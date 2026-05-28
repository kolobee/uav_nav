"""Script 05: Simulate GNSS loss and evaluate VIO trajectory accuracy.

Usage::

    python -m uav_nav.scripts.simulate_gnss_loss \\
        dataset.midair_root=D:/data/MidAir \\
        simulate.gnss_loss_fraction=0.5 \\
        simulate.frame_stride=5

Replays a MidAir test trajectory in two phases:
  1. Mapping  — GNSS active, TILMBuilder accumulates landmarks.
  2. Navigation — GNSS lost, EKF corrected via TILM matching.

Prints a comparison table and saves metrics to ``results/gnss_loss/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from uav_nav.data.calibration import CameraCalibration
from uav_nav.data.midair_loader import MidAirLoader
from uav_nav.eval.trajectory_eval import TrajectoryEvaluator
from uav_nav.runtime.logger import get_logger, setup_logger
from uav_nav.runtime.pseudo_simulator import PseudoSimulator, SimulationConfig

logger = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run GNSS loss simulation and evaluate trajectory accuracy.

    Args:
        cfg: Hydra configuration object.
    """
    setup_logger(log_dir=Path(cfg.runtime.log_dir), level=cfg.runtime.log_level)
    logger.info("=== GNSS Loss Simulation ===")

    midair_root = Path(cfg.dataset.midair_root)
    if not midair_root.exists():
        logger.error("MidAir root not found: {}", midair_root)
        sys.exit(1)

    # ── loader ─────────────────────────────────────────────────────────────
    loader = MidAirLoader(
        root=midair_root,
        sequences=list(cfg.dataset.test_sequences),
        load_depth=True,
        load_segmentation=True,
        load_right=False,
        frame_stride=int(cfg.get("simulate", {}).get("frame_stride", 5)),
    )
    loader.build_index()
    logger.info("Loaded {} frames from {}", len(loader), midair_root)

    if len(loader) < 10:
        logger.error("Not enough frames ({}). Check dataset path and sequences.", len(loader))
        sys.exit(1)

    # ── calibration ────────────────────────────────────────────────────────
    calibration = CameraCalibration.from_midair_defaults()

    # ── optional YOLO + embedding ──────────────────────────────────────────
    yolo_segmenter = None
    embedding_head = None

    yolo_path = Path(cfg.get("yolo", {}).get("weights_path", "weights/yolo_midair.pt"))
    emb_path = Path(cfg.get("embedding", {}).get("weights_path", "weights/embedding_head.pt"))

    if yolo_path.exists():
        from uav_nav.perception.yolo_segmenter import YOLOSegmenter
        yolo_segmenter = YOLOSegmenter(device=cfg.get("yolo", {}).get("device", "cpu"))
        yolo_segmenter.load(yolo_path)
        logger.info("YOLO loaded from {}", yolo_path)
    else:
        logger.info("YOLO weights not found — using GT segmentation mode")

    if emb_path.exists():
        from uav_nav.perception.embedding_head import EmbeddingHead
        embedding_head = EmbeddingHead.load(
            emb_path, device=cfg.get("yolo", {}).get("device", "cpu")
        )
        logger.info("EmbeddingHead loaded from {}", emb_path)

    # ── simulation config ──────────────────────────────────────────────────
    sim_cfg_raw = cfg.get("simulate", {})
    sim_config = SimulationConfig(
        gnss_loss_fraction=float(sim_cfg_raw.get("gnss_loss_fraction", 0.5)),
        gnss_noise_std=float(sim_cfg_raw.get("gnss_noise_std", 1.0)),
        frame_stride=int(sim_cfg_raw.get("frame_stride", 1)),
    )

    # ── run simulator ──────────────────────────────────────────────────────
    def _progress(idx: int, total: int) -> None:
        if idx % 100 == 0:
            logger.info("  Frame {}/{} ({:.0f}%)", idx, total, idx / total * 100)

    simulator = PseudoSimulator(
        loader=loader,
        calibration=calibration,
        embedding_head=embedding_head,
        yolo_segmenter=yolo_segmenter,
        config=sim_config,
        progress_callback=_progress,
    )

    result = simulator.run()

    # ── evaluate ───────────────────────────────────────────────────────────
    evaluator = TrajectoryEvaluator(align=bool(cfg.get("eval", {}).get("align_trajectory", True)))
    comparison = evaluator.compare(
        gt_positions=result.gt_positions,
        with_tilm=result.estimated_positions,
        without_tilm=result.dead_reckoning_positions,
        timestamps=result.timestamps,
    )

    metrics_tilm = comparison["tilm"]
    metrics_dr = comparison["dead_reckoning"]
    improvement = comparison["improvement_pct"]

    # ── print results ──────────────────────────────────────────────────────
    print("\n=== Simulation Results ===")
    print(f"Trajectory:         {result.sequence_id}")
    print(f"TILM nodes built:   {result.tilm_node_count}")
    print(f"GNSS loss at frame: {result.gnss_loss_idx}/{len(result.timestamps)}")
    if result.corrector_stats:
        cs = result.corrector_stats
        print(f"TILM acceptance:    {cs.tilm_acceptance_rate:.1%} "
              f"({cs.n_tilm_corrections}/{cs.n_tilm_attempts} attempts)")
    print()
    print(f"{'Metric':<28} {'IMU-only':>12} {'IMU+TILM':>12}")
    print("-" * 54)
    print(f"{'ATE RMSE (m)':<28} {metrics_dr.ate_rmse:>12.2f} {metrics_tilm.ate_rmse:>12.2f}")
    print(f"{'ATE mean (m)':<28} {metrics_dr.ate_mean:>12.2f} {metrics_tilm.ate_mean:>12.2f}")
    print(f"{'ATE max (m)':<28} {metrics_dr.ate_max:>12.2f} {metrics_tilm.ate_max:>12.2f}")
    print(f"{'RPE RMSE (m)':<28} {metrics_dr.rpe_rmse:>12.2f} {metrics_tilm.rpe_rmse:>12.2f}")
    print("-" * 54)
    print(f"{'TILM improvement':<28} {'':>12} {improvement:>11.1f}%")
    print()

    # ── save results ───────────────────────────────────────────────────────
    output_dir = Path("results/gnss_loss")
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator.save_report(
        [metrics_dr, metrics_tilm],
        output_dir / "metrics.csv",
    )
    evaluator.save_report(
        [metrics_dr, metrics_tilm],
        output_dir / "metrics.json",
        format="json",
    )

    np.save(output_dir / "gt_positions.npy", result.gt_positions)
    np.save(output_dir / "estimated_positions.npy", result.estimated_positions)
    np.save(output_dir / "dead_reckoning_positions.npy", result.dead_reckoning_positions)

    evaluator.plot_trajectories(
        positions_est=result.estimated_positions,
        positions_gt=result.gt_positions,
        dead_reckoning=result.dead_reckoning_positions,
        output_path=output_dir / "trajectory.png",
        title=f"GNSS loss simulation — improvement {improvement:.1f}%",
    )

    logger.info("Results saved to {}", output_dir)


if __name__ == "__main__":
    main()
