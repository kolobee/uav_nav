"""Script 05: Simulate GNSS loss scenarios through the pseudo-simulator.

Usage::

    python uav_nav/scripts/05_simulate_gnss_loss.py
    python uav_nav/scripts/05_simulate_gnss_loss.py outage_duration=120.0

Replays MidAir test sequences with injected GNSS outages and records
position errors for evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from uav_nav.data.midair_loader import MidAirLoader
from uav_nav.eval.trajectory_eval import TrajectoryEvaluator
from uav_nav.runtime.pipeline import NavigationPipeline, PipelineConfig
from uav_nav.runtime.pseudo_simulator import GNSSOutageConfig, PseudoSimulator
from uav_nav.runtime.logger import setup_logger, get_logger

logger = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run GNSS loss simulation and evaluate trajectory accuracy.

    Args:
        cfg: Hydra configuration object.

    Raises:
        NotImplementedError: Simulation implementation pending.
    """
    setup_logger(log_dir=Path(cfg.runtime.log_dir), level=cfg.runtime.log_level)
    logger.info("Starting GNSS loss simulation")

    midair_root = Path(cfg.dataset.midair_root)
    tilm_path = Path("data/tilm/tilm.hdf5")

    if not tilm_path.exists():
        logger.error("TILM not found at {}. Run 04_build_tilm.py first.", tilm_path)
        sys.exit(1)

    loader = MidAirLoader(
        root=midair_root,
        sequences=list(cfg.dataset.test_sequences),
        load_depth=True,
        load_segmentation=True,
    )
    loader.build_index()

    pipeline_cfg = PipelineConfig(
        yolo_weights_path=Path(cfg.yolo.weights_path),
        embedding_weights_path=Path(cfg.embedding.weights_path),
        tilm_path=tilm_path,
        log_dir=Path(cfg.runtime.log_dir),
        device=cfg.yolo.device,
    )
    pipeline = NavigationPipeline(config=pipeline_cfg)
    pipeline.initialise()

    outages = [
        GNSSOutageConfig(start_time=60.0, duration=120.0, noise_std=0.0),
    ]

    def progress(idx: int, total: int) -> None:
        if idx % 50 == 0:
            logger.info("  Frame {}/{}", idx, total)

    simulator = PseudoSimulator(
        pipeline=pipeline,
        loader=loader,
        outages=outages,
        progress_callback=progress,
    )

    result = simulator.run()
    logger.info("Simulation complete: RMSE = {:.2f} m, max error = {:.2f} m",
                result.rmse, result.max_error)

    evaluator = TrajectoryEvaluator(align=cfg.eval.align_trajectory)
    metrics = evaluator.evaluate(
        positions_est=result.positions_est,
        positions_gt=result.positions_gt,
        timestamps=result.timestamps,
        sequence_name="gnss_loss_simulation",
    )

    output_dir = Path("results/gnss_loss")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator.save_report([metrics], output_dir / "metrics.csv")
    logger.info("Results saved to {}", output_dir)


if __name__ == "__main__":
    main()
