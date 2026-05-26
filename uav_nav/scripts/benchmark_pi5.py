"""Script 07: Benchmark pipeline latency on Raspberry Pi 5.

Usage::

    python uav_nav/scripts/07_benchmark_pi5.py
    python uav_nav/scripts/07_benchmark_pi5.py --config uav_nav/configs/pi5.yaml

Measures per-module and end-to-end latency, reports pass/fail against
Pi 5 real-time targets, and saves results to benchmark_results/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from uav_nav.deployment.benchmark_pi5 import Pi5Benchmark
from uav_nav.runtime.pipeline import NavigationPipeline, PipelineConfig
from uav_nav.runtime.logger import setup_logger, get_logger

logger = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="pi5", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run Pi 5 hardware benchmark suite.

    Args:
        cfg: Hydra configuration object (pi5.yaml by default).

    Raises:
        NotImplementedError: Benchmark implementation pending.
    """
    setup_logger(log_dir=Path(cfg.runtime.log_dir), level=cfg.runtime.log_level)
    logger.info("Pi 5 benchmark starting")
    logger.info("Target FPS: {:.1f}", cfg.runtime.target_fps)

    tilm_path = Path("data/tilm/tilm.hdf5")
    if not tilm_path.exists():
        logger.error("TILM not found at {}. Run 04_build_tilm.py first.", tilm_path)
        sys.exit(1)

    pipeline_cfg = PipelineConfig(
        yolo_weights_path=Path(cfg.yolo.weights_path),
        embedding_weights_path=Path(cfg.embedding.weights_path),
        tilm_path=tilm_path,
        log_dir=Path(cfg.runtime.log_dir),
        device=cfg.yolo.device,
        target_fps=cfg.runtime.target_fps,
    )
    pipeline = NavigationPipeline(config=pipeline_cfg)
    pipeline.initialise()

    output_dir = Path("benchmark_results")
    benchmarker = Pi5Benchmark(
        output_dir=output_dir,
        n_runs=cfg.eval.benchmark_n_runs,
        warmup=cfg.eval.benchmark_warmup,
    )

    results = benchmarker.run_all(pipeline)

    print("\n" + benchmarker.report())

    csv_path = benchmarker.save_csv()
    logger.info("Benchmark results saved to: {}", csv_path)

    failed = [r for r in results if not r.passes]
    if failed:
        logger.warning(
            "{} module(s) failed latency targets: {}",
            len(failed),
            [r.module_name for r in failed],
        )
        sys.exit(1)

    logger.info("All modules pass Pi 5 latency targets.")


if __name__ == "__main__":
    main()
