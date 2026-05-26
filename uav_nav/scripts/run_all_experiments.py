"""Script 08: Run all dissertation experiments in sequence.

Usage::

    python uav_nav/scripts/08_run_all_experiments.py
    python uav_nav/scripts/08_run_all_experiments.py --experiments 5_3 5_5

Executes all configured experiment scenarios via ScenarioRunner and
produces a consolidated results table in results/all_experiments.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from uav_nav.eval.ablation import AblationStudy, AblationConfig
from uav_nav.eval.scenario_runner import ScenarioConfig, ScenarioRunner
from uav_nav.runtime.logger import setup_logger, get_logger

logger = get_logger(__name__)

# Registry of all experiments to run
EXPERIMENT_CONFIGS: dict[str, str] = {
    "5_3": "uav_nav/configs/experiments/5_3_cross_weather.yaml",
    "5_5": "uav_nav/configs/experiments/5_5_tilm_correction.yaml",
}


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run all dissertation experiments.

    Args:
        cfg: Hydra configuration object.

    Raises:
        NotImplementedError: Experiment runner implementation pending.
    """
    setup_logger(log_dir=Path(cfg.runtime.log_dir), level=cfg.runtime.log_level)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENT_CONFIGS.keys()),
        help="Experiment IDs to run (e.g. 5_3 5_5).",
    )
    args, _ = parser.parse_known_args()

    selected = {k: v for k, v in EXPERIMENT_CONFIGS.items() if k in args.experiments}
    if not selected:
        logger.error("No valid experiments selected. Available: {}", list(EXPERIMENT_CONFIGS))
        sys.exit(1)

    logger.info("Running {} experiment(s): {}", len(selected), list(selected.keys()))

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    midair_root = Path(cfg.dataset.midair_root)

    # TODO: Implement pipeline factory and scenario runner loop
    # runner = ScenarioRunner(
    #     pipeline_factory=build_pipeline,
    #     output_dir=results_dir,
    #     midair_root=midair_root,
    # )
    # all_results = runner.run_all(scenario_configs)
    # runner.save_summary(results_dir / "all_experiments.csv")

    raise NotImplementedError(
        "08_run_all_experiments.py: Experiment runner not yet implemented."
    )


if __name__ == "__main__":
    main()
