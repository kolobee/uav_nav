"""Script: Evaluate YOLO segmentation model (Experiments 5.3 and 5.4).

Usage::

    # Experiment 5.3 — cross-weather evaluation
    uav-nav-eval-yolo +experiment=5_3_cross_weather \\
        eval.weights_path=weights/yolo_midair.pt \\
        eval.yolo_root=data/yolo

    # Experiment 5.4 — leave-one-trajectory-out
    uav-nav-eval-yolo +experiment=5_4_leave_one_out \\
        eval.weights_path=weights/yolo_midair.pt \\
        eval.yolo_root=data/yolo

Results are saved to ``experiments/eval_<date>/``.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from uav_nav.eval.yolo_eval import YOLOEvaluator
from uav_nav.runtime.logger import get_logger, setup_logger

logger = get_logger(__name__)

_CONFIGS_DIR = str(Path(__file__).resolve().parent.parent / "configs")


@hydra.main(config_path=_CONFIGS_DIR, config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run YOLO evaluation for Experiments 5.3 or 5.4.

    Args:
        cfg: Hydra configuration object.
    """
    setup_logger(log_dir=Path(cfg.runtime.log_dir), level=cfg.runtime.log_level)

    weights_path = Path(OmegaConf.select(cfg, "eval.weights_path", default="weights/yolo_midair.pt"))
    yolo_root = Path(OmegaConf.select(cfg, "eval.yolo_root", default="data/yolo"))
    experiment_name = OmegaConf.select(cfg, "experiment.name", default="yolo_eval")
    imgsz = int(OmegaConf.select(cfg, "yolo.imgsz", default=640))
    device = str(OmegaConf.select(cfg, "yolo.device", default="cpu"))

    if not weights_path.exists():
        logger.error("Weights not found: {}", weights_path)
        sys.exit(1)

    # Output directory
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("experiments") / f"{stamp}_{experiment_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: {}", out_dir)

    # Load class names from label_map or config
    class_names: list[str] = list(
        OmegaConf.select(cfg, "place_descriptor.class_order", default=["tree", "rock", "river"])
    )

    evaluator = YOLOEvaluator(
        class_names=class_names,
        imgsz=imgsz,
        device=device,
    )

    if "5_3" in experiment_name or "cross_weather" in experiment_name:
        _run_cross_weather(cfg, evaluator, weights_path, yolo_root, out_dir)
    elif "5_4" in experiment_name or "leave_one_out" in experiment_name:
        _run_leave_one_out(cfg, evaluator, weights_path, yolo_root, out_dir)
    else:
        # Generic: single evaluation on the default dataset YAML
        yaml_path = yolo_root / "midair.yaml"
        if not yaml_path.exists():
            logger.error("Dataset YAML not found: {}", yaml_path)
            sys.exit(1)
        metrics = [evaluator.evaluate_on_yaml(weights_path, yaml_path, split="test")]
        evaluator.save_table_csv(metrics, out_dir / "metrics.csv")
        evaluator.save_table_json(metrics, out_dir / "metrics.json")
        logger.info("mAP50-mask: {:.4f}", metrics[0].map50_mask)


def _run_cross_weather(
    cfg: DictConfig,
    evaluator: YOLOEvaluator,
    weights: Path,
    yolo_root: Path,
    out_dir: Path,
) -> None:
    """Run Experiment 5.3: cross-weather evaluation."""
    # Expect weather-specific YAML files: yolo_root/{weather}/midair.yaml
    # or a flat yolo_root/midair_{weather}.yaml
    test_weathers: list[str] = list(
        OmegaConf.select(cfg, "scenario.test_weather", default=["cloudy", "foggy", "sunny"])
    )

    weather_yamls: dict[str, Path] = {}
    for w in test_weathers:
        candidates = [
            yolo_root / w / "midair.yaml",
            yolo_root / f"midair_{w}.yaml",
        ]
        for c in candidates:
            if c.exists():
                weather_yamls[w] = c
                break
        else:
            logger.warning("No dataset YAML found for weather '{}', skipping.", w)

    if not weather_yamls:
        logger.error(
            "No weather-specific dataset YAMLs found under {}. "
            "Expected {weather}/midair.yaml or midair_{weather}.yaml.",
            yolo_root,
        )
        return

    metrics = evaluator.cross_weather_table(weights, weather_yamls)
    evaluator.save_table_csv(metrics, out_dir / "cross_weather.csv")
    evaluator.save_table_json(metrics, out_dir / "cross_weather.json")

    stats = evaluator.summary_stats(metrics)
    logger.info(
        "Cross-weather summary — mean mAP50-mask: {:.4f} ± {:.4f}", stats["mean"], stats["std"]
    )


def _run_leave_one_out(
    cfg: DictConfig,
    evaluator: YOLOEvaluator,
    weights: Path,
    yolo_root: Path,
    out_dir: Path,
) -> None:
    """Run Experiment 5.4: leave-one-trajectory-out evaluation.

    Assumes a folder structure:
        yolo_root/
            loo_fold_<traj>/
                midair.yaml       ← dataset with that trajectory held out
                weights/best.pt   ← weights trained on the rest
    """
    fold_dirs = sorted(yolo_root.glob("loo_fold_*"))
    if not fold_dirs:
        logger.warning(
            "No LOO fold directories found under {}. "
            "Expected 'loo_fold_*' subdirectories.",
            yolo_root,
        )
        # Fall back: use the same weights but evaluate per trajectory YAML
        logger.info("Falling back to single-weights LOO evaluation.")
        fold_dirs_yaml = sorted(yolo_root.glob("*/midair.yaml"))
        folds = {p.parent.name: (weights, p) for p in fold_dirs_yaml}
    else:
        folds: dict[str, tuple[Path, Path]] = {}
        for d in fold_dirs:
            fold_weights = d / "weights" / "best.pt"
            fold_yaml = d / "midair.yaml"
            if fold_weights.exists() and fold_yaml.exists():
                folds[d.name] = (fold_weights, fold_yaml)
            else:
                logger.warning("Fold '{}' missing weights or YAML, using global weights.", d.name)
                if fold_yaml.exists():
                    folds[d.name] = (weights, fold_yaml)

    if not folds:
        logger.error("No folds found for LOO evaluation.")
        return

    metrics = evaluator.leave_one_out_table(folds)
    evaluator.save_table_csv(metrics, out_dir / "leave_one_out.csv")
    evaluator.save_table_json(metrics, out_dir / "leave_one_out.json")

    stats = evaluator.summary_stats(metrics)
    logger.info(
        "LOO summary — mean mAP50-mask: {:.4f} ± {:.4f}", stats["mean"], stats["std"]
    )


if __name__ == "__main__":
    main()
