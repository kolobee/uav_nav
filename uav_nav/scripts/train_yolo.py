"""Script: Fine-tune YOLOv8n-seg on MidAir aerial imagery.

Usage::

    # Basic (uses default config)
    uav-nav-train-yolo

    # Override parameters
    uav-nav-train-yolo yolo.imgsz=640 yolo.epochs=100 yolo.device=cuda

    # Custom dataset YAML
    uav-nav-train-yolo yolo.dataset_yaml=data/yolo/midair.yaml

The script expects a YOLO-format dataset at ``yolo.dataset_yaml``.
Run ``uav-nav-build-dataset`` first if the dataset does not yet exist.

Outputs are written to ``experiments/<date>_yolo/train/weights/``.
The best checkpoint is also copied to ``weights/yolo_midair.pt``.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from uav_nav.runtime.logger import get_logger, setup_logger

logger = get_logger(__name__)

_CONFIGS_DIR = str(Path(__file__).resolve().parent.parent / "configs")

_DEFAULTS: dict[str, object] = {
    "model": "yolov8n-seg.pt",
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "device": "cpu",
    "workers": 4,
    "patience": 20,
    "lr0": 1e-3,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 5e-4,
    "warmup_epochs": 3,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "dataset_yaml": "data/yolo/midair.yaml",
}


@hydra.main(config_path=_CONFIGS_DIR, config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Fine-tune YOLO segmentation on the MidAir dataset.

    Args:
        cfg: Hydra configuration object.
    """
    setup_logger(log_dir=Path(cfg.runtime.log_dir), level=cfg.runtime.log_level)

    yolo_cfg: dict = dict(_DEFAULTS)
    if hasattr(cfg, "yolo"):
        user = OmegaConf.to_container(cfg.yolo, resolve=True)
        assert isinstance(user, dict)
        yolo_cfg.update({k: v for k, v in user.items() if v is not None})

    dataset_yaml = Path(str(yolo_cfg["dataset_yaml"]))
    if not dataset_yaml.exists():
        logger.error(
            "Dataset YAML not found: {}. Run 'uav-nav-build-dataset' first.",
            dataset_yaml,
        )
        sys.exit(1)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("experiments") / f"{stamp}_yolo"
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Experiment dir: {}", exp_dir)
    logger.info("Training config: {}", yolo_cfg)

    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(str(yolo_cfg["model"]))
    results = model.train(
        data=str(dataset_yaml.resolve()),
        epochs=int(yolo_cfg["epochs"]),
        imgsz=int(yolo_cfg["imgsz"]),
        batch=int(yolo_cfg["batch"]),
        device=yolo_cfg["device"],
        workers=int(yolo_cfg["workers"]),
        patience=int(yolo_cfg["patience"]),
        lr0=float(yolo_cfg["lr0"]),
        lrf=float(yolo_cfg["lrf"]),
        momentum=float(yolo_cfg["momentum"]),
        weight_decay=float(yolo_cfg["weight_decay"]),
        warmup_epochs=float(yolo_cfg["warmup_epochs"]),
        hsv_h=float(yolo_cfg["hsv_h"]),
        hsv_s=float(yolo_cfg["hsv_s"]),
        hsv_v=float(yolo_cfg["hsv_v"]),
        flipud=float(yolo_cfg["flipud"]),
        fliplr=float(yolo_cfg["fliplr"]),
        mosaic=float(yolo_cfg["mosaic"]),
        project=str(exp_dir),
        name="train",
        exist_ok=True,
        verbose=True,
    )

    # Copy best weights to canonical location.
    # ultralytics may save to its own CWD-relative path (e.g. runs/segment/...),
    # so check results.save_dir first, then fall back to rglob.
    best_pt: Path | None = None
    if results is not None and hasattr(results, "save_dir"):
        candidate = Path(results.save_dir) / "weights" / "best.pt"
        if candidate.exists():
            best_pt = candidate
    if best_pt is None:
        found = list(exp_dir.rglob("best.pt"))
        if found:
            best_pt = found[0]

    if best_pt is not None:
        weights_dir = Path("weights")
        weights_dir.mkdir(parents=True, exist_ok=True)
        dst = weights_dir / "yolo_midair.pt"
        shutil.copy2(best_pt, dst)
        logger.success("Best weights → {}", dst)
    else:
        logger.warning("best.pt not found after training — copy it manually.")

    if results is not None and hasattr(results, "results_dict"):
        rd = results.results_dict
        logger.info(
            "Final — mAP50-box: {:.4f}  mAP50-mask: {:.4f}",
            rd.get("metrics/mAP50(B)", float("nan")),
            rd.get("metrics/mAP50(M)", float("nan")),
        )


if __name__ == "__main__":
    main()
