"""Script: Build dataset splits from raw MidAir data.

Produces two artefacts:
  1. HDF5 files (``data/processed/``) — for EKF/TILM training with depth and poses.
  2. YOLO-format dataset (``data/yolo/``) — for YOLOv8n-seg training.

Usage::

    uav-nav-build-dataset dataset.midair_root="D:/data/MidAir"

    # Override YOLO output dir
    uav-nav-build-dataset dataset.midair_root="D:/data/MidAir" dataset.yolo_root="data/yolo"

    # Skip HDF5 (only build YOLO dataset)
    uav-nav-build-dataset dataset.midair_root="D:/data/MidAir" +dataset.yolo_only=true
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from uav_nav.data.calibration import CameraCalibration
from uav_nav.data.dataset_builder import DatasetBuilder, SplitConfig
from uav_nav.data.midair_loader import MidAirLoader
from uav_nav.runtime.logger import get_logger, setup_logger

logger = get_logger(__name__)

# Absolute path so Hydra finds configs regardless of CWD or entry-point install dir.
_CONFIGS_DIR = str(Path(__file__).resolve().parent.parent / "configs")


@hydra.main(config_path=_CONFIGS_DIR, config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Build HDF5 and YOLO dataset splits from raw MidAir data.

    Args:
        cfg: Hydra configuration object.
    """
    setup_logger(log_dir=Path(cfg.runtime.log_dir), level=cfg.runtime.log_level)
    logger.info("Dataset config:\n{}", OmegaConf.to_yaml(cfg.dataset))

    midair_root = Path(cfg.dataset.midair_root)
    if not midair_root.exists():
        logger.error("MidAir root not found: {}", midair_root)
        sys.exit(1)

    # Load label map
    label_map_path = Path(cfg.dataset.label_map_path)
    if not label_map_path.exists():
        # Fallback: look relative to project root
        label_map_path = Path(__file__).resolve().parent.parent / "configs" / "label_map.json"
    with open(label_map_path, encoding="utf-8") as f:
        label_json = json.load(f)
    label_map: dict[int, str] = {int(k): v for k, v in label_json["label_map"].items()}

    calibration = CameraCalibration.from_midair_defaults()

    train_seqs = list(cfg.dataset.train_sequences)
    val_seqs = list(cfg.dataset.val_sequences)
    test_seqs = list(cfg.dataset.test_sequences)
    target_size: tuple[int, int] = tuple(cfg.dataset.target_size)  # type: ignore[assignment]
    yolo_only: bool = bool(OmegaConf.select(cfg, "dataset.yolo_only", default=False))
    max_frames: int | None = OmegaConf.select(cfg, "dataset.max_frames_per_traj", default=None)
    camera_dir: str = str(OmegaConf.select(cfg, "dataset.camera_dir", default="color_left"))
    frame_stride: int | None = OmegaConf.select(cfg, "dataset.frame_stride", default=None)

    # Collect all sequences we actually need (union of all splits)
    all_sequences = list(dict.fromkeys(train_seqs + val_seqs + test_seqs))

    loader = MidAirLoader(
        root=midair_root,
        sequences=all_sequences,
        load_depth=not yolo_only,   # depth not needed for YOLO polygon export
        load_segmentation=True,
        load_stereo=False,
        max_frames=max_frames,
        camera_dir=camera_dir,
        frame_stride=frame_stride,
    )
    loader.build_index(skip_on_error=True)
    logger.info("MidAir index built: {} total samples", len(loader))

    builder = DatasetBuilder(
        loader=loader,
        calibration=calibration,
        output_dir=Path("data/processed"),
        label_map=label_map,
        target_size=target_size,
    )

    # ── HDF5 splits ───────────────────────────────────────────────────────────
    if not yolo_only:
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        hdf5_paths = builder.build_all(
            train_sequences=train_seqs,
            val_sequences=val_seqs,
            test_sequences=test_seqs,
        )
        for split, path in hdf5_paths.items():
            logger.info("HDF5 split '{}' → {}", split, path)

        # Print class distribution for the train split
        if "train" in hdf5_paths:
            stats = builder.compute_statistics(hdf5_paths["train"])
            logger.info("Train channel stats: {}", stats)
    else:
        logger.info("--yolo_only mode: skipping HDF5 build")

    # ── YOLO-format dataset ───────────────────────────────────────────────────
    yolo_root = Path(OmegaConf.select(cfg, "dataset.yolo_root", default="data/yolo"))
    logger.info("Building YOLO dataset → {}", yolo_root)

    builder.export_yolo_all(
        train_sequences=train_seqs,
        val_sequences=val_seqs,
        test_sequences=test_seqs,
        yolo_root=yolo_root,
    )

    yaml_path = yolo_root / "midair.yaml"
    logger.success("YOLO dataset ready. Dataset YAML: {}", yaml_path)
    logger.info("Next step: uav-nav-train-yolo yolo.dataset_yaml={}", yaml_path)


if __name__ == "__main__":
    main()
