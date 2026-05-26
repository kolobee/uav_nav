"""Script 04: Build the Topological Invariant Landmark Map (TILM).

Usage::

    python uav_nav/scripts/04_build_tilm.py
    python uav_nav/scripts/04_build_tilm.py tilm.min_node_distance=8.0

Processes the training sequences through the full perception stack and
incrementally builds the TILM, saving it to data/tilm/tilm.hdf5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from uav_nav.data.calibration import CameraCalibration
from uav_nav.data.midair_loader import MidAirLoader
from uav_nav.memory.tilm import TILM
from uav_nav.memory.tilm_builder import TILMBuilder, BuilderConfig
from uav_nav.runtime.logger import setup_logger, get_logger

logger = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Build TILM from training sequences.

    Args:
        cfg: Hydra configuration object.

    Raises:
        NotImplementedError: TILM build implementation pending.
    """
    setup_logger(log_dir=Path(cfg.runtime.log_dir), level=cfg.runtime.log_level)
    logger.info("TILM build configuration:\n{}", OmegaConf.to_yaml(cfg.tilm))

    midair_root = Path(cfg.dataset.midair_root)
    if not midair_root.exists():
        logger.error("MidAir root not found: {}", midair_root)
        sys.exit(1)

    loader = MidAirLoader(
        root=midair_root,
        sequences=list(cfg.dataset.train_sequences),
        load_depth=True,
        load_segmentation=True,
    )
    loader.build_index()
    logger.info("Loaded {} training frames", len(loader))

    tilm = TILM(
        min_node_distance=cfg.tilm.min_node_distance,
        max_nodes=cfg.tilm.max_nodes,
    )
    tilm.initialise()

    builder_cfg = BuilderConfig(
        min_node_distance=cfg.tilm.min_node_distance,
        min_landmarks_per_node=cfg.tilm.min_landmarks_per_node,
        edge_max_distance=cfg.tilm.edge_max_distance,
        use_loop_closure=cfg.tilm.use_loop_closure,
        loop_closure_threshold=cfg.tilm.loop_closure_threshold,
    )
    builder = TILMBuilder(tilm=tilm, config=builder_cfg)

    logger.info("Processing frames to build TILM...")

    # TODO: Implement full build loop:
    # for sample in loader:
    #     landmarks = landmark_extractor.extract(...)
    #     builder.process_frame(pose_ned=..., landmarks=landmarks, timestamp=...)

    tilm_out = Path("data/tilm")
    tilm_out.mkdir(parents=True, exist_ok=True)
    output_path = tilm_out / "tilm.hdf5"

    raise NotImplementedError("04_build_tilm.py: TILM build not yet implemented.")


if __name__ == "__main__":
    main()
