"""Main navigation pipeline orchestrating all subsystems.

NavigationPipeline wires together: perception → memory → estimation →
planning, and drives the processing loop at the configured frame rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from uav_nav.data.calibration import CameraCalibration, StereoCalibration
from uav_nav.estimation.ekf_vio import EKFVIO, EKFState
from uav_nav.memory.tilm import TILM
from uav_nav.memory.temporal_matcher import TemporalMatcher
from uav_nav.perception.landmark_extractor import LandmarkExtractor
from uav_nav.perception.stereo_depth import StereoDepthEstimator
from uav_nav.perception.yolo_segmenter import YOLOSegmenter
from uav_nav.planning.adaptive_corridor import AdaptiveCorridor
from uav_nav.planning.mission_modes import MissionManager, MissionMode
from uav_nav.planning.path_follower import PathFollower, PathFollowerConfig
from uav_nav.runtime.monitor import SystemMonitor


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration.

    Attributes:
        camera_calibration_path: Path to camera calibration YAML.
        stereo_calibration_path: Path to stereo calibration YAML.
        yolo_weights_path: Path to YOLO segmentation weights.
        embedding_weights_path: Path to embedding head checkpoint.
        tilm_path: Path to pre-built TILM file.
        waypoints_path: Path to planned-path waypoints (NPY or CSV).
        log_dir: Directory for logs and telemetry output.
        device: Torch device string.
        target_fps: Target processing frame rate.
        enable_rerun: Whether to stream data to Rerun SDK visualiser.
    """

    camera_calibration_path: Optional[Path] = None
    stereo_calibration_path: Optional[Path] = None
    yolo_weights_path: Optional[Path] = None
    embedding_weights_path: Optional[Path] = None
    tilm_path: Optional[Path] = None
    waypoints_path: Optional[Path] = None
    log_dir: Path = Path("logs")
    device: str = "cpu"
    target_fps: float = 10.0
    enable_rerun: bool = False


class NavigationPipeline:
    """Full UAV navigation pipeline with VIO and semantic landmark matching.

    Args:
        config: PipelineConfig with all subsystem paths and settings.
        path_follower_config: Optional PathFollowerConfig override.
    """

    def __init__(
        self,
        config: PipelineConfig,
        path_follower_config: Optional[PathFollowerConfig] = None,
    ) -> None:
        self.config = config
        self._pf_config = path_follower_config or PathFollowerConfig()
        self._initialised: bool = False

        # Subsystem handles (populated in initialise())
        self._segmenter: Optional[YOLOSegmenter] = None
        self._depth_estimator: Optional[StereoDepthEstimator] = None
        self._landmark_extractor: Optional[LandmarkExtractor] = None
        self._tilm: Optional[TILM] = None
        self._matcher: Optional[TemporalMatcher] = None
        self._ekf: Optional[EKFVIO] = None
        self._path_follower: Optional[PathFollower] = None
        self._corridor: Optional[AdaptiveCorridor] = None
        self._mission: Optional[MissionManager] = None
        self._monitor: Optional[SystemMonitor] = None

    def initialise(self) -> None:
        """Load all subsystems and prepare the pipeline for processing.

        Raises:
            NotImplementedError: Not yet implemented.
            FileNotFoundError: If any required weight/config file is missing.
        """
        raise NotImplementedError("NavigationPipeline.initialise is not yet implemented.")

    def process_frame(
        self,
        image_rgb: np.ndarray,
        image_right: Optional[np.ndarray],
        imu_acc: np.ndarray,
        imu_gyro: np.ndarray,
        dt: float,
        gnss_position: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
    ) -> dict[str, object]:
        """Process one sensor frame through the full pipeline.

        Args:
            image_rgb: Left RGB image, shape (H, W, 3), uint8.
            image_right: Optional right image for stereo depth.
            imu_acc: Accelerometer measurement, shape (3,), m/s^2.
            imu_gyro: Gyroscope measurement, shape (3,), rad/s.
            dt: Time since last frame, seconds.
            gnss_position: Optional GNSS NED position, shape (3,).
            timestamp: Frame timestamp in seconds.

        Returns:
            Dictionary with keys: "pose", "velocity_cmd", "mode",
            "n_landmarks", "tilm_quality", "corridor_state", "state".

        Raises:
            NotImplementedError: Not yet implemented.
            RuntimeError: If ``initialise()`` has not been called.
        """
        raise NotImplementedError("NavigationPipeline.process_frame is not yet implemented.")

    def shutdown(self) -> None:
        """Gracefully shut down all subsystems and flush logs.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("NavigationPipeline.shutdown is not yet implemented.")

    @property
    def current_state(self) -> Optional[EKFState]:
        """Current EKF state, or None if not yet initialised."""
        if self._ekf is None:
            return None
        return self._ekf.get_state()

    @property
    def current_mode(self) -> Optional[MissionMode]:
        """Current mission mode, or None if not yet initialised."""
        if self._mission is None:
            return None
        return self._mission.current_mode
