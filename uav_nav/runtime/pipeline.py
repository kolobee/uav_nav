"""Main navigation pipeline orchestrating all subsystems.

NavigationPipeline wires together: perception → memory → estimation →
planning, and drives the processing loop at the configured frame rate.

Intended for real-time use via AirSimBridge or Pi5 runtime.  For offline
experiment replay, use ``PseudoSimulator`` instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from uav_nav.data.calibration import CameraCalibration, StereoCalibration
from uav_nav.estimation.ekf_vio import EKFVIO, EKFState
from uav_nav.estimation.imu_preintegrator import IMUPreintegrator
from uav_nav.memory.tilm import TILM
from uav_nav.memory.temporal_matcher import TemporalMatcher
from uav_nav.perception.embedding_head import EmbeddingHead
from uav_nav.perception.landmark_extractor import LandmarkExtractor
from uav_nav.perception.stereo_depth import DepthResult, StereoDepthEstimator
from uav_nav.perception.yolo_segmenter import SegmentationResult, YOLOSegmenter
from uav_nav.planning.adaptive_corridor import AdaptiveCorridor
from uav_nav.planning.mission_modes import MissionManager, MissionMode
from uav_nav.planning.path_follower import PathFollower, PathFollowerConfig
from uav_nav.runtime.matching_corrector import MatchingCorrector
from uav_nav.runtime.monitor import SystemMonitor


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration.

    Attributes:
        camera_calibration_path: Path to camera calibration YAML.
        stereo_calibration_path: Path to stereo calibration YAML.
        yolo_weights_path: Path to YOLO segmentation weights (.pt / .onnx).
        embedding_weights_path: Path to embedding head checkpoint.
        tilm_path: Path to pre-built TILM file.  If None, starts empty.
        waypoints_path: Path to planned-path waypoints (NPY or CSV).
        log_dir: Directory for logs and telemetry output.
        device: Torch device string (``"cpu"`` or ``"cuda"``).
        target_fps: Target processing frame rate.
        enable_rerun: Whether to stream data to Rerun SDK visualiser.
        gnss_noise_std: Expected GNSS noise (m), used for EKF update weight.
        min_landmarks_for_correction: Minimum landmark count to attempt TILM
            correction.
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
    gnss_noise_std: float = 3.0
    min_landmarks_for_correction: int = 1


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

        # Subsystems (populated in initialise())
        self._calibration: Optional[CameraCalibration] = None
        self._stereo_cal: Optional[StereoCalibration] = None
        self._segmenter: Optional[YOLOSegmenter] = None
        self._embedding_head: Optional[EmbeddingHead] = None
        self._depth_estimator: Optional[StereoDepthEstimator] = None
        self._landmark_extractor: Optional[LandmarkExtractor] = None
        self._tilm: Optional[TILM] = None
        self._matcher: Optional[TemporalMatcher] = None
        self._ekf: Optional[EKFVIO] = None
        self._imu_preint: Optional[IMUPreintegrator] = None
        self._corrector: Optional[MatchingCorrector] = None
        self._path_follower: Optional[PathFollower] = None
        self._corridor: Optional[AdaptiveCorridor] = None
        self._mission: Optional[MissionManager] = None
        self._monitor: Optional[SystemMonitor] = None

        self._gnss_active: bool = True
        self._frame_count: int = 0

    def initialise(self) -> None:
        """Load all subsystems and prepare the pipeline for processing.

        Raises:
            FileNotFoundError: If a required weight/config file is missing.
            RuntimeError: If YOLO weights are required but not provided.
        """
        logger.info("NavigationPipeline: initialising subsystems")

        # ── Camera calibration ─────────────────────────────────────────────
        if self.config.camera_calibration_path and self.config.camera_calibration_path.exists():
            self._calibration = CameraCalibration.load(self.config.camera_calibration_path)
        else:
            self._calibration = CameraCalibration.from_midair_defaults()
            logger.debug("Using MidAir default camera calibration")

        if self.config.stereo_calibration_path and self.config.stereo_calibration_path.exists():
            self._stereo_cal = StereoCalibration.load(self.config.stereo_calibration_path)
        else:
            right = CameraCalibration(
                fx=self._calibration.fx,
                fy=self._calibration.fy,
                cx=self._calibration.cx,
                cy=self._calibration.cy,
                width=self._calibration.width,
                height=self._calibration.height,
                camera_id="right",
            )
            self._stereo_cal = StereoCalibration(
                left=self._calibration,
                right=right,
                R=np.eye(3),
                T=np.array([-0.12, 0.0, 0.0]),
            )

        # ── YOLO segmenter ─────────────────────────────────────────────────
        if self.config.yolo_weights_path and self.config.yolo_weights_path.exists():
            self._segmenter = YOLOSegmenter(device=self.config.device)
            self._segmenter.load(self.config.yolo_weights_path)
            logger.info("YOLO loaded from {}", self.config.yolo_weights_path)
        else:
            logger.warning("YOLO weights not found — perception disabled")

        # ── Embedding head ─────────────────────────────────────────────────
        if self.config.embedding_weights_path and self.config.embedding_weights_path.exists():
            self._embedding_head = EmbeddingHead.load(
                self.config.embedding_weights_path, device=self.config.device
            )
            logger.info("EmbeddingHead loaded from {}", self.config.embedding_weights_path)
        else:
            self._embedding_head = EmbeddingHead()
            logger.warning("Embedding weights not found — using random init")

        # ── Stereo depth ───────────────────────────────────────────────────
        self._depth_estimator = StereoDepthEstimator(calibration=self._stereo_cal)
        self._depth_estimator.load()

        # ── Landmark extractor ─────────────────────────────────────────────
        self._landmark_extractor = LandmarkExtractor(
            embedding_head=self._embedding_head,
            camera_K=self._calibration.K,
        )

        # ── TILM + matcher ─────────────────────────────────────────────────
        if self.config.tilm_path and self.config.tilm_path.exists():
            self._tilm = TILM.load(self.config.tilm_path)
            logger.info("TILM loaded: {} nodes", self._tilm.n_nodes())
        else:
            self._tilm = TILM()
            logger.warning("No TILM file — starting with empty map")

        self._matcher = TemporalMatcher(self._tilm)
        self._matcher.build_descriptor_index()

        # ── EKF ───────────────────────────────────────────────────────────
        initial_state = EKFState(
            position=np.zeros(3, dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            bias_acc=np.zeros(3, dtype=np.float64),
            bias_gyro=np.zeros(3, dtype=np.float64),
            P=np.eye(18, dtype=np.float64) * 1e-2,
            is_gnss_active=True,
        )
        self._ekf = EKFVIO(initial_state)

        # ── IMU preintegrator ──────────────────────────────────────────────
        self._imu_preint = IMUPreintegrator(
            noise_acc=0.1,
            noise_gyro=0.01,
            noise_bias_acc=0.001,
            noise_bias_gyro=0.0001,
        )
        self._imu_preint.reset(np.zeros(3), np.zeros(3))

        # ── Matching corrector ─────────────────────────────────────────────
        self._corrector = MatchingCorrector(
            matcher=self._matcher,
            ekf=self._ekf,
            min_confidence=0.3,
            correction_interval=1.0,
        )

        # ── Path follower (optional) ───────────────────────────────────────
        waypoints: Optional[np.ndarray] = None
        if self.config.waypoints_path and self.config.waypoints_path.exists():
            wp_path = self.config.waypoints_path
            if wp_path.suffix == ".npy":
                waypoints = np.load(wp_path)
            else:
                waypoints = np.loadtxt(wp_path, delimiter=",")

        if waypoints is not None and len(waypoints):
            self._path_follower = PathFollower(
                reference_path=waypoints,
                config=self._pf_config,
            )
            self._corridor = AdaptiveCorridor(tilm=self._tilm)

        # ── Mission manager ────────────────────────────────────────────────
        self._mission = MissionManager()

        # ── Monitor ───────────────────────────────────────────────────────
        self._monitor = SystemMonitor()

        self._initialised = True
        logger.info("NavigationPipeline: ready")

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
            imu_acc: Accelerometer measurement (3,), m/s².
            imu_gyro: Gyroscope measurement (3,), rad/s.
            dt: Time since last frame, seconds.
            gnss_position: Optional GNSS NED position (3,).
            timestamp: Frame timestamp in seconds.

        Returns:
            Dictionary with keys ``"position"``, ``"mode"``,
            ``"n_landmarks"``, ``"tilm_quality"``, ``"latency_ms"``.

        Raises:
            RuntimeError: If ``initialise()`` has not been called.
        """
        if not self._initialised:
            raise RuntimeError("Call initialise() before process_frame().")

        t_start = time.perf_counter()
        self._frame_count += 1

        # ── 1. IMU propagation ─────────────────────────────────────────────
        state = self._ekf.get_state()  # type: ignore[union-attr]
        self._imu_preint.reset(state.bias_acc, state.bias_gyro)  # type: ignore[union-attr]
        self._imu_preint.integrate(  # type: ignore[union-attr]
            np.asarray(imu_acc, dtype=np.float64),
            np.asarray(imu_gyro, dtype=np.float64),
            max(dt, 1e-4),
        )
        self._ekf.propagate(self._imu_preint.get_state())  # type: ignore[union-attr]

        # ── 2. Depth estimation ────────────────────────────────────────────
        depth_result: Optional[DepthResult] = None
        if image_right is not None and self._depth_estimator is not None:
            try:
                depth_result = self._depth_estimator.estimate(image_rgb, image_right)
            except Exception as exc:
                logger.debug("Depth estimation failed: {}", exc)

        # ── 3. Segmentation ────────────────────────────────────────────────
        seg_result: Optional[SegmentationResult] = None
        if self._segmenter is not None:
            try:
                seg_result = self._segmenter.predict(image_rgb)
            except Exception as exc:
                logger.debug("Segmentation failed: {}", exc)

        # ── 4. Landmark extraction ─────────────────────────────────────────
        landmarks = []
        if seg_result is not None and depth_result is not None and self._landmark_extractor is not None:
            try:
                landmarks = self._landmark_extractor.extract(
                    image_rgb=image_rgb,
                    segmentation=seg_result,
                    depth=depth_result,
                    frame_id=f"frame_{self._frame_count:06d}",
                )
            except Exception as exc:
                logger.debug("Landmark extraction failed: {}", exc)

        # ── 5. GNSS / TILM update ──────────────────────────────────────────
        if gnss_position is not None and self._gnss_active:
            self._corrector.set_gnss_available(True)  # type: ignore[union-attr]
        else:
            self._corrector.set_gnss_available(False)  # type: ignore[union-attr]

        match_result = self._corrector.update(  # type: ignore[union-attr]
            observations=landmarks,
            timestamp=timestamp,
            gnss_position=gnss_position,
            gnss_accuracy=self.config.gnss_noise_std,
        )

        # ── 6. Mission mode ────────────────────────────────────────────────
        mode = self._mission.current_mode if self._mission else MissionMode.NORMAL  # type: ignore[attr-defined]

        # ── 7. Monitor update ──────────────────────────────────────────────
        if self._monitor:
            self._monitor.update(
                n_landmarks=len(landmarks),
                tilm_match=match_result,
                ekf_state=self._ekf.get_state(),
            )

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        current_pos = self._ekf.get_state().position  # type: ignore[union-attr]
        quality = (match_result.confidence if (match_result is not None and match_result.is_valid) else 0.0)

        return {
            "position": current_pos.copy(),
            "mode": str(mode),
            "n_landmarks": len(landmarks),
            "tilm_quality": quality,
            "latency_ms": latency_ms,
            "gnss_active": self._gnss_active,
        }

    def set_gnss_active(self, active: bool) -> None:
        """Switch GNSS availability (e.g., to simulate GPS loss).

        Args:
            active: True = GNSS available; False = GNSS lost.
        """
        self._gnss_active = active
        if self._corrector is not None:
            self._corrector.set_gnss_available(active)
        logger.info("GNSS {}", "active" if active else "lost")

    def shutdown(self) -> None:
        """Gracefully shut down all subsystems and flush logs."""
        logger.info("NavigationPipeline: shutting down (processed {} frames)", self._frame_count)
        self._initialised = False

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
        return self._mission.current_mode  # type: ignore[attr-defined]
