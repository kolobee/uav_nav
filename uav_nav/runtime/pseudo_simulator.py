"""Pseudo-simulator for offline two-phase navigation experiments.

Replays a MidAir dataset sequence through the navigation pipeline in two
phases:

1. **Mapping phase** (GNSS active): EKF is corrected via GT GPS, and
   ``TILMBuilder`` accumulates landmarks into a topological map.
2. **Navigation phase** (GNSS lost): EKF is corrected only by TILM matching
   (``MatchingCorrector``).  A parallel dead-reckoning EKF runs with IMU only
   to quantify the drift that TILM corrections suppress.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from loguru import logger

from uav_nav.data.calibration import CameraCalibration
from uav_nav.data.midair_loader import IMUSample, MidAirLoader, MidAirSample
from uav_nav.estimation.ekf_vio import EKFVIO, EKFState
from uav_nav.estimation.imu_preintegrator import IMUPreintegrator
from uav_nav.memory.tilm import TILM
from uav_nav.memory.tilm_builder import TILMBuilder
from uav_nav.memory.temporal_matcher import TemporalMatcher
from uav_nav.perception.embedding_head import EmbeddingHead
from uav_nav.perception.landmark_extractor import Landmark, LandmarkExtractor
from uav_nav.perception.stereo_depth import DepthResult
from uav_nav.perception.yolo_segmenter import SegmentationResult, YOLOSegmenter
from uav_nav.runtime.matching_corrector import CorrectorStats, MatchingCorrector


# ── configuration ─────────────────────────────────────────────────────────────

@dataclass
class GNSSOutageConfig:
    """Configuration for a synthetic GNSS outage injection.

    Attributes:
        start_time: Outage start time in seconds.
        duration: Outage duration in seconds.
        noise_std: Gaussian noise std applied before outage (m). Zero = perfect.
    """

    start_time: float
    duration: float
    noise_std: float = 0.0


@dataclass
class SimulationConfig:
    """Top-level simulation parameters.

    Attributes:
        gnss_loss_fraction: Fraction of trajectory at which GNSS is lost
            (0.5 = midpoint).
        gnss_noise_std: GNSS measurement noise std (m) during mapping phase.
        frame_stride: Step size through the loader (1 = every frame).
        imu_noise_acc: Accelerometer noise std added on top of GT IMU (m/s²).
        imu_noise_gyro: Gyroscope noise std added on top of GT IMU (rad/s).
        min_landmarks_for_node: Minimum landmarks to create a TILM node.
        tilm_min_node_distance: Minimum travel (m) between TILM nodes.
    """

    gnss_loss_fraction: float = 0.5
    gnss_noise_std: float = 1.0
    frame_stride: int = 1
    imu_noise_acc: float = 0.02
    imu_noise_gyro: float = 0.002
    min_landmarks_for_node: int = 1
    tilm_min_node_distance: float = 5.0


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Full results from a two-phase pseudo-simulation run.

    Attributes:
        timestamps: Frame timestamps, shape (N,).
        gt_positions: Ground-truth NED positions, shape (N, 3).
        estimated_positions: EKF+TILM positions, shape (N, 3).
        dead_reckoning_positions: IMU-only positions, shape (N, 3).
        gnss_active_mask: True where GNSS was available, shape (N,).
        tilm_qualities: TILM match quality (0–1) per frame, shape (N,).
        gnss_loss_idx: Index of the first navigation-phase frame.
        tilm_node_count: Number of nodes in the built TILM.
        corrector_stats: Stats from ``MatchingCorrector``.
        sequence_id: Identifier of the source sequence.
    """

    timestamps: np.ndarray = field(default_factory=lambda: np.empty(0))
    gt_positions: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    estimated_positions: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    dead_reckoning_positions: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    gnss_active_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    tilm_qualities: np.ndarray = field(default_factory=lambda: np.empty(0))
    gnss_loss_idx: int = 0
    tilm_node_count: int = 0
    corrector_stats: Optional[CorrectorStats] = None
    sequence_id: str = ""

    # ── convenience properties ─────────────────────────────────────────────

    @property
    def position_errors_tilm(self) -> np.ndarray:
        """L2 error (m) between EKF+TILM and GT, shape (N,)."""
        return np.linalg.norm(self.estimated_positions - self.gt_positions, axis=1)

    @property
    def position_errors_dr(self) -> np.ndarray:
        """L2 error (m) between dead reckoning and GT, shape (N,)."""
        return np.linalg.norm(self.dead_reckoning_positions - self.gt_positions, axis=1)

    @property
    def ate_tilm(self) -> float:
        """ATE RMSE for the TILM-corrected trajectory (m)."""
        e = self.position_errors_tilm
        return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")

    @property
    def ate_dr(self) -> float:
        """ATE RMSE for the dead-reckoning trajectory (m)."""
        e = self.position_errors_dr
        return float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")

    @property
    def improvement_pct(self) -> float:
        """Percentage ATE improvement of TILM over dead reckoning."""
        if self.ate_dr == 0.0 or np.isnan(self.ate_dr):
            return 0.0
        return (self.ate_dr - self.ate_tilm) / self.ate_dr * 100.0

    # back-compat aliases (used in existing stub references)
    @property
    def positions_est(self) -> np.ndarray:
        return self.estimated_positions

    @property
    def positions_gt(self) -> np.ndarray:
        return self.gt_positions

    @property
    def position_errors(self) -> np.ndarray:
        return self.position_errors_tilm

    @property
    def rmse(self) -> float:
        return self.ate_tilm

    @property
    def max_error(self) -> float:
        e = self.position_errors_tilm
        return float(np.max(e)) if len(e) else float("nan")

    @property
    def modes(self) -> list[str]:
        mask = self.gnss_active_mask
        return ["mapping" if g else "navigation" for g in mask]


# ── simulator ─────────────────────────────────────────────────────────────────

class PseudoSimulator:
    """Two-phase offline simulator over MidAir GT data.

    Args:
        loader: Initialised ``MidAirLoader`` with index already built.
        calibration: Camera intrinsics used for 3-D landmark back-projection.
        embedding_head: Optional trained ``EmbeddingHead``.  If ``None``,
            a randomly-initialised head is used (sufficient for structural
            testing; TILM matching quality will be poor).
        yolo_segmenter: Optional ``YOLOSegmenter``.  If ``None``, GT
            segmentation from MidAir is used as observations.
        config: Simulation parameters.
        outages: Legacy per-outage config (overrides ``config.gnss_loss_fraction``
            if provided).
        progress_callback: Called with ``(frame_idx, total)`` on each frame.
    """

    # Label-map path relative to the package root
    _LABEL_MAP_PATH = Path(__file__).parent.parent / "configs" / "label_map.json"

    def __init__(
        self,
        loader: MidAirLoader,
        calibration: CameraCalibration,
        embedding_head: Optional[EmbeddingHead] = None,
        yolo_segmenter: Optional[YOLOSegmenter] = None,
        config: Optional[SimulationConfig] = None,
        outages: Optional[list[GNSSOutageConfig]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self.loader = loader
        self.calibration = calibration
        self.embedding_head = embedding_head
        self.yolo_segmenter = yolo_segmenter
        self.config = config or SimulationConfig()
        self.outages = outages or []
        self.progress_callback = progress_callback

        self._rng = np.random.default_rng(seed=42)
        self._label_map: dict[int, str] = {}
        self._class_to_id: dict[str, int] = {}
        self._load_label_map()

    # ── public interface ───────────────────────────────────────────────────

    def run(self) -> SimulationResult:
        """Run the full two-phase simulation.

        Returns:
            SimulationResult with trajectories, errors, and TILM stats.

        Raises:
            RuntimeError: If the loader has no frames.
        """
        frames = list(self._iter_frames())
        if not frames:
            raise RuntimeError("MidAirLoader returned no frames.")

        gnss_loss_idx = int(len(frames) * self.config.gnss_loss_fraction)
        gnss_loss_idx = max(1, gnss_loss_idx)

        logger.info(
            "Simulation: {} frames total, GNSS lost at frame {} ({:.0f}%)",
            len(frames),
            gnss_loss_idx,
            self.config.gnss_loss_fraction * 100,
        )

        # ── initialise shared components ──────────────────────────────────
        embedding_head = self.embedding_head or _make_random_embedding_head()
        landmark_extractor = LandmarkExtractor(
            embedding_head=embedding_head,
            camera_K=self.calibration.K,
        )

        tilm = TILM(min_node_distance=self.config.tilm_min_node_distance)
        tilm_builder = TILMBuilder(tilm)

        initial_pose = _gt_pose_or_identity(frames[0])
        initial_pos = initial_pose[:3, 3]

        main_ekf = _make_ekf(initial_pos, timestamp=frames[0].timestamp)
        dr_ekf = _make_ekf(initial_pos, timestamp=frames[0].timestamp)
        main_preint = _make_preintegrator()
        dr_preint = _make_preintegrator()

        # ── result accumulators ───────────────────────────────────────────
        n = len(frames)
        timestamps = np.zeros(n)
        gt_positions = np.zeros((n, 3))
        estimated_positions = np.zeros((n, 3))
        dead_reckoning_positions = np.zeros((n, 3))
        gnss_active_mask = np.zeros(n, dtype=bool)
        tilm_qualities = np.zeros(n)

        # ── Phase 1: mapping ──────────────────────────────────────────────
        logger.info("Phase 1: mapping ({} frames)", gnss_loss_idx)

        for i, sample in enumerate(frames[:gnss_loss_idx]):
            if self.progress_callback:
                self.progress_callback(i, n)

            dt = _dt(frames, i)
            imu_samples = _get_imu(sample)
            _propagate(main_preint, main_ekf, imu_samples, dt, self.config)
            _propagate(dr_preint, dr_ekf, imu_samples, dt, self.config)

            gt_pos = _gt_position(sample)
            gnss_pos = gt_pos + self._rng.normal(0.0, self.config.gnss_noise_std, 3)
            main_ekf.update_gnss(gnss_pos, accuracy=self.config.gnss_noise_std)
            dr_ekf.update_gnss(gnss_pos, accuracy=self.config.gnss_noise_std)

            landmarks = self._get_landmarks(sample, landmark_extractor)
            gt_pose = _gt_pose_or_identity(sample)
            tilm_builder.process_frame(
                pose_ned=gt_pose,
                landmarks=landmarks,
                timestamp=sample.timestamp,
                keyframe_id=f"frame_{i:05d}",
            )

            timestamps[i] = sample.timestamp
            gt_positions[i] = gt_pos
            estimated_positions[i] = main_ekf.get_state().position
            dead_reckoning_positions[i] = dr_ekf.get_state().position
            gnss_active_mask[i] = True

        # ── Build TILM index ───────────────────────────────────────────────
        tilm = tilm_builder.finalise()
        logger.info("TILM built: {} nodes", tilm.n_nodes())

        matcher = TemporalMatcher(tilm)
        matcher.build_descriptor_index()

        corrector = MatchingCorrector(matcher=matcher, ekf=main_ekf)
        corrector.set_gnss_available(False)

        # Sync dead reckoning EKF state at GNSS loss point
        main_state = main_ekf.get_state()
        dr_ekf_synced = _make_ekf(main_state.position, timestamp=main_state.timestamp)

        # Copy bias estimates so IMU integration starts from the same point
        dr_state = dr_ekf_synced.get_state()
        dr_state.bias_acc = main_state.bias_acc.copy()
        dr_state.bias_gyro = main_state.bias_gyro.copy()
        dr_preint2 = _make_preintegrator()
        dr_preint2.reset(main_state.bias_acc, main_state.bias_gyro)

        # ── Phase 2: navigation ────────────────────────────────────────────
        logger.info("Phase 2: navigation ({} frames)", n - gnss_loss_idx)

        for j, sample in enumerate(frames[gnss_loss_idx:]):
            i = gnss_loss_idx + j
            if self.progress_callback:
                self.progress_callback(i, n)

            dt = _dt(frames, i)
            imu_samples = _get_imu(sample)
            _propagate(main_preint, main_ekf, imu_samples, dt, self.config)
            _propagate(dr_preint2, dr_ekf_synced, imu_samples, dt, self.config)

            landmarks = self._get_landmarks(sample, landmark_extractor)
            match_result = corrector.update(
                observations=landmarks,
                timestamp=sample.timestamp,
                gnss_position=None,
            )

            quality = match_result.confidence if (match_result is not None and match_result.is_valid) else 0.0

            timestamps[i] = sample.timestamp
            gt_positions[i] = _gt_position(sample)
            estimated_positions[i] = main_ekf.get_state().position
            dead_reckoning_positions[i] = dr_ekf_synced.get_state().position
            gnss_active_mask[i] = False
            tilm_qualities[i] = quality

        result = SimulationResult(
            timestamps=timestamps,
            gt_positions=gt_positions,
            estimated_positions=estimated_positions,
            dead_reckoning_positions=dead_reckoning_positions,
            gnss_active_mask=gnss_active_mask,
            tilm_qualities=tilm_qualities,
            gnss_loss_idx=gnss_loss_idx,
            tilm_node_count=tilm.n_nodes(),
            corrector_stats=corrector.stats,
            sequence_id=frames[0].sequence_id if frames else "",
        )

        logger.info(
            "Simulation done — ATE TILM: {:.2f} m | ATE DR: {:.2f} m | improvement: {:.1f}%",
            result.ate_tilm,
            result.ate_dr,
            result.improvement_pct,
        )
        return result

    # ── private helpers ────────────────────────────────────────────────────

    def _iter_frames(self):
        """Yield frames at the configured stride."""
        stride = max(1, self.config.frame_stride)
        for i in range(0, len(self.loader), stride):
            yield self.loader[i]

    def _get_landmarks(
        self,
        sample: MidAirSample,
        extractor: LandmarkExtractor,
    ) -> list[Landmark]:
        """Extract landmarks using YOLO or GT segmentation."""
        depth_result = _sample_to_depth(sample, self.calibration)
        if depth_result is None:
            return []

        if self.yolo_segmenter is not None and sample.image_rgb is not None:
            try:
                seg = self.yolo_segmenter.predict(sample.image_rgb)
            except Exception:
                seg = _empty_seg(sample)
        elif sample.segmentation is not None:
            seg = self._gt_seg_to_result(sample)
        else:
            return []

        if seg.n_instances == 0:
            return []

        return extractor.extract(
            image_rgb=sample.image_rgb,
            segmentation=seg,
            depth=depth_result,
            frame_id=f"{sample.sequence_id}_{sample.frame_idx:05d}",
        )

    def _gt_seg_to_result(self, sample: MidAirSample) -> SegmentationResult:
        """Convert MidAir GT segmentation map to per-instance SegmentationResult."""
        seg_map = sample.segmentation
        if seg_map is None:
            return _empty_seg(sample)

        H, W = seg_map.shape
        masks: list[np.ndarray] = []
        class_ids: list[int] = []
        class_names_out: list[str] = []
        scores: list[float] = []
        boxes: list[list[float]] = []

        for raw_id, cls_name in self._label_map.items():
            if cls_name == "background":
                continue
            cls_id = self._class_to_id.get(cls_name, 0)
            binary = (seg_map == raw_id).astype(np.uint8)
            if binary.sum() == 0:
                continue
            n_comp, labels = cv2.connectedComponents(binary)
            for comp_id in range(1, n_comp):
                comp_mask = labels == comp_id
                if int(comp_mask.sum()) < 200:
                    continue
                rows, cols = np.where(comp_mask)
                x1, y1 = int(cols.min()), int(rows.min())
                x2, y2 = int(cols.max()), int(rows.max())
                masks.append(comp_mask)
                class_ids.append(cls_id)
                class_names_out.append(cls_name)
                scores.append(1.0)
                boxes.append([float(x1), float(y1), float(x2), float(y2)])

        if not masks:
            return _empty_seg(sample)

        return SegmentationResult(
            masks=np.stack(masks, axis=0),
            class_ids=np.array(class_ids, dtype=int),
            class_names=class_names_out,
            scores=np.array(scores, dtype=np.float32),
            boxes=np.array(boxes, dtype=np.float32),
            image_hw=(H, W),
        )

    def _load_label_map(self) -> None:
        """Load label_map.json for class-ID mapping."""
        if not self._LABEL_MAP_PATH.exists():
            logger.warning("label_map.json not found at {}", self._LABEL_MAP_PATH)
            return
        data = json.loads(self._LABEL_MAP_PATH.read_text())
        self._label_map = {int(k): v for k, v in data["label_map"].items()}
        self._class_to_id = {v: int(k) for k, v in data["id_to_class"].items()}

    def _is_gnss_active(self, timestamp: float) -> bool:
        """Return False if timestamp falls inside any configured outage."""
        for outage in self.outages:
            if outage.start_time <= timestamp < outage.start_time + outage.duration:
                return False
        return True


# ── module-level helpers ───────────────────────────────────────────────────────

def _gt_position(sample: MidAirSample) -> np.ndarray:
    if sample.pose_gt is not None:
        return sample.pose_gt[:3, 3].astype(np.float64)
    return np.zeros(3, dtype=np.float64)


def _gt_pose_or_identity(sample: MidAirSample) -> np.ndarray:
    if sample.pose_gt is not None:
        return np.asarray(sample.pose_gt, dtype=np.float64)
    return np.eye(4, dtype=np.float64)


def _dt(frames: list[MidAirSample], i: int) -> float:
    if i == 0:
        return 0.1
    return max(float(frames[i].timestamp - frames[i - 1].timestamp), 1e-4)


def _get_imu(sample: MidAirSample) -> list[IMUSample]:
    return sample.imu_samples or []


def _make_ekf(position: np.ndarray, timestamp: float = 0.0) -> EKFVIO:
    state = EKFState(
        position=np.asarray(position, dtype=np.float64).copy(),
        velocity=np.zeros(3, dtype=np.float64),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        bias_acc=np.zeros(3, dtype=np.float64),
        bias_gyro=np.zeros(3, dtype=np.float64),
        P=np.eye(18, dtype=np.float64) * 1e-2,
        timestamp=timestamp,
        is_gnss_active=True,
    )
    return EKFVIO(state)


def _make_preintegrator() -> IMUPreintegrator:
    return IMUPreintegrator(
        noise_acc=0.1,
        noise_gyro=0.01,
        noise_bias_acc=0.001,
        noise_bias_gyro=0.0001,
    )


def _propagate(
    preint: IMUPreintegrator,
    ekf: EKFVIO,
    imu_samples: list[IMUSample],
    dt_frame: float,
    cfg: SimulationConfig,
) -> None:
    """Preintegrate IMU samples and propagate EKF."""
    state = ekf.get_state()
    preint.reset(state.bias_acc, state.bias_gyro)

    if imu_samples:
        for s in imu_samples:
            dt_imu = max(float(getattr(s, "dt", 0.005)), 1e-5)
            acc = np.asarray(s.acc, dtype=np.float64)
            gyro = np.asarray(s.gyro, dtype=np.float64)
            # Add simulation noise
            if cfg.imu_noise_acc > 0:
                acc = acc + np.random.normal(0, cfg.imu_noise_acc, 3)
            if cfg.imu_noise_gyro > 0:
                gyro = gyro + np.random.normal(0, cfg.imu_noise_gyro, 3)
            preint.integrate(acc, gyro, dt_imu)
    else:
        # Gravity-aligned IMU if no samples (static assumption)
        acc = np.array([0.0, 0.0, 9.81])
        preint.integrate(acc, np.zeros(3), dt_frame)

    ekf.propagate(preint.get_state())


def _sample_to_depth(
    sample: MidAirSample,
    calibration: CameraCalibration,
) -> Optional[DepthResult]:
    """Convert MidAir GT depth array to DepthResult."""
    if sample.depth is None:
        return None
    depth = np.asarray(sample.depth, dtype=np.float32)
    # Synthetic disparity: d = f*B/z (B = 0.12 m baseline, f = fx)
    f = float(calibration.fx)
    B = 0.12
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = np.where(depth > 0, f * B / depth, 0.0).astype(np.float32)
    conf = np.ones_like(depth, dtype=np.float32)
    conf[~np.isfinite(depth) | (depth <= 0)] = 0.0
    return DepthResult(
        disparity=disp,
        depth=depth,
        confidence=conf,
        baseline=B,
        focal_length=f,
    )


def _empty_seg(sample: MidAirSample) -> SegmentationResult:
    H, W = (sample.image_rgb.shape[:2] if sample.image_rgb is not None else (512, 512))
    return SegmentationResult(
        masks=np.zeros((0, H, W), dtype=bool),
        class_ids=np.zeros(0, dtype=int),
        class_names=[],
        scores=np.zeros(0, dtype=np.float32),
        boxes=np.zeros((0, 4), dtype=np.float32),
        image_hw=(H, W),
    )


def _make_random_embedding_head() -> EmbeddingHead:
    """Return a randomly-initialised EmbeddingHead (no trained weights)."""
    return EmbeddingHead()
