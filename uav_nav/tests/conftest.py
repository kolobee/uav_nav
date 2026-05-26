"""Shared pytest fixtures for the uav_nav test suite.

Provides synthetic sensor data, minimal calibration, and minimal TILM
instances that can be reused across all test modules without requiring
real dataset files.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_nav.data.calibration import CameraCalibration, StereoCalibration
from uav_nav.estimation.ekf_vio import EKFState
from uav_nav.estimation.imu_preintegrator import IMUPreintegrator, PreintegratedState
from uav_nav.memory.tilm import TILM, TILMNode
from uav_nav.perception.landmark_extractor import Landmark


# ── Camera and Calibration Fixtures ──────────────────────────────────────────

@pytest.fixture()
def camera_calibration() -> CameraCalibration:
    """Return a minimal CameraCalibration with MidAir default values.

    Returns:
        CameraCalibration instance ready for use in tests.
    """
    return CameraCalibration.from_midair_defaults()


@pytest.fixture()
def stereo_calibration(camera_calibration: CameraCalibration) -> StereoCalibration:
    """Return a minimal StereoCalibration with 12 cm baseline.

    Args:
        camera_calibration: Left camera calibration (from fixture).

    Returns:
        StereoCalibration instance.
    """
    right_cal = CameraCalibration(
        fx=camera_calibration.fx,
        fy=camera_calibration.fy,
        cx=camera_calibration.cx,
        cy=camera_calibration.cy,
        width=camera_calibration.width,
        height=camera_calibration.height,
        camera_id="midair_right",
    )
    return StereoCalibration(
        left=camera_calibration,
        right=right_cal,
        R=np.eye(3),
        T=np.array([-0.12, 0.0, 0.0]),
    )


# ── Synthetic Image Fixtures ──────────────────────────────────────────────────

@pytest.fixture()
def synthetic_rgb_frame() -> np.ndarray:
    """Return a synthetic 512×512 RGB image with random pixel values.

    Returns:
        Array of shape (512, 512, 3), dtype uint8.
    """
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 255, size=(512, 512, 3), dtype=np.uint8)


@pytest.fixture()
def synthetic_depth_frame() -> np.ndarray:
    """Return a synthetic 512×512 depth map with values in [5, 100] m.

    Returns:
        Array of shape (512, 512), dtype float32.
    """
    rng = np.random.default_rng(seed=1)
    return rng.uniform(5.0, 100.0, size=(512, 512)).astype(np.float32)


@pytest.fixture()
def synthetic_segmentation_frame() -> np.ndarray:
    """Return a synthetic 512×512 segmentation mask with 4 classes (0–3).

    Returns:
        Array of shape (512, 512), dtype uint8.
    """
    rng = np.random.default_rng(seed=2)
    return rng.integers(0, 4, size=(512, 512), dtype=np.uint8)


# ── Synthetic IMU Data Fixtures ───────────────────────────────────────────────

@pytest.fixture()
def synthetic_imu_batch() -> list[dict[str, object]]:
    """Return a batch of 100 synthetic IMU measurements at 200 Hz.

    Returns:
        List of dicts with keys "acc" (3,), "gyro" (3,), "dt" (float).
    """
    rng = np.random.default_rng(seed=3)
    measurements = []
    for _ in range(100):
        measurements.append({
            "acc":  rng.normal([0.0, 0.0, 9.81], 0.05, size=(3,)).astype(np.float64),
            "gyro": rng.normal([0.0, 0.0, 0.0], 0.005, size=(3,)).astype(np.float64),
            "dt":   0.005,  # 200 Hz
        })
    return measurements


@pytest.fixture()
def imu_preintegrator() -> IMUPreintegrator:
    """Return an IMUPreintegrator with default MidAir-like noise params.

    Returns:
        IMUPreintegrator instance (not yet reset to a keyframe).
    """
    return IMUPreintegrator(
        noise_acc=0.1,
        noise_gyro=0.01,
        noise_bias_acc=0.001,
        noise_bias_gyro=0.0001,
    )


# ── EKF State Fixtures ────────────────────────────────────────────────────────

@pytest.fixture()
def initial_ekf_state() -> EKFState:
    """Return a default-initialised EKFState at the origin.

    Returns:
        EKFState with zero position and identity attitude.
    """
    return EKFState(
        position=np.zeros(3),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        bias_acc=np.zeros(3),
        bias_gyro=np.zeros(3),
        P=np.eye(18) * 1e-3,
        timestamp=0.0,
        is_gnss_active=True,
    )


# ── Synthetic Landmark Fixtures ───────────────────────────────────────────────

@pytest.fixture()
def synthetic_landmarks() -> list[Landmark]:
    """Return a list of 5 synthetic Landmark observations.

    Returns:
        List of 5 Landmark instances with random positions and descriptors.
    """
    rng = np.random.default_rng(seed=4)
    classes = ["tree", "tree", "rock", "river", "tree"]
    class_ids = [1, 1, 2, 3, 1]
    landmarks = []
    for i, (cls, cid) in enumerate(zip(classes, class_ids)):
        desc = rng.standard_normal(128).astype(np.float32)
        desc /= np.linalg.norm(desc)
        landmarks.append(
            Landmark(
                position_3d=rng.uniform(-50, 50, size=(3,)).astype(np.float32),
                semantic_class=cls,
                class_id=cid,
                descriptor=desc,
                confidence=float(rng.uniform(0.5, 0.95)),
                mask_area=int(rng.integers(500, 5000)),
                centroid_uv=(float(rng.uniform(50, 450)), float(rng.uniform(50, 450))),
                frame_id="synthetic_0",
                instance_id=i,
            )
        )
    return landmarks


# ── Minimal TILM Fixture ──────────────────────────────────────────────────────

@pytest.fixture()
def minimal_tilm(synthetic_landmarks: list[Landmark]) -> TILM:
    """Return a minimal TILM with 3 nodes connected in a chain.

    Nodes are placed at NED positions (0, 0, -50), (10, 0, -50), (20, 0, -50).

    Args:
        synthetic_landmarks: Synthetic landmarks from fixture.

    Returns:
        Initialised TILM with 3 nodes and 2 directed edges.

    Note:
        This fixture calls TILM methods that are marked NotImplementedError.
        Mark tests using this fixture with ``pytest.mark.skip`` until the
        TILM implementation is complete.
    """
    tilm = TILM(min_node_distance=5.0)
    # NOTE: tilm.initialise() and add_node() raise NotImplementedError until
    # implemented. Tests that use this fixture should be skipped or marked xfail.
    return tilm


# ── Miscellaneous Fixtures ────────────────────────────────────────────────────

@pytest.fixture()
def ned_waypoints() -> np.ndarray:
    """Return a straight 500 m north NED path with 10 waypoints.

    Returns:
        Array of shape (10, 3), NED metres, altitude -50 m.
    """
    waypoints = np.zeros((10, 3))
    waypoints[:, 0] = np.linspace(0, 500, 10)
    waypoints[:, 2] = -50.0
    return waypoints
