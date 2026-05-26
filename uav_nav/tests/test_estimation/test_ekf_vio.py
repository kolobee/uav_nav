"""Unit tests for EKFState and EKFVIO."""

from __future__ import annotations

import numpy as np
import pytest

from uav_nav.estimation.ekf_vio import EKFState, EKFVIO
from uav_nav.estimation.imu_preintegrator import IMUPreintegrator


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_ekf(p0=None, v0=None, q0=None) -> EKFVIO:
    state = EKFState(
        position=np.array(p0 or [0., 0., 0.], dtype=np.float64),
        velocity=np.array(v0 or [0., 0., 0.], dtype=np.float64),
        quaternion=np.array(q0 or [1., 0., 0., 0.], dtype=np.float64),
        P=np.eye(18, dtype=np.float64) * 1e-2,
    )
    return EKFVIO(state)


def _integrate_imu(n: int = 100, dt: float = 0.01,
                    acc=None, gyro=None) -> "PreintegratedState":
    integrator = IMUPreintegrator()
    integrator.reset(np.zeros(3), np.zeros(3))
    a = np.array(acc or [0.0, 0.0, 0.0], dtype=np.float64)
    w = np.array(gyro or [0.0, 0.0, 0.0], dtype=np.float64)
    for _ in range(n):
        integrator.integrate(a, w, dt)
    return integrator.get_state()


# ── EKFState tests ────────────────────────────────────────────────────────────

class TestEKFState:
    def test_rotation_matrix_identity_quat(self) -> None:
        s = EKFState()
        R = s.rotation_matrix()
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_rotation_matrix_is_orthogonal(self) -> None:
        # 90° rotation around Z axis: q = [cos45°, 0, 0, sin45°]
        c, sv = np.cos(np.pi / 4), np.sin(np.pi / 4)
        s = EKFState(quaternion=np.array([c, 0., 0., sv]))
        R = s.rotation_matrix()
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_to_pose_matrix_shape(self) -> None:
        s = EKFState(position=np.array([1., 2., 3.]))
        T = s.to_pose_matrix()
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T[3, :], [0., 0., 0., 1.])

    def test_to_pose_matrix_translation(self) -> None:
        p = np.array([10., -5., 3.])
        s = EKFState(position=p)
        T = s.to_pose_matrix()
        np.testing.assert_allclose(T[:3, 3], p)

    def test_to_pose_matrix_rotation_block(self) -> None:
        s = EKFState()
        T = s.to_pose_matrix()
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)


# ── EKFVIO propagation tests ──────────────────────────────────────────────────

class TestPropagation:
    def test_propagate_changes_position(self) -> None:
        ekf = _make_ekf()
        pis = _integrate_imu(100, 0.01, acc=[1., 0., 0.])
        p_before = ekf.state.position.copy()
        ekf.propagate(pis)
        assert not np.allclose(ekf.state.position, p_before)

    def test_propagate_gravity_increases_z_position(self) -> None:
        # Stationary IMU with zero acc (gravity not compensated in body frame here)
        # After 1s, position Z should increase due to gravity in predict_pose
        ekf = _make_ekf()
        pis = _integrate_imu(100, 0.01, acc=[0., 0., 0.])
        ekf.propagate(pis)
        # NED: z positive = down; gravity [0,0,9.81] → p_z should be > 0
        assert ekf.state.position[2] > 0.0

    def test_propagate_increases_cov_trace(self) -> None:
        ekf = _make_ekf()
        pis = _integrate_imu(100, 0.01)
        trace_before = np.trace(ekf.state.P)
        ekf.propagate(pis)
        assert np.trace(ekf.state.P) > trace_before

    def test_propagate_zero_dt_noop(self) -> None:
        ekf = _make_ekf()
        integrator = IMUPreintegrator()
        integrator.reset(np.zeros(3), np.zeros(3))
        pis = integrator.get_state()  # dt=0, n_samples=0
        p_before = ekf.state.position.copy()
        ekf.propagate(pis)
        np.testing.assert_allclose(ekf.state.position, p_before)

    def test_propagate_updates_timestamp(self) -> None:
        ekf = _make_ekf()
        pis = _integrate_imu(100, 0.01)
        ekf.propagate(pis)
        assert abs(ekf.state.timestamp - 1.0) < 1e-9

    def test_propagate_quaternion_stays_unit(self) -> None:
        ekf = _make_ekf()
        pis = _integrate_imu(100, 0.01, gyro=[0.1, 0.05, 0.0])
        ekf.propagate(pis)
        norm = float(np.linalg.norm(ekf.state.quaternion))
        assert abs(norm - 1.0) < 1e-6


# ── GNSS update tests ─────────────────────────────────────────────────────────

class TestGNSSUpdate:
    def test_gnss_update_moves_position_toward_measurement(self) -> None:
        ekf = _make_ekf(p0=[0., 0., 0.])
        z = np.array([10., 0., 0.])
        ekf.update_gnss(z, accuracy=3.0)
        # Position should shift toward z
        assert ekf.state.position[0] > 0.0

    def test_gnss_update_reduces_position_uncertainty(self) -> None:
        ekf = _make_ekf()
        # Large initial position uncertainty
        ekf.state.P[0:3, 0:3] = np.eye(3) * 100.0
        trace_before = np.trace(ekf.state.P[0:3, 0:3])
        ekf.update_gnss(np.zeros(3), accuracy=3.0)
        assert np.trace(ekf.state.P[0:3, 0:3]) < trace_before

    def test_gnss_inactive_ignores_update(self) -> None:
        ekf = _make_ekf(p0=[0., 0., 0.])
        ekf.set_gnss_active(False)
        p_before = ekf.state.position.copy()
        ekf.update_gnss(np.array([100., 0., 0.]), accuracy=3.0)
        np.testing.assert_allclose(ekf.state.position, p_before)

    def test_gnss_reactivate(self) -> None:
        ekf = _make_ekf(p0=[0., 0., 0.])
        ekf.set_gnss_active(False)
        ekf.set_gnss_active(True)
        ekf.update_gnss(np.array([10., 0., 0.]), accuracy=3.0)
        assert ekf.state.position[0] > 0.0

    def test_get_state_returns_copy(self) -> None:
        ekf = _make_ekf(p0=[1., 2., 3.])
        s = ekf.get_state()
        s.position[0] = 999.0
        assert ekf.state.position[0] != 999.0


# ── Landmark update tests ─────────────────────────────────────────────────────

class TestLandmarkUpdate:
    def test_landmark_applies_positive_correction(self) -> None:
        ekf = _make_ekf(p0=[0., 0., 0.])
        correction = np.array([5., 0., 0.])
        ekf.update_landmark(correction)
        assert ekf.state.position[0] > 0.0

    def test_landmark_uses_custom_covariance(self) -> None:
        ekf1 = _make_ekf()
        ekf2 = _make_ekf()
        correction = np.array([1., 0., 0.])
        ekf1.update_landmark(correction, correction_covariance=np.eye(3) * 0.01)
        ekf2.update_landmark(correction, correction_covariance=np.eye(3) * 100.0)
        # Low noise → stronger correction (closer to measurement)
        assert ekf1.state.position[0] > ekf2.state.position[0]

    def test_zero_correction_noop(self) -> None:
        ekf = _make_ekf(p0=[1., 2., 3.])
        p_before = ekf.state.position.copy()
        # A zero correction means z = p_nominal → innovation = 0
        ekf.update_landmark(np.zeros(3))
        np.testing.assert_allclose(ekf.state.position, p_before, atol=1e-10)


# ── Visual update ─────────────────────────────────────────────────────────────

class TestVisualUpdate:
    def test_update_visual_raises(self) -> None:
        ekf = _make_ekf()
        with pytest.raises(NotImplementedError):
            ekf.update_visual(np.zeros((5, 2)), np.zeros((5, 2)), np.eye(3))
