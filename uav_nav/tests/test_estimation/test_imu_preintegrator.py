"""Unit tests for IMUPreintegrator and PreintegratedState."""

from __future__ import annotations

import numpy as np
import pytest

from uav_nav.estimation.imu_preintegrator import IMUPreintegrator, PreintegratedState
from uav_nav.estimation.pose_utils import so3_exp


@pytest.fixture
def integrator() -> IMUPreintegrator:
    return IMUPreintegrator(noise_acc=0.1, noise_gyro=0.01)


class TestReset:
    def test_reset_zeros_deltas(self, integrator: IMUPreintegrator) -> None:
        integrator.integrate(np.array([0.0, 0.0, 9.81]), np.zeros(3), 0.01)
        integrator.reset(np.zeros(3), np.zeros(3))
        s = integrator.get_state()
        np.testing.assert_allclose(s.delta_R, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(s.delta_v, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(s.delta_p, np.zeros(3), atol=1e-12)
        assert s.dt == 0.0
        assert s.n_samples == 0

    def test_reset_stores_bias(self, integrator: IMUPreintegrator) -> None:
        ba = np.array([0.01, -0.02, 0.03])
        bg = np.array([0.001, 0.002, -0.001])
        integrator.reset(ba, bg)
        s = integrator.get_state()
        np.testing.assert_allclose(s.bias_acc, ba)
        np.testing.assert_allclose(s.bias_gyro, bg)

    def test_reset_zeros_cov(self, integrator: IMUPreintegrator) -> None:
        integrator.integrate(np.ones(3), np.ones(3), 0.1)
        integrator.reset(np.zeros(3), np.zeros(3))
        np.testing.assert_allclose(integrator.get_state().cov, np.zeros((9, 9)), atol=1e-30)


class TestIntegrate:
    def test_zero_gyro_no_rotation(self, integrator: IMUPreintegrator) -> None:
        integrator.reset(np.zeros(3), np.zeros(3))
        for _ in range(10):
            integrator.integrate(np.array([0.0, 0.0, 9.81]), np.zeros(3), 0.01)
        s = integrator.get_state()
        np.testing.assert_allclose(s.delta_R, np.eye(3), atol=1e-10)

    def test_rotation_around_z(self, integrator: IMUPreintegrator) -> None:
        integrator.reset(np.zeros(3), np.zeros(3))
        omega = np.array([0.0, 0.0, np.pi / 2])
        dt = 0.01
        n = 100  # 1 second → 90° rotation around Z
        for _ in range(n):
            integrator.integrate(np.zeros(3), omega, dt)
        s = integrator.get_state()
        expected = so3_exp(np.array([0.0, 0.0, np.pi / 2]))
        np.testing.assert_allclose(s.delta_R, expected, atol=1e-2)

    def test_acc_integrates_to_velocity(self, integrator: IMUPreintegrator) -> None:
        integrator.reset(np.zeros(3), np.zeros(3))
        a = np.array([1.0, 0.0, 0.0])
        dt = 0.01
        n = 100  # 1 second
        for _ in range(n):
            integrator.integrate(a, np.zeros(3), dt)
        s = integrator.get_state()
        # delta_v ≈ a * T = 1.0 m/s in X (gravity is not in acc here)
        np.testing.assert_allclose(s.delta_v[0], 1.0, atol=1e-2)

    def test_cov_grows_with_integration(self, integrator: IMUPreintegrator) -> None:
        integrator.reset(np.zeros(3), np.zeros(3))
        cov_before = integrator.get_state().cov.copy()
        for _ in range(100):
            integrator.integrate(np.zeros(3), np.zeros(3), 0.01)
        cov_after = integrator.get_state().cov
        assert np.trace(cov_after) > np.trace(cov_before)

    def test_sample_count_increments(self, integrator: IMUPreintegrator) -> None:
        integrator.reset(np.zeros(3), np.zeros(3))
        for _ in range(5):
            integrator.integrate(np.zeros(3), np.zeros(3), 0.01)
        assert integrator.get_state().n_samples == 5

    def test_dt_accumulates(self, integrator: IMUPreintegrator) -> None:
        integrator.reset(np.zeros(3), np.zeros(3))
        for _ in range(10):
            integrator.integrate(np.zeros(3), np.zeros(3), 0.01)
        assert abs(integrator.get_state().dt - 0.1) < 1e-10


class TestPredictPose:
    def test_stationary_gravity_effect(self, integrator: IMUPreintegrator) -> None:
        # No IMU → delta_* = 0, but gravity accumulates in predict_pose
        integrator.reset(np.zeros(3), np.zeros(3))
        R0 = np.eye(3)
        v0 = np.zeros(3)
        p0 = np.zeros(3)
        # After reset, dt=0 → no gravity effect
        R1, v1, p1 = integrator.predict_pose(R0, v0, p0)
        np.testing.assert_allclose(R1, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(v1, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(p1, np.zeros(3), atol=1e-12)

    def test_predict_with_forward_acc(self, integrator: IMUPreintegrator) -> None:
        integrator.reset(np.zeros(3), np.zeros(3))
        a = np.array([1.0, 0.0, 0.0])
        dt = 0.01
        for _ in range(100):  # 1 second
            integrator.integrate(a, np.zeros(3), dt)
        R0, v0, p0 = np.eye(3), np.zeros(3), np.zeros(3)
        R1, v1, p1 = integrator.predict_pose(R0, v0, p0)
        # p ≈ 0.5 * a * T² = 0.5 m in X (gravity in Z in NED)
        assert p1[0] > 0.4

    def test_rotation_is_body_to_ned(self, integrator: IMUPreintegrator) -> None:
        integrator.reset(np.zeros(3), np.zeros(3))
        omega = np.array([0.0, 0.0, np.pi / 2])
        for _ in range(100):
            integrator.integrate(np.zeros(3), omega, 0.01)
        R1, _, _ = integrator.predict_pose(np.eye(3), np.zeros(3), np.zeros(3))
        # R1 should be a valid rotation matrix
        np.testing.assert_allclose(R1 @ R1.T, np.eye(3), atol=1e-6)
        assert abs(np.linalg.det(R1) - 1.0) < 1e-6


class TestBiasCorrection:
    def test_correct_for_bias_update_returns_new_state(self, integrator: IMUPreintegrator) -> None:
        integrator.reset(np.zeros(3), np.zeros(3))
        for _ in range(50):
            integrator.integrate(np.array([0.0, 0.0, 9.81]), np.zeros(3), 0.01)
        s = integrator.get_state()
        corrected = s.correct_for_bias_update(np.zeros(3), np.zeros(3))
        # Zero delta_bias → state unchanged
        np.testing.assert_allclose(corrected.delta_p, s.delta_p, atol=1e-12)
        np.testing.assert_allclose(corrected.delta_v, s.delta_v, atol=1e-12)

    def test_bias_update_changes_delta_v(self, integrator: IMUPreintegrator) -> None:
        integrator.reset(np.zeros(3), np.zeros(3))
        for _ in range(100):
            integrator.integrate(np.array([1.0, 0.0, 0.0]), np.zeros(3), 0.01)
        s = integrator.get_state()
        # Small acc bias change should shift delta_v
        delta_ba = np.array([0.1, 0.0, 0.0])
        corrected = s.correct_for_bias_update(delta_ba, np.zeros(3))
        assert not np.allclose(corrected.delta_v, s.delta_v)
