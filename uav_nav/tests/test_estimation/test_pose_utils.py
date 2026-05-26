"""Unit tests for pose utility functions.

Tests functions in uav_nav.estimation.pose_utils that can be verified
without requiring the full pipeline to be implemented.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_nav.estimation.pose_utils import (
    skew,
    so3_exp,
    quat_to_rot,
    ned_to_enu,
    enu_to_ned,
    compute_relative_pose,
)


class TestSkew:
    """Tests for the skew-symmetric matrix function."""

    def test_shape(self) -> None:
        """skew() returns a (3, 3) matrix."""
        v = np.array([1.0, 2.0, 3.0])
        K = skew(v)
        assert K.shape == (3, 3)

    def test_antisymmetric(self) -> None:
        """skew(v) satisfies K + K.T == 0."""
        v = np.array([1.0, -2.0, 3.0])
        K = skew(v)
        np.testing.assert_allclose(K + K.T, np.zeros((3, 3)), atol=1e-12)

    def test_cross_product(self) -> None:
        """skew(a) @ b equals a × b."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        expected = np.cross(a, b)
        result = skew(a) @ b
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_zero_vector(self) -> None:
        """skew(0) is the zero matrix."""
        K = skew(np.zeros(3))
        np.testing.assert_allclose(K, np.zeros((3, 3)), atol=1e-12)


class TestSO3Exp:
    """Tests for the SO(3) exponential map."""

    def test_identity_for_zero(self) -> None:
        """so3_exp(0) returns the identity matrix."""
        R = so3_exp(np.zeros(3))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_orthogonality(self) -> None:
        """so3_exp(v) satisfies R @ R.T == I."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            v = rng.uniform(-np.pi, np.pi, size=(3,))
            R = so3_exp(v)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_determinant_one(self) -> None:
        """so3_exp(v) has det == +1."""
        v = np.array([0.1, 0.2, 0.3])
        R = so3_exp(v)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_90_degree_rotation_x(self) -> None:
        """so3_exp([π/2, 0, 0]) rotates (0,1,0) → (0,0,1)."""
        R = so3_exp(np.array([np.pi / 2, 0.0, 0.0]))
        result = R @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(result, np.array([0.0, 0.0, 1.0]), atol=1e-10)


class TestQuatToRot:
    """Tests for quaternion to rotation matrix conversion."""

    def test_identity_quaternion(self) -> None:
        """quat_to_rot([1,0,0,0]) returns the identity matrix."""
        R = quat_to_rot(np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_orthogonality(self) -> None:
        """quat_to_rot(q) produces an orthogonal matrix."""
        q = np.array([0.707, 0.0, 0.707, 0.0])
        q /= np.linalg.norm(q)
        R = quat_to_rot(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


class TestCoordinateFrames:
    """Tests for NED ↔ ENU frame conversions."""

    def test_ned_to_enu_roundtrip(self) -> None:
        """ned_to_enu followed by enu_to_ned returns original vector."""
        v_ned = np.array([10.0, -5.0, 3.0])
        v_enu = ned_to_enu(v_ned)
        v_back = enu_to_ned(v_enu)
        np.testing.assert_allclose(v_back, v_ned, atol=1e-12)

    def test_ned_to_enu_axes(self) -> None:
        """NED (1, 0, 0) → ENU (0, 1, 0) (North → East becomes Y)."""
        ned = np.array([1.0, 0.0, 0.0])
        enu = ned_to_enu(ned)
        np.testing.assert_allclose(enu, np.array([0.0, 1.0, 0.0]), atol=1e-12)

    def test_down_to_up(self) -> None:
        """NED (0, 0, 1) → ENU (0, 0, -1) (Down → negated Up)."""
        ned = np.array([0.0, 0.0, 1.0])
        enu = ned_to_enu(ned)
        np.testing.assert_allclose(enu, np.array([0.0, 0.0, -1.0]), atol=1e-12)


class TestRelativePose:
    """Tests for relative SE(3) pose computation."""

    def test_identity_relative(self) -> None:
        """Relative pose of T with itself is the identity."""
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        rel = compute_relative_pose(T, T)
        np.testing.assert_allclose(rel, np.eye(4), atol=1e-10)

    def test_known_translation(self) -> None:
        """Relative pose of two pure-translation matrices gives correct offset."""
        T_a = np.eye(4)
        T_a[0, 3] = 5.0
        T_b = np.eye(4)
        T_b[0, 3] = 8.0
        rel = compute_relative_pose(T_a, T_b)
        expected = np.eye(4)
        expected[0, 3] = 3.0
        np.testing.assert_allclose(rel, expected, atol=1e-10)
