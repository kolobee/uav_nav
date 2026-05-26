"""Pose and rotation utility functions.

Provides SO(3) and SE(3) exponential/logarithm maps, quaternion conversions,
and coordinate frame transformations used throughout the estimation stack.
"""

from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix of a 3-vector.

    Args:
        v: Input vector, shape (3,).

    Returns:
        Skew-symmetric matrix, shape (3, 3).
    """
    return np.array([
        [0.0,  -v[2],  v[1]],
        [v[2],  0.0,  -v[0]],
        [-v[1], v[0],  0.0 ],
    ], dtype=np.float64)


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Exponential map from so(3) to SO(3) (Rodrigues' formula).

    Args:
        omega: Rotation vector (axis-angle), shape (3,).

    Returns:
        Rotation matrix, shape (3, 3).
    """
    theta = float(np.linalg.norm(omega))
    if theta < 1e-9:
        return np.eye(3) + skew(omega)
    axis = omega / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithm map from SO(3) to so(3).

    Args:
        R: Rotation matrix, shape (3, 3).

    Returns:
        Rotation vector, shape (3,).

    Raises:
        NotImplementedError: Not yet implemented for edge cases.
    """
    raise NotImplementedError("so3_log is not yet implemented.")


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map from se(3) to SE(3).

    Args:
        xi: Twist vector [rho (3), omega (3)], shape (6,).

    Returns:
        Homogeneous transform matrix, shape (4, 4).

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError("se3_exp is not yet implemented.")


def se3_log(T: np.ndarray) -> np.ndarray:
    """Logarithm map from SE(3) to se(3).

    Args:
        T: Homogeneous transform matrix, shape (4, 4).

    Returns:
        Twist vector [rho (3), omega (3)], shape (6,).

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError("se3_log is not yet implemented.")


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion to a rotation matrix.

    Args:
        q: Quaternion (w, x, y, z), shape (4,). Must be unit norm.

    Returns:
        Rotation matrix, shape (3, 3).
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*w*z,     2*x*z + 2*w*y  ],
        [2*x*y + 2*w*z,        1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x  ],
        [2*x*z - 2*w*y,        2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a unit quaternion (w, x, y, z).

    Uses Shepperd's method for numerical stability.

    Args:
        R: Rotation matrix, shape (3, 3).

    Returns:
        Unit quaternion (w, x, y, z), shape (4,).

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError("rot_to_quat is not yet implemented.")


def ned_to_enu(ned: np.ndarray) -> np.ndarray:
    """Convert a position/velocity vector from NED to ENU frame.

    Args:
        ned: Vector in NED frame, shape (3,) or (..., 3).

    Returns:
        Vector in ENU frame, same shape.
    """
    if ned.ndim == 1:
        return np.array([ned[1], ned[0], -ned[2]], dtype=np.float64)
    return np.stack([ned[..., 1], ned[..., 0], -ned[..., 2]], axis=-1)


def enu_to_ned(enu: np.ndarray) -> np.ndarray:
    """Convert a position/velocity vector from ENU to NED frame.

    Args:
        enu: Vector in ENU frame, shape (3,) or (..., 3).

    Returns:
        Vector in NED frame, same shape.
    """
    if enu.ndim == 1:
        return np.array([enu[1], enu[0], -enu[2]], dtype=np.float64)
    return np.stack([enu[..., 1], enu[..., 0], -enu[..., 2]], axis=-1)


def interpolate_poses(
    T0: np.ndarray,
    T1: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Linearly interpolate between two SE(3) poses.

    Uses slerp for the rotation component.

    Args:
        T0: Start pose, shape (4, 4).
        T1: End pose, shape (4, 4).
        alpha: Interpolation factor in [0, 1]. 0 returns T0, 1 returns T1.

    Returns:
        Interpolated pose, shape (4, 4).

    Raises:
        NotImplementedError: Not yet implemented.
    """
    raise NotImplementedError("interpolate_poses is not yet implemented.")


def compute_relative_pose(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """Compute the relative SE(3) transform T_a_to_b = T_a^{-1} @ T_b.

    Args:
        T_a: Pose A, shape (4, 4).
        T_b: Pose B, shape (4, 4).

    Returns:
        Relative transform, shape (4, 4).
    """
    return np.linalg.inv(T_a) @ T_b
