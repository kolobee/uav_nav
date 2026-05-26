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
    """Logarithm map from SO(3) to so(3) (inverse Rodrigues).

    Args:
        R: Rotation matrix, shape (3, 3).

    Returns:
        Rotation vector (axis-angle), shape (3,).
    """
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        return np.zeros(3, dtype=np.float64)
    if np.pi - theta < 1e-6:
        # Near 180°: extract axis from symmetric part (R + Rᵀ)/2 = I + cos(θ)(I - ω⊗ω)
        B = (R + R.T) / 2.0 - cos_theta * np.eye(3)
        i = int(np.argmax(np.diag(B)))
        denom = float(B[i, i] * (1.0 - cos_theta))
        omega = B[:, i] / np.sqrt(max(denom, 1e-30))
        return theta * omega / float(np.linalg.norm(omega))
    factor = theta / (2.0 * np.sin(theta))
    return factor * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """Exponential map from se(3) to SE(3).

    Args:
        xi: Twist vector [rho (3), omega (3)], shape (6,).

    Returns:
        Homogeneous transform matrix, shape (4, 4).
    """
    rho = xi[:3]
    omega = xi[3:]
    R = so3_exp(omega)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-9:
        V = np.eye(3)
    else:
        K = skew(omega / theta)
        V = (np.eye(3)
             + (1.0 - np.cos(theta)) / theta * K
             + (theta - np.sin(theta)) / theta * (K @ K))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


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
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / float(np.linalg.norm(q))


def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two unit quaternions (w, x, y, z).

    Args:
        q1: First quaternion, shape (4,).
        q2: Second quaternion, shape (4,).

    Returns:
        Product quaternion, shape (4,), unit norm.
    """
    w1, x1, y1, z1 = float(q1[0]), float(q1[1]), float(q1[2]), float(q1[3])
    w2, x2, y2, z2 = float(q2[0]), float(q2[1]), float(q2[2]), float(q2[3])
    q = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)
    return q / float(np.linalg.norm(q))


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
    """
    p = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
    R0 = T0[:3, :3]
    R1 = T1[:3, :3]
    omega = so3_log(R0.T @ R1)
    R_interp = R0 @ so3_exp(alpha * omega)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_interp
    T[:3, 3] = p
    return T


def compute_relative_pose(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """Compute the relative SE(3) transform T_a_to_b = T_a^{-1} @ T_b.

    Args:
        T_a: Pose A, shape (4, 4).
        T_b: Pose B, shape (4, 4).

    Returns:
        Relative transform, shape (4, 4).
    """
    return np.linalg.inv(T_a) @ T_b
