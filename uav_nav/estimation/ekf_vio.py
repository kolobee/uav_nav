"""Extended Kalman Filter for Visual-Inertial Odometry.

Fuses IMU pre-integrated states, semantic landmark position corrections,
and optional GNSS measurements into a consistent pose and bias estimate.

Implementation: Error-State EKF (ESKF), 15-DoF active error state
plus 3 reserved DoF → 18×18 covariance matrix P.

Error state layout (indices into P):
    0:3   δp   — position error (NED, m)
    3:6   δv   — velocity error (NED, m/s)
    6:9   δθ   — attitude error (rotation vector, rad)
    9:12  δba  — accelerometer bias error (m/s²)
    12:15 δbg  — gyroscope bias error (rad/s)
    15:18 δbr  — reserved (near-zero, for future barometer bias)

Nominal state is stored in EKFState (position, velocity, quaternion, biases).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from uav_nav.estimation.imu_preintegrator import PreintegratedState
from uav_nav.estimation.pose_utils import quat_to_rot, rot_to_quat, quat_mult, skew, so3_exp

_G_NED = np.array([0.0, 0.0, 9.81], dtype=np.float64)

# ── EKF state ─────────────────────────────────────────────────────────────────

@dataclass
class EKFState:
    """Full EKF state with mean and covariance.

    Attributes:
        position: NED position, shape (3,), float64.
        velocity: NED velocity, shape (3,), float64.
        quaternion: Body-to-NED quaternion (w,x,y,z), shape (4,), float64.
        bias_acc: Accelerometer bias, shape (3,), float64.
        bias_gyro: Gyroscope bias, shape (3,), float64.
        P: Error-state covariance matrix, shape (18, 18), float64.
        timestamp: State timestamp in seconds.
        is_gnss_active: Whether GNSS corrections are currently used.
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1., 0., 0., 0.], dtype=np.float64)
    )
    bias_acc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    bias_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    P: np.ndarray = field(default_factory=lambda: np.eye(18, dtype=np.float64) * 1e-3)
    timestamp: float = 0.0
    is_gnss_active: bool = True

    def rotation_matrix(self) -> np.ndarray:
        """Convert stored quaternion to a 3×3 rotation matrix (body-to-NED).

        Returns:
            Rotation matrix, shape (3, 3).
        """
        return quat_to_rot(self.quaternion)

    def to_pose_matrix(self) -> np.ndarray:
        """Return full SE(3) pose as a 4×4 homogeneous matrix.

        Returns:
            Pose matrix [R | p; 0 | 1], shape (4, 4), float64.
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.rotation_matrix()
        T[:3, 3] = self.position
        return T


# ── EKF VIO ───────────────────────────────────────────────────────────────────

class EKFVIO:
    """Error-State EKF for Visual-Inertial Odometry.

    Supports three update types:
    - IMU propagation via pre-integrated states (propagate).
    - Semantic landmark position correction from TILM (update_landmark).
    - GNSS position fix (update_gnss).

    The visual feature update (update_visual) is reserved for future work.

    Args:
        initial_state: Initial EKFState (position, attitude, covariance).
        noise_acc: Accelerometer noise density.
        noise_gyro: Gyroscope noise density.
        noise_bias_acc: Accelerometer bias process noise.
        noise_bias_gyro: Gyroscope bias process noise.
        R_visual: Visual measurement noise covariance (2×2 per feature).
        R_landmark: Landmark position measurement noise covariance (3×3).
        R_gnss: GNSS position measurement noise covariance (3×3).
        gravity_ned: Gravity vector in NED frame, shape (3,).
    """

    def __init__(
        self,
        initial_state: EKFState,
        noise_acc: float = 0.1,
        noise_gyro: float = 0.01,
        noise_bias_acc: float = 0.001,
        noise_bias_gyro: float = 0.0001,
        R_visual: Optional[np.ndarray] = None,
        R_landmark: Optional[np.ndarray] = None,
        R_gnss: Optional[np.ndarray] = None,
        gravity_ned: Optional[np.ndarray] = None,
    ) -> None:
        self.state = initial_state
        self.noise_acc = noise_acc
        self.noise_gyro = noise_gyro
        self.noise_bias_acc = noise_bias_acc
        self.noise_bias_gyro = noise_bias_gyro
        self.R_visual = R_visual if R_visual is not None else np.eye(2) * 0.5
        self.R_landmark = R_landmark if R_landmark is not None else np.eye(3) * 1.0
        self.R_gnss = R_gnss if R_gnss is not None else np.eye(3) * 4.0
        self.gravity_ned = (
            gravity_ned if gravity_ned is not None else _G_NED.copy()
        )

    # ── Propagation ───────────────────────────────────────────────────────────

    def propagate(self, preintegrated: PreintegratedState) -> None:
        """Propagate the EKF state using a pre-integrated IMU state.

        Updates nominal state via kinematic equations and propagates the
        error-state covariance using a linearised state transition matrix F.

        Args:
            preintegrated: Pre-integrated state from IMUPreintegrator.
        """
        if preintegrated.dt <= 0.0 or preintegrated.n_samples == 0:
            return

        s = self.state
        pis = preintegrated
        dt = pis.dt

        R0 = quat_to_rot(s.quaternion)
        g = self.gravity_ned

        # ── Nominal state prediction ──────────────────────────────────────────
        p_new = s.position + s.velocity * dt + 0.5 * g * dt**2 + R0 @ pis.delta_p
        v_new = s.velocity + g * dt + R0 @ pis.delta_v
        R_new = R0 @ pis.delta_R
        q_new = rot_to_quat(R_new)
        # biases: random walk (unchanged by prediction)

        # ── State transition Jacobian F (18×18, error state) ─────────────────
        # Linearization of ESKF around current nominal state.
        # Active 15×15 block (δp, δv, δθ, δba, δbg):
        F = np.eye(18, dtype=np.float64)

        # δp_new depends on δv (velocity integrates to position)
        F[0:3, 3:6] = np.eye(3) * dt

        # δv_new depends on δθ (rotation error mixes gravity-subtracted acc)
        F[3:6, 6:9] = -R0 @ skew(pis.delta_v)

        # δv_new depends on δba (acc bias enters via R0)
        F[3:6, 9:12] = -R0 * dt

        # δθ_new = delta_R.T @ δθ (rotation error propagates via pre-integrated rotation)
        F[6:9, 6:9] = pis.delta_R.T

        # δθ_new depends on δbg (gyro bias integrates to rotation error)
        F[6:9, 12:15] = -np.eye(3) * dt

        # ── Process noise Q (18×18) ───────────────────────────────────────────
        # From preintegrated.cov [Δp(0:3), Δv(3:6), Δθ(6:9)] + bias random walk
        Q = np.zeros((18, 18), dtype=np.float64)
        Q[0:3, 0:3] = pis.cov[0:3, 0:3]
        Q[3:6, 3:6] = pis.cov[3:6, 3:6]
        Q[6:9, 6:9] = pis.cov[6:9, 6:9]
        Q[9:12, 9:12] = self.noise_bias_acc**2 * dt * np.eye(3)
        Q[12:15, 12:15] = self.noise_bias_gyro**2 * dt * np.eye(3)
        Q[15:18, 15:18] = 1e-10 * np.eye(3)

        # ── Covariance prediction ─────────────────────────────────────────────
        s.P = F @ s.P @ F.T + Q

        # ── Apply nominal state update ────────────────────────────────────────
        s.position = p_new
        s.velocity = v_new
        s.quaternion = q_new
        s.timestamp += dt

    # ── Update helpers ────────────────────────────────────────────────────────

    def _position_update(
        self,
        z: np.ndarray,
        R_meas: np.ndarray,
    ) -> None:
        """Standard EKF position update (measurement = NED position).

        H = [I₃ | 0 | 0 | 0 | 0 | 0]  (3×18)

        Args:
            z: Measured NED position, shape (3,).
            R_meas: Measurement noise covariance, shape (3, 3).
        """
        s = self.state
        H = np.zeros((3, 18), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)

        innovation = z - s.position                  # (3,)
        S = H @ s.P @ H.T + R_meas                  # (3, 3)
        K = s.P @ H.T @ np.linalg.inv(S)            # (18, 3)
        delta_x = K @ innovation                     # (18,)

        # Apply error-state correction to nominal state
        s.position += delta_x[0:3]
        s.velocity += delta_x[3:6]

        # Small-angle quaternion update: δq ≈ [1, δθ/2]
        dtheta = delta_x[6:9]
        delta_q = np.concatenate([[1.0], dtheta / 2.0])
        delta_q /= float(np.linalg.norm(delta_q))
        s.quaternion = quat_mult(s.quaternion, delta_q)

        s.bias_acc += delta_x[9:12]
        s.bias_gyro += delta_x[12:15]

        # Joseph form for numerical stability: P = (I - K H) P (I - K H)ᵀ + K R Kᵀ
        IKH = np.eye(18) - K @ H
        s.P = IKH @ s.P @ IKH.T + K @ R_meas @ K.T

    # ── Public update interfaces ──────────────────────────────────────────────

    def update_visual(
        self,
        keypoints_curr: np.ndarray,
        keypoints_prev: np.ndarray,
        K: np.ndarray,
    ) -> None:
        """Apply visual feature reprojection error correction.

        Args:
            keypoints_curr: Current frame keypoints, shape (N, 2).
            keypoints_prev: Matched previous frame keypoints, shape (N, 2).
            K: Camera intrinsic matrix, shape (3, 3).

        Raises:
            NotImplementedError: Visual update not implemented in this version.
        """
        raise NotImplementedError("EKFVIO.update_visual is not yet implemented.")

    def update_landmark(
        self,
        position_correction_ned: np.ndarray,
        correction_covariance: Optional[np.ndarray] = None,
    ) -> None:
        """Apply a position correction from TILM landmark matching.

        Interprets position_correction_ned as the innovation (difference
        between map-predicted position and current estimate). Applies a
        standard Kalman position update with the landmark noise model.

        Args:
            position_correction_ned: Correction vector, shape (3,), NED metres.
            correction_covariance: Optional 3×3 covariance override.
        """
        R_meas = (
            correction_covariance if correction_covariance is not None
            else self.R_landmark
        )
        z = self.state.position + np.asarray(position_correction_ned, dtype=np.float64)
        self._position_update(z, R_meas)

    def update_gnss(
        self,
        position_ned: np.ndarray,
        accuracy: float = 3.0,
    ) -> None:
        """Apply a GNSS position fix update.

        Args:
            position_ned: GNSS position in NED frame, shape (3,).
            accuracy: Reported GNSS horizontal accuracy in metres (1-sigma).
        """
        if not self.state.is_gnss_active:
            return
        R_gnss = self.R_gnss.copy()
        R_gnss[0, 0] = accuracy**2
        R_gnss[1, 1] = accuracy**2
        self._position_update(
            np.asarray(position_ned, dtype=np.float64), R_gnss
        )

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_state(self) -> EKFState:
        """Return a deep copy of the current filter state.

        Returns:
            Current EKFState instance.
        """
        return copy.deepcopy(self.state)

    def set_gnss_active(self, active: bool) -> None:
        """Enable or disable the GNSS update channel.

        Args:
            active: If False, update_gnss calls are silently ignored.
        """
        self.state.is_gnss_active = active
