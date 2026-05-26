"""Extended Kalman Filter for Visual-Inertial Odometry.

Fuses IMU pre-integrated states, visual feature tracks, semantic
landmark position corrections, and optional GNSS measurements into
a consistent pose and bias estimate.

State vector (18-DoF):
    p_nb_n  (3): Position of body in NED frame.
    v_nb_n  (3): Velocity of body in NED frame.
    q_nb    (4): Quaternion (body-to-NED rotation), stored as (w,x,y,z).
    b_a     (3): Accelerometer bias.
    b_g     (3): Gyroscope bias.
    b_r     (3): Reserved (e.g. barometer bias).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from uav_nav.estimation.imu_preintegrator import PreintegratedState


@dataclass
class EKFState:
    """Full EKF state with mean and covariance.

    Attributes:
        position: NED position, shape (3,), float64.
        velocity: NED velocity, shape (3,), float64.
        quaternion: Body-to-NED quaternion (w,x,y,z), shape (4,), float64.
        bias_acc: Accelerometer bias, shape (3,), float64.
        bias_gyro: Gyroscope bias, shape (3,), float64.
        P: State covariance matrix, shape (18, 18), float64.
        timestamp: State timestamp in seconds.
        is_gnss_active: Whether GNSS corrections are currently used.
    """

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    bias_acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bias_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    P: np.ndarray = field(default_factory=lambda: np.eye(18) * 1e-3)
    timestamp: float = 0.0
    is_gnss_active: bool = True

    def rotation_matrix(self) -> np.ndarray:
        """Convert stored quaternion to a 3x3 rotation matrix.

        Returns:
            Rotation matrix body-to-NED, shape (3, 3).

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("EKFState.rotation_matrix is not yet implemented.")

    def to_pose_matrix(self) -> np.ndarray:
        """Return full SE(3) pose as a 4x4 homogeneous matrix.

        Returns:
            Pose matrix, shape (4, 4), float64.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("EKFState.to_pose_matrix is not yet implemented.")


class EKFVIO:
    """EKF-based Visual-Inertial Odometry estimator.

    Supports four update types:
    - IMU propagation via pre-integrated states.
    - Visual update from tracked feature reprojection error.
    - Semantic landmark update from TILM matching.
    - GNSS update from GPS position fix.

    Args:
        initial_state: Initial EKFState (position, attitude, covariance).
        noise_acc: Accelerometer noise density.
        noise_gyro: Gyroscope noise density.
        noise_bias_acc: Accelerometer bias process noise.
        noise_bias_gyro: Gyroscope bias process noise.
        R_visual: Visual measurement noise covariance (2x2 per feature).
        R_landmark: Landmark position measurement noise covariance (3x3).
        R_gnss: GNSS position measurement noise covariance (3x3).
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
    ) -> None:
        self.state = initial_state
        self.noise_acc = noise_acc
        self.noise_gyro = noise_gyro
        self.noise_bias_acc = noise_bias_acc
        self.noise_bias_gyro = noise_bias_gyro
        self.R_visual = R_visual if R_visual is not None else np.eye(2) * 0.5
        self.R_landmark = R_landmark if R_landmark is not None else np.eye(3) * 1.0
        self.R_gnss = R_gnss if R_gnss is not None else np.eye(3) * 2.0

    def propagate(self, preintegrated: PreintegratedState) -> None:
        """Propagate the EKF state using a pre-integrated IMU state.

        Args:
            preintegrated: Pre-integrated state from IMUPreintegrator.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("EKFVIO.propagate is not yet implemented.")

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
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("EKFVIO.update_visual is not yet implemented.")

    def update_landmark(
        self,
        position_correction_ned: np.ndarray,
        correction_covariance: Optional[np.ndarray] = None,
    ) -> None:
        """Apply a position correction from TILM landmark matching.

        Args:
            position_correction_ned: Correction vector, shape (3,), NED metres.
            correction_covariance: Optional 3x3 covariance override.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("EKFVIO.update_landmark is not yet implemented.")

    def update_gnss(
        self,
        position_ned: np.ndarray,
        accuracy: float = 3.0,
    ) -> None:
        """Apply a GNSS position fix update.

        Args:
            position_ned: GNSS position in NED frame, shape (3,).
            accuracy: Reported GNSS horizontal accuracy in metres (1-sigma).

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("EKFVIO.update_gnss is not yet implemented.")

    def get_state(self) -> EKFState:
        """Return a copy of the current filter state.

        Returns:
            Current EKFState instance.
        """
        import copy
        return copy.deepcopy(self.state)

    def set_gnss_active(self, active: bool) -> None:
        """Enable or disable the GNSS update channel.

        Args:
            active: If False, GNSS updates are silently ignored.
        """
        self.state.is_gnss_active = active
