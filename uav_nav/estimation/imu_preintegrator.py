"""IMU pre-integration on the SO(3) manifold.

Implements the IMU pre-integration theory from Forster et al. (2017)
"On-Manifold Preintegration for Real-Time Visual-Inertial Odometry"
(IEEE TRO). Pre-integrates accelerometer and gyroscope measurements
between two keyframes without re-integrating from the start.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PreintegratedState:
    """Pre-integrated IMU state between two keyframes.

    Represents the change in rotation, velocity, and position accumulated
    from integrating IMU measurements.

    Attributes:
        delta_R: Pre-integrated rotation increment, shape (3, 3), SO(3).
        delta_v: Pre-integrated velocity increment, shape (3,), m/s.
        delta_p: Pre-integrated position increment, shape (3,), m.
        dt: Total integration time in seconds.
        n_samples: Number of IMU samples integrated.
        cov: Covariance of the pre-integrated state, shape (9, 9).
        bias_acc: Accelerometer bias at integration time, shape (3,).
        bias_gyro: Gyroscope bias at integration time, shape (3,).
        Ja_R: Jacobian of delta_R w.r.t. acc bias, shape (3, 3).
        Ja_v: Jacobian of delta_v w.r.t. acc bias, shape (3, 3).
        Ja_p: Jacobian of delta_p w.r.t. acc bias, shape (3, 3).
        Jg_R: Jacobian of delta_R w.r.t. gyro bias, shape (3, 3).
        Jg_v: Jacobian of delta_v w.r.t. gyro bias, shape (3, 3).
        Jg_p: Jacobian of delta_p w.r.t. gyro bias, shape (3, 3).
    """

    delta_R: np.ndarray = field(default_factory=lambda: np.eye(3))   # (3, 3)
    delta_v: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,)
    delta_p: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (3,)
    dt: float = 0.0
    n_samples: int = 0
    cov: np.ndarray = field(default_factory=lambda: np.zeros((9, 9)))
    bias_acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bias_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Ja_R: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    Ja_v: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    Ja_p: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    Jg_R: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    Jg_v: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    Jg_p: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))

    def correct_for_bias_update(
        self,
        delta_ba: np.ndarray,
        delta_bg: np.ndarray,
    ) -> "PreintegratedState":
        """Apply a first-order bias correction without re-integration.

        Args:
            delta_ba: Accelerometer bias change, shape (3,).
            delta_bg: Gyroscope bias change, shape (3,).

        Returns:
            New PreintegratedState with corrected increments.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError(
            "PreintegratedState.correct_for_bias_update is not yet implemented."
        )


class IMUPreintegrator:
    """Pre-integrates IMU measurements between keyframes on SO(3).

    Args:
        noise_acc: Accelerometer noise density (m/s^2 / sqrt(Hz)).
        noise_gyro: Gyroscope noise density (rad/s / sqrt(Hz)).
        noise_bias_acc: Accelerometer bias random walk (m/s^3 / sqrt(Hz)).
        noise_bias_gyro: Gyroscope bias random walk (rad/s^2 / sqrt(Hz)).
        gravity_ned: Gravity vector in NED frame, shape (3,).
    """

    def __init__(
        self,
        noise_acc: float = 0.1,
        noise_gyro: float = 0.01,
        noise_bias_acc: float = 0.001,
        noise_bias_gyro: float = 0.0001,
        gravity_ned: np.ndarray = None,  # type: ignore[assignment]
    ) -> None:
        self.noise_acc = noise_acc
        self.noise_gyro = noise_gyro
        self.noise_bias_acc = noise_bias_acc
        self.noise_bias_gyro = noise_bias_gyro
        self.gravity_ned = (
            gravity_ned if gravity_ned is not None else np.array([0.0, 0.0, 9.81])
        )
        self._state: PreintegratedState = PreintegratedState()

    def reset(self, bias_acc: np.ndarray, bias_gyro: np.ndarray) -> None:
        """Reset the integrator for a new keyframe interval.

        Args:
            bias_acc: Current accelerometer bias estimate, shape (3,).
            bias_gyro: Current gyroscope bias estimate, shape (3,).

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("IMUPreintegrator.reset is not yet implemented.")

    def integrate(
        self,
        acc: np.ndarray,
        gyro: np.ndarray,
        dt: float,
    ) -> None:
        """Integrate a single IMU measurement.

        Uses the Euler (or mid-point) method to update the pre-integrated
        state and propagate the covariance.

        Args:
            acc: Accelerometer measurement, shape (3,), m/s^2.
            gyro: Gyroscope measurement, shape (3,), rad/s.
            dt: Time step in seconds.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("IMUPreintegrator.integrate is not yet implemented.")

    def get_state(self) -> PreintegratedState:
        """Return the current pre-integrated state.

        Returns:
            A copy of the current PreintegratedState.
        """
        return self._state

    def predict_pose(
        self,
        R0: np.ndarray,
        v0: np.ndarray,
        p0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Propagate a pose forward using the pre-integrated state.

        Args:
            R0: Initial rotation matrix, shape (3, 3).
            v0: Initial velocity in world frame, shape (3,).
            p0: Initial position in world frame, shape (3,).

        Returns:
            Tuple (R1, v1, p1) predicted rotation, velocity, and position.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("IMUPreintegrator.predict_pose is not yet implemented.")
