"""IMU pre-integration on the SO(3) manifold.

Implements a simplified version of Forster et al. (2017)
"On-Manifold Preintegration for Real-Time Visual-Inertial Odometry"
(IEEE TRO). Pre-integrates accelerometer and gyroscope measurements
between two keyframes without re-integrating from the start.

Integration uses first-order Euler on SO(3). Covariance is propagated
using a diagonal noise model (sufficient for thesis-level evaluation).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from uav_nav.estimation.pose_utils import so3_exp, skew


@dataclass
class PreintegratedState:
    """Pre-integrated IMU state between two keyframes.

    Attributes:
        delta_R: Pre-integrated rotation increment, shape (3, 3), SO(3).
        delta_v: Pre-integrated velocity increment, shape (3,), m/s.
        delta_p: Pre-integrated position increment, shape (3,), m.
        dt: Total integration time in seconds.
        n_samples: Number of IMU samples integrated.
        cov: Covariance of [Δp, Δv, Δθ], shape (9, 9).
        bias_acc: Accelerometer bias at integration time, shape (3,).
        bias_gyro: Gyroscope bias at integration time, shape (3,).
        Ja_R: Jacobian of delta_R w.r.t. acc bias, shape (3, 3).
        Ja_v: Jacobian of delta_v w.r.t. acc bias, shape (3, 3).
        Ja_p: Jacobian of delta_p w.r.t. acc bias, shape (3, 3).
        Jg_R: Jacobian of delta_R w.r.t. gyro bias, shape (3, 3).
        Jg_v: Jacobian of delta_v w.r.t. gyro bias, shape (3, 3).
        Jg_p: Jacobian of delta_p w.r.t. gyro bias, shape (3, 3).
    """

    delta_R: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    delta_v: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    delta_p: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    dt: float = 0.0
    n_samples: int = 0
    cov: np.ndarray = field(default_factory=lambda: np.zeros((9, 9), dtype=np.float64))
    bias_acc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    bias_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    Ja_R: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))
    Ja_v: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))
    Ja_p: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))
    Jg_R: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))
    Jg_v: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))
    Jg_p: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))

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
        """
        corrected = copy.copy(self)
        corrected.delta_R = self.delta_R @ so3_exp(self.Jg_R @ delta_bg)
        corrected.delta_v = self.delta_v + self.Ja_v @ delta_ba + self.Jg_v @ delta_bg
        corrected.delta_p = self.delta_p + self.Ja_p @ delta_ba + self.Jg_p @ delta_bg
        corrected.bias_acc = self.bias_acc + delta_ba
        corrected.bias_gyro = self.bias_gyro + delta_bg
        return corrected


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
            gravity_ned if gravity_ned is not None
            else np.array([0.0, 0.0, 9.81], dtype=np.float64)
        )
        self._state: PreintegratedState = PreintegratedState()

    def reset(self, bias_acc: np.ndarray, bias_gyro: np.ndarray) -> None:
        """Reset the integrator for a new keyframe interval.

        Args:
            bias_acc: Current accelerometer bias estimate, shape (3,).
            bias_gyro: Current gyroscope bias estimate, shape (3,).
        """
        self._state = PreintegratedState(
            bias_acc=np.array(bias_acc, dtype=np.float64),
            bias_gyro=np.array(bias_gyro, dtype=np.float64),
        )

    def integrate(
        self,
        acc: np.ndarray,
        gyro: np.ndarray,
        dt: float,
    ) -> None:
        """Integrate a single IMU measurement (Euler on SO(3)).

        Removes bias, integrates delta_R/v/p, propagates covariance.

        Args:
            acc: Accelerometer measurement, shape (3,), m/s^2.
            gyro: Gyroscope measurement, shape (3,), rad/s.
            dt: Time step in seconds.
        """
        s = self._state
        a = np.asarray(acc, dtype=np.float64) - s.bias_acc
        w = np.asarray(gyro, dtype=np.float64) - s.bias_gyro

        # Rotation increment on SO(3) (Euler)
        dR = so3_exp(w * dt)

        # Pre-integrated increments (Euler, using old delta_R)
        new_delta_p = s.delta_p + s.delta_v * dt + 0.5 * (s.delta_R @ a) * dt**2
        new_delta_v = s.delta_v + (s.delta_R @ a) * dt
        new_delta_R = s.delta_R @ dR

        # First-order Jacobians for bias correction (accumulated)
        # Jg_R: ∂ΔR/∂δbg ≈ accumulated -I*dt (gyro bias → rotation error)
        s.Jg_R = s.Jg_R - np.eye(3) * dt
        # Ja_v: ∂Δv/∂δba ≈ accumulated -ΔR*dt
        s.Ja_v = s.Ja_v - s.delta_R * dt
        # Ja_p: ∂Δp/∂δba (from velocity integration)
        s.Ja_p = s.Ja_p + s.Ja_v * dt

        # Covariance propagation (diagonal noise model, indices: [Δp(0:3), Δv(3:6), Δθ(6:9)])
        # Discrete noise variance: σ² * dt (noise density → variance per step)
        s.cov[0:3, 0:3] += self.noise_acc**2 * (0.5 * dt**2)**2 * np.eye(3)
        s.cov[3:6, 3:6] += self.noise_acc**2 * dt**2 * np.eye(3)
        s.cov[6:9, 6:9] += self.noise_gyro**2 * dt**2 * np.eye(3)

        s.delta_R = new_delta_R
        s.delta_v = new_delta_v
        s.delta_p = new_delta_p
        s.dt += dt
        s.n_samples += 1

    def get_state(self) -> PreintegratedState:
        """Return the current pre-integrated state (not a copy).

        Returns:
            Current PreintegratedState.
        """
        return self._state

    def predict_pose(
        self,
        R0: np.ndarray,
        v0: np.ndarray,
        p0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Propagate a pose forward using the pre-integrated state.

        Applies standard IMU kinematic equations in NED frame:
            p1 = p0 + v0*Δt + 0.5*g*Δt² + R0 @ Δp
            v1 = v0 + g*Δt + R0 @ Δv
            R1 = R0 @ ΔR

        Args:
            R0: Initial rotation matrix (body-to-NED), shape (3, 3).
            v0: Initial velocity in NED frame, shape (3,), m/s.
            p0: Initial position in NED frame, shape (3,), m.

        Returns:
            Tuple (R1, v1, p1) — predicted rotation, velocity, and position.
        """
        s = self._state
        g = self.gravity_ned
        dt = s.dt
        R1 = R0 @ s.delta_R
        v1 = v0 + g * dt + R0 @ s.delta_v
        p1 = p0 + v0 * dt + 0.5 * g * dt**2 + R0 @ s.delta_p
        return R1, v1, p1
