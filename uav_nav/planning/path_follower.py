"""Path follower with adaptive velocity and corridor management.

Implements a carrot-chasing path follower that tracks a pre-defined
waypoint path using the TILM-corrected pose estimate.

Key parameters (from dissertation):
    D_min     = 5.0  m  — minimum look-ahead distance.
    D_max_cap = 50.0 m  — maximum allowed look-ahead cap.
    dt_window = 5.0  s  — velocity smoothing window.
    THRESHOLD = 0.7      — corridor quality threshold for adaptive lookahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PathFollowerConfig:
    """Configuration for the path follower.

    Attributes:
        D_min: Minimum look-ahead distance in metres.
        D_max_cap: Maximum look-ahead distance cap in metres.
        dt_window: Velocity estimation smoothing window in seconds.
        THRESHOLD: Corridor quality threshold in [0, 1].
        max_speed: Maximum path-following speed in m/s.
        min_speed: Minimum path-following speed in m/s.
        heading_gain: Proportional gain for heading correction.
        altitude_gain: Proportional gain for altitude correction.
        cross_track_gain: Proportional gain for cross-track error.
        waypoint_radius: Radius within which a waypoint is considered reached.
    """

    D_min: float = 5.0
    D_max_cap: float = 50.0
    dt_window: float = 5.0
    THRESHOLD: float = 0.7
    max_speed: float = 15.0
    min_speed: float = 2.0
    heading_gain: float = 1.5
    altitude_gain: float = 1.0
    cross_track_gain: float = 0.8
    waypoint_radius: float = 3.0


@dataclass
class FollowerState:
    """Internal state of the path follower.

    Attributes:
        current_segment_idx: Index of the current path segment.
        look_ahead_distance: Current adaptive look-ahead distance.
        cross_track_error: Signed cross-track error in metres.
        along_track_progress: Fraction [0, 1] through the current segment.
        commanded_velocity: Current velocity command in m/s.
        corridor_quality: Current corridor matching quality in [0, 1].
        elapsed_since_gnss_loss: Time since last GNSS update, seconds.
    """

    current_segment_idx: int = 0
    look_ahead_distance: float = 10.0
    cross_track_error: float = 0.0
    along_track_progress: float = 0.0
    commanded_velocity: float = 5.0
    corridor_quality: float = 1.0
    elapsed_since_gnss_loss: float = 0.0


class PathFollower:
    """Adaptive look-ahead path follower for UAV navigation.

    Tracks a waypoint path, adjusting look-ahead distance and speed based
    on VIO quality and corridor matching confidence from the TILM.

    Args:
        config: PathFollowerConfig with gains and thresholds.
        waypoints: Path waypoints in NED frame, shape (N, 3).
    """

    def __init__(
        self,
        config: PathFollowerConfig,
        waypoints: Optional[np.ndarray] = None,
    ) -> None:
        self.config = config
        self.waypoints: np.ndarray = (
            waypoints if waypoints is not None else np.empty((0, 3))
        )
        self._state: FollowerState = FollowerState()
        self._velocity_history: list[tuple[float, float]] = []  # (time, speed)

    def set_path(self, waypoints: np.ndarray) -> None:
        """Set a new waypoint path and reset follower state.

        Args:
            waypoints: Path waypoints in NED frame, shape (N, 3).
        """
        self.waypoints = waypoints.copy()
        self._state = FollowerState()
        self._velocity_history.clear()

    def step(
        self,
        position_ned: np.ndarray,
        heading_rad: float,
        corridor_quality: float,
        dt: float,
    ) -> np.ndarray:
        """Compute a velocity command for the current timestep.

        Args:
            position_ned: Current NED position, shape (3,).
            heading_rad: Current heading in radians (NED convention).
            corridor_quality: Corridor matching quality from TILM, in [0, 1].
            dt: Time step in seconds.

        Returns:
            Velocity command in NED frame, shape (3,), m/s.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("PathFollower.step is not yet implemented.")

    def _compute_look_ahead(self, corridor_quality: float) -> float:
        """Compute adaptive look-ahead distance from corridor quality.

        Scales look-ahead between D_min and D_max_cap depending on quality.

        Args:
            corridor_quality: Matching quality in [0, 1].

        Returns:
            Look-ahead distance in metres.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("PathFollower._compute_look_ahead is not yet implemented.")

    def _carrot_point(
        self,
        position_ned: np.ndarray,
        look_ahead: float,
    ) -> np.ndarray:
        """Find the carrot (look-ahead target) point on the path.

        Args:
            position_ned: Current NED position, shape (3,).
            look_ahead: Look-ahead distance in metres.

        Returns:
            Carrot point in NED frame, shape (3,).

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("PathFollower._carrot_point is not yet implemented.")

    def cross_track_error(self, position_ned: np.ndarray) -> float:
        """Compute signed cross-track error from the nearest segment.

        Args:
            position_ned: Current NED position, shape (3,).

        Returns:
            Signed cross-track error in metres (positive = right of path).

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("PathFollower.cross_track_error is not yet implemented.")

    def is_path_complete(self) -> bool:
        """Return True if the final waypoint has been reached.

        Returns:
            Boolean completion flag.
        """
        if len(self.waypoints) == 0:
            return True
        return self._state.current_segment_idx >= len(self.waypoints) - 1

    @property
    def state(self) -> FollowerState:
        """Current follower state (read-only view)."""
        return self._state
