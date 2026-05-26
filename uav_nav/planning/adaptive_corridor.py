"""Adaptive corridor management for GNSS-denied path following.

Maintains a virtual corridor around the planned path and uses TILM
matching quality to dynamically adjust the corridor width and the
path-follower's look-ahead distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CorridorState:
    """Current state of the adaptive corridor.

    Attributes:
        width: Current corridor half-width in metres.
        quality: Composite quality score in [0, 1].
        deviation: Lateral deviation from path centreline in metres.
        n_matched_landmarks: Number of landmarks matched to TILM this step.
        is_violated: Whether the UAV has left the corridor.
        violation_duration: Continuous time (s) outside the corridor.
    """

    width: float = 20.0
    quality: float = 1.0
    deviation: float = 0.0
    n_matched_landmarks: int = 0
    is_violated: bool = False
    violation_duration: float = 0.0


class AdaptiveCorridor:
    """Manages an adaptive path corridor based on localisation confidence.

    The corridor width contracts when TILM matching is reliable, giving a
    tighter constraint on the path follower. When matching degrades (GNSS-
    denied, few landmarks), the corridor widens to avoid false violations.

    Args:
        initial_width: Starting corridor half-width in metres.
        min_width: Minimum corridor half-width in metres.
        max_width: Maximum corridor half-width in metres.
        contraction_rate: Rate of width contraction per second (m/s).
        expansion_rate: Rate of width expansion per second (m/s).
        quality_threshold: TILM quality below which corridor expands.
        violation_timeout: Max continuous violation time before emergency stop.
    """

    def __init__(
        self,
        initial_width: float = 20.0,
        min_width: float = 5.0,
        max_width: float = 50.0,
        contraction_rate: float = 0.5,
        expansion_rate: float = 2.0,
        quality_threshold: float = 0.7,
        violation_timeout: float = 10.0,
    ) -> None:
        self.initial_width = initial_width
        self.min_width = min_width
        self.max_width = max_width
        self.contraction_rate = contraction_rate
        self.expansion_rate = expansion_rate
        self.quality_threshold = quality_threshold
        self.violation_timeout = violation_timeout
        self._state = CorridorState(width=initial_width)

    def update(
        self,
        position_ned: np.ndarray,
        path_waypoints: np.ndarray,
        tilm_quality: float,
        n_matched_landmarks: int,
        dt: float,
    ) -> CorridorState:
        """Update corridor state for the current timestep.

        Args:
            position_ned: Current UAV NED position, shape (3,).
            path_waypoints: Reference path waypoints, shape (N, 3).
            tilm_quality: TILM matching quality score in [0, 1].
            n_matched_landmarks: Number of landmarks matched this step.
            dt: Time step in seconds.

        Returns:
            Updated CorridorState.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("AdaptiveCorridor.update is not yet implemented.")

    def is_inside(self, position_ned: np.ndarray, path_waypoints: np.ndarray) -> bool:
        """Check whether the UAV is within the current corridor.

        Args:
            position_ned: Current NED position, shape (3,).
            path_waypoints: Reference path waypoints, shape (N, 3).

        Returns:
            True if the UAV is inside the corridor, False otherwise.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("AdaptiveCorridor.is_inside is not yet implemented.")

    def _nearest_path_distance(
        self,
        position_ned: np.ndarray,
        path_waypoints: np.ndarray,
    ) -> float:
        """Compute the minimum perpendicular distance to any path segment.

        Args:
            position_ned: Query NED position, shape (3,).
            path_waypoints: Path waypoints, shape (N, 3).

        Returns:
            Minimum lateral distance to the path centreline in metres.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError(
            "AdaptiveCorridor._nearest_path_distance is not yet implemented."
        )

    @property
    def state(self) -> CorridorState:
        """Current corridor state (read-only view)."""
        return self._state
