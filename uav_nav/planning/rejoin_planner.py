"""Re-join planner for returning to the planned path after deviation.

When the UAV has deviated from the planned path (e.g. due to obstacle
avoidance or GNSS loss), this planner computes a smooth re-join trajectory
that merges back onto the original path at a suitable merge point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RejoinResult:
    """Result from the re-join planner.

    Attributes:
        rejoin_waypoints: Waypoints of the re-join trajectory in NED,
            shape (M, 3). Empty if re-join is not possible.
        merge_idx: Index into the original path where re-join merges.
        estimated_distance: Estimated re-join path length in metres.
        estimated_time: Estimated re-join time in seconds.
        is_feasible: Whether a feasible re-join trajectory was found.
        reason: Human-readable explanation if not feasible.
    """

    rejoin_waypoints: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    merge_idx: int = 0
    estimated_distance: float = 0.0
    estimated_time: float = 0.0
    is_feasible: bool = False
    reason: str = ""


class RejoinPlanner:
    """Computes a re-join trajectory from current position to planned path.

    Uses a heuristic merge point selection followed by cubic Bezier or
    minimum-snap polynomial trajectory generation.

    Args:
        max_rejoin_distance: Maximum allowed re-join path length in metres.
        merge_lookahead: How far ahead on the original path (m) to merge.
        max_speed: Maximum speed during re-join in m/s.
        safety_altitude_margin: Extra altitude to maintain above terrain (m).
    """

    def __init__(
        self,
        max_rejoin_distance: float = 500.0,
        merge_lookahead: float = 50.0,
        max_speed: float = 10.0,
        safety_altitude_margin: float = 5.0,
    ) -> None:
        self.max_rejoin_distance = max_rejoin_distance
        self.merge_lookahead = merge_lookahead
        self.max_speed = max_speed
        self.safety_altitude_margin = safety_altitude_margin

    def plan(
        self,
        current_position_ned: np.ndarray,
        current_velocity_ned: np.ndarray,
        original_path: np.ndarray,
        nearest_segment_idx: int,
    ) -> RejoinResult:
        """Compute a re-join trajectory from current position to planned path.

        Args:
            current_position_ned: Current UAV NED position, shape (3,).
            current_velocity_ned: Current UAV NED velocity, shape (3,), m/s.
            original_path: Original planned path waypoints, shape (N, 3).
            nearest_segment_idx: Index of the nearest original path segment.

        Returns:
            RejoinResult with the computed re-join trajectory.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("RejoinPlanner.plan is not yet implemented.")

    def _select_merge_point(
        self,
        current_position_ned: np.ndarray,
        original_path: np.ndarray,
        nearest_segment_idx: int,
    ) -> tuple[np.ndarray, int]:
        """Select the best merge point on the original path.

        Args:
            current_position_ned: Current NED position, shape (3,).
            original_path: Original path waypoints, shape (N, 3).
            nearest_segment_idx: Index of the nearest path segment.

        Returns:
            Tuple (merge_point_ned, merge_segment_idx).

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("RejoinPlanner._select_merge_point is not yet implemented.")

    def _generate_bezier(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        p1: np.ndarray,
        n_points: int = 20,
    ) -> np.ndarray:
        """Generate a cubic Bezier curve from p0 to p1 with initial velocity v0.

        Args:
            p0: Start position, shape (3,).
            v0: Initial velocity direction (used for first control point).
            p1: End position, shape (3,).
            n_points: Number of sample points along the curve.

        Returns:
            Sampled waypoints along the Bezier curve, shape (n_points, 3).

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("RejoinPlanner._generate_bezier is not yet implemented.")
