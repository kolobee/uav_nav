"""Unit tests for PathFollowerConfig and PathFollower initialisation.

Tests the configuration defaults and the properties of the PathFollower
that do not require the full implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_nav.planning.path_follower import PathFollower, PathFollowerConfig, FollowerState


class TestPathFollowerConfig:
    """Tests for PathFollowerConfig default values."""

    def test_d_min_default(self) -> None:
        """Default D_min is 5.0 m."""
        cfg = PathFollowerConfig()
        assert cfg.D_min == 5.0

    def test_d_max_cap_default(self) -> None:
        """Default D_max_cap is 50.0 m."""
        cfg = PathFollowerConfig()
        assert cfg.D_max_cap == 50.0

    def test_dt_window_default(self) -> None:
        """Default dt_window is 5.0 s."""
        cfg = PathFollowerConfig()
        assert cfg.dt_window == 5.0

    def test_threshold_default(self) -> None:
        """Default THRESHOLD is 0.7."""
        cfg = PathFollowerConfig()
        assert cfg.THRESHOLD == pytest.approx(0.7)

    def test_max_speed_above_min(self) -> None:
        """max_speed must be greater than min_speed."""
        cfg = PathFollowerConfig()
        assert cfg.max_speed > cfg.min_speed


class TestPathFollowerInit:
    """Tests for PathFollower initialisation."""

    def test_empty_path_completion(self) -> None:
        """is_path_complete() returns True when no waypoints are set."""
        cfg = PathFollowerConfig()
        follower = PathFollower(config=cfg)
        assert follower.is_path_complete() is True

    def test_initial_state_defaults(self) -> None:
        """Initial follower state has segment_idx=0."""
        cfg = PathFollowerConfig()
        follower = PathFollower(config=cfg)
        assert follower.state.current_segment_idx == 0

    def test_set_path_resets_state(self, ned_waypoints: np.ndarray) -> None:
        """set_path() resets the internal FollowerState."""
        cfg = PathFollowerConfig()
        follower = PathFollower(config=cfg)
        follower.set_path(ned_waypoints)
        assert follower.state.current_segment_idx == 0
        assert len(follower.waypoints) == len(ned_waypoints)

    def test_path_not_complete_with_waypoints(self, ned_waypoints: np.ndarray) -> None:
        """is_path_complete() returns False when waypoints are loaded."""
        cfg = PathFollowerConfig()
        follower = PathFollower(config=cfg, waypoints=ned_waypoints)
        assert follower.is_path_complete() is False

    def test_config_stored(self) -> None:
        """PathFollower stores its configuration."""
        cfg = PathFollowerConfig(D_min=8.0, D_max_cap=40.0)
        follower = PathFollower(config=cfg)
        assert follower.config.D_min == 8.0
        assert follower.config.D_max_cap == 40.0


class TestMissionModeEnum:
    """Tests for MissionMode enum completeness."""

    def test_required_modes_exist(self) -> None:
        """All required mission modes are defined."""
        from uav_nav.planning.mission_modes import MissionMode
        required = {"NOMINAL", "GNSS_DEGRADED", "GNSS_LOST", "REJOIN", "EMERGENCY_LAND"}
        actual = {m.name for m in MissionMode}
        assert required.issubset(actual)
