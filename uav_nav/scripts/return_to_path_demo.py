"""Script 06: Demonstrate re-join manoeuvre after path deviation.

Usage::

    python uav_nav/scripts/06_return_to_path_demo.py
    python uav_nav/scripts/06_return_to_path_demo.py deviation_distance=30.0

Simulates a UAV deviating from its planned path (e.g. due to obstacle avoidance)
and demonstrates the RejoinPlanner generating a smooth return trajectory.
Outputs visualisation via Rerun SDK or saved trajectory plots.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from uav_nav.planning.rejoin_planner import RejoinPlanner
from uav_nav.planning.path_follower import PathFollower, PathFollowerConfig
from uav_nav.planning.mission_modes import MissionManager, MissionMode, MissionStatus
from uav_nav.runtime.logger import setup_logger, get_logger

logger = get_logger(__name__)


def _build_example_path() -> np.ndarray:
    """Build a simple 1 km straight NED path for demonstration.

    Returns:
        Waypoints array, shape (10, 3), NED metres.
    """
    waypoints = np.zeros((10, 3))
    waypoints[:, 0] = np.linspace(0, 1000, 10)  # North
    waypoints[:, 2] = -50.0                       # 50 m altitude AGL
    return waypoints


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Demonstrate path re-join after planned-path deviation.

    Args:
        cfg: Hydra configuration object.

    Raises:
        NotImplementedError: Full demo implementation pending.
    """
    setup_logger(log_dir=Path(cfg.runtime.log_dir), level=cfg.runtime.log_level)
    logger.info("Return-to-path demonstration starting")

    waypoints = _build_example_path()

    pf_cfg = PathFollowerConfig(
        D_min=cfg.path_follower.D_min,
        D_max_cap=cfg.path_follower.D_max_cap,
        dt_window=cfg.path_follower.dt_window,
        THRESHOLD=cfg.path_follower.THRESHOLD,
        max_speed=cfg.path_follower.max_speed,
    )
    follower = PathFollower(config=pf_cfg, waypoints=waypoints)

    rejoin = RejoinPlanner(
        max_rejoin_distance=500.0,
        merge_lookahead=50.0,
        max_speed=pf_cfg.max_speed,
    )

    # Simulate deviation from path at waypoint 3
    deviated_position = waypoints[3].copy()
    deviated_position[1] += 40.0  # 40 m east lateral deviation
    deviated_velocity = np.array([5.0, 0.5, 0.0])  # m/s

    logger.info("Simulated position: {}", deviated_position)
    logger.info("Computing re-join trajectory...")

    result = rejoin.plan(
        current_position_ned=deviated_position,
        current_velocity_ned=deviated_velocity,
        original_path=waypoints,
        nearest_segment_idx=3,
    )

    if result.is_feasible:
        logger.info(
            "Re-join trajectory found: {} waypoints, {:.1f} m, {:.1f} s",
            len(result.rejoin_waypoints),
            result.estimated_distance,
            result.estimated_time,
        )
    else:
        logger.warning("Re-join not feasible: {}", result.reason)

    # TODO: Visualise via rerun or matplotlib
    raise NotImplementedError(
        "06_return_to_path_demo.py: Full simulation loop not yet implemented."
    )


if __name__ == "__main__":
    main()
