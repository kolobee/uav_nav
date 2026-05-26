"""Trajectory evaluation using the evo library.

Computes standard VIO/SLAM trajectory metrics: ATE (Absolute Trajectory
Error) and RPE (Relative Pose Error) using the ``evo`` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class TrajectoryMetrics:
    """Trajectory evaluation metrics.

    Attributes:
        ate_rmse: Absolute Trajectory Error RMSE in metres.
        ate_mean: ATE mean error in metres.
        ate_std: ATE standard deviation in metres.
        ate_max: ATE maximum error in metres.
        rpe_rmse: Relative Pose Error RMSE in metres.
        rpe_mean: RPE mean error in metres.
        rpe_trans_rmse: RPE translational RMSE in metres.
        rpe_rot_rmse: RPE rotational RMSE in degrees.
        n_frames: Number of evaluated frames.
        sequence_name: Name of the evaluated sequence.
    """

    ate_rmse: float = 0.0
    ate_mean: float = 0.0
    ate_std: float = 0.0
    ate_max: float = 0.0
    rpe_rmse: float = 0.0
    rpe_mean: float = 0.0
    rpe_trans_rmse: float = 0.0
    rpe_rot_rmse: float = 0.0
    n_frames: int = 0
    sequence_name: str = ""

    def to_dict(self) -> dict[str, float | int | str]:
        """Serialise metrics to a plain dictionary.

        Returns:
            Dictionary of all metric fields.
        """
        return {
            "ate_rmse": self.ate_rmse,
            "ate_mean": self.ate_mean,
            "ate_std": self.ate_std,
            "ate_max": self.ate_max,
            "rpe_rmse": self.rpe_rmse,
            "rpe_mean": self.rpe_mean,
            "rpe_trans_rmse": self.rpe_trans_rmse,
            "rpe_rot_rmse": self.rpe_rot_rmse,
            "n_frames": self.n_frames,
            "sequence_name": self.sequence_name,
        }


class TrajectoryEvaluator:
    """Computes ATE and RPE between estimated and ground-truth trajectories.

    Wraps the ``evo`` library with a convenient API for batch evaluation
    of multiple sequences and export to CSV/plots.

    Args:
        align: Whether to SE(3)-align the estimated trajectory before computing
            ATE (recommended True for VIO where initial frame is arbitrary).
        align_scale: Whether to additionally solve for scale alignment.
        rpe_delta: Frame delta for RPE computation (relative pose pairs).
        rpe_delta_unit: Unit for RPE delta ("frames", "meters", "seconds").
    """

    def __init__(
        self,
        align: bool = True,
        align_scale: bool = False,
        rpe_delta: int = 1,
        rpe_delta_unit: str = "frames",
    ) -> None:
        self.align = align
        self.align_scale = align_scale
        self.rpe_delta = rpe_delta
        self.rpe_delta_unit = rpe_delta_unit

    def evaluate(
        self,
        positions_est: np.ndarray,
        positions_gt: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        sequence_name: str = "",
    ) -> TrajectoryMetrics:
        """Compute trajectory metrics for a single sequence.

        Args:
            positions_est: Estimated NED positions, shape (N, 3).
            positions_gt: Ground-truth NED positions, shape (N, 3).
            timestamps: Optional timestamps, shape (N,). Used for RPE with
                time-based delta.
            sequence_name: Optional sequence identifier.

        Returns:
            TrajectoryMetrics with ATE and RPE statistics.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TrajectoryEvaluator.evaluate is not yet implemented.")

    def evaluate_batch(
        self,
        results: list[dict[str, np.ndarray]],
    ) -> list[TrajectoryMetrics]:
        """Evaluate multiple sequences and return a list of metrics.

        Args:
            results: List of dicts, each with keys "positions_est",
                "positions_gt", "timestamps" (optional), "sequence_name" (opt).

        Returns:
            List of TrajectoryMetrics, one per sequence.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TrajectoryEvaluator.evaluate_batch is not yet implemented.")

    def save_report(
        self,
        metrics_list: list[TrajectoryMetrics],
        output_path: Path,
        format: str = "csv",
    ) -> None:
        """Save evaluation results to a file.

        Args:
            metrics_list: List of TrajectoryMetrics to save.
            output_path: Output file path.
            format: Output format ("csv" or "json").

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TrajectoryEvaluator.save_report is not yet implemented.")

    def plot_trajectories(
        self,
        positions_est: np.ndarray,
        positions_gt: np.ndarray,
        output_path: Optional[Path] = None,
    ) -> None:
        """Plot estimated vs ground-truth trajectory.

        Args:
            positions_est: Estimated positions, shape (N, 3).
            positions_gt: Ground-truth positions, shape (N, 3).
            output_path: If given, save the figure to this path.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TrajectoryEvaluator.plot_trajectories is not yet implemented.")
