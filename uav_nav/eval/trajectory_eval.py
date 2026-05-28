"""Trajectory evaluation using the evo library.

Computes standard VIO/SLAM trajectory metrics: ATE (Absolute Trajectory
Error) and RPE (Relative Pose Error) using the ``evo`` package.
"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from evo.core import metrics as evo_metrics
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core.units import Unit

    _EVO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _EVO_AVAILABLE = False


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

    def improvement_over(self, baseline: "TrajectoryMetrics") -> float:
        """Percentage ATE RMSE improvement relative to a baseline.

        Returns:
            Positive value = improvement; negative = regression. Zero if
            baseline ATE is zero (avoids division by zero).
        """
        if baseline.ate_rmse == 0.0:
            return 0.0
        return (baseline.ate_rmse - self.ate_rmse) / baseline.ate_rmse * 100.0

    def to_dict(self) -> dict[str, float | int | str]:
        """Serialise metrics to a plain dictionary."""
        return asdict(self)  # type: ignore[return-value]


def _build_pose_trajectory(
    positions: np.ndarray,
    timestamps: np.ndarray,
) -> "PoseTrajectory3D":
    """Build a PoseTrajectory3D with identity rotations from NED positions."""
    N = len(positions)
    quats = np.zeros((N, 4), dtype=np.float64)
    quats[:, 0] = 1.0  # w=1, x=y=z=0 → identity
    return PoseTrajectory3D(
        positions_xyz=positions.astype(np.float64),
        orientations_quat_wxyz=quats,
        timestamps=timestamps.astype(np.float64),
    )


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
            timestamps: Optional timestamps, shape (N,). Synthesised if None.
            sequence_name: Optional sequence identifier.

        Returns:
            TrajectoryMetrics with ATE and RPE statistics.
        """
        positions_est = np.asarray(positions_est, dtype=np.float64)
        positions_gt = np.asarray(positions_gt, dtype=np.float64)
        N = len(positions_est)

        if N < 2:
            return TrajectoryMetrics(n_frames=N, sequence_name=sequence_name)

        if timestamps is None:
            timestamps = np.arange(N, dtype=np.float64)
        else:
            timestamps = np.asarray(timestamps, dtype=np.float64)

        if not _EVO_AVAILABLE:
            return self._evaluate_fallback(positions_est, positions_gt, timestamps, sequence_name)

        traj_ref = _build_pose_trajectory(positions_gt, timestamps)
        traj_est = _build_pose_trajectory(positions_est, timestamps)

        if self.align:
            traj_est = copy.deepcopy(traj_est)
            try:
                traj_est.align(traj_ref, correct_scale=self.align_scale)
            except Exception:
                pass  # degenerate trajectory (e.g. straight line) — skip alignment

        # ATE
        ape = evo_metrics.APE(evo_metrics.PoseRelation.translation_part)
        ape.process_data((traj_ref, traj_est))
        ape_stats = ape.get_all_statistics()

        # RPE
        delta_unit = _parse_delta_unit(self.rpe_delta_unit)
        rpe = evo_metrics.RPE(
            pose_relation=evo_metrics.PoseRelation.translation_part,
            delta=float(self.rpe_delta),
            delta_unit=delta_unit,
        )
        try:
            rpe.process_data((traj_ref, traj_est))
            rpe_stats = rpe.get_all_statistics()
            rpe_rmse = float(rpe_stats.get("rmse", 0.0))
            rpe_mean = float(rpe_stats.get("mean", 0.0))
        except Exception:
            rpe_rmse = 0.0
            rpe_mean = 0.0

        return TrajectoryMetrics(
            ate_rmse=float(ape_stats.get("rmse", 0.0)),
            ate_mean=float(ape_stats.get("mean", 0.0)),
            ate_std=float(ape_stats.get("std", 0.0)),
            ate_max=float(ape_stats.get("max", 0.0)),
            rpe_rmse=rpe_rmse,
            rpe_mean=rpe_mean,
            rpe_trans_rmse=rpe_rmse,
            rpe_rot_rmse=0.0,
            n_frames=N,
            sequence_name=sequence_name,
        )

    def _evaluate_fallback(
        self,
        positions_est: np.ndarray,
        positions_gt: np.ndarray,
        timestamps: np.ndarray,
        sequence_name: str,
    ) -> TrajectoryMetrics:
        """Simple RMSE fallback when evo is not installed."""
        errors = np.linalg.norm(positions_est - positions_gt, axis=1)
        # Frame-to-frame RPE (no alignment in fallback)
        diffs_est = np.diff(positions_est, axis=0)
        diffs_gt = np.diff(positions_gt, axis=0)
        rpe_errors = np.linalg.norm(diffs_est - diffs_gt, axis=1)
        return TrajectoryMetrics(
            ate_rmse=float(np.sqrt(np.mean(errors ** 2))),
            ate_mean=float(np.mean(errors)),
            ate_std=float(np.std(errors)),
            ate_max=float(np.max(errors)),
            rpe_rmse=float(np.sqrt(np.mean(rpe_errors ** 2))) if len(rpe_errors) else 0.0,
            rpe_mean=float(np.mean(rpe_errors)) if len(rpe_errors) else 0.0,
            rpe_trans_rmse=float(np.sqrt(np.mean(rpe_errors ** 2))) if len(rpe_errors) else 0.0,
            rpe_rot_rmse=0.0,
            n_frames=len(errors),
            sequence_name=sequence_name,
        )

    def compare(
        self,
        gt_positions: np.ndarray,
        with_tilm: np.ndarray,
        without_tilm: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> dict[str, object]:
        """Compare TILM-corrected vs dead-reckoning trajectories.

        Args:
            gt_positions: Ground-truth NED positions, shape (N, 3).
            with_tilm: TILM-corrected positions, shape (N, 3).
            without_tilm: Dead-reckoning (IMU-only) positions, shape (N, 3).
            timestamps: Optional timestamps, shape (N,).

        Returns:
            Dictionary with keys "tilm", "dead_reckoning", "improvement_pct".
        """
        metrics_tilm = self.evaluate(with_tilm, gt_positions, timestamps, "tilm")
        metrics_dr = self.evaluate(without_tilm, gt_positions, timestamps, "dead_reckoning")
        return {
            "tilm": metrics_tilm,
            "dead_reckoning": metrics_dr,
            "improvement_pct": metrics_tilm.improvement_over(metrics_dr),
        }

    def evaluate_batch(
        self,
        results: list[dict[str, object]],
    ) -> list[TrajectoryMetrics]:
        """Evaluate multiple sequences and return a list of metrics.

        Args:
            results: List of dicts, each with keys ``positions_est``,
                ``positions_gt``, ``timestamps`` (optional),
                ``sequence_name`` (optional).

        Returns:
            List of TrajectoryMetrics, one per sequence.
        """
        return [
            self.evaluate(
                positions_est=r["positions_est"],  # type: ignore[arg-type]
                positions_gt=r["positions_gt"],  # type: ignore[arg-type]
                timestamps=r.get("timestamps"),  # type: ignore[arg-type]
                sequence_name=str(r.get("sequence_name", "")),
            )
            for r in results
        ]

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
            format: ``"csv"`` or ``"json"``.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [m.to_dict() for m in metrics_list]

        if format == "json":
            output_path.write_text(json.dumps(rows, indent=2))
        else:
            if not rows:
                output_path.write_text("")
                return
            with output_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    def plot_trajectories(
        self,
        positions_est: np.ndarray,
        positions_gt: np.ndarray,
        output_path: Optional[Path] = None,
        dead_reckoning: Optional[np.ndarray] = None,
        title: str = "Trajectory comparison",
    ) -> None:
        """Plot estimated vs ground-truth trajectory (top-down XY view).

        Args:
            positions_est: Estimated NED positions, shape (N, 3).
            positions_gt: Ground-truth NED positions, shape (N, 3).
            output_path: If given, save the figure to this path; else show.
            dead_reckoning: Optional dead-reckoning positions, shape (N, 3).
            title: Figure title.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plotting.") from exc

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(positions_gt[:, 1], positions_gt[:, 0], "k-", linewidth=1.5, label="GT")
        ax.plot(positions_est[:, 1], positions_est[:, 0], "b-", linewidth=1.0, label="EKF+TILM")
        if dead_reckoning is not None:
            ax.plot(dead_reckoning[:, 1], dead_reckoning[:, 0], "r--", linewidth=1.0, label="Dead reckoning")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_title(title)
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if output_path is not None:
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()


def _parse_delta_unit(unit_str: str) -> "Unit":
    """Convert string delta unit to evo Unit enum."""
    if not _EVO_AVAILABLE:
        return None  # type: ignore[return-value]
    mapping = {
        "frames": Unit.frames,
        "meters": Unit.meters,
        "seconds": Unit.seconds,
        "m": Unit.meters,
        "s": Unit.seconds,
    }
    return mapping.get(unit_str.lower(), Unit.frames)
