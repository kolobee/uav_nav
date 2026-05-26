"""Lightweight pseudo-simulator for offline pipeline testing.

Replays a MidAir dataset sequence through the NavigationPipeline,
optionally injecting synthetic GNSS outages or noise, and records
all outputs for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from uav_nav.data.midair_loader import MidAirLoader, MidAirSample
from uav_nav.runtime.pipeline import NavigationPipeline


@dataclass
class GNSSOutageConfig:
    """Configuration for a synthetic GNSS outage injection.

    Attributes:
        start_time: Outage start time in seconds.
        duration: Outage duration in seconds.
        noise_std: Gaussian noise std applied before outage (m). Zero = perfect.
    """

    start_time: float
    duration: float
    noise_std: float = 0.0


@dataclass
class SimulationResult:
    """Results from a full pseudo-simulation run.

    Attributes:
        timestamps: Frame timestamps, shape (N,).
        positions_est: Estimated NED positions, shape (N, 3).
        positions_gt: Ground-truth NED positions, shape (N, 3).
        modes: Mission mode at each frame as string list, length N.
        tilm_qualities: TILM matching quality at each frame, shape (N,).
        gnss_active_mask: Boolean array indicating GNSS availability, shape (N,).
        position_errors: L2 position error at each frame (m), shape (N,).
    """

    timestamps: np.ndarray = field(default_factory=lambda: np.empty(0))
    positions_est: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    positions_gt: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    modes: list[str] = field(default_factory=list)
    tilm_qualities: np.ndarray = field(default_factory=lambda: np.empty(0))
    gnss_active_mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    position_errors: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def rmse(self) -> float:
        """Root-mean-square position error over all frames.

        Returns:
            RMSE in metres.
        """
        if len(self.position_errors) == 0:
            return float("nan")
        return float(np.sqrt(np.mean(self.position_errors ** 2)))

    @property
    def max_error(self) -> float:
        """Maximum position error across all frames.

        Returns:
            Maximum error in metres.
        """
        if len(self.position_errors) == 0:
            return float("nan")
        return float(np.max(self.position_errors))


class PseudoSimulator:
    """Replays a MidAir dataset through the navigation pipeline.

    Args:
        pipeline: Initialised NavigationPipeline instance.
        loader: MidAirLoader with index already built.
        outages: List of GNSS outage configurations to inject.
        progress_callback: Optional function called with (frame_idx, total).
    """

    def __init__(
        self,
        pipeline: NavigationPipeline,
        loader: MidAirLoader,
        outages: Optional[list[GNSSOutageConfig]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self.pipeline = pipeline
        self.loader = loader
        self.outages = outages or []
        self.progress_callback = progress_callback

    def run(self) -> SimulationResult:
        """Run the full simulation and return results.

        Returns:
            SimulationResult with trajectories, errors, and mode history.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("PseudoSimulator.run is not yet implemented.")

    def _is_gnss_active(self, timestamp: float) -> bool:
        """Determine GNSS availability at the given timestamp.

        Args:
            timestamp: Current simulation timestamp in seconds.

        Returns:
            True if GNSS is active (no active outage).
        """
        for outage in self.outages:
            if outage.start_time <= timestamp < outage.start_time + outage.duration:
                return False
        return True

    def _process_sample(
        self,
        sample: MidAirSample,
        prev_sample: Optional[MidAirSample],
    ) -> dict[str, object]:
        """Process a single MidAir sample through the pipeline.

        Args:
            sample: Current MidAirSample to process.
            prev_sample: Previous sample (for dt computation). None for first frame.

        Returns:
            Pipeline output dictionary.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("PseudoSimulator._process_sample is not yet implemented.")
