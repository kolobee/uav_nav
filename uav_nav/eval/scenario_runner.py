"""Scenario runner for structured experiments.

Executes defined experiment scenarios (cross-weather, GNSS loss, etc.)
through the pseudo-simulator and aggregates results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from uav_nav.runtime.pseudo_simulator import GNSSOutageConfig, SimulationResult


@dataclass
class ScenarioConfig:
    """Configuration for a single experiment scenario.

    Attributes:
        name: Scenario identifier (e.g. "5_3_cross_weather").
        description: Human-readable description.
        train_weather: Weather condition IDs used to build the TILM.
        test_weather: Weather condition IDs used for evaluation sequences.
        gnss_outages: GNSS outage configurations to inject.
        sequence_ids: MidAir sequence identifiers for evaluation.
        tilm_path: Optional path to a pre-built TILM for this scenario.
        extra_params: Additional scenario-specific parameters.
    """

    name: str
    description: str = ""
    train_weather: list[str] = field(default_factory=list)
    test_weather: list[str] = field(default_factory=list)
    gnss_outages: list[GNSSOutageConfig] = field(default_factory=list)
    sequence_ids: list[str] = field(default_factory=list)
    tilm_path: Optional[Path] = None
    extra_params: dict = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Results from executing a single scenario.

    Attributes:
        config: The scenario configuration that was run.
        simulation_results: SimulationResult from the pseudo-simulator.
        ate_rmse: Absolute Trajectory Error RMSE in metres.
        max_position_error: Maximum position error during GNSS loss (m).
        recovery_time_s: Time to recover to within 10 m after GNSS restoration.
        tilm_match_rate: Fraction of frames with valid TILM match.
    """

    config: ScenarioConfig
    simulation_results: Optional[SimulationResult] = None
    ate_rmse: float = float("nan")
    max_position_error: float = float("nan")
    recovery_time_s: float = float("nan")
    tilm_match_rate: float = float("nan")


class ScenarioRunner:
    """Executes and records results for a set of experiment scenarios.

    Args:
        pipeline_factory: Callable that builds a fresh NavigationPipeline.
            Called once per scenario to ensure clean state.
        output_dir: Directory for scenario result files.
        midair_root: Root directory of the MidAir dataset.
    """

    def __init__(
        self,
        pipeline_factory: object,
        output_dir: Path,
        midair_root: Path,
    ) -> None:
        self.pipeline_factory = pipeline_factory
        self.output_dir = Path(output_dir)
        self.midair_root = Path(midair_root)
        self._results: list[ScenarioResult] = []

    def run_scenario(self, config: ScenarioConfig) -> ScenarioResult:
        """Execute a single scenario and return its result.

        Args:
            config: ScenarioConfig defining the experiment.

        Returns:
            ScenarioResult with metrics for the scenario.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("ScenarioRunner.run_scenario is not yet implemented.")

    def run_all(self, configs: list[ScenarioConfig]) -> list[ScenarioResult]:
        """Execute a list of scenarios sequentially.

        Args:
            configs: List of ScenarioConfig instances.

        Returns:
            List of ScenarioResult, one per config.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("ScenarioRunner.run_all is not yet implemented.")

    def save_summary(self, output_path: Optional[Path] = None) -> Path:
        """Save a summary CSV of all scenario results.

        Args:
            output_path: Optional override for the output path.

        Returns:
            Path to the written CSV file.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("ScenarioRunner.save_summary is not yet implemented.")
