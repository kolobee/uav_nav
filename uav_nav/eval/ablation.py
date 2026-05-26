"""Ablation study framework.

Runs systematic ablation experiments by toggling individual pipeline
components (e.g. semantic matching, embedding head, stereo depth) and
measuring the impact on trajectory accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from uav_nav.eval.trajectory_eval import TrajectoryMetrics


@dataclass
class AblationConfig:
    """Configuration for a single ablation variant.

    Attributes:
        name: Variant name (e.g. "no_semantic_matching").
        description: Human-readable description of what is disabled.
        modifications: Dict mapping pipeline parameter names to override values.
            Used to disable/modify specific components.
    """

    name: str
    description: str = ""
    modifications: dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationResult:
    """Result from a single ablation variant.

    Attributes:
        config: The ablation configuration that was run.
        metrics: TrajectoryMetrics for the variant.
        mean_tilm_quality: Mean TILM matching quality during evaluation.
        mean_n_landmarks: Mean number of landmarks per frame.
        compute_time_mean_ms: Mean per-frame compute time.
    """

    config: AblationConfig
    metrics: TrajectoryMetrics
    mean_tilm_quality: float = 0.0
    mean_n_landmarks: float = 0.0
    compute_time_mean_ms: float = 0.0

    @property
    def ate_rmse(self) -> float:
        """Convenience accessor for ATE RMSE.

        Returns:
            ATE RMSE in metres.
        """
        return self.metrics.ate_rmse


class AblationStudy:
    """Runs and compares ablation variants of the pipeline.

    Args:
        baseline_config: AblationConfig representing the full system (baseline).
        pipeline_factory: Callable(AblationConfig) → NavigationPipeline that
            constructs a pipeline from an ablation config.
        scenario_runner: ScenarioRunner to execute each variant.
        output_dir: Directory for ablation result files.
    """

    def __init__(
        self,
        baseline_config: AblationConfig,
        pipeline_factory: Callable[[AblationConfig], object],
        scenario_runner: object,
        output_dir: Path,
    ) -> None:
        self.baseline_config = baseline_config
        self.pipeline_factory = pipeline_factory
        self.scenario_runner = scenario_runner
        self.output_dir = Path(output_dir)
        self._results: list[AblationResult] = []

    def run_variant(self, config: AblationConfig) -> AblationResult:
        """Run a single ablation variant and return its result.

        Args:
            config: AblationConfig for the variant.

        Returns:
            AblationResult with metrics.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("AblationStudy.run_variant is not yet implemented.")

    def run_all(self, variants: list[AblationConfig]) -> list[AblationResult]:
        """Run all ablation variants including the baseline.

        Args:
            variants: List of AblationConfig instances to evaluate.

        Returns:
            List of AblationResult starting with the baseline.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("AblationStudy.run_all is not yet implemented.")

    def delta_table(self) -> dict[str, dict[str, float]]:
        """Compute performance delta of each variant relative to baseline.

        Returns:
            Nested dict mapping variant_name → {metric_name → delta_value}.
            Positive delta means degradation, negative means improvement.

        Raises:
            NotImplementedError: Not yet implemented.
            RuntimeError: If run_all has not been called yet.
        """
        raise NotImplementedError("AblationStudy.delta_table is not yet implemented.")

    def save_table(self, output_path: Optional[Path] = None) -> Path:
        """Save ablation results as a formatted LaTeX or CSV table.

        Args:
            output_path: Output file path. Defaults to output_dir/ablation.csv.

        Returns:
            Path to the written file.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("AblationStudy.save_table is not yet implemented.")
