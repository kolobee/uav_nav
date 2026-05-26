"""Evaluation of the TILM temporal matching system.

Measures place recognition accuracy (precision/recall) and position
correction quality when matching current observations against the TILM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class MatchingMetrics:
    """Metrics for TILM temporal matching evaluation.

    Attributes:
        precision: Fraction of match attempts that are correct.
        recall: Fraction of query locations that are successfully matched.
        f1: Harmonic mean of precision and recall.
        position_error_mean: Mean position correction error in metres.
        position_error_rmse: RMSE of position correction error.
        position_error_max: Maximum position correction error.
        false_positive_rate: Rate of incorrect matches.
        n_total_queries: Total number of matching queries.
        n_successful_matches: Number of queries with valid match result.
        n_correct_matches: Number of correct match results.
        mean_latency_ms: Mean TILM matching latency in milliseconds.
    """

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    position_error_mean: float = 0.0
    position_error_rmse: float = 0.0
    position_error_max: float = 0.0
    false_positive_rate: float = 0.0
    n_total_queries: int = 0
    n_successful_matches: int = 0
    n_correct_matches: int = 0
    mean_latency_ms: float = 0.0


class MatchingEvaluator:
    """Evaluates TILM matching performance against ground-truth node labels.

    Args:
        position_tolerance: Maximum position error (m) to count a match as
            correct.
        node_id_tolerance: Optional: if provided, correct match is defined by
            matching node ID rather than position.
    """

    def __init__(
        self,
        position_tolerance: float = 10.0,
        node_id_tolerance: Optional[int] = None,
    ) -> None:
        self.position_tolerance = position_tolerance
        self.node_id_tolerance = node_id_tolerance

    def evaluate(
        self,
        query_positions_gt: np.ndarray,
        match_positions: np.ndarray,
        match_valid: np.ndarray,
        latencies_ms: Optional[np.ndarray] = None,
    ) -> MatchingMetrics:
        """Compute matching metrics over a sequence of queries.

        Args:
            query_positions_gt: Ground-truth NED positions, shape (N, 3).
            match_positions: Matched/corrected NED positions, shape (N, 3).
                Rows corresponding to invalid matches may be NaN.
            match_valid: Boolean array indicating valid match, shape (N,).
            latencies_ms: Optional per-query matching latency, shape (N,).

        Returns:
            MatchingMetrics with precision, recall, and position error stats.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("MatchingEvaluator.evaluate is not yet implemented.")

    def precision_recall_curve(
        self,
        scores: np.ndarray,
        is_correct: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute a precision-recall curve over varying score thresholds.

        Args:
            scores: Match confidence scores, shape (N,).
            is_correct: Boolean ground-truth correctness, shape (N,).

        Returns:
            Tuple (precision, recall, thresholds) arrays.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError(
            "MatchingEvaluator.precision_recall_curve is not yet implemented."
        )
