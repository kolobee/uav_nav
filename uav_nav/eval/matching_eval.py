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
        """
        query_positions_gt = np.asarray(query_positions_gt, dtype=np.float64)
        match_positions = np.asarray(match_positions, dtype=np.float64)
        match_valid = np.asarray(match_valid, dtype=bool)

        N = len(query_positions_gt)
        n_matched = int(match_valid.sum())

        if n_matched == 0:
            return MatchingMetrics(
                n_total_queries=N,
                n_successful_matches=0,
                n_correct_matches=0,
            )

        valid_gt = query_positions_gt[match_valid]
        valid_pred = match_positions[match_valid]
        errors = np.linalg.norm(valid_pred - valid_gt, axis=1)

        is_correct = errors <= self.position_tolerance
        n_correct = int(is_correct.sum())

        precision = n_correct / n_matched
        recall = n_correct / N
        f1 = (2.0 * precision * recall / (precision + recall)
              if (precision + recall) > 0.0 else 0.0)
        fpr = (n_matched - n_correct) / N if N > 0 else 0.0

        mean_lat = 0.0
        if latencies_ms is not None and len(latencies_ms) > 0:
            mean_lat = float(np.asarray(latencies_ms, dtype=np.float64).mean())

        return MatchingMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            position_error_mean=float(errors.mean()),
            position_error_rmse=float(np.sqrt((errors ** 2).mean())),
            position_error_max=float(errors.max()),
            false_positive_rate=fpr,
            n_total_queries=N,
            n_successful_matches=n_matched,
            n_correct_matches=n_correct,
            mean_latency_ms=mean_lat,
        )

    def precision_recall_curve(
        self,
        scores: np.ndarray,
        is_correct: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute a precision-recall curve over varying score thresholds.

        Sorts queries by score descending and computes cumulative precision
        and recall at each threshold.

        Args:
            scores: Match confidence scores, shape (N,).
            is_correct: Boolean ground-truth correctness, shape (N,).

        Returns:
            Tuple (precision, recall, thresholds) where each array has
            shape (N,). Threshold array is sorted descending.
        """
        scores = np.asarray(scores, dtype=np.float64)
        is_correct = np.asarray(is_correct, dtype=bool)

        if len(scores) == 0:
            empty = np.zeros(0, dtype=np.float64)
            return empty, empty, empty

        order = np.argsort(-scores)
        sorted_correct = is_correct[order].astype(np.float64)
        sorted_scores = scores[order]

        cumulative_correct = np.cumsum(sorted_correct)
        cumulative_total = np.arange(1, len(scores) + 1, dtype=np.float64)
        n_positive = max(int(is_correct.sum()), 1)

        precision = cumulative_correct / cumulative_total
        recall = cumulative_correct / n_positive
        return precision, recall, sorted_scores
