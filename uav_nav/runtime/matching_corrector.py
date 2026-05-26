"""TILM-to-EKF correction bridge for GNSS-denied navigation.

MatchingCorrector drives the temporal landmark matcher and applies the
resulting position corrections to the EKF when GNSS is unavailable.

State machine:
    GNSS_ACTIVE  → EKF receives GNSS updates; TILM matching is skipped.
    GNSS_LOST    → EKF receives TILM corrections when match confidence
                   exceeds min_confidence and correction_interval elapsed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from uav_nav.estimation.ekf_vio import EKFVIO
from uav_nav.memory.temporal_matcher import MatchResult, TemporalMatcher
from uav_nav.perception.landmark_extractor import Landmark


# ── statistics dataclass ──────────────────────────────────────────────────────

@dataclass
class CorrectorStats:
    """Running statistics for the MatchingCorrector.

    Attributes:
        n_frames: Total frames processed.
        n_gnss_updates: Number of GNSS EKF updates applied.
        n_tilm_attempts: Number of TILM match attempts.
        n_tilm_corrections: Number of TILM corrections applied to EKF.
        n_tilm_rejected: Attempts rejected (low confidence or throttled).
        correction_errors: List of ||correction|| values (metres) applied.
    """

    n_frames: int = 0
    n_gnss_updates: int = 0
    n_tilm_attempts: int = 0
    n_tilm_corrections: int = 0
    n_tilm_rejected: int = 0
    correction_errors: list[float] = field(default_factory=list)

    @property
    def mean_correction_m(self) -> float:
        """Mean applied correction magnitude in metres."""
        if not self.correction_errors:
            return 0.0
        return float(np.mean(self.correction_errors))

    @property
    def tilm_acceptance_rate(self) -> float:
        """Fraction of TILM attempts that resulted in a correction."""
        if self.n_tilm_attempts == 0:
            return 0.0
        return self.n_tilm_corrections / self.n_tilm_attempts


# ── main class ────────────────────────────────────────────────────────────────

class MatchingCorrector:
    """Applies TILM-derived position corrections to an EKFVIO instance.

    Args:
        matcher: Built TemporalMatcher with descriptor index.
        ekf: EKFVIO instance to update.
        min_confidence: Minimum match confidence in [0, 1] to apply
            a correction. Rejects low-quality matches.
        correction_interval: Minimum seconds between two consecutive TILM
            corrections. Prevents over-correction at high frame rates.
        correction_covariance: Optional 3×3 covariance matrix for the
            landmark update. If None, uses EKFVIO default (R_landmark).
    """

    def __init__(
        self,
        matcher: TemporalMatcher,
        ekf: EKFVIO,
        min_confidence: float = 0.3,
        correction_interval: float = 1.0,
        correction_covariance: Optional[np.ndarray] = None,
    ) -> None:
        self.matcher = matcher
        self.ekf = ekf
        self.min_confidence = min_confidence
        self.correction_interval = correction_interval
        self.correction_covariance = correction_covariance

        self._gnss_available: bool = True
        self._last_correction_time: float = -np.inf
        self._last_match: Optional[MatchResult] = None
        self.stats = CorrectorStats()

    # ── mode control ──────────────────────────────────────────────────────────

    def set_gnss_available(self, available: bool) -> None:
        """Switch between GNSS-active and GNSS-lost modes.

        Args:
            available: True = GNSS updates are applied; False = TILM mode.
        """
        self._gnss_available = available
        self.ekf.set_gnss_active(available)

    @property
    def is_gnss_available(self) -> bool:
        """Whether GNSS is currently active."""
        return self._gnss_available

    @property
    def is_using_tilm(self) -> bool:
        """Whether the corrector is operating in TILM correction mode."""
        return not self._gnss_available

    # ── main update ───────────────────────────────────────────────────────────

    def update(
        self,
        observations: list[Landmark],
        timestamp: float,
        gnss_position: Optional[np.ndarray] = None,
        gnss_accuracy: float = 3.0,
    ) -> Optional[MatchResult]:
        """Apply one navigation update step.

        In GNSS-active mode, applies a GNSS measurement to the EKF and
        skips TILM matching. In GNSS-lost mode, attempts TILM matching
        and applies the correction if confidence and timing criteria pass.

        Args:
            observations: Current frame landmark observations.
            timestamp: Frame timestamp in seconds.
            gnss_position: Optional GNSS NED position (3,). If provided and
                GNSS is active, this is fed to the EKF.
            gnss_accuracy: GNSS horizontal accuracy (1-sigma) in metres.

        Returns:
            MatchResult if a TILM match was attempted, else None.
        """
        self.stats.n_frames += 1

        if self._gnss_available:
            if gnss_position is not None:
                self.ekf.update_gnss(
                    np.asarray(gnss_position, dtype=np.float64),
                    accuracy=gnss_accuracy,
                )
                self.stats.n_gnss_updates += 1
            return None

        # ── GNSS-lost: attempt TILM correction ────────────────────────────────
        self.stats.n_tilm_attempts += 1

        position_prior = self.ekf.state.position.copy()
        result = self.matcher.match(observations, position_prior=position_prior)
        self._last_match = result

        if not result.is_valid:
            self.stats.n_tilm_rejected += 1
            return result

        if result.confidence < self.min_confidence:
            self.stats.n_tilm_rejected += 1
            return result

        elapsed = timestamp - self._last_correction_time
        if elapsed < self.correction_interval:
            self.stats.n_tilm_rejected += 1
            return result

        # Apply correction to EKF
        self.ekf.update_landmark(
            result.position_correction_ned,
            correction_covariance=self.correction_covariance,
        )
        self._last_correction_time = timestamp
        self.stats.n_tilm_corrections += 1
        self.stats.correction_errors.append(
            float(np.linalg.norm(result.position_correction_ned))
        )

        return result

    # ── accessors ─────────────────────────────────────────────────────────────

    @property
    def last_match(self) -> Optional[MatchResult]:
        """Most recent MatchResult from a TILM attempt, or None."""
        return self._last_match

    def reset_stats(self) -> None:
        """Reset running statistics."""
        self.stats = CorrectorStats()
