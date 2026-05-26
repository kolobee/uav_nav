"""Temporal landmark matching for relocalization.

Matches current landmark observations against the TILM to estimate
position corrections when GNSS is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from uav_nav.memory.tilm import TILM, TILMNode
from uav_nav.perception.landmark_extractor import Landmark


@dataclass
class MatchResult:
    """Result of matching current observations against the TILM.

    Attributes:
        matched_node_id: ID of the best-matching TILM node, or None.
        score: Similarity score in [0, 1]. Higher is better.
        matched_pairs: List of (observed_landmark_idx, tilm_landmark_idx) pairs.
        position_correction_ned: Estimated position correction, shape (3,).
            Zero vector if matching failed.
        confidence: Overall confidence in the match in [0, 1].
        n_inliers: Number of geometrically consistent matches.
        is_valid: Whether the match quality exceeds the minimum threshold.
    """

    matched_node_id: Optional[int]
    score: float
    matched_pairs: list[tuple[int, int]] = field(default_factory=list)
    position_correction_ned: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    confidence: float = 0.0
    n_inliers: int = 0
    is_valid: bool = False


class TemporalMatcher:
    """Matches a set of current landmark observations to the TILM.

    Uses descriptor similarity and geometric consistency checking to
    identify the most likely TILM node for place recognition and to
    compute a position correction for the EKF update.

    Args:
        tilm: The built TILM to match against.
        descriptor_threshold: Maximum descriptor distance to consider a
            candidate match pair (0–2 for L2-normalised vectors).
        geometric_threshold: Maximum 3-D distance (m) inconsistency for
            RANSAC geometric verification.
        min_inliers: Minimum inlier count for a valid match.
        top_k_nodes: Number of candidate nodes to retrieve before
            geometric verification.
    """

    def __init__(
        self,
        tilm: TILM,
        descriptor_threshold: float = 0.5,
        geometric_threshold: float = 3.0,
        min_inliers: int = 3,
        top_k_nodes: int = 5,
    ) -> None:
        self.tilm = tilm
        self.descriptor_threshold = descriptor_threshold
        self.geometric_threshold = geometric_threshold
        self.min_inliers = min_inliers
        self.top_k_nodes = top_k_nodes

    def match(
        self,
        observations: list[Landmark],
        position_prior: Optional[np.ndarray] = None,
    ) -> MatchResult:
        """Match current observations to the best TILM node.

        Args:
            observations: Current frame landmark observations.
            position_prior: Optional NED position prior to restrict
                candidate search, shape (3,).

        Returns:
            MatchResult with best node, score, and position correction.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TemporalMatcher.match is not yet implemented.")

    def _retrieve_candidates(
        self,
        observations: list[Landmark],
        position_prior: Optional[np.ndarray],
    ) -> list[TILMNode]:
        """Retrieve top-K candidate TILM nodes via descriptor voting.

        Args:
            observations: Current landmark observations.
            position_prior: Optional position to bias retrieval.

        Returns:
            List of candidate TILMNode instances.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TemporalMatcher._retrieve_candidates is not yet implemented.")

    def _geometric_verify(
        self,
        observations: list[Landmark],
        candidate: TILMNode,
    ) -> tuple[list[tuple[int, int]], np.ndarray]:
        """Geometric consistency check via RANSAC on 3-D correspondences.

        Args:
            observations: Current observations.
            candidate: Candidate TILM node.

        Returns:
            Tuple (inlier_pairs, position_correction) where inlier_pairs is a
            list of (obs_idx, map_idx) tuples and position_correction is a
            NED offset estimate, shape (3,).

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("TemporalMatcher._geometric_verify is not yet implemented.")

    def build_descriptor_index(self) -> None:
        """Pre-build a FAISS (or brute-force) index of all TILM descriptors.

        Should be called after the TILM is fully built for fast retrieval.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError(
            "TemporalMatcher.build_descriptor_index is not yet implemented."
        )
