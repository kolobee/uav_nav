"""Temporal landmark matching for relocalization.

Matches current landmark observations against the TILM to estimate
position corrections when GNSS is unavailable.

Algorithm:
1. Descriptor voting: each observation votes for the TILM node whose
   landmark embedding is most similar (within the same semantic class).
2. Candidate retrieval: top-K nodes by vote count, optionally filtered
   by distance to a position prior.
3. Geometric verification: greedy descriptor matching within each
   candidate node, count consistent pairs.
4. Correction: position_correction = best_node.position_ned - position_prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from uav_nav.memory.tilm import TILM, TILMNode
from uav_nav.perception.landmark_extractor import Landmark


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    """Result of matching current observations against the TILM.

    Attributes:
        matched_node_id: ID of the best-matching TILM node, or None.
        score: Similarity score in [0, 1]. Higher is better.
        matched_pairs: List of (observed_landmark_idx, tilm_landmark_idx).
        position_correction_ned: Estimated position correction, shape (3,).
        confidence: Overall confidence in the match in [0, 1].
        n_inliers: Number of geometrically consistent matches.
        is_valid: Whether the match quality exceeds the minimum threshold.
    """

    matched_node_id: Optional[int]
    score: float
    matched_pairs: list[tuple[int, int]] = field(default_factory=list)
    position_correction_ned: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    confidence: float = 0.0
    n_inliers: int = 0
    is_valid: bool = False


# ── matcher ───────────────────────────────────────────────────────────────────

class TemporalMatcher:
    """Matches a set of current landmark observations to the TILM.

    Uses descriptor similarity and geometric consistency checking to
    identify the most likely TILM node and compute a position correction.

    Args:
        tilm: The built TILM to match against.
        descriptor_threshold: Minimum cosine similarity to accept a match
            (0–1 for L2-normalised vectors; 1 − cosine_sim used internally).
        geometric_threshold: Maximum 3-D distance (m) for geometric inlier
            check (currently used as soft filter on descriptor distance).
        min_inliers: Minimum matched pairs for a valid result.
        top_k_nodes: Number of candidate nodes to score before selecting best.
    """

    def __init__(
        self,
        tilm: TILM,
        descriptor_threshold: float = 0.5,
        geometric_threshold: float = 3.0,
        min_inliers: int = 2,
        top_k_nodes: int = 5,
    ) -> None:
        self.tilm = tilm
        self.descriptor_threshold = descriptor_threshold
        self.geometric_threshold = geometric_threshold
        self.min_inliers = min_inliers
        self.top_k_nodes = top_k_nodes

        # Built by build_descriptor_index()
        self._entries: list[tuple[int, int, int]] = []  # (node_id, lm_idx, class_id)
        self._desc_matrix: Optional[np.ndarray] = None  # (N_entries, D)

    # ── index ─────────────────────────────────────────────────────────────────

    def build_descriptor_index(self) -> None:
        """Pre-build a brute-force index of all TILM landmark descriptors.

        Call this once after the TILM is fully built. Automatically re-called
        by ``match()`` if the index is stale (None).
        """
        self._entries = []
        descs: list[np.ndarray] = []
        for node in self.tilm.all_nodes:
            for j, lm in enumerate(node.landmarks):
                self._entries.append((node.node_id, j, lm.class_id))
                descs.append(lm.descriptor)
        if descs:
            self._desc_matrix = np.stack(descs, axis=0).astype(np.float32)
        else:
            self._desc_matrix = np.empty((0, 1), dtype=np.float32)

    # ── retrieval ─────────────────────────────────────────────────────────────

    def _retrieve_candidates(
        self,
        observations: list[Landmark],
        position_prior: Optional[np.ndarray],
    ) -> list[TILMNode]:
        """Retrieve top-K candidate TILM nodes via descriptor voting."""
        if self._desc_matrix is None or self._desc_matrix.shape[0] == 0:
            return []
        if not observations:
            return []

        obs_descs = np.stack(
            [lm.descriptor for lm in observations], axis=0
        ).astype(np.float32)                                  # (M, D)
        sims = obs_descs @ self._desc_matrix.T                # (M, N_entries)

        vote_scores: dict[int, float] = {}
        for m_idx, obs in enumerate(observations):
            row = sims[m_idx].copy()
            # Mask entries with different semantic class
            for e_idx, (nid, lm_idx, cls) in enumerate(self._entries):
                if cls != obs.class_id:
                    row[e_idx] = -1.0
            best_e = int(np.argmax(row))
            nid = self._entries[best_e][0]
            vote_scores[nid] = vote_scores.get(nid, 0.0) + float(row[best_e])

        # Sort by vote score; optionally bias by proximity to position_prior
        ranked = sorted(vote_scores.items(), key=lambda x: -x[1])

        # If position prior given, down-rank very distant nodes
        if position_prior is not None:
            p = np.asarray(position_prior, dtype=np.float64)

            def _rank_key(item: tuple[int, float]) -> float:
                nid, score = item
                d = float(np.linalg.norm(
                    self.tilm.get_node(nid).position_ned - p
                ))
                return -(score / (1.0 + d * 0.1))

            ranked = sorted(vote_scores.items(), key=_rank_key)

        result: list[TILMNode] = []
        seen: set[int] = set()
        for nid, _ in ranked:
            if nid not in seen:
                seen.add(nid)
                result.append(self.tilm.get_node(nid))
            if len(result) >= self.top_k_nodes:
                break
        return result

    # ── geometric verification ────────────────────────────────────────────────

    def _geometric_verify(
        self,
        observations: list[Landmark],
        candidate: TILMNode,
    ) -> tuple[list[tuple[int, int]], np.ndarray]:
        """Greedy descriptor matching with same-class constraint.

        Args:
            observations: Current landmark observations.
            candidate: Candidate TILM node.

        Returns:
            (inlier_pairs, position_correction) where inlier_pairs is a list
            of (obs_idx, map_idx) tuples and position_correction is zeros
            (the caller computes the NED correction from node position).
        """
        if not observations or not candidate.landmarks:
            return [], np.zeros(3, dtype=np.float64)

        obs_descs = np.stack(
            [lm.descriptor for lm in observations], axis=0
        ).astype(np.float32)
        map_descs = np.stack(
            [lm.descriptor for lm in candidate.landmarks], axis=0
        ).astype(np.float32)
        sims = obs_descs @ map_descs.T                        # (M_obs, M_map)

        inliers: list[tuple[int, int]] = []
        used_map: set[int] = set()

        # Greedy matching: best-first by cosine similarity
        pairs = []
        for obs_idx, obs in enumerate(observations):
            for map_idx, map_lm in enumerate(candidate.landmarks):
                if map_lm.class_id != obs.class_id:
                    continue
                cos_sim = float(sims[obs_idx, map_idx])
                if cos_sim >= (1.0 - self.descriptor_threshold):
                    pairs.append((cos_sim, obs_idx, map_idx))

        pairs.sort(key=lambda x: -x[0])
        used_obs: set[int] = set()
        for cos_sim, obs_idx, map_idx in pairs:
            if obs_idx in used_obs or map_idx in used_map:
                continue
            inliers.append((obs_idx, map_idx))
            used_obs.add(obs_idx)
            used_map.add(map_idx)

        return inliers, np.zeros(3, dtype=np.float64)

    # ── main entry ────────────────────────────────────────────────────────────

    def match(
        self,
        observations: list[Landmark],
        position_prior: Optional[np.ndarray] = None,
    ) -> MatchResult:
        """Match current observations to the best TILM node.

        Args:
            observations: Current frame landmark observations.
            position_prior: Optional NED position estimate to bias retrieval,
                shape (3,). Also used to compute position_correction_ned.

        Returns:
            MatchResult with best node, score, and position correction.
        """
        if self._desc_matrix is None:
            self.build_descriptor_index()

        _invalid = MatchResult(matched_node_id=None, score=0.0, is_valid=False)

        if not observations or self.tilm.n_nodes() == 0:
            return _invalid

        candidates = self._retrieve_candidates(observations, position_prior)
        if not candidates:
            return _invalid

        best: Optional[MatchResult] = None
        for candidate in candidates:
            inliers, _ = self._geometric_verify(observations, candidate)
            if len(inliers) < self.min_inliers:
                continue

            denom = max(len(observations), candidate.n_landmarks)
            score = len(inliers) / denom

            correction = np.zeros(3, dtype=np.float64)
            if position_prior is not None:
                correction = (
                    candidate.position_ned
                    - np.asarray(position_prior, dtype=np.float64)
                )

            result = MatchResult(
                matched_node_id=candidate.node_id,
                score=score,
                matched_pairs=inliers,
                position_correction_ned=correction,
                confidence=score,
                n_inliers=len(inliers),
                is_valid=True,
            )
            if best is None or score > best.score:
                best = result

        return best if best is not None else _invalid
