"""Holistic place descriptor computation and querying.

Aggregates per-landmark descriptors into a single compact place vector
suitable for fast global retrieval (e.g. with NetVLAD-style aggregation
or simple mean pooling over semantic-class-conditioned sub-vectors).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from uav_nav.perception.landmark_extractor import Landmark


class AggregationMethod(str, Enum):
    """Supported descriptor aggregation strategies."""

    MEAN = "mean"                # Simple mean of all descriptors
    CLASS_MEAN = "class_mean"    # Mean per class, then concatenate
    VLAD = "vlad"               # VLAD aggregation with learned codebook
    FISHER = "fisher"           # Fisher Vector encoding


@dataclass
class PlaceQuery:
    """A query for place retrieval.

    Attributes:
        descriptor: Holistic place descriptor, shape (D,), float32.
        position_prior: Optional NED position prior, shape (3,).
        timestamp: Query timestamp in seconds.
        source_landmarks: Original landmarks used to compute the descriptor.
    """

    descriptor: np.ndarray   # (D,)
    position_prior: Optional[np.ndarray] = None
    timestamp: float = 0.0
    source_landmarks: list[Landmark] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.source_landmarks is None:
            self.source_landmarks = []


class PlaceDescriptor:
    """Computes and manages holistic place descriptors.

    Args:
        method: Aggregation method to use.
        output_dim: Output descriptor dimension.
        class_order: Ordered list of semantic class names for class-conditioned
            aggregation. Required for CLASS_MEAN method.
        codebook_size: Number of VLAD/Fisher codebook centroids.
    """

    def __init__(
        self,
        method: AggregationMethod = AggregationMethod.CLASS_MEAN,
        output_dim: int = 256,
        class_order: Optional[list[str]] = None,
        codebook_size: int = 64,
    ) -> None:
        self.method = method
        self.output_dim = output_dim
        self.class_order = class_order or ["tree", "rock", "river"]
        self.codebook_size = codebook_size
        self._codebook: Optional[np.ndarray] = None

    def compute(self, landmarks: list[Landmark]) -> np.ndarray:
        """Compute a holistic place descriptor from a list of landmarks.

        Args:
            landmarks: Landmark observations from the current frame.

        Returns:
            L2-normalised place descriptor, shape (output_dim,), float32.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("PlaceDescriptor.compute is not yet implemented.")

    def build_query(
        self,
        landmarks: list[Landmark],
        position_prior: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
    ) -> PlaceQuery:
        """Build a PlaceQuery from landmark observations.

        Args:
            landmarks: Current frame landmark observations.
            position_prior: Optional NED position prior.
            timestamp: Frame timestamp.

        Returns:
            PlaceQuery ready for retrieval.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("PlaceDescriptor.build_query is not yet implemented.")

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two place descriptors.

        Args:
            a: First descriptor, shape (D,), float32.
            b: Second descriptor, shape (D,), float32.

        Returns:
            Cosine similarity in [-1, 1].
        """
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-9:
            return 0.0
        return float(np.dot(a, b) / denom)

    def fit_codebook(self, descriptors: np.ndarray) -> None:
        """Fit the VLAD/Fisher codebook on a set of training descriptors.

        Args:
            descriptors: Training descriptor matrix, shape (N, D), float32.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("PlaceDescriptor.fit_codebook is not yet implemented.")
