"""Unit tests for PlaceDescriptor utility methods.

Tests the methods that can be verified without NotImplementedError:
- similarity() computation (pure numpy, already implemented).
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_nav.memory.place_descriptor import PlaceDescriptor, AggregationMethod


class TestPlaceDescriptorSimilarity:
    """Tests for the cosine similarity utility."""

    def test_identical_vectors(self) -> None:
        """Cosine similarity of a vector with itself is 1.0."""
        pd = PlaceDescriptor()
        a = np.array([1.0, 0.0, 0.0, 1.0])
        a = a / np.linalg.norm(a)
        assert abs(pd.similarity(a, a) - 1.0) < 1e-9

    def test_orthogonal_vectors(self) -> None:
        """Cosine similarity of orthogonal vectors is 0.0."""
        pd = PlaceDescriptor()
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(pd.similarity(a, b)) < 1e-9

    def test_opposite_vectors(self) -> None:
        """Cosine similarity of antiparallel vectors is -1.0."""
        pd = PlaceDescriptor()
        a = np.array([1.0, 0.0, 0.0])
        b = -a
        assert abs(pd.similarity(a, b) + 1.0) < 1e-9

    def test_zero_vector_returns_zero(self) -> None:
        """Cosine similarity with a zero vector returns 0.0 (not NaN)."""
        pd = PlaceDescriptor()
        a = np.array([1.0, 2.0, 3.0])
        z = np.zeros(3)
        result = pd.similarity(a, z)
        assert result == 0.0

    def test_symmetry(self) -> None:
        """Cosine similarity is symmetric: sim(a, b) == sim(b, a)."""
        pd = PlaceDescriptor()
        rng = np.random.default_rng(0)
        a = rng.standard_normal(128)
        b = rng.standard_normal(128)
        assert abs(pd.similarity(a, b) - pd.similarity(b, a)) < 1e-12


class TestPlaceDescriptorConfig:
    """Tests for PlaceDescriptor configuration."""

    def test_default_class_order(self) -> None:
        """Default class order is ['tree', 'rock', 'river']."""
        pd = PlaceDescriptor()
        assert pd.class_order == ["tree", "rock", "river"]

    def test_custom_class_order(self) -> None:
        """Custom class order is stored correctly."""
        pd = PlaceDescriptor(class_order=["rock", "river"])
        assert pd.class_order == ["rock", "river"]

    def test_default_method(self) -> None:
        """Default aggregation method is CLASS_MEAN."""
        pd = PlaceDescriptor()
        assert pd.method == AggregationMethod.CLASS_MEAN
