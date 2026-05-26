"""Unit tests for TILM, TILMBuilder, and TemporalMatcher."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from uav_nav.memory.tilm import TILM, TILMEdge, TILMNode
from uav_nav.memory.tilm_builder import BuilderConfig, TILMBuilder
from uav_nav.memory.temporal_matcher import MatchResult, TemporalMatcher
from uav_nav.perception.landmark_extractor import Landmark


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_descriptor(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(dim).astype(np.float32)
    return d / np.linalg.norm(d)


def _make_landmark(
    class_id: int = 0,
    seed: int = 0,
    dim: int = 8,
) -> Landmark:
    return Landmark(
        position_3d=np.zeros(3, dtype=np.float32),
        semantic_class="tree",
        class_id=class_id,
        descriptor=_make_descriptor(seed, dim),
        confidence=0.9,
        mask_area=500,
        centroid_uv=(100.0, 100.0),
    )


def _make_node(
    node_id: int,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_landmarks: int = 3,
    class_id: int = 0,
    dim: int = 8,
) -> TILMNode:
    lms = [_make_landmark(class_id=class_id, seed=node_id * 100 + i, dim=dim)
           for i in range(n_landmarks)]
    place_desc = _make_descriptor(node_id, dim)
    return TILMNode(
        node_id=node_id,
        position_ned=np.array(position, dtype=np.float64),
        landmarks=lms,
        place_descriptor=place_desc,
    )


def _make_pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def _make_tilm_with_nodes(n: int = 3, spacing: float = 10.0, dim: int = 8) -> TILM:
    tilm = TILM()
    for i in range(n):
        node = _make_node(i, position=(i * spacing, 0.0, 0.0), dim=dim)
        tilm.add_node(node)
    return tilm


# ── TILM tests ────────────────────────────────────────────────────────────────

class TestTILMBasic:
    def test_empty_tilm_has_zero_nodes(self) -> None:
        tilm = TILM()
        assert tilm.n_nodes() == 0

    def test_add_node_returns_id(self) -> None:
        tilm = TILM()
        node = _make_node(0)
        nid = tilm.add_node(node)
        assert nid == 0

    def test_add_node_increments_count(self) -> None:
        tilm = TILM()
        for i in range(4):
            tilm.add_node(_make_node(i))
        assert tilm.n_nodes() == 4

    def test_get_node_returns_correct_node(self) -> None:
        tilm = TILM()
        node = _make_node(0, position=(1.0, 2.0, 3.0))
        tilm.add_node(node)
        retrieved = tilm.get_node(0)
        np.testing.assert_allclose(retrieved.position_ned, [1.0, 2.0, 3.0])

    def test_get_node_missing_raises(self) -> None:
        tilm = TILM()
        with pytest.raises(KeyError):
            tilm.get_node(99)

    def test_all_nodes_returns_all(self) -> None:
        tilm = _make_tilm_with_nodes(5)
        assert len(tilm.all_nodes) == 5

    def test_initialise_clears_nodes(self) -> None:
        tilm = _make_tilm_with_nodes(3)
        tilm.initialise()
        assert tilm.n_nodes() == 0


class TestTILMEdge:
    def test_add_edge_valid(self) -> None:
        tilm = _make_tilm_with_nodes(2)
        edge = TILMEdge(src_id=0, dst_id=1, relative_pose=np.eye(4), distance=10.0)
        tilm.add_edge(edge)  # should not raise

    def test_add_edge_missing_node_raises(self) -> None:
        tilm = _make_tilm_with_nodes(2)
        edge = TILMEdge(src_id=0, dst_id=99, relative_pose=np.eye(4), distance=5.0)
        with pytest.raises(KeyError):
            tilm.add_edge(edge)


class TestTILMNearestNode:
    def test_nearest_single_node(self) -> None:
        tilm = TILM()
        tilm.add_node(_make_node(0, position=(0.0, 0.0, 0.0)))
        result = tilm.nearest_node(np.array([1.0, 0.0, 0.0]))
        assert len(result) == 1
        assert result[0][0] == 0

    def test_nearest_correct_node(self) -> None:
        tilm = _make_tilm_with_nodes(3, spacing=10.0)
        result = tilm.nearest_node(np.array([19.0, 0.0, 0.0]), k=1)
        assert result[0][0] == 2  # node at (20, 0, 0) is nearest

    def test_nearest_k_returns_k(self) -> None:
        tilm = _make_tilm_with_nodes(5, spacing=10.0)
        result = tilm.nearest_node(np.array([20.0, 0.0, 0.0]), k=3)
        assert len(result) == 3

    def test_nearest_sorted_ascending(self) -> None:
        tilm = _make_tilm_with_nodes(4, spacing=10.0)
        result = tilm.nearest_node(np.array([15.0, 0.0, 0.0]), k=4)
        dists = [d for _, d in result]
        assert dists == sorted(dists)

    def test_nearest_empty_tilm(self) -> None:
        tilm = TILM()
        result = tilm.nearest_node(np.array([0.0, 0.0, 0.0]))
        assert result == []


class TestTILMShortestPath:
    def test_path_connected_nodes(self) -> None:
        tilm = _make_tilm_with_nodes(3)
        tilm.add_edge(TILMEdge(0, 1, np.eye(4), 10.0))
        tilm.add_edge(TILMEdge(1, 2, np.eye(4), 10.0))
        path = tilm.shortest_path(0, 2)
        assert path == [0, 1, 2]

    def test_path_disconnected_returns_none(self) -> None:
        tilm = _make_tilm_with_nodes(2)
        path = tilm.shortest_path(0, 1)
        assert path is None


class TestTILMSaveLoad:
    def test_save_load_roundtrip(self) -> None:
        tilm = _make_tilm_with_nodes(3)
        with tempfile.NamedTemporaryFile(suffix=".tilm", delete=False) as f:
            p = Path(f.name)
        try:
            tilm.save(p)
            loaded = TILM.load(p)
            assert loaded.n_nodes() == 3
            np.testing.assert_allclose(
                loaded.get_node(1).position_ned,
                tilm.get_node(1).position_ned,
            )
        finally:
            p.unlink(missing_ok=True)


# ── TILMNode tests ────────────────────────────────────────────────────────────

class TestTILMNode:
    def test_n_landmarks(self) -> None:
        node = _make_node(0, n_landmarks=4)
        assert node.n_landmarks == 4

    def test_landmark_classes(self) -> None:
        lm = _make_landmark(class_id=1)
        lm.semantic_class = "rock"
        node = TILMNode(
            node_id=0,
            position_ned=np.zeros(3),
            landmarks=[lm, _make_landmark(class_id=0)],
        )
        classes = node.landmark_classes()
        assert "rock" in classes
        assert "tree" in classes


# ── TILMBuilder tests ─────────────────────────────────────────────────────────

class TestTILMBuilderNodeCreation:
    def _make_builder(self, **kwargs) -> tuple[TILM, TILMBuilder]:
        cfg = BuilderConfig(**kwargs)
        tilm = TILM()
        builder = TILMBuilder(tilm, cfg)
        return tilm, builder

    def _landmarks(self, n: int = 3, dim: int = 8) -> list[Landmark]:
        return [_make_landmark(seed=i, dim=dim) for i in range(n)]

    def test_first_frame_creates_node(self) -> None:
        tilm, builder = self._make_builder(min_node_distance=5.0)
        nid = builder.process_frame(_make_pose(0, 0, 0), self._landmarks(), 0.0)
        assert nid == 0
        assert tilm.n_nodes() == 1

    def test_close_frame_skipped(self) -> None:
        tilm, builder = self._make_builder(min_node_distance=5.0)
        builder.process_frame(_make_pose(0, 0, 0), self._landmarks(), 0.0)
        nid = builder.process_frame(_make_pose(2, 0, 0), self._landmarks(), 0.1)
        assert nid is None
        assert tilm.n_nodes() == 1

    def test_far_frame_creates_node(self) -> None:
        tilm, builder = self._make_builder(min_node_distance=5.0)
        builder.process_frame(_make_pose(0, 0, 0), self._landmarks(), 0.0)
        nid = builder.process_frame(_make_pose(10, 0, 0), self._landmarks(), 1.0)
        assert nid == 1
        assert tilm.n_nodes() == 2

    def test_sequential_edge_added(self) -> None:
        tilm, builder = self._make_builder(
            min_node_distance=5.0, use_loop_closure=False
        )
        builder.process_frame(_make_pose(0, 0, 0), self._landmarks(), 0.0)
        builder.process_frame(_make_pose(10, 0, 0), self._landmarks(), 1.0)
        assert tilm._graph.has_edge(0, 1)

    def test_too_few_landmarks_skips_node(self) -> None:
        tilm, builder = self._make_builder(min_landmarks_per_node=4)
        nid = builder.process_frame(_make_pose(0, 0, 0), self._landmarks(2), 0.0)
        assert nid is None
        assert tilm.n_nodes() == 0

    def test_multiple_nodes_in_sequence(self) -> None:
        tilm, builder = self._make_builder(
            min_node_distance=5.0, use_loop_closure=False
        )
        for i in range(5):
            builder.process_frame(_make_pose(i * 10, 0, 0), self._landmarks(), float(i))
        assert tilm.n_nodes() == 5

    def test_finalise_returns_tilm(self) -> None:
        _, builder = self._make_builder()
        result = builder.finalise()
        assert isinstance(result, TILM)


class TestTILMBuilderLoopClosure:
    def test_loop_closure_disabled(self) -> None:
        cfg = BuilderConfig(min_node_distance=5.0, use_loop_closure=False)
        tilm = TILM()
        builder = TILMBuilder(tilm, cfg)
        lms = [_make_landmark(seed=i) for i in range(3)]
        for i in range(4):
            builder.process_frame(_make_pose(i * 10, 0, 0), lms, float(i))
        # Only sequential edges
        assert tilm._graph.number_of_edges() == 3

    def test_loop_closure_creates_extra_edge(self) -> None:
        dim = 8

        def _similar_lms() -> list[Landmark]:
            # Fixed seed every call → identical descriptors across nodes → sim=1.0
            rng = np.random.default_rng(42)
            d = rng.standard_normal(dim).astype(np.float32)
            d /= np.linalg.norm(d)
            lms = []
            for _ in range(3):
                noise = rng.standard_normal(dim).astype(np.float32) * 0.01
                di = d + noise
                di /= np.linalg.norm(di)
                lms.append(Landmark(
                    position_3d=np.zeros(3, dtype=np.float32),
                    semantic_class="tree",
                    class_id=0,
                    descriptor=di,
                    confidence=0.9,
                    mask_area=500,
                    centroid_uv=(0.0, 0.0),
                ))
            return lms

        cfg = BuilderConfig(
            min_node_distance=5.0,
            use_loop_closure=True,
            loop_closure_threshold=0.7,
            edge_max_distance=20.0,
        )
        tilm = TILM()
        builder = TILMBuilder(tilm, cfg)
        # Node 0 at (0, 0, 0)
        builder.process_frame(_make_pose(0, 0, 0), _similar_lms(), 0.0)
        # Node 1 at (10, 0, 0) — dist=10 ≥ 5 → created; sequential edge 0→1
        builder.process_frame(_make_pose(10, 0, 0), _similar_lms(), 1.0)
        # Node 2 at (5, 0, 0) — dist=5 from node 1 ≥ 5 → created; sequential edge 1→2
        # dist from node 0 = 5, in [1, 20] → loop closure candidate
        builder.process_frame(_make_pose(5, 0, 0), _similar_lms(), 2.0)
        assert tilm.n_nodes() == 3
        # Sequential: 0→1, 1→2 (2 edges) + at least 1 loop closure
        assert tilm._graph.number_of_edges() > 2


# ── TemporalMatcher tests ─────────────────────────────────────────────────────

class TestTemporalMatcherIndex:
    def test_build_index_empty_tilm(self) -> None:
        tilm = TILM()
        matcher = TemporalMatcher(tilm)
        matcher.build_descriptor_index()
        assert matcher._desc_matrix is not None
        assert matcher._desc_matrix.shape[0] == 0

    def test_build_index_correct_size(self) -> None:
        tilm = _make_tilm_with_nodes(3, dim=8)  # 3 nodes × 3 landmarks = 9 entries
        matcher = TemporalMatcher(tilm)
        matcher.build_descriptor_index()
        assert matcher._desc_matrix.shape[0] == 9
        assert matcher._desc_matrix.shape[1] == 8

    def test_build_index_entries_match_nodes(self) -> None:
        tilm = _make_tilm_with_nodes(2, dim=8)
        matcher = TemporalMatcher(tilm)
        matcher.build_descriptor_index()
        node_ids = {e[0] for e in matcher._entries}
        assert node_ids == {0, 1}


class TestTemporalMatcherRetrieve:
    def _make_matcher(self, n_nodes: int = 3) -> tuple[TILM, TemporalMatcher]:
        tilm = _make_tilm_with_nodes(n_nodes, spacing=10.0, dim=8)
        matcher = TemporalMatcher(tilm, top_k_nodes=n_nodes)
        matcher.build_descriptor_index()
        return tilm, matcher

    def test_no_observations_returns_invalid(self) -> None:
        tilm, matcher = self._make_matcher()
        result = matcher.match([], position_prior=None)
        assert not result.is_valid
        assert result.matched_node_id is None

    def test_empty_tilm_returns_invalid(self) -> None:
        tilm = TILM()
        matcher = TemporalMatcher(tilm)
        matcher.build_descriptor_index()
        obs = [_make_landmark(seed=0)]
        result = matcher.match(obs)
        assert not result.is_valid

    def test_match_correct_node(self) -> None:
        dim = 16
        tilm = TILM()

        # Build 3 nodes with clearly different descriptors
        rng = np.random.default_rng(0)
        prototypes = []
        for i in range(3):
            base = np.zeros(dim, dtype=np.float32)
            base[i * (dim // 3)] = 1.0
            prototypes.append(base)
            lms = []
            for j in range(3):
                noise = rng.standard_normal(dim).astype(np.float32) * 0.05
                d = base + noise
                d /= np.linalg.norm(d)
                lms.append(Landmark(
                    position_3d=np.zeros(3, dtype=np.float32),
                    semantic_class="tree",
                    class_id=0,
                    descriptor=d,
                    confidence=0.9,
                    mask_area=500,
                    centroid_uv=(0.0, 0.0),
                ))
            place = prototypes[i].copy() / np.linalg.norm(prototypes[i])
            node = TILMNode(
                node_id=i,
                position_ned=np.array([i * 20.0, 0.0, 0.0]),
                landmarks=lms,
                place_descriptor=place,
            )
            tilm.add_node(node)

        matcher = TemporalMatcher(
            tilm,
            descriptor_threshold=0.5,
            min_inliers=1,
            top_k_nodes=3,
        )
        matcher.build_descriptor_index()

        # Observe descriptors similar to node 1's prototype
        obs_d = prototypes[1].copy()
        obs_d /= np.linalg.norm(obs_d)
        observations = [
            Landmark(
                position_3d=np.zeros(3, dtype=np.float32),
                semantic_class="tree",
                class_id=0,
                descriptor=obs_d,
                confidence=0.9,
                mask_area=500,
                centroid_uv=(0.0, 0.0),
            )
        ]
        result = matcher.match(observations)
        assert result.is_valid
        assert result.matched_node_id == 1


class TestTemporalMatcherResult:
    def _build_simple_matcher(self) -> tuple[TILM, TemporalMatcher, list[Landmark]]:
        dim = 8
        tilm = TILM()
        rng = np.random.default_rng(7)

        d = rng.standard_normal(dim).astype(np.float32)
        d /= np.linalg.norm(d)
        lms = []
        for _ in range(3):
            noise = rng.standard_normal(dim).astype(np.float32) * 0.02
            di = d + noise
            di /= np.linalg.norm(di)
            lms.append(Landmark(
                position_3d=np.zeros(3, dtype=np.float32),
                semantic_class="tree",
                class_id=0,
                descriptor=di,
                confidence=0.9,
                mask_area=500,
                centroid_uv=(0.0, 0.0),
            ))
        node = TILMNode(
            node_id=0,
            position_ned=np.array([10.0, 5.0, 0.0]),
            landmarks=lms,
            place_descriptor=d.copy(),
        )
        tilm.add_node(node)

        matcher = TemporalMatcher(tilm, descriptor_threshold=0.5, min_inliers=1, top_k_nodes=1)
        matcher.build_descriptor_index()

        obs_d = d.copy()
        observations = [
            Landmark(
                position_3d=np.zeros(3, dtype=np.float32),
                semantic_class="tree",
                class_id=0,
                descriptor=obs_d,
                confidence=0.9,
                mask_area=500,
                centroid_uv=(0.0, 0.0),
            )
        ]
        return tilm, matcher, observations

    def test_match_result_is_valid(self) -> None:
        _, matcher, obs = self._build_simple_matcher()
        result = matcher.match(obs)
        assert result.is_valid

    def test_match_result_score_in_range(self) -> None:
        _, matcher, obs = self._build_simple_matcher()
        result = matcher.match(obs)
        assert 0.0 <= result.score <= 1.0

    def test_match_result_n_inliers_positive(self) -> None:
        _, matcher, obs = self._build_simple_matcher()
        result = matcher.match(obs)
        assert result.n_inliers >= 1

    def test_position_correction_with_prior(self) -> None:
        _, matcher, obs = self._build_simple_matcher()
        prior = np.array([8.0, 4.0, 0.0])
        result = matcher.match(obs, position_prior=prior)
        expected_correction = np.array([10.0, 5.0, 0.0]) - prior
        np.testing.assert_allclose(
            result.position_correction_ned, expected_correction, atol=1e-9
        )

    def test_position_correction_zero_without_prior(self) -> None:
        _, matcher, obs = self._build_simple_matcher()
        result = matcher.match(obs, position_prior=None)
        np.testing.assert_allclose(result.position_correction_ned, np.zeros(3))

    def test_index_rebuilt_on_match_if_none(self) -> None:
        tilm = _make_tilm_with_nodes(2)
        matcher = TemporalMatcher(tilm, min_inliers=1)
        # Do NOT call build_descriptor_index — match() should auto-build
        obs = [_make_landmark(seed=0)]
        result = matcher.match(obs)
        assert matcher._desc_matrix is not None  # was built


class TestTemporalMatcherMinInliers:
    def test_insufficient_inliers_returns_invalid(self) -> None:
        dim = 8
        tilm = TILM()
        # Node with very different descriptors from the observation
        lm = Landmark(
            position_3d=np.zeros(3, dtype=np.float32),
            semantic_class="rock",
            class_id=1,  # different class
            descriptor=np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            confidence=0.9,
            mask_area=500,
            centroid_uv=(0.0, 0.0),
        )
        node = TILMNode(node_id=0, position_ned=np.zeros(3), landmarks=[lm, lm])
        tilm.add_node(node)

        obs = Landmark(
            position_3d=np.zeros(3, dtype=np.float32),
            semantic_class="tree",
            class_id=0,  # different class → no match
            descriptor=np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            confidence=0.9,
            mask_area=500,
            centroid_uv=(0.0, 0.0),
        )
        matcher = TemporalMatcher(tilm, min_inliers=2)
        matcher.build_descriptor_index()
        result = matcher.match([obs])
        assert not result.is_valid


class TestMatchResult:
    def test_default_result_invalid(self) -> None:
        r = MatchResult(matched_node_id=None, score=0.0)
        assert not r.is_valid
        assert r.n_inliers == 0
        assert r.confidence == 0.0

    def test_position_correction_default_zeros(self) -> None:
        r = MatchResult(matched_node_id=None, score=0.0)
        np.testing.assert_allclose(r.position_correction_ned, np.zeros(3))
