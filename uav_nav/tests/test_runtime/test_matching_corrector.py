"""Unit tests for MatchingCorrector and MatchingEvaluator."""

from __future__ import annotations

import numpy as np
import pytest

from uav_nav.estimation.ekf_vio import EKFState, EKFVIO
from uav_nav.memory.tilm import TILM, TILMNode
from uav_nav.memory.temporal_matcher import TemporalMatcher
from uav_nav.perception.landmark_extractor import Landmark
from uav_nav.runtime.matching_corrector import CorrectorStats, MatchingCorrector
from uav_nav.eval.matching_eval import MatchingEvaluator, MatchingMetrics


# ── helpers ───────────────────────────────────────────────────────────────────

DIM = 16


def _make_descriptor(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(DIM).astype(np.float32)
    return d / np.linalg.norm(d)


def _make_landmark(class_id: int = 0, seed: int = 0) -> Landmark:
    return Landmark(
        position_3d=np.zeros(3, dtype=np.float32),
        semantic_class="tree",
        class_id=class_id,
        descriptor=_make_descriptor(seed),
        confidence=0.9,
        mask_area=500,
        centroid_uv=(0.0, 0.0),
    )


def _make_ekf(position=(0.0, 0.0, 0.0)) -> EKFVIO:
    state = EKFState(
        position=np.array(position, dtype=np.float64),
        velocity=np.zeros(3, dtype=np.float64),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        P=np.eye(18, dtype=np.float64) * 1e-2,
    )
    return EKFVIO(state)


def _build_tilm_and_matcher(
    node_position=(10.0, 5.0, 0.0),
    obs_seed: int = 42,
) -> tuple[TILM, TemporalMatcher, list[Landmark]]:
    """Build a single-node TILM and observations similar to that node."""
    rng = np.random.default_rng(obs_seed)
    base = rng.standard_normal(DIM).astype(np.float32)
    base /= np.linalg.norm(base)

    def _noisy(noise_scale=0.02):
        n = rng.standard_normal(DIM).astype(np.float32) * noise_scale
        d = base + n
        return (d / np.linalg.norm(d)).astype(np.float32)

    lms = [
        Landmark(
            position_3d=np.zeros(3, dtype=np.float32),
            semantic_class="tree", class_id=0,
            descriptor=_noisy(), confidence=0.9,
            mask_area=500, centroid_uv=(0.0, 0.0),
        )
        for _ in range(3)
    ]
    node = TILMNode(
        node_id=0,
        position_ned=np.array(node_position, dtype=np.float64),
        landmarks=lms,
        place_descriptor=base.copy(),
    )
    tilm = TILM()
    tilm.add_node(node)

    matcher = TemporalMatcher(tilm, descriptor_threshold=0.5, min_inliers=1, top_k_nodes=1)
    matcher.build_descriptor_index()

    # Observations identical to the node's first landmark → guaranteed match
    observations = [
        Landmark(
            position_3d=np.zeros(3, dtype=np.float32),
            semantic_class="tree", class_id=0,
            descriptor=lms[0].descriptor.copy(),
            confidence=0.9, mask_area=500, centroid_uv=(0.0, 0.0),
        )
    ]
    return tilm, matcher, observations


def _make_corrector(
    node_position=(10.0, 5.0, 0.0),
    ekf_position=(8.0, 4.0, 0.0),
    min_confidence=0.1,
    correction_interval=0.0,
) -> tuple[MatchingCorrector, EKFVIO]:
    _, matcher, _ = _build_tilm_and_matcher(node_position)
    ekf = _make_ekf(ekf_position)
    corrector = MatchingCorrector(
        matcher, ekf,
        min_confidence=min_confidence,
        correction_interval=correction_interval,
    )
    return corrector, ekf


# ── CorrectorStats tests ──────────────────────────────────────────────────────

class TestCorrectorStats:
    def test_initial_zeros(self) -> None:
        s = CorrectorStats()
        assert s.n_frames == 0
        assert s.n_gnss_updates == 0
        assert s.n_tilm_corrections == 0

    def test_mean_correction_empty(self) -> None:
        assert CorrectorStats().mean_correction_m == 0.0

    def test_acceptance_rate_zero_attempts(self) -> None:
        assert CorrectorStats().tilm_acceptance_rate == 0.0

    def test_acceptance_rate_computed(self) -> None:
        s = CorrectorStats()
        s.n_tilm_attempts = 4
        s.n_tilm_corrections = 3
        assert abs(s.tilm_acceptance_rate - 0.75) < 1e-9

    def test_mean_correction_computed(self) -> None:
        s = CorrectorStats()
        s.correction_errors = [1.0, 3.0]
        assert abs(s.mean_correction_m - 2.0) < 1e-9


# ── MatchingCorrector mode tests ──────────────────────────────────────────────

class TestMatchingCorrectorMode:
    def test_default_gnss_active(self) -> None:
        corrector, _ = _make_corrector()
        assert corrector.is_gnss_available
        assert not corrector.is_using_tilm

    def test_set_gnss_unavailable(self) -> None:
        corrector, _ = _make_corrector()
        corrector.set_gnss_available(False)
        assert not corrector.is_gnss_available
        assert corrector.is_using_tilm

    def test_set_gnss_back_active(self) -> None:
        corrector, _ = _make_corrector()
        corrector.set_gnss_available(False)
        corrector.set_gnss_available(True)
        assert corrector.is_gnss_available

    def test_ekf_gnss_flag_synced(self) -> None:
        corrector, ekf = _make_corrector()
        corrector.set_gnss_available(False)
        assert not ekf.state.is_gnss_active
        corrector.set_gnss_available(True)
        assert ekf.state.is_gnss_active


# ── GNSS update path ──────────────────────────────────────────────────────────

class TestGNSSUpdatePath:
    def test_gnss_update_shifts_position(self) -> None:
        corrector, ekf = _make_corrector(ekf_position=(0.0, 0.0, 0.0))
        z = np.array([10.0, 0.0, 0.0])
        corrector.update([], timestamp=1.0, gnss_position=z)
        assert ekf.state.position[0] > 0.0

    def test_gnss_update_increments_counter(self) -> None:
        corrector, _ = _make_corrector()
        corrector.update([], timestamp=1.0, gnss_position=np.zeros(3))
        assert corrector.stats.n_gnss_updates == 1

    def test_gnss_update_returns_none(self) -> None:
        corrector, _ = _make_corrector()
        result = corrector.update([], timestamp=1.0, gnss_position=np.zeros(3))
        assert result is None

    def test_gnss_no_position_no_update(self) -> None:
        corrector, _ = _make_corrector()
        corrector.update([], timestamp=1.0, gnss_position=None)
        assert corrector.stats.n_gnss_updates == 0

    def test_frame_counter_increments(self) -> None:
        corrector, _ = _make_corrector()
        for _ in range(5):
            corrector.update([], timestamp=0.0)
        assert corrector.stats.n_frames == 5


# ── TILM correction path ──────────────────────────────────────────────────────

class TestTILMCorrectionPath:
    def _obs(self) -> list[Landmark]:
        _, _, obs = _build_tilm_and_matcher()
        return obs

    def test_tilm_correction_applied(self) -> None:
        _, matcher, obs = _build_tilm_and_matcher(
            node_position=(10.0, 0.0, 0.0),
        )
        ekf = _make_ekf(position=(8.0, 0.0, 0.0))
        corrector = MatchingCorrector(matcher, ekf, min_confidence=0.0,
                                      correction_interval=0.0)
        corrector.set_gnss_available(False)
        p_before = ekf.state.position.copy()
        corrector.update(obs, timestamp=1.0)
        assert not np.allclose(ekf.state.position, p_before)

    def test_tilm_correction_toward_node(self) -> None:
        _, matcher, obs = _build_tilm_and_matcher(
            node_position=(10.0, 0.0, 0.0),
        )
        ekf = _make_ekf(position=(5.0, 0.0, 0.0))
        corrector = MatchingCorrector(matcher, ekf, min_confidence=0.0,
                                      correction_interval=0.0)
        corrector.set_gnss_available(False)
        corrector.update(obs, timestamp=1.0)
        # EKF position should move toward node at x=10
        assert ekf.state.position[0] > 5.0

    def test_tilm_increments_attempts(self) -> None:
        corrector, ekf = _make_corrector(min_confidence=0.0,
                                         correction_interval=0.0)
        corrector.set_gnss_available(False)
        _, _, obs = _build_tilm_and_matcher()
        corrector.update(obs, timestamp=1.0)
        assert corrector.stats.n_tilm_attempts == 1

    def test_tilm_increments_corrections(self) -> None:
        _, matcher, obs = _build_tilm_and_matcher()
        ekf = _make_ekf()
        corrector = MatchingCorrector(matcher, ekf, min_confidence=0.0,
                                      correction_interval=0.0)
        corrector.set_gnss_available(False)
        corrector.update(obs, timestamp=1.0)
        assert corrector.stats.n_tilm_corrections == 1

    def test_tilm_returns_match_result(self) -> None:
        _, matcher, obs = _build_tilm_and_matcher()
        ekf = _make_ekf()
        corrector = MatchingCorrector(matcher, ekf, min_confidence=0.0,
                                      correction_interval=0.0)
        corrector.set_gnss_available(False)
        result = corrector.update(obs, timestamp=1.0)
        assert result is not None

    def test_gnss_active_skips_tilm(self) -> None:
        _, matcher, obs = _build_tilm_and_matcher()
        ekf = _make_ekf()
        corrector = MatchingCorrector(matcher, ekf)
        # GNSS is still active
        corrector.update(obs, timestamp=1.0)
        assert corrector.stats.n_tilm_attempts == 0


# ── confidence gate ───────────────────────────────────────────────────────────

class TestConfidenceGate:
    def test_low_confidence_rejected(self) -> None:
        _, matcher, obs = _build_tilm_and_matcher()
        ekf = _make_ekf()
        corrector = MatchingCorrector(matcher, ekf, min_confidence=0.99,
                                      correction_interval=0.0)
        corrector.set_gnss_available(False)
        corrector.update(obs, timestamp=1.0)
        assert corrector.stats.n_tilm_corrections == 0
        assert corrector.stats.n_tilm_rejected == 1

    def test_high_confidence_accepted(self) -> None:
        _, matcher, obs = _build_tilm_and_matcher()
        ekf = _make_ekf()
        corrector = MatchingCorrector(matcher, ekf, min_confidence=0.0,
                                      correction_interval=0.0)
        corrector.set_gnss_available(False)
        corrector.update(obs, timestamp=1.0)
        assert corrector.stats.n_tilm_corrections == 1


# ── correction interval ───────────────────────────────────────────────────────

class TestCorrectionInterval:
    def test_too_soon_rejected(self) -> None:
        _, matcher, obs = _build_tilm_and_matcher()
        ekf = _make_ekf()
        corrector = MatchingCorrector(matcher, ekf, min_confidence=0.0,
                                      correction_interval=5.0)
        corrector.set_gnss_available(False)
        corrector.update(obs, timestamp=1.0)  # first: accepted
        corrector.update(obs, timestamp=2.0)  # too soon: rejected
        assert corrector.stats.n_tilm_corrections == 1
        assert corrector.stats.n_tilm_rejected == 1

    def test_after_interval_accepted(self) -> None:
        _, matcher, obs = _build_tilm_and_matcher()
        ekf = _make_ekf()
        corrector = MatchingCorrector(matcher, ekf, min_confidence=0.0,
                                      correction_interval=5.0)
        corrector.set_gnss_available(False)
        corrector.update(obs, timestamp=1.0)   # accepted
        corrector.update(obs, timestamp=6.1)   # after interval: accepted
        assert corrector.stats.n_tilm_corrections == 2

    def test_zero_interval_always_accepted(self) -> None:
        _, matcher, obs = _build_tilm_and_matcher()
        ekf = _make_ekf()
        corrector = MatchingCorrector(matcher, ekf, min_confidence=0.0,
                                      correction_interval=0.0)
        corrector.set_gnss_available(False)
        for t in [0.1, 0.2, 0.3]:
            corrector.update(obs, timestamp=t)
        assert corrector.stats.n_tilm_corrections == 3


# ── no observations ───────────────────────────────────────────────────────────

class TestNoObservations:
    def test_empty_obs_no_correction(self) -> None:
        _, matcher, _ = _build_tilm_and_matcher()
        ekf = _make_ekf()
        corrector = MatchingCorrector(matcher, ekf, min_confidence=0.0,
                                      correction_interval=0.0)
        corrector.set_gnss_available(False)
        corrector.update([], timestamp=1.0)
        assert corrector.stats.n_tilm_corrections == 0

    def test_last_match_stored(self) -> None:
        _, matcher, obs = _build_tilm_and_matcher()
        ekf = _make_ekf()
        corrector = MatchingCorrector(matcher, ekf, min_confidence=0.0,
                                      correction_interval=0.0)
        corrector.set_gnss_available(False)
        corrector.update(obs, timestamp=1.0)
        assert corrector.last_match is not None


# ── reset_stats ───────────────────────────────────────────────────────────────

class TestResetStats:
    def test_reset_clears_counters(self) -> None:
        _, matcher, obs = _build_tilm_and_matcher()
        ekf = _make_ekf()
        corrector = MatchingCorrector(matcher, ekf, min_confidence=0.0,
                                      correction_interval=0.0)
        corrector.set_gnss_available(False)
        corrector.update(obs, timestamp=1.0)
        corrector.reset_stats()
        assert corrector.stats.n_frames == 0
        assert corrector.stats.n_tilm_corrections == 0


# ── MatchingEvaluator tests ───────────────────────────────────────────────────

class TestMatchingEvaluator:
    def test_all_correct(self) -> None:
        ev = MatchingEvaluator(position_tolerance=5.0)
        gt = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        pred = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]])
        valid = np.array([True, True])
        m = ev.evaluate(gt, pred, valid)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.n_correct_matches == 2

    def test_all_wrong(self) -> None:
        ev = MatchingEvaluator(position_tolerance=1.0)
        gt = np.zeros((3, 3))
        pred = np.ones((3, 3)) * 100.0
        valid = np.ones(3, dtype=bool)
        m = ev.evaluate(gt, pred, valid)
        assert m.precision == 0.0
        assert m.n_correct_matches == 0

    def test_no_valid_matches(self) -> None:
        ev = MatchingEvaluator()
        gt = np.zeros((3, 3))
        pred = np.zeros((3, 3))
        valid = np.zeros(3, dtype=bool)
        m = ev.evaluate(gt, pred, valid)
        assert m.n_successful_matches == 0
        assert m.precision == 0.0

    def test_mixed(self) -> None:
        ev = MatchingEvaluator(position_tolerance=5.0)
        gt = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        pred = np.array([[0.1, 0.0, 0.0], [50.0, 0.0, 0.0], [20.1, 0.0, 0.0]])
        valid = np.array([True, True, True])
        m = ev.evaluate(gt, pred, valid)
        assert m.n_correct_matches == 2
        assert abs(m.precision - 2/3) < 1e-9
        assert abs(m.recall - 2/3) < 1e-9

    def test_partial_valid(self) -> None:
        ev = MatchingEvaluator(position_tolerance=5.0)
        gt = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        pred = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]])
        valid = np.array([True, False])  # only first is valid
        m = ev.evaluate(gt, pred, valid)
        assert m.n_successful_matches == 1
        assert m.n_correct_matches == 1
        assert m.precision == 1.0
        assert m.recall == 0.5  # 1 correct out of 2 total queries

    def test_latency_mean(self) -> None:
        ev = MatchingEvaluator(position_tolerance=5.0)
        gt = np.zeros((2, 3))
        pred = np.zeros((2, 3))
        valid = np.ones(2, dtype=bool)
        lat = np.array([2.0, 4.0])
        m = ev.evaluate(gt, pred, valid, latencies_ms=lat)
        assert abs(m.mean_latency_ms - 3.0) < 1e-9

    def test_position_error_rmse(self) -> None:
        ev = MatchingEvaluator(position_tolerance=100.0)
        gt = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        pred = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])  # errors: 5, 0
        valid = np.ones(2, dtype=bool)
        m = ev.evaluate(gt, pred, valid)
        # rmse = sqrt((25 + 0)/2) = sqrt(12.5)
        assert abs(m.position_error_rmse - np.sqrt(12.5)) < 1e-6
        assert m.position_error_max == 5.0

    def test_false_positive_rate(self) -> None:
        ev = MatchingEvaluator(position_tolerance=1.0)
        gt = np.zeros((4, 3))
        pred = np.ones((4, 3)) * 100.0  # all wrong
        valid = np.ones(4, dtype=bool)
        m = ev.evaluate(gt, pred, valid)
        # 4 matched, 0 correct → fpr = 4/4 = 1.0
        assert m.false_positive_rate == 1.0


class TestPrecisionRecallCurve:
    def test_returns_three_arrays(self) -> None:
        ev = MatchingEvaluator()
        scores = np.array([0.9, 0.7, 0.5, 0.3])
        correct = np.array([True, False, True, False])
        p, r, t = ev.precision_recall_curve(scores, correct)
        assert len(p) == len(r) == len(t) == 4

    def test_thresholds_sorted_descending(self) -> None:
        ev = MatchingEvaluator()
        scores = np.array([0.3, 0.9, 0.5])
        correct = np.array([True, True, False])
        _, _, t = ev.precision_recall_curve(scores, correct)
        assert list(t) == sorted(t, reverse=True)

    def test_all_correct_recall_reaches_one(self) -> None:
        ev = MatchingEvaluator()
        scores = np.array([0.8, 0.6, 0.4])
        correct = np.array([True, True, True])
        _, r, _ = ev.precision_recall_curve(scores, correct)
        assert abs(r[-1] - 1.0) < 1e-9

    def test_empty_input(self) -> None:
        ev = MatchingEvaluator()
        p, r, t = ev.precision_recall_curve(np.array([]), np.array([]))
        assert len(p) == 0

    def test_precision_first_entry_correct(self) -> None:
        ev = MatchingEvaluator()
        # Highest score is correct → first precision = 1.0
        scores = np.array([0.9, 0.5, 0.3])
        correct = np.array([True, False, False])
        p, _, _ = ev.precision_recall_curve(scores, correct)
        assert abs(p[0] - 1.0) < 1e-9
