"""Unit tests for TrajectoryEvaluator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from uav_nav.eval.trajectory_eval import TrajectoryEvaluator, TrajectoryMetrics, _EVO_AVAILABLE


# ── helpers ───────────────────────────────────────────────────────────────────

def _straight_traj(n: int = 100, speed: float = 1.0) -> np.ndarray:
    """Straight-line trajectory along the North axis."""
    t = np.linspace(0.0, float(n - 1), n)
    pos = np.zeros((n, 3))
    pos[:, 0] = t * speed
    return pos


def _noisy_traj(gt: np.ndarray, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return gt + rng.normal(0.0, sigma, gt.shape)


def _drifting_traj(gt: np.ndarray, drift_rate: float = 0.05) -> np.ndarray:
    """Trajectory that drifts linearly away from GT."""
    n = len(gt)
    drift = np.zeros_like(gt)
    drift[:, 0] = np.arange(n) * drift_rate  # accumulating North drift
    return gt + drift


# ── TrajectoryMetrics ─────────────────────────────────────────────────────────

class TestTrajectoryMetrics:
    def test_to_dict_has_all_fields(self):
        m = TrajectoryMetrics(ate_rmse=1.0, ate_mean=0.8, n_frames=50, sequence_name="test")
        d = m.to_dict()
        assert "ate_rmse" in d
        assert "n_frames" in d
        assert d["sequence_name"] == "test"

    def test_improvement_over_baseline(self):
        baseline = TrajectoryMetrics(ate_rmse=10.0)
        improved = TrajectoryMetrics(ate_rmse=5.0)
        pct = improved.improvement_over(baseline)
        assert abs(pct - 50.0) < 1e-6

    def test_improvement_zero_baseline(self):
        baseline = TrajectoryMetrics(ate_rmse=0.0)
        improved = TrajectoryMetrics(ate_rmse=2.0)
        assert improved.improvement_over(baseline) == 0.0

    def test_improvement_negative_when_worse(self):
        baseline = TrajectoryMetrics(ate_rmse=5.0)
        worse = TrajectoryMetrics(ate_rmse=10.0)
        assert worse.improvement_over(baseline) < 0.0


# ── TrajectoryEvaluator.evaluate ──────────────────────────────────────────────

class TestEvaluate:
    def test_perfect_estimate_zero_ate(self):
        ev = TrajectoryEvaluator(align=False)
        gt = _straight_traj(50)
        m = ev.evaluate(gt, gt)
        assert m.ate_rmse < 1e-6
        assert m.n_frames == 50

    def test_noisy_estimate_positive_ate(self):
        ev = TrajectoryEvaluator(align=False)
        gt = _straight_traj(100)
        est = _noisy_traj(gt, sigma=2.0)
        m = ev.evaluate(est, gt)
        assert m.ate_rmse > 0.5

    def test_returns_correct_n_frames(self):
        ev = TrajectoryEvaluator()
        gt = _straight_traj(80)
        est = _noisy_traj(gt)
        m = ev.evaluate(est, gt, sequence_name="seq1")
        assert m.n_frames == 80
        assert m.sequence_name == "seq1"

    def test_ate_lower_with_alignment(self):
        # Offset trajectory → alignment should reduce ATE
        ev_align = TrajectoryEvaluator(align=True)
        ev_noa = TrajectoryEvaluator(align=False)
        gt = _straight_traj(100)
        est = gt + np.array([5.0, 3.0, 0.0])  # constant offset
        m_align = ev_align.evaluate(est, gt)
        m_noa = ev_noa.evaluate(est, gt)
        assert m_align.ate_rmse <= m_noa.ate_rmse + 1e-3

    def test_fallback_when_evo_missing(self, monkeypatch):
        import uav_nav.eval.trajectory_eval as module
        monkeypatch.setattr(module, "_EVO_AVAILABLE", False)
        ev = TrajectoryEvaluator(align=False)
        gt = _straight_traj(50)
        est = _noisy_traj(gt, sigma=2.0)
        m = ev.evaluate(est, gt)
        assert m.ate_rmse > 0

    def test_timestamps_accepted(self):
        ev = TrajectoryEvaluator()
        n = 60
        gt = _straight_traj(n)
        est = _noisy_traj(gt, sigma=1.0)
        ts = np.linspace(0.0, 6.0, n)
        m = ev.evaluate(est, gt, timestamps=ts, sequence_name="ts_test")
        assert m.n_frames == n

    def test_single_frame_returns_zero(self):
        ev = TrajectoryEvaluator()
        gt = np.array([[0.0, 0.0, 0.0]])
        m = ev.evaluate(gt, gt)
        assert m.ate_rmse == 0.0
        assert m.n_frames == 1


# ── TrajectoryEvaluator.compare ───────────────────────────────────────────────

class TestCompare:
    def test_compare_tilm_beats_dr(self):
        ev = TrajectoryEvaluator(align=False)
        gt = _straight_traj(100)
        tilm = _noisy_traj(gt, sigma=1.0, seed=0)
        dr = _drifting_traj(gt, drift_rate=0.3)  # large drift
        result = ev.compare(gt, with_tilm=tilm, without_tilm=dr)
        assert result["improvement_pct"] > 0
        assert result["tilm"].ate_rmse < result["dead_reckoning"].ate_rmse

    def test_compare_returns_both_metrics(self):
        ev = TrajectoryEvaluator()
        gt = _straight_traj(50)
        result = ev.compare(gt, with_tilm=gt, without_tilm=gt)
        assert "tilm" in result
        assert "dead_reckoning" in result
        assert "improvement_pct" in result


# ── TrajectoryEvaluator.evaluate_batch ────────────────────────────────────────

class TestEvaluateBatch:
    def test_batch_returns_one_per_sequence(self):
        ev = TrajectoryEvaluator(align=False)
        gt = _straight_traj(50)
        records = [
            {"positions_est": _noisy_traj(gt, sigma=1.0, seed=i),
             "positions_gt": gt,
             "sequence_name": f"seq{i}"}
            for i in range(3)
        ]
        metrics = ev.evaluate_batch(records)
        assert len(metrics) == 3
        for i, m in enumerate(metrics):
            assert m.sequence_name == f"seq{i}"

    def test_batch_empty_input(self):
        ev = TrajectoryEvaluator()
        assert ev.evaluate_batch([]) == []


# ── TrajectoryEvaluator.save_report ──────────────────────────────────────────

class TestSaveReport:
    def test_save_csv(self):
        ev = TrajectoryEvaluator(align=False)
        gt = _straight_traj(30)
        m = ev.evaluate(_noisy_traj(gt), gt, sequence_name="s1")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "metrics.csv"
            ev.save_report([m], p, format="csv")
            text = p.read_text()
            assert "ate_rmse" in text
            assert "s1" in text

    def test_save_json(self):
        ev = TrajectoryEvaluator(align=False)
        gt = _straight_traj(30)
        m = ev.evaluate(_noisy_traj(gt), gt, sequence_name="s2")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "metrics.json"
            ev.save_report([m], p, format="json")
            data = json.loads(p.read_text())
            assert len(data) == 1
            assert data[0]["sequence_name"] == "s2"

    def test_save_empty_list(self):
        ev = TrajectoryEvaluator()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "empty.csv"
            ev.save_report([], p)
            assert p.exists()

    def test_creates_parent_dirs(self):
        ev = TrajectoryEvaluator(align=False)
        gt = _straight_traj(20)
        m = ev.evaluate(_noisy_traj(gt), gt)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "deep" / "nested" / "metrics.csv"
            ev.save_report([m], p)
            assert p.exists()
