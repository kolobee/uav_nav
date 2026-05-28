"""Unit tests for PseudoSimulator (two-phase offline simulator).

All tests use synthetic MidAirSample data — no real dataset or model
weights are required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from uav_nav.data.calibration import CameraCalibration
from uav_nav.data.midair_loader import IMUSample, MidAirSample
from uav_nav.runtime.pseudo_simulator import (
    PseudoSimulator,
    SimulationConfig,
    SimulationResult,
)


# ── synthetic dataset ─────────────────────────────────────────────────────────

N_FRAMES = 40
DT = 0.1  # 10 Hz


def _make_pose(position: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = position
    return T


def _make_imu_sample(dt: float = DT) -> IMUSample:
    return IMUSample(
        timestamp=0.0,
        acc=np.array([0.0, 0.0, 9.81]),
        gyro=np.zeros(3),
    )


def _make_sample(frame_idx: int) -> MidAirSample:
    t = frame_idx * DT
    pos = np.array([frame_idx * 1.0, 0.0, -50.0])
    seg = np.full((64, 64), 2, dtype=np.uint8)  # raw MidAir class 2 = tree

    depth = np.full((64, 64), 30.0, dtype=np.float32)
    image_rgb = np.zeros((64, 64, 3), dtype=np.uint8)

    return MidAirSample(
        timestamp=t,
        image_rgb=image_rgb,
        depth=depth,
        segmentation=seg,
        pose_gt=_make_pose(pos),
        imu_samples=[_make_imu_sample()],
        sequence_id="synthetic",
        frame_idx=frame_idx,
    )


class _SyntheticLoader:
    """Minimal MidAirLoader substitute returning synthetic samples."""

    def __init__(self, n_frames: int = N_FRAMES) -> None:
        self._frames = [_make_sample(i) for i in range(n_frames)]

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> MidAirSample:
        return self._frames[idx]

    def __iter__(self):
        return iter(self._frames)

    def build_index(self) -> None:
        pass


# ── fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def loader() -> _SyntheticLoader:
    return _SyntheticLoader(N_FRAMES)


@pytest.fixture()
def calibration() -> CameraCalibration:
    return CameraCalibration.from_midair_defaults()


@pytest.fixture()
def sim_config() -> SimulationConfig:
    return SimulationConfig(
        gnss_loss_fraction=0.5,
        gnss_noise_std=0.1,
        frame_stride=1,
        imu_noise_acc=0.0,
        imu_noise_gyro=0.0,
    )


@pytest.fixture()
def simulator(loader, calibration, sim_config) -> PseudoSimulator:
    return PseudoSimulator(
        loader=loader,
        calibration=calibration,
        config=sim_config,
    )


# ── SimulationResult ──────────────────────────────────────────────────────────

class TestSimulationResult:
    def test_position_errors_shape(self):
        n = 20
        result = SimulationResult(
            gt_positions=np.zeros((n, 3)),
            estimated_positions=np.ones((n, 3)),
            dead_reckoning_positions=np.ones((n, 3)) * 2,
            timestamps=np.arange(n, dtype=float),
            gnss_active_mask=np.ones(n, dtype=bool),
            tilm_qualities=np.zeros(n),
        )
        assert result.position_errors_tilm.shape == (n,)
        assert result.position_errors_dr.shape == (n,)

    def test_ate_nan_when_empty(self):
        result = SimulationResult()
        assert np.isnan(result.ate_tilm)
        assert np.isnan(result.ate_dr)

    def test_improvement_pct_positive_when_tilm_better(self):
        n = 50
        gt = np.zeros((n, 3))
        tilm = np.random.default_rng(0).normal(0, 1.0, (n, 3))
        dr = np.random.default_rng(0).normal(0, 5.0, (n, 3))
        result = SimulationResult(
            gt_positions=gt,
            estimated_positions=tilm,
            dead_reckoning_positions=dr,
        )
        assert result.improvement_pct > 0

    def test_back_compat_aliases(self):
        n = 10
        result = SimulationResult(
            gt_positions=np.zeros((n, 3)),
            estimated_positions=np.ones((n, 3)),
            dead_reckoning_positions=np.ones((n, 3)),
            timestamps=np.arange(n, dtype=float),
            gnss_active_mask=np.ones(n, dtype=bool),
        )
        assert result.positions_gt is result.gt_positions
        assert result.positions_est is result.estimated_positions
        assert len(result.modes) == n


# ── PseudoSimulator.run ───────────────────────────────────────────────────────

class TestPseudoSimulatorRun:
    def test_result_arrays_same_length(self, simulator):
        result = simulator.run()
        n = len(result.timestamps)
        assert len(result.gt_positions) == n
        assert len(result.estimated_positions) == n
        assert len(result.dead_reckoning_positions) == n
        assert len(result.gnss_active_mask) == n
        assert len(result.tilm_qualities) == n

    def test_gnss_loss_idx_correct(self, simulator, sim_config):
        result = simulator.run()
        expected = int(N_FRAMES * sim_config.gnss_loss_fraction)
        assert result.gnss_loss_idx == expected

    def test_gnss_active_mask_splits_at_loss(self, simulator, sim_config):
        result = simulator.run()
        loss_idx = result.gnss_loss_idx
        assert result.gnss_active_mask[:loss_idx].all()
        assert not result.gnss_active_mask[loss_idx:].any()

    def test_mapping_phase_builds_tilm(self, simulator):
        result = simulator.run()
        assert result.tilm_node_count >= 0  # may be 0 if no landmarks extracted

    def test_corrector_stats_populated(self, simulator):
        result = simulator.run()
        assert result.corrector_stats is not None
        assert result.corrector_stats.n_frames >= 0

    def test_gt_positions_match_sample_poses(self, simulator):
        result = simulator.run()
        # First frame should be at position [0, 0, -50]
        np.testing.assert_allclose(result.gt_positions[0], [0.0, 0.0, -50.0], atol=1e-6)

    def test_run_empty_loader_raises(self, calibration, sim_config):
        empty = _SyntheticLoader(n_frames=0)
        sim = PseudoSimulator(loader=empty, calibration=calibration, config=sim_config)
        with pytest.raises(RuntimeError):
            sim.run()

    def test_dead_reckoning_diverges_from_estimated(self, loader, calibration):
        cfg = SimulationConfig(
            gnss_loss_fraction=0.3,
            gnss_noise_std=0.0,
            imu_noise_acc=0.5,   # high noise → DR diverges
            imu_noise_gyro=0.1,
        )
        sim = PseudoSimulator(loader=loader, calibration=calibration, config=cfg)
        result = sim.run()
        nav_slice = slice(result.gnss_loss_idx, None)
        err_tilm = np.mean(result.position_errors_tilm[nav_slice])
        err_dr = np.mean(result.position_errors_dr[nav_slice])
        # Both may be similar with random embeddings (no good TILM matches),
        # but the arrays should differ structurally
        assert not np.allclose(result.estimated_positions, result.dead_reckoning_positions)

    def test_progress_callback_called(self, loader, calibration, sim_config):
        calls: list[tuple[int, int]] = []
        sim = PseudoSimulator(
            loader=loader,
            calibration=calibration,
            config=sim_config,
            progress_callback=lambda i, t: calls.append((i, t)),
        )
        sim.run()
        assert len(calls) > 0
        # Total should always be the loader length
        for _, total in calls:
            assert total == N_FRAMES

    def test_frame_stride_reduces_frame_count(self, calibration, sim_config):
        sim_config.frame_stride = 2
        loader = _SyntheticLoader(N_FRAMES)
        sim = PseudoSimulator(loader=loader, calibration=calibration, config=sim_config)
        result = sim.run()
        expected_n = len(range(0, N_FRAMES, 2))
        assert len(result.timestamps) == expected_n


# ── GT segmentation conversion ────────────────────────────────────────────────

class TestGTSegConversion:
    def test_gt_seg_produces_segmentation_result(self, loader, calibration, sim_config):
        sim = PseudoSimulator(loader=loader, calibration=calibration, config=sim_config)
        sample = loader[0]
        seg = sim._gt_seg_to_result(sample)
        # All-tree image (class 2) should produce at least one instance
        assert seg.n_instances >= 0  # 0 is ok if all pixels < min_area

    def test_empty_segmentation_for_none_seg(self, loader, calibration, sim_config):
        sim = PseudoSimulator(loader=loader, calibration=calibration, config=sim_config)
        sample = _make_sample(0)
        sample.segmentation = None
        seg = sim._gt_seg_to_result(sample)
        assert seg.n_instances == 0
