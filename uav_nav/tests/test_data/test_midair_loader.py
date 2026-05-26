"""Tests for MidAirLoader and DatasetBuilder.

Tests that do NOT require the real dataset use synthetic zip archives
built in-memory via zipfile + h5py.

Tests marked with ``pytest.mark.requires_midair`` are skipped automatically
unless the environment variable ``MIDAIR_ROOT`` points to a valid dataset.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
import yaml

from uav_nav.data.calibration import CameraCalibration
from uav_nav.data.dataset_builder import DatasetBuilder, SplitConfig
from uav_nav.data.midair_loader import (
    MidAirLoader,
    MidAirSample,
    _build_se3,
    _decode_jpeg,
)

# ---------------------------------------------------------------------------
# Helpers to build a synthetic MidAir-like directory in a tmp folder
# ---------------------------------------------------------------------------

_N_FRAMES = 8       # camera frames per synthetic trajectory
_N_GT = 32          # groundtruth samples (4× camera)
_IMG_H = _IMG_W = 64  # small images for speed


def _make_jpeg_bytes(h: int = _IMG_H, w: int = _IMG_W) -> bytes:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _make_png_bytes(h: int = _IMG_H, w: int = _IMG_W, dtype=np.uint8) -> bytes:
    rng = np.random.default_rng(1)
    if dtype == np.uint8:
        img = rng.integers(0, 4, (h, w), dtype=np.uint8)
    else:
        img = rng.integers(0, 5000, (h, w), dtype=np.uint16)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _make_frames_zip(n_frames: int, ext: str = ".JPEG") -> bytes:
    """Return bytes of a zip archive containing n_frames images."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(n_frames):
            fname = f"{i:06d}{ext}"
            if ext.upper() in (".JPEG", ".JPG"):
                zf.writestr(fname, _make_jpeg_bytes())
            else:
                zf.writestr(fname, _make_png_bytes())
    return buf.getvalue()


def _make_sensor_records_zip(traj_names: list[str], n_frames: int, n_gt: int) -> bytes:
    """Return bytes of a sensor_records.zip with a synthetic HDF5."""
    rng = np.random.default_rng(42)
    hdf_buf = io.BytesIO()
    with h5py.File(hdf_buf, "w") as f:
        for traj in traj_names:
            grp = f.create_group(traj)
            cam = grp.create_group("camera_data")
            # Store dummy filenames (shape (n_frames,) object)
            dt = h5py.special_dtype(vlen=str)
            for cam_key in ("color_down", "color_left", "depth", "segmentation"):
                ds = cam.create_dataset(cam_key, shape=(n_frames,), dtype=dt)
                for i in range(n_frames):
                    ds[i] = f"{i:06d}.JPEG"
            gt = grp.create_group("groundtruth")
            gt.create_dataset("position", data=rng.standard_normal((n_gt, 3)))
            # quaternions — normalise so they're valid rotations
            q = rng.standard_normal((n_gt, 4))
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            gt.create_dataset("attitude", data=q)
            gt.create_dataset("angular_velocity", data=rng.standard_normal((n_gt, 3)))
            gt.create_dataset("acceleration", data=rng.standard_normal((n_gt, 3)))
            imu = grp.create_group("imu")
            imu.create_dataset("accelerometer", data=rng.standard_normal((n_gt, 3)))
            imu.create_dataset("gyroscope", data=rng.standard_normal((n_gt, 3)))

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("sensor_records.hdf5", hdf_buf.getvalue())
    return zip_buf.getvalue()


def _build_synthetic_root(
    tmp_path: Path,
    env: str = "Kite_training",
    cond: str = "cloudy",
    traj_names: list[str] | None = None,
    n_frames: int = _N_FRAMES,
    n_gt: int = _N_GT,
) -> Path:
    """Write a minimal MidAir directory tree under tmp_path."""
    if traj_names is None:
        traj_names = ["trajectory_3000", "trajectory_3001"]

    cond_path = tmp_path / env / cond
    cond_path.mkdir(parents=True)

    # sensor_records.zip
    (cond_path / "sensor_records.zip").write_bytes(
        _make_sensor_records_zip(traj_names, n_frames, n_gt)
    )

    # color_down / depth / segmentation frames
    for folder in ("color_down", "depth", "segmentation"):
        for traj in traj_names:
            tdir = cond_path / folder / traj
            tdir.mkdir(parents=True)
            (tdir / "frames.zip").write_bytes(
                _make_frames_zip(n_frames, ext=".JPEG" if folder != "segmentation" else ".png")
            )

    return tmp_path


# ---------------------------------------------------------------------------
# Tests: MidAirLoader — no real dataset required
# ---------------------------------------------------------------------------


class TestMidAirLoaderSynthetic:
    def test_discover_sequences(self, tmp_path):
        root = _build_synthetic_root(tmp_path)
        loader = MidAirLoader(root=root)
        seqs = loader._discover_sequences()
        assert seqs == ["Kite_training/cloudy"]

    def test_build_index_sample_count(self, tmp_path):
        n_frames = 8
        n_trajs = 2
        root = _build_synthetic_root(tmp_path, n_frames=n_frames, n_gt=n_frames * 4)
        loader = MidAirLoader(root=root, load_depth=False, load_segmentation=False)
        loader.build_index()
        assert len(loader) == n_frames * n_trajs

    def test_build_index_sequence_ids(self, tmp_path):
        root = _build_synthetic_root(tmp_path)
        loader = MidAirLoader(root=root, load_depth=False, load_segmentation=False)
        loader.build_index()
        sid = loader._refs[0].sequence_id
        assert sid.startswith("Kite_training/cloudy/trajectory_")

    def test_getitem_returns_midair_sample(self, tmp_path):
        root = _build_synthetic_root(tmp_path, n_frames=4, n_gt=16)
        loader = MidAirLoader(root=root, load_depth=False, load_segmentation=False)
        loader.build_index()
        sample = loader[0]
        assert isinstance(sample, MidAirSample)

    def test_getitem_image_shape(self, tmp_path):
        root = _build_synthetic_root(tmp_path, n_frames=4, n_gt=16)
        loader = MidAirLoader(root=root, load_depth=False, load_segmentation=False)
        loader.build_index()
        sample = loader[0]
        assert sample.image_rgb.shape == (_IMG_H, _IMG_W, 3)
        assert sample.image_rgb.dtype == np.uint8

    def test_getitem_pose_shape(self, tmp_path):
        root = _build_synthetic_root(tmp_path, n_frames=4, n_gt=16)
        loader = MidAirLoader(root=root, load_depth=False, load_segmentation=False)
        loader.build_index()
        sample = loader[0]
        assert sample.pose_gt is not None
        assert sample.pose_gt.shape == (4, 4)

    def test_iter_yields_all_samples(self, tmp_path):
        root = _build_synthetic_root(tmp_path, n_frames=4, n_gt=16)
        loader = MidAirLoader(root=root, load_depth=False, load_segmentation=False)
        loader.build_index()
        samples = list(loader)
        assert len(samples) == 8  # 4 frames × 2 trajectories

    def test_max_frames_limit(self, tmp_path):
        root = _build_synthetic_root(tmp_path, n_frames=8, n_gt=32)
        loader = MidAirLoader(
            root=root, max_frames=3, load_depth=False, load_segmentation=False
        )
        loader.build_index()
        assert len(loader) == 6  # 3 frames × 2 trajectories

    def test_timestamps_monotone_within_trajectory(self, tmp_path):
        root = _build_synthetic_root(tmp_path, n_frames=4, n_gt=16,
                                     traj_names=["trajectory_3000"])
        loader = MidAirLoader(root=root, load_depth=False, load_segmentation=False)
        loader.build_index()
        ts = [r.timestamp for r in loader._refs]
        assert all(ts[i] < ts[i + 1] for i in range(len(ts) - 1))

    def test_imu_slice_not_empty_after_first_frame(self, tmp_path):
        root = _build_synthetic_root(tmp_path, n_frames=4, n_gt=16,
                                     traj_names=["trajectory_3000"])
        loader = MidAirLoader(root=root, load_depth=False, load_segmentation=False)
        loader.build_index()
        # Frame 0 has no prior frame → empty slice; frame 1+ should have IMU
        assert len(loader._refs[1].imu_slice) > 0

    def test_build_index_without_sequences_argument(self, tmp_path):
        root = _build_synthetic_root(tmp_path, n_frames=2, n_gt=8)
        loader = MidAirLoader(root=root, load_depth=False, load_segmentation=False)
        loader.build_index()
        assert len(loader) > 0

    def test_raises_without_build_index(self, tmp_path):
        root = _build_synthetic_root(tmp_path)
        loader = MidAirLoader(root=root)
        with pytest.raises(RuntimeError):
            _ = loader[0]

    def test_raises_on_bad_sequence_id(self, tmp_path):
        root = _build_synthetic_root(tmp_path)
        loader = MidAirLoader(root=root, sequences=["bad_format"])
        with pytest.raises(ValueError):
            loader.build_index()


class TestBuildSe3:
    def test_identity_quaternion_gives_identity_rotation(self):
        pos = np.zeros(3)
        q = np.array([1.0, 0.0, 0.0, 0.0])
        T = _build_se3(pos, q)
        assert np.allclose(T[:3, :3], np.eye(3))
        assert np.allclose(T[:3, 3], np.zeros(3))
        assert T[3, 3] == 1.0

    def test_translation_is_stored_correctly(self):
        pos = np.array([1.0, 2.0, 3.0])
        q = np.array([1.0, 0.0, 0.0, 0.0])
        T = _build_se3(pos, q)
        assert np.allclose(T[:3, 3], pos)

    def test_bottom_row(self):
        T = _build_se3(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        assert np.allclose(T[3], [0, 0, 0, 1])


# ---------------------------------------------------------------------------
# Tests: DatasetBuilder — no real dataset required
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_loader(tmp_path) -> MidAirLoader:
    root = _build_synthetic_root(
        tmp_path, n_frames=6, n_gt=24,
        traj_names=["trajectory_3000", "trajectory_3001"],
    )
    loader = MidAirLoader(root=root, load_depth=False, load_segmentation=False)
    loader.build_index()
    return loader


@pytest.fixture()
def label_map() -> dict[int, str]:
    return {
        0: "background", 1: "tree", 2: "tree", 3: "rock",
        4: "rock", 5: "river", 6: "background",
    }


class TestDatasetBuilderSynthetic:
    def test_build_split_creates_file(self, tmp_path, synthetic_loader, label_map):
        calib = CameraCalibration.from_midair_defaults()
        builder = DatasetBuilder(
            loader=synthetic_loader,
            calibration=calib,
            output_dir=tmp_path / "out",
            label_map=label_map,
            target_size=(64, 64),
        )
        cfg = SplitConfig(name="train", sequences=["Kite_training/cloudy"])
        out = builder.build_split(cfg)
        assert out.exists()

    def test_build_split_hdf5_keys(self, tmp_path, synthetic_loader, label_map):
        calib = CameraCalibration.from_midair_defaults()
        builder = DatasetBuilder(
            loader=synthetic_loader,
            calibration=calib,
            output_dir=tmp_path / "out",
            label_map=label_map,
            target_size=(64, 64),
        )
        cfg = SplitConfig(name="train", sequences=["Kite_training/cloudy"])
        out = builder.build_split(cfg)
        with h5py.File(out, "r") as f:
            for key in ("images", "segmentation", "depth", "poses", "timestamps",
                        "frame_indices", "sequence_ids"):
                assert key in f, f"missing dataset: {key}"

    def test_build_split_image_shape(self, tmp_path, synthetic_loader, label_map):
        calib = CameraCalibration.from_midair_defaults()
        builder = DatasetBuilder(
            loader=synthetic_loader,
            calibration=calib,
            output_dir=tmp_path / "out",
            label_map=label_map,
            target_size=(32, 48),
        )
        cfg = SplitConfig(name="train", sequences=["Kite_training/cloudy"],
                          shuffle=False)
        out = builder.build_split(cfg)
        with h5py.File(out, "r") as f:
            assert f["images"].shape[1:] == (32, 48, 3)

    def test_build_split_pose_shape(self, tmp_path, synthetic_loader, label_map):
        calib = CameraCalibration.from_midair_defaults()
        builder = DatasetBuilder(
            loader=synthetic_loader,
            calibration=calib,
            output_dir=tmp_path / "out",
            label_map=label_map,
            target_size=(64, 64),
        )
        cfg = SplitConfig(name="train", sequences=["Kite_training/cloudy"],
                          shuffle=False)
        out = builder.build_split(cfg)
        with h5py.File(out, "r") as f:
            assert f["poses"].shape[1:] == (4, 4)

    def test_remap_labels(self, tmp_path, synthetic_loader, label_map):
        calib = CameraCalibration.from_midair_defaults()
        builder = DatasetBuilder(
            loader=synthetic_loader,
            calibration=calib,
            output_dir=tmp_path / "out",
            label_map=label_map,
            target_size=(64, 64),
        )
        mask = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
        remapped = builder._remap_labels(mask)
        assert remapped.shape == mask.shape
        assert remapped.dtype == np.uint8
        # All values must be in [0, n_classes)
        n_classes = len(set(label_map.values()))
        assert int(remapped.max()) < n_classes

    def test_undistort_resizes_correctly(self, tmp_path, synthetic_loader, label_map):
        calib = CameraCalibration.from_midair_defaults()
        builder = DatasetBuilder(
            loader=synthetic_loader,
            calibration=calib,
            output_dir=tmp_path / "out",
            label_map=label_map,
            target_size=(48, 64),
        )
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        out = builder._undistort_image(img)
        assert out.shape == (48, 64, 3)

    def test_build_all_creates_three_splits(self, tmp_path, synthetic_loader, label_map):
        calib = CameraCalibration.from_midair_defaults()
        builder = DatasetBuilder(
            loader=synthetic_loader,
            calibration=calib,
            output_dir=tmp_path / "out",
            label_map=label_map,
            target_size=(32, 32),
        )
        paths = builder.build_all(
            train_sequences=["Kite_training/cloudy"],
            val_sequences=["Kite_training/cloudy"],
            test_sequences=["Kite_training/cloudy"],
        )
        assert set(paths.keys()) == {"train", "val", "test"}
        for p in paths.values():
            assert p.exists()

    def test_compute_statistics_returns_six_floats(self, tmp_path, synthetic_loader,
                                                    label_map):
        calib = CameraCalibration.from_midair_defaults()
        builder = DatasetBuilder(
            loader=synthetic_loader,
            calibration=calib,
            output_dir=tmp_path / "out",
            label_map=label_map,
            target_size=(16, 16),
        )
        cfg = SplitConfig(name="train", sequences=["Kite_training/cloudy"],
                          shuffle=False)
        out = builder.build_split(cfg)
        stats = builder.compute_statistics(out)
        expected_keys = {"mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"}
        assert set(stats.keys()) == expected_keys
        for v in stats.values():
            assert np.isfinite(v)


# ---------------------------------------------------------------------------
# Tests: YOLO exporter — synthetic data
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_loader_with_seg(tmp_path) -> MidAirLoader:
    """Loader with segmentation enabled for YOLO export tests."""
    root = _build_synthetic_root(
        tmp_path, n_frames=4, n_gt=16,
        traj_names=["trajectory_3000"],
    )
    loader = MidAirLoader(root=root, load_depth=False, load_segmentation=True)
    loader.build_index()
    return loader


class TestYoloExporter:
    def test_export_creates_images_and_labels(self, tmp_path,
                                               synthetic_loader_with_seg, label_map):
        calib = CameraCalibration.from_midair_defaults()
        builder = DatasetBuilder(
            loader=synthetic_loader_with_seg,
            calibration=calib,
            output_dir=tmp_path / "hdf",
            label_map=label_map,
            target_size=(64, 64),
        )
        cfg = SplitConfig(name="train", sequences=["Kite_training/cloudy"],
                          shuffle=False)
        yolo_root = builder.export_yolo_split(cfg, tmp_path / "yolo")

        imgs = list((yolo_root / "images" / "train").glob("*.jpg"))
        lbls = list((yolo_root / "labels" / "train").glob("*.txt"))
        assert len(imgs) > 0
        assert len(lbls) == len(imgs)

    def test_image_and_label_names_match(self, tmp_path,
                                          synthetic_loader_with_seg, label_map):
        calib = CameraCalibration.from_midair_defaults()
        builder = DatasetBuilder(
            loader=synthetic_loader_with_seg,
            calibration=calib,
            output_dir=tmp_path / "hdf",
            label_map=label_map,
            target_size=(64, 64),
        )
        cfg = SplitConfig(name="train", sequences=["Kite_training/cloudy"],
                          shuffle=False)
        yolo_root = builder.export_yolo_split(cfg, tmp_path / "yolo")

        img_stems = {p.stem for p in (yolo_root / "images" / "train").glob("*.jpg")}
        lbl_stems = {p.stem for p in (yolo_root / "labels" / "train").glob("*.txt")}
        assert img_stems == lbl_stems

    def test_exported_images_are_valid_jpeg(self, tmp_path,
                                             synthetic_loader_with_seg, label_map):
        calib = CameraCalibration.from_midair_defaults()
        builder = DatasetBuilder(
            loader=synthetic_loader_with_seg,
            calibration=calib,
            output_dir=tmp_path / "hdf",
            label_map=label_map,
            target_size=(64, 64),
        )
        cfg = SplitConfig(name="train", sequences=["Kite_training/cloudy"],
                          shuffle=False)
        yolo_root = builder.export_yolo_split(cfg, tmp_path / "yolo")
        for p in (yolo_root / "images" / "train").glob("*.jpg"):
            img = cv2.imread(str(p))
            assert img is not None
            assert img.shape == (64, 64, 3)
            break

    def test_label_coords_in_unit_range(self, tmp_path,
                                         synthetic_loader_with_seg, label_map):
        calib = CameraCalibration.from_midair_defaults()
        builder = DatasetBuilder(
            loader=synthetic_loader_with_seg,
            calibration=calib,
            output_dir=tmp_path / "hdf",
            label_map=label_map,
            target_size=(64, 64),
        )
        cfg = SplitConfig(name="train", sequences=["Kite_training/cloudy"],
                          shuffle=False)
        yolo_root = builder.export_yolo_split(cfg, tmp_path / "yolo")
        for lbl_path in (yolo_root / "labels" / "train").glob("*.txt"):
            text = lbl_path.read_text().strip()
            if not text:
                continue
            for line in text.splitlines():
                parts = line.split()
                coords = list(map(float, parts[1:]))
                assert all(0.0 <= c <= 1.0 for c in coords), \
                    f"coord out of range in {lbl_path}: {line}"

    def test_label_class_ids_are_valid(self, tmp_path,
                                        synthetic_loader_with_seg, label_map):
        calib = CameraCalibration.from_midair_defaults()
        n_fg = len(set(label_map.values())) - 1  # subtract background
        builder = DatasetBuilder(
            loader=synthetic_loader_with_seg,
            calibration=calib,
            output_dir=tmp_path / "hdf",
            label_map=label_map,
            target_size=(64, 64),
        )
        cfg = SplitConfig(name="train", sequences=["Kite_training/cloudy"],
                          shuffle=False)
        yolo_root = builder.export_yolo_split(cfg, tmp_path / "yolo")
        for lbl_path in (yolo_root / "labels" / "train").glob("*.txt"):
            for line in lbl_path.read_text().splitlines():
                if not line.strip():
                    continue
                class_id = int(line.split()[0])
                assert 0 <= class_id < n_fg

    def test_write_yolo_yaml(self, tmp_path, synthetic_loader_with_seg, label_map):
        calib = CameraCalibration.from_midair_defaults()
        builder = DatasetBuilder(
            loader=synthetic_loader_with_seg,
            calibration=calib,
            output_dir=tmp_path / "hdf",
            label_map=label_map,
            target_size=(64, 64),
        )
        yolo_root = tmp_path / "yolo"
        yolo_root.mkdir()
        builder._write_yolo_yaml(yolo_root)
        yaml_path = yolo_root / "midair.yaml"
        assert yaml_path.exists()
        with open(yaml_path) as f:
            cfg_yaml = yaml.safe_load(f)
        assert "nc" in cfg_yaml
        assert "names" in cfg_yaml
        assert cfg_yaml["nc"] == len(cfg_yaml["names"])
        n_fg = len(set(label_map.values())) - 1
        assert cfg_yaml["nc"] == n_fg

    def test_export_yolo_all_creates_yaml(self, tmp_path,
                                           synthetic_loader_with_seg, label_map):
        calib = CameraCalibration.from_midair_defaults()
        builder = DatasetBuilder(
            loader=synthetic_loader_with_seg,
            calibration=calib,
            output_dir=tmp_path / "hdf",
            label_map=label_map,
            target_size=(32, 32),
        )
        yolo_root = builder.export_yolo_all(
            train_sequences=["Kite_training/cloudy"],
            val_sequences=["Kite_training/cloudy"],
            test_sequences=["Kite_training/cloudy"],
            yolo_root=tmp_path / "yolo",
        )
        assert (yolo_root / "midair.yaml").exists()
        assert (yolo_root / "images" / "train").is_dir()
        assert (yolo_root / "labels" / "val").is_dir()

    def test_raises_if_segmentation_disabled(self, tmp_path, label_map):
        root = _build_synthetic_root(tmp_path, n_frames=4, n_gt=16,
                                     traj_names=["trajectory_3000"])
        loader_no_seg = MidAirLoader(root=root, load_segmentation=False)
        loader_no_seg.build_index()
        calib = CameraCalibration.from_midair_defaults()
        builder = DatasetBuilder(
            loader=loader_no_seg,
            calibration=calib,
            output_dir=tmp_path / "hdf",
            label_map=label_map,
        )
        cfg = SplitConfig(name="train", sequences=["Kite_training/cloudy"])
        with pytest.raises(RuntimeError, match="load_segmentation=True"):
            builder.export_yolo_split(cfg, tmp_path / "yolo")

    def test_extract_yolo_labels_all_fg_mask(self, tmp_path, label_map):
        """Dense foreground mask should produce at least one label."""
        calib = CameraCalibration.from_midair_defaults()
        root = _build_synthetic_root(tmp_path, n_frames=2, n_gt=8,
                                     traj_names=["trajectory_3000"])
        loader = MidAirLoader(root=root, load_segmentation=False)
        loader.build_index()
        builder = DatasetBuilder(
            loader=loader,
            calibration=calib,
            output_dir=tmp_path / "hdf",
            label_map=label_map,
            target_size=(64, 64),
        )
        # Create a mask with a large foreground blob (class 1 = rock in alphabetical order)
        seg = np.ones((64, 64), dtype=np.uint8)  # compact_id=1 everywhere
        labels = builder._extract_yolo_labels(seg, 64, 64, min_area=10, epsilon_frac=0.01)
        assert len(labels) >= 1
        # Each label must have class_id + at least 3 coordinate pairs
        for line in labels:
            parts = line.split()
            assert len(parts) >= 1 + 3 * 2  # class + 3 (x,y) pairs


# ---------------------------------------------------------------------------
# Tests: real MidAir dataset (skipped unless MIDAIR_ROOT is set)
# ---------------------------------------------------------------------------

_MIDAIR_ROOT = os.environ.get("MIDAIR_ROOT", "")


@pytest.mark.requires_midair
@pytest.mark.skipif(not _MIDAIR_ROOT, reason="MIDAIR_ROOT not set")
class TestMidAirLoaderReal:
    def test_discover_sequences_not_empty(self):
        loader = MidAirLoader(
            root=Path(_MIDAIR_ROOT),
            load_depth=False,
            load_segmentation=False,
        )
        seqs = loader._discover_sequences()
        assert len(seqs) > 0

    def test_build_index_first_trajectory(self):
        loader = MidAirLoader(
            root=Path(_MIDAIR_ROOT),
            sequences=["Kite_training/cloudy"],
            max_frames=5,
            load_depth=False,
            load_segmentation=False,
        )
        loader.build_index()
        assert len(loader) > 0

    def test_first_sample_image_size(self):
        loader = MidAirLoader(
            root=Path(_MIDAIR_ROOT),
            sequences=["Kite_training/cloudy"],
            max_frames=3,
            load_depth=False,
            load_segmentation=False,
        )
        loader.build_index()
        sample = loader[0]
        # MidAir images are 1024×1024 (90° FoV, colour_down camera)
        assert sample.image_rgb.shape == (1024, 1024, 3)

    def test_pose_is_valid_se3(self):
        loader = MidAirLoader(
            root=Path(_MIDAIR_ROOT),
            sequences=["Kite_training/cloudy"],
            max_frames=3,
            load_depth=False,
            load_segmentation=False,
        )
        loader.build_index()
        T = loader[0].pose_gt
        assert T is not None
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.allclose(T[3], [0, 0, 0, 1])
