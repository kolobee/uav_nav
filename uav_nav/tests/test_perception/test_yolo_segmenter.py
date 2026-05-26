"""Unit tests for YOLOSegmenter and SegmentationResult.

All tests run without GPU or the MidAir dataset.
Heavy ultralytics calls are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from uav_nav.perception.yolo_segmenter import SegmentationResult, YOLOSegmenter, _empty_result


# ── SegmentationResult tests ─────────────────────────────────────────────────


class TestSegmentationResult:
    def _make(self, n: int = 3) -> SegmentationResult:
        H, W = 64, 64
        masks = np.zeros((n, H, W), dtype=bool)
        for i in range(n):
            masks[i, i * 10 : i * 10 + 10, i * 10 : i * 10 + 10] = True
        return SegmentationResult(
            masks=masks,
            class_ids=np.array([0, 1, 2], dtype=int)[:n],
            class_names=["tree", "rock", "river"][:n],
            scores=np.array([0.9, 0.7, 0.5], dtype=np.float32)[:n],
            boxes=np.zeros((n, 4), dtype=np.float32),
            image_hw=(H, W),
        )

    def test_n_instances(self):
        r = self._make(3)
        assert r.n_instances == 3

    def test_n_instances_zero(self):
        r = _empty_result((64, 64))
        assert r.n_instances == 0

    def test_filter_by_class_returns_matching(self):
        r = self._make(3)
        filtered = r.filter_by_class("rock")
        assert filtered.n_instances == 1
        assert filtered.class_names == ["rock"]
        assert filtered.class_ids[0] == 1

    def test_filter_by_class_missing_returns_empty(self):
        r = self._make(3)
        filtered = r.filter_by_class("nonexistent")
        assert filtered.n_instances == 0
        assert filtered.masks.shape == (0, 64, 64)

    def test_filter_preserves_mask_shape(self):
        r = self._make(3)
        filtered = r.filter_by_class("tree")
        assert filtered.masks.shape == (1, 64, 64)

    def test_merged_mask_zero_instances(self):
        r = _empty_result((32, 32))
        out = r.merged_mask()
        assert out.shape == (32, 32)
        assert out.dtype == np.uint8
        assert out.sum() == 0

    def test_merged_mask_higher_score_wins(self):
        H, W = 16, 16
        masks = np.zeros((2, H, W), dtype=bool)
        masks[0, 0:8, 0:8] = True   # score 0.5, class_id=0
        masks[1, 4:12, 4:12] = True  # score 0.9, class_id=1 (overlaps corner)
        r = SegmentationResult(
            masks=masks,
            class_ids=np.array([0, 1]),
            class_names=["tree", "rock"],
            scores=np.array([0.5, 0.9], dtype=np.float32),
            boxes=np.zeros((2, 4), dtype=np.float32),
            image_hw=(H, W),
        )
        out = r.merged_mask()
        # Overlapping pixels should have class_id=1 (higher score wins)
        assert out[6, 6] == 1
        # Non-overlapping pixels of mask0 should have class_id=0
        assert out[1, 1] == 0
        # Background
        assert out[15, 15] == 0

    def test_merged_mask_no_overlap(self):
        H, W = 32, 32
        masks = np.zeros((2, H, W), dtype=bool)
        masks[0, 0:10, 0:10] = True
        masks[1, 20:30, 20:30] = True
        r = SegmentationResult(
            masks=masks,
            class_ids=np.array([1, 2]),
            class_names=["rock", "river"],
            scores=np.array([0.8, 0.6], dtype=np.float32),
            boxes=np.zeros((2, 4), dtype=np.float32),
            image_hw=(H, W),
        )
        out = r.merged_mask()
        assert out[5, 5] == 1
        assert out[25, 25] == 2
        assert out[15, 15] == 0


# ── YOLOSegmenter unit tests (mocked) ────────────────────────────────────────


class TestYOLOSegmenterLoad:
    def test_load_raises_file_not_found(self, tmp_path):
        seg = YOLOSegmenter(
            weights_path=tmp_path / "nonexistent.pt",
            class_names=["tree"],
        )
        with pytest.raises(FileNotFoundError):
            seg.load()

    def test_load_unsupported_extension(self, tmp_path):
        bad_path = tmp_path / "model.xyz"
        bad_path.write_bytes(b"")
        seg = YOLOSegmenter(weights_path=bad_path, class_names=["tree"])
        with pytest.raises(ValueError, match="Unsupported weight format"):
            seg.load()

    def test_check_loaded_raises_before_load(self, tmp_path):
        seg = YOLOSegmenter(weights_path=tmp_path / "x.pt", class_names=["tree"])
        with pytest.raises(RuntimeError, match="load\\(\\)"):
            seg._check_loaded()

    @patch("uav_nav.perception.yolo_segmenter.YOLOSegmenter._load_ultralytics")
    def test_load_calls_ultralytics_for_pt(self, mock_load, tmp_path):
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"")
        seg = YOLOSegmenter(weights_path=pt, class_names=["tree"])
        seg.load()
        mock_load.assert_called_once()

    @patch("uav_nav.perception.yolo_segmenter.YOLOSegmenter._load_onnx")
    def test_load_calls_onnx_for_onnx(self, mock_load, tmp_path):
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"")
        seg = YOLOSegmenter(weights_path=onnx_path, class_names=["tree"])
        seg.load()
        mock_load.assert_called_once()


class TestYOLOSegmenterPredict:
    def _make_mock_result(self, H: int = 64, W: int = 64, n: int = 2) -> MagicMock:
        """Build a mock ultralytics Results object."""
        import torch

        r = MagicMock()
        r.masks = MagicMock()
        # masks.data: (N, H_m, W_m) float32
        r.masks.data = torch.ones(n, H // 4, W // 4)
        r.boxes = MagicMock()
        r.boxes.__len__ = MagicMock(return_value=n)
        r.boxes.cls = torch.tensor([0.0, 1.0])[:n]
        r.boxes.conf = torch.tensor([0.9, 0.7])[:n]
        r.boxes.xyxy = torch.zeros(n, 4)
        return r

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_predict_returns_segmentation_result(self, tmp_path):
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"")

        seg = YOLOSegmenter(
            weights_path=pt,
            class_names=["tree", "rock", "river"],
            imgsz=64,
        )
        seg._backend = "ultralytics"

        mock_results = [self._make_mock_result(64, 64, 2)]
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_results
        seg._model = mock_model

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        result = seg.predict(image)

        assert isinstance(result, SegmentationResult)
        assert result.image_hw == (64, 64)
        assert result.n_instances == 2
        assert result.masks.shape == (2, 64, 64)
        assert result.masks.dtype == bool

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_predict_empty_when_no_masks(self, tmp_path):
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"")

        seg = YOLOSegmenter(weights_path=pt, class_names=["tree"])
        seg._backend = "ultralytics"

        r = MagicMock()
        r.masks = None
        r.boxes = MagicMock()
        r.boxes.__len__ = MagicMock(return_value=0)

        seg._model = MagicMock()
        seg._model.predict.return_value = [r]

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        result = seg.predict(image)
        assert result.n_instances == 0

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_predict_batch_returns_list(self, tmp_path):
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"")

        seg = YOLOSegmenter(weights_path=pt, class_names=["tree", "rock"])
        seg._backend = "ultralytics"

        images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
        seg._model = MagicMock()
        seg._model.predict.return_value = [
            self._make_mock_result(64, 64, 1) for _ in range(3)
        ]

        results = seg.predict_batch(images)
        assert len(results) == 3
        assert all(isinstance(r, SegmentationResult) for r in results)

    def test_predict_batch_empty_list(self, tmp_path):
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"")

        seg = YOLOSegmenter(weights_path=pt, class_names=["tree"])
        seg._backend = "ultralytics"
        seg._model = MagicMock()

        assert seg.predict_batch([]) == []


# ── NMS helper tests ──────────────────────────────────────────────────────────


class TestNMS:
    def test_nms_no_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = YOLOSegmenter._nms(boxes, scores, 0.5)
        assert sorted(keep) == [0, 1]

    def test_nms_full_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = YOLOSegmenter._nms(boxes, scores, 0.5)
        assert len(keep) == 1
        assert keep[0] == 0  # higher score kept

    def test_nms_empty(self):
        keep = YOLOSegmenter._nms(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            0.5,
        )
        assert keep == []

    def test_xywh2xyxy(self):
        boxes = np.array([[10, 10, 4, 4]], dtype=np.float32)
        out = YOLOSegmenter._xywh2xyxy(boxes)
        np.testing.assert_allclose(out, [[8, 8, 12, 12]])


# ── Letterbox helper ──────────────────────────────────────────────────────────


class TestLetterbox:
    def test_output_shape(self, tmp_path):
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"")
        seg = YOLOSegmenter(weights_path=pt, class_names=["tree"], imgsz=64)
        image = np.zeros((128, 64, 3), dtype=np.uint8)
        blob, ratio, (pad_w, pad_h) = seg._letterbox(image, 64)
        assert blob.shape == (1, 3, 64, 64)
        assert blob.dtype == np.float32
        assert 0.0 <= blob.min() and blob.max() <= 1.0

    def test_square_image_no_padding(self, tmp_path):
        pt = tmp_path / "model.pt"
        pt.write_bytes(b"")
        seg = YOLOSegmenter(weights_path=pt, class_names=["tree"], imgsz=64)
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        _, ratio, (pad_w, pad_h) = seg._letterbox(image, 64)
        assert ratio == 1.0
