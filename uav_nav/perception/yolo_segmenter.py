"""YOLO-based semantic segmentation for aerial imagery.

Wraps Ultralytics YOLOv8/v11 segmentation models and provides a clean
interface for per-class mask extraction used by the landmark pipeline.

Supports three backends:
  - ``ultralytics`` (.pt weights) — default, GPU-capable.
  - ``onnxruntime`` (.onnx weights) — CPU/GPU cross-platform.
  - ``ncnn`` (.param/.bin pair) — ARM-optimised for Raspberry Pi 5.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from loguru import logger


@dataclass
class SegmentationResult:
    """Output of one segmentation inference pass.

    Attributes:
        masks: Binary masks per detected instance, shape (N, H, W), bool.
        class_ids: Class ID for each instance, shape (N,), int.
        class_names: Human-readable class name for each instance.
        scores: Confidence score for each instance, shape (N,), float32.
        boxes: Bounding boxes [x1, y1, x2, y2] per instance, shape (N, 4).
        image_hw: Input image size as (H, W).
    """

    masks: np.ndarray        # (N, H, W) bool
    class_ids: np.ndarray    # (N,) int
    class_names: list[str]
    scores: np.ndarray       # (N,) float32
    boxes: np.ndarray        # (N, 4) float32
    image_hw: tuple[int, int]

    @property
    def n_instances(self) -> int:
        """Number of detected instances."""
        return int(self.masks.shape[0])

    def filter_by_class(self, class_name: str) -> "SegmentationResult":
        """Return a new SegmentationResult keeping only a specific class.

        Args:
            class_name: Semantic class name to keep (e.g. "tree").

        Returns:
            Filtered SegmentationResult containing only matching instances.
        """
        keep = [i for i, n in enumerate(self.class_names) if n == class_name]
        if not keep:
            H, W = self.image_hw
            return SegmentationResult(
                masks=np.zeros((0, H, W), dtype=bool),
                class_ids=np.zeros(0, dtype=int),
                class_names=[],
                scores=np.zeros(0, dtype=np.float32),
                boxes=np.zeros((0, 4), dtype=np.float32),
                image_hw=self.image_hw,
            )
        idx = np.array(keep, dtype=int)
        return SegmentationResult(
            masks=self.masks[idx],
            class_ids=self.class_ids[idx],
            class_names=[self.class_names[i] for i in keep],
            scores=self.scores[idx],
            boxes=self.boxes[idx],
            image_hw=self.image_hw,
        )

    def merged_mask(self) -> np.ndarray:
        """Return a single HxW mask with class IDs (0 = background).

        Higher-confidence instances win on overlapping pixels.

        Returns:
            Integer mask of shape (H, W), uint8, where each pixel holds the
            class ID of the highest-confidence overlapping instance or 0.
        """
        H, W = self.image_hw
        out = np.zeros((H, W), dtype=np.uint8)
        if self.n_instances == 0:
            return out
        # Paint from lowest to highest score so high-conf instances win.
        order = np.argsort(self.scores)
        for i in order:
            out[self.masks[i]] = int(self.class_ids[i])
        return out


def _empty_result(image_hw: tuple[int, int]) -> SegmentationResult:
    H, W = image_hw
    return SegmentationResult(
        masks=np.zeros((0, H, W), dtype=bool),
        class_ids=np.zeros(0, dtype=int),
        class_names=[],
        scores=np.zeros(0, dtype=np.float32),
        boxes=np.zeros((0, 4), dtype=np.float32),
        image_hw=image_hw,
    )


class YOLOSegmenter:
    """Wraps a YOLO segmentation model for semantic landmark extraction.

    Supports inference on single frames or batches. Handles model loading
    from a weights file and optional NCNN/ONNX backend for deployment.

    Args:
        weights_path: Path to YOLO weights file (.pt or .onnx).
        class_names: Ordered list of class names matching the model output.
        confidence_threshold: Minimum score to keep a detection.
        iou_threshold: NMS IoU threshold.
        imgsz: Model inference image size (square side, pixels).
        device: Torch device string, e.g. "cpu", "cuda:0".
    """

    def __init__(
        self,
        weights_path: Path,
        class_names: list[str],
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        self.weights_path = Path(weights_path)
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self._model: Optional[Any] = None
        self._backend: str = "unloaded"

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load model weights into memory.

        Raises:
            FileNotFoundError: If ``self.weights_path`` does not exist.
            ImportError: If the required backend library is not installed.
        """
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        ext = self.weights_path.suffix.lower()

        if ext == ".pt":
            self._load_ultralytics()
        elif ext == ".onnx":
            self._load_onnx()
        else:
            raise ValueError(
                f"Unsupported weight format '{ext}'. "
                "Use .pt (ultralytics) or .onnx (onnxruntime)."
            )
        logger.info(
            "YOLOSegmenter loaded: backend={}, weights={}",
            self._backend,
            self.weights_path,
        )

    def _load_ultralytics(self) -> None:
        from ultralytics import YOLO  # type: ignore[import]

        self._model = YOLO(str(self.weights_path))
        self._backend = "ultralytics"

    def _load_onnx(self) -> None:
        import onnxruntime as ort  # type: ignore[import]

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device != "cpu"
            else ["CPUExecutionProvider"]
        )
        self._model = ort.InferenceSession(str(self.weights_path), providers=providers)
        self._input_name: str = self._model.get_inputs()[0].name
        self._backend = "onnxruntime"

    def _check_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() before predict().")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, image: np.ndarray) -> SegmentationResult:
        """Run segmentation inference on a single RGB image.

        Args:
            image: Input image, shape (H, W, 3), uint8, RGB order.

        Returns:
            SegmentationResult containing all detected instances.

        Raises:
            RuntimeError: If ``load()`` has not been called.
        """
        self._check_loaded()

        if self._backend == "ultralytics":
            return self._predict_ultralytics_single(image)
        elif self._backend == "onnxruntime":
            return self._predict_onnx_single(image)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    def predict_batch(self, images: list[np.ndarray]) -> list[SegmentationResult]:
        """Run segmentation on a batch of images.

        Args:
            images: List of RGB images, each (H, W, 3) uint8.

        Returns:
            List of SegmentationResult, one per input image.

        Raises:
            RuntimeError: If ``load()`` has not been called.
        """
        self._check_loaded()
        if not images:
            return []

        if self._backend == "ultralytics":
            return self._predict_ultralytics_batch(images)
        else:
            return [self.predict(img) for img in images]

    def warmup(self, n_iters: int = 3) -> None:
        """Run dummy inferences to warm up the model and measure latency.

        Args:
            n_iters: Number of warm-up inference passes.

        Raises:
            RuntimeError: If ``load()`` has not been called.
        """
        self._check_loaded()
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        t_start = time.perf_counter()
        for _ in range(n_iters):
            self.predict(dummy)
        elapsed = (time.perf_counter() - t_start) * 1000.0
        logger.debug(
            "Warmup: {} iterations, {:.1f} ms total ({:.1f} ms/iter)",
            n_iters,
            elapsed,
            elapsed / n_iters,
        )

    # ── Backend implementations ───────────────────────────────────────────────

    def _predict_ultralytics_single(self, image: np.ndarray) -> SegmentationResult:
        results = self._model.predict(  # type: ignore[union-attr]
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._parse_ultralytics_result(results[0], image.shape[:2])

    def _predict_ultralytics_batch(
        self, images: list[np.ndarray]
    ) -> list[SegmentationResult]:
        results = self._model.predict(  # type: ignore[union-attr]
            source=images,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return [
            self._parse_ultralytics_result(r, img.shape[:2])
            for r, img in zip(results, images)
        ]

    def _parse_ultralytics_result(
        self, result: Any, image_hw: tuple[int, int]
    ) -> SegmentationResult:
        """Convert a single ultralytics Results object to SegmentationResult."""
        H, W = image_hw

        if result.masks is None or len(result.boxes) == 0:
            return _empty_result(image_hw)

        # Masks: (N, H_m, W_m) float32 in [0, 1]
        masks_raw = result.masks.data.cpu().numpy()
        resized_masks = np.stack(
            [
                cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR) >= 0.5
                for m in masks_raw
            ],
            axis=0,
        ).astype(bool)  # (N, H, W)

        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy().astype(np.float32)
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        names = [
            self.class_names[c] if c < len(self.class_names) else str(c)
            for c in class_ids
        ]

        return SegmentationResult(
            masks=resized_masks,
            class_ids=class_ids,
            class_names=names,
            scores=scores,
            boxes=boxes,
            image_hw=image_hw,
        )

    def _predict_onnx_single(self, image: np.ndarray) -> SegmentationResult:
        """Run inference via onnxruntime session.

        YOLO ONNX export from ultralytics produces an output of shape
        (1, 4+nc+32, num_anchors) for detections + (1, 32, Hm, Wm) for
        proto masks.  We apply the standard post-processing here.
        """
        import onnxruntime as ort  # type: ignore[import]  # noqa: F401

        H_orig, W_orig = image.shape[:2]
        # Letterbox + normalise to (1, 3, imgsz, imgsz) float32 [0, 1]
        blob, ratio, (pad_w, pad_h) = self._letterbox(image, self.imgsz)

        outputs = self._model.run(  # type: ignore[union-attr]
            None, {self._input_name: blob}
        )

        # outputs[0]: (1, 4+nc+32, na)   predictions
        # outputs[1]: (1, 32, Hm, Wm)    proto masks
        preds = outputs[0][0]   # (4+nc+32, na)
        protos = outputs[1][0]  # (32, Hm, Wm)

        return self._decode_onnx_output(preds, protos, H_orig, W_orig, ratio, pad_w, pad_h)

    def _letterbox(
        self, image: np.ndarray, target: int
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        """Letterbox-resize to square keeping aspect ratio."""
        H, W = image.shape[:2]
        ratio = min(target / H, target / W)
        new_H, new_W = int(round(H * ratio)), int(round(W * ratio))
        resized = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

        pad_h = (target - new_H) / 2
        pad_w = (target - new_W) / 2
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        # (H, W, 3) uint8 BGR → (1, 3, H, W) float32
        blob = (
            padded[:, :, ::-1].astype(np.float32) / 255.0
        ).transpose(2, 0, 1)[np.newaxis]
        return blob, ratio, (pad_w, pad_h)

    def _decode_onnx_output(
        self,
        preds: np.ndarray,
        protos: np.ndarray,
        H_orig: int,
        W_orig: int,
        ratio: float,
        pad_w: float,
        pad_h: float,
    ) -> SegmentationResult:
        """Decode raw ONNX outputs into a SegmentationResult."""
        nc = len(self.class_names)
        # preds: (4+nc+32, na) — transpose to (na, 4+nc+32)
        preds = preds.T  # (na, 4+nc+32)

        # Scores: max over nc classes
        box_pred = preds[:, :4]
        cls_pred = preds[:, 4 : 4 + nc]
        mask_coeff = preds[:, 4 + nc :]

        scores = cls_pred.max(axis=1)
        class_ids = cls_pred.argmax(axis=1)
        keep = scores >= self.confidence_threshold

        if not keep.any():
            return _empty_result((H_orig, W_orig))

        box_pred = box_pred[keep]
        scores = scores[keep].astype(np.float32)
        class_ids = class_ids[keep].astype(int)
        mask_coeff = mask_coeff[keep]

        # Convert cx,cy,w,h → x1,y1,x2,y2 (letterboxed space)
        boxes_xyxy = self._xywh2xyxy(box_pred)

        # NMS per class
        final_idx = self._nms(boxes_xyxy, scores, self.iou_threshold)
        if len(final_idx) == 0:
            return _empty_result((H_orig, W_orig))

        boxes_xyxy = boxes_xyxy[final_idx]
        scores = scores[final_idx]
        class_ids = class_ids[final_idx]
        mask_coeff = mask_coeff[final_idx]

        # Scale boxes back to original image coordinates
        boxes_orig = boxes_xyxy.copy()
        boxes_orig[:, [0, 2]] = (boxes_orig[:, [0, 2]] - pad_w) / ratio
        boxes_orig[:, [1, 3]] = (boxes_orig[:, [1, 3]] - pad_h) / ratio
        boxes_orig = np.clip(boxes_orig, 0, [W_orig, H_orig, W_orig, H_orig])

        # Proto masks: (32, Hm, Wm) → (Hm*Wm, 32)
        _, Hm, Wm = protos.shape
        protos_flat = protos.reshape(32, -1).T  # (Hm*Wm, 32)
        mask_pred = (mask_coeff @ protos_flat.T).reshape(
            len(final_idx), Hm, Wm
        )  # (N, Hm, Wm)

        # Sigmoid + resize to original
        mask_pred = 1.0 / (1.0 + np.exp(-mask_pred))
        masks_bool = np.stack(
            [
                cv2.resize(m, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR) >= 0.5
                for m in mask_pred
            ]
        ).astype(bool)

        # Crop masks to their bounding box (cleaner edges)
        for i, box in enumerate(boxes_orig.astype(int)):
            x1, y1, x2, y2 = box
            crop = np.zeros((H_orig, W_orig), dtype=bool)
            crop[y1:y2, x1:x2] = masks_bool[i, y1:y2, x1:x2]
            masks_bool[i] = crop

        class_names = [
            self.class_names[c] if c < len(self.class_names) else str(c)
            for c in class_ids
        ]

        return SegmentationResult(
            masks=masks_bool,
            class_ids=class_ids,
            class_names=class_names,
            scores=scores,
            boxes=boxes_orig.astype(np.float32),
            image_hw=(H_orig, W_orig),
        )

    @staticmethod
    def _xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
        out = np.empty_like(boxes)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return out

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        """Simple greedy NMS. Returns indices to keep."""
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while len(order):
            i = int(order[0])
            keep.append(i)
            if len(order) == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
            order = rest[iou <= iou_threshold]
        return keep
