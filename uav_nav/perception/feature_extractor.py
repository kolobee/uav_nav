"""YOLO + EmbeddingHead perception pipeline.

Runs YOLOv8n-seg on a frame, then embeds each detected instance with
EmbeddingHead to produce a list of Detection objects ready for TILM lookup.

This is the main entry point for the perception stack at runtime.

Usage::

    from uav_nav.perception.feature_extractor import PerceptionPipeline, Detection

    pipeline = PerceptionPipeline()
    pipeline.load(
        yolo_weights="weights/yolo_midair.pt",
        embedding_weights="weights/embedding_head.pt",
    )
    detections = pipeline.process(img_bgr)
    for d in detections:
        print(d.class_name, d.score, d.embedding.shape)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from uav_nav.perception.yolo_segmenter import SegmentationResult, YOLOSegmenter
from uav_nav.perception.embedding_head import EmbeddingHead


# ── Detection dataclass ───────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single detected landmark instance with its semantic embedding.

    Attributes:
        class_id: Integer class ID (0=river, 1=road, 2=rock, 3=tree).
        class_name: Human-readable class name.
        score: YOLO confidence score in [0, 1].
        bbox_xyxy: Bounding box [x1, y1, x2, y2] in pixel coords, float32.
        mask: Binary instance mask, shape (H, W) bool, or None.
        polygon_flat: Flat list of normalised polygon coords [x0,y0,x1,y1,...].
        embedding: L2-normalised descriptor, shape (128,) float32.
            All-zeros until ``PerceptionPipeline.load()`` is called with
            embedding weights.
    """

    class_id: int
    class_name: str
    score: float
    bbox_xyxy: np.ndarray                  # (4,) float32
    mask: Optional[np.ndarray] = None      # (H, W) bool
    polygon_flat: list[float] = field(default_factory=list)
    embedding: np.ndarray = field(
        default_factory=lambda: np.zeros(128, dtype=np.float32)
    )

    @property
    def bbox_center(self) -> np.ndarray:
        """Centre of the bounding box as (cx, cy) in pixels."""
        return np.array([
            (self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2.0,
            (self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2.0,
        ], dtype=np.float32)

    @property
    def bbox_area(self) -> float:
        """Bounding box area in pixels²."""
        w = float(self.bbox_xyxy[2] - self.bbox_xyxy[0])
        h = float(self.bbox_xyxy[3] - self.bbox_xyxy[1])
        return max(0.0, w * h)


# ── helpers ───────────────────────────────────────────────────────────────────

def _mask_to_polygon(mask: np.ndarray, img_h: int, img_w: int) -> list[float]:
    """Convert a binary instance mask to a flat normalised polygon.

    Uses the largest contour as the polygon approximation.

    Args:
        mask: Boolean mask, shape (H, W).
        img_h: Original image height (for normalisation).
        img_w: Original image width.

    Returns:
        Flat list ``[x0, y0, x1, y1, …]`` normalised to [0, 1].
        Returns an empty list if the mask is empty.
    """
    import cv2
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    eps = 0.005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(float)
    approx[:, 0] /= img_w
    approx[:, 1] /= img_h
    return approx.flatten().tolist()


# ── pipeline ──────────────────────────────────────────────────────────────────

class PerceptionPipeline:
    """YOLO segmentation + semantic embedding in one call.

    Args:
        conf_threshold: YOLO confidence threshold for detections.
        min_bbox_area: Minimum bounding-box area in pixels² to keep a detection.
        device: Torch device for embedding inference.
    """

    def __init__(
        self,
        conf_threshold: float = 0.25,
        min_bbox_area: float = 400.0,
        device: str = "cpu",
    ) -> None:
        self.conf_threshold = conf_threshold
        self.min_bbox_area = min_bbox_area
        self.device = device
        self._segmenter: Optional[YOLOSegmenter] = None
        self._embedder: Optional[EmbeddingHead] = None

    # ── loading ───────────────────────────────────────────────────────────────

    def load(
        self,
        yolo_weights: str | Path,
        embedding_weights: Optional[str | Path] = None,
    ) -> None:
        """Load YOLO and optionally the embedding head.

        Args:
            yolo_weights: Path to ``.pt`` or ``.onnx`` YOLO weights.
            embedding_weights: Path to ``embedding_head.pt``.  If ``None``,
                embeddings are zero vectors (class-only mode).
        """
        self._segmenter = YOLOSegmenter(conf=self.conf_threshold)
        self._segmenter.load(Path(yolo_weights))

        if embedding_weights is not None:
            self._embedder = EmbeddingHead.load(Path(embedding_weights), device=self.device)

    def _check_loaded(self) -> None:
        if self._segmenter is None:
            raise RuntimeError("Call load() before process().")

    # ── inference ─────────────────────────────────────────────────────────────

    def process(self, img_bgr: np.ndarray) -> list[Detection]:
        """Run full perception stack on one BGR frame.

        Args:
            img_bgr: Input image, shape (H, W, 3), uint8 BGR (OpenCV convention).

        Returns:
            List of Detection objects, one per detected instance.
            Sorted by descending confidence score.
        """
        self._check_loaded()
        seg: SegmentationResult = self._segmenter.predict(img_bgr)
        h, w = img_bgr.shape[:2]

        detections: list[Detection] = []
        for i in range(seg.n_instances):
            bbox = seg.boxes[i].astype(np.float32)          # (4,) xyxy
            area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            if area < self.min_bbox_area:
                continue

            mask = seg.masks[i] if seg.masks is not None else None
            poly = _mask_to_polygon(mask, h, w) if mask is not None else []

            emb = np.zeros(128, dtype=np.float32)
            if self._embedder is not None and poly:
                emb = self._embedder.encode_crop(img_bgr, poly)

            detections.append(Detection(
                class_id=int(seg.class_ids[i]),
                class_name=seg.class_names[i] if seg.class_names else str(seg.class_ids[i]),
                score=float(seg.scores[i]),
                bbox_xyxy=bbox,
                mask=mask,
                polygon_flat=poly,
                embedding=emb,
            ))

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def process_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Run ``process`` on a list of frames.

        Args:
            images: List of BGR frames.

        Returns:
            List of detection lists, one per input frame.
        """
        return [self.process(img) for img in images]

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._segmenter is not None

    @property
    def has_embedding(self) -> bool:
        return self._embedder is not None
