"""Semantic landmark extraction from segmentation and depth results.

Combines YOLO segmentation masks with stereo depth to produce 3-D
landmark observations with associated semantic class and descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from uav_nav.perception.embedding_head import EmbeddingHead
from uav_nav.perception.stereo_depth import DepthResult
from uav_nav.perception.yolo_segmenter import SegmentationResult


@dataclass
class Landmark:
    """A single 3-D semantic landmark observation.

    Attributes:
        position_3d: Landmark position in camera frame, shape (3,), float32.
        semantic_class: Semantic class name (e.g. "tree", "rock").
        class_id: Integer class identifier.
        descriptor: L2-normalised embedding vector, shape (D,), float32.
        confidence: Detection confidence score in [0, 1].
        mask_area: Area of the instance mask in pixels.
        centroid_uv: Mask centroid in image coordinates (u, v).
        frame_id: Identifier of the source frame.
        instance_id: Per-frame instance index.
    """

    position_3d: np.ndarray     # (3,)
    semantic_class: str
    class_id: int
    descriptor: np.ndarray      # (D,)
    confidence: float
    mask_area: int
    centroid_uv: tuple[float, float]
    frame_id: str = ""
    instance_id: int = 0

    def distance_to(self, other: "Landmark") -> float:
        """Euclidean distance in 3-D to another landmark.

        Args:
            other: Another Landmark instance.

        Returns:
            3-D distance in metres.
        """
        return float(np.linalg.norm(self.position_3d - other.position_3d))

    def descriptor_distance(self, other: "Landmark") -> float:
        """L2 descriptor distance to another landmark.

        Args:
            other: Another Landmark instance.

        Returns:
            Descriptor L2 distance (0 = identical).
        """
        return float(np.linalg.norm(self.descriptor - other.descriptor))


class LandmarkExtractor:
    """Extracts 3-D semantic landmark observations from a single frame.

    Combines segmentation masks, depth maps, and the embedding head to
    produce a list of Landmark objects per frame.

    Args:
        embedding_head: Trained EmbeddingHead for descriptor computation.
        min_mask_area: Minimum instance mask area in pixels.
        max_landmarks_per_frame: Cap on the number of extracted landmarks.
        depth_percentile: Depth aggregation percentile within each mask
            (e.g. 50 for median depth).
        valid_classes: Set of semantic class names to extract. If None, all
            classes are extracted.
        camera_K: Camera intrinsic matrix (3×3). If None, inferred from
            image size and DepthResult.focal_length at inference time.
    """

    def __init__(
        self,
        embedding_head: EmbeddingHead,
        min_mask_area: int = 200,
        max_landmarks_per_frame: int = 20,
        depth_percentile: float = 50.0,
        valid_classes: Optional[set[str]] = None,
        camera_K: Optional[np.ndarray] = None,
    ) -> None:
        self.embedding_head = embedding_head
        self.min_mask_area = min_mask_area
        self.max_landmarks_per_frame = max_landmarks_per_frame
        self.depth_percentile = depth_percentile
        self.valid_classes = valid_classes
        self._camera_K = camera_K

    def extract(
        self,
        image_rgb: np.ndarray,
        segmentation: SegmentationResult,
        depth: DepthResult,
        frame_id: str = "",
    ) -> list[Landmark]:
        """Extract landmarks from a single frame.

        Args:
            image_rgb: Source RGB image, shape (H, W, 3), uint8.
            segmentation: Segmentation result for the frame.
            depth: Depth result for the frame.
            frame_id: Optional frame identifier string.

        Returns:
            List of Landmark observations, sorted by confidence descending.
        """
        H, W = image_rgb.shape[:2]
        K = self._camera_K if self._camera_K is not None else np.array(
            [[depth.focal_length, 0.0, W / 2.0],
             [0.0, depth.focal_length, H / 2.0],
             [0.0, 0.0, 1.0]], dtype=np.float64
        )

        # Sort instances by confidence descending
        order = np.argsort(-segmentation.scores)

        landmarks: list[Landmark] = []
        for rank, idx in enumerate(order):
            if len(landmarks) >= self.max_landmarks_per_frame:
                break

            cls_name = segmentation.class_names[idx]
            if self.valid_classes is not None and cls_name not in self.valid_classes:
                continue

            mask = segmentation.masks[idx]          # (H, W) bool
            area = int(mask.sum())
            if area < self.min_mask_area:
                continue

            # Centroid in (u, v) = (col, row)
            rows, cols = np.where(mask)
            u = float(cols.mean())
            v = float(rows.mean())

            # Depth aggregation over valid mask pixels
            mask_depths = depth.depth[mask]
            valid = mask_depths[np.isfinite(mask_depths)]
            if valid.size == 0:
                continue
            depth_val = float(np.percentile(valid, self.depth_percentile))
            if depth_val <= 0.0:
                continue

            pos3d = self._unproject((u, v), depth_val, K)
            descriptor = self._crop_and_embed(image_rgb, mask)

            landmarks.append(Landmark(
                position_3d=pos3d,
                semantic_class=cls_name,
                class_id=int(segmentation.class_ids[idx]),
                descriptor=descriptor,
                confidence=float(segmentation.scores[idx]),
                mask_area=area,
                centroid_uv=(u, v),
                frame_id=frame_id,
                instance_id=rank,
            ))

        return landmarks

    def _crop_and_embed(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Crop the masked region and compute its embedding descriptor.

        Args:
            image_rgb: Source image, shape (H, W, 3), uint8.
            mask: Binary instance mask, shape (H, W), bool.

        Returns:
            L2-normalised descriptor vector, shape (D,), float32.
        """
        rows, cols = np.where(mask)
        if rows.size == 0:
            return np.zeros(self.embedding_head.config.embedding_dim, dtype=np.float32)

        H, W = image_rgb.shape[:2]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())

        # Add padding around tight bbox
        pad_r = max(1, int((r1 - r0) * 0.20))
        pad_c = max(1, int((c1 - c0) * 0.20))
        r0 = max(0, r0 - pad_r); r1 = min(H - 1, r1 + pad_r)
        c0 = max(0, c0 - pad_c); c1 = min(W - 1, c1 + pad_c)

        if r1 - r0 < 4 or c1 - c0 < 4:
            return np.zeros(self.embedding_head.config.embedding_dim, dtype=np.float32)

        crop = image_rgb[r0:r1 + 1, c0:c1 + 1].copy()
        crop_mask = mask[r0:r1 + 1, c0:c1 + 1]

        # Fill background with neutral gray
        bg = np.uint8(127)
        for ch in range(3):
            crop[..., ch] = np.where(crop_mask, crop[..., ch], bg)

        cfg = self.embedding_head.config
        import cv2
        import torch
        crop_resized = cv2.resize(crop, (cfg.crop_size, cfg.crop_size),
                                  interpolation=cv2.INTER_LINEAR)

        tensor = self.embedding_head._transform(crop_resized).unsqueeze(0)
        tensor = tensor.to(self.embedding_head.device_str)
        was_training = self.embedding_head.training
        self.embedding_head.eval()
        try:
            with torch.no_grad():
                emb = self.embedding_head.forward(tensor)
        finally:
            if was_training:
                self.embedding_head.train()
        return emb.squeeze(0).cpu().numpy().astype(np.float32)

    def _unproject(
        self,
        centroid_uv: tuple[float, float],
        depth_value: float,
        K: np.ndarray,
    ) -> np.ndarray:
        """Unproject a 2-D pixel to 3-D using the depth value and intrinsics.

        Args:
            centroid_uv: Pixel coordinate (u, v).
            depth_value: Depth in metres.
            K: Camera intrinsic matrix, shape (3, 3).

        Returns:
            3-D position in camera frame, shape (3,), float32.
        """
        u, v = centroid_uv
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        X = (u - cx) * depth_value / fx
        Y = (v - cy) * depth_value / fy
        Z = depth_value
        return np.array([X, Y, Z], dtype=np.float32)
