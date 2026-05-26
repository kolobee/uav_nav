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
    """

    def __init__(
        self,
        embedding_head: EmbeddingHead,
        min_mask_area: int = 200,
        max_landmarks_per_frame: int = 20,
        depth_percentile: float = 50.0,
        valid_classes: Optional[set[str]] = None,
    ) -> None:
        self.embedding_head = embedding_head
        self.min_mask_area = min_mask_area
        self.max_landmarks_per_frame = max_landmarks_per_frame
        self.depth_percentile = depth_percentile
        self.valid_classes = valid_classes

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

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("LandmarkExtractor.extract is not yet implemented.")

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

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("LandmarkExtractor._crop_and_embed is not yet implemented.")

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

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("LandmarkExtractor._unproject is not yet implemented.")
