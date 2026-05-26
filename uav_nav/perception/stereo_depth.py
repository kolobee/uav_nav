"""Stereo depth estimation from rectified image pairs.

Provides both classical SGM/BM disparity estimation and optional learned
stereo networks (CREStereo, RAFT-Stereo) for accurate depth maps used in
3-D landmark position triangulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from uav_nav.data.calibration import StereoCalibration


class StereoBackend(str, Enum):
    """Available stereo depth estimation backends."""

    SGBM = "sgbm"          # OpenCV Semi-Global Block Matching
    BM = "bm"              # OpenCV Block Matching
    RAFT_STEREO = "raft"   # RAFT-Stereo learned network
    CRES = "cres"          # CREStereo learned network


@dataclass
class DepthResult:
    """Output of one stereo depth estimation pass.

    Attributes:
        disparity: Raw disparity map, shape (H, W), float32.
        depth: Metric depth map in metres, shape (H, W), float32.
            Pixels with invalid depth hold np.nan.
        confidence: Per-pixel confidence (0–1), shape (H, W), float32.
            May be all-ones for classical backends.
        baseline: Stereo baseline in metres used for depth computation.
        focal_length: Focal length in pixels used for depth computation.
    """

    disparity: np.ndarray   # (H, W) float32
    depth: np.ndarray       # (H, W) float32
    confidence: np.ndarray  # (H, W) float32
    baseline: float
    focal_length: float

    def point_cloud(self, K: np.ndarray) -> np.ndarray:
        """Back-project depth map to a 3-D point cloud.

        Args:
            K: Camera intrinsic matrix, shape (3, 3).

        Returns:
            Point cloud array of shape (H * W, 3), float32, with invalid
            depth pixels removed.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("DepthResult.point_cloud is not yet implemented.")

    def depth_at(self, u: float, v: float) -> float:
        """Return the depth at pixel coordinate (u, v) with bilinear interpolation.

        Args:
            u: Horizontal pixel coordinate.
            v: Vertical pixel coordinate.

        Returns:
            Interpolated depth in metres, or np.nan if invalid.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("DepthResult.depth_at is not yet implemented.")


class StereoDepthEstimator:
    """Estimates per-pixel depth from a rectified stereo image pair.

    Args:
        calibration: Stereo calibration providing baseline and focal length.
        backend: Stereo matching algorithm to use.
        min_disparity: Minimum valid disparity value (pixels).
        max_disparity: Maximum valid disparity value (pixels).
        device: Torch device for learned backends.
    """

    def __init__(
        self,
        calibration: StereoCalibration,
        backend: StereoBackend = StereoBackend.SGBM,
        min_disparity: int = 0,
        max_disparity: int = 96,
        device: str = "cpu",
    ) -> None:
        self.calibration = calibration
        self.backend = backend
        self.min_disparity = min_disparity
        self.max_disparity = max_disparity
        self.device = device
        self._matcher: Optional[object] = None

    def load(self) -> None:
        """Initialise the stereo matching backend.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("StereoDepthEstimator.load is not yet implemented.")

    def estimate(
        self, left: np.ndarray, right: np.ndarray
    ) -> DepthResult:
        """Compute depth from a rectified stereo pair.

        Args:
            left: Left rectified image, shape (H, W, 3) or (H, W), uint8.
            right: Right rectified image, same shape as ``left``.

        Returns:
            DepthResult with disparity, depth, and confidence maps.

        Raises:
            NotImplementedError: Not yet implemented.
            RuntimeError: If ``load()`` has not been called.
        """
        raise NotImplementedError("StereoDepthEstimator.estimate is not yet implemented.")

    def rectify_and_estimate(
        self, left_raw: np.ndarray, right_raw: np.ndarray
    ) -> DepthResult:
        """Rectify a raw stereo pair then compute depth.

        Args:
            left_raw: Unrectified left image, shape (H, W, 3), uint8.
            right_raw: Unrectified right image, same shape.

        Returns:
            DepthResult computed on rectified images.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError(
            "StereoDepthEstimator.rectify_and_estimate is not yet implemented."
        )
