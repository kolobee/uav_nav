"""Raspberry Pi 5 optimised runtime configuration.

Provides a Pi5Runtime class that wraps the NavigationPipeline with
Pi 5-specific optimisations: NCNN inference, reduced resolution,
ARM-optimised thread counts, and power-mode management.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from uav_nav.runtime.pipeline import NavigationPipeline, PipelineConfig


@dataclass
class Pi5RuntimeConfig:
    """Configuration for Pi 5 optimised runtime.

    Attributes:
        ncnn_yolo_param: Path to NCNN YOLO .param file.
        ncnn_yolo_bin: Path to NCNN YOLO .bin file.
        ncnn_embed_param: Path to NCNN embedding head .param file.
        ncnn_embed_bin: Path to NCNN embedding head .bin file.
        n_threads: Number of CPU threads for NCNN inference (recommended: 4).
        use_vulkan: Whether to use Vulkan GPU acceleration.
        camera_index: V4L2 camera device index for live capture.
        capture_width: Camera capture width in pixels.
        capture_height: Camera capture height in pixels.
        target_fps: Target processing frame rate (recommended: 6-10 for Pi 5).
        governor: CPU frequency governor ("performance" or "ondemand").
    """

    ncnn_yolo_param: Optional[Path] = None
    ncnn_yolo_bin: Optional[Path] = None
    ncnn_embed_param: Optional[Path] = None
    ncnn_embed_bin: Optional[Path] = None
    n_threads: int = 4
    use_vulkan: bool = False
    camera_index: int = 0
    capture_width: int = 640
    capture_height: int = 480
    target_fps: float = 8.0
    governor: str = "performance"


class Pi5Runtime:
    """Raspberry Pi 5 optimised navigation runtime.

    Wraps NavigationPipeline with NCNN backends, camera capture,
    and performance tuning for ARM Cortex-A76 deployment.

    Args:
        pi5_config: Pi5-specific runtime configuration.
        pipeline_config: Base pipeline configuration.
    """

    def __init__(
        self,
        pi5_config: Pi5RuntimeConfig,
        pipeline_config: Optional[PipelineConfig] = None,
    ) -> None:
        self.pi5_config = pi5_config
        self.pipeline_config = pipeline_config or PipelineConfig()
        self._pipeline: Optional[NavigationPipeline] = None
        self._camera: Optional[object] = None

    def setup(self) -> None:
        """Initialise camera, set CPU governor, and load NCNN models.

        Raises:
            NotImplementedError: Not yet implemented.
            RuntimeError: If camera device cannot be opened.
        """
        raise NotImplementedError("Pi5Runtime.setup is not yet implemented.")

    def set_cpu_governor(self, governor: str) -> None:
        """Set the Linux CPU frequency governor.

        Args:
            governor: Governor name (e.g. "performance", "ondemand").

        Raises:
            NotImplementedError: Not yet implemented.
            PermissionError: If not running as root.
        """
        raise NotImplementedError("Pi5Runtime.set_cpu_governor is not yet implemented.")

    def run(self, waypoints: np.ndarray, max_frames: Optional[int] = None) -> None:
        """Start the real-time navigation loop.

        Reads frames from the camera, processes them through the pipeline,
        and outputs velocity commands.

        Args:
            waypoints: Planned path waypoints in NED, shape (N, 3).
            max_frames: Maximum number of frames to process. None = unlimited.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("Pi5Runtime.run is not yet implemented.")

    def shutdown(self) -> None:
        """Release camera, restore CPU governor, flush logs.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("Pi5Runtime.shutdown is not yet implemented.")

    def get_system_info(self) -> dict[str, object]:
        """Collect Pi 5 system information for logging.

        Returns:
            Dictionary with keys: "cpu_temp_c", "cpu_freq_mhz",
            "memory_mb", "governor", "hostname".

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("Pi5Runtime.get_system_info is not yet implemented.")
