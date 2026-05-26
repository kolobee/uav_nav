"""System performance monitoring and telemetry.

Tracks per-frame timing, memory usage, and algorithm quality metrics.
Optionally streams to Rerun SDK for real-time visualisation.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np


@dataclass
class MonitorMetrics:
    """Snapshot of system performance metrics.

    Attributes:
        frame_idx: Current frame index.
        timestamp: Frame timestamp in seconds.
        latency_ms: Total pipeline latency for this frame in milliseconds.
        segmentation_ms: Segmentation module latency (ms).
        depth_ms: Depth estimation module latency (ms).
        embedding_ms: Embedding computation latency (ms).
        matching_ms: TILM matching latency (ms).
        ekf_ms: EKF update latency (ms).
        memory_mb: Estimated RAM usage in megabytes.
        n_landmarks: Number of landmarks extracted this frame.
        tilm_quality: TILM matching quality score.
        position_error: Ground-truth position error (m), or NaN if unavailable.
        cpu_percent: CPU usage percentage.
    """

    frame_idx: int = 0
    timestamp: float = 0.0
    latency_ms: float = 0.0
    segmentation_ms: float = 0.0
    depth_ms: float = 0.0
    embedding_ms: float = 0.0
    matching_ms: float = 0.0
    ekf_ms: float = 0.0
    memory_mb: float = 0.0
    n_landmarks: int = 0
    tilm_quality: float = 0.0
    position_error: float = float("nan")
    cpu_percent: float = 0.0


class SystemMonitor:
    """Collects, aggregates, and optionally exports pipeline telemetry.

    Args:
        window_size: Number of frames in the rolling statistics window.
        enable_rerun: Whether to push metrics to a Rerun SDK session.
        export_csv_path: Optional path to write a CSV telemetry log.
    """

    def __init__(
        self,
        window_size: int = 100,
        enable_rerun: bool = False,
        export_csv_path: Optional[str] = None,
    ) -> None:
        self.window_size = window_size
        self.enable_rerun = enable_rerun
        self.export_csv_path = export_csv_path
        self._history: Deque[MonitorMetrics] = deque(maxlen=window_size)
        self._start_time: float = time.monotonic()

    def record(self, metrics: MonitorMetrics) -> None:
        """Record a frame's metrics snapshot.

        Args:
            metrics: Populated MonitorMetrics for the current frame.
        """
        self._history.append(metrics)
        if self.export_csv_path is not None:
            self._append_csv(metrics)
        if self.enable_rerun:
            self._push_rerun(metrics)

    def mean_latency_ms(self) -> float:
        """Compute mean end-to-end latency over the rolling window.

        Returns:
            Mean latency in milliseconds, or 0.0 if no data.
        """
        if not self._history:
            return 0.0
        return float(np.mean([m.latency_ms for m in self._history]))

    def fps(self) -> float:
        """Estimate effective frames per second from the rolling window.

        Returns:
            FPS estimate, or 0.0 if fewer than 2 frames recorded.
        """
        if len(self._history) < 2:
            return 0.0
        dt_total = self._history[-1].timestamp - self._history[0].timestamp
        if dt_total <= 0:
            return 0.0
        return float((len(self._history) - 1) / dt_total)

    def summary(self) -> dict[str, float]:
        """Return a summary dictionary of rolling-window statistics.

        Returns:
            Dictionary with keys: "mean_latency_ms", "fps",
            "mean_tilm_quality", "mean_n_landmarks", "mean_position_error".

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("SystemMonitor.summary is not yet implemented.")

    def _append_csv(self, metrics: MonitorMetrics) -> None:
        """Append a metrics row to the CSV export file.

        Args:
            metrics: Metrics to write.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("SystemMonitor._append_csv is not yet implemented.")

    def _push_rerun(self, metrics: MonitorMetrics) -> None:
        """Push metrics to a Rerun SDK session for live visualisation.

        Args:
            metrics: Metrics to push.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError("SystemMonitor._push_rerun is not yet implemented.")

    def reset(self) -> None:
        """Clear the rolling history and reset the start time."""
        self._history.clear()
        self._start_time = time.monotonic()
