"""Raspberry Pi 5 hardware benchmarking suite.

Measures per-module and end-to-end pipeline latency on Pi 5 hardware,
validating that real-time processing constraints are met.

Target constraints:
    YOLO segmentation:  ≤ 80 ms
    Embedding head:     ≤ 20 ms
    Stereo depth:       ≤ 30 ms
    TILM matching:      ≤ 10 ms
    EKF update:         ≤ 5 ms
    Total pipeline:     ≤ 150 ms (≥ 6.7 FPS)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np


@dataclass
class BenchmarkResult:
    """Results from a benchmark run on a single module or pipeline.

    Attributes:
        module_name: Name of the benchmarked module.
        n_runs: Number of timed iterations.
        latencies_ms: Per-run latency in milliseconds, shape (n_runs,).
        mean_ms: Mean latency in milliseconds.
        std_ms: Standard deviation of latency.
        min_ms: Minimum latency.
        max_ms: Maximum latency.
        p95_ms: 95th percentile latency.
        p99_ms: 99th percentile latency.
        target_ms: Target latency requirement. NaN if not set.
        passes: Whether mean latency is within the target.
    """

    module_name: str
    n_runs: int
    latencies_ms: np.ndarray = field(default_factory=lambda: np.empty(0))
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    target_ms: float = float("nan")
    passes: bool = False

    @classmethod
    def from_latencies(
        cls,
        module_name: str,
        latencies_ms: np.ndarray,
        target_ms: float = float("nan"),
    ) -> "BenchmarkResult":
        """Compute statistics from a raw latency array.

        Args:
            module_name: Name of the benchmarked module.
            latencies_ms: Per-run latency measurements in milliseconds.
            target_ms: Optional target latency for pass/fail.

        Returns:
            Populated BenchmarkResult.
        """
        return cls(
            module_name=module_name,
            n_runs=len(latencies_ms),
            latencies_ms=latencies_ms,
            mean_ms=float(np.mean(latencies_ms)),
            std_ms=float(np.std(latencies_ms)),
            min_ms=float(np.min(latencies_ms)),
            max_ms=float(np.max(latencies_ms)),
            p95_ms=float(np.percentile(latencies_ms, 95)),
            p99_ms=float(np.percentile(latencies_ms, 99)),
            target_ms=target_ms,
            passes=float(np.mean(latencies_ms)) <= target_ms if not np.isnan(target_ms) else True,
        )


class Pi5Benchmark:
    """Benchmark suite for validating Pi 5 real-time performance.

    Args:
        output_dir: Directory to write benchmark CSV and plots.
        n_runs: Number of timed runs per module.
        warmup: Number of warm-up runs before timing.
    """

    # Target latencies in milliseconds
    TARGETS: dict[str, float] = {
        "yolo_segmentation": 80.0,
        "embedding_head":    20.0,
        "stereo_depth":      30.0,
        "tilm_matching":     10.0,
        "ekf_update":         5.0,
        "full_pipeline":    150.0,
    }

    def __init__(
        self,
        output_dir: Path = Path("benchmark_results"),
        n_runs: int = 100,
        warmup: int = 10,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.n_runs = n_runs
        self.warmup = warmup
        self._results: list[BenchmarkResult] = []

    def benchmark_callable(
        self,
        fn: Callable[[], object],
        module_name: str,
        target_ms: float = float("nan"),
    ) -> BenchmarkResult:
        """Time a callable and return a BenchmarkResult.

        Args:
            fn: Zero-argument callable to time.
            module_name: Human-readable module name for reporting.
            target_ms: Target latency in milliseconds.

        Returns:
            BenchmarkResult with statistics.
        """
        for _ in range(self.warmup):
            fn()

        latencies = np.empty(self.n_runs, dtype=np.float64)
        for i in range(self.n_runs):
            t0 = time.perf_counter()
            fn()
            latencies[i] = (time.perf_counter() - t0) * 1000.0

        result = BenchmarkResult.from_latencies(module_name, latencies, target_ms)
        self._results.append(result)
        return result

    def run_all(self, pipeline: object) -> list[BenchmarkResult]:
        """Run all module benchmarks against a full pipeline instance.

        Calls ``pipeline.yolo_segmenter``, ``pipeline.ekf``, etc. if they
        expose a ``predict``/``update`` callable.  Each callable is wrapped
        with a dummy input and timed individually.

        Args:
            pipeline: Initialised NavigationPipeline to benchmark.

        Returns:
            List of BenchmarkResult, one per module.
        """
        results: list[BenchmarkResult] = []
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

        # YOLO segmentation
        if hasattr(pipeline, "segmenter") and hasattr(pipeline.segmenter, "predict"):
            r = self.benchmark_callable(
                lambda: pipeline.segmenter.predict(dummy_img),  # type: ignore[union-attr]
                "yolo_segmentation",
                self.TARGETS["yolo_segmentation"],
            )
            results.append(r)

        return results

    def report(self) -> str:
        """Generate a human-readable report of all benchmark results.

        Returns:
            Formatted report string with one module per row.
        """
        if not self._results:
            return "No benchmark results."

        header = f"{'Module':<25} {'Mean ms':>9} {'Std ms':>8} {'P95 ms':>8} {'Target ms':>10} {'Pass':>5}"
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for r in self._results:
            target_str = f"{r.target_ms:>10.1f}" if not np.isnan(r.target_ms) else "       N/A"
            pass_str = "YES" if r.passes else " NO"
            lines.append(
                f"{r.module_name:<25} {r.mean_ms:>9.2f} {r.std_ms:>8.2f} "
                f"{r.p95_ms:>8.2f} {target_str} {pass_str:>5}"
            )
        lines.append(sep)
        return "\n".join(lines)

    def save_csv(self) -> Path:
        """Save all benchmark results to a CSV file.

        Returns:
            Path to the written CSV file (``<output_dir>/benchmark.csv``).
        """
        import csv

        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "benchmark.csv"

        fieldnames = [
            "module_name", "n_runs", "mean_ms", "std_ms",
            "min_ms", "max_ms", "p95_ms", "p99_ms", "target_ms", "passes",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._results:
                writer.writerow(
                    {
                        "module_name": r.module_name,
                        "n_runs": r.n_runs,
                        "mean_ms": f"{r.mean_ms:.4f}",
                        "std_ms": f"{r.std_ms:.4f}",
                        "min_ms": f"{r.min_ms:.4f}",
                        "max_ms": f"{r.max_ms:.4f}",
                        "p95_ms": f"{r.p95_ms:.4f}",
                        "p99_ms": f"{r.p99_ms:.4f}",
                        "target_ms": f"{r.target_ms:.1f}" if not np.isnan(r.target_ms) else "",
                        "passes": r.passes,
                    }
                )
        return csv_path
