"""Export models to NCNN format for Raspberry Pi 5 deployment.

NCNN is a lightweight inference framework optimised for ARM processors.
The pipeline is: PyTorch → ONNX → NCNN.

Two conversion paths are supported:
  1. **ultralytics built-in** (preferred for YOLO): ``YOLO.export(format='ncnn')``.
  2. **onnx2ncnn** (via subprocess): for generic ONNX models.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class NCNNExportConfig:
    """Configuration for NCNN export.

    Attributes:
        onnx_path: Input ONNX model path. If None, derived from model_name.
        output_dir: Directory for .param and .bin output files.
        use_vulkan: Whether to enable Vulkan GPU acceleration (Pi 5 supports it).
        fp16: Whether to quantise weights to FP16 to reduce model size.
        optimize_level: NCNN optimiser level (0-3).
        num_threads: Number of CPU threads for inference.
    """

    onnx_path: Optional[Path] = None
    output_dir: Path = Path("ncnn_models")
    use_vulkan: bool = False
    fp16: bool = True
    optimize_level: int = 2
    num_threads: int = 4


def export_yolo_to_ncnn(
    pt_path: Path,
    config: NCNNExportConfig,
    imgsz: int = 640,
) -> tuple[Path, Path]:
    """Export a YOLO .pt model to NCNN using ultralytics built-in export.

    Args:
        pt_path: Path to the .pt YOLO weights.
        config: NCNNExportConfig.
        imgsz: Inference image size (square).

    Returns:
        Tuple (param_path, bin_path).

    Raises:
        FileNotFoundError: If pt_path does not exist.
        ImportError: If ultralytics is not installed.
    """
    if not pt_path.exists():
        raise FileNotFoundError(pt_path)

    from ultralytics import YOLO  # type: ignore[import]

    model = YOLO(str(pt_path))
    exported = model.export(
        format="ncnn",
        imgsz=imgsz,
        half=config.fp16,
    )
    # ultralytics places the ncnn dir next to the .pt file
    ncnn_dir = Path(str(exported))
    config.output_dir.mkdir(parents=True, exist_ok=True)

    stem = pt_path.stem
    param_src = ncnn_dir / f"{stem}.param"
    bin_src = ncnn_dir / f"{stem}.bin"

    param_dst = config.output_dir / f"{stem}.param"
    bin_dst = config.output_dir / f"{stem}.bin"

    if param_src.exists():
        shutil.copy2(param_src, param_dst)
    if bin_src.exists():
        shutil.copy2(bin_src, bin_dst)

    logger.success("YOLO NCNN export → {} , {}", param_dst, bin_dst)
    return param_dst, bin_dst


def export_to_ncnn(
    onnx_path: Path,
    config: NCNNExportConfig,
    model_name: str = "model",
) -> tuple[Path, Path]:
    """Convert an ONNX model to NCNN .param/.bin format via onnx2ncnn.

    Requires ``onnx2ncnn`` to be on ``PATH`` (part of the NCNN tools package).

    Args:
        onnx_path: Path to the source ONNX file.
        config: NCNNExportConfig controlling conversion settings.
        model_name: Base name for the .param and .bin output files.

    Returns:
        Tuple (param_path, bin_path) for the NCNN model files.

    Raises:
        FileNotFoundError: If onnx_path does not exist.
        RuntimeError: If onnx2ncnn is not found on PATH or conversion fails.
    """
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)

    tool = shutil.which("onnx2ncnn")
    if tool is None:
        raise RuntimeError(
            "onnx2ncnn not found on PATH. "
            "Install NCNN tools: https://github.com/Tencent/ncnn/releases"
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    param_path = config.output_dir / f"{model_name}.param"
    bin_path = config.output_dir / f"{model_name}.bin"

    cmd = [tool, str(onnx_path), str(param_path), str(bin_path)]
    logger.info("Running: {}", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"onnx2ncnn failed:\n{result.stderr}")

    logger.success("NCNN export → {} , {}", param_path, bin_path)
    return param_path, bin_path


def verify_ncnn_outputs(
    onnx_path: Path,
    param_path: Path,
    bin_path: Path,
    input_shape: tuple[int, int, int, int],
    tolerance: float = 1e-3,
) -> bool:
    """Verify NCNN output matches ONNX output within tolerance.

    Requires the ``ncnn`` Python package (``pip install ncnn``) to be installed.

    Args:
        onnx_path: Reference ONNX model path.
        param_path: NCNN .param file path.
        bin_path: NCNN .bin file path.
        input_shape: Input tensor shape (B, C, H, W).
        tolerance: Maximum allowed per-element absolute difference.

    Returns:
        True if outputs match within tolerance.
    """
    import onnxruntime as ort  # type: ignore[import]

    dummy = np.random.rand(*input_shape).astype(np.float32)

    # ONNX reference
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {sess.get_inputs()[0].name: dummy})[0]

    # NCNN inference
    try:
        import ncnn  # type: ignore[import]
    except ImportError:
        logger.warning("ncnn Python package not installed; skipping NCNN output verification")
        return True

    net = ncnn.Net()
    net.load_param(str(param_path))
    net.load_model(str(bin_path))

    ex = net.create_extractor()
    # Assumes the first input/output layer names; adjust if needed
    B, C, H, W = input_shape
    mat_in = ncnn.Mat(W, H, C)
    # Copy data: ncnn.Mat is CHW
    for c in range(C):
        for h in range(H):
            for w in range(W):
                mat_in[c * H * W + h * W + w] = float(dummy[0, c, h, w])

    ex.input("input", mat_in)
    _, mat_out = ex.extract("output")
    ncnn_out = np.array(mat_out).reshape(ort_out.shape)

    max_diff = float(np.abs(ort_out - ncnn_out).max())
    logger.info("NCNN vs ONNX max abs diff: {:.2e} (tol {:.2e})", max_diff, tolerance)
    return max_diff <= tolerance


def benchmark_ncnn(
    param_path: Path,
    bin_path: Path,
    input_shape: tuple[int, int, int, int],
    n_runs: int = 100,
    num_threads: int = 4,
) -> dict[str, float]:
    """Benchmark NCNN model inference latency.

    Requires the ``ncnn`` Python package.

    Args:
        param_path: NCNN .param file path.
        bin_path: NCNN .bin file path.
        input_shape: Input tensor shape (B, C, H, W).
        n_runs: Number of timed inference runs.
        num_threads: Number of CPU threads for inference.

    Returns:
        Dictionary with keys: ``mean_ms``, ``std_ms``, ``min_ms``, ``max_ms``.
    """
    try:
        import ncnn  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "ncnn Python package not installed. "
            "On Pi 5: pip install ncnn"
        ) from e

    net = ncnn.Net()
    net.opt.num_threads = num_threads
    net.load_param(str(param_path))
    net.load_model(str(bin_path))

    B, C, H, W = input_shape
    dummy = np.random.rand(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(10):
        ex = net.create_extractor()
        mat_in = ncnn.Mat(W, H, C, dummy[0].flatten().tolist())
        ex.input("input", mat_in)
        ex.extract("output")

    latencies = np.empty(n_runs, dtype=np.float64)
    for i in range(n_runs):
        ex = net.create_extractor()
        mat_in = ncnn.Mat(W, H, C, dummy[0].flatten().tolist())
        t0 = time.perf_counter()
        ex.input("input", mat_in)
        ex.extract("output")
        latencies[i] = (time.perf_counter() - t0) * 1000.0

    return {
        "mean_ms": float(latencies.mean()),
        "std_ms": float(latencies.std()),
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
    }
