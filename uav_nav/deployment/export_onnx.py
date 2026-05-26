"""Export PyTorch models to ONNX format for cross-platform inference.

Handles exporting both the YOLO segmenter (via Ultralytics built-in export)
and the custom EmbeddingHead to ONNX with opset 17 and dynamic batch axes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class ONNXExportConfig:
    """Configuration for ONNX export.

    Attributes:
        opset_version: ONNX opset version. Recommended: 17.
        dynamic_axes: Whether to use dynamic batch axis.
        simplify: Whether to run onnx-simplifier after export.
        fp16: Whether to export in half precision (FP16).
        input_shape: Model input shape (B, C, H, W). B may be dynamic.
        output_path: Path for the exported .onnx file.
        verify: Whether to run a forward pass comparison after export.
    """

    opset_version: int = 17
    dynamic_axes: bool = True
    simplify: bool = True
    fp16: bool = False
    input_shape: tuple[int, int, int, int] = (1, 3, 640, 640)
    output_path: Optional[Path] = None
    verify: bool = True


def export_to_onnx(
    model: object,
    config: ONNXExportConfig,
    model_name: str = "model",
) -> Path:
    """Export a PyTorch model to ONNX.

    For ultralytics YOLO models the built-in ``model.export(format='onnx')``
    path is used; for generic ``nn.Module`` objects ``torch.onnx.export`` is
    called instead.

    Args:
        model: Ultralytics YOLO or PyTorch nn.Module (eval mode).
        config: ONNXExportConfig controlling export settings.
        model_name: Base name for the output file when config.output_path is None.

    Returns:
        Path to the written .onnx file.

    Raises:
        ImportError: If torch or ultralytics is not installed.
        RuntimeError: If the model cannot be exported.
    """
    output_path = config.output_path or Path(f"{model_name}.onnx")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Branch: ultralytics YOLO
    try:
        from ultralytics import YOLO  # type: ignore[import]

        if isinstance(model, YOLO):
            imgsz = config.input_shape[2]
            exported = model.export(
                format="onnx",
                opset=config.opset_version,
                dynamic=config.dynamic_axes,
                half=config.fp16,
                imgsz=imgsz,
                simplify=config.simplify,
            )
            exported_path = Path(str(exported))
            if exported_path != output_path:
                import shutil

                shutil.move(str(exported_path), str(output_path))
            logger.success("YOLO ONNX export → {}", output_path)
            return output_path
    except ImportError:
        pass

    # Branch: generic nn.Module
    import torch  # type: ignore[import]

    assert hasattr(model, "forward"), "model must be a torch.nn.Module"

    dummy = torch.zeros(config.input_shape, dtype=torch.float16 if config.fp16 else torch.float32)
    dyn_axes = (
        {"input": {0: "batch_size"}, "output": {0: "batch_size"}} if config.dynamic_axes else None
    )
    torch.onnx.export(
        model,  # type: ignore[arg-type]
        dummy,
        str(output_path),
        opset_version=config.opset_version,
        dynamic_axes=dyn_axes,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )
    logger.info("torch.onnx.export → {}", output_path)

    if config.simplify:
        try:
            import onnx  # type: ignore[import]
            import onnxsim  # type: ignore[import]

            model_onnx = onnx.load(str(output_path))
            simplified, ok = onnxsim.simplify(model_onnx)
            if ok:
                onnx.save(simplified, str(output_path))
                logger.info("onnx-simplifier applied")
            else:
                logger.warning("onnx-simplifier could not simplify the model")
        except ImportError:
            logger.warning("onnx-simplifier not installed; skipping simplification")

    return output_path


def verify_onnx_outputs(
    pytorch_model: object,
    onnx_path: Path,
    input_shape: tuple[int, int, int, int],
    tolerance: float = 1e-4,
) -> bool:
    """Verify that ONNX and PyTorch outputs agree within tolerance.

    Args:
        pytorch_model: Reference PyTorch model (eval mode).
        onnx_path: Path to the exported ONNX file.
        input_shape: Input tensor shape (B, C, H, W).
        tolerance: Maximum allowed per-element absolute difference.

    Returns:
        True if outputs match within tolerance, False otherwise.
    """
    import torch  # type: ignore[import]
    import onnxruntime as ort  # type: ignore[import]

    dummy_np = np.random.rand(*input_shape).astype(np.float32)
    dummy_pt = torch.from_numpy(dummy_np)

    with torch.no_grad():
        pt_out = pytorch_model(dummy_pt)  # type: ignore[operator]
    if isinstance(pt_out, (list, tuple)):
        pt_out = pt_out[0]
    pt_np = pt_out.cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {input_name: dummy_np})[0]

    max_diff = float(np.abs(pt_np - ort_out).max())
    logger.info("ONNX vs PyTorch max abs diff: {:.2e} (tolerance {:.2e})", max_diff, tolerance)
    return max_diff <= tolerance


def benchmark_onnx(
    onnx_path: Path,
    input_shape: tuple[int, int, int, int],
    n_runs: int = 100,
    warmup: int = 10,
) -> dict[str, float]:
    """Benchmark ONNX model inference latency.

    Args:
        onnx_path: Path to the ONNX model file.
        input_shape: Input tensor shape (B, C, H, W).
        n_runs: Number of timed inference runs.
        warmup: Number of warm-up runs before timing.

    Returns:
        Dictionary with keys: ``mean_ms``, ``std_ms``, ``min_ms``, ``max_ms``.
    """
    import onnxruntime as ort  # type: ignore[import]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy = np.random.rand(*input_shape).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {input_name: dummy})

    latencies = np.empty(n_runs, dtype=np.float64)
    for i in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        latencies[i] = (time.perf_counter() - t0) * 1000.0

    return {
        "mean_ms": float(latencies.mean()),
        "std_ms": float(latencies.std()),
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
    }
