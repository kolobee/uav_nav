"""YOLO segmentation evaluation: mAP tables for Experiments 5.3 and 5.4.

Experiment 5.3 — Cross-weather:
    Maps built under one weather condition; evaluated under others.
    Produces a table: rows = test weather, columns = mAP50-mask / mAP-mask / per-class.

Experiment 5.4 — Leave-one-trajectory-out (LOO):
    One trajectory held out from training; the model is evaluated on it.
    Produces a table: rows = held-out trajectory, columns = mAP50-mask / mAP-mask.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class SegMetrics:
    """Per-split segmentation metrics from one ultralytics val() run.

    Attributes:
        split_name: Identifier for this evaluation split.
        map50_mask: mAP@IoU=0.5 for instance segmentation masks.
        map_mask: mAP@IoU=0.5:0.95 for masks.
        map50_box: mAP@IoU=0.5 for bounding boxes.
        map_box: mAP@IoU=0.5:0.95 for boxes.
        map50_per_class: Per-class mAP50-mask list (same order as class_names).
        class_names: Class names corresponding to map50_per_class entries.
        n_images: Number of images evaluated.
        extra: Additional metrics dict from ultralytics results_dict.
    """

    split_name: str = ""
    map50_mask: float = float("nan")
    map_mask: float = float("nan")
    map50_box: float = float("nan")
    map_box: float = float("nan")
    map50_per_class: list[float] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    n_images: int = 0
    extra: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a flat dictionary for CSV/JSON output."""
        d: dict[str, object] = {
            "split": self.split_name,
            "mAP50_mask": self.map50_mask,
            "mAP_mask": self.map_mask,
            "mAP50_box": self.map50_box,
            "mAP_box": self.map_box,
            "n_images": self.n_images,
        }
        for cls, v in zip(self.class_names, self.map50_per_class):
            d[f"mAP50_{cls}"] = v
        return d


class YOLOEvaluator:
    """Evaluate a trained YOLO segmentation model on MidAir test splits.

    Args:
        class_names: Ordered class names expected by the model (e.g. ["tree", "rock", "river"]).
        imgsz: Inference image size (square).
        device: Torch device string (e.g. "cpu", "cuda:0").
        confidence_threshold: Detection confidence threshold for evaluation.
        iou_threshold: NMS IoU threshold.
        batch_size: Batch size used during validation.
    """

    def __init__(
        self,
        class_names: list[str],
        imgsz: int = 640,
        device: str = "cpu",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        batch_size: int = 16,
    ) -> None:
        self.class_names = class_names
        self.imgsz = imgsz
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.batch_size = batch_size

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate_on_yaml(
        self,
        weights: Path,
        dataset_yaml: Path,
        split: str = "test",
        split_name: Optional[str] = None,
    ) -> SegMetrics:
        """Run ultralytics validation on a YOLO-format dataset.

        Args:
            weights: Path to .pt weights file.
            dataset_yaml: Path to the dataset YAML (midair.yaml format).
            split: Dataset split to evaluate: "train", "val", or "test".
            split_name: Human-readable label for this evaluation (defaults to
                ``dataset_yaml.stem + '/' + split``).

        Returns:
            SegMetrics populated from ultralytics val() results.

        Raises:
            FileNotFoundError: If weights or dataset_yaml do not exist.
            ImportError: If ultralytics is not installed.
        """
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

        from ultralytics import YOLO  # type: ignore[import]

        model = YOLO(str(weights))
        results = model.val(
            data=str(dataset_yaml),
            split=split,
            imgsz=self.imgsz,
            batch=self.batch_size,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        name = split_name or f"{dataset_yaml.stem}/{split}"
        return self._parse_val_results(results, name)

    def cross_weather_table(
        self,
        weights: Path,
        weather_yamls: dict[str, Path],
        split: str = "test",
    ) -> list[SegMetrics]:
        """Evaluate a single model across multiple weather-specific datasets.

        Experiment 5.3: model trained on one weather, evaluated on all others.

        Args:
            weights: Path to trained .pt weights.
            weather_yamls: Mapping {weather_label: dataset_yaml_path}.
            split: Dataset split to evaluate for each yaml.

        Returns:
            List of SegMetrics, one per weather condition.
        """
        metrics: list[SegMetrics] = []
        for weather, yaml_path in sorted(weather_yamls.items()):
            logger.info("Evaluating on weather '{}' …", weather)
            m = self.evaluate_on_yaml(weights, yaml_path, split=split, split_name=weather)
            metrics.append(m)
            logger.info("  mAP50-mask: {:.4f}", m.map50_mask)
        return metrics

    def leave_one_out_table(
        self,
        folds: dict[str, tuple[Path, Path]],
        split: str = "test",
    ) -> list[SegMetrics]:
        """Evaluate LOO-trained models on their held-out trajectory.

        Experiment 5.4: for each fold ``{fold_name: (weights_path, yaml_path)}``.

        Args:
            folds: Mapping {fold_name: (weights_path, dataset_yaml_path)}.
            split: Dataset split to evaluate for each fold.

        Returns:
            List of SegMetrics, one per fold.
        """
        metrics: list[SegMetrics] = []
        for fold_name, (weights, yaml_path) in sorted(folds.items()):
            logger.info("LOO fold '{}' …", fold_name)
            m = self.evaluate_on_yaml(weights, yaml_path, split=split, split_name=fold_name)
            metrics.append(m)
            logger.info("  mAP50-mask: {:.4f}", m.map50_mask)
        return metrics

    # ── Output helpers ────────────────────────────────────────────────────────

    def save_table_csv(self, metrics: list[SegMetrics], out_path: Path) -> Path:
        """Write a list of SegMetrics to a CSV file.

        Args:
            metrics: List of SegMetrics to serialise.
            out_path: Destination CSV path.

        Returns:
            Path to the written file.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [m.to_dict() for m in metrics]
        if not rows:
            out_path.write_text("")
            return out_path

        fieldnames = list(rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Saved table → {}", out_path)
        return out_path

    def save_table_json(self, metrics: list[SegMetrics], out_path: Path) -> Path:
        """Write a list of SegMetrics to a JSON file.

        Args:
            metrics: List of SegMetrics to serialise.
            out_path: Destination JSON path.

        Returns:
            Path to the written file.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [m.to_dict() for m in metrics]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=str)
        logger.info("Saved table → {}", out_path)
        return out_path

    def summary_stats(self, metrics: list[SegMetrics]) -> dict[str, float]:
        """Compute mean, std, min, max of mAP50-mask across multiple runs.

        Args:
            metrics: List of SegMetrics.

        Returns:
            Dict with keys: ``mean``, ``std``, ``min``, ``max``.
        """
        vals = np.array([m.map50_mask for m in metrics], dtype=np.float64)
        if len(vals) == 0:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_val_results(self, results: object, split_name: str) -> SegMetrics:
        """Extract SegMetrics from an ultralytics Validator results object."""
        # ultralytics stores per-class metrics in results.seg.maps (array)
        seg = getattr(results, "seg", None)
        box = getattr(results, "box", None)

        map50_mask = float(getattr(getattr(seg, "map50", None), "__float__", lambda: float("nan"))()) if seg is not None else float("nan")
        map_mask = float(getattr(getattr(seg, "map", None), "__float__", lambda: float("nan"))()) if seg is not None else float("nan")
        map50_box = float(getattr(getattr(box, "map50", None), "__float__", lambda: float("nan"))()) if box is not None else float("nan")
        map_box = float(getattr(getattr(box, "map", None), "__float__", lambda: float("nan"))()) if box is not None else float("nan")

        # Try the simpler attribute path used in recent ultralytics versions
        if np.isnan(map50_mask) and seg is not None:
            try:
                map50_mask = float(seg.map50)
                map_mask = float(seg.map)
            except Exception:
                pass
        if np.isnan(map50_box) and box is not None:
            try:
                map50_box = float(box.map50)
                map_box = float(box.map)
            except Exception:
                pass

        # Per-class mAP50 masks
        per_class: list[float] = []
        if seg is not None and hasattr(seg, "maps"):
            try:
                per_class = [float(v) for v in seg.maps]
            except Exception:
                pass

        # n_images from speed dict or results_dict
        n_images = 0
        if hasattr(results, "results_dict"):
            rd = results.results_dict
            n_images = int(rd.get("n", 0))

        return SegMetrics(
            split_name=split_name,
            map50_mask=map50_mask,
            map_mask=map_mask,
            map50_box=map50_box,
            map_box=map_box,
            map50_per_class=per_class,
            class_names=self.class_names[: len(per_class)],
            n_images=n_images,
        )
