"""Exp 5.3 — Cross-weather per-condition evaluation of YOLOv8n-seg.

Reads test images from the main multi-weather dataset, groups them by the
weather/season prefix embedded in each filename, then runs YOLO val once per
group and prints a summary table.

Usage (from project root, Windows):
    python -m uav_nav.scripts.eval_cross_weather ^
        --weights runs/segment/experiments/20260525_110834_yolo/train/weights/best.pt ^
        --yolo_root data/yolo_v44 ^
        --device 0 ^
        --imgsz 640

Output:
    experiments/<date>_cross_weather/cross_weather_results.csv
    experiments/<date>_cross_weather/cross_weather_results.json
    Console table of per-condition mAP50 and per-class breakdown
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from loguru import logger


def _parse_condition(filename: str) -> str:
    """Extract the environment+weather prefix from a MidAir filename.

    e.g. "Kite_training_cloudy_trajectory_3027_000000.jpg" → "Kite_training_cloudy"
         "PLE_training_fall_trajectory_3000_000000.jpg"   → "PLE_training_fall"
    """
    # Match everything before '_trajectory_'
    m = re.match(r"^(.+?)_trajectory_", filename)
    if m:
        return m.group(1)
    return "unknown"


def _build_condition_dirs(
    images_test_dir: Path,
    labels_test_dir: Path,
    tmp_root: Path,
) -> dict[str, Path]:
    """Create per-condition subdirectories with symlinks to images and labels.

    Returns mapping {condition_name: yaml_dir} where each yaml_dir contains
    images/test/ and labels/test/ symlinks for that condition.
    """
    condition_map: dict[str, list[Path]] = {}
    for img_path in sorted(images_test_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        cond = _parse_condition(img_path.name)
        condition_map.setdefault(cond, []).append(img_path)

    cond_dirs: dict[str, Path] = {}
    for cond, img_paths in condition_map.items():
        cond_dir = tmp_root / cond
        img_out = cond_dir / "images" / "test"
        lbl_out = cond_dir / "labels" / "test"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in img_paths:
            dst_img = img_out / img_path.name
            if not dst_img.exists():
                try:
                    os.link(img_path.resolve(), dst_img)
                except (OSError, NotImplementedError):
                    try:
                        os.symlink(img_path.resolve(), dst_img)
                    except (OSError, NotImplementedError):
                        shutil.copy2(img_path, dst_img)

            lbl_name = img_path.stem + ".txt"
            src_lbl = labels_test_dir / lbl_name
            if src_lbl.exists():
                dst_lbl = lbl_out / lbl_name
                if not dst_lbl.exists():
                    try:
                        os.link(src_lbl.resolve(), dst_lbl)
                    except (OSError, NotImplementedError):
                        try:
                            os.symlink(src_lbl.resolve(), dst_lbl)
                        except (OSError, NotImplementedError):
                            shutil.copy2(src_lbl, dst_lbl)

        cond_dirs[cond] = cond_dir
    return cond_dirs


def _write_cond_yaml(cond_dir: Path, class_names: list[str]) -> Path:
    yaml_path = cond_dir / "midair_cond.yaml"
    lines = [
        f"path: {cond_dir.resolve()}",
        "test: images/test",
        "train: images/test",
        "val: images/test",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for name in class_names:
        lines.append(f"  - {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def _run_val(weights: Path, yaml_path: Path, device: str, imgsz: int, batch: int) -> dict:
    from ultralytics import YOLO  # type: ignore[import]

    model = YOLO(str(weights))
    results = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=imgsz,
        batch=batch,
        workers=0,
        conf=0.35,
        iou=0.45,
        device=device,
        verbose=False,
        plots=False,
    )

    seg = getattr(results, "seg", None)
    box = getattr(results, "box", None)

    map50_m = float("nan")
    map_m = float("nan")
    map50_b = float("nan")
    per_class: list[float] = []

    if seg is not None:
        try:
            map50_m = float(seg.map50)
            map_m = float(seg.map)
        except Exception:
            pass
        try:
            per_class = [float(v) for v in seg.maps]
        except Exception:
            pass
    if box is not None:
        try:
            map50_b = float(box.map50)
        except Exception:
            pass

    return {
        "map50_mask": map50_m,
        "map_mask": map_m,
        "map50_box": map50_b,
        "per_class_map50": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-weather YOLO evaluation (Exp 5.3)")
    parser.add_argument("--weights", required=True, type=Path, help="Path to best.pt")
    parser.add_argument("--yolo_root", required=True, type=Path, help="Root of multi-weather YOLO dataset")
    parser.add_argument("--device", default="0", help="CUDA device or 'cpu'")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--out_dir", default=None, type=Path)
    args = parser.parse_args()

    weights: Path = args.weights
    yolo_root: Path = args.yolo_root
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    images_test = yolo_root / "images" / "test"
    labels_test = yolo_root / "labels" / "test"
    if not images_test.exists():
        raise FileNotFoundError(f"Test images not found: {images_test}")

    # Read class names from yaml
    yaml_path = yolo_root / "midair.yaml"
    class_names: list[str] = []
    if yaml_path.exists():
        import re as _re
        text = yaml_path.read_text(encoding="utf-8")
        # Parse simple list under 'names:'
        in_names = False
        for line in text.splitlines():
            if line.strip().startswith("names:"):
                in_names = True
                continue
            if in_names:
                m = _re.match(r"\s*-\s*(\S+)", line)
                if m:
                    class_names.append(m.group(1))
                elif line.strip() and not line.strip().startswith("-"):
                    break
    if not class_names:
        class_names = ["river", "road", "rock", "tree"]

    out_dir = args.out_dir or Path("experiments") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cross_weather"
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="yolo_cw_"))
    logger.info("Temp dir: {}", tmp_root)

    try:
        logger.info("Building per-condition directories…")
        cond_dirs = _build_condition_dirs(images_test, labels_test, tmp_root)
        logger.info("Found {} conditions: {}", len(cond_dirs), sorted(cond_dirs))

        rows: list[dict] = []
        for cond in sorted(cond_dirs):
            cond_dir = cond_dirs[cond]
            n_imgs = len(list((cond_dir / "images" / "test").iterdir()))
            logger.info("Evaluating '{}' ({} images)…", cond, n_imgs)
            yaml = _write_cond_yaml(cond_dir, class_names)
            metrics = _run_val(weights, yaml, args.device, args.imgsz, args.batch)
            row: dict = {"condition": cond, "n_images": n_imgs}
            row["mAP50_mask"] = metrics["map50_mask"]
            row["mAP_mask"] = metrics["map_mask"]
            row["mAP50_box"] = metrics["map50_box"]
            for i, cls in enumerate(class_names):
                pc = metrics["per_class_map50"]
                row[f"mAP50_{cls}"] = pc[i] if i < len(pc) else float("nan")
            rows.append(row)
            logger.info("  mAP50-mask: {:.4f}", metrics["map50_mask"])

        # Save CSV
        csv_path = out_dir / "cross_weather_results.csv"
        if rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        logger.info("Saved CSV → {}", csv_path)

        # Save JSON
        json_path = out_dir / "cross_weather_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=str)
        logger.info("Saved JSON → {}", json_path)

        # Print summary table
        header = f"{'Condition':<30} {'N':>6} {'mAP50-M':>9} {'mAP-M':>7}"
        for cls in class_names:
            header += f" {cls[:8]:>9}"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))
        for row in rows:
            line = f"{row['condition']:<30} {row['n_images']:>6} {row['mAP50_mask']:>9.4f} {row['mAP_mask']:>7.4f}"
            for cls in class_names:
                v = row.get(f"mAP50_{cls}", float("nan"))
                line += f" {v:>9.4f}"
            print(line)
        print("=" * len(header))

        maps = [r["mAP50_mask"] for r in rows]
        import math
        valid = [v for v in maps if not math.isnan(v)]
        if valid:
            mean_v = sum(valid) / len(valid)
            std_v = (sum((v - mean_v) ** 2 for v in valid) / len(valid)) ** 0.5
            print(f"\nMean mAP50-mask: {mean_v:.4f} ± {std_v:.4f}")

        print(f"\nResults saved to: {out_dir}")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
