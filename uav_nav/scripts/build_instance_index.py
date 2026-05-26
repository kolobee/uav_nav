"""Build instance index for embedding head training.

Parses YOLO segmentation labels, matches instances across consecutive frames
in the same trajectory via bounding-box IoU, and writes CSV tables for
triplet sampling.

Output layout::

    data/instance_index/
        train.csv          – all training detections with instance_id
        val.csv
        test.csv
        stats.json         – per-split / per-class instance counts

CSV columns:
    image_path, class_id, instance_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
    polygon_flat, frame_idx, condition, trajectory, split

Usage::

    python -m uav_nav.scripts.build_instance_index
    python -m uav_nav.scripts.build_instance_index --yolo_root data/yolo_v5 --iou_thresh 0.30
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from loguru import logger

# ── constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES: dict[int, str] = {0: "river", 1: "road", 2: "rock", 3: "tree"}
_STEM_RE = re.compile(r"^(.+)_(trajectory_\d+)_(\d+)$")
_SPLITS = ("train", "val", "test")
_CSV_FIELDS = [
    "image_path", "class_id", "instance_id",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "polygon_flat", "frame_idx", "condition", "trajectory", "split",
]


# ── label parsing ─────────────────────────────────────────────────────────────

def _parse_stem(stem: str) -> tuple[str, str, int] | tuple[None, None, None]:
    m = _STEM_RE.match(stem)
    if not m:
        return None, None, None
    return m.group(1), m.group(2), int(m.group(3))


def _parse_label(path: Path) -> list[tuple[int, str, float, float, float, float]]:
    """Return list of (class_id, polygon_flat, x1, y1, x2, y2) per instance."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        coords = list(map(float, parts[1:]))
        if len(coords) < 4 or len(coords) % 2 != 0:
            continue
        xs = coords[0::2]
        ys = coords[1::2]
        out.append((cid, " ".join(parts[1:]), min(xs), min(ys), max(xs), max(ys)))
    return out


# ── IoU matching ──────────────────────────────────────────────────────────────

def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def _greedy_match(
    prev: list[tuple[float, float, float, float, str]],  # (x1,y1,x2,y2, iid)
    curr_bboxes: list[tuple[float, float, float, float]],
    thresh: float,
) -> list[str | None]:
    """One-to-one IoU matching; returns instance_id or None for each curr bbox."""
    result: list[str | None] = [None] * len(curr_bboxes)
    candidates: list[tuple[float, int, int, str]] = []
    for ci, cb in enumerate(curr_bboxes):
        for pi, (*pbbox, pid) in enumerate(prev):
            iou = _iou(cb, tuple(pbbox))  # type: ignore[arg-type]
            if iou >= thresh:
                candidates.append((-iou, ci, pi, pid))
    candidates.sort()
    used_c: set[int] = set()
    used_p: set[int] = set()
    for _, ci, pi, pid in candidates:
        if ci not in used_c and pi not in used_p:
            result[ci] = pid
            used_c.add(ci)
            used_p.add(pi)
    return result


# ── trajectory processing ─────────────────────────────────────────────────────

def _process_trajectory(
    frames: list[tuple[Path, Path]],
    condition: str,
    trajectory: str,
    split: str,
    counter: list[int],
    iou_thresh: float,
) -> list[dict]:
    """Assign instance_ids to all detections in one trajectory.

    Args:
        frames: (img_path, label_path) pairs sorted by frame_idx.
        condition: Environment condition string (e.g. ``Kite_training_cloudy``).
        trajectory: Trajectory string (e.g. ``trajectory_3000``).
        split: Dataset split name.
        counter: Single-element list used as a mutable global id counter.
        iou_thresh: Minimum IoU to consider the same instance across frames.

    Returns:
        List of record dicts, one per detection.
    """
    records: list[dict] = []
    # class_id → [(x1,y1,x2,y2, instance_id)] for previous frame
    prev_by_class: dict[int, list[tuple]] = defaultdict(list)

    for img_path, label_path in frames:
        frame_idx = int(img_path.stem.split("_")[-1])
        raw = _parse_label(label_path)

        # Group current detections by class
        curr_by_class: dict[int, list[tuple]] = defaultdict(list)
        for cid, polygon, x1, y1, x2, y2 in raw:
            curr_by_class[cid].append((cid, polygon, x1, y1, x2, y2))

        for class_id, instances in curr_by_class.items():
            curr_bboxes = [(x1, y1, x2, y2) for _, _, x1, y1, x2, y2 in instances]
            matched_ids = _greedy_match(prev_by_class[class_id], curr_bboxes, iou_thresh)

            new_prev: list[tuple] = []
            for idx, (cid, polygon, x1, y1, x2, y2) in enumerate(instances):
                iid = matched_ids[idx]
                if iid is None:
                    iid = f"{condition}_{trajectory}_{CLASS_NAMES[cid]}_{counter[0]:07d}"
                    counter[0] += 1
                new_prev.append((x1, y1, x2, y2, iid))
                records.append({
                    "image_path": str(img_path),
                    "class_id": cid,
                    "instance_id": iid,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "polygon_flat": polygon,
                    "frame_idx": frame_idx,
                    "condition": condition,
                    "trajectory": trajectory,
                    "split": split,
                })
            prev_by_class[class_id] = new_prev

    return records


# ── split processing ──────────────────────────────────────────────────────────

def _process_split(
    images_dir: Path,
    labels_dir: Path,
    split: str,
    counter: list[int],
    iou_thresh: float,
) -> list[dict]:
    """Process one dataset split and return all records."""
    img_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    logger.info("  {} images in {} split", len(img_paths), split)

    # Group by (condition, trajectory)
    groups: dict[tuple[str, str], list[tuple[Path, Path]]] = defaultdict(list)
    skipped = 0
    for img_path in img_paths:
        condition, trajectory, _ = _parse_stem(img_path.stem)
        if condition is None:
            skipped += 1
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        groups[(condition, trajectory)].append((img_path, label_path))

    if skipped:
        logger.warning("  {} files did not match naming pattern", skipped)

    # Sort frames within each trajectory by frame_idx
    for key in groups:
        groups[key].sort(key=lambda pair: int(pair[0].stem.split("_")[-1]))

    all_records: list[dict] = []
    for (condition, trajectory), frames in sorted(groups.items()):
        recs = _process_trajectory(frames, condition, trajectory, split, counter, iou_thresh)
        all_records.extend(recs)

    return all_records


# ── stats ────────────────────────────────────────────────────────────────────

def _compute_stats(records: list[dict], split: str) -> dict:
    iid_set: set[str] = set()
    class_counts: dict[int, int] = defaultdict(int)
    class_iids: dict[int, set[str]] = defaultdict(set)
    for r in records:
        iid_set.add(r["instance_id"])
        class_counts[r["class_id"]] += 1
        class_iids[r["class_id"]].add(r["instance_id"])

    # Count instances with ≥ 2 observations (usable for triplets)
    iid_obs: dict[str, int] = defaultdict(int)
    for r in records:
        iid_obs[r["instance_id"]] += 1
    multi_obs = sum(1 for cnt in iid_obs.values() if cnt >= 2)

    return {
        "split": split,
        "total_detections": len(records),
        "total_instances": len(iid_set),
        "instances_with_2plus_observations": multi_obs,
        "per_class": {
            CLASS_NAMES.get(cid, str(cid)): {
                "detections": class_counts[cid],
                "instances": len(class_iids[cid]),
            }
            for cid in sorted(class_counts)
        },
    }


# ── I/O ───────────────────────────────────────────────────────────────────────

def _write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build instance index CSV for embedding training")
    parser.add_argument("--yolo_root", default="data/yolo_v5", help="Root of YOLO dataset")
    parser.add_argument("--out_dir", default="data/instance_index", help="Output directory")
    parser.add_argument("--iou_thresh", type=float, default=0.30,
                        help="Min bbox IoU to match instance across frames")
    args = parser.parse_args()

    yolo_root = Path(args.yolo_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("YOLO root: {}", yolo_root)
    logger.info("Output:    {}", out_dir)
    logger.info("IoU threshold: {}", args.iou_thresh)

    counter = [0]  # global instance counter
    all_stats: list[dict] = []

    for split in _SPLITS:
        images_dir = yolo_root / "images" / split
        labels_dir = yolo_root / "labels" / split
        if not images_dir.exists():
            logger.warning("Split '{}' not found, skipping", split)
            continue

        logger.info("Processing split: {}", split)
        records = _process_split(images_dir, labels_dir, split, counter, args.iou_thresh)

        csv_path = out_dir / f"{split}.csv"
        _write_csv(records, csv_path)
        logger.success("  Wrote {} rows → {}", len(records), csv_path)

        stats = _compute_stats(records, split)
        all_stats.append(stats)
        logger.info(
            "  Instances: {} total, {} with ≥2 obs (usable for triplets)",
            stats["total_instances"],
            stats["instances_with_2plus_observations"],
        )

    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False))
    logger.success("Stats → {}", stats_path)
    logger.info("Total unique instance IDs assigned: {}", counter[0])


if __name__ == "__main__":
    main()
