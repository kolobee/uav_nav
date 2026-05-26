"""
Standalone YOLO dataset builder for MidAir.

Based on copyFromVDS/build_yolo_dataset.py — rewritten for the actual
MidAir zip-archive layout:
  - sensor_records.zip  → contains sensor_records.hdf5 (trajectory list)
  - color_left/trajectory_XXXX/frames.zip  → JPEG frames
  - segmentation/trajectory_XXXX/frames.zip → segmentation masks

Split strategy: ALL 6 weather conditions contribute to every split.
Split is done PER TRAJECTORY within each condition:
  - first TRAIN_FRAC  (80 %) of sorted trajectories → train
  - remaining         (20 %) of sorted trajectories → val & test

Usage:
    python build_dataset_standalone.py
    python build_dataset_standalone.py --out data/yolo_v4
    python build_dataset_standalone.py --out data/yolo_v4 --stride 15
"""

import argparse
import io
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml
from PIL import Image

# ── Config ──────────────��───────────────────────────────���─────────────────────

MIDAIR_ROOT  = r"D:\data\MidAir"
OUT_DIR      = Path("data/yolo_v5")
TARGET_H     = 640
TARGET_W     = 640
FRAME_STRIDE = 15
CAMERA_DIR   = "color_left"
TRAIN_FRAC   = 0.8   # первые 80 % траекторий → train, остальные → val/test

LABEL_MAP = {
    2:  "tree",
    6:  "rock",
    7:  "river",
    8:  "river",
    10: "road",
    11: "road",
}

CLASS_NAMES = sorted(set(LABEL_MAP.values()))  # river=0, road=1, rock=2, tree=3
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

MIN_AREA = 400
MIN_BOX  = 15
EPSILON  = 0.002  # Douglas-Peucker fraction (same as copyFromVDS original)

# Для этих классов применяем watershed-разделение слившихся blob'ов.
# Реки и дороги — протяжённые объекты, их делить не нужно.
WATERSHED_CLASSES = {"tree"}

ALL_CONDITIONS = [
    ("Kite_training", "cloudy"),
    ("Kite_training", "sunny"),
    ("Kite_training", "foggy"),
    ("PLE_training",  "fall"),
    ("PLE_training",  "spring"),
    ("PLE_training",  "winter"),
]

# Per-split trajectory slice: (start_frac, end_frac)
SPLIT_RANGES = {
    "train": (0.0,        0.8),
    "val":   (0.8,        0.9),
    "test":  (0.9,        1.0),
}

# ─�� Image decoding ─────────────────────────────────────────────��──────────────

def decode_jpeg(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_seg(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

# ── Watershed splitting ───────────────────────────────────────────────────────

def split_with_watershed(comp_mask):

    kernel = np.ones((3, 3), np.uint8)

    # немного раздвигаем объекты
    eroded = cv2.erode(comp_mask, kernel, iterations=1)

    dist = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)

    dist_max = dist.max()

    if dist_max < 3:
        return [comp_mask]

    # более агрессивное разделение
    _, sure_fg = cv2.threshold(
        dist,
        0.45 * dist_max,
        255,
        cv2.THRESH_BINARY
    )

    sure_fg = sure_fg.astype(np.uint8)

    # ищем центры объектов
    n_markers, markers = cv2.connectedComponents(
        sure_fg,
        connectivity=4
    )

    if n_markers <= 2:
        return [comp_mask]

    unknown = cv2.subtract(comp_mask, sure_fg)

    markers = markers + 1
    markers[unknown == 1] = 0

    color = np.stack([comp_mask * 255] * 3, axis=-1)

    markers = cv2.watershed(color, markers)

    result_masks = []

    for m in range(2, n_markers + 1):

        obj = (markers == m).astype(np.uint8)

        if obj.sum() < MIN_AREA:
            continue

        result_masks.append(obj)

    if len(result_masks) == 0:
        return [comp_mask]

    return result_masks

# ── Label extraction (logic from copyFromVDS/build_yolo_dataset.py) ───────────

def _poly_lines(sub_mask: np.ndarray, yolo_cls: int, h: int, w: int) -> list:
    """Convert one binary mask to YOLO polygon label lines."""
    contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if len(cnt) < 6:
            continue
        eps = EPSILON * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2).astype(float)
        pts[:, 0] = np.clip(pts[:, 0] / w, 0.0, 1.0)
        pts[:, 1] = np.clip(pts[:, 1] / h, 0.0, 1.0)
        lines.append(f"{yolo_cls} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in pts))
    return lines


def extract_labels(seg: np.ndarray, h: int, w: int) -> list:
    lines = []
    for raw_id, class_name in LABEL_MAP.items():
        yolo_cls = CLASS_TO_ID[class_name]
        mask = (seg == raw_id).astype(np.uint8)
        if class_name == "tree":
            kernel = np.ones((3, 3), np.uint8)

            mask = cv2.erode(mask, kernel, iterations=1)
        if mask.sum() == 0:
            continue
        n_comp, cc, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=4
        )
        for comp in range(1, n_comp):
            _, _, wb, hb, area = stats[comp]
            if area < MIN_AREA or wb < MIN_BOX or hb < MIN_BOX:
                continue
            comp_mask = (cc == comp).astype(np.uint8)

            if class_name in WATERSHED_CLASSES:
                sub_masks = split_with_watershed(comp_mask)
            else:
                sub_masks = [comp_mask]

            for sub in sub_masks:
                lines.extend(_poly_lines(sub, yolo_cls, h, w))
    return lines

# ── Open sensor_records.zip → HDF5 ���───────────────────��──────────────────────

def open_hdf5(zip_path: Path) -> h5py.File:
    """Extract HDF5 from sensor_records.zip and return open h5py.File."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            hdf_name = next(
                (n for n in zf.namelist() if n.endswith((".hdf5", ".hdf", ".h5"))),
                None,
            )
            if hdf_name is not None:
                return h5py.File(io.BytesIO(zf.read(hdf_name)), "r")
    except zipfile.BadZipFile:
        pass
    # Some distributions ship a bare HDF5 with .zip extension
    return h5py.File(zip_path, "r")

# ── Per-condition processing ────────────��─────────────────────────────────────

def process_condition(
    cond_path: Path,
    split: str,
    out_dir: Path,
    tag: str,
    stride: int,
    traj_range: tuple = (0.0, 1.0),
) -> int:
    sensor_zip = cond_path / "sensor_records.zip"
    if not sensor_zip.exists():
        print(f"  [SKIP] not found: {sensor_zip}")
        return 0

    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with open_hdf5(sensor_zip) as hdf:
        all_trajs = sorted(k for k in hdf.keys() if k.startswith("trajectory_"))
        n = len(all_trajs)
        start = int(n * traj_range[0])
        end   = int(n * traj_range[1])
        trajs = all_trajs[start:end]
        print(f"    trajectories {start}–{end-1} of {n} (range {traj_range})")
        for traj in trajs:
            # Frame count from any camera_data dataset
            cam_grp = hdf[f"{traj}/camera_data"]
            ref_key = next(iter(cam_grp.keys()))
            n_frames = int(cam_grp[ref_key].shape[0])

            frames_zip_rgb = cond_path / CAMERA_DIR / traj / "frames.zip"
            frames_zip_seg = cond_path / "segmentation" / traj / "frames.zip"

            if not frames_zip_rgb.exists():
                print(f"    [SKIP] missing rgb zip: {frames_zip_rgb}")
                continue
            if not frames_zip_seg.exists():
                print(f"    [SKIP] missing seg zip: {frames_zip_seg}")
                continue

            indices = list(range(0, n_frames, stride))

            with zipfile.ZipFile(frames_zip_rgb, "r") as zf_rgb, \
                 zipfile.ZipFile(frames_zip_seg, "r") as zf_seg:

                # Detect segmentation extension from first file in archive
                seg_ext = Path(zf_seg.namelist()[0]).suffix  # .JPEG or .PNG

                for idx in indices:
                    fname_rgb = f"{idx:06d}.JPEG"
                    fname_seg = f"{idx:06d}{seg_ext}"

                    try:
                        rgb = decode_jpeg(zf_rgb.read(fname_rgb))
                        seg = decode_seg(zf_seg.read(fname_seg))
                    except KeyError:
                        continue  # frame missing in archive

                    if rgb.shape[:2] != (TARGET_H, TARGET_W):
                        rgb = cv2.resize(rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
                        seg = cv2.resize(seg, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)

                    stem = f"{tag}_{traj}_{idx:06d}"

                    cv2.imwrite(
                        str(img_dir / f"{stem}.jpg"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    )

                    labels = extract_labels(seg, TARGET_H, TARGET_W)
                    (lbl_dir / f"{stem}.txt").write_text("\n".join(labels), encoding="utf-8")
                    written += 1

            print(f"    {traj}: {len(indices)} frames")
    return written

# ── YAML ──────────────���──────────────────────────────────��────────────────────

def write_yaml(out_dir: Path):
    data = {
        "path":  str(out_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(out_dir / "midair.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    print(f"YAML written: {out_dir / 'midair.yaml'}")

# ── Entry point ────────────���──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",    default=str(OUT_DIR))
    parser.add_argument("--midair", default=MIDAIR_ROOT)
    parser.add_argument("--stride", type=int, default=FRAME_STRIDE)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, traj_range in SPLIT_RANGES.items():
        print(f"\n=== {split} (trajs {int(traj_range[0]*100)}%–{int(traj_range[1]*100)}%) ===")
        split_total = 0
        for dataset, condition in ALL_CONDITIONS:
            cond_path = Path(args.midair) / dataset / condition
            tag = f"{dataset}_{condition}"
            print(f"  {dataset}/{condition}")
            n = process_condition(cond_path, split, out_dir, tag,
                                  stride=args.stride, traj_range=traj_range)
            split_total += n
            print(f"  subtotal: {n}")
        print(f"  {split} total: {split_total}")

    write_yaml(out_dir)
    print(f"\nDone → {out_dir.resolve()}")
    print(f"Train: uav-nav-train-yolo yolo.dataset_yaml={out_dir}/midair.yaml yolo.device=cuda yolo.batch=64 yolo.workers=0")


if __name__ == "__main__":
    main()
