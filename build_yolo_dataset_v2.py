"""
YOLO dataset builder for MidAir.

Infrastructure (folder structure, zip-archive layout, splits, YAML)
from build_dataset_standalone.py.

Label-generation logic (mask_to_polygon / normalize_polygon, no watershed)
from build_yolo_dataset.py.

MidAir zip-archive layout expected:
  <condition>/sensor_records.zip          → sensor_records.hdf5 (trajectory list)
  <condition>/color_left/<traj>/frames.zip  → JPEG frames
  <condition>/segmentation/<traj>/frames.zip → segmentation masks

Split strategy: ALL 6 weather conditions contribute to every split.
Split is done PER TRAJECTORY within each condition:
  - first 80 % of sorted trajectories → train
  - next  10 %                        → val
  - last  10 %                        → test

Usage:
    python build_yolo_dataset_v2.py
    python build_yolo_dataset_v2.py --out data/yolo_v4
    python build_yolo_dataset_v2.py --out data/yolo_v4 --stride 15
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

# ── Config ────────────────────────────────────────────────────────────────────

MIDAIR_ROOT  = r"D:\data\MidAir"
OUT_DIR      = Path("data/yolo_v6")
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

MIN_AREA = 200
MIN_BOX  = 10
EPSILON  = 0.002  # Douglas-Peucker fraction

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
    "train": (0.0, 0.8),
    "val":   (0.8, 0.9),
    "test":  (0.9, 1.0),
}

# ── Image decoding ────────────────────────────────────────────────────────────

def decode_jpeg(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_seg(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

# ── Label logic (from build_yolo_dataset.py) ──────────────────────────────────

def mask_to_polygon(mask: np.ndarray) -> list:
    """Find external contours and simplify with Douglas-Peucker.

    Returns a list of (N, 2) int arrays — one per valid contour.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) < 6:
            continue
        eps = EPSILON * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        polygons.append(approx.reshape(-1, 2))
    return polygons


def normalize_polygon(poly: np.ndarray, w: int, h: int) -> np.ndarray:
    """Normalize polygon coordinates to [0, 1] and clip."""
    poly = poly.astype(float)
    poly[:, 0] = np.clip(poly[:, 0] / w, 0.0, 1.0)
    poly[:, 1] = np.clip(poly[:, 1] / h, 0.0, 1.0)
    return poly


def extract_labels(seg: np.ndarray, h: int, w: int) -> list:
    """Convert a segmentation mask to YOLO polygon label lines."""
    lines = []
    for raw_id, class_name in LABEL_MAP.items():
        yolo_cls = CLASS_TO_ID[class_name]
        mask = (seg == raw_id).astype(np.uint8)
        if mask.sum() == 0:
            continue

        n_comp, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for comp in range(1, n_comp):
            _, _, wb, hb, area = stats[comp]
            if area < MIN_AREA or wb < MIN_BOX or hb < MIN_BOX:
                continue

            comp_mask = (cc == comp).astype(np.uint8)
            polygons = mask_to_polygon(comp_mask)

            for poly in polygons:
                poly_norm = normalize_polygon(poly, w, h).reshape(-1)
                lines.append(
                    f"{yolo_cls} " + " ".join(f"{v:.6f}" for v in poly_norm)
                )
    return lines

# ── Open sensor_records.zip → HDF5 ───────────────────────────────────────────

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

# ── Per-condition processing ──────────────────────────────────────────────────

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

                    if written % 50 == 0:
                        print(f"    processed {written} frames so far")

            print(f"    {traj}: {len(indices)} frames sampled")
    return written

# ── YAML ──────────────────────────────────────────────────────────────────────

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

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",    default=str(OUT_DIR),    help="Output directory")
    parser.add_argument("--midair", default=MIDAIR_ROOT,     help="MidAir root directory")
    parser.add_argument("--stride", type=int, default=FRAME_STRIDE, help="Frame stride")
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
            n = process_condition(
                cond_path, split, out_dir, tag,
                stride=args.stride, traj_range=traj_range,
            )
            split_total += n
            print(f"  subtotal: {n}")
        print(f"  {split} total: {split_total}")

    write_yaml(out_dir)
    print(f"\nDone → {out_dir.resolve()}")
    print(
        f"Train: uav-nav-train-yolo "
        f"yolo.dataset_yaml={out_dir}/midair.yaml "
        f"yolo.device=cuda yolo.batch=64 yolo.workers=0"
    )


if __name__ == "__main__":
    main()
