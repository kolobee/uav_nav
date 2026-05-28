"""Dataset builder for constructing training/evaluation splits.

Reads raw MidAir data via MidAirLoader and produces:
  - Processed HDF5 files for embedding-head training and VIO evaluation.
  - YOLO segmentation dataset (images + polygon labels) for YOLOv8n-seg.

Output HDF5 layout per split::

    images        (N, H, W, 3) uint8
    segmentation  (N, H, W)    uint8   (remapped class IDs)
    depth         (N, H, W)    float32 (metres)
    poses         (N, 4, 4)    float64 (SE3)
    timestamps    (N,)         float64
    frame_indices (N,)         int32
    sequence_ids  (N,)         bytes   (variable-length UTF-8)

YOLO output layout::

    yolo_root/
    ├── images/{train,val,test}/*.jpg
    ├── labels/{train,val,test}/*.txt   (YOLOv8-seg polygon format)
    └── midair.yaml
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import yaml

from uav_nav.data.calibration import CameraCalibration
from uav_nav.data.midair_loader import MidAirLoader, MidAirSample


@dataclass
class SplitConfig:
    """Configuration for a dataset split."""

    name: str
    sequences: list[str]
    max_samples: Optional[int] = None
    shuffle: bool = True
    random_seed: int = 42


class DatasetBuilder:
    """Constructs processed HDF5 dataset files from raw MidAir data.

    Args:
        loader: Configured MidAirLoader over the raw data.
        calibration: Camera calibration (used for undistortion if needed).
        output_dir: Directory where output HDF5 files will be written.
        label_map: Mapping from raw MidAir class IDs to semantic class names.
        target_size: Output image size as (height, width). Default (480, 640).
    """

    def __init__(
        self,
        loader: MidAirLoader,
        calibration: CameraCalibration,
        output_dir: Path,
        label_map: dict[int, str],
        target_size: tuple[int, int] = (480, 640),
    ) -> None:
        self.loader = loader
        self.calibration = calibration
        self.output_dir = Path(output_dir)
        self.label_map = label_map
        self.target_size = target_size

        # Reverse map: class name → compact 0-based class ID
        unique_classes = sorted(set(label_map.values()))
        self._class_to_id: dict[str, int] = {c: i for i, c in enumerate(unique_classes)}
        # raw_id → compact_id lookup array (size = max raw ID + 1)
        max_raw = max(label_map.keys()) if label_map else 0
        self._remap_table = np.zeros(max_raw + 1, dtype=np.uint8)
        for raw_id, class_name in label_map.items():
            self._remap_table[raw_id] = self._class_to_id[class_name]

    # ── public API ────────────────────────────────────────────────────────────

    def build_split(self, config: SplitConfig) -> Path:
        """Build and write an HDF5 file for a single dataset split.

        Returns:
            Path to the written HDF5 file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"{config.name}.hdf5"

        # Collect sample refs matching this split's sequences
        matching_refs = [
            ref for ref in self.loader._refs
            if any(ref.sequence_id.startswith(seq) for seq in config.sequences)
        ]

        if config.shuffle:
            rng = np.random.default_rng(config.random_seed)
            indices = rng.permutation(len(matching_refs)).tolist()
        else:
            indices = list(range(len(matching_refs)))

        if config.max_samples is not None:
            indices = indices[: config.max_samples]

        n = len(indices)
        if n == 0:
            raise ValueError(
                f"No samples found for split '{config.name}' "
                f"with sequences {config.sequences}"
            )

        H, W = self.target_size
        vlen_str = h5py.special_dtype(vlen=str)

        with h5py.File(out_path, "w") as f:
            ds_images = f.create_dataset(
                "images", shape=(n, H, W, 3), dtype=np.uint8,
                chunks=(1, H, W, 3), compression="gzip", compression_opts=4,
            )
            ds_seg = f.create_dataset(
                "segmentation", shape=(n, H, W), dtype=np.uint8,
                chunks=(1, H, W), compression="gzip", compression_opts=4,
            )
            ds_depth = f.create_dataset(
                "depth", shape=(n, H, W), dtype=np.float32,
                chunks=(1, H, W), compression="gzip", compression_opts=4,
            )
            ds_poses = f.create_dataset(
                "poses", shape=(n, 4, 4), dtype=np.float64,
            )
            ds_ts = f.create_dataset("timestamps", shape=(n,), dtype=np.float64)
            ds_fi = f.create_dataset("frame_indices", shape=(n,), dtype=np.int32)
            ds_sid = f.create_dataset("sequence_ids", shape=(n,), dtype=vlen_str)

            for write_idx, src_idx in enumerate(indices):
                ref = matching_refs[src_idx]
                sample = self.loader._load_sample(ref)

                img = self._undistort_image(sample.image_rgb)

                seg = np.zeros((H, W), dtype=np.uint8)
                if sample.segmentation is not None:
                    seg = self._remap_labels(
                        cv2.resize(
                            sample.segmentation, (W, H),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    )

                depth = np.zeros((H, W), dtype=np.float32)
                if sample.depth is not None:
                    depth = cv2.resize(
                        sample.depth, (W, H),
                        interpolation=cv2.INTER_LINEAR,
                    )

                ds_images[write_idx] = img
                ds_seg[write_idx] = seg
                ds_depth[write_idx] = depth
                ds_poses[write_idx] = (
                    sample.pose_gt if sample.pose_gt is not None
                    else np.eye(4, dtype=np.float64)
                )
                ds_ts[write_idx] = sample.timestamp
                ds_fi[write_idx] = sample.frame_idx
                ds_sid[write_idx] = sample.sequence_id

            # Store class map metadata
            f.attrs["class_to_id"] = json.dumps(self._class_to_id)
            f.attrs["target_height"] = H
            f.attrs["target_width"] = W
            f.attrs["n_samples"] = n

        return out_path

    def build_all(
        self,
        train_sequences: list[str],
        val_sequences: list[str],
        test_sequences: list[str],
    ) -> dict[str, Path]:
        """Build train, val, and test HDF5 splits."""
        results: dict[str, Path] = {}
        for name, seqs in [
            ("train", train_sequences),
            ("val", val_sequences),
            ("test", test_sequences),
        ]:
            cfg = SplitConfig(name=name, sequences=seqs)
            results[name] = self.build_split(cfg)
        return results

    def _undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Undistort and resize an image to target_size.

        MidAir cameras have zero distortion (pinhole model), so this only
        performs a resize.
        """
        H, W = self.target_size
        src_h, src_w = image.shape[:2]
        if src_h == H and src_w == W:
            return image
        return cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)

    def _remap_labels(self, mask: np.ndarray) -> np.ndarray:
        """Remap raw MidAir class IDs to compact 0-based semantic indices."""
        flat = mask.ravel()
        # Clip to known range before lookup
        clamped = np.clip(flat, 0, len(self._remap_table) - 1)
        return self._remap_table[clamped].reshape(mask.shape)

    # ── YOLO export ───────────────────────────────────────────────────────────

    def export_yolo_split(
        self,
        config: SplitConfig,
        yolo_root: Path,
        min_component_area: int = 400,
        polygon_epsilon: float = 0.002,
    ) -> Path:
        """Export a split to YOLOv8-seg polygon format.

        Writes JPEG images and .txt label files (one polygon per instance,
        background class is skipped). Connected components separate semantic
        regions into individual instances.

        Args:
            config: Split configuration (name, sequences, shuffle, seed).
            yolo_root: Root directory for the YOLO dataset tree.
            min_component_area: Minimum pixel area to include a component.
            polygon_epsilon: Douglas-Peucker epsilon as fraction of arc length.

        Returns:
            Path to yolo_root (for chaining).

        Raises:
            RuntimeError: If loader was created with load_segmentation=False.
        """
        if not self.loader.load_segmentation:
            raise RuntimeError(
                "MidAirLoader must be created with load_segmentation=True for YOLO export."
            )

        img_dir = yolo_root / "images" / config.name
        lbl_dir = yolo_root / "labels" / config.name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        matching_refs = [
            ref for ref in self.loader._refs
            if any(ref.sequence_id.startswith(seq) for seq in config.sequences)
        ]

        if config.shuffle:
            rng = np.random.default_rng(config.random_seed)
            indices = rng.permutation(len(matching_refs)).tolist()
        else:
            indices = list(range(len(matching_refs)))

        if config.max_samples is not None:
            indices = indices[: config.max_samples]

        H, W = self.target_size

        for src_idx in indices:
            ref = matching_refs[src_idx]
            sample = self.loader._load_sample(ref)

            img = self._undistort_image(sample.image_rgb)  # (H, W, 3) RGB

            # Build a filesystem-safe unique stem
            stem = ref.sequence_id.replace("/", "_") + f"_{ref.frame_idx:06d}"

            # Write JPEG (cv2 expects BGR)
            cv2.imwrite(
                str(img_dir / f"{stem}.jpg"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            # Write label file (empty if no segmentation available)
            labels: list[str] = []
            if sample.segmentation is not None:
                seg_r = cv2.resize(
                    sample.segmentation, (W, H), interpolation=cv2.INTER_NEAREST
                )
                labels = self._extract_yolo_labels(
                    seg_r, H, W, min_component_area, polygon_epsilon
                )

            (lbl_dir / f"{stem}.txt").write_text("\n".join(labels), encoding="utf-8")

        return yolo_root

    def export_yolo_all(
        self,
        train_sequences: list[str],
        val_sequences: list[str],
        test_sequences: list[str],
        yolo_root: Path,
        min_component_area: int = 400,
        polygon_epsilon: float = 0.002,
    ) -> Path:
        """Export train / val / test splits and write midair.yaml.

        Returns:
            Path to yolo_root.
        """
        for name, seqs in [
            ("train", train_sequences),
            ("val", val_sequences),
            ("test", test_sequences),
        ]:
            cfg = SplitConfig(name=name, sequences=seqs)
            self.export_yolo_split(
                cfg, yolo_root,
                min_component_area=min_component_area,
                polygon_epsilon=polygon_epsilon,
            )

        self._write_yolo_yaml(yolo_root)
        return yolo_root

    # Classes where adjacent instances can merge — split with watershed.
    # Rivers and roads are contiguous by nature and should not be split.
    _WATERSHED_CLASSES: frozenset[str] = frozenset({"tree", "rock"})

    @staticmethod
    def _split_with_watershed(comp_mask: np.ndarray, min_area: int) -> list[np.ndarray]:
        """Split a merged blob into individuals via distance transform + watershed.

        Returns a list of sub-masks. If only one peak is found the original
        mask is returned unchanged as a single-element list.
        """
        dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
        dist_max = float(dist.max())
        if dist_max == 0:
            return [comp_mask]

        _, sure_fg = cv2.threshold(dist, 0.35 * dist_max, 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)

        n_markers, markers = cv2.connectedComponents(sure_fg)
        if n_markers <= 2:          # background + one peak → nothing to split
            return [comp_mask]

        bgr = cv2.cvtColor(comp_mask * 255, cv2.COLOR_GRAY2BGR)
        markers_ws = cv2.watershed(bgr, markers.astype(np.int32))

        sub_masks = []
        for lbl in range(1, n_markers):
            sub = (markers_ws == lbl).astype(np.uint8) & comp_mask  # не выходим за исходный blob
            if sub.sum() >= min_area:
                sub_masks.append(sub)

        return sub_masks if sub_masks else [comp_mask]

    def _extract_yolo_labels(
        self,
        seg_raw: np.ndarray,
        img_h: int,
        img_w: int,
        min_area: int,
        epsilon_frac: float,
    ) -> list[str]:
        """Convert a raw segmentation mask to YOLO polygon label lines.

        Iterates over each raw MidAir ID separately so natural pixel
        boundaries between IDs act as instance separators. For tree/rock
        classes a watershed pass splits merged blobs into individuals.

        Each connected component becomes one label line:
        ``<yolo_class> x1 y1 x2 y2 ... xN yN`` (coords in [0, 1]).
        """
        labels: list[str] = []

        for raw_id, class_name in self.label_map.items():
            if class_name == "background":
                continue

            compact_id = self._class_to_id[class_name]
            yolo_class = compact_id - 1  # YOLO is 0-indexed (no background slot)

            raw_mask = (seg_raw == raw_id).astype(np.uint8)
            if raw_mask.sum() == 0:
                continue

            n_comp, label_map_cc, stats, _ = cv2.connectedComponentsWithStats(
                raw_mask, connectivity=8
            )

            for comp_id in range(1, n_comp):
                bx, by, w_box, h_box, area = stats[comp_id]
                if area < min_area:
                    continue
                if w_box < 15 or h_box < 15:
                    continue

                comp_mask = (label_map_cc == comp_id).astype(np.uint8)

                if class_name in self._WATERSHED_CLASSES:
                    sub_masks = self._split_with_watershed(comp_mask, min_area)
                else:
                    sub_masks = [comp_mask]

                for sub_mask in sub_masks:
                    contours, _ = cv2.findContours(
                        sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if not contours:
                        continue
                    for contour in contours:
                        if len(contour) < 3:
                            continue
                        arc = cv2.arcLength(contour, closed=True)
                        if arc == 0:
                            continue
                        simplified = cv2.approxPolyDP(
                            contour, epsilon_frac * arc, closed=True
                        )
                        if len(simplified) < 3:
                            continue
                        pts = simplified.reshape(-1, 2).astype(np.float32)
                        pts[:, 0] /= img_w
                        pts[:, 1] /= img_h
                        pts = pts.clip(0.0, 1.0)
                        coord_str = " ".join(f"{px:.6f} {py:.6f}" for px, py in pts)
                        labels.append(f"{yolo_class} {coord_str}")

        return labels

    def _write_yolo_yaml(self, yolo_root: Path) -> None:
        """Write midair.yaml for ultralytics training."""
        # Ordered class names (compact_id 1, 2, 3 … → yolo 0, 1, 2 …)
        id_to_name = {v: k for k, v in self._class_to_id.items()}
        class_names = [
            id_to_name[i] for i in sorted(id_to_name)
            if id_to_name[i] != "background"
        ]

        data = {
            "path": str(yolo_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(class_names),
            "names": class_names,
        }

        with open(yolo_root / "midair.yaml", "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def compute_statistics(self, split_path: Path) -> dict[str, float]:
        """Compute per-channel mean and std over all images in a split.

        Returns:
            Dict with keys mean_r, mean_g, mean_b, std_r, std_g, std_b.
        """
        with h5py.File(split_path, "r") as f:
            images = f["images"]  # (N, H, W, 3) uint8
            n = images.shape[0]

            # Online mean/variance per channel (one image at a time to bound RAM)
            mean = np.zeros(3, dtype=np.float64)
            M2 = np.zeros(3, dtype=np.float64)
            count = 0

            for i in range(n):
                img = images[i].astype(np.float64)  # (H, W, 3)
                pixels = img.reshape(-1, 3)          # (H*W, 3)
                batch = pixels.shape[0]
                batch_mean = pixels.mean(axis=0)
                batch_var = pixels.var(axis=0)
                # Chan parallel combine formula
                delta = batch_mean - mean
                new_count = count + batch
                mean += delta * batch / new_count
                M2 += batch_var * batch + delta ** 2 * count * batch / new_count
                count = new_count

        std = np.sqrt(M2 / max(count - 1, 1))
        return {
            "mean_r": float(mean[0]),
            "mean_g": float(mean[1]),
            "mean_b": float(mean[2]),
            "std_r": float(std[0]),
            "std_g": float(std[1]),
            "std_b": float(std[2]),
        }
