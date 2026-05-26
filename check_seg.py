import zipfile
import glob
import numpy as np
import cv2
from pathlib import Path

label_map = {
    0: "background", 1: "tree", 2: "tree", 3: "tree",
    4: "rock", 5: "rock", 6: "rock",
    7: "river", 8: "river",
    9: "background", 10: "background", 11: "tree", 12: "rock", 13: "river", 14: "background"
}

def check_dataset(pattern, title):
    zips = glob.glob(pattern)
    if not zips:
        print(f"{title}: no zips found for {pattern}")
        return
    print(f"\n=== {title} ({len(zips)} trajectories) ===")
    total_counts = {}
    total_pixels = 0
    for zip_path in zips[:5]:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            raw = zf.read(names[0])
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        vals, counts = np.unique(img, return_counts=True)
        for v, c in zip(vals, counts):
            total_counts[int(v)] = total_counts.get(int(v), 0) + int(c)
        total_pixels += img.size

    print("Raw pixel ID breakdown:")
    for raw_id, count in sorted(total_counts.items()):
        name = label_map.get(raw_id, f"UNKNOWN_{raw_id}")
        print(f"  ID {raw_id:3d} → {name:15s}: {100*count/total_pixels:.1f}%")

check_dataset("D:/data/MidAir/Kite_training/cloudy/segmentation/*/frames.zip", "Kite_training/cloudy")
check_dataset("D:/data/MidAir/Kite_training/sunny/segmentation/*/frames.zip",  "Kite_training/sunny")
check_dataset("D:/data/MidAir/PLE_training/fall/segmentation/*/frames.zip",    "PLE_training/fall")
check_dataset("D:/data/MidAir/PLE_training/spring/segmentation/*/frames.zip",  "PLE_training/spring")
