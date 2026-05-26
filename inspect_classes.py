"""
Для каждого raw class ID создаёт папку с примерами кадров,
где выделен только этот класс. Запускай и смотри — что реально
попало в каждый ID, чтобы сделать правильный label_map.
"""
import zipfile
import numpy as np
import cv2
from pathlib import Path

STEP = 300         # каждый N-й кадр
MAX_TRAJ = None    # None = все траектории
MAX_FRAMES_PER_CLASS = 6   # максимум кадров на класс на условие
OUT_DIR = Path("viz_classes")

DATASETS = [
    ("Kite", "D:/data/MidAir/Kite_training/cloudy"),
    ("PLE",  "D:/data/MidAir/PLE_training/fall"),
]


def decode(raw: bytes, flag=cv2.IMREAD_COLOR):
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, flag)


def process_condition(label, cond_path):
    cond = Path(cond_path)
    all_traj_dirs = sorted((cond / "color_left").iterdir())
    traj_dirs = all_traj_dirs if MAX_TRAJ is None else all_traj_dirs[:MAX_TRAJ]

    saved_counts = {}  # class_id -> int

    for traj_dir in traj_dirs:
        traj = traj_dir.name
        rgb_zip = cond / "color_left"  / traj / "frames.zip"
        seg_zip = cond / "segmentation" / traj / "frames.zip"
        if not rgb_zip.exists() or not seg_zip.exists():
            continue

        with zipfile.ZipFile(rgb_zip, "r") as zf_rgb, \
             zipfile.ZipFile(seg_zip, "r") as zf_seg:
            rgb_names = sorted(zf_rgb.namelist())
            seg_names = sorted(zf_seg.namelist())
            n = min(len(rgb_names), len(seg_names))

            # Равномерно берём до MAX_FRAMES_PER_CLASS кадров из траектории
            indices = list(range(0, n, STEP))[:MAX_FRAMES_PER_CLASS]

            for i in indices:
                rgb = decode(zf_rgb.read(rgb_names[i]), cv2.IMREAD_COLOR)
                seg = decode(zf_seg.read(seg_names[i]), cv2.IMREAD_GRAYSCALE)

                for uid in np.unique(seg):
                    class_id = int(uid)
                    out = OUT_DIR / label / f"class_{class_id:02d}"
                    out.mkdir(parents=True, exist_ok=True)

                    # Затемняем фон, подсвечиваем класс
                    vis = (rgb * 0.2).astype(np.uint8)
                    mask = seg == class_id
                    vis[mask] = rgb[mask]

                    contour_mask = mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)

                    pct = 100.0 * mask.sum() / mask.size
                    text = f"ID={class_id}  {traj}  frame={i}  cover={pct:.1f}%"
                    cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (0, 0, 0), -1)
                    cv2.putText(vis, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    fname = out / f"{traj}_f{i:04d}.jpg"
                    cv2.imwrite(str(fname), vis)
                    saved_counts[class_id] = saved_counts.get(class_id, 0) + 1

    for class_id, cnt in sorted(saved_counts.items()):
        print(f"  [{label}] class_id={class_id:2d} → {cnt} frames saved → {OUT_DIR/label/f'class_{class_id:02d}'}")


print("Обрабатываю датасеты...")
for label, path in DATASETS:
    print(f"\n--- {label}: {path} ---")
    process_condition(label, path)

print(f"\nГотово! Открой папку: {OUT_DIR.resolve()}")
