"""
Визуализирует YOLO-полигоны из data/yolo_v2 поверх изображений.
Каждый класс — свой цвет, каждая инстанция — отдельный полигон.
Особый акцент на деревья — сохраняются в отдельную папку.
"""
import random
import cv2
import numpy as np
from pathlib import Path

DATASET     = Path("data/yolo_v6")
OUT_DIR     = Path("viz_labels")
N_SAMPLES   = 20      # сколько случайных изображений взять
SHOW_TREES_ONLY_DIR = OUT_DIR / "trees_only"  # папка только с деревьями

# Классы по midair.yaml: river=0, road=1, rock=2, tree=3
CLASS_NAMES  = {0: "river", 1: "road", 2: "rock", 3: "tree"}
CLASS_COLORS = {
    0: (255, 144,  30),   # river — оранжевый
    1: (128, 128, 128),   # road  — серый
    2: ( 43,  90, 139),   # rock  — коричневый
    3: ( 34, 139,  34),   # tree  — зелёный
}

OUT_DIR.mkdir(exist_ok=True)
SHOW_TREES_ONLY_DIR.mkdir(exist_ok=True)

def draw_instance_label(img: np.ndarray, pts: np.ndarray, label: str, color: tuple):
    """Рисует bbox и подпись для одного инстанса поверх img (in-place)."""
    x, y, bw, bh = cv2.boundingRect(pts)
    # bbox пунктиром (через drawContours с маленькой толщиной)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 1)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Фон под текст
    tx = max(x, 0)
    ty = max(y - 3, th + 2)
    cv2.rectangle(img, (tx, ty - th - baseline), (tx + tw + 2, ty + baseline), (20, 20, 20), -1)
    cv2.putText(img, label, (tx + 1, ty), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_labels(img_path: Path, lbl_path: Path, highlight_class: int = None):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    H, W = img.shape[:2]

    overlay = img.copy()

    if not lbl_path.exists():
        return img

    lines = lbl_path.read_text().strip().splitlines()
    class_counts = {}

    # Первый проход — заливка полигонов
    instances = []  # (cls, pts, color, dimmed)
    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))
        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= W
        pts[:, 1] *= H
        pts = pts.astype(np.int32)

        color  = CLASS_COLORS.get(cls, (255, 255, 255))
        dimmed = highlight_class is not None and cls != highlight_class

        if dimmed:
            cv2.fillPoly(overlay, [pts], (30, 30, 30))
            cv2.polylines(overlay, [pts], True, (60, 60, 60), 1)
        else:
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(overlay, [pts], True, (255, 255, 255), 1)

        instance_idx = class_counts.get(cls, 0)
        class_counts[cls] = instance_idx + 1
        instances.append((cls, pts, color, dimmed, instance_idx))

    result = cv2.addWeighted(img, 0.45, overlay, 0.55, 0)

    # Второй проход — bbox + подписи инстансов
    for cls, pts, color, dimmed, idx in instances:
        if dimmed:
            continue
        name  = CLASS_NAMES.get(cls, str(cls))
        label = f"{name}{idx}"
        draw_instance_label(result, pts, label, color)

    # Легенда
    y = 25
    for cls, cnt in sorted(class_counts.items()):
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        name  = CLASS_NAMES.get(cls, str(cls))
        cv2.rectangle(result, (8, y - 14), (24, y + 4), color, -1)
        cv2.putText(result, f"{name}: {cnt}", (28, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 22

    # Имя файла
    cv2.putText(result, img_path.stem[-40:], (8, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return result


# Берём случайные изображения из train
all_imgs = sorted((DATASET / "images" / "train").glob("*.jpg"))
# Отдельно — берём PLE изображения (там больше деревьев)
ple_imgs  = [p for p in all_imgs if "PLE" in p.stem]
kite_imgs = [p for p in all_imgs if "Kite" in p.stem]

random.seed(42)
selected = random.sample(ple_imgs, min(10, len(ple_imgs))) + \
           random.sample(kite_imgs, min(10, len(kite_imgs)))

print(f"Визуализирую {len(selected)} изображений...")

for img_path in selected:
    lbl_path = DATASET / "labels" / "train" / img_path.with_suffix(".txt").name

    # Все классы
    vis = draw_labels(img_path, lbl_path)
    if vis is not None:
        cv2.imwrite(str(OUT_DIR / img_path.name), vis)

    # Только деревья
    vis_tree = draw_labels(img_path, lbl_path, highlight_class=3)
    if vis_tree is not None:
        cv2.imwrite(str(SHOW_TREES_ONLY_DIR / img_path.name), vis_tree)

print(f"\nГотово!")
print(f"  Все классы : {OUT_DIR.resolve()}")
print(f"  Только деревья: {SHOW_TREES_ONLY_DIR.resolve()}")
