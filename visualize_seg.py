"""Визуализация сегментации YOLOv8n-seg на примерах из тест-сета.

Запуск:
    python visualize_seg.py
"""

import random
from pathlib import Path

import cv2
import numpy as np

WEIGHTS = "runs/segment/experiments/20260525_110834_yolo/train/weights/best.pt"
IMAGES_DIR = Path("data/yolo_v44/images/test")
OUT_DIR = Path("docs/seg_examples")
N_SAMPLES = 12
CONF = 0.15
SEED = 42

CLASS_COLORS = {
    "river": (255, 100,   0),
    "road":  (200, 200,   0),
    "rock":  (100, 100, 255),
    "tree":  ( 50, 200,  50),
}
ALPHA = 0.45

if __name__ == "__main__":
    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(WEIGHTS)

    all_images = sorted(IMAGES_DIR.glob("*.jpg"))
    random.seed(SEED)
    samples = random.sample(all_images, min(N_SAMPLES, len(all_images)))

    print(f"Обрабатываю {len(samples)} изображений → {OUT_DIR}/")

    for img_path in samples:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        results = model.predict(img_bgr, imgsz=640, conf=CONF, iou=0.45, verbose=False)
        result = results[0]

        overlay = img_bgr.copy()

        if result.masks is not None and len(result.masks):
            masks = result.masks.data.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            names = model.names

            H, W = img_bgr.shape[:2]

            for mask, cls_id in zip(masks, cls_ids):
                cls_name = names.get(cls_id, str(cls_id))
                color_rgb = CLASS_COLORS.get(cls_name, (200, 200, 200))
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_bool = mask_resized > 0.5
                overlay[mask_bool] = (
                    (1 - ALPHA) * img_bgr[mask_bool] + ALPHA * np.array(color_bgr)
                ).astype(np.uint8)

            for mask, cls_id, conf in zip(masks, cls_ids, confs):
                cls_name = names.get(cls_id, str(cls_id))
                color_rgb = CLASS_COLORS.get(cls_name, (200, 200, 200))
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_u8 = (mask_resized > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color_bgr, 2)
                M = cv2.moments(mask_u8)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(overlay, label, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(overlay, label, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color_bgr, 1, cv2.LINE_AA)

        lx, ly = 10, 10
        for cls_name, color_rgb in CLASS_COLORS.items():
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            cv2.rectangle(overlay, (lx, ly), (lx + 16, ly + 16), color_bgr, -1)
            cv2.putText(overlay, cls_name, (lx + 20, ly + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(overlay, cls_name, (lx + 20, ly + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            ly += 22

        n_det = len(result.boxes) if result.boxes is not None else 0
        out_path = OUT_DIR / f"{img_path.stem}_seg.jpg"
        cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  {out_path.name}  det={n_det}")

    print(f"\nГотово — {len(samples)} изображений в {OUT_DIR}/")
