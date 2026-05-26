"""Экспорт best.pt в ONNX и NCNN.

Запуск:
    python export_model.py
"""
from pathlib import Path
from ultralytics import YOLO

WEIGHTS = "runs/segment/experiments/20260525_110834_yolo/train/weights/best.pt"

if __name__ == "__main__":
    model = YOLO(WEIGHTS)

    print("=== ONNX export ===")
    onnx_path = model.export(format="onnx", imgsz=640, simplify=True, opset=17)
    print(f"ONNX saved: {onnx_path}")

    print("\n=== NCNN export ===")
    ncnn_path = model.export(format="ncnn", imgsz=640)
    print(f"NCNN saved: {ncnn_path}")

    print("\nГотово. Файлы:")
    base = Path(WEIGHTS).parent
    for f in base.iterdir():
        if f.suffix in {".onnx", ".param", ".bin"} or f.name.endswith("_ncnn_model"):
            print(f"  {f}")
