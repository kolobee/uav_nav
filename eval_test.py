from ultralytics import YOLO

WEIGHTS = "runs/segment/experiments/20260525_110834_yolo/train/weights/best.pt"
DATA = "data/yolo_v44/midair.yaml"

if __name__ == "__main__":
    model = YOLO(WEIGHTS)
    common = dict(imgsz=640, batch=16, workers=0, device=0, conf=0.001, iou=0.6, verbose=False)

    print("\n=== VAL (conf=0.001, как при обучении) ===")
    rv = model.val(data=DATA, split="val", **common)
    print(f"  all   mAP50-mask: {rv.seg.map50:.4f}   mAP-mask: {rv.seg.map:.4f}")
    names = rv.names
    for i, name in names.items():
        try:
            m50 = float(rv.seg.maps[i])
        except Exception:
            m50 = float("nan")
        print(f"  {name:<8} mAP50-mask: {m50:.4f}")

    print("\n=== TEST (conf=0.001, как при обучении) ===")
    rt = model.val(data=DATA, split="test", **common)
    print(f"  all   mAP50-mask: {rt.seg.map50:.4f}   mAP-mask: {rt.seg.map:.4f}")
    for i, name in names.items():
        try:
            m50 = float(rt.seg.maps[i])
        except Exception:
            m50 = float("nan")
        print(f"  {name:<8} mAP50-mask: {m50:.4f}")

    print("\n=== ИТОГ ===")
    print(f"  val  mAP50-mask: {rv.seg.map50:.4f}   mAP50-box: {rv.box.map50:.4f}")
    print(f"  test mAP50-mask: {rt.seg.map50:.4f}   mAP50-box: {rt.box.map50:.4f}")
    gap = rv.seg.map50 - rt.seg.map50
    print(f"  val→test gap:    {gap:.4f}")
