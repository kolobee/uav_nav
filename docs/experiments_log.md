# Experiments Log

Records all experiments run, their configuration, and results.

## Format

```
## EXP-NNN: <short title>
**Date:** YYYY-MM-DD
**Config:** configs/experiments/<name>.yaml
**Hypothesis:** ...
**Result:** ...
**Notes:** ...
```

---

## EXP-001: YOLOv8n-seg baseline training (Stage 2)

**Date:** 2026-05-25  
**Run dir:** `runs/segment/experiments/20260525_110834_yolo`  
**Hypothesis:** YOLOv8n-seg обученная на multi-weather MidAir датасете (6 условий, split по траекториям) покажет mAP50-mask ≥ 0.55 на val.

**Config:**
- model: YOLOv8n-seg (pretrained COCO)
- data: `data/yolo_v44/midair.yaml` (28 515 train / 3 330 val / 3 978 test)
- imgsz=640, batch=64, epochs=100, patience=20, lr0=1e-3

**Result:**
- Остановилась на эпохе 33 (best ~15–16)
- val mAP50-mask: **0.41** (порог 0.55 не достигнут)
- test mAP50-mask: **0.18** (val→test gap = −0.23)
- Лучший класс: rock (val 0.23, test 0.34)
- Худший класс: tree (val 0.03, test 0.04 — confidence calibration issue)

**Notes:** Принят как baseline. Параллельно запущена вторая тренировка (`20260525_222650_yolo`).

---

## EXP-002: Cross-weather evaluation (Эксп. 5.3)

**Date:** 2026-05-26  
**Config:** `configs/experiments/5_3_cross_weather.yaml`  
**Script:** `uav_nav/scripts/eval_cross_weather.py`  
**Results dir:** `experiments/20260526_005114_cross_weather/`

**Hypothesis:** Модель, обученная на всех 6 условиях, покажет стабильные метрики при оценке по каждому условию отдельно.

**Result:**

| Условие | mAP50-mask |
|---|---|
| Kite_training_cloudy | 0.315 |
| Kite_training_foggy  | 0.307 |
| Kite_training_sunny  | 0.276 |
| PLE_training_fall    | 0.073 |
| PLE_training_spring  | 0.111 |
| PLE_training_winter  | 0.166 |
| **Mean**             | **0.208 ± 0.096** |

**Вывод:** Kite (лесная среда) значительно лучше PLE (городская/сельская). Разброс 0.096 — высокий. Гипотеза о стабильности опровергнута: модель имеет существенный domain gap между типами среды.
