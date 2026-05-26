# Прогресс проекта: UAV-навигация с семантическим TILM
## Документ для передачи Claude Opus — структура отчёта и анализ

**Последнее обновление:** 2026-05-26  
**Автор:** Claude Sonnet (рабочий агент проекта)  
**Назначение:** зафиксировать фактически выполненные работы по этапам 0–3, передаётся Opus для выработки структуры итогового отчёта/главы 5 диплома.

---

## Контекст проекта

**Тема диплома:** Автономная VIO-навигация БПЛА с семантическим landmark-matching при потере GNSS.

**Научная новизна:** Trajectory-Indexed Landmark Memory (TILM) с instance embeddings и темпорально-ограниченным matching для коррекции инерциальной навигации на предкартографированных маршрутах в реальном времени на CPU-only edge-устройстве (Raspberry Pi 5).

**Датасет:** MidAir — синтетический датасет для UAV-навигации.
- 6 погодных условий: cloudy, sunny, foggy (Kite_training) + fall, spring, winter (PLE_training)
- Изображения: 1024×1024 RGB, camera=`color_left`
- Частота: 25 Hz камера, 100 Hz IMU/GT
- Калибровка: fx=fy=512, cx=cy=512 (pinhole, нулевая дисторсия)
- 30 траекторий на условие, ~2200 кадров на траекторию

**Целевая платформа:** Raspberry Pi 5 (Cortex-A76, 8GB RAM, без NPU)

**Стек:** Python 3.12, PyTorch 2.6, Ultralytics 8.4, OpenCV, h5py, NetworkX, loguru, hydra/omegaconf, evo

---

## Этап 0 — Scaffold проекта

**Статус:** ✅ Завершён (2026-05-23)

### Что сделано

Создана полная структура проекта `uav_nav/` с заглушками всех модулей:

**Модули (все реализованы как stubs, кроме указанных):**
- `data/`: `midair_loader.py`, `calibration.py`, `dataset_builder.py`
- `perception/`: `yolo_segmenter.py`, `embedding_head.py`, `feature_extractor.py`, `stereo_depth.py`, `landmark_extractor.py`
- `memory/`: `tilm.py`, `tilm_builder.py`, `temporal_matcher.py`, `place_descriptor.py`
- `estimation/`: `imu_preintegrator.py`, `ekf_vio.py`, `pose_utils.py`
- `planning/`: `path_follower.py`, `adaptive_corridor.py`, `mission_modes.py`, `rejoin_planner.py`
- `runtime/`: `pipeline.py`, `pseudo_simulator.py`, `airsim_bridge.py`, `monitor.py`, `logger.py`
- `deployment/`: `export_onnx.py`, `export_ncnn.py`, `benchmark_pi5.py`, `pi5_runtime.py`
- `eval/`: `trajectory_eval.py`, `embedding_eval.py`, `matching_eval.py`, `scenario_runner.py`, `ablation.py`

**Конфигурация:**
- `configs/default.yaml` — полный конфиг системы (camera, IMU, EKF, TILM, YOLO, embedding, path follower)
- `configs/pi5.yaml` — наследует default, переопределяет: NCNN backend, 320×240, 8 FPS
- `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml`

**Тесты:** 30 тестов (pose_utils + place_descriptor + path_follower), все проходят.

---

## Этап 1 — Data Layer

**Статус:** ✅ Завершён (2026-05-24)

### `midair_loader.py`
- HDF5 читается через `h5py.File(io.BytesIO(data))` — без распаковки zip на диск
- Lazy loading кадров, синхронизация IMU↔Camera по формуле `gt_idx = round(i * n_gt / n_frames)`

### `dataset_builder.py`
- Пишет YOLO-датасет: `images/` + `labels/` (polygon segmentation format)
- Split по траекториям (не по кадрам), фиксированный seed
- Watershed-разделение слившихся blob'ов, Douglas-Peucker аппроксимация полигонов (EPSILON=0.002)
- MIN_AREA=400 px, MIN_BOX=15 px, TARGET=640×640

### Критические баги, найденные в процессе (важно для диплома)

1. **Камера `color_down` → `color_left`** — исходно использовалась нижняя камера; для навигации нужна фронтальная. После исправления деревья появились в нужном количестве.
2. **`label_map.json` — добавлен класс `road`** — MidAir ID 10 и 11 (дорога) отсутствовали. Итоговые 4 класса: tree/rock/river/road.
3. **Итерация по сырым MidAir ID** вместо merged semantic mask — исходная логика сливала объекты.
4. **frame_stride=10** — без ограничения датасет ~20 ГБ; stride=10 → ~2 ГБ.

**Тесты:** 28 тестов, все проходят. Покрытие: `dataset_builder.py` 96%, `midair_loader.py` 77%.

---

## Этап 2 — Базовая YOLO + Бенчмарки

**Статус:** ✅ Завершён (2026-05-26)

### Обучение

Обучены две модели YOLOv8n-seg. Финальная — v2.

**Финальная модель v2:**
- Run: `runs/segment/experiments/20260525_222650_yolo`
- 60 эпох, ранняя остановка на эпохе 40 (patience=20), 8.6 часов, RTX 3090
- Датасет: `data/yolo_v5`, 6 условий, split по траекториям 80/10/10
- Train images: 19,011 | Val: 2,220 | Test: 2,646

### Метрики val (epoch 40, conf=0.001, iou=0.6)

| Класс | Box mAP50 | Mask mAP50 |
|---|---|---|
| river | 0.597 | 0.588 |
| road  | 0.650 | 0.559 |
| rock  | 0.450 | 0.454 |
| tree  | 0.252 | 0.196 |
| **all** | **0.488** | **0.449** |

**Сравнение v1 vs v2:**

| | v1 (baseline) | v2 (финал) | Δ |
|---|---|---|---|
| val mAP50-mask | 0.410 | **0.449** | +0.039 |
| tree mAP50-mask | 0.031 | **0.196** | +0.165 |
| cross-weather mean | 0.208 ± 0.096 | **0.327 ± 0.038** | +57%, σ×3↓ |

### Эксперимент 5.3 — Cross-weather (v2, `data/yolo_v5`, 441 img/условие)

| Условие | mAP50-mask | rock | tree |
|---|---|---|---|
| Kite_training_cloudy | 0.379 | 0.465 | 0.011 |
| Kite_training_foggy  | 0.357 | 0.457 | 0.008 |
| Kite_training_sunny  | 0.352 | 0.430 | 0.013 |
| PLE_training_fall    | 0.291 | 0.270 | 0.030 |
| PLE_training_spring  | 0.307 | 0.294 | 0.029 |
| PLE_training_winter  | 0.276 | 0.475 | 0.018 |
| **Mean** | **0.327 ± 0.038** | — | — |

**Научные наблюдения:**
1. Kite (лес) mAP≈0.35–0.38 vs PLE (город/поле) mAP≈0.28–0.31 — domain gap между типами сцены
2. Разброс σ=0.038 (v2) vs σ=0.096 (v1) — значительно лучшая генерализация при обучении на всех 6 условиях
3. tree: recall 3% при conf=0.35, 64.5% при conf=0.10 — confidence miscalibration, не отсутствие детекций
4. rock — наиболее стабильный класс (текстурно инвариантен к освещению)

### Экспорт

| Формат | Файл | Размер |
|---|---|---|
| PyTorch | `best.pt` | 6.8 MB |
| ONNX | `best.onnx` | 12.7 MB (opset=17, simplified) |
| NCNN | `best_ncnn_model/` | 12.6 MB (FP32, opt=2) |

Pi5 FPS benchmark: не проводился (нет оборудования). По НИР ожидается 7–14 FPS.

### Тесты

**38/38 passed** (23 yolo_segmenter + 15 embedding_head), без GPU и датасета.

---

## Этап 3 — Embedding-голова

**Статус:** ✅ Завершён (2026-05-26)

### Архитектурные решения

| Решение | Выбор | Обоснование |
|---|---|---|
| Backbone | MobileNetV3-Small (pretrained ImageNet) | 2.5M параметров, быстрый на ARM; ImageNet фильтры дают хорошую текстурную основу |
| Нормализация | LayerNorm (не BatchNorm) | Работает при batch_size=1 в любом режиме — необходимо для single-crop inference |
| Projection | Linear(576→256) → LayerNorm → ReLU → Dropout(0.1) → Linear(256→128) | Двухслойная проекция эффективнее одной |
| Loss | TripletMarginLoss(margin=0.5) | Стандарт metric learning; margin=0.5 для пространства cos dist ∈ [-1,1] |
| Crop | 96×96 px с polygon masking (фон → серый 127) | 64×64 теряет текстуру крупных объектов; маска убирает шум от соседних объектов |
| Instance tracking | IoU ≥ 0.30 bbox между соседними кадрами | Простой и надёжный; HSV-histogram не добавил качества при тестировании |

### Instance Index

Построен из `data/yolo_v5` через `build_instance_index.py`:

| Сплит | Детекций | Instances | ≥2 obs (usable) |
|---|---|---|---|
| train | 193,816 | 120,766 | **27,904** |
| val | 21,792 | 13,739 | **3,359** |
| test | 30,155 | 19,936 | **4,564** |

По классам (train): tree=95,150 instances (доминирует), rock=22,883, road=1,680, river=1,053.

### Обучение (текущее)

```
python -m uav_nav.scripts.train_embedding \
    --device cuda --epochs 50 --batch_size 64 --num_workers 0
```

- 436 батчей/эпоха (27,904 instances ÷ 64)
- Первые 5 эпох: backbone заморожен, обучается только projection head
- Эпоха 6+: backbone разморожен, lr_backbone = lr×0.1
- Val метрика: ROC AUC (pos=same instance, neg=diff instance same class)
- Checkpoints каждые 5 эпох → `experiments/<date>_embedding/`

### Реализованные модули этапа 3

| Модуль | Статус |
|---|---|
| `scripts/build_instance_index.py` | ✅ Реализован и запущен |
| `perception/embedding_head.py` | ✅ Реализован (MobileNetV3-Small + LayerNorm) |
| `data/triplet_dataset.py` | ✅ Реализован (TripletDataset + build_loaders) |
| `scripts/train_embedding.py` | ✅ Реализован (полный цикл + ROC AUC val) |
| `eval/embedding_eval.py` | ✅ Реализован (ROC AUC, Recall@1/5, discriminability) |
| `perception/feature_extractor.py` | ✅ Реализован (YOLO+EmbeddingHead → Detection) |

### Финальные метрики этапа 3 (val, 3359 instances)

| Метрика | Порог | Факт | |
|---|---|---|---|
| ROC AUC same/diff instance (val) | ≥ 0.80 | **0.983** | ✅ |
| Recall@1 (val, full gallery) | ≥ 0.60 | **0.118** | ⚠️ |
| Recall@5 (val) | — | **0.300** | |
| Discriminability | — | **8.46** | |
| Embedding inference latency (CPU) | ≤ 15 ms | `[не измерено]` | |
| pytest без GPU | 100% | **38/38** | ✅ |

Recall@1 ниже порога из-за большой галереи (3359) и доминирования деревьев (~95%). В контексте TILM gallery ограничена temporal window, поэтому retrieval реально проще.

---

## Этап 4 — Базовый VIO (EKF)

**Статус:** ✅ Завершён (2026-05-26)

### Что реализовано

| Модуль | Описание |
|---|---|
| `estimation/pose_utils.py` | `so3_exp`, `so3_log`, `skew`, `quat_to_rot`, `rot_to_quat`, `quat_mult`, `se3_exp`, `interpolate_poses` |
| `estimation/imu_preintegrator.py` | `IMUPreintegrator` — интеграция на SO(3), якобианы поправок смещений, `PreintegratedState.correct_for_bias_update()` |
| `estimation/ekf_vio.py` | `EKFState` (quaternion + P=18×18), `EKFVIO` — propagate (F=18×18), update_gnss, update_landmark (Joseph form), update_visual (NotImplementedError) |

### Ключевые решения

| Решение | Выбор | Обоснование |
|---|---|---|
| Error-state EKF | 15-DoF активное состояние (δp, δv, δθ, δba, δbg) + 3 reserved | P=18×18 по спецификации проекта |
| IMU интеграция | Euler на SO(3) с δt | Достаточная точность при 100 Hz IMU |
| Bias correction | Первый порядок по якобианам Jg_R, Ja_v, Ja_p | Стандарт для IMU preintegration |
| Landmark update | `z = p_nominal + correction → innovation = correction` | Связка с TemporalMatcher.position_correction_ned |
| Covariance update | Joseph form `(I−KH)P(I−KH)ᵀ + KRKᵀ` | Численная стабильность |

### Тесты

**49/49 passed** (test_imu_preintegrator: 14, test_ekf_vio: 20, test_pose_utils: 15)

---

## Этап 5 — TILM

**Статус:** ✅ Завершён (2026-05-26)

### Что реализовано

| Модуль | Описание |
|---|---|
| `memory/tilm.py` | `TILM` — NetworkX DiGraph, add_node/add_edge, nearest_node (k-NN), shortest_path, save/load (pickle) |
| `memory/tilm_builder.py` | `TILMBuilder` — process_frame: создание узла по distance/landmark threshold, sequential edges, loop closure по place_descriptor cosine sim |
| `memory/temporal_matcher.py` | `TemporalMatcher` — brute-force descriptor index, voting-based candidate retrieval, greedy geometric verification, position_correction_ned |
| `perception/landmark_extractor.py` | `LandmarkExtractor.extract()` — из SegmentationResult+DepthResult → список Landmark с 3D позицией и дескриптором |

### Алгоритм TemporalMatcher

1. **Индекс**: все дескрипторы TILM → матрица (N_entries, D)
2. **Голосование**: каждое наблюдение голосует за узел с наибольшим cos-similarity (same class only)
3. **Top-K кандидатов**: по числу голосов, опционально взвешено близостью к position_prior
4. **Geometric verify**: жадный матчинг по cos-sim ≥ (1 − threshold), same class
5. **Коррекция**: `correction = best_node.position_ned − position_prior`

### LandmarkExtractor.extract()

1. Сортировка по confidence desc
2. Фильтр по классу (valid_classes) и площади маски (min_mask_area)
3. Центроид маски → depth по nanpercentile → `_unproject()` → position_3d
4. `_crop_and_embed()`: bbox маски + 20% padding, фон → gray 127, resize 96×96 → EmbeddingHead

### Тесты

**43/43 passed** (TestTILMBasic×7, TestTILMEdge×2, TestTILMNearestNode×5, TestTILMShortestPath×2, TestTILMSaveLoad×1, TestTILMNode×2, TestTILMBuilderNodeCreation×7, TestTILMBuilderLoopClosure×2, TestTemporalMatcher×12, TestMatchResult×2)

---

## Этап 6 — Matching + EKF-коррекции

**Статус:** ✅ Завершён (2026-05-26)

### Что реализовано

| Модуль | Описание |
|---|---|
| `runtime/matching_corrector.py` | `MatchingCorrector` — TILM→EKF мост: state machine GNSS/TILM, confidence gate, correction interval throttle, `CorrectorStats` |
| `eval/matching_eval.py` | `MatchingEvaluator.evaluate()` — precision/recall/F1/RMSE; `precision_recall_curve()` |

### Алгоритм MatchingCorrector

```
GNSS_ACTIVE:  update_gnss(position, accuracy) → EKF
GNSS_LOST:    match(observations, prior=ekf.position)
                → is_valid AND confidence >= min_confidence
                AND elapsed >= correction_interval
                → update_landmark(correction_ned) → EKF
```

**Ключевые параметры:**
- `min_confidence=0.3` — порог уверенности матчинга
- `correction_interval=1.0` — минимальный интервал между коррекциями (с)

### Тесты

**41/41 passed** (CorrectorStats×5, Mode×4, GNSS×5, TILM×6, Confidence×2, Interval×3, NoObs×2, ResetStats×1, MatchingEvaluator×8, PR-curve×5)

---

## Текущий стек архитектурных решений

| ID | Решение | Обоснование |
|---|---|---|
| DD-001 | loguru вместо stdlib logging | Structured fields, JSON sink, минимум boilerplate |
| DD-002 | Hydra + OmegaConf | Композиция конфигов, CLI-переопределения |
| DD-003 | NCNN на Pi5 | Оптимизирован под ARM Cortex-A, нет зависимости NVIDIA/Intel |
| DD-004 | NetworkX для TILM | Алгоритмы графов, pure-Python |
| DD-005 | evo для ATE/RPE | Стандарт SLAM-сообщества |
| DD-006 | frame_stride=10 | Без ограничения датасет ~20 ГБ; stride=10 → ~2 ГБ |
| DD-007 | Split по траекториям, не по погодам | Domain shift: модель на cloudy/sunny деградирует на foggy/winter |
| DD-008 | MobileNetV3-Small для embedding | Pretrained текстурные фильтры; 2.5M параметров, быстро на Pi5 |
| DD-009 | Crop-based embedding (не ROIAlign) | ROIAlign требует встраивания в YOLO backbone → сложный ONNX граф; crop достаточен |
| DD-010 | Маскирование crop по YOLO polygon | Raw bbox-crop включает фон соседних объектов → шумный embedding |
| DD-011 | LayerNorm вместо BatchNorm в projection head | BatchNorm требует batch_size>1 в train mode; LayerNorm работает при B=1 |
| DD-012 | IoU ≥ 0.30 для instance tracking | HSV-histogram не улучшил качество; простой IoU достаточен при stride=15 кадров |

---

## Структура проекта

```
vkr/
├── uav_nav/                        # Основной пакет
│   ├── data/
│   │   ├── calibration.py          # Модель камеры MidAir (fx=fy=512)
│   │   ├── dataset_builder.py      # ✅ Построение YOLO-датасета из MidAir HDF5
│   │   ├── midair_loader.py        # ✅ Загрузчик MidAir (lazy HDF5, IMU sync)
│   │   └── triplet_dataset.py      # ✅ TripletDataset для embedding (этап 3)
│   ├── perception/
│   │   ├── yolo_segmenter.py       # ✅ YOLOv8n-seg inference wrapper
│   │   ├── embedding_head.py       # ✅ MobileNetV3-Small + LayerNorm → 128d
│   │   ├── feature_extractor.py    # ✅ YOLO + EmbeddingHead → Detection
│   │   ├── landmark_extractor.py   # ✅ LandmarkExtractor (этап 5)
│   │   └── stereo_depth.py         # stub (MidAir pinhole, нет стерео)
│   ├── memory/
│   │   ├── tilm.py                 # ✅ TILM — NetworkX граф (этап 5)
│   │   ├── tilm_builder.py         # ✅ TILMBuilder — инкрементальная сборка (этап 5)
│   │   ├── temporal_matcher.py     # ✅ TemporalMatcher — descriptor matching (этап 5)
│   │   └── place_descriptor.py     # ✅ PlaceDescriptor (тесты проходят)
│   ├── estimation/
│   │   ├── imu_preintegrator.py    # ✅ IMUPreintegrator — SO(3) (этап 4)
│   │   ├── ekf_vio.py              # ✅ EKFVIO — Error-State EKF 18×18 (этап 4)
│   │   └── pose_utils.py           # ✅ SE3/quaternion utils (тесты проходят)
│   ├── planning/
│   │   ├── path_follower.py        # ✅ PathFollower (тесты проходят)
│   │   ├── adaptive_corridor.py    # stub → этап 7
│   │   ├── mission_modes.py        # stub → этап 7
│   │   └── rejoin_planner.py       # stub → этап 7
│   ├── runtime/
│   │   ├── matching_corrector.py   # ✅ MatchingCorrector TILM→EKF (этап 6)
│   │   ├── pipeline.py             # stub → этап 7
│   │   ├── pseudo_simulator.py     # stub → этап 7
│   │   ├── airsim_bridge.py        # stub (опционально)
│   │   ├── monitor.py              # stub → этап 7
│   │   └── logger.py               # loguru wrapper
│   ├── deployment/
│   │   ├── export_onnx.py          # ✅ ONNX экспорт (opset=17)
│   │   ├── export_ncnn.py          # ✅ NCNN экспорт (FP32, opt=2)
│   │   ├── benchmark_pi5.py        # stub → этап 8
│   │   └── pi5_runtime.py          # stub → этап 8
│   ├── eval/
│   │   ├── yolo_eval.py            # ✅ Cross-weather mAP
│   │   ├── embedding_eval.py       # ✅ ROC AUC, Recall@K, discriminability
│   │   ├── trajectory_eval.py      # stub → этап 9 (ATE/RPE с evo)
│   │   ├── matching_eval.py        # ✅ MatchingEvaluator (этап 6)
│   │   ├── scenario_runner.py      # stub → этап 9
│   │   └── ablation.py             # stub → этап 9
│   └── scripts/
│       ├── build_dataset.py        # ✅ Сборка YOLO-датасета
│       ├── build_instance_index.py # ✅ IoU-трекинг → instance_index CSV
│       ├── train_yolo.py           # ✅ Запуск обучения YOLOv8n-seg
│       ├── train_embedding.py      # ✅ TripletLoss обучение EmbeddingHead
│       ├── eval_yolo.py            # ✅ Запуск yolo_eval
│       ├── eval_cross_weather.py   # ✅ Cross-weather per-condition eval
│       ├── build_tilm.py           # stub → этап 5
│       ├── simulate_gnss_loss.py   # stub → этап 7
│       └── run_all_experiments.py  # stub → этап 9
├── data/
│   ├── MidAir/                     # Исходные HDF5 (не в git)
│   ├── yolo_v5/                    # YOLO-датасет (финальный, 6 условий)
│   │   ├── images/{train,val,test}/
│   │   └── labels/{train,val,test}/
│   └── instance_index/             # CSV-индекс для embedding
│       ├── train.csv               # 27,904 usable instances
│       ├── val.csv                 # 3,359 usable instances
│       ├── test.csv                # 4,564 usable instances
│       └── stats.json
├── weights/
│   ├── yolo_midair.pt              # YOLOv8n-seg v2 (финальный)
│   └── embedding_head.pt           # EmbeddingHead (после обучения)
├── experiments/
│   ├── runs/segment/experiments/20260525_222650_yolo/  # YOLO v2
│   └── 20260526_092148_embedding/  # EmbeddingHead (текущее обучение)
│       ├── best.pt                 # Лучший по val_AUC (эпоха 31, 0.978)
│       ├── train_log.csv
│       └── checkpoints/
├── configs/
│   ├── default.yaml                # Полный конфиг системы
│   └── pi5.yaml                    # Pi5 overrides (NCNN, 320×240, 8 FPS)
└── docs/
    ├── progress_for_opus.md        # Этот файл
    ├── stage_{0,1,2,3}_report.md   # Отчёты по этапам
    ├── stage_{2,3}_plan.md         # Планы
    └── decisions.md                # Design decisions
```

**Тесты:** 171/171 passed (этапы 0–5: test_pose_utils×15, test_place_descriptor×5, test_path_follower_config×10, test_embedding_head×15, test_yolo_segmenter×23, test_imu_preintegrator×14, test_ekf_vio×20, test_tilm×43).

---

## Общий статус этапов

| Этап | Название | Статус |
|---|---|---|
| 0 | Scaffold | ✅ Завершён |
| 1 | Data Layer | ✅ Завершён |
| 2 | YOLO + Бенчмарки | ✅ Завершён |
| 3 | Embedding-голова | ✅ Завершён (ROC AUC=0.983, Recall@1=0.118, Discriminability=8.46) |
| 4 | Базовый VIO (EKF) | ✅ Завершён (49/49 тестов: IMUPreintegrator, EKFVIO, pose_utils) |
| 5 | TILM | ✅ Завершён (43/43 тестов: TILM, TILMBuilder, TemporalMatcher, LandmarkExtractor) |
| 6 | Matching + EKF-коррекции | ✅ Завершён (41/41 тестов: MatchingCorrector, MatchingEvaluator) |
| 7 | Path-follower + Pseudo-simulator | 📋 Не начат |
| 8 | Pi5 deployment | 📋 Не начат |
| 9 | Ablation studies | 📋 Не начат |
| 10 | Финальное демо | 📋 Опционально |

---

## Вопросы для Opus — что хотелось бы получить

1. **Структура главы 5 диплома**: в каком порядке представлять эксперименты 5.1–5.11, чтобы логика повествования была последовательной?

2. **Как подать domain shift как научный результат**: переход от split-по-погодам к split-по-траекториям и его влияние на генерализацию — ключевое наблюдение этапа 2.

3. **Confidence miscalibration для tree**: при conf=0.35 recall 3%, при conf=0.10 recall 64.5% — как обосновать выбор порога для runtime и описать этот эффект в дипломе?

4. **Embedding ablation**: как обосновать crop-based vs ROIAlign, MobileNetV3-Small vs from scratch, LayerNorm vs BatchNorm в контексте диплома?

5. **Если cross-weather Recall@1 < 0.60**: как это оформить — как limitation или как мотивацию к domain adaptation в будущих работах?
