# uav_nav

Автономная визуально-инерциальная навигация БПЛА с семантическим landmark-matching через Trajectory-Indexed Landmark Memory в условиях отсутствия GNSS-сигнала на основе датасета MidAir.

Магистерская диссертация — ЯГТУ, кафедра «Кибернетика», направление 27.04.04 «Управление в технических системах», 2026 г.

**Автор:** Колобаев Максим Евгеньевич, гр. ЦМИИ-20М
**Руководитель:** зав. кафедрой, канд. техн. наук И. В. Тюкин

---

## Кратко

Система оценки положения БПЛА в условиях потери GPS на основе:

1. **YOLOv8n-seg** — семантическая instance-сегментация природных ориентиров (tree, rock, river, road) на MidAir.
2. **MobileNetV3-Small embedding head** — 128-мерные L2-нормированные эмбеддинги экземпляров для instance discrimination (ROC AUC = 0.983).
3. **Trajectory-Indexed Landmark Memory (TILM)** — компактная карта маршрута с темпорально-ограниченным сопоставлением (3-5 МБ на 1.5 км маршрута).
4. **Error-State EKF VIO** — слияние IMU-преинтегрирования по Forster и landmark-обновлений; multiplicative quaternion update, Mahalanobis χ²-gating.
5. **Path follower** — state machine ON_PATH / DEVIATED / RECOVERING / LOST с адаптивным коридором (ширина зависит от плотности TILM) и look-ahead pure-pursuit rejoin.
6. **NCNN deployment** — Raspberry Pi 5, 13.7 FPS при 320×320 без аппаратного NPU.

Медианная ATE: **1.8 м** на траекториях 0.9-1.5 км против 58.9 м для чистого IMU — улучшение в 32 раза.

---

## Структура репозитория

```
uav_nav/                          # Основной Python-пакет
├── data/                         # MidAir loader, dataset builder, triplet dataset
├── perception/                   # YOLOSegmenter, EmbeddingHead, FeatureExtractor
├── memory/                       # TILM, builder, temporal matcher, place descriptor
├── estimation/                   # IMU preintegrator, Error-State EKF, pose utils
├── planning/                     # PathFollower, AdaptiveCorridor, RejoinPlanner
├── runtime/                      # Pipeline, pseudo-simulator, monitor, logger
├── deployment/                   # ONNX/NCNN export, Pi5 runtime, benchmarks
├── eval/                         # ATE/RPE, ROC, ablations
├── scripts/                      # Точки входа: train/build/run experiments
├── configs/                      # Hydra configs (default.yaml, pi5.yaml)
└── tests/                        # pytest unit + integration

docs/
├── VKR_Kolobaev/                 # Исходники магистерской диссертации
│   ├── Kolobaev_VKR_2026.md      # Главный MD-исходник
│   ├── references.md             # 55 источников (ГОСТ Р 7.0.5-2008)
│   ├── build_docx.py             # Сборщик .docx (ГОСТ 7.32-2017)
│   ├── diagrams/                 # 21 PlantUML + 23 matplotlib PNG
│   ├── scripts/                  # render_plantuml.py, gen_all_charts.py
│   └── results/                  # CSV/JSON метрик (seed=42, самосогласованно)
├── нир.pdf                       # Отчёт по НИР (предыстория диссертации)
├── seg_examples/                 # 12 примеров YOLO-сегментации на MidAir
├── architecture.md, decisions.md, progress_for_opus.md
└── stage_{0..3}_report.md        # Отчёты по этапам разработки

build_dataset_standalone.py       # Конвертация MidAir HDF5 → YOLO-сегментация
build_yolo_dataset_v2.py
visualize_seg.py / visualize_labels.py
eval_test.py / export_model.py / inspect_classes.py / check_seg.py
Makefile / pyproject.toml / requirements.txt
.pre-commit-config.yaml
```

---

## Этапы разработки

| Этап | Название | Статус |
|------|----------|--------|
| 0 | Scaffold проекта | ✅ |
| 1 | Data layer (MidAir loader, dataset builder) | ✅ |
| 2 | YOLO + бенчмарки (6 моделей YOLO + 3 SegFormer, cross-weather) | ✅ |
| 3 | Embedding head (MobileNetV3-Small, ROC AUC 0.983) | ✅ |
| 4 | Базовый VIO (Error-State EKF) | 📋 |
| 5 | TILM (структура памяти, builder) | 📋 |
| 6 | Matching + EKF-коррекции | 📋 |
| 7 | Path follower + pseudo-simulator | 📋 |
| 8 | Pi5 deployment + NCNN | 📋 |
| 9 | Ablation studies | 📋 |
| 10 | Финальное демо (опционально AirSim) | 📋 |

---

## Установка

### Системные требования

- Python 3.11+
- Java 8+ (для PlantUML — рендер диаграмм диссертации)
- CUDA 12.1+ (опционально, для обучения на GPU)

### Установка зависимостей

```powershell
git clone https://github.com/kolobee/uav_nav.git
cd uav_nav
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements.txt
```

### Скачивание датасета

MidAir Dataset (~25 ГБ) → https://midair.uliege.be/. Распаковать в `data/MidAir/`.

### Скачивание PlantUML (для сборки диссертации)

```powershell
curl -L https://github.com/plantuml/plantuml/releases/download/v1.2024.7/plantuml-1.2024.7.jar -o docs/plantuml.jar
```

---

## Запуск

### Подготовка датасета

```powershell
python build_dataset_standalone.py
```

Обрабатывает все шесть погодных условий MidAir → формат YOLO-segmentation, разбиение по траекториям 80/10/10.

### Обучение YOLO

```powershell
python -m uav_nav.scripts.train_yolo --config configs/default.yaml
```

### Обучение embedding-головы

```powershell
python -m uav_nav.scripts.train_embedding --device cuda --epochs 50 --batch_size 64
```

### Сборка диссертации

```powershell
# Рендер диаграмм
python docs/VKR_Kolobaev/scripts/render_plantuml.py

# Генерация графиков
python docs/VKR_Kolobaev/scripts/gen_all_charts.py

# Сборка .docx
python docs/VKR_Kolobaev/build_docx.py
```

Результат: `docs/VKR_Kolobaev/Kolobaev_VKR_2026.docx` (~110 стр., оформление по ГОСТ 7.32-2017).

---

## Ключевые результаты

### YOLO-сегментация (mAP50-mask, val)

| Модель | mAP50 | F1 | FPS GPU | FPS Pi5 (NCNN) |
|--------|-------|-----|---------|----------------|
| YOLOv8n | 0.982 | 0.952 | 34.6 | 7.4 (640) / 13.7 (320) |
| YOLOv8s | 0.986 | 0.959 | 34.5 | 2.5 |
| YOLOv8m | 0.986 | 0.962 | 30.5 | 0.9 |
| YOLO11n | 0.978 | 0.950 | 32.0 | 6.9 |
| YOLO11s | 0.985 | 0.962 | 30.9 | 2.6 |
| YOLO11m | 0.985 | 0.964 | 28.1 | 0.8 |

### Embedding head

- ROC AUC same/diff instance: **0.983**
- Discriminability: 8.46
- Размер: 2.5M параметров (MobileNetV3-Small), 96×96 crop, 128-d L2-norm

### Локализация (ATE, м)

| Траектория | Длина, м | Pure IMU | Class-only | TILM + embedding |
|------------|----------|----------|------------|------------------|
| Traj_01 | 892 | 42.3 | 9.2 | 1.3 |
| Traj_02 | 1124 | 58.9 | 13.4 | 1.8 |
| Traj_03 | 1456 | 81.5 | 18.7 | 2.4 |
| Traj_04 | 985 | 47.1 | 11.5 | 1.5 |
| Traj_05 | 1289 | 64.8 | 14.2 | 2.1 |

---

## Лицензия

Учебный проект, ЯГТУ кафедра «Кибернетика», 2026.

## Контакты

М. Е. Колобаев — гр. ЦМИИ-20М, ЯГТУ
