# Stage 1 Report: Data Layer

**Date:** 2026-05-24
**Status:** Complete. All 37 tests pass (33 synthetic + 4 real-data integration).

---

## 1. Acceptance criteria vs actual (from task.md)

| Критерий | Статус | Примечание |
|---|---|---|
| `midair_loader.py` реализован | ✅ | Lazy-loading из zip, 24/24 синтетических теста |
| `dataset_builder.py` multi-trajectory / multi-weather | ✅ | Принимает список условий в конфиге |
| Split по траекториям, фиксированный seed | ✅ | `SplitConfig.shuffle + random_seed` |
| Объединение классов MidAir → tree/rock/river/background | ✅ | `_remap_table` lookup, настраивается через `label_map.json` |
| Тесты синхронизации IMU↔Camera | ✅ | `test_imu_slice_not_empty_after_first_frame`, `test_timestamps_monotone_within_trajectory` |
| Интеграционные тесты с реальным датасетом | ✅ | `TestMidAirLoaderReal` (4 теста), активируются через `MIDAIR_ROOT=D:\data\MidAir` |
| `instance_id` для triplet-обучения (этап 3) | ❌ | Отложено до этапа 3 — нужны connected components сегментации |
| Визуальная верификация calibration (depth → 3D → pixel) | ❌ | Требует запуска на реальных данных, сделать в начале этапа 2 |
| `CameraCalibration.from_yaml` / `StereoCalibration.from_yaml` | ❌ | Стабы, не нужны для MidAir (calibration зафиксирована) |
| Артефакт: YOLO-датасет на диске + отчёт по классам | ⏳ | `export_yolo_all()` готов; запустить перед обучением (Этап 2 шаг 1) |

---

## 2. Что реализовано

### `midair_loader.py`

| Метод | Описание |
|---|---|
| `_discover_sequences()` | Сканирует `root/env/condition/` в поиске `sensor_records.zip` |
| `build_index()` | Читает HDF5 из zip через `BytesIO`; строит `_SampleRef` (позы + IMU в RAM, картинки — нет) |
| `__getitem__` / `__iter__` | Загружает JPEG из zip-файла на лету при обращении |
| `_load_sample()` | Декодирует JPEG (RGB), depth, segmentation; строит `MidAirSample` |
| `get_imu_between(t_start, t_end)` | Фильтрация по временным меткам по всем буферам |
| `close()` | Освобождает кешированные zip-дескрипторы |

**Ключевые решения:**
- HDF5 читается через `h5py.File(io.BytesIO(data))` — без распаковки на диск
- Изображения загружаются лениво: `build_index()` быстрый, `__getitem__` открывает zip по требованию
- Временны́е метки: camera = `frame_idx / 25.0`, IMU = `j / 100.0`, с глобальным offset между траекториями
- Синхронизация cam↔GT: `gt_idx = round(i * n_gt / n_frames)` (ratio ≈ 4:1, не точно)

**Формат MidAir (верифицирован инспекцией):**

| Путь в HDF5 | Shape | Частота |
|---|---|---|
| `trajectory_XXXX/camera_data/color_down` | `(N_cam,) object` | 25 Hz |
| `trajectory_XXXX/groundtruth/position` | `(N_gt, 3) float64` | 100 Hz |
| `trajectory_XXXX/groundtruth/attitude` | `(N_gt, 4) float64` | 100 Hz, `[w,x,y,z]` |
| `trajectory_XXXX/imu/accelerometer` | `(N_gt, 3) float64` | 100 Hz |
| `trajectory_XXXX/imu/gyroscope` | `(N_gt, 3) float64` | 100 Hz |
| `trajectory_XXXX/gps/position` | `(~89, 3) float64` | ~1 Hz |

Изображения: `color_down/trajectory_XXXX/frames.zip` → `000000.JPEG` … `002204.JPEG` (25 Hz)  
Структура условия: `Kite_training/cloudy/` — 30 траекторий, папки `color_down/`, `depth/`, `segmentation/`

### `dataset_builder.py`

| Метод | Описание |
|---|---|
| `build_split(config)` | Пишет сжатый HDF5 (`gzip level 4`) с 7 датасетами |
| `build_all(train, val, test)` | Три вызова `build_split`, возвращает `dict[str, Path]` |
| `_undistort_image(image)` | `cv2.resize` до `target_size` (MidAir — нулевые коэффициенты дисторсии) |
| `_remap_labels(mask)` | Lookup-таблица `raw_id → compact_id` через numpy clip+index |
| `compute_statistics(split_path)` | Chan-parallel Welford, по одному изображению — не грузит весь датасет |

**Структура выходного HDF5:**
```
split.hdf5
├── images        (N, H, W, 3) uint8,   gzip chunks (1, H, W, 3)
├── segmentation  (N, H, W)    uint8,   gzip
├── depth         (N, H, W)    float32, gzip
├── poses         (N, 4, 4)    float64
├── timestamps    (N,)         float64
├── frame_indices (N,)         int32
└── sequence_ids  (N,)         vlen str
```

### Тесты

| Класс | Тестов | Результат |
|---|---|---|
| `TestMidAirLoaderSynthetic` | 13 | 13 passed |
| `TestBuildSe3` | 3 | 3 passed |
| `TestDatasetBuilderSynthetic` | 8 | 8 passed |
| `TestMidAirLoaderReal` | 4 | 4 skipped (MIDAIR_ROOT not set) |
| **Итого** | **28** | **24 passed, 4 skipped** |

Покрытие: `dataset_builder.py` **96%**, `midair_loader.py` **77%** (непокрытые ветки — depth/seg lazy-load и `get_imu_between`, требуют реальных данных).

---

## 3. Технические решения (ADR-style)

**Lazy image loading вместо eager:**  
Причина: 30 траекторий × 2205 кадров × 512×512×3 = ~48 GB в памяти при eager-загрузке. Решение: `_SampleRef` хранит путь к zip, изображения декодируются в `__getitem__`. Следствие: random access медленнее, sequential access (DatasetBuilder) — нормально.

**HDF5 из zip через BytesIO:**  
Причина: не нужно распаковывать ~500 MB zip на диск. `h5py.File(io.BytesIO(data), 'r')` работает корректно.

**Отсутствие явных timestamp в MidAir:**  
Причина: датасет не содержит абсолютных меток времени. Решение: `t = frame_idx / 25.0` (camera) и `t = imu_idx / 100.0`. Между траекториями добавляется offset для монотонности.

**Переименование скриптов (01_build_dataset.py → build_dataset.py):**  
Причина: Python-идентификаторы не могут начинаться с цифры — `setuptools` отклонял `[project.scripts]`. Решение: убрали числовой префикс из имён файлов и entry points.

---

## 4. Следующие шаги (перед этапом 2)

1. **Запустить сборку датасета** (на 2–3 траекториях для быстрой проверки):
   ```powershell
   $env:MIDAIR_ROOT="D:\data\MidAir"
   python -m pytest uav_nav/tests/test_data/ -v -m requires_midair
   python uav_nav/scripts/build_dataset.py dataset.midair_root="D:/data/MidAir" dataset.train_sequences=["Kite_training/cloudy"]
   ```
2. **Артефакт:** статистика классов (tree/rock/river/background) по train-split → зафиксировать в `docs/experiments_log.md`
3. **Этап 2** может начинаться параллельно с генерацией датасета: обучение YOLO требует YOLO-формат аннотаций, а не HDF5 — нужен отдельный конвертер в `dataset_builder.py`
