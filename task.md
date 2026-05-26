# Проект: автономная VIO-навигация БПЛА с семантическим landmark-matching при потере GNSS

## Контекст и предыстория
Диплом магистра. Существует НИР, в котором уже выполнено:
- Подготовлен пайплайн извлечения YOLO-формата разметки из MidAir HDF5 (label_map с классами tree, rock, river).
- Обучены и сравнены 9 моделей сегментации (YOLOv8 n/s/m, YOLO11 n/s/m, SegFormer B0/B1/B2).
- Выбрана YOLOv8n-seg + формат NCNN на Raspberry Pi 5 как оптимум по FPS/качеству (~7-14 FPS).
- Обоснована достаточность 13-14 FPS через геометрический расчёт межкадрового смещения + IMU-компенсацию.

Текущий проект достраивает на этом фундаменте полноценную систему навигации с уникальной научной новизной: **Trajectory-Indexed Landmark Memory с Instance Embeddings и темпорально-ограниченным matching**.

## Научная новизна (для формулировок в коде, документации, README)
Семантико-инстансный place recognition с темпорально-ограниченным сопоставлением для коррекции инерциальной навигации БПЛА на пред-картографированных маршрутах при работе в реальном времени на CPU-only edge-устройстве (Raspberry Pi 5).

## Целевая платформа
- **Разработка:** Linux/Windows, Python 3.11, GPU CUDA для обучения.
- **Развёртывание:** Raspberry Pi 5 (Cortex-A76, 8GB RAM), без NPU, без AI HAT.
- **Симуляция:** pseudo-closed-loop поверх MidAir для воспроизводимых экспериментов. Опционально AirSim для качественного видео-демо.

## Датасет
**MidAir** (https://midair.uliege.be).
HDF5 структура:
- `/trajectory_XXXX/camera_data/color_left` — RGB левая (1024×1024, ссылки на PNG)
- `/trajectory_XXXX/camera_data/color_right` — RGB правая
- `/trajectory_XXXX/camera_data/depth` — GT depth maps
- `/trajectory_XXXX/camera_data/stereo_disparity` — GT disparity (если есть)
- `/trajectory_XXXX/camera_data/segmentation` — GT semantic segmentation
- `/trajectory_XXXX/imu/accelerometer`, `/imu/gyroscope` — IMU 100 Hz
- `/trajectory_XXXX/groundtruth/position`, `/attitude`, `/velocity` — GT 100 Hz
- `/trajectory_XXXX/gps/position` — GPS

Калибровка: проверить в HDF5 атрибутах и spec на сайте MidAir. Стерео baseline должен быть указан в документации — найти и зафиксировать в конфиге.

## Технологический стек
- Python 3.11, NumPy, SciPy
- OpenCV 4.x (StereoSGBM, ORB как референс)
- PyTorch для embedding-головы (обучение), ONNX/NCNN для развёртывания
- Ultralytics YOLO для базовой YOLOv8n-seg
- h5py для MidAir
- NetworkX для графа TILM
- evo для метрик ATE/RPE
- rerun.io для визуализации (предпочтительно перед Open3D — лучше для робототехники)
- loguru для логирования
- hydra или omegaconf для конфигов
- pytest для тестов

**Не использовать GTSAM** — установка на Pi5 проблематична, EKF реализуем на NumPy.

## Архитектура проекта (пакет `uav_nav/`)

```
uav_nav/
├── data/
│   ├── midair_loader.py            # HDF5 reader, синхронизация камер+IMU
│   ├── calibration.py              # Camera intrinsics, baseline, IMU noise
│   └── dataset_builder.py          # Улучшенный конвертер из НИР: multi-trajectory, multi-weather, корректный split
│
├── perception/
│   ├── yolo_segmenter.py           # YOLOv8n-seg, backends: ultralytics/ONNX/NCNN
│   ├── embedding_head.py           # Triplet-обучаемая голова для instance discrimination
│   ├── feature_extractor.py        # Связка YOLO+embedding: возвращает Detection с embedding
│   ├── stereo_depth.py             # StereoSGBM для real-time + GT-depth для валидации
│   └── landmark_extractor.py       # Detection → 3D Landmark через depth+pose
│
├── memory/
│   ├── tilm.py                     # Trajectory-Indexed Landmark Memory
│   ├── tilm_builder.py             # Build/merge mode: первый пролёт собирает карту
│   ├── temporal_matcher.py         # Поиск кандидатов в temporal window + embedding cosine
│   └── place_descriptor.py         # Опциональный глобальный дескриптор узла (для re-localize в LOST)
│
├── estimation/
│   ├── imu_preintegrator.py        # Forster preintegration между кадрами
│   ├── ekf_vio.py                  # Error-state EKF на NumPy (16-state nominal, 15-state error)
│   └── pose_utils.py               # SO(3), кватернионы, утилиты для манипуляций позой
│
├── planning/
│   ├── path_follower.py            # State machine ON_PATH/DEVIATED/RECOVERING/LOST + waypoint emission
│   ├── adaptive_corridor.py        # Динамическая ширина D_max по плотности TILM
│   ├── mission_modes.py            # NORMAL, MAPPING, GNSS_LOST, RTH, EMERGENCY
│   └── rejoin_planner.py           # Геометрический расчёт rejoin point на reference path
│
├── runtime/
│   ├── pipeline.py                 # Главный цикл интеграции всех модулей
│   ├── pseudo_simulator.py         # Pseudo-closed-loop: подмена кадров MidAir по требуемой позе
│   ├── airsim_bridge.py            # Опциональная интеграция с AirSim (если будет реализовываться)
│   ├── monitor.py                  # Метрики качества VIO в runtime: cov, innovation, matches
│   └── logger.py                   # rerun.io интеграция
│
├── deployment/
│   ├── export_onnx.py              # YOLO + embedding-голова → ONNX
│   ├── export_ncnn.py              # ONNX → NCNN с FP16
│   ├── benchmark_pi5.py            # Профилирование на целевой платформе
│   └── pi5_runtime.py              # Урезанный pipeline для Pi5
│
├── eval/
│   ├── trajectory_eval.py          # ATE/RPE через evo
│   ├── embedding_eval.py           # ROC, mAP@k для embedding-головы
│   ├── matching_eval.py            # Precision/Recall для matching
│   ├── scenario_runner.py          # Запуск экспериментов 5.5-5.9 из плана
│   └── ablation.py                 # Все ablation studies
│
├── configs/
│   ├── default.yaml
│   ├── pi5.yaml                    # Урезанные параметры для целевой платформы
│   ├── label_map.json
│   └── experiments/                # Конфиги для каждого эксперимента из плана
│       ├── 5_3_cross_weather.yaml
│       ├── 5_5_tilm_correction.yaml
│       └── ...
│
├── scripts/
│   ├── 01_build_dataset.py         # Эксп. 1 шаг 1: multi-trajectory YOLO dataset
│   ├── 02_train_yolo.py            # Обучение базовой YOLOv8n-seg
│   ├── 03_train_embedding.py       # Обучение embedding-головы через triplet loss
│   ├── 04_build_tilm.py            # Mapping mission на test-траектории
│   ├── 05_simulate_gnss_loss.py    # Главный демо-скрипт + Эксп. 5.5
│   ├── 06_return_to_path_demo.py   # Сценарии возврата 5.7
│   ├── 07_benchmark_pi5.py         # Замеры 5.8
│   └── 08_run_all_experiments.py   # Все эксперименты главы 5
│
├── tests/                           # pytest unit + integration
│
└── docs/
    ├── architecture.md
    ├── stage_N_report.md            # Отчёты после каждого этапа разработки
    └── experiments_log.md
```

## Требования по компонентам (ключевые)

### `dataset_builder.py` (важно — переделать дизайн НИР)
Текущий скрипт НИР работает на одной траектории. Переписать:
- Принимать список (траектория, погода) пар в конфиге.
- Параллельная обработка через `multiprocessing.Pool`.
- Split **по траекториям, не по кадрам**. Конфиг train/val/test split явный, фиксированный seed.
- Объединение классов из MidAir label_map в целевые `tree/rock/river` (некоторые подклассы MidAir могут смежать).
- Балансировка через oversampling редких классов.
- Для эксп. 5.4 — отдельный режим «leave-one-trajectory-out».
- **Дополнительный выход:** для каждой детекции в обучающем датасете embedding-головы сохранить `instance_id` через connected components, чтобы потом строить triplets «one-instance / different-instance».

### `embedding_head.py`
Архитектура:
```
Input: YOLO backbone features at ROI (через ROIAlign по bbox), 7×7×N
  ↓
Conv 3×3, 256 channels, ReLU
  ↓
Conv 3×3, 128 channels, ReLU
  ↓
Adaptive avg pool → 128-d
  ↓
L2-normalize
```
Обучение через **TripletMarginLoss** (margin=0.5) или **NT-Xent**:
- Anchor — crop экземпляра объекта.
- Positive — тот же экземпляр в соседнем кадре (instance_id из dataset_builder).
- Negative — другой экземпляр того же класса.
- Hard negative mining после первой эпохи.
Сохранение весов отдельно, инференс через тот же YOLO backbone (без дублирования).
Экспорт в ONNX как часть единого графа `YOLO + embedding`.

### `tilm.py`
Класс `TrajectoryIndexedLandmarkMap`:
- Список `Landmark` объектов, упорядоченных по `expected_time_s`.
- Каждый Landmark: `id`, `class`, `embedding (128-d)`, `expected_time_s`, `expected_arc_length_m`, `position_3d_world`, `bearing_from_trajectory`, `observation_count`, `confidence`.
- Метод `query_candidates(t_current, dt_window, class_filter)` → список Landmark в окне.
- Метод `match(detection, t_current, dt_window)` → лучший кандидат по cosine + score.
- Сериализация в HDF5 или pickle, версионирование (атрибут format_version).
- Метод `density(arc_length, radius)` для adaptive_corridor.

### `tilm_builder.py`
Режим mapping: дрон летит с GPS, на каждой детекции:
- Вычисляется 3D позиция в мире (через depth + pose из GPS+IMU).
- Считается embedding.
- Если есть близкий по позиции и классу landmark в TILM с похожим embedding → обновляется (накапливаем больше observation_count, усредняем embedding и позицию через EMA).
- Иначе создаётся новый.
По завершении: pruning редких landmarks (observation_count < 2 → удалить), сохранение TILM на диск.

### `temporal_matcher.py`
```python
def match(detection, current_time_estimate, tilm, dt_window=5.0, class_filter=True) -> Match|None:
    candidates = tilm.query_candidates(current_time_estimate, dt_window, class_filter=detection.class)
    if not candidates: return None
    best = argmax_cosine(detection.embedding, [c.embedding for c in candidates])
    if cosine(detection.embedding, best.embedding) < THRESHOLD: return None
    return Match(detection, best, score)
```
THRESHOLD выбирается по эксп. 5.6 — порог, на котором precision matching ≥ 0.95.

### `ekf_vio.py`
Error-state EKF на NumPy. State (nominal 16D):
- position (3), velocity (3), attitude quaternion (4), accel bias (3), gyro bias (3).
Error-state (15D в касательном пространстве, attitude через small-angle).
**Predict step:** IMU preintegration между кадрами.
**Update steps:**
1. **TILM-correction:** при match — позиция landmark известна, наблюдается через камеру + depth → виртуальное наблюдение позы (или хотя бы position) дрона.
2. **GPS-correction:** в режиме MAPPING.
3. **Zero-velocity update:** при детектировании зависания (опционально).
Корректная multiplicative update для кватерниона. Все Jacobians выводятся аналитически и тестируются numerically через `scipy.optimize.check_grad`.

### `path_follower.py`
State machine:
```python
state: ON_PATH | DEVIATED | RECOVERING | LOST

def update(current_pose, current_detections, time_since_gnss_loss):
    e_xt, e_at, nearest_segment = compute_path_errors(current_pose, reference_path)
    D_max = adaptive_corridor.compute(nearest_segment.arc_length, tilm)
    
    expected_landmarks = tilm.query_candidates(time_since_gnss_loss, dt_window=5.0)
    visible_match = any(match(d, time_since_gnss_loss, tilm) for d in current_detections)
    
    # Переходы с гистерезисом
    if state == ON_PATH:
        if e_xt > D_max * 0.4: state = DEVIATED
    elif state == DEVIATED:
        if e_xt > D_max: state = RECOVERING
        elif e_xt < D_max * 0.2 for N consecutive frames: state = ON_PATH
    elif state == RECOVERING:
        if visible_match and back_in_corridor: state = ON_PATH
        elif time_in_state > T_lost_threshold and not visible_match: state = LOST
    elif state == LOST:
        if global_relocalize_succeeded: state = RECOVERING
    
    return emit_waypoint(state, current_pose, reference_path, e_xt, e_at)
```
emit_waypoint выдаёт целевую точку через 1-2 секунды полёта вдоль референса (или к rejoin_point если RECOVERING).

### `rejoin_planner.py`
Не «возврат к ближайшей точке» (это создаёт перпендикулярный заход). Используется look-ahead pure-pursuit:
- Найти точку на reference path впереди по маршруту на расстоянии L = max(20м, current_speed × 2с).
- Это и есть rejoin_point.
- Вектор от текущей позы к rejoin_point — желаемое направление.
Это даёт плавный «касательный» возврат, не оверкоррекцию.

### `adaptive_corridor.py`
```python
def compute(arc_length: float, tilm: TILM) -> float:
    density = tilm.density(arc_length, radius=50.0)
    # density: landmarks per meter of path
    
    D_min = 5.0  # м, минимум независимо от плотности
    D_max_cap = 50.0  # м, максимум
    
    # Чем плотнее — тем шире коридор (мы уверены в локализации)
    D_max = D_min + (D_max_cap - D_min) * min(1.0, density / DENSITY_REFERENCE)
    return D_max
```
DENSITY_REFERENCE подбирается так, чтобы средняя плотность TILM на типичной траектории давала D_max ≈ 25м.

### `pseudo_simulator.py`
Главный инструмент воспроизводимых экспериментов. Алгоритм:
- Берётся MidAir test-траектория со всеми GT pose.
- На каждом шаге симуляции pipeline выдаёт желаемый waypoint.
- Симулятор симулирует движение к waypoint через простую модель (точечная масса с ограничениями по скорости/ускорению) → получает новую целевую позу.
- Из MidAir достаётся кадр с **ближайшей по позиции и направлению** GT pose.
- Если ближайший кадр слишком далёк (>15м или >30°) — возвращается «ошибка камеры» (как будто bad data) и pipeline должен обработать. Это автоматически ограничивает дальность отклонения и моделирует деградацию восприятия.
- IMU данные синтезируются из разностей симулированных поз с добавлением шума по модели MEMS-IMU.

### `airsim_bridge.py` (опционально)
Если будет реализовываться:
- Подключение к AirSim Multirotor API.
- Команды через `moveToPositionAsync`.
- Получение Image, Lidar, Imu сенсоров.
- Заглушка с тем же интерфейсом, что `pseudo_simulator.py`, чтобы pipeline не различал источник.

## Этапы разработки

Выполнять последовательно. После каждого этапа — отчёт в `docs/stage_N_report.md` с метриками и графиками.

### Этап 0. Setup
- Создание структуры проекта.
- pyproject.toml, requirements.txt.
- Базовая конфигурация loguru + hydra.
- Заготовки pytest.
- Pre-commit hooks (black, ruff).

### Этап 1. Data layer
- `midair_loader.py`, `calibration.py`.
- Верификация calibration: визуально проверить, что 3D точка через depth+intrinsic совпадает с pixel-проекцией.
- Тест синхронизации IMU↔Camera.
- `dataset_builder.py` с multi-trajectory, multi-weather, instance_id сохраняется.
- **Артефакт этапа:** YOLO-датасет на нескольких траекториях/погодах, отчёт с распределением классов и количеством instance-IDs.

### Этап 2. Базовая YOLO + бенчмарки
- Обучить YOLOv8n-seg на новом мульти-датасете.
- Сравнить с моделью НИР: разница в mAP на одиночной траектории и на новой test-траектории.
- Экспорт в ONNX и NCNN.
- Замеры FPS на Pi5.
- **Артефакт:** улучшенные веса, отчёт с таблицей mAP cross-weather/cross-trajectory.
- **Эксперимент 5.3 и 5.4 из плана выполняются здесь.**

### Этап 3. Embedding-голова
- Реализация архитектуры.
- Sampler для triplets из dataset_builder данных.
- Обучение через TripletMarginLoss.
- Hard negative mining.
- Метрики на validation: ROC AUC same/diff instance.
- Экспорт совмещённого графа YOLO+embedding в ONNX.
- **Артефакт:** обученная embedding-голова, отчёт с ROC и mAP@k.
- **Эксперимент 5.5 из плана выполняется здесь.**

### Этап 4. Базовый VIO
- `imu_preintegrator.py` с numerical jacobians.
- `ekf_vio.py` с predict/update GPS.
- Тест: на mapping mission с GPS — ATE ≤ 1м (тривиальный, ground-truth).
- Тест без GPS: накопление дрейфа в пределах теории (1-2% от пути).
- **Артефакт:** работающий EKF.

### Этап 5. TILM
- `tilm.py`, `tilm_builder.py`.
- Mapping mission: пролёт по test-траектории с GPS, построение TILM.
- Визуализация TILM в rerun.io (позиции, classes, embeddings через t-SNE).
- **Артефакт:** TILM-файлы для всех test-траекторий, визуализация.

### Этап 6. Matching и коррекции
- `temporal_matcher.py`.
- Интеграция в EKF: TILM-match → update step.
- **Эксперимент 5.6 из плана здесь:** подбор THRESHOLD и dt_window.
- **Эксперимент 5.7 из плана здесь:** главный — pure IMU vs class-only vs full TILM.
- **Артефакт:** таблица ATE для всех режимов и дистанций.

### Этап 7. Path-follower и режимы миссии
- `path_follower.py`, `adaptive_corridor.py`, `rejoin_planner.py`, `mission_modes.py`.
- `pseudo_simulator.py`.
- **Эксперимент 5.8 и 5.9 из плана здесь:** adaptive vs fixed corridor, 4 сценария возврата на маршрут.
- **Артефакт:** видео сценариев возврата, таблицы метрик.

### Этап 8. Pi5 deployment
- Полный pipeline в `pi5_runtime.py`.
- Профилирование, оптимизация bottlenecks.
- **Эксперимент 5.10 из плана здесь:** end-to-end latency, FPS, RAM, термотест.
- **Артефакт:** работающий pipeline на Pi5, отчёт.

### Этап 9. Ablations
- `ablation.py` запускает все варианты из плана 5.11.
- Сводная таблица для главы 5.11 диплома.

### Этап 10. Финальное демо (опционально AirSim)
- Если развернуто — `airsim_bridge.py`, записанное демо-видео.
- Если нет — расширенный набор pseudo-simulator сценариев в видео-формате через rerun.io recording.

## Требования к качеству кода
- Type hints везде (mypy strict).
- Docstrings в Google style.
- Logging через loguru, не print.
- Конфиги через hydra/omegaconf, никаких magic numbers в коде.
- Unit-тесты на критичные модули (EKF, matcher, path_follower) с >80% coverage.
- Integration test: полный 30-секундный пролёт в pseudo-simulator должен укладываться в фиксированное время и давать воспроизводимые метрики (фиксированный seed).
- `Makefile` или `pyproject.toml [project.scripts]` с командами: `train_yolo`, `train_embedding`, `build_tilm`, `run_experiment`, `benchmark_pi5`.

## Что НЕ делать
- Не использовать GTSAM/Ceres — EKF на NumPy.
- Не пытаться реализовать полётный контроллер — выдача только waypoints.
- Не использовать GT depth в runtime — только в валидации. В runtime — StereoSGBM (когда есть stereo) или фиксированная глубина для landmark height-estimation (когда mono).
- Не делать плотные voxel/mesh карты — только sparse TILM.
- Не вшивать AirSim как обязательную зависимость — основная разработка через pseudo_simulator.

## Поэтапная стратегия работы
В начале каждого этапа создавать `docs/stage_N_plan.md` с:
1. Декомпозицией задач,
2. Интерфейсами модулей,
3. Acceptance criteria.
После реализации — `docs/stage_N_report.md` с фактическими результатами и сравнением с критериями.

Если по ходу обнаруживаются архитектурные проблемы — фиксировать в `docs/decisions.md` (ADR-style: контекст, решение, последствия). Не молча менять интерфейсы между модулями.

При запуске тяжёлых обучений (YOLO, embedding) логи и checkpoints — в `experiments/<date>_<name>/`, не в основной репозиторий.