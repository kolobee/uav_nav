# План: Этап 7 — Pseudo-simulator + NavigationPipeline

**Статус:** 📋 Не начат  
**Цель:** Сквозной тест всей системы на MidAir GT-данных без реального дрона.  
**Ключевой результат:** Скрипт `simulate_gnss_loss.py` выдаёт ATE с коррекцией TILM vs без.

---

## Мотивация

Этапы 0–6 реализовали все алгоритмические блоки по отдельности. Этап 7 их соединяет:
- Берём реальную MidAir-траекторию с GT-позициями и IMU
- Первая половина — "картографирование" (GNSS активен, строим TILM)
- Вторая половина — "навигация" (GNSS отключён, EKF дрейфует, TILM корректирует)
- Считаем ATE/RPE: IMU-only vs IMU+TILM

---

## Файлы для реализации

### 1. `runtime/pseudo_simulator.py` — `PseudoSimulator`

**Что делает:** воспроизводит MidAir-траекторию покадрово, имитируя потерю GNSS.

```python
@dataclass
class SimulationConfig:
    gnss_loss_start_frac: float = 0.5   # потеря GNSS с 50% траектории
    gnss_noise_std: float = 1.0          # шум GNSS (м), σ=1.0
    imu_noise_acc: float = 0.1
    imu_noise_gyro: float = 0.01
    frame_stride: int = 5                # каждые 5 кадров = 5 Hz при 25 Hz видео
    min_landmarks_per_frame: int = 2

@dataclass
class SimulationResult:
    timestamps: np.ndarray              # (N,)
    gt_positions: np.ndarray            # (N, 3) NED
    estimated_positions: np.ndarray     # (N, 3) с TILM
    dead_reckoning_positions: np.ndarray # (N, 3) IMU-only (без коррекций)
    match_results: list[MatchResult | None]
    gnss_loss_idx: int                  # кадр потери GNSS
    tilm_node_count: int
    corrector_stats: CorrectorStats

class PseudoSimulator:
    def __init__(self, loader, segmenter, embedding_head, config):
        ...

    def run(self, trajectory_id, weather) -> SimulationResult:
        # Фаза 1: mapping (кадры 0..gnss_loss_start_frac)
        #   - EKF с GNSS
        #   - TILMBuilder.process_frame() на каждый кадр
        # Фаза 2: navigation (кадры gnss_loss_start_frac..end)
        #   - EKF без GNSS
        #   - MatchingCorrector.update() на каждый кадр
        #   - Параллельно пишем dead_reckoning (копия EKF без TILM)
```

**Зависимости:** `MidAirLoader`, `YOLOSegmenter`, `EmbeddingHead`,  
`LandmarkExtractor`, `TILMBuilder`, `MatchingCorrector`

**Сложность:** высокая — нужны реальные веса YOLO и EmbeddingHead

**Решение для тестов:** mock-версия `MockSegmenter` + `MockEmbeddingHead`  
(возвращают синтетические маски и дескрипторы без модели)

---

### 2. `eval/trajectory_eval.py` — `TrajectoryEvaluator`

**Что делает:** считает ATE/RPE через `evo`.

```python
@dataclass
class TrajectoryMetrics:
    ate_rmse: float          # Absolute Trajectory Error RMSE (м)
    ate_mean: float
    ate_max: float
    rpe_rmse: float          # Relative Pose Error RMSE (м)
    rpe_mean: float
    improvement_pct: float   # (ate_dead_reckoning - ate_tilm) / ate_dead_reckoning * 100

class TrajectoryEvaluator:
    def evaluate(self, gt, estimated) -> TrajectoryMetrics
    def compare(self, gt, with_tilm, without_tilm) -> dict
    def plot_trajectory_xy(self, result: SimulationResult, save_path)
```

**Зависимости:** `evo` (уже в requirements), `matplotlib`

---

### 3. `runtime/pipeline.py` — `NavigationPipeline` (частичная реализация)

**Что делает:** полный сквозной пайплайн для реального времени.

```python
class NavigationPipeline:
    def initialise(self):
        # Загрузка YOLO weights → YOLOSegmenter
        # Загрузка embedding_head.pt → EmbeddingHead
        # Загрузка TILM файла → TILM + TemporalMatcher
        # Инициализация EKF + MatchingCorrector
        # Инициализация PathFollower (если waypoints заданы)

    def process_frame(self, image_rgb, image_right, imu_acc, imu_gyro,
                      dt, gnss_position=None, timestamp=0.0) -> dict:
        # 1. IMU preintegration → ekf.propagate()
        # 2. YOLO segmentation → SegmentationResult
        # 3. Depth (stub/dummy если нет стерео) → DepthResult
        # 4. LandmarkExtractor.extract() → observations
        # 5. corrector.update(observations, timestamp, gnss_position)
        # 6. PathFollower.compute_command() (опционально)
        # return {"pose": ..., "n_landmarks": ..., "mode": ...}

    def shutdown(self):
        # flush logs, save TILM snapshot
```

**Примечание:** Для ВКР достаточно реализовать шаги 1–5. PathFollower — опционально.

---

### 4. `scripts/simulate_gnss_loss.py`

**Что делает:** end-to-end скрипт для запуска симуляции и вывода результатов.

```bash
python -m uav_nav.scripts.simulate_gnss_loss \
    --weather cloudy \
    --trajectory 3000 \
    --gnss_loss_frac 0.5 \
    --output results/sim_cloudy_3000.json
```

**Вывод:**
```
=== Simulation Results ===
Trajectory: cloudy/trajectory_3000
TILM nodes built: 47
GNSS loss at frame 550/1100

ATE RMSE:
  IMU-only (dead reckoning):  42.3 m
  IMU + TILM correction:       8.7 m
  Improvement:                79.4%

TILM acceptance rate: 68.2%
Mean correction magnitude: 5.2 m
```

---

### 5. Тесты — `tests/test_runtime/test_pseudo_simulator.py`

**Без реальных моделей** — только синтетические данные:

| Тест | Что проверяет |
|---|---|
| `test_mapping_phase_builds_tilm` | После фазы 1 TILM содержит узлы |
| `test_navigation_phase_applies_corrections` | В фазе 2 n_tilm_corrections > 0 |
| `test_dead_reckoning_diverges` | DR дрейфует дальше, чем TILM-версия |
| `test_result_arrays_same_length` | Все массивы результата одной длины |
| `test_gnss_loss_idx_correct` | gnss_loss_idx соответствует конфигу |

---

## Порядок реализации

```
1. trajectory_eval.py   — чистая математика, нет зависимостей на модели
2. pseudo_simulator.py  — mock-версия (для тестов), потом настоящая
3. pipeline.py          — initialise() + process_frame()
4. simulate_gnss_loss.py — финальный скрипт
5. Тесты
6. Запуск на реальных данных (нужны веса YOLO + EmbeddingHead)
```

---

## Что нужно перед запуском на реальных данных

| Артефакт | Где взять | Статус |
|---|---|---|
| `weights/yolo_midair.pt` | Уже обучен (этап 2) | ✅ |
| `weights/embedding_head.pt` | Уже обучен (этап 3) | ✅ |
| MidAir HDF5 данные | Локально на PC | ✅ |
| NCNN-веса для Pi5 | Конвертация через `export_ncnn.py` | 📋 этап 8 |

---

## Критерии завершения этапа 7

- [ ] `PseudoSimulator` работает на синтетических данных (тесты)
- [ ] `TrajectoryEvaluator.evaluate()` считает ATE корректно
- [ ] `NavigationPipeline.initialise()` + `process_frame()` реализованы
- [ ] `simulate_gnss_loss.py` запускается на реальной MidAir-траектории
- [ ] ATE с TILM < ATE без TILM (хотя бы на одной траектории)
- [ ] Все тесты проходят

---

## После этапа 7 → этап 8 (Pi5 deployment)

```
PC:
  python deployment/export_ncnn.py --model yolo  → yolo.param / yolo.bin
  python deployment/export_ncnn.py --model emb   → embedding.param / embedding.bin
  rsync -avz weights/ pi@<IP>:/home/pi/vkr/weights/
  rsync -avz . pi@<IP>:/home/pi/vkr/ --exclude venv --exclude data

Pi5 (SSH):
  pip install -r requirements_pi5.txt   # без torch, с ncnn
  python deployment/benchmark_pi5.py   # замер latency
  python scripts/simulate_gnss_loss.py --device ncnn
```

**Целевые метрики Pi5:**
- YOLO inference: ≤ 100 ms (320×240, NCNN FP32)
- Embedding: ≤ 15 ms (96×96, NCNN)
- EKF + matching: ≤ 5 ms
- **Total pipeline: ≤ 125 ms (~8 FPS)**
