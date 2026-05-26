# Stage 2 Plan: Базовая YOLO + Бенчмарки

**Дата:** 2026-05-24  
**Зависит от:** Stage 1 (Data Layer) — Complete

---

## 0. Статус выполнения (обновлено 2026-05-24)

| # | Задача | Статус |
|---|--------|--------|
| 2.1 | YOLOSegmenter | ✅ Готово |
| 2.2 | Обучить YOLOv8n-seg | 🔄 Датасет пересобирается |
| 2.3 | ONNX-экспорт | ✅ Готово (скрипт) |
| 2.4 | NCNN-экспорт | ✅ Готово (скрипт) |
| 2.5 | Pi5Benchmark | ✅ Готово (скрипт) |
| 2.6 | YOLOEvaluator | ✅ Готово (скрипт) |
| 2.7 | eval_yolo.py | ✅ Готово |
| 2.8 | Конфиг 5.4 LOO | ✅ Готово |
| 2.9 | Юнит-тесты | ⏳ После обучения |
| 2.10 | Отчёт этапа | ⏳ После обучения |

**Ключевые исправления датасета:**
- Камера: `color_left` (было `color_down` — критический баг, деревьев почти не было)
- `label_map.json`: 4 класса — tree/rock/river/road; ID5→background; ID10,11→road
- `dataset_builder.py` `_extract_yolo_labels`: итерация по сырым MidAir ID вместо merged semantic mask → деревья не сливаются, no erosion, в разы быстрее
- `frame_stride=10` (2026-05-24): датасет без ограничений вырастал до ~20 ГБ; добавлен параметр `frame_stride` в `MidAirLoader` и конфиг — берётся каждый 10-й кадр траектории, размер датасета ~в 10 раз меньше. См. DD-006.

**Следующий шаг:** пересобрать `data/yolo_v2` (с `frame_stride=10`), проверить `visualize_labels.py`, запустить обучение.

---

## 1. Декомпозиция задач

| # | Задача | Файл(ы) | Критерий готовности |
|---|--------|---------|----------------------|
| 2.1 | Реализовать `YOLOSegmenter` (load/predict/predict_batch/warmup) | `perception/yolo_segmenter.py` | Все методы работают; тесты проходят без GPU |
| 2.2 | Обучить YOLOv8n-seg на мульти-датасете | `scripts/train_yolo.py` | Скрипт запускается; checkpoint сохраняется |
| 2.3 | Реализовать ONNX-экспорт и верификацию | `deployment/export_onnx.py` | ONNX-файл создаётся; выходы совпадают с PT |
| 2.4 | Реализовать NCNN-экспорт | `deployment/export_ncnn.py` | .param + .bin создаются через ultralytics/pnnx |
| 2.5 | Реализовать `Pi5Benchmark` (benchmark_callable / report / save_csv) | `deployment/benchmark_pi5.py` | Замеряет любой callable; CSV пишется |
| 2.6 | Реализовать `YOLOEvaluator` (mAP, cross-weather/trajectory таблицы) | `eval/yolo_eval.py` | Таблица mAP генерируется из ultralytics val() |
| 2.7 | Скрипт оценки (Эксп 5.3, 5.4) | `scripts/eval_yolo.py` | Гидра-скрипт запускает cross-weather и LOO |
| 2.8 | Конфиг Эксп 5.4 leave-one-trajectory-out | `configs/experiments/5_4_leave_one_out.yaml` | Конфиг загружается, переопределяет dataset |
| 2.9 | Юнит-тесты (без GPU, без датасета) | `tests/test_perception/` | pytest -m "not requires_gpu and not requires_midair" green |
| 2.10 | Отчёт этапа | `docs/stage_2_report.md` | Заполняется после запуска обучения |

---

## 2. Интерфейсы модулей

### `YOLOSegmenter`

```python
class YOLOSegmenter:
    def load(self) -> None:
        # Поддерживает .pt (ultralytics), .onnx (onnxruntime)
        
    def predict(self, image: np.ndarray) -> SegmentationResult:
        # image: (H, W, 3) uint8 RGB
        
    def predict_batch(self, images: list[np.ndarray]) -> list[SegmentationResult]:
        # Пакетный инференс (одним вызовом ultralytics)
        
    def warmup(self, n_iters: int = 3) -> None:
        # n_iters dummy-прогонов для прогрева JIT/кеша
```

### `SegmentationResult`

```python
@dataclass
class SegmentationResult:
    masks: np.ndarray        # (N, H, W) bool
    class_ids: np.ndarray    # (N,) int
    class_names: list[str]
    scores: np.ndarray       # (N,) float32
    boxes: np.ndarray        # (N, 4) float32 xyxy
    image_hw: tuple[int, int]
    
    def filter_by_class(self, class_name: str) -> SegmentationResult
    def merged_mask(self) -> np.ndarray  # (H, W) uint8 class IDs
```

### `YOLOEvaluator`

```python
class YOLOEvaluator:
    def evaluate_on_yaml(
        self, weights: Path, dataset_yaml: Path, split: str = "test"
    ) -> SegMetrics

    def cross_weather_table(
        self, weights: Path, yolo_roots: dict[str, Path]
    ) -> pd.DataFrame  # строки = weather, столбцы = mAP50/mAP/per-class

    def leave_one_out_table(
        self, weights_per_fold: dict[str, Path], dataset_yamls: dict[str, Path]
    ) -> pd.DataFrame
    
    def save_table(self, df: pd.DataFrame, out_path: Path) -> None
```

### `Pi5Benchmark`

```python
class Pi5Benchmark:
    def benchmark_callable(
        self, fn: Callable, module_name: str, target_ms: float = nan
    ) -> BenchmarkResult
    
    def report(self) -> str        # таблица в консоль
    def save_csv(self) -> Path     # сохранение в output_dir/benchmark.csv
```

---

## 3. Acceptance Criteria

| Критерий | Метрика | Порог |
|----------|---------|-------|
| mAP50-mask (val, multi-weather train) | ultralytics val() | ≥ 0.55 (ориентировочно; точное пороговое значение после первого прогона) |
| Прирост mAP vs НИР-модель (single-weather) | Δ mAP50 | ≥ +0.03 на новой test-траектории |
| ONNX выход совпадает с PT | max abs diff | ≤ 1e-4 |
| FPS на Pi5 @ imgsz=640 | 1000/mean_ms | ≥ 7 FPS (≤ 143 ms) |
| Все тесты без GPU/датасета | pytest | 100% pass |
| Эксп 5.3 таблица | cross-weather mAP50 | Присутствует для 2 weather пар |
| Эксп 5.4 таблица | LOO mAP50 среднее | ≤ 0.05 разброс между фолдами |

---

## 4. Порядок запуска (руководство)

```bash
# 1. Построить YOLO датасет (если не сделано на этапе 1)
MIDAIR_ROOT=D:/data/MidAir uav-nav-build-dataset \
  dataset.midair_root="D:/data/MidAir" \
  dataset.yolo_root="data/yolo"

# 2. Обучение
uav-nav-train-yolo \
  yolo.imgsz=640 \
  yolo.epochs=100 \
  yolo.device=cuda

# 3. Оценка (кросс-погодная)
uav-nav-eval-yolo \
  +experiment=5_3_cross_weather \
  eval.weights_path=weights/yolo_midair/weights/best.pt

# 4. Оценка (leave-one-out)
uav-nav-eval-yolo \
  +experiment=5_4_leave_one_out \
  eval.weights_path=weights/yolo_midair/weights/best.pt

# 5. Экспорт
uav-nav-export \
  export.weights=weights/yolo_midair/weights/best.pt \
  export.format=onnx

uav-nav-export \
  export.weights=weights/yolo_midair/weights/best.pt \
  export.format=ncnn

# 6. Benchmark (запускать на Pi5)
uav-nav-benchmark \
  benchmark.weights=weights/yolo_midair.ncnn \
  benchmark.n_runs=200
```
