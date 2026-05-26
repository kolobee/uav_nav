# Stage 3 Plan: Embedding-голова

**Дата старта:** 2026-05-26  
**Основано на:** `runs/segment/experiments/20260525_222650_yolo/train/weights/best.pt` (v2, mAP50-mask=0.449)

---

## Acceptance criteria

| Критерий | Порог |
|---|---|
| ROC AUC same/diff instance (val) | ≥ 0.80 |
| Recall@1 (same class, val) | ≥ 0.60 |
| Embedding inference latency (CPU, batch=1) | ≤ 15 ms |
| ONNX embedding экспорт | создан и верифицирован |
| Triplet train loss сходится | da (loss ↓ по эпохам) |
| pytest без GPU/датасета | 100% pass |

---

## Задачи

### 0. Экспорт YOLO v2 в ONNX + NCNN (подготовка)

Перед embedding нужно иметь актуальный ONNX для deployment.

```
python -m uav_nav.scripts.train_yolo --export-only ^
    --weights runs\segment\experiments\20260525_222650_yolo\train\weights\best.pt
```

Если скрипт не поддерживает `--export-only`, запустить через Ultralytics напрямую:

```python
from ultralytics import YOLO
model = YOLO(r"runs\segment\experiments\20260525_222650_yolo\train\weights\best.pt")
model.export(format="onnx", opset=17, simplify=True)
model.export(format="ncnn")
```

### 1. Построение instance index

**Скрипт:** `uav_nav/scripts/build_instance_index.py`

**Задача:** для каждого экземпляра объекта в датасете назначить `instance_id`, который одинаков для одного физического объекта в соседних кадрах.

**Алгоритм:**
1. Итерируем по траекториям в порядке кадров.
2. Для каждого кадра парсим YOLO-label: каждая строка = один полигон = один instance.
3. Строим bbox из полигона, вычисляем IoU с инстансами предыдущего кадра той же траектории и класса.
4. Если IoU ≥ 0.4 → тот же instance_id. Иначе → новый.
5. Сохраняем индекс в `data/instance_index/train.parquet` и `val.parquet`.

**Схема parquet:**
```
image_path | class_id | instance_id | bbox_x1y1x2y2 | polygon_flat | frame_idx | condition | trajectory
```

**Acceptance:** каждый `instance_id` встречается ≥ 2 раза (anchor+positive).

### 2. Реализация EmbeddingHead

**Файл:** `uav_nav/perception/embedding_head.py` — переписать полностью.

**Архитектура:**
```
Input: masked crop 96×96 RGB (background → gray [127,127,127])
  ↓
MobileNetV3-Small (pretrained ImageNet, заморожен первые 5 эпох)
  ↓ features (576-d из adaptive avg pool)
Linear(576, 256) → BatchNorm1d(256) → ReLU
  ↓
Linear(256, 128)
  ↓
L2-normalize → embedding 128-d
```

**Интерфейс:**
```python
class EmbeddingHead(nn.Module):
    def forward(self, crops: Tensor) -> Tensor  # (B,3,96,96) → (B,128) L2-norm
    def encode_crop(self, img_bgr, polygon_pts) -> np.ndarray  # удобная обёртка
    def save(self, path: Path) -> None
    @classmethod
    def load(cls, path: Path, device: str) -> "EmbeddingHead"
```

### 3. Triplet Dataset

**Файл:** `uav_nav/data/triplet_dataset.py` (новый)

**Sampling:**
- Anchor: случайный instance из index
- Positive: другой кадр того же instance_id
- Negative: случайный instance того же class_id, другой instance_id

**Crop pipeline:**
1. Загрузить изображение (resize до 640 если не в памяти)
2. Вырезать bbox + 20% padding
3. Залить пиксели вне polygon маски серым (127)
4. Resize → 96×96
5. Нормализовать ImageNet mean/std

### 4. Обучение

**Скрипт:** `uav_nav/scripts/train_embedding.py` — заменить заглушку реализацией.

**Параметры:**
- Loss: `TripletMarginLoss(margin=0.5, p=2)`
- Optimizer: AdamW(lr=1e-4, weight_decay=1e-5)
- Scheduler: CosineAnnealingLR(T_max=50)
- Epochs: 50, batch=64
- Hard negative mining: включить после эпохи 5 (online mining через `BatchHardTripletLoss`)
- Backbone: разморозить полностью после эпохи 5

**Чекпоинты:** `experiments/<date>_embedding/checkpoints/epoch_{N:02d}.pt`  
**Лучшая модель:** `weights/embedding_head.pt`

**Логирование (каждую эпоху):**
- train triplet loss
- val ROC AUC
- val Recall@1

### 5. Оценка

**Файл:** `uav_nav/eval/embedding_eval.py` — реализовать.

**Метрики:**
- ROC AUC: попарное сравнение (same instance vs diff instance), cosine similarity как score
- Recall@k: для каждого query embedding найти k ближайших в val gallery, проверить совпадение instance_id
- t-SNE визуализация (опционально, для диплома)

### 6. ONNX экспорт embedding

**Файл:** `uav_nav/deployment/export_onnx.py` — добавить поддержку EmbeddingHead.

```python
torch.onnx.export(
    head, dummy_input,
    output_path,
    input_names=["crop"],
    output_names=["embedding"],
    dynamic_axes={"crop": {0: "batch"}, "embedding": {0: "batch"}},
    opset_version=17,
)
```

### 7. Тесты

**Файл:** `uav_nav/tests/test_perception/test_embedding_head.py` (новый)

Тест-кейсы (без GPU, без датасета):
- `test_forward_shape`: выход (B, 128)
- `test_l2_normalized`: norm ≈ 1.0
- `test_save_load`: сохранение и восстановление весов
- `test_crop_pipeline`: маскирование polygon работает корректно
- `test_triplet_loss_decreases`: мок-данные, loss идёт вниз за 3 шага

---

## Интерфейсы модулей

### `build_instance_index.py` → выход
```
data/instance_index/
    train.parquet   — ~50k строк
    val.parquet     — ~10k строк
    stats.json      — кол-во instance_id по классам
```

### `EmbeddingHead` → для следующих этапов
Используется в `feature_extractor.py` (связка YOLO + embedding) и `tilm_builder.py`.  
Возвращает `np.ndarray` shape `(128,)` для каждого детекции.

---

## Порядок реализации

1. `build_instance_index.py` — запустить, получить parquet (данные уже есть)
2. `embedding_head.py` — реализовать архитектуру и crop pipeline
3. `triplet_dataset.py` — DataLoader поверх parquet
4. `train_embedding.py` — тренировочный цикл (заменить заглушку)
5. Обучение: запустить, дождаться сходимости
6. `embedding_eval.py` — ROC AUC, Recall@1
7. ONNX экспорт
8. Тесты

---

## Что НЕ делать

- Не пытаться подключать к внутренним фичам YOLO backbone через хуки — слишком хрупко и сломается при смене версии Ultralytics. Использовать независимый MobileNetV3.
- Не хранить 640×640 кропы — только 96×96 masked.
- Не экспортировать единый граф YOLO+embedding сейчас — это делается в этапе 8 (Pi5 deployment).
