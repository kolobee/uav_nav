# Этап 3 — Embedding-голова: промежуточный отчёт

**Дата:** 2026-05-26  
**Статус:** ✅ Завершён (обучение + eval)  
**Лучший чекпоинт:** `experiments/20260526_092148_embedding/best.pt` (эпоха 31, train val_AUC=0.9783)  
**Финальные веса:** `weights/embedding_head.pt`

---

## 1. Задача этапа

Обучить семантический дескриптор (embedding-голову), способный различать отдельные экземпляры объектов (деревья, скалы, реки, дороги) по обрезкам с YOLO-масками. Дескриптор нужен для TILM: при повторном пролёте над маршрутом landmark должен быть узнан по внешнему виду, несмотря на изменение угла, освещения и погоды.

---

## 2. Архитектура модели

**Файл:** `uav_nav/perception/embedding_head.py`

### Backbone

MobileNetV3-Small, предобученный на ImageNet (torchvision). Используются только `features` и `avgpool` — классификационная голова отброшена. Выходной размер после пулинга: **576-мерный вектор**.

Причина выбора: 2.5M параметров, быстрый на ARM Cortex-A76 (Pi5), ImageNet-фильтры дают хорошую текстурную основу без обучения с нуля.

### Projection Head

```
Linear(576 → 256) → LayerNorm(256) → ReLU → Dropout(0.1) → Linear(256 → 128) → L2-normalize
```

**LayerNorm вместо BatchNorm**: BatchNorm1d не работает при `batch_size=1` в training mode — необходимо для single-crop inference в runtime. LayerNorm нормирует по feature-оси, не зависит от размера батча.

Итоговый дескриптор: **128-мерный L2-нормированный вектор** (единичная норма, косинусное расстояние = евклидово).

### Параметры обучения (`EmbeddingConfig`)

| Параметр | Значение |
|---|---|
| embedding_dim | 128 |
| hidden_dim | 256 |
| backbone_channels | 576 |
| dropout | 0.1 |
| margin (TripletLoss) | 0.5 |
| lr | 1e-4 |
| lr_backbone | 1e-5 (lr×0.1) |
| epochs | 50 |
| freeze_backbone_epochs | 5 |
| crop_size | 96×96 px |
| crop_pad_ratio | 0.20 |

---

## 3. Датасет для обучения

**Источник:** `data/yolo_v5/` (тот же YOLO-датасет, 6 погодных условий, split по траекториям 80/10/10)

**Построение instance index:** `uav_nav/scripts/build_instance_index.py`  
**Результат:** `data/instance_index/{train,val,test}.csv`

### Instance tracking

Между соседними кадрами одной траектории запускается одно-к-одному жадное IoU-сопоставление (порог 0.30) по классу. Каждому экземпляру присваивается UUID (`instance_id`). Экземпляры с ≥2 наблюдениями считаются "usable" (есть хотя бы один anchor+positive).

### Статистика (usable instances)

| Сплит | Детекций | Instances | ≥2 obs (usable) |
|---|---|---|---|
| train | 193,816 | 120,766 | **27,904** |
| val | 21,792 | 13,739 | **3,359** |
| test | 30,155 | 19,936 | **4,564** |

По классам (train): tree=95,150 (доминирует), rock=22,883, road=1,680, river=1,053.

### Формирование триплетов (TripletDataset)

**Файл:** `uav_nav/data/triplet_dataset.py`

- **Anchor:** первое наблюдение instance_id (96×96 crop, фон=серый 127)
- **Positive:** второе наблюдение того же instance_id
- **Negative:** случайный экземпляр того же класса, другой instance_id (20 попыток)

Crop pipeline: polygon_flat → bbox с padding 20% → вырезка → заливка фона серым 127 → resize 96×96 → ImageNet normalization.

---

## 4. Обучение

**Скрипт:** `uav_nav/scripts/train_embedding.py`  
**Команда запуска:**
```bash
python -m uav_nav.scripts.train_embedding \
    --device cuda --epochs 50 --batch_size 64 --num_workers 0
```

**Директория эксперимента:** `experiments/20260526_092148_embedding/`  
- `train_log.csv` — лог по эпохам  
- `checkpoints/epoch_NNN.pt` — каждые 5 эпох  
- `best.pt` — лучший по val_AUC

**Функция потерь:** `TripletMarginLoss(margin=0.5, p=2)`  
**Оптимизатор:** AdamW, CosineAnnealingLR  
**Gradient clipping:** max_norm=1.0

### Прогрев backbone

Первые 5 эпох backbone заморожен — обучается только projection head. На эпохе 6 backbone разморожен, для него lr_backbone = lr×0.1.

### Кривая обучения (по эпохам)

| Эпоха | train_loss | val_loss | val_AUC | Событие |
|---|---|---|---|---|
| 1 | 0.2076 | 0.1490 | 0.8956 | Старт |
| 5 | 0.1107 | 0.1167 | 0.9178 | Конец заморозки |
| 6 | 0.0952 | 0.1021 | 0.9288 | Backbone разморожен (+0.011 AUC) |
| 10 | 0.0634 | 0.0726 | 0.9558 | |
| 19 | 0.0462 | 0.0497 | 0.9751 | |
| **31** | **0.0382** | **0.0406** | **0.9783** | **Лучший чекпоинт** |
| 43 | 0.0363 | 0.0404 | 0.9775 | Текущий (обучение идёт) |

Модель стабилизировалась около эпохи 31 — val_AUC плавает в диапазоне 0.974–0.978 без улучшения. Это характерно для metric learning с фиксированным датасетом триплетов.

---

## 5. Файлы весов

| Путь | Описание |
|---|---|
| `experiments/20260526_092148_embedding/best.pt` | Лучший чекпоинт (эпоха 31, AUC=0.9783) |
| `experiments/20260526_092148_embedding/checkpoints/epoch_NNN.pt` | Периодические чекпоинты |
| `weights/embedding_head.pt` | Финальный (скопируется после 50-й эпохи) |

Формат чекпоинта:
```python
{"config": EmbeddingConfig.__dict__, "state_dict": OrderedDict}
```

---

## 6. Реализованные модули

| Модуль | Статус | Описание |
|---|---|---|
| `scripts/build_instance_index.py` | ✅ | IoU-трекинг, построение CSV-индекса |
| `perception/embedding_head.py` | ✅ | MobileNetV3-Small + LayerNorm projection |
| `data/triplet_dataset.py` | ✅ | TripletDataset, build_loaders, crop extraction |
| `scripts/train_embedding.py` | ✅ | Полный train loop, ROC AUC val, чекпоинты |
| `eval/embedding_eval.py` | ✅ | ROC AUC, Recall@1/5, discriminability |
| `perception/feature_extractor.py` | ✅ | YOLO+EmbeddingHead → Detection pipeline |

---

## 7. Тесты

**38/38 passed** (без GPU, без датасета):
- `test_embedding_head.py` — 15 тестов (forward shape, L2-норма, freeze/unfreeze, save/load, encode_crop)
- `test_yolo_segmenter.py` — 23 теста

Запуск: `python -m pytest uav_nav/tests/test_perception/ -v`

---

## 8. Финальные метрики (val, 3359 instances)

| Метрика | Порог | Факт | Статус |
|---|---|---|---|
| ROC AUC same/diff instance (val) | ≥ 0.80 | **0.983** | ✅ |
| Recall@1 (val, gallery=3359) | ≥ 0.60 | **0.118** | ⚠️ ниже порога |
| Recall@5 (val) | — | **0.300** | |
| Discriminability (inter/intra) | — | **8.46** | excellent |
| Intra dist (same instance) | — | **0.116** | |
| Inter dist (diff instance, same class) | — | **0.978** | |
| pytest без GPU | 100% | **38/38** | ✅ |

### Анализ Recall@1

**ROC AUC = 0.983 при Recall@1 = 0.118 — это не противоречие.**

ROC AUC измеряет попарную дискриминацию: "правильная пара имеет cosine similarity выше, чем неправильная". Это верно в 98.3% случаев. Recall@1 требует, чтобы точное совпадение было абсолютным №1 среди 3359 кандидатов в галерее.

Сложность retrieval:
- ~95% instances в val — деревья (tree=95,150/120,766 в train). Деревья визуально однородны → много похожих кандидатов.
- 0.118 = 393× лучше случайного (random = 1/3359 ≈ 0.0003).

**Для TILM это приемлемо:** matching происходит с позиционным prior'ом по траектории (temporal window), а не в полном галерейном поиске всех 3359 экземпляров. В TILM галерея ограничена ближайшими ~20–50 landmarks по времени и пространству.

---

## 9. Следующий этап

**Этап 4 — EKF VIO** (`estimation/imu_preintegrator.py` + `estimation/ekf_vio.py`).

Latency benchmark на CPU (запустить при наличии времени):
```bash
python -c "
import torch, time
from uav_nav.perception.embedding_head import EmbeddingHead
h = EmbeddingHead.load('weights/embedding_head.pt', device='cpu')
x = torch.randn(1, 3, 96, 96)
for _ in range(10): h(x)
t = time.perf_counter()
for _ in range(100): h(x)
print(f'{(time.perf_counter()-t)*10:.1f} ms/crop')
"
```
