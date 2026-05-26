# Отчёт: Этап 5 — TILM (Topological Invariant Landmark Map)

**Дата завершения:** 2026-05-26  
**Статус:** ✅ Завершён  
**Тесты:** 43/43 passed

---

## Цель этапа

Построить топологическую карту семантических ориентиров (TILM), накапливаемую при полётах с GNSS, и реализовать механизм поиска похожего места для коррекции позиции при потере GNSS.

---

## Реализованные модули

### `memory/tilm.py` — `TILM`

Топологический граф на NetworkX DiGraph:

```
TILMNode:
  node_id        — уникальный идентификатор
  position_ned   (3,) — NED-позиция при создании узла
  landmarks      list[Landmark] — семантические ориентиры
  place_descriptor (D,) — L2-норм. средний дескриптор для place recognition
  timestamp, keyframe_id, visit_count

TILMEdge:
  src_id, dst_id — направленное ребро
  relative_pose  (4×4) — SE(3) трансформ
  distance       — метрическое расстояние
  weight         — вес для A* (1 − sim для loop closure рёбер)

TILM:
  add_node(node) → node_id
  add_edge(edge)
  get_node(node_id) → TILMNode
  nearest_node(position_ned, k) → [(id, dist), ...]
  shortest_path(src_id, dst_id) → [id, ...]
  n_nodes() → int
  save(path) / load(path) — pickle сериализация
```

### `memory/tilm_builder.py` — `TILMBuilder`

Инкрементальная сборка карты:

```
BuilderConfig:
  min_node_distance        = 5.0 м
  min_landmarks_per_node   = 2
  edge_max_distance        = 20.0 м
  use_loop_closure         = True
  loop_closure_threshold   = 0.75

TILMBuilder.process_frame(pose_ned, landmarks, timestamp):
  1. Проверить _should_create_node():
     - len(landmarks) >= min_landmarks_per_node
     - dist(pos, prev_pos) >= min_node_distance
  2. Создать TILMNode с place_descriptor = L2-норм. среднее эмбеддингов
  3. Добавить sequential edge от предыдущего узла
  4. Попытаться loop closure:
     - Для каждого узла в радиусе [1, edge_max_distance]
     - cos-similarity(pd_new, pd_existing) >= loop_closure_threshold
     - Добавить edge с weight = 1 − sim
```

**Place descriptor** = L2-нормализованное среднее по всем landmark дескрипторам узла — инвариантен к порядку и числу ориентиров.

### `memory/temporal_matcher.py` — `TemporalMatcher`

Поиск соответствия наблюдений с картой:

```
TemporalMatcher(tilm, descriptor_threshold=0.5, min_inliers=2, top_k_nodes=5)

build_descriptor_index():
  → _entries: list[(node_id, lm_idx, class_id)]
  → _desc_matrix: (N_entries, D) float32

match(observations, position_prior) → MatchResult:
  1. Voting: каждое наблюдение → best matching entry (same class)
     → vote_scores[node_id] += cos_sim
  2. Ranking: top_k_nodes по score, взвешен proximity к prior
  3. Geometric verify каждого кандидата:
     - Жадный матчинг: cos_sim >= (1 − threshold), same class
     - inliers = непересекающиеся пары
  4. Выбор best: max(n_inliers / max(|obs|, |map|))
  5. correction = best_node.position_ned − position_prior

MatchResult:
  matched_node_id, score, matched_pairs
  position_correction_ned, confidence, n_inliers, is_valid
```

### `perception/landmark_extractor.py` — `LandmarkExtractor`

Извлечение 3D-ориентиров из одного кадра:

```
LandmarkExtractor(embedding_head, min_mask_area=200,
                  max_landmarks_per_frame=20,
                  depth_percentile=50.0,
                  valid_classes=None,
                  camera_K=None)

extract(image_rgb, segmentation, depth, frame_id) → list[Landmark]:
  Для каждого instance (sorted by confidence desc):
    1. Фильтр по классу и площади маски
    2. centroid(u,v) = mean(mask pixels)
    3. depth_val = nanpercentile(depth[mask], percentile)
    4. position_3d = _unproject(centroid, depth_val, K)
    5. descriptor = _crop_and_embed(image_rgb, mask)
    → Landmark(position_3d, class_id, descriptor, confidence, ...)

_unproject(centroid_uv, depth, K):
  X = (u − cx) × depth / fx
  Y = (v − cy) × depth / fy
  Z = depth

_crop_and_embed(image_rgb, mask):
  bbox(mask) + 20% padding → crop
  фон = 127 (neutral gray)
  resize 96×96 → EmbeddingHead.forward()
```

---

## Архитектурные решения

| Решение | Выбор | Обоснование |
|---|---|---|
| Топология | NetworkX DiGraph | Pure-Python, алгоритмы графов включены, легко сериализовать |
| Place descriptor | L2-норм. среднее эмбеддингов | Инвариантен к порядку, не требует отдельной модели |
| Loop closure | Косинусная схожесть place descriptor ≥ threshold | Быстро, без RANSAC; threshold настраивается |
| Descriptor voting | Each observation votes for best matching node | O(M×N) brute force, достаточно для edge device |
| Depth aggregation | nanpercentile (медиана по умолчанию) | Устойчив к шуму и невалидным пикселям |
| Landmark crop | Bbox + 20% padding, фон gray 127 | Консистентно с EmbeddingHead.encode_crop() |

---

## Результаты тестирования

| Класс тестов | Тестов | Результат |
|---|---|---|
| `TestTILMBasic` | 7 | ✅ |
| `TestTILMEdge` | 2 | ✅ |
| `TestTILMNearestNode` | 5 | ✅ |
| `TestTILMShortestPath` | 2 | ✅ |
| `TestTILMSaveLoad` | 1 | ✅ |
| `TestTILMNode` | 2 | ✅ |
| `TestTILMBuilderNodeCreation` | 7 | ✅ |
| `TestTILMBuilderLoopClosure` | 2 | ✅ |
| `TestTemporalMatcherIndex` | 3 | ✅ |
| `TestTemporalMatcherRetrieve` | 3 | ✅ |
| `TestTemporalMatcherResult` | 6 | ✅ |
| `TestTemporalMatcherMinInliers` | 1 | ✅ |
| `TestMatchResult` | 2 | ✅ |
| **Итого** | **43** | **43/43** |

---

## Следующий этап

**Этап 6 — Matching + EKF коррекции:**
- `runtime/pipeline.py` — связать `LandmarkExtractor → TemporalMatcher → EKFVIO.update_landmark()`
- `eval/matching_eval.py` — метрики качества matching на симулированных траекториях
- Тест сквозного пайплайна: frame → landmarks → match → EKF correction
