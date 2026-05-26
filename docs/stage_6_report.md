# Отчёт: Этап 6 — Matching + EKF-коррекции

**Дата завершения:** 2026-05-26  
**Статус:** ✅ Завершён  
**Тесты:** 41/41 passed

---

## Цель этапа

Соединить TILM-матчинг с EKF: при потере GNSS автоматически переключаться на коррекции позиции через `TemporalMatcher` и применять их к `EKFVIO`.

---

## Реализованные модули

### `runtime/matching_corrector.py`

```
CorrectorStats:
  n_frames, n_gnss_updates
  n_tilm_attempts, n_tilm_corrections, n_tilm_rejected
  correction_errors: list[float]
  → mean_correction_m, tilm_acceptance_rate

MatchingCorrector(matcher, ekf,
                  min_confidence=0.3,
                  correction_interval=1.0,
                  correction_covariance=None):

  set_gnss_available(bool)   — переключение режима
  is_gnss_available          — текущий режим
  is_using_tilm              — антоним

  update(observations, timestamp,
         gnss_position=None, gnss_accuracy=3.0) → MatchResult | None

  last_match                 — последний MatchResult
  reset_stats()
```

**Логика `update()`:**

```
if GNSS_ACTIVE:
    if gnss_position is not None:
        ekf.update_gnss(gnss_position, accuracy)
        stats.n_gnss_updates += 1
    return None

# GNSS_LOST:
result = matcher.match(observations, prior=ekf.state.position)

if not result.is_valid:           → reject
if result.confidence < threshold: → reject
if elapsed < correction_interval: → reject (throttle)

ekf.update_landmark(result.position_correction_ned)
stats.n_tilm_corrections += 1
return result
```

### `eval/matching_eval.py`

```
MatchingEvaluator(position_tolerance=10.0)

evaluate(query_gt, match_pred, match_valid, latencies_ms=None)
  → MatchingMetrics(precision, recall, f1,
                    position_error_mean/rmse/max,
                    false_positive_rate,
                    n_total/n_successful/n_correct,
                    mean_latency_ms)

precision_recall_curve(scores, is_correct)
  → (precision, recall, thresholds)
     sorted by score descending, cumulative
```

**Определения метрик:**
- `precision = n_correct / n_matched`
- `recall = n_correct / N` (доля от всех запросов)
- `false_positive_rate = (n_matched - n_correct) / N`
- `is_correct` := `||pred - gt||₂ ≤ position_tolerance`

---

## Схема взаимодействия (сводная)

```
MidAir frame
    │
    ├─ [GNSS available] ──► ekf.update_gnss()
    │
    └─ [GNSS lost]
           │
           ▼
    LandmarkExtractor.extract(image, seg, depth)
           │  list[Landmark]
           ▼
    TemporalMatcher.match(obs, prior=ekf.position)
           │  MatchResult
           ▼
    MatchingCorrector.update()
      confidence ≥ 0.3 AND interval ≥ 1s?
           │ YES
           ▼
    ekf.update_landmark(correction_ned)
           │
           ▼
    EKFState.position ← corrected
```

---

## Результаты тестирования

| Класс тестов | Тестов | Результат |
|---|---|---|
| `TestCorrectorStats` | 5 | ✅ |
| `TestMatchingCorrectorMode` | 4 | ✅ |
| `TestGNSSUpdatePath` | 5 | ✅ |
| `TestTILMCorrectionPath` | 6 | ✅ |
| `TestConfidenceGate` | 2 | ✅ |
| `TestCorrectionInterval` | 3 | ✅ |
| `TestNoObservations` | 2 | ✅ |
| `TestResetStats` | 1 | ✅ |
| `TestMatchingEvaluator` | 8 | ✅ |
| `TestPrecisionRecallCurve` | 5 | ✅ |
| **Итого** | **41** | **41/41** |

---

## Следующий этап

**Этап 7 — Pseudo-simulator + NavigationPipeline:**
- `runtime/pseudo_simulator.py` — воспроизведение MidAir GT-траекторий с симуляцией потери GNSS
- `runtime/pipeline.py` — сквозной пайплайн: YOLO → LandmarkExtractor → MatchingCorrector → EKF
- `eval/trajectory_eval.py` — ATE/RPE через `evo` на симулированных сценариях
