# Design Decisions Log

Records architectural decisions and their rationale.

## DD-001: Loguru over standard logging

**Date:** 2026-05-23
**Decision:** Use loguru instead of Python stdlib `logging`.
**Rationale:** Structured fields, colourised output, rotating files, and JSON
serialisation in one library with minimal configuration boilerplate.

## DD-002: Hydra for configuration

**Date:** 2026-05-23
**Decision:** Use Hydra + OmegaConf for config management.
**Rationale:** Supports config composition (pi5.yaml inherits default.yaml),
command-line overrides, and experiment sweep support.

## DD-003: NCNN for Pi 5 deployment

**Date:** 2026-05-23
**Decision:** Target NCNN as the inference backend on Raspberry Pi 5.
**Rationale:** NCNN is optimised for ARM Cortex-A series and has no heavy
runtime dependencies, unlike TensorRT (NVIDIA only) or OpenVINO (Intel only).

## DD-004: NetworkX for TILM topology

**Date:** 2026-05-23
**Decision:** Use NetworkX DiGraph as the TILM graph backend.
**Rationale:** Rich graph algorithms (shortest path, connected components),
easy serialisation, and pure-Python so it runs on Pi 5 without native libs.

## DD-005: evo for trajectory evaluation

**Date:** 2026-05-23
**Decision:** Use the `evo` package for ATE/RPE computation.
**Rationale:** Standard tool in the SLAM community, handles pose alignment,
scale ambiguity, and produces publication-quality plots.

## DD-006: frame_stride для контроля размера YOLO-датасета

**Date:** 2026-05-24
**Decision:** Добавить параметр `frame_stride` в `MidAirLoader` и конфиг `default.yaml` (по умолчанию `10`).
**Проблема:** Без ограничения датасет из MidAir вырастает до ~20 ГБ — каждая траектория содержит 3000+ кадров, все они выгружались в YOLO-формат.
**Решение:** `frame_stride=N` означает `range(0, n_frames, N)` — каждый N-й кадр.
  При `stride=10` датасет уменьшается примерно в 10 раз (~2 ГБ), при `stride=15` — в 15 раз.
  `frame_stride` имеет приоритет над `max_frames_per_traj`; если задан `max_frames`, он применяется как cap уже после stride-отбора.
**Затронутые файлы:**
  - `uav_nav/data/midair_loader.py` — новый параметр `frame_stride` в `__init__`, логика в `_index_trajectory`
  - `uav_nav/scripts/build_dataset.py` — читает `dataset.frame_stride` из Hydra-конфига
  - `uav_nav/configs/default.yaml` — добавлено `frame_stride: 10`
**Команда запуска с переопределением:**
  ```
  uav-nav-build-dataset dataset.midair_root="D:/data/MidAir" dataset.yolo_root="data/yolo_v2" +dataset.yolo_only=true dataset.frame_stride=15
  ```
