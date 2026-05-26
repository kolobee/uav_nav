# Stage 0 Plan: Project Structure and Scaffold

**Date:** 2026-05-23
**Author:** UAV Nav Research
**Status:** In Progress

---

## 1. Overview

Stage 0 establishes the entire project scaffold for the magistrate dissertation
on UAV navigation using Visual-Inertial Odometry (VIO) with semantic
Topological Invariant Landmark Matching (TILM).

No functional algorithm code is written at this stage.
All module files contain correctly-typed class/function stubs with
`raise NotImplementedError(...)` bodies and Google-style docstrings.

---

## 2. Task Decomposition

| ID   | Task                               | Output                              | Status |
|------|------------------------------------|-------------------------------------|--------|
| S0-1 | Create directory tree              | All dirs under `uav_nav/`           | Done   |
| S0-2 | Package stubs (`data/`)            | calibration, loader, builder        | Done   |
| S0-3 | Package stubs (`perception/`)      | yolo, embedding, features, depth, landmark | Done |
| S0-4 | Package stubs (`memory/`)          | TILM, builder, matcher, descriptor  | Done   |
| S0-5 | Package stubs (`estimation/`)      | IMU preint, EKF, pose utils         | Done   |
| S0-6 | Package stubs (`planning/`)        | path follower, corridor, modes, rejoin | Done |
| S0-7 | Package stubs (`runtime/`)         | pipeline, simulator, AirSim, monitor, logger | Done |
| S0-8 | Package stubs (`deployment/`)      | ONNX export, NCNN export, benchmark, Pi5 runtime | Done |
| S0-9 | Package stubs (`eval/`)            | trajectory, embedding, matching, scenario, ablation | Done |
| S0-10 | Hydra configs                     | default.yaml, pi5.yaml, label_map.json, experiment YAMLs | Done |
| S0-11 | Scripts                            | 01–08_*.py with Hydra entry points  | Done   |
| S0-12 | pyproject.toml + requirements.txt  | Build system, deps, pytest config   | Done   |
| S0-13 | pytest setup + conftest.py         | Fixtures for all major data types   | Done   |
| S0-14 | .pre-commit-config.yaml            | black + ruff hooks                  | Done   |
| S0-15 | Makefile                           | 8+ targets including pipeline stages| Done   |
| S0-16 | docs/stage_0_plan.md               | This document                       | Done   |
| S0-17 | docs/stage_0_report.md             | Completion report                   | Done   |

---

## 3. Module Interface Specifications

### 3.1 `uav_nav.data`

| Class / Function         | Signature                                         | Returns               |
|--------------------------|---------------------------------------------------|-----------------------|
| `CameraCalibration`      | `(fx, fy, cx, cy, width, height, ...)`            | dataclass             |
| `CameraCalibration.K`    | property                                          | `np.ndarray (3,3)`    |
| `CameraCalibration.from_midair_defaults` | classmethod                      | `CameraCalibration`   |
| `StereoCalibration`      | `(left, right, R, T)`                             | dataclass             |
| `StereoCalibration.baseline` | property                                      | `float`               |
| `MidAirLoader`           | `(root, sequences, load_depth, load_segmentation, ...)` | iterable       |
| `MidAirLoader.build_index` | `() -> None`                                    | —                     |
| `MidAirLoader.__getitem__` | `(idx: int) -> MidAirSample`                   | `MidAirSample`        |
| `DatasetBuilder.build_all` | `(train_seq, val_seq, test_seq) -> dict[str, Path]` | `dict`            |

### 3.2 `uav_nav.perception`

| Class / Function           | Signature                                      | Returns               |
|----------------------------|------------------------------------------------|-----------------------|
| `YOLOSegmenter.predict`    | `(image: ndarray) -> SegmentationResult`       | `SegmentationResult`  |
| `EmbeddingHead.forward`    | `(crops: ndarray) -> ndarray`                  | `(B, D)` float32      |
| `FeatureExtractor.extract` | `(image: ndarray) -> KeypointSet`              | `KeypointSet`         |
| `StereoDepthEstimator.estimate` | `(left, right) -> DepthResult`            | `DepthResult`         |
| `LandmarkExtractor.extract` | `(image, seg, depth, frame_id) -> list[Landmark]` | `list[Landmark]`  |

### 3.3 `uav_nav.memory`

| Class / Function           | Signature                                      | Returns               |
|----------------------------|------------------------------------------------|-----------------------|
| `TILM.add_node`            | `(node: TILMNode) -> int`                      | node ID               |
| `TILM.nearest_node`        | `(pos: ndarray, k: int) -> list[tuple]`        | `[(id, dist)]`        |
| `TILM.save`                | `(path: Path) -> None`                         | —                     |
| `TILM.load`                | `(path: Path) -> TILM`                         | `TILM`                |
| `TILMBuilder.process_frame` | `(pose_ned, landmarks, timestamp) -> int?`    | optional node ID      |
| `TemporalMatcher.match`    | `(observations, position_prior) -> MatchResult` | `MatchResult`        |

### 3.4 `uav_nav.estimation`

| Class / Function           | Signature                                      | Returns               |
|----------------------------|------------------------------------------------|-----------------------|
| `IMUPreintegrator.integrate` | `(acc, gyro, dt) -> None`                   | —                     |
| `IMUPreintegrator.predict_pose` | `(R0, v0, p0) -> (R1, v1, p1)`          | 3-tuple               |
| `EKFVIO.propagate`         | `(preintegrated: PreintegratedState) -> None`  | —                     |
| `EKFVIO.update_landmark`   | `(correction_ned: ndarray) -> None`            | —                     |
| `EKFVIO.update_gnss`       | `(position_ned: ndarray, accuracy) -> None`    | —                     |

### 3.5 `uav_nav.planning`

| Class / Function           | Signature                                      | Returns               |
|----------------------------|------------------------------------------------|-----------------------|
| `PathFollower.step`        | `(pos, heading, quality, dt) -> ndarray(3,)`   | velocity NED          |
| `PathFollower._compute_look_ahead` | `(quality: float) -> float`          | distance (m)          |
| `AdaptiveCorridor.update`  | `(pos, path, quality, n_lm, dt) -> CorridorState` | `CorridorState`   |
| `MissionManager.update`    | `(status: MissionStatus) -> MissionMode`       | `MissionMode`         |
| `RejoinPlanner.plan`       | `(pos, vel, path, idx) -> RejoinResult`        | `RejoinResult`        |

### 3.6 `uav_nav.runtime`

| Class / Function           | Signature                                      | Returns               |
|----------------------------|------------------------------------------------|-----------------------|
| `NavigationPipeline.initialise` | `() -> None`                             | —                     |
| `NavigationPipeline.process_frame` | `(image, image_right, imu_acc, imu_gyro, dt, gnss, ts) -> dict` | dict |
| `setup_logger`             | `(log_dir, level, ...) -> None`                | —                     |

---

## 4. Key Parameters (Dissertation Values)

| Parameter     | Value   | Location                         |
|---------------|---------|----------------------------------|
| D_min         | 5.0 m   | `PathFollowerConfig.D_min`       |
| D_max_cap     | 50.0 m  | `PathFollowerConfig.D_max_cap`   |
| dt_window     | 5.0 s   | `PathFollowerConfig.dt_window`   |
| THRESHOLD     | 0.7     | `PathFollowerConfig.THRESHOLD`   |
| Embedding dim | 128     | `EmbeddingConfig.embedding_dim`  |
| TILM node dist| 5.0 m   | `TILM.min_node_distance`         |
| LC threshold  | 0.75    | `BuilderConfig.loop_closure_threshold` |

---

## 5. Acceptance Criteria (Stage 0)

- [ ] `python -m pytest uav_nav/tests/ -m "not requires_midair"` passes (all tests that don't need the dataset run).
- [ ] `black --check uav_nav/` exits with 0.
- [ ] `ruff check uav_nav/ --select E,F,I` exits with 0.
- [ ] `python -c "import uav_nav; print(uav_nav.__version__)"` prints `0.1.0`.
- [ ] All stub methods raise `NotImplementedError` with a descriptive message.
- [ ] Hydra can load `default.yaml` and `pi5.yaml` without errors.
- [ ] `label_map.json` contains mappings for `tree`, `rock`, `river`.
- [ ] `Makefile` targets `train_yolo`, `train_embedding`, `build_tilm`, `run_experiment`, `benchmark_pi5` exist and invoke the correct scripts.

---

## 6. Subsequent Stage Dependencies

Stage 0 output is a prerequisite for all subsequent stages:

- **Stage 1**: Implement `data/` module — `MidAirLoader`, `DatasetBuilder`.
- **Stage 2**: Implement `perception/` — YOLO segmenter, stereo depth, landmark extractor.
- **Stage 3**: Implement `estimation/` — IMU preintegration, EKF.
- **Stage 4**: Implement `memory/` — TILM construction and matching.
- **Stage 5**: Implement `planning/` — path follower, corridor, mission modes.
- **Stage 6**: Implement `eval/` — trajectory metrics, ablation studies.
- **Stage 7**: Implement `deployment/` — ONNX/NCNN export, Pi 5 benchmark.
