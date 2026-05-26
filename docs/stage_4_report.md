# Отчёт: Этап 4 — Базовый VIO (EKF + IMU Preintegration)

**Дата завершения:** 2026-05-26  
**Статус:** ✅ Завершён  
**Тесты:** 49/49 passed

---

## Цель этапа

Реализовать инерциальную навигацию без GNSS:
- IMU preintegration на группе SO(3) для накопления приращений позиции/скорости/ориентации между ключевыми кадрами
- Error-State EKF (18×18) с поддержкой обновлений GNSS, landmark и visual

---

## Реализованные модули

### `estimation/pose_utils.py`

Утилиты геометрии SE(3)/SO(3):

| Функция | Описание |
|---|---|
| `skew(v)` | Кососимметричная матрица 3×3 из вектора ω |
| `so3_exp(omega)` | Rodrigues: SO(3) экспонента вектора угловой скорости |
| `so3_log(R)` | Обратная Rodrigues: вектор из матрицы вращения |
| `quat_to_rot(q)` | Кватернион [w,x,y,z] → матрица вращения 3×3 |
| `rot_to_quat(R)` | Матрица вращения → кватернион (метод Шеппера, 4 случая) |
| `quat_mult(q1, q2)` | Произведение Гамильтона |
| `se3_exp(xi)` | SE(3) экспонента (twist вектор 6-d) |
| `interpolate_poses(T0, T1, alpha)` | SLERP+LERP интерполяция поз |

### `estimation/imu_preintegrator.py`

```
PreintegratedState:
  delta_R  (3×3) — накопленное вращение
  delta_v  (3,)  — накопленное изменение скорости
  delta_p  (3,)  — накопленное изменение позиции
  dt             — суммарный интервал времени
  n_samples      — число интегрированных измерений
  cov      (9×9) — ковариация [δp, δv, δθ]
  Jg_R, Ja_v, Ja_p — якобианы для поправки смещений
  bias_acc, bias_gyro — текущие смещения

IMUPreintegrator.reset(bias_acc, bias_gyro)
IMUPreintegrator.integrate(acc, gyro, dt)  — Euler на SO(3)
IMUPreintegrator.predict_pose(R0, v0, p0) → (R1, v1, p1)
PreintegratedState.correct_for_bias_update(delta_ba, delta_bg)
```

**Формулы интеграции (NED, g=[0,0,9.81]):**
```
a_corr = acc - bias_acc
ω_corr = gyro - bias_gyro
dR     = so3_exp(ω_corr × dt)

δp_new = δp + δv·dt + 0.5·(δR @ a_corr)·dt²
δv_new = δv + (δR @ a_corr)·dt
δR_new = δR @ dR
```

### `estimation/ekf_vio.py`

```
EKFState:
  position    (3,)   — NED позиция
  velocity    (3,)   — NED скорость
  quaternion  (4,)   — [w,x,y,z]
  bias_acc    (3,)   — смещение акселерометра
  bias_gyro   (3,)   — смещение гироскопа
  P           (18×18)— ковариация ошибок
  is_gnss_active     — флаг активности GNSS

EKFVIO.propagate(preintegrated)    — predict с F=18×18
EKFVIO.update_gnss(position_ned, accuracy)
EKFVIO.update_landmark(correction_ned, covariance)
EKFVIO.update_visual(...)          — NotImplementedError (этап 6)
EKFVIO.set_gnss_active(flag)
EKFVIO.get_state() → копия EKFState
```

**F-матрица перехода (18×18):**
```
F[0:3, 3:6]   = I·dt              (p ← v)
F[3:6, 6:9]   = -R0·skew(δv)      (v ← θ)
F[3:6, 9:12]  = -R0·dt            (v ← ba)
F[6:9, 6:9]   = δR.T              (θ ← θ)
F[6:9, 12:15] = -I·dt             (θ ← bg)
```

**Обновление позиции (Joseph form):**
```
H[0:3, 0:3] = I
K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹
IKH = I − K·H
P_new = IKH·P·IKHᵀ + K·R·Kᵀ
```

---

## Связь с этапом 5 (TILM)

`EKFVIO.update_landmark(correction)` принимает `position_correction_ned` из `TemporalMatcher.match()`:
```python
result = matcher.match(observations, position_prior=ekf.state.position)
if result.is_valid:
    ekf.update_landmark(result.position_correction_ned)
```

Вычисление коррекции: `correction = best_node.position_ned − position_prior`  
Эффект: `z = position + correction = best_node.position_ned` → innovation = коррекция

---

## Результаты тестирования

| Класс тестов | Тестов | Результат |
|---|---|---|
| `TestReset` | 3 | ✅ |
| `TestIntegrate` | 6 | ✅ |
| `TestPredictPose` | 3 | ✅ |
| `TestBiasCorrection` | 2 | ✅ |
| `TestEKFState` | 5 | ✅ |
| `TestPropagation` | 6 | ✅ |
| `TestGNSSUpdate` | 5 | ✅ |
| `TestLandmarkUpdate` | 3 | ✅ |
| `TestVisualUpdate` | 1 | ✅ |
| **Итого** | **34** | **34/34** |

*(+ 15 тестов pose_utils = 49 всего)*
