"""벡터 물리 기반 활 트릭샷 계산 함수 모음.

단위는 SI(m, s, m/s, rad)를 사용한다. 각도(theta, phi)는 별도 명시가 없으면
degree 단위 인자를 받고, 내부적으로 radian으로 변환해 계산한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve

G_DEFAULT = 9.8


# ---------------------------------------------------------------------------
# 1. 더블 하위처 샷 (수직 평면 포물선 운동)
# ---------------------------------------------------------------------------

def projectile_range(v0: float, theta_deg: float, g: float = G_DEFAULT) -> float:
    """사거리 R = v0^2 sin(2theta) / g."""
    theta = math.radians(theta_deg)
    return v0 ** 2 * math.sin(2 * theta) / g


def projectile_time_of_flight(v0: float, theta_deg: float, g: float = G_DEFAULT) -> float:
    """체공 시간 T = 2 v0 sin(theta) / g."""
    theta = math.radians(theta_deg)
    return 2 * v0 * math.sin(theta) / g


def projectile_position(v0: float, theta_deg: float, t, g: float = G_DEFAULT):
    """시각 t(스칼라 또는 배열)에서의 (x, y) 위치. 발사 전(t<0)은 발사점에 고정."""
    theta = math.radians(theta_deg)
    t = np.asarray(t, dtype=float)
    t_clamped = np.clip(t, 0.0, None)
    x = v0 * math.cos(theta) * t_clamped
    y = v0 * math.sin(theta) * t_clamped - 0.5 * g * t_clamped ** 2
    y = np.clip(y, 0.0, None)  # 착지 후 지면 아래로 내려가지 않도록
    return x, y


@dataclass
class DoubleHawkeyeResult:
    v0: float
    g: float
    theta1: float  # 고각
    theta2: float  # 저각 = 90 - theta1
    R: float
    T1: float
    T2: float
    delay: float  # T1 - T2, 저각 화살의 발사 지연 시간


def double_hawkeye(v0: float, theta1_deg: float, g: float = G_DEFAULT) -> DoubleHawkeyeResult:
    """같은 v0, 상보각(theta1, 90-theta1)으로 동시 착탄시키는 데 필요한 값 계산."""
    if not 45.0 < theta1_deg < 90.0:
        raise ValueError("theta1은 (45, 90) 구간이어야 고각/저각이 구분됩니다.")
    theta2_deg = 90.0 - theta1_deg
    R1 = projectile_range(v0, theta1_deg, g)
    T1 = projectile_time_of_flight(v0, theta1_deg, g)
    T2 = projectile_time_of_flight(v0, theta2_deg, g)
    return DoubleHawkeyeResult(
        v0=v0, g=g, theta1=theta1_deg, theta2=theta2_deg,
        R=R1, T1=T1, T2=T2, delay=T1 - T2,
    )


# ---------------------------------------------------------------------------
# 2. S자 우회 샷 (수평 평면, 매그누스 힘에 의한 횡방향 진동)
# ---------------------------------------------------------------------------

def lateral_position(A: float, omega: float, t, phi: float = 0.0):
    """y(t) = A sin(omega t + phi)."""
    t = np.asarray(t, dtype=float)
    return A * np.sin(omega * t + phi)


def lateral_velocity(A: float, omega: float, t, phi: float = 0.0):
    """dy/dt = A omega cos(omega t + phi)."""
    t = np.asarray(t, dtype=float)
    return A * omega * np.cos(omega * t + phi)


def s_curve_position(vx: float, A: float, omega: float, t, phi: float = 0.0):
    """전진/횡방향을 합친 (x(t), y(t))."""
    t = np.asarray(t, dtype=float)
    x = vx * t
    y = lateral_position(A, omega, t, phi)
    return x, y


def _obstacle_pass_side(y_obs: float) -> int:
    """장애물의 반대쪽(비어 있는 쪽)으로 피해가도록 회피 방향 부호 결정."""
    return -1 if y_obs >= 0 else 1


def obstacle_clearance_target(obstacle: dict, margin: float = 0.15) -> float:
    """장애물을 안전하게 피하는 목표 y값 (장애물 중심 반대편, radius+margin만큼 이격)."""
    side = _obstacle_pass_side(obstacle["y"])
    return obstacle["y"] + side * (obstacle.get("radius", 0.0) + margin)


def solve_s_curve_avoidance(vx: float, obstacles: list[dict], margin: float = 0.15,
                             initial_guess: tuple[float, float] = (1.5, 1.5),
                             phi: float = 0.0):
    """장애물 2개를 피하도록 A, omega를 연립해서 구한다.

    obstacles: [{"x":, "y":, "radius":}, {"x":, "y":, "radius":}]
    각 장애물을 지나는 시각 t_obs = x_obs/vx 에서, y(t_obs)가 장애물 반대편의
    안전 지점(obstacle_clearance_target)에 오도록 하는 연립방정식을 푼다.
    """
    if len(obstacles) != 2:
        raise ValueError("장애물은 정확히 2개여야 합니다.")

    t_obs = [obs["x"] / vx for obs in obstacles]
    y_target = [obstacle_clearance_target(obs, margin) for obs in obstacles]

    def equations(vars_):
        A, omega = vars_
        eq1 = A * math.sin(omega * t_obs[0] + phi) - y_target[0]
        eq2 = A * math.sin(omega * t_obs[1] + phi) - y_target[1]
        return [eq1, eq2]

    A_sol, omega_sol = fsolve(equations, initial_guess)
    return abs(A_sol), abs(omega_sol)


def check_collision(A: float, omega: float, vx: float, obstacle: dict,
                     phi: float = 0.0, t_end: float | None = None,
                     samples: int = 400) -> bool:
    """S자 궤적이 장애물 원(circle) 내부를 지나는지 샘플링으로 검사."""
    if t_end is None:
        t_end = obstacle["x"] / vx * 2.0 if vx > 0 else 1.0
    t = np.linspace(0.0, max(t_end, 1e-6), samples)
    x, y = s_curve_position(vx, A, omega, t, phi)
    dist = np.hypot(x - obstacle["x"], y - obstacle["y"])
    return bool(np.any(dist <= obstacle.get("radius", 0.0)))


# ---------------------------------------------------------------------------
# 3. 결합 모델 (3차원: 다운레인지 + 높이 + 좌우)
# ---------------------------------------------------------------------------

def arrow3d_position(v0: float, theta_deg: float, A: float, omega: float, t,
                      g: float = G_DEFAULT, phi: float = 0.0):
    """3차원 위치 (x, y, z) = (다운레인지, 좌우, 높이)."""
    x, z = projectile_position(v0, theta_deg, t, g)
    y = lateral_position(A, omega, t, phi)
    return x, y, z


def lateral_error(T1: float, T2: float, A1: float, omega1: float, phi1: float,
                   A2: float, omega2: float, phi2: float) -> float:
    """착탄 순간 두 화살의 좌우 위치 차이 |y1(T1) - y2(T2)|."""
    y1 = A1 * math.sin(omega1 * T1 + phi1)
    y2 = A2 * math.sin(omega2 * T2 + phi2)
    return abs(y1 - y2)


def suggest_omega(T1: float, T2: float, n: int = 0, mode: str = "sum") -> float:
    """탄착 시 좌우 위치가 일치하도록 하는 omega 후보값.

    mode="sum": omega * (T1+T2) = (2n+1)*pi  (n = 0, 1, 2, ...)
    mode="diff": omega * (T1-T2) = 2*n*pi    (n = 1, 2, ...  ; n=0은 omega=0 자명해)
    """
    if mode == "sum":
        return (2 * n + 1) * math.pi / (T1 + T2)
    elif mode == "diff":
        if n == 0:
            raise ValueError("mode='diff'에서 n=0은 omega=0인 자명해입니다. n>=1을 사용하세요.")
        diff = T1 - T2
        if abs(diff) < 1e-9:
            raise ValueError("T1과 T2가 같으면 diff 조건은 정의되지 않습니다.")
        return 2 * n * math.pi / diff
    raise ValueError("mode는 'sum' 또는 'diff' 여야 합니다.")


def nearest_suggested_omega(T1: float, T2: float, current_omega: float,
                             n_range: range = range(0, 12)) -> float:
    """현재 omega 슬라이더 값에 가장 가까운 '동일 탄착점' 조건 omega를 찾는다."""
    candidates = [suggest_omega(T1, T2, n, "sum") for n in n_range]
    diff_range = [n for n in n_range if n >= 1]
    candidates += [suggest_omega(T1, T2, n, "diff") for n in diff_range]
    return min(candidates, key=lambda w: abs(w - current_omega))
