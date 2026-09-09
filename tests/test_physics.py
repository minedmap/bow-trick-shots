import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import physics as ph


def test_double_hawkeye_spec_example():
    # 스펙 예시값: v0=30, theta1=70, g=9.8 -> R~=59, T1~=5.75, T2~=2.09, delay~=3.66
    res = ph.double_hawkeye(v0=30.0, theta1_deg=70.0, g=9.8)
    assert res.theta2 == pytest.approx(20.0)
    assert res.R == pytest.approx(59.0, abs=0.5)
    assert res.T1 == pytest.approx(5.75, abs=0.01)
    assert res.T2 == pytest.approx(2.09, abs=0.01)
    assert res.delay == pytest.approx(3.66, abs=0.01)


def test_complementary_angles_same_range():
    for theta1 in (50, 60, 75, 89):
        r1 = ph.projectile_range(25.0, theta1)
        r2 = ph.projectile_range(25.0, 90 - theta1)
        assert r1 == pytest.approx(r2)


def test_double_hawkeye_invalid_theta():
    with pytest.raises(ValueError):
        ph.double_hawkeye(v0=30.0, theta1_deg=45.0)
    with pytest.raises(ValueError):
        ph.double_hawkeye(v0=30.0, theta1_deg=90.0)


def test_projectile_position_apex_and_landing():
    v0, theta = 30.0, 70.0
    T = ph.projectile_time_of_flight(v0, theta)
    x_land, y_land = ph.projectile_position(v0, theta, T)
    assert y_land == pytest.approx(0.0, abs=1e-6)
    x_apex, y_apex = ph.projectile_position(v0, theta, T / 2)
    assert y_apex > 0


def test_lateral_position_matches_formula():
    A, omega, phi = 2.0, 1.5, 0.3
    t = 1.234
    expected = A * math.sin(omega * t + phi)
    assert ph.lateral_position(A, omega, t, phi) == pytest.approx(expected)


def test_solve_s_curve_avoidance_clears_obstacles():
    vx = 20.0
    obstacles = [
        {"x": 15.0, "y": 1.0, "radius": 0.5},
        {"x": 30.0, "y": -1.0, "radius": 0.5},
    ]
    A, omega = ph.solve_s_curve_avoidance(vx, obstacles, margin=0.3)
    assert A > 0
    assert omega > 0
    for obs in obstacles:
        assert not ph.check_collision(A, omega, vx, obs)


def test_check_collision_true_when_arrow_hits_center():
    obstacle = {"x": 10.0, "y": 0.0, "radius": 1.0}
    # A=0이면 궤적이 y=0 직선이라 장애물 중심(0,10)을 그대로 통과
    assert ph.check_collision(A=0.0, omega=1.0, vx=5.0, obstacle=obstacle)


def test_suggest_omega_sum_condition_gives_matching_lateral_position():
    T1, T2 = 5.75, 2.09
    omega = ph.suggest_omega(T1, T2, n=0, mode="sum")
    err = ph.lateral_error(T1, T2, A1=1.0, omega1=omega, phi1=0.0,
                            A2=1.0, omega2=omega, phi2=0.0)
    assert err == pytest.approx(0.0, abs=1e-9)


def test_suggest_omega_diff_condition_gives_matching_lateral_position():
    T1, T2 = 5.75, 2.09
    omega = ph.suggest_omega(T1, T2, n=1, mode="diff")
    err = ph.lateral_error(T1, T2, A1=1.0, omega1=omega, phi1=0.0,
                            A2=1.0, omega2=omega, phi2=0.0)
    assert err == pytest.approx(0.0, abs=1e-9)


def test_suggest_omega_diff_zero_raises():
    with pytest.raises(ValueError):
        ph.suggest_omega(5.0, 2.0, n=0, mode="diff")


def test_nearest_suggested_omega_returns_closest_candidate():
    T1, T2 = 5.75, 2.09
    current = 0.5
    best = ph.nearest_suggested_omega(T1, T2, current)
    err = ph.lateral_error(T1, T2, 1.0, best, 0.0, 1.0, best, 0.0)
    assert err == pytest.approx(0.0, abs=1e-9)
