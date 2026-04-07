import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np


def test_pendulum_small_angle():
    """작은 각도에서 T = 2π√(L/g) 검증"""
    from week4_app import calculate_pendulum_period
    g = 9.81
    L = 1.0
    T = calculate_pendulum_period(L, 5.0)
    T_expected = 2 * np.pi * np.sqrt(L / g)
    assert abs(T - T_expected) / T_expected < 0.01


def test_pendulum_large_angle_longer():
    """큰 각도에서 주기가 더 길어야 함"""
    from week4_app import calculate_pendulum_period
    T_small = calculate_pendulum_period(1.0, 10.0)
    T_large = calculate_pendulum_period(1.0, 75.0)
    assert T_large > T_small


def test_projectile_data_shape():
    """포물선 데이터 생성 shape 검증"""
    from week4_app import generate_projectile_data
    X, Y = generate_projectile_data(n_samples=200)
    assert X.shape[1] == 3
    assert Y.shape[1] == 2


def test_1d_function_keys():
    """1D 함수 dict에 3개 키 존재 확인"""
    from week4_app import FUNCTIONS_1D
    assert 'sin(x)' in FUNCTIONS_1D
    assert len(FUNCTIONS_1D) == 3
