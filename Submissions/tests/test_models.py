import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np


def test_build_1d_model_output_shape():
    from week4_app import build_1d_model
    model = build_1d_model([64, 64], activation='tanh', lr=0.001)
    out = model.predict(np.array([[0.5]]), verbose=0)
    assert out.shape == (1, 1)


def test_build_projectile_model_output_shape():
    from week4_app import build_projectile_model
    model = build_projectile_model()
    out = model.predict(np.zeros((1, 3)), verbose=0)
    assert out.shape == (1, 2)


def test_build_pendulum_model_output_shape():
    from week4_app import build_pendulum_model
    model = build_pendulum_model()
    out = model.predict(np.zeros((1, 2)), verbose=0)
    assert out.shape == (1, 1)
