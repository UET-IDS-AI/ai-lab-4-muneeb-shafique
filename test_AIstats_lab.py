import numpy as np
import pytest
import pandas as pd
from sklearn import datasets

import AI_stats_lab as A


def test_q1_gradient_descent_shapes_and_decrease():
    out = A.visualize_gradient_descent(lr=0.1, epochs=80, seed=0)

    theta_path = out["theta_path"]
    losses = out["losses"]
    X = out["X"]
    y = out["y"]

    assert isinstance(theta_path, np.ndarray) and theta_path.shape == (80, 2)
    assert isinstance(losses, np.ndarray) and losses.shape == (80,)
    assert isinstance(X, np.ndarray) and X.shape[1] == 2
    assert isinstance(y, np.ndarray) and y.shape[0] == X.shape[0]

    # Must improve substantially (not necessarily strictly monotone)
    assert losses[-1] < 0.3 * losses[0]


def test_q1_gradient_descent_linreg_basic():
    # Simple perfect line: y = 2 + 3x
    x = np.linspace(-1, 1, 50).reshape(-1, 1)
    X = A.add_bias_column(x)
    y = 2 + 3 * x.reshape(-1)

    res = A.gradient_descent_linreg(X, y, lr=0.2, epochs=400)

    assert res.theta.shape == (2,)
    assert res.losses.shape == (400,)
    assert res.thetas.shape == (400, 2)
    assert res.losses[-1] < 1e-4

    # parameters close
    assert np.allclose(res.theta[0], 2.0, atol=1e-2)
    assert np.allclose(res.theta[1], 3.0, atol=1e-2)


def test_q2_diabetes_gd_reasonable_metrics():
    train_mse, test_mse, train_r2, test_r2, theta = A.diabetes_linear_gd(
        lr=0.05, epochs=4000, test_size=0.2, seed=0
    )
    assert np.isfinite(train_mse) and np.isfinite(test_mse)
    assert np.isfinite(train_r2) and np.isfinite(test_r2)
    assert theta.ndim == 1

    # sanity: training error should usually be <= test error
    assert train_mse <= test_mse * 1.5

    # should do better than predicting mean (R2 > 0 often)
    assert test_r2 > 0.25


def test_q3_diabetes_analytical_reasonable_metrics():
    train_mse, test_mse, train_r2, test_r2, theta = A.diabetes_linear_analytical(
        ridge_lambda=1e-8, test_size=0.2, seed=0
    )
    assert np.isfinite(train_mse) and np.isfinite(test_mse)
    assert np.isfinite(train_r2) and np.isfinite(test_r2)
    assert theta.ndim == 1

    assert test_r2 > 0.25


def test_q4_compare_gd_vs_analytical_close():
    comp = A.diabetes_compare_gd_vs_analytical(
        lr=0.05, epochs=6000, test_size=0.2, seed=0
    )

    # expected keys
    for k in [
        "theta_l2_diff",
        "train_mse_diff",
        "test_mse_diff",
        "train_r2_diff",
        "test_r2_diff",
        "theta_cosine_sim",
    ]:
        assert k in comp
        assert np.isfinite(comp[k])

    # close solutions
    assert comp["theta_cosine_sim"] > 0.98
    assert comp["test_mse_diff"] < 50.0
    assert comp["test_r2_diff"] < 0.05
