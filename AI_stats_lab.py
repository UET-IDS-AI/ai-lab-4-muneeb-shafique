from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

# =========================
# Helpers
# =========================

def add_bias_column(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return np.hstack([np.ones((X.shape[0], 1)), X])

def standardize_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)

@dataclass
class GDResult:
    theta: np.ndarray
    losses: np.ndarray
    thetas: np.ndarray

# =========================
# Q1: Gradient descent + visualization
# =========================

def gradient_descent_linreg(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 200,
    theta0: Optional[np.ndarray] = None,
) -> GDResult:
    n, d = X.shape
    y = y.reshape(-1)
    theta = np.zeros(d) if theta0 is None else theta0.copy()
    losses = []
    thetas = []

    for _ in range(epochs):
        y_pred = X @ theta
        error = y_pred - y
        loss = np.mean(error ** 2)
        losses.append(loss)
        thetas.append(theta.copy())
        grad = (2 / n) * (X.T @ error)
        theta = theta - lr * grad

    return GDResult(
        theta=theta,
        losses=np.array(losses),
        thetas=np.array(thetas)
    )

def visualize_gradient_descent(
    lr: float = 0.1,
    epochs: int = 60,
    seed: int = 0
) -> Dict[str, np.ndarray]:
    np.random.seed(seed)
    n = 50
    X = np.random.uniform(-1, 1, size=(n, 1))
    y = 1.5 + 2.5 * X[:, 0] + np.random.randn(n) * 0.2
    X_b = add_bias_column(X)
    res = gradient_descent_linreg(X_b, y, lr=lr, epochs=epochs)
    return {
        "theta_path": res.thetas,
        "losses": res.losses,
        "X": X_b,
        "y": y
    }

# =========================
# Q2: Diabetes regression using GD
# =========================

def diabetes_linear_gd(
    lr: float = 0.05,
    epochs: int = 2000,
    test_size: float = 0.2,
    seed: int = 0
) -> Tuple[float, float, float, float, np.ndarray]:
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train_std, X_test_std, _, _ = standardize_train_test(X_train, X_test)
    X_train_std = add_bias_column(X_train_std)
    X_test_std = add_bias_column(X_test_std)
    res = gradient_descent_linreg(X_train_std, y_train, lr=lr, epochs=epochs)
    y_train_pred = X_train_std @ res.theta
    y_test_pred = X_test_std @ res.theta
    return (
        mse(y_train, y_train_pred),
        mse(y_test, y_test_pred),
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred),
        res.theta
    )

# =========================
# Q3: Diabetes regression using analytical solution
# =========================

def diabetes_linear_analytical(
    ridge_lambda: float = 1e-8,
    test_size: float = 0.2,
    seed: int = 0
) -> Tuple[float, float, float, float, np.ndarray]:
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train_std, X_test_std, _, _ = standardize_train_test(X_train, X_test)
    X_train_std = add_bias_column(X_train_std)
    X_test_std = add_bias_column(X_test_std)
    d = X_train_std.shape[1]
    I = np.eye(d)
    theta = np.linalg.inv(X_train_std.T @ X_train_std + ridge_lambda * I) @ X_train_std.T @ y_train
    y_train_pred = X_train_std @ theta
    y_test_pred = X_test_std @ theta
    return (
        mse(y_train, y_train_pred),
        mse(y_test, y_test_pred),
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred),
        theta
    )

# =========================
# Q4: Compare GD vs analytical
# =========================

def diabetes_compare_gd_vs_analytical(
    lr: float = 0.05,
    epochs: int = 4000,
    test_size: float = 0.2,
    seed: int = 0
) -> Dict[str, float]:
    train_mse_gd, test_mse_gd, train_r2_gd, test_r2_gd, theta_gd = diabetes_linear_gd(
        lr=lr, epochs=epochs, test_size=test_size, seed=seed
    )
    train_mse_an, test_mse_an, train_r2_an, test_r2_an, theta_an = diabetes_linear_analytical(
        ridge_lambda=1e-8, test_size=test_size, seed=seed
    )
    theta_diff = np.linalg.norm(theta_gd - theta_an)
    cosine_sim = (theta_gd @ theta_an) / (np.linalg.norm(theta_gd) * np.linalg.norm(theta_an))
    return {
        "theta_l2_diff": theta_diff,
        "train_mse_diff": abs(train_mse_gd - train_mse_an),
        "test_mse_diff": abs(test_mse_gd - test_mse_an),
        "train_r2_diff": abs(train_r2_gd - train_r2_an),
        "test_r2_diff": abs(test_r2_gd - test_r2_an),
        "theta_cosine_sim": cosine_sim
    }
