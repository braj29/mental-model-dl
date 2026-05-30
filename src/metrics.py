"""
Continual-learning metrics (Lopez-Paz & Ranzato, 2017).

Given an accuracy matrix R where R[i, j] = test accuracy on task j after
training on task i:

    Average Accuracy (ACC) = mean over j of R[T-1, j]
    Backward Transfer (BWT) = mean_{j<T-1} ( R[T-1, j] - R[j, j] )
        negative => forgetting
    Forward Transfer (FWT) = mean_{j>0} ( R[j-1, j] - random_acc )
"""

from __future__ import annotations

import numpy as np


def average_accuracy(R: np.ndarray) -> float:
    return float(R[-1].mean())


def backward_transfer(R: np.ndarray) -> float:
    T = R.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([R[-1, j] - R[j, j] for j in range(T - 1)]))


def forward_transfer(R: np.ndarray, random_acc: float = 0.0) -> float:
    T = R.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([R[j - 1, j] - random_acc for j in range(1, T)]))


def summarize(R: np.ndarray, random_acc: float = 0.0) -> dict:
    return {
        "ACC": average_accuracy(R),
        "BWT": backward_transfer(R),
        "FWT": forward_transfer(R, random_acc),
    }
