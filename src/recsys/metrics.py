"""Evaluation metrics (explained in notebooks/08_evaluation_and_conclusions.ipynb)."""
import numpy as np


def mae(y_true, y_pred):
    """Mean Absolute Error: average size of the rating mistake, in stars."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    """Root Mean Squared Error: like MAE but punishes big mistakes more."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def precision_at_k(recommended, relevant, k=10):
    """Share of the top-k recommendations that were relevant."""
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended, relevant, k=10):
    """Share of the relevant events that appeared in the top-k."""
    return len(set(recommended[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended, relevant, k=10):
    """Normalized Discounted Cumulative Gain: rewards relevant events placed near the TOP.

    DCG  = sum over hits of 1 / log2(position + 1)        (position starts at 1)
    IDCG = DCG of the perfect ordering (all relevant events first)
    NDCG = DCG / IDCG, between 0 and 1.
    """
    dcg = sum(1.0 / np.log2(pos + 2) for pos, e in enumerate(recommended[:k]) if e in relevant)
    idcg = sum(1.0 / np.log2(pos + 2) for pos in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0
