"""Evaluation protocol shared by all notebooks."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import POSITIVE_TYPES
from .metrics import mae, ndcg_at_k, precision_at_k, recall_at_k, rmse


@dataclass
class RankingTask:
    candidates: np.ndarray      # events that take place in the evaluation window
    relevant: dict              # user_id -> set of events the user went to / attended in the window
    seen: dict                  # user_id -> events the user already interacted with before the cut-off


def make_ranking_task(history, future, events, window_start, window_end):
    """At time `window_start`, recommend events happening in (window_start, window_end].

    A recommendation is 'relevant' if the user later marked that event going/attended.
    Only information available at `window_start` (the history) may be used by the models.
    """
    start = events["start_datetime"]
    candidates = events.loc[(start > window_start) & (start <= window_end), "event_id"].to_numpy()
    pos = future[future["interaction_type"].isin(POSITIVE_TYPES) & future["event_id"].isin(candidates)]
    relevant = pos.groupby("user_id")["event_id"].apply(set).to_dict()
    seen = history.groupby("user_id")["event_id"].apply(set).to_dict()
    return RankingTask(candidates, relevant, seen)


def recommend(model, user_id, candidates, seen=(), k=10):
    """Top-k events for one user, never re-recommending events they already interacted with."""
    cands = np.array([e for e in candidates if e not in seen])
    scores = model.score(user_id, cands)
    return cands[np.argsort(-scores, kind="stable")[:k]]


def evaluate_ranking(model, task, k=10):
    """Per-user Precision@k, Recall@k and NDCG@k (one row per evaluated user)."""
    rows = []
    for uid, rel in task.relevant.items():
        top = list(recommend(model, uid, task.candidates, task.seen.get(uid, set()), k))
        rows.append({"user_id": uid, "n_relevant": len(rel),
                     f"precision@{k}": precision_at_k(top, rel, k),
                     f"recall@{k}": recall_at_k(top, rel, k),
                     f"ndcg@{k}": ndcg_at_k(top, rel, k)})
    return pd.DataFrame(rows)


def evaluate_rating(model, pairs):
    """MAE and RMSE on (user, event, rating) rows."""
    pred = model.predict(pairs["user_id"].to_numpy(), pairs["event_id"].to_numpy())
    return {"MAE": mae(pairs["rating"], pred), "RMSE": rmse(pairs["rating"], pred)}


def bootstrap_ci(values, n_boot=1000, seed=42, alpha=0.05):
    """95% confidence interval of a mean by resampling users with replacement."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
