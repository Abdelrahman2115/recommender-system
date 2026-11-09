"""Recommendation models.

Ranking models expose  score(user_id, event_ids) -> np.ndarray  (higher = recommend first).
Rating models expose   predict(user_ids, event_ids) -> np.ndarray  (predicted 1-5 rating).

Explained in notebooks 05 (content-based), 06 (SVD) and 07 (hybrid).
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from .features import EventFeaturizer

# Weight of each interaction when building a content profile. Attended events use the rating
# instead: rating - 2  ->  1 star = -1 (pushes the profile AWAY), 5 stars = +3.
PROFILE_WEIGHT = {"view": 0.5, "interested": 1.0, "going": 2.0}


def minmax(x):
    """Rescale scores to [0, 1] so scores from different models can be mixed."""
    x = np.asarray(x, dtype=float)
    span = x.max() - x.min()
    return np.zeros_like(x) if span == 0 else (x - x.min()) / span


# ------------------------------------------------------------------ baselines
class RandomRecommender:
    """Baseline 1: random order. Any useful model must beat this."""
    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, *args, **kwargs):
        return self

    def score(self, user_id, event_ids):
        return np.random.default_rng(self.seed + int(user_id)).random(len(event_ids))


class PopularityRecommender:
    """Baseline 2: recommend what most people engaged with (same list for everyone)."""
    def fit(self, train, *args, **kwargs):
        self.pop = train.groupby("event_id")["weight"].sum()
        return self

    def score(self, user_id, event_ids):
        return self.pop.reindex(event_ids).fillna(0).to_numpy()


class GlobalMeanPredictor:
    """Rating baseline: predict the average training rating for everything."""
    def fit(self, ratings):
        self.mu = ratings["rating"].mean()
        return self

    def predict(self, user_ids, event_ids):
        return np.full(len(user_ids), self.mu)


# ------------------------------------------------------------------ content-based
class ContentBasedRecommender:
    """Content-Based Filtering (CBF).

    * Event vectors  : TF-IDF text + category + city + price (EventFeaturizer).
    * User profile   : weighted average of the vectors of events the user engaged with,
                       blended with their stated interests + home city. New users rely only on
                       interests; the more history a user has, the more the history counts.
    * Ranking score  : cosine similarity between user profile and event vector.
    * Rating predict : item-kNN on content - start from the user's (shrunk) average rating and
                       correct it with the ratings they gave to the most similar events.
    """

    def __init__(self, featurizer=None, interest_strength=5.0, city_strength=0.5,
                 k_neighbors=20, shrinkage=5.0, neighbor_shrinkage=10.0):
        self.featurizer = featurizer or EventFeaturizer()
        self.interest_strength = interest_strength   # how many interactions the stated interests are "worth"
        self.city_strength = city_strength           # weight of the home city in the interest profile
        self.k_neighbors = k_neighbors               # neighbours used for rating prediction
        self.shrinkage = shrinkage                   # pulls users with few ratings toward the global mean
        self.neighbor_shrinkage = neighbor_shrinkage # damps the neighbour correction when similarities are weak

    def fit(self, train, users, events):
        self.X = self.featurizer.fit_transform(events).toarray()       # (n_events, n_features), unit rows
        self.row = pd.Series(np.arange(len(events)), index=events["event_id"].to_numpy())
        cats = events["category"].to_numpy()
        self.centroids = {c: normalize(self.X[cats == c].mean(axis=0, keepdims=True))[0]
                          for c in np.unique(cats)}

        # interaction weights -> behaviour profile = sum(weight * event_vector)
        w = train["interaction_type"].map(PROFILE_WEIGHT)
        w = w.where(train["interaction_type"] != "attended", train["rating"] - 2).to_numpy(float)
        rows = self.row.loc[train["event_id"]].to_numpy()
        self.behaviour, self.n_hist = {}, {}
        for uid, idx in train.groupby("user_id").indices.items():
            self.behaviour[uid] = w[idx] @ self.X[rows[idx]]
            self.n_hist[uid] = len(idx)

        # stated-interest profile = mean of the category centroids + home-city vector
        self.interest = {}
        for uid, interests, city in users[["user_id", "interests", "city"]].itertuples(index=False):
            cents = [self.centroids[c] for c in str(interests).split("|") if c in self.centroids]
            vec = normalize(np.mean(cents, axis=0, keepdims=True))[0] if cents else np.zeros(self.X.shape[1])
            self.interest[uid] = vec + self.city_strength * self.featurizer.city_vector(city)

        # data for rating prediction
        rated = train.dropna(subset=["rating"])
        self.mu = rated["rating"].mean() if len(rated) else 3.0
        self.user_ratings = {uid: (self.row.loc[g["event_id"]].to_numpy(), g["rating"].to_numpy())
                             for uid, g in rated.groupby("user_id")}
        return self

    def profile(self, user_id):
        n = self.n_hist.get(user_id, 0)
        b = self.behaviour.get(user_id)
        i = self.interest.get(user_id, np.zeros(self.X.shape[1]))
        i = normalize(i[None])[0] if np.any(i) else i
        s = self.interest_strength
        p = i * s / (n + s)
        if b is not None and np.any(b):
            p = p + normalize(b[None])[0] * n / (n + s)
        return p

    def score(self, user_id, event_ids):
        return self.X[self.row.loc[event_ids].to_numpy()] @ self.profile(user_id)

    def predict(self, user_ids, event_ids):
        out = np.empty(len(user_ids))
        for j, (u, e) in enumerate(zip(user_ids, event_ids)):
            rows, r = self.user_ratings.get(u, (np.array([], int), np.array([])))
            b_u = (r.sum() + self.shrinkage * self.mu) / (len(r) + self.shrinkage)  # shrunk user mean
            if len(r) == 0 or e not in self.row.index:
                out[j] = b_u
                continue
            sims = self.X[rows] @ self.X[self.row[e]]
            top = np.argsort(-sims)[: self.k_neighbors]
            top = top[sims[top] > 0]
            # similarity-weighted correction, damped: few/weak neighbours -> stay near the user mean
            denom = sims[top].sum() + self.neighbor_shrinkage
            out[j] = b_u if denom == 0 else b_u + sims[top] @ (r[top] - b_u) / denom
        return np.clip(out, 1, 5)


# ------------------------------------------------------------------ collaborative (SVD)
class FunkSVD:
    """Matrix factorisation for RATING prediction (the 'SVD' of the Netflix Prize).

    rating(u, e) ~ mu + b_u + b_e + p_u . q_e
    Trained only on observed ratings with stochastic gradient descent (SGD) - missing ratings are
    unknown, NOT zero, which is why we cannot simply use numpy/sklearn SVD here.
    """

    def __init__(self, n_factors=20, lr=0.01, reg=0.05, n_epochs=40, init_std=0.1, seed=42):
        self.n_factors, self.lr, self.reg = n_factors, lr, reg
        self.n_epochs, self.init_std, self.seed = n_epochs, init_std, seed

    def fit(self, ratings, val=None):
        rng = np.random.default_rng(self.seed)
        self.mu = ratings["rating"].mean()
        self.u_idx = {u: k for k, u in enumerate(ratings["user_id"].unique())}
        self.i_idx = {e: k for k, e in enumerate(ratings["event_id"].unique())}
        u = ratings["user_id"].map(self.u_idx).to_numpy()
        i = ratings["event_id"].map(self.i_idx).to_numpy()
        r = ratings["rating"].to_numpy(float)
        self.bu, self.bi = np.zeros(len(self.u_idx)), np.zeros(len(self.i_idx))
        self.P = rng.normal(0, self.init_std, (len(self.u_idx), self.n_factors))
        self.Q = rng.normal(0, self.init_std, (len(self.i_idx), self.n_factors))
        self.history = []
        lr, reg = self.lr, self.reg
        for epoch in range(self.n_epochs):
            for k in rng.permutation(len(r)):
                uu, ii = u[k], i[k]
                err = r[k] - (self.mu + self.bu[uu] + self.bi[ii] + self.P[uu] @ self.Q[ii])
                self.bu[uu] += lr * (err - reg * self.bu[uu])
                self.bi[ii] += lr * (err - reg * self.bi[ii])
                pu = self.P[uu].copy()
                self.P[uu] += lr * (err * self.Q[ii] - reg * self.P[uu])
                self.Q[ii] += lr * (err * pu - reg * self.Q[ii])
            rec = {"epoch": epoch + 1, "train_rmse": _rmse(r, self.predict(ratings["user_id"], ratings["event_id"]))}
            if val is not None:
                rec["val_rmse"] = _rmse(val["rating"].to_numpy(), self.predict(val["user_id"], val["event_id"]))
            self.history.append(rec)
        self.history = pd.DataFrame(self.history)
        return self

    def predict(self, user_ids, event_ids):
        u = np.array([self.u_idx.get(x, -1) for x in user_ids])
        i = np.array([self.i_idx.get(x, -1) for x in event_ids])
        pred = np.full(len(u), self.mu)
        ku, ki = u >= 0, i >= 0
        pred[ku] += self.bu[u[ku]]                       # unknown user -> no user bias
        pred[ki] += self.bi[i[ki]]                       # unknown event -> no event bias
        both = ku & ki
        pred[both] += np.einsum("ij,ij->i", self.P[u[both]], self.Q[i[both]])
        return np.clip(pred, 1, 5)


class PureSVDRecommender:
    """SVD for top-N RANKING ('PureSVD', Cremonesi et al., 2010).

    Build the user x event engagement matrix (missing = 0, because for ranking 'no interaction'
    really is information), factorise it with truncated SVD, and score events by the reconstructed
    row:  score(u) = r_u V^T V.
    """

    def __init__(self, n_factors=20, seed=42):
        self.n_factors, self.seed = n_factors, seed

    def fit(self, train, users, events):
        self.u_idx = pd.Series(np.arange(len(users)), index=users["user_id"].to_numpy())
        self.i_idx = pd.Series(np.arange(len(events)), index=events["event_id"].to_numpy())
        R = csr_matrix((train["weight"].to_numpy(float),
                        (self.u_idx.loc[train["user_id"]].to_numpy(), self.i_idx.loc[train["event_id"]].to_numpy())),
                       shape=(len(users), len(events)))
        self.svd = TruncatedSVD(n_components=self.n_factors, random_state=self.seed).fit(R)
        self.V = self.svd.components_                          # (k, n_events): event factors
        self.user_emb = R @ self.V.T                           # (n_users, k): user factors
        self.has_hist = np.asarray(R.sum(axis=1)).ravel() > 0
        return self

    def has_history(self, user_id):
        return user_id in self.u_idx.index and self.has_hist[self.u_idx[user_id]]

    def score(self, user_id, event_ids):
        if not self.has_history(user_id):
            return np.zeros(len(event_ids))
        return self.user_emb[self.u_idx[user_id]] @ self.V[:, self.i_idx.loc[event_ids].to_numpy()]


# ------------------------------------------------------------------ hybrid
class HybridRanker:
    """Weighted hybrid for ranking (both scores min-max scaled to 0-1 per user first).

    final = a_e * CF + (1 - a_e) * CBF
      * confidence_k > 0 : a_e = cf_weight * n_e / (n_e + k)   (confidence-weighted: a brand-new event
                           with n_e = 0 is ranked purely by content; trust in CF grows with history)
      * confidence_k = 0 : a_e = cf_weight for every event        (static weighted blend)
    n_e = training interactions of the event. Users with no history are ranked purely by content.
    """

    def __init__(self, content, collaborative, cf_weight=0.5, confidence_k=10):
        self.content, self.collaborative = content, collaborative
        self.cf_weight, self.confidence_k = cf_weight, confidence_k

    def fit(self, train, *args, **kwargs):
        self.n_event = train.groupby("event_id").size()
        return self

    def score(self, user_id, event_ids):
        s_cb = minmax(self.content.score(user_id, event_ids))
        if not self.collaborative.has_history(user_id):
            return s_cb
        s_cf = minmax(self.collaborative.score(user_id, event_ids))
        if self.confidence_k == 0:
            a = self.cf_weight
        else:
            n = self.n_event.reindex(event_ids).fillna(0).to_numpy()
            a = self.cf_weight * n / (n + self.confidence_k)
        return a * s_cf + (1 - a) * s_cb


class HybridRatingPredictor:
    """Weighted hybrid for ratings: w * SVD + (1 - w) * CBF, with w tuned on validation data.

    Unlike ranking, SVD stays useful for brand-new events here: its user bias b_u (is this person
    a generous or a harsh rater?) does not need any history of the event.
    """

    def __init__(self, content, svd, svd_weight=0.5):
        self.content, self.svd, self.svd_weight = content, svd, svd_weight

    def fit(self, *args, **kwargs):
        return self

    def predict(self, user_ids, event_ids):
        w = self.svd_weight
        return w * self.svd.predict(user_ids, event_ids) + (1 - w) * self.content.predict(user_ids, event_ids)


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
