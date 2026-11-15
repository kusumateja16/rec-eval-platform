"""Recommenders. Every model returns a score per item with NaN where it cannot predict."""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import Dataset


def _knn_predict(S: np.ndarray, rated: np.ndarray, ratings: np.ndarray, k: int) -> np.ndarray:
    """Similarity-weighted average of the user's ratings over each item's top-k rated neighbours."""
    sub = np.clip(S[:, rated], 0, None)
    if sub.shape[1] > k:
        thr = -np.partition(-sub, k - 1, axis=1)[:, k - 1:k]
        sub = np.where(sub >= thr, sub, 0.0)
    den = sub.sum(1)
    out = np.full(S.shape[0], np.nan)
    ok = den > 1e-9
    out[ok] = (sub @ ratings)[ok] / den[ok]
    return out


class Base:
    name = "base"

    def fit(self, ds: Dataset):
        self.ds, self.R = ds, ds.matrix()
        self.global_mean = float(np.nanmean(self.R))
        counts = (~np.isnan(self.R)).sum(0)
        sums = np.nansum(self.R, 0)
        # popularity prior: smoothed item mean (shrinks rarely-rated items toward the global mean)
        self.item_mean = (sums + 5 * self.global_mean) / (counts + 5)
        self._fit()
        return self

    def _fit(self): ...

    def n_rated(self, u: int) -> int:
        return int((~np.isnan(self.R[u])).sum())

    def predict(self, u: int) -> np.ndarray:  # raw, may contain NaN
        raise NotImplementedError

    def _user(self, u):
        m = ~np.isnan(self.R[u])
        return np.where(m)[0], self.R[u, m]


class Popularity(Base):
    name = "popularity"

    def predict(self, u):
        return self.item_mean.copy()


class ContentBased(Base):
    """Item-item kNN on TF-IDF genre vectors."""
    name = "content"

    def __init__(self, k=30):
        self.k = k

    def _fit(self):
        X = TfidfVectorizer(token_pattern=r"[^|]+", lowercase=True).fit_transform(self.ds.movies.genres)
        self.S = cosine_similarity(X)
        np.fill_diagonal(self.S, 0)

    def predict(self, u):
        rated, r = self._user(u)
        return _knn_predict(self.S, rated, r, self.k) if len(rated) else np.full(self.ds.n_items, np.nan)


class Collaborative(Base):
    """Item-based CF: cosine over user-mean-centred rating columns."""
    name = "collaborative"

    def __init__(self, k=30, min_overlap=2):
        self.k, self.min_overlap = k, min_overlap

    def _fit(self):
        mask = ~np.isnan(self.R)
        user_mean = np.where(mask.any(1), np.nansum(self.R, 1) / np.maximum(mask.sum(1), 1), self.global_mean)
        C = np.where(mask, self.R - user_mean[:, None], 0.0)
        self.S = cosine_similarity(C.T)
        overlap = mask.T.astype(float) @ mask.astype(float)  # co-rating counts
        self.S[overlap < self.min_overlap] = 0.0           # drop similarities backed by too little evidence
        np.fill_diagonal(self.S, 0)

    def predict(self, u):
        rated, r = self._user(u)
        return _knn_predict(self.S, rated, r, self.k) if len(rated) else np.full(self.ds.n_items, np.nan)


class Hybrid(Base):
    """alpha * CF + (1 - alpha) * CB.  alpha='adaptive' => alpha_u = n_u / (n_u + c)."""
    name = "hybrid"

    def __init__(self, alpha="adaptive", c=15.0, k=30, min_overlap=2):
        self.alpha, self.c = alpha, float(c)
        self.cb, self.cf = ContentBased(k), Collaborative(k, min_overlap)

    def _fit(self):
        self.cb.fit(self.ds)
        self.cf.fit(self.ds)

    def alpha_for(self, u: int) -> float:
        if self.alpha == "adaptive":
            n = self.n_rated(u)
            return n / (n + self.c)  # few ratings -> lean on content; many -> trust collaborative
        return float(self.alpha)

    def predict(self, u):
        a = self.alpha_for(u)
        cf, cb = self.cf.predict(u), self.cb.predict(u)
        out = a * cf + (1 - a) * cb
        out = np.where(np.isnan(cf), cb, out)   # CF missing -> content only
        out = np.where(np.isnan(cb), cf, out)   # CB missing -> CF only
        return out                               # both missing stays NaN -> handled by fill_missing


def fill_missing(model: Base, scores: np.ndarray, strategy: str) -> np.ndarray:
    """strategy: 'popularity' | 'global_mean' | 'drop' (leave NaN, item is not ranked)."""
    out = scores.copy()
    nan = np.isnan(out)
    if strategy == "popularity":
        out[nan] = model.item_mean[nan] - 1e-3   # sits just below real predictions on ties
    elif strategy == "global_mean":
        out[nan] = model.global_mean
    return out


REGISTRY = {"popularity": Popularity, "content": ContentBased,
            "collaborative": Collaborative, "hybrid": Hybrid}


def build(kind: str, params: dict | None = None) -> Base:
    if kind not in REGISTRY:
        raise ValueError(f"unknown model '{kind}', choose from {list(REGISTRY)}")
    return REGISTRY[kind](**(params or {}))
