"""Data loading, train/test splitting and SQL persistence."""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller", "Animation", "Horror"]
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DB_PATH = DATA_DIR / "platform.db"


@dataclass
class Dataset:
    movies: pd.DataFrame      # item_idx, movie_id, title, genres
    train: pd.DataFrame       # user_idx, item_idx, rating
    test: pd.DataFrame
    n_users: int
    n_items: int
    user_ids: np.ndarray      # user_idx -> original id

    def matrix(self) -> np.ndarray:
        """Dense train rating matrix, NaN = unrated."""
        R = np.full((self.n_users, self.n_items), np.nan)
        R[self.train.user_idx, self.train.item_idx] = self.train.rating
        return R


def synthetic(n_users=300, n_items=400, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genre-driven synthetic ratings so models have real signal to find."""
    rng = np.random.default_rng(seed)
    item_g = rng.random((n_items, len(GENRES))) > 0.78
    for i in np.where(~item_g.any(1))[0]:
        item_g[i, rng.integers(0, len(GENRES))] = True
    movies = pd.DataFrame({
        "movie_id": np.arange(n_items),
        "title": [f"Movie {i}" for i in range(n_items)],
        "genres": ["|".join(g for g, f in zip(GENRES, row) if f) for row in item_g],
    })
    pref = rng.normal(size=(n_users, len(GENRES)))
    rows = []
    for u in range(n_users):
        n = int(rng.integers(5, 80))  # mix of cold and heavy users
        items = rng.choice(n_items, n, replace=False)
        score = item_g[items] @ pref[u] + rng.normal(0, 0.6, n)
        rating = np.clip(np.round(3 + score), 1, 5)
        rows += [(u, i, r) for i, r in zip(items, rating)]
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"]), movies


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use MovieLens (data/ratings.csv + data/movies.csv) if present, else synthetic."""
    r, m = DATA_DIR / "ratings.csv", DATA_DIR / "movies.csv"
    if r.exists() and m.exists():
        ratings = pd.read_csv(r).rename(columns={"userId": "user_id", "movieId": "movie_id"})
        movies = pd.read_csv(m).rename(columns={"movieId": "movie_id"})
        return ratings[["user_id", "movie_id", "rating"]], movies[["movie_id", "title", "genres"]]
    return synthetic()


def build_dataset(test_frac=0.2, seed=42) -> Dataset:
    ratings, movies = load_raw()
    ratings = ratings[ratings.movie_id.isin(movies.movie_id)]
    user_ids = np.sort(ratings.user_id.unique())
    movies = movies.reset_index(drop=True).assign(item_idx=lambda d: np.arange(len(d)))
    uidx = {u: i for i, u in enumerate(user_ids)}
    iidx = dict(zip(movies.movie_id, movies.item_idx))
    df = ratings.assign(user_idx=ratings.user_id.map(uidx), item_idx=ratings.movie_id.map(iidx))
    df = df[["user_idx", "item_idx", "rating"]].sample(frac=1, random_state=seed)
    # per-user holdout; users with < 5 ratings stay fully in train
    rank = df.groupby("user_idx").cumcount()
    size = df.groupby("user_idx")["rating"].transform("size")
    is_test = (size >= 5) & (rank < (size * test_frac).round().clip(lower=1))
    return Dataset(movies, df[~is_test], df[is_test], len(user_ids), len(movies), user_ids)


# ---------- SQL persistence (SQLite) ----------
def _conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)


def save_experiment(rows: pd.DataFrame) -> None:
    c = _conn()
    try:
        rows.to_sql("experiments", c, if_exists="append", index=False)
        c.commit()
    finally:
        c.close()


def list_experiments(limit=200) -> list[dict]:
    c = _conn()
    try:
        q = "SELECT * FROM experiments ORDER BY run_at DESC, ndcg DESC LIMIT ?"
        return pd.read_sql(q, c, params=(limit,)).to_dict("records")
    except Exception:  # table not created yet
        return []
    finally:
        c.close()


def user_activity(ds: Dataset) -> pd.DataFrame:
    """SQL aggregation: ratings per user (shows cold vs. heavy users)."""
    c = sqlite3.connect(":memory:")
    try:
        ds.train.to_sql("train", c, index=False)
        return pd.read_sql(
            "SELECT user_idx, COUNT(*) AS n_ratings, AVG(rating) AS mean_rating "
            "FROM train GROUP BY user_idx ORDER BY n_ratings DESC", c)
    finally:
        c.close()
