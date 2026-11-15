from __future__ import annotations
import time
import numpy as np

from .data import Dataset
from .metrics import precision_at_k, recall_at_k, ndcg_at_k
from .models import Base, build, fill_missing

RELEVANT_THRESHOLD = 4.0  # a held-out rating >= 4 counts as relevant


def recommend(model: Base, u: int, k: int, missing: str = "popularity") -> list[tuple[int, float]]:
    scores = fill_missing(model, model.predict(u), missing)
    scores[~np.isnan(model.R[u])] = -np.inf          # never recommend already-rated items
    scores = np.where(np.isnan(scores), -np.inf, scores)  # 'drop' strategy: unpredictable items are skipped
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top if np.isfinite(scores[i])]


def evaluate(model: Base, ds: Dataset, k: int = 10, missing: str = "popularity") -> dict:
    relevant = (ds.test[ds.test.rating >= RELEVANT_THRESHOLD]
                .groupby("user_idx").item_idx.apply(set).to_dict())
    p, r, n, miss, seen = [], [], [], [], set()
    for u, rel in relevant.items():
        raw = model.predict(u)
        miss.append(np.isnan(raw[np.isnan(model.R[u])]).mean())
        recs = [i for i, _ in recommend(model, u, k, missing)]
        seen.update(recs)
        p.append(precision_at_k(recs, rel, k)); r.append(recall_at_k(recs, rel, k)); n.append(ndcg_at_k(recs, rel, k))
    return {"precision": float(np.mean(p)), "recall": float(np.mean(r)), "ndcg": float(np.mean(n)),
            "missing_rate": float(np.mean(miss)), "coverage": len(seen) / ds.n_items,
            "users_evaluated": len(p)}


def run_configs(configs: list[dict], ds: Dataset, k: int) -> list[dict]:
    results = []
    for cfg in configs:
        t = time.time()
        model = build(cfg["type"], cfg.get("params")).fit(ds)
        m = evaluate(model, ds, k, cfg.get("missing", "popularity"))
        results.append({"name": cfg["name"], "type": cfg["type"], "params": str(cfg.get("params") or {}),
                        "missing_strategy": cfg.get("missing", "popularity"), "k": k, **m,
                        "seconds": round(time.time() - t, 2)})
    return results
