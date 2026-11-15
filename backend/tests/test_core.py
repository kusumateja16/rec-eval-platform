import numpy as np
import pytest
from app.metrics import precision_at_k, recall_at_k, ndcg_at_k
from app import data
from app.models import build
from app.evaluate import evaluate, recommend


def test_metrics_known_values():
    rec, rel = [1, 2, 3, 4], {1, 3, 9}
    assert precision_at_k(rec, rel, 4) == 0.5
    assert recall_at_k(rec, rel, 4) == pytest.approx(2 / 3)
    dcg = 1 + 1 / np.log2(4)
    ideal = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert ndcg_at_k(rec, rel, 4) == pytest.approx(dcg / ideal)
    assert ndcg_at_k([5, 6], {1}, 2) == 0.0


@pytest.fixture(scope="module")
def ds():
    return data.build_dataset()


def test_no_leakage(ds):
    tr = set(zip(ds.train.user_idx, ds.train.item_idx))
    assert not tr & set(zip(ds.test.user_idx, ds.test.item_idx))


def test_adaptive_alpha_grows_with_activity(ds):
    h = build("hybrid").fit(ds)
    counts = {u: h.n_rated(u) for u in range(ds.n_users)}
    lo, hi = min(counts, key=counts.get), max(counts, key=counts.get)
    assert h.alpha_for(lo) < h.alpha_for(hi)


def test_missing_handling_and_no_rated_items(ds):
    cf = build("collaborative", {"min_overlap": 50}).fit(ds)   # strict -> many NaNs
    assert np.isnan(cf.predict(0)).any()
    recs = recommend(cf, 0, 10, "popularity")
    assert len(recs) == 10 and all(np.isnan(cf.R[0, i]) for i, _ in recs)
    assert len(recommend(cf, 0, 10, "drop")) <= 10


def test_models_beat_popularity(ds):
    pop = evaluate(build("popularity").fit(ds), ds)
    for kind in ("content", "hybrid"):
        assert evaluate(build(kind).fit(ds), ds)["ndcg"] > pop["ndcg"]
