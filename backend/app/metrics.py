"""Ranking metrics. `recommended` is an ordered list of item ids, `relevant` a set."""
from __future__ import annotations
import numpy as np


def precision_at_k(recommended, relevant, k):
    return sum(i in relevant for i in recommended[:k]) / k


def recall_at_k(recommended, relevant, k):
    return sum(i in relevant for i in recommended[:k]) / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended, relevant, k):
    dcg = sum(1.0 / np.log2(r + 2) for r, i in enumerate(recommended[:k]) if i in relevant)
    ideal = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0
