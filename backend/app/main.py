from __future__ import annotations
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import data
from .evaluate import recommend, run_configs
from .models import REGISTRY, build

app = FastAPI(title="Recommendation Evaluation Platform")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=1)
def dataset() -> data.Dataset:
    return data.build_dataset()


class Config(BaseModel):
    name: str
    type: str = Field(description=f"one of {list(REGISTRY)}")
    params: dict = {}
    missing: str = Field("popularity", pattern="^(popularity|global_mean|drop)$")


class EvalRequest(BaseModel):
    configs: list[Config]
    k: int = Field(10, ge=1, le=50)


@app.get("/api/health")
def health():
    ds = dataset()
    return {"users": ds.n_users, "items": ds.n_items, "train": len(ds.train), "test": len(ds.test)}


@app.get("/api/models")
def models():
    return {"models": list(REGISTRY)}


@app.post("/api/evaluate")
def evaluate_configs(req: EvalRequest):
    try:
        results = run_configs([c.model_dump() for c in req.configs], dataset(), req.k)
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))
    data.save_experiment(pd.DataFrame(results).assign(run_at=datetime.now(timezone.utc).isoformat()))
    return {"results": results}


@app.get("/api/experiments")
def experiments():
    return {"experiments": data.list_experiments()}


@app.get("/api/recommend/{user_idx}")
def recommend_for_user(user_idx: int, model: str = "hybrid", k: int = Query(10, ge=1, le=50),
                       missing: str = "popularity"):
    ds = dataset()
    if not 0 <= user_idx < ds.n_users:
        raise HTTPException(404, "unknown user")
    try:
        m = build(model).fit(ds)
    except ValueError as e:
        raise HTTPException(400, str(e))
    titles = ds.movies.set_index("item_idx")
    return {"user_idx": user_idx, "model": model, "ratings_in_train": m.n_rated(user_idx),
            "alpha": getattr(m, "alpha_for", lambda u: None)(user_idx),
            "recommendations": [{"item_idx": i, "title": titles.loc[i, "title"],
                                 "genres": titles.loc[i, "genres"], "score": round(s, 3)}
                                for i, s in recommend(m, user_idx, k, missing)]}


@app.get("/api/users")
def users(limit: int = 30):
    return {"users": data.user_activity(dataset()).head(limit).to_dict("records")}
