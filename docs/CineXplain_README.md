# CineXplain: Hybrid Movie Recommendation System

<!-- One sentence: what it recommends, for whom, and the headline idea (hybrid + adaptive weighting). -->

**Live demo / screenshots:** <!-- link or docs/ images -->

## Problem
<!-- Why single-method recommenders fail: cold-start for content-light users, sparsity for CF. -->

## Approach

### Content-based filtering
<!-- Features used (genres, overview, cast...), vectorisation (TF-IDF?), similarity measure, how a user profile/prediction is formed. -->

### Collaborative filtering
<!-- User-based or item-based? Similarity measure, neighbourhood size, mean-centring, sparsity handling. -->

### Hybrid architecture
<!-- Diagram (docs/architecture.png): how both predictions combine, e.g. final = alpha * CF + (1 - alpha) * CB -->

### Adaptive alpha weighting
<!-- Exact rule for alpha per user/item (e.g. based on number of ratings), why, and what you observed vs fixed alpha. -->

### Missing prediction handling
<!-- When CF/CB can't predict, what you fall back to and in what order. Report how often it happens. -->

## Evaluation
<!-- Split method, metrics (Precision@K, Recall@K, NDCG@K, RMSE/MAE if used), results table vs baselines. Only real numbers. -->

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Content-based | | | |
| Collaborative | | | |
| Hybrid (fixed alpha) | | | |
| Hybrid (adaptive alpha) | | | |

## Tech stack
<!-- Python, libraries, dataset (MovieLens?), UI framework -->

## Setup
```bash
git clone <repo> && cd CineXplain
pip install -r requirements.txt
# command to run
```

## My contributions
<!-- Team project? Be specific and honest: "I implemented X, designed Y, evaluated Z". Mentors read this section closely. -->
- 
- 

## Future work
-
