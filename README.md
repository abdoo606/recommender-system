# 25 · reco

Collaborative filtering with an offline evaluation harness.

## Run

```bash
from reco.cf import Matrix, Interaction, recommend_user_based
```

## Test

```bash
pytest projects/25-recommender -q
```

## What it demonstrates

- User-based and item-based cosine CF on a sparse matrix
- precision@k and catalogue coverage, not just 'it looks right'
- Popularity baseline included because it is hard to beat
- Unknown users return empty lists instead of crashing

## Repository
- GitHub: https://github.com/{owner}/recommender-system
- Python 3.11+ · tests: `pytest` · runtime dependencies: none (standard library only)
