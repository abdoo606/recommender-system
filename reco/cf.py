"""User-based and item-based CF, plus precision@k against a popularity baseline."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Interaction:
    user: str
    item: str
    rating: float = 1.0


class Matrix:
    """Sparse user-item interactions with cosine similarity helpers."""

    def __init__(self, interactions: list[Interaction]) -> None:
        self.by_user: dict[str, dict[str, float]] = defaultdict(dict)
        self.by_item: dict[str, dict[str, float]] = defaultdict(dict)
        for row in interactions:
            self.by_user[row.user][row.item] = row.rating
            self.by_item[row.item][row.user] = row.rating
        self._user_norms = {u: math.sqrt(sum(v * v for v in items.values())) for u, items in self.by_user.items()}
        self._item_norms = {i: math.sqrt(sum(v * v for v in users.values())) for i, users in self.by_item.items()}

    def user_similarity(self, a: str, b: str) -> float:
        return self._cosine(self.by_user.get(a, {}), self.by_user.get(b, {}),
                            self._user_norms.get(a, 0.0), self._user_norms.get(b, 0.0))

    def item_similarity(self, a: str, b: str) -> float:
        return self._cosine(self.by_item.get(a, {}), self.by_item.get(b, {}),
                            self._item_norms.get(a, 0.0), self._item_norms.get(b, 0.0))

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float], ln: float, rn: float) -> float:
        if not ln or not rn:
            return 0.0
        shared = set(left) & set(right)
        if not shared:
            return 0.0
        dot = sum(left[k] * right[k] for k in shared)
        return dot / (ln * rn)


def recommend_user_based(
    matrix: Matrix, user: str, *, k: int = 3, neighbours: int = 10, exclude_seen: bool = True
) -> list[tuple[str, float]]:
    """Scores unseen items by the ratings of the most similar users."""
    if user not in matrix.by_user:
        return []
    seen = set(matrix.by_user[user])
    scores: dict[str, float] = defaultdict(float)
    weights: dict[str, float] = defaultdict(float)

    similar = sorted(
        ((other, matrix.user_similarity(user, other)) for other in matrix.by_user if other != user),
        key=lambda pair: -pair[1],
    )[:neighbours]

    for other, similarity in similar:
        if similarity <= 0:
            continue
        for item, rating in matrix.by_user[other].items():
            if exclude_seen and item in seen:
                continue
            scores[item] += similarity * rating
            weights[item] += similarity

    ranked = [(item, round(scores[item] / weights[item], 4)) for item in scores if weights[item]]
    return sorted(ranked, key=lambda pair: (-pair[1], pair[0]))[:k]


def recommend_item_based(matrix: Matrix, user: str, *, k: int = 3) -> list[tuple[str, float]]:
    """Scores unseen items by how similar they are to what the user already likes."""
    if user not in matrix.by_user:
        return []
    profile = matrix.by_user[user]
    scores: dict[str, float] = defaultdict(float)
    for candidate in matrix.by_item:
        if candidate in profile:
            continue
        numerator = sum(matrix.item_similarity(candidate, liked) * rating for liked, rating in profile.items())
        denominator = sum(matrix.item_similarity(candidate, liked) for liked in profile) or 1.0
        if numerator:
            scores[candidate] = round(numerator / denominator, 4)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:k]


def popularity_baseline(matrix: Matrix, user: str, *, k: int = 3) -> list[str]:
    """Recommending the most popular items is a shockingly strong baseline."""
    seen = set(matrix.by_user.get(user, {}))
    ranked = sorted(matrix.by_item, key=lambda item: -len(matrix.by_item[item]))
    return [item for item in ranked if item not in seen][:k]


def precision_at_k(
    recommended: list[str], relevant: set[str], *, k: int
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    top = recommended[:k]
    return sum(item in relevant for item in top) / k if top else 0.0


def evaluate(
    matrix: Matrix, *, users: list[str], relevant: dict[str, set[str]], k: int = 3, strategy: str = "user"
) -> dict[str, float]:
    """Mean precision@k, plus coverage - the share of the catalogue ever recommended."""
    scored = []
    recommended_items: set[str] = set()
    for user in users:
        if strategy == "user":
            recs = [item for item, _ in recommend_user_based(matrix, user, k=k)]
        elif strategy == "item":
            recs = [item for item, _ in recommend_item_based(matrix, user, k=k)]
        else:
            recs = popularity_baseline(matrix, user, k=k)
        scored.append(precision_at_k(recs, relevant.get(user, set()), k=k))
        recommended_items.update(recs)
    return {
        f"precision@{k}": round(sum(scored) / len(scored), 4) if scored else 0.0,
        "coverage": round(len(recommended_items) / len(matrix.by_item), 4) if matrix.by_item else 0.0,
        "users_evaluated": float(len(scored)),
    }
