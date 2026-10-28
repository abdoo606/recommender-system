from __future__ import annotations

import pytest

from reco.cf import Interaction, Matrix, evaluate, popularity_baseline, precision_at_k, recommend_item_based, recommend_user_based

# Two taste clusters: users 1-3 like sci-fi, users 4-6 like romance.
DATA = [
    Interaction("u1", "dune"), Interaction("u1", "foundation"),
    Interaction("u2", "dune"), Interaction("u2", "foundation"), Interaction("u2", "hyperion"),
    Interaction("u3", "dune"), Interaction("u3", "hyperion"),
    Interaction("u4", "emma"), Interaction("u4", "persuasion"),
    Interaction("u5", "emma"), Interaction("u5", "persuasion"), Interaction("u5", "sense"),
    Interaction("u6", "emma"), Interaction("u6", "sense"),
]


@pytest.fixture()
def matrix():
    return Matrix(DATA)


def test_same_cluster_is_more_similar(matrix):
    assert matrix.user_similarity("u1", "u2") > matrix.user_similarity("u1", "u4")


def test_similarity_is_symmetric(matrix):
    assert matrix.user_similarity("u2", "u3") == pytest.approx(matrix.user_similarity("u3", "u2"))


def test_unknown_user_has_zero_similarity(matrix):
    assert matrix.user_similarity("ghost", "u1") == 0.0


def test_user_based_recommends_from_the_same_taste_cluster(matrix):
    recs = [item for item, _ in recommend_user_based(matrix, "u1", k=2)]
    assert "hyperion" in recs
    assert "emma" not in recs


def test_seen_items_are_excluded(matrix):
    recs = [item for item, _ in recommend_user_based(matrix, "u2", k=5)]
    assert "dune" not in recs and "foundation" not in recs


def test_item_based_recommendation(matrix):
    recs = [item for item, _ in recommend_item_based(matrix, "u1", k=3)]
    assert "hyperion" in recs


def test_popularity_baseline(matrix):
    assert popularity_baseline(matrix, "u1", k=1) == ["emma"]


def test_unknown_user_gets_nothing(matrix):
    assert recommend_user_based(matrix, "ghost") == []
    assert recommend_item_based(matrix, "ghost") == []


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, k=2) == 0.5
    assert precision_at_k([], {"a"}, k=3) == 0.0
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, k=0)


def test_evaluate_reports_metrics(matrix):
    relevant = {"u1": {"hyperion"}, "u4": {"sense"}}
    result = evaluate(matrix, users=["u1", "u4"], relevant=relevant, k=2, strategy="user")
    assert 0 <= result["precision@2"] <= 1
    assert 0 < result["coverage"] <= 1
    assert result["users_evaluated"] == 2.0


def test_evaluate_popularity_baseline_has_low_coverage(matrix):
    relevant = {"u1": {"hyperion"}}
    item = evaluate(matrix, users=["u1"], relevant=relevant, k=1, strategy="item")
    popular = evaluate(matrix, users=["u1"], relevant=relevant, k=1, strategy="popular")
    assert item["coverage"] >= popular["coverage"]
