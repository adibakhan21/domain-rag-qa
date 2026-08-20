"""Evaluation metrics, checked against hand-computed values."""
from __future__ import annotations

import math

import pytest

from rag_system.evaluation.qa_metrics import (answer_in_context, context_relevance,
                                              exact_match, groundedness,
                                              normalize_answer, token_f1)
from rag_system.evaluation.retrieval_metrics import (evaluate_run, hit_at_k, ndcg_at_k,
                                                     paired_bootstrap_test,
                                                     recall_at_k, reciprocal_rank)


# --- retrieval -------------------------------------------------------------
def test_recall_vs_hit_differ_when_gold_is_partially_found():
    ranked, gold = ["a", "b", "c"], ["c", "z"]
    assert recall_at_k(ranked, gold, 3) == 0.5   # 1 of 2 gold found
    assert hit_at_k(ranked, gold, 3) == 1.0      # at least one found


def test_reciprocal_rank_uses_first_relevant_position():
    assert reciprocal_rank(["a", "b", "c"], ["c"], 10) == pytest.approx(1 / 3)
    assert reciprocal_rank(["c", "b"], ["c"], 10) == 1.0
    assert reciprocal_rank(["a", "b"], ["z"], 10) == 0.0


def test_reciprocal_rank_respects_the_cutoff():
    assert reciprocal_rank(["a", "b", "c"], ["c"], k=2) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["a", "b", "c"], ["a", "b"], 10) == pytest.approx(1.0)


def test_ndcg_penalises_lower_placement():
    good = ndcg_at_k(["a", "x", "y"], ["a"], 10)
    bad = ndcg_at_k(["x", "y", "a"], ["a"], 10)
    assert good > bad


def test_ndcg_hand_computed():
    # relevant at ranks 3 and 5 -> DCG = 1/log2(4) + 1/log2(6)
    dcg = 1 / math.log2(4) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["a", "b", "c", "d", "e"], ["c", "e"], 5) == pytest.approx(dcg / idcg)


def test_missing_question_counts_as_failure_not_skip():
    scored = evaluate_run({"q1": ["a"]}, {"q1": ["a"], "q2": ["b"]}, k_values=(1,), n_bootstrap=0)
    assert scored["summary"]["n_queries"] == 2
    assert scored["summary"]["hit@1"] == 0.5


def test_evaluate_run_rejects_empty_qrels():
    with pytest.raises(ValueError, match="no question"):
        evaluate_run({}, {}, k_values=(1,), n_bootstrap=0)


def test_bootstrap_ci_brackets_the_mean():
    run = {f"q{i}": (["gold"] if i % 2 == 0 else ["other"]) for i in range(100)}
    qrels = {f"q{i}": ["gold"] for i in range(100)}
    scored = evaluate_run(run, qrels, k_values=(1,), n_bootstrap=500)
    lo, hi = scored["summary"]["ci95"]["hit@1"]
    assert lo <= scored["summary"]["hit@1"] <= hi


def test_paired_bootstrap_detects_a_real_difference():
    a = {f"q{i}": {"hit@5": 0.0} for i in range(60)}
    b = {f"q{i}": {"hit@5": 1.0} for i in range(60)}
    result = paired_bootstrap_test(a, b, "hit@5", n_bootstrap=2000)
    assert result["mean_delta"] == pytest.approx(1.0)
    assert result["significant_at_0.05"]


def test_paired_bootstrap_reports_no_difference_when_identical():
    a = {f"q{i}": {"hit@5": float(i % 2)} for i in range(60)}
    result = paired_bootstrap_test(a, a, "hit@5", n_bootstrap=2000)
    assert result["mean_delta"] == pytest.approx(0.0)
    assert not result["significant_at_0.05"]


# --- QA --------------------------------------------------------------------
def test_squad_normalisation_removes_articles_case_and_punctuation():
    assert normalize_answer("The  Virus, indeed!") == "virus indeed"


def test_unicode_punctuation_is_normalised():
    # CORD-19 uses U+2010 hyphens; the official ASCII-only script would miss this.
    assert exact_match("HCoV-HKU1", "HCoV‐HKU1") == 1.0
    assert exact_match("the virus's spike", "the virus’s spike") == 1.0


def test_normalisation_still_separates_different_answers():
    assert exact_match("cats", "dogs") == 0.0


def test_token_f1_partial_overlap():
    # pred 5 tokens, gold 5 tokens, 4 shared -> P=R=0.8, F1=0.8
    assert token_f1("main cause of HIV infection",
                    "main cause of HIV transmission") == pytest.approx(0.8)


def test_token_f1_is_zero_without_overlap():
    assert token_f1("alpha beta", "gamma delta") == 0.0


def test_empty_prediction_scores_zero_against_nonempty_gold():
    assert token_f1("", "something") == 0.0
    assert exact_match("", "something") == 0.0


def test_answer_in_context_is_normalisation_aware():
    assert answer_in_context("SARS-CoV-2", "the virus sars cov 2 spreads") == 0.0
    assert answer_in_context("SARS-CoV-2", "we studied SARS-CoV-2 closely") == 1.0


def test_groundedness_measures_token_overlap_with_context():
    assert groundedness("mother to child", "mother to child transmission") == 1.0
    assert groundedness("mother zebra", "mother to child") == pytest.approx(0.5)
    assert groundedness("", "anything") == 0.0


def test_context_relevance_is_precision_over_retrieved():
    assert context_relevance(["a", "b"], ["a", "x", "y", "b"]) == 0.5
    assert context_relevance(["a"], []) == 0.0
