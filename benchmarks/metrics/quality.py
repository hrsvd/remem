from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from benchmarks.model import DecisionObservation


def _divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(observations: Iterable[DecisionObservation]) -> dict[str, Any]:
    rows = list(observations)
    predicted_reuse = [row for row in rows if row.predicted != "miss"]
    expected_reuse = [row for row in rows if row.expected != "miss"]
    true_positive = sum(row.safe and row.predicted != "miss" for row in rows)
    false_positive = sum(not row.safe and row.predicted != "miss" for row in rows)
    false_negative = sum(
        row.expected != "miss" and row.predicted == "miss" for row in rows
    )
    true_negative = sum(
        row.expected == "miss" and row.predicted == "miss" for row in rows
    )
    precision = _divide(true_positive, true_positive + false_positive)
    recall = _divide(true_positive, true_positive + false_negative)
    f1 = _divide(2 * precision * recall, precision + recall)
    latencies_ms = [row.latency_ns / 1_000_000 for row in rows]
    response_rows = [row for row in rows if row.predicted == "response_reuse"]
    retrieval_rows = [row for row in rows if row.predicted == "retrieval_reuse"]
    expected_response_rows = [row for row in rows if row.expected == "response_reuse"]
    expected_retrieval_rows = [row for row in rows if row.expected == "retrieval_reuse"]
    correct_response = sum(row.safe for row in response_rows)
    correct_retrieval = sum(row.safe for row in retrieval_rows)
    response_precision = _divide(correct_response, len(response_rows))
    response_recall = _divide(correct_response, len(expected_response_rows))
    retrieval_precision = _divide(correct_retrieval, len(retrieval_rows))
    retrieval_recall = _divide(correct_retrieval, len(expected_retrieval_rows))
    total_seconds = sum(row.latency_ns for row in rows) / 1_000_000_000
    return {
        "total": len(rows),
        "confusion": {
            "true_positive_reuse": true_positive,
            "false_positive_reuse": false_positive,
            "true_negative_miss": true_negative,
            "false_negative_miss": false_negative,
        },
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_reuse_rate": _divide(false_positive, len(rows)),
        "false_negative_reuse_rate": _divide(false_negative, len(expected_reuse)),
        "unsafe_reuse_rate": _divide(false_positive, len(predicted_reuse)),
        "response_reuse_precision": response_precision,
        "response_reuse_recall": response_recall,
        "response_reuse_f1": _divide(
            2 * response_precision * response_recall,
            response_precision + response_recall,
        ),
        "retrieval_reuse_precision": retrieval_precision,
        "retrieval_reuse_recall": retrieval_recall,
        "retrieval_reuse_f1": _divide(
            2 * retrieval_precision * retrieval_recall,
            retrieval_precision + retrieval_recall,
        ),
        "unsafe_response_reuse_rate": _divide(
            len(response_rows) - correct_response, len(response_rows)
        ),
        "response_reuse_count": len(response_rows),
        "retrieval_reuse_count": len(retrieval_rows),
        "miss_count": sum(row.predicted == "miss" for row in rows),
        "miss_rate": _divide(sum(row.predicted == "miss" for row in rows), len(rows)),
        "reuse_hit_rate": _divide(len(predicted_reuse), len(rows)),
        "latency_ms": {
            "p50": percentile(latencies_ms, 50),
            "p95": percentile(latencies_ms, 95),
            "p99": percentile(latencies_ms, 99),
            "mean": _divide(sum(latencies_ms), len(latencies_ms)),
            "minimum": min(latencies_ms, default=0.0),
            "maximum": max(latencies_ms, default=0.0),
        },
        "total_query_seconds": total_seconds,
        "throughput_queries_per_second": _divide(len(rows), total_seconds),
    }


def retrieval_ranking_metrics(
    ranked_groups: Sequence[Sequence[str]],
    relevant_groups: Sequence[str | None],
    k: int,
) -> dict[str, float]:
    recall_values: list[float] = []
    precision_values: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []
    for ranked, relevant in zip(ranked_groups, relevant_groups):
        if relevant is None:
            continue
        top = list(ranked[:k])
        hits = [index for index, group in enumerate(top) if group == relevant]
        recall_values.append(1.0 if hits else 0.0)
        precision_values.append(len(hits) / k)
        reciprocal_ranks.append(1.0 / (hits[0] + 1) if hits else 0.0)
        ndcg_values.append(1.0 / math.log2(hits[0] + 2) if hits else 0.0)
    count = len(recall_values)
    return {
        f"recall@{k}": _divide(sum(recall_values), count),
        f"precision@{k}": _divide(sum(precision_values), count),
        "mrr": _divide(sum(reciprocal_ranks), count),
        f"ndcg@{k}": _divide(sum(ndcg_values), count),
    }
