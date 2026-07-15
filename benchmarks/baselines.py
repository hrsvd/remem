from __future__ import annotations

import re
from collections.abc import Iterable

from benchmarks.model import BenchmarkCase, DecisionObservation, SeedRecord


def normalize_key(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def no_reuse(cases: Iterable[BenchmarkCase]) -> list[DecisionObservation]:
    return [
        DecisionObservation(
            case_id=case.id,
            expected=case.expected.value,
            predicted="miss",
            score=0.0,
            matched_record_id=None,
            matched_response_group=None,
            matched_retrieval_group=None,
            safe=case.expected.value == "miss",
            latency_ns=0,
            tags=case.tags,
        )
        for case in cases
    ]


def exact_key(
    seeds: Iterable[SeedRecord], cases: Iterable[BenchmarkCase]
) -> list[DecisionObservation]:
    lookup = {
        (
            normalize_key(seed.text),
            seed.context.namespace,
            seed.context.kb_version,
            seed.context.prompt_version,
            seed.context.model,
        ): seed
        for seed in seeds
    }
    observations = []
    for case in cases:
        matched = lookup.get(
            (
                normalize_key(case.query),
                case.context.namespace,
                case.context.kb_version,
                case.context.prompt_version,
                case.context.model,
            )
        )
        predicted = "response_reuse" if matched else "miss"
        safe = (
            case.expected.value == "miss"
            if matched is None
            else case.response_group == matched.response_group
        )
        observations.append(
            DecisionObservation(
                case_id=case.id,
                expected=case.expected.value,
                predicted=predicted,
                score=1.0 if matched else 0.0,
                matched_record_id=matched.id if matched else None,
                matched_response_group=matched.response_group if matched else None,
                matched_retrieval_group=matched.retrieval_group if matched else None,
                safe=safe,
                latency_ns=0,
                tags=case.tags,
            )
        )
    return observations
