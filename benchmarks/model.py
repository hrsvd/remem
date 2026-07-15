from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ExpectedDecision(str, Enum):
    RESPONSE_REUSE = "response_reuse"
    RETRIEVAL_REUSE = "retrieval_reuse"
    MISS = "miss"


@dataclass(frozen=True)
class ContextSpec:
    namespace: str = "benchmark"
    kb_version: str = "1"
    prompt_version: str = "1"
    model: str | None = "benchmark-model"


@dataclass(frozen=True)
class SeedRecord:
    id: str
    text: str
    response: str
    references: list[str]
    response_group: str
    retrieval_group: str
    context: ContextSpec = field(default_factory=ContextSpec)


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    dataset: str
    split: str
    query: str
    expected: ExpectedDecision
    response_group: str | None
    retrieval_group: str | None
    context: ContextSpec = field(default_factory=ContextSpec)
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Workload:
    name: str
    dataset: str
    dataset_version: str
    license: str
    seed: int
    seeds: list[SeedRecord]
    cases: list[BenchmarkCase]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CostAssumptions:
    model_cost_per_call: float = 0.0
    retrieval_cost_per_call: float = 0.0
    input_tokens_per_call: int = 0
    output_tokens_per_call: int = 0


@dataclass(frozen=True)
class DecisionObservation:
    case_id: str
    expected: str
    predicted: str
    score: float
    matched_record_id: str | None
    matched_response_group: str | None
    matched_retrieval_group: str | None
    safe: bool
    latency_ns: int
    tags: list[str]
