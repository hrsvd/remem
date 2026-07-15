from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

from benchmarks.model import (
    BenchmarkCase,
    ContextSpec,
    ExpectedDecision,
    SeedRecord,
    Workload,
)

T = TypeVar("T")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(destination)


def load_workload(path: str | Path) -> Workload:
    raw = read_json(path)
    seeds = [
        SeedRecord(
            **{key: value for key, value in item.items() if key != "context"},
            context=ContextSpec(**item.get("context", {})),
        )
        for item in raw["seeds"]
    ]
    cases = [
        BenchmarkCase(
            **{
                key: value
                for key, value in item.items()
                if key not in {"context", "expected"}
            },
            expected=ExpectedDecision(item["expected"]),
            context=ContextSpec(**item.get("context", {})),
        )
        for item in raw["cases"]
    ]
    allowed = {item.name for item in fields(Workload)} - {"seeds", "cases"}
    return Workload(
        **{key: value for key, value in raw.items() if key in allowed},
        seeds=seeds,
        cases=cases,
    )
