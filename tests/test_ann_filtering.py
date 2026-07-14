from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from remem import (
    AnnConfig,
    Client,
    ExecutionContext,
    InMemoryStorage,
    ReusePolicy,
)
from remem.models.execution_record import ExecutionRecord


class TrackingStorage(InMemoryStorage):
    def __init__(self) -> None:
        super().__init__()
        self.requested_ids = []

    def get_many(self, entry_ids):
        self.requested_ids.append(list(entry_ids))
        return super().get_many(entry_ids)


def record(
    vector: list[float],
    name: str,
    context: ExecutionContext,
    *,
    record_id=None,
) -> ExecutionRecord:
    return ExecutionRecord(
        id=record_id or uuid4(),
        embedding=vector,
        references=[],
        response=name,
        context=context,
    )


def context(
    namespace: str = "tenant-a",
    *,
    kb_version: str = "kb-current",
    prompt_version: str = "prompt-current",
    model: str | None = "model-current",
    metadata=None,
) -> ExecutionContext:
    return ExecutionContext(
        namespace=namespace,
        kb_version=kb_version,
        prompt_version=prompt_version,
        model=model,
        metadata=metadata or {},
    )


def client_with(records, *, candidate_count: int = 1, policy=None) -> Client:
    storage = InMemoryStorage()
    for stored in records:
        storage.put(stored)
    return Client(
        storage_backend=storage,
        search_mode="hnsw_cosine",
        ann_config=AnnConfig(candidate_count=candidate_count),
        policy=policy,
    )


def test_namespace_partition_prevents_global_candidate_crowding() -> None:
    pytest.importorskip("usearch.index")
    accepted = record([0.85, 0.15], "accepted", context("tenant-a"))
    incompatible = [
        record([1.0, float(index) / 10000], f"other-{index}", context("tenant-b"))
        for index in range(40)
    ]
    client = client_with([*incompatible, accepted], candidate_count=1)
    other_partition = client.similarity._index._partitions["tenant-b"]

    def fail_if_searched(*args, **kwargs):
        raise AssertionError("unselected namespace partition was searched")

    other_partition.candidate_ids = fail_if_searched
    outcome = client.check([1.0, 0.0], context("tenant-a"))

    assert outcome.matched_record_id == accepted.id
    assert outcome.result == "accepted"


def test_filtering_fetches_only_eligible_ids_without_query_storage_scan() -> None:
    pytest.importorskip("usearch.index")
    storage = TrackingStorage()
    accepted = record([0.8, 0.2], "accepted", context())
    storage.put(accepted)
    for index in range(16):
        storage.put(
            record(
                [1.0, float(index) / 100000],
                f"incompatible-{index}",
                context(kb_version="kb-old"),
            )
        )
    client = Client(
        storage_backend=storage,
        search_mode="hnsw_cosine",
        ann_config=AnnConfig(candidate_count=1),
    )

    def fail_all():
        raise AssertionError("query path called storage.all()")

    storage.all = fail_all
    outcome = client.check([1.0, 0.0], context())

    assert outcome.matched_record_id == accepted.id
    assert storage.requested_ids == [[accepted.id]]


@pytest.mark.parametrize(
    "mismatch",
    [
        {"kb_version": "kb-old"},
        {"prompt_version": "prompt-old"},
        {"model": "model-old"},
    ],
)
def test_structured_filter_expands_until_compatible_candidate_is_found(
    mismatch: dict[str, str],
) -> None:
    pytest.importorskip("usearch.index")
    current = context()
    accepted = record([0.8, 0.2], "accepted", current)
    incompatible_context = context(**mismatch)
    incompatible = [
        record(
            [1.0, float(index) / 100000],
            f"incompatible-{index}",
            incompatible_context,
        )
        for index in range(24)
    ]
    client = client_with([*incompatible, accepted], candidate_count=1)
    partition = client.similarity._index._partitions[current.namespace]
    real_search = partition._index.search
    requested_counts = []

    def track_search(query, count):
        requested_counts.append(count)
        return real_search(query, count=count)

    partition._index.search = track_search
    outcome = client.check([1.0, 0.0], current)

    assert outcome.matched_record_id == accepted.id
    assert requested_counts[0] == 1
    assert requested_counts[-1] == len(incompatible) + 1


def test_filter_aware_hnsw_matches_exact_mode_on_mixed_contexts() -> None:
    pytest.importorskip("usearch.index")
    current = context()
    accepted = record([0.8, 0.2], "accepted", current)
    records = [
        record([1.0, 0.0], "wrong-namespace", context("tenant-b")),
        record([0.99, 0.01], "wrong-kb", context(kb_version="kb-old")),
        record(
            [0.98, 0.02],
            "wrong-prompt",
            context(prompt_version="prompt-old"),
        ),
        record([0.97, 0.03], "wrong-model", context(model="model-old")),
        accepted,
    ]
    storage = InMemoryStorage()
    for stored in records:
        storage.put(stored)
    exact = Client(storage_backend=storage, search_mode="exact_cosine")
    ann = Client(
        storage_backend=storage,
        search_mode="hnsw_cosine",
        ann_config=AnnConfig(candidate_count=1),
    )

    exact_outcome = exact.check([1.0, 0.0], current)
    ann_outcome = ann.check([1.0, 0.0], current)

    assert ann_outcome.decision == exact_outcome.decision
    assert (
        ann_outcome.matched_record_id == exact_outcome.matched_record_id == accepted.id
    )
    assert ann_outcome.similarity_score == pytest.approx(exact_outcome.similarity_score)


def test_relaxed_policy_searches_across_namespace_partitions() -> None:
    pytest.importorskip("usearch.index")
    local = record([0.8, 0.2], "local", context("tenant-a"))
    global_best = record([1.0, 0.0], "global", context("tenant-b"))
    policy = ReusePolicy(
        require_same_namespace=False,
        require_same_kb_version=False,
        require_same_prompt_version=False,
        require_same_model=False,
    )
    client = client_with([local, global_best], candidate_count=1, policy=policy)

    outcome = client.check([1.0, 0.0], context("tenant-a"))

    assert outcome.matched_record_id == global_best.id


def test_namespace_change_moves_record_between_partitions_incrementally() -> None:
    pytest.importorskip("usearch.index")
    stored = record([1.0, 0.0], "before", context("tenant-a"))
    client = client_with([stored])
    moved = record(
        [1.0, 0.0],
        "after",
        context("tenant-b"),
        record_id=stored.id,
    )

    client.store(moved)

    assert client.check([1.0, 0.0], context("tenant-a")).matched_record_id is None
    assert client.check([1.0, 0.0], context("tenant-b")).result == "after"
    assert (
        stored.id
        not in client.similarity._index._partitions["tenant-a"]._key_by_record_id
    )
    assert (
        stored.id in client.similarity._index._partitions["tenant-b"]._key_by_record_id
    )


def test_partitioned_persistence_fast_loads_each_namespace(tmp_path: Path) -> None:
    pytest.importorskip("usearch.index")
    storage = InMemoryStorage()
    first = record([1.0, 0.0], "first", context("tenant-a"))
    second = record([0.0, 1.0], "second", context("tenant-b"))
    storage.put(first)
    storage.put(second)
    index_path = tmp_path / "records.usearch"
    config = AnnConfig(persistence_path=index_path)
    Client(
        storage_backend=storage,
        search_mode="hnsw_cosine",
        ann_config=config,
    )

    reloaded = Client(
        storage_backend=storage,
        search_mode="hnsw_cosine",
        ann_config=config,
    )

    assert reloaded.ann_index_stats.load_count == 2
    assert reloaded.ann_index_stats.rebuild_count == 0
    assert reloaded.ann_persistence_recovery_reason is None
    partition_dir = tmp_path / "records.usearch.partitions"
    assert len(list(partition_dir.glob("*.usearch"))) == 2


def test_one_corrupt_namespace_partition_does_not_rebuild_others(
    tmp_path: Path,
) -> None:
    pytest.importorskip("usearch.index")
    storage = InMemoryStorage()
    storage.put(record([1.0, 0.0], "first", context("tenant-a")))
    storage.put(record([0.0, 1.0], "second", context("tenant-b")))
    index_path = tmp_path / "records.usearch"
    config = AnnConfig(persistence_path=index_path)
    Client(
        storage_backend=storage,
        search_mode="hnsw_cosine",
        ann_config=config,
    )
    partition_files = list((tmp_path / "records.usearch.partitions").glob("*.usearch"))
    with partition_files[0].open("ab") as handle:
        handle.write(b"corrupt")

    recovered = Client(
        storage_backend=storage,
        search_mode="hnsw_cosine",
        ann_config=config,
    )

    assert recovered.ann_index_stats.load_count == 1
    assert recovered.ann_index_stats.rebuild_count == 1
    assert "checksum" in recovered.ann_persistence_recovery_reason


def test_arbitrary_context_metadata_is_not_an_implicit_filter() -> None:
    pytest.importorskip("usearch.index")
    cached = record(
        [1.0, 0.0],
        "cached",
        context(metadata={"region": "us"}),
    )
    client = client_with([cached])

    outcome = client.check(
        [1.0, 0.0],
        context(metadata={"region": "eu"}),
    )

    assert outcome.matched_record_id == cached.id
