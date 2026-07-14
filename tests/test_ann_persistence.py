from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from remem import AnnConfig, Client, ExecutionContext, InMemoryStorage, JsonStorage
from remem.models.execution_record import ExecutionRecord
from remem.similarity.index import AnnMutationError


def record(
    vector: list[float],
    name: str,
    *,
    record_id=None,
) -> ExecutionRecord:
    return ExecutionRecord(
        id=record_id or uuid4(),
        embedding=vector,
        references=[],
        response=name,
        context=ExecutionContext(),
    )


def persistent_client(storage, path: Path, **config) -> Client:
    return Client(
        storage_backend=storage,
        search_mode="hnsw_cosine",
        ann_config=AnnConfig(persistence_path=path, **config),
    )


def test_valid_persistent_index_fast_loads_without_rebuild(tmp_path: Path) -> None:
    pytest.importorskip("usearch.index")
    storage = InMemoryStorage()
    stored = record([1.0, 0.0], "persisted")
    storage.put(stored)
    index_path = tmp_path / "ann" / "records.usearch"

    initial = persistent_client(storage, index_path)
    assert initial.similarity._index.rebuild_count == 1
    assert index_path.is_file()
    assert Path(f"{index_path}.meta.json").is_file()

    reloaded = persistent_client(storage, index_path)

    assert reloaded.ann_index_stats.load_count == 1
    assert reloaded.ann_index_stats.rebuild_count == 0
    assert reloaded.ann_persistence_recovery_reason is None
    assert reloaded.check([1.0, 0.0]).matched_record_id == stored.id


def test_json_storage_restart_uses_persisted_native_index(tmp_path: Path) -> None:
    pytest.importorskip("usearch.index")
    storage_path = tmp_path / "records.json"
    index_path = tmp_path / "records.usearch"
    stored = record([1.0, 0.0], "durable")
    initial = persistent_client(JsonStorage(str(storage_path)), index_path)
    initial.store(stored)

    reloaded = persistent_client(JsonStorage(str(storage_path)), index_path)

    assert reloaded.similarity._index.load_count == 1
    assert reloaded.similarity._index.rebuild_count == 0
    assert reloaded.check([1.0, 0.0]).result == "durable"


def test_incremental_mutations_are_available_after_fast_reload(tmp_path: Path) -> None:
    pytest.importorskip("usearch.index")
    storage = InMemoryStorage()
    index_path = tmp_path / "records.usearch"
    stored = record([1.0, 0.0], "first")
    client = persistent_client(storage, index_path)
    client.store(stored)

    replacement = record([0.0, 1.0], "updated", record_id=stored.id)
    client.store(replacement)
    reloaded = persistent_client(storage, index_path)
    assert reloaded.similarity._index.rebuild_count == 0
    assert reloaded.check([0.0, 1.0]).result == "updated"

    assert reloaded.delete(stored.id)
    empty = persistent_client(storage, index_path)
    assert empty.similarity._index.load_count == 1
    assert empty.similarity._index.rebuild_count == 0
    assert empty.check([0.0, 1.0]).matched_record_id is None


def test_stale_storage_identity_rebuilds_and_rewrites_cache(tmp_path: Path) -> None:
    pytest.importorskip("usearch.index")
    storage = InMemoryStorage()
    index_path = tmp_path / "records.usearch"
    first = record([1.0, 0.0], "first")
    storage.put(first)
    persistent_client(storage, index_path)

    second = record([0.0, 1.0], "second")
    storage.put(second)
    recovered = persistent_client(storage, index_path)

    assert recovered.similarity._index.rebuild_count == 1
    assert "stale" in recovered.ann_persistence_recovery_reason
    assert recovered.check([0.0, 1.0]).matched_record_id == second.id
    fast = persistent_client(storage, index_path)
    assert fast.similarity._index.load_count == 1
    assert fast.similarity._index.rebuild_count == 0


@pytest.mark.parametrize("corrupt_target", ["metadata", "index"])
def test_corrupt_artifact_recovers_by_rebuilding(
    tmp_path: Path, corrupt_target: str
) -> None:
    pytest.importorskip("usearch.index")
    storage = InMemoryStorage()
    stored = record([1.0, 0.0], "safe")
    storage.put(stored)
    index_path = tmp_path / "records.usearch"
    persistent_client(storage, index_path)

    if corrupt_target == "metadata":
        Path(f"{index_path}.meta.json").write_text("{bad", encoding="utf-8")
    else:
        with index_path.open("ab") as handle:
            handle.write(b"corrupt")

    recovered = persistent_client(storage, index_path)

    assert recovered.similarity._index.rebuild_count == 1
    assert recovered.ann_persistence_recovery_reason is not None
    assert recovered.check([1.0, 0.0]).matched_record_id == stored.id


def test_incompatible_configuration_rebuilds_cache(tmp_path: Path) -> None:
    pytest.importorskip("usearch.index")
    storage = InMemoryStorage()
    storage.put(record([1.0, 0.0], "stored"))
    index_path = tmp_path / "records.usearch"
    persistent_client(storage, index_path, m=8)

    recovered = persistent_client(storage, index_path, m=12)

    assert recovered.similarity._index.rebuild_count == 1
    assert "configuration" in recovered.ann_persistence_recovery_reason


@pytest.mark.parametrize(
    ("mutate", "expected_reason"),
    [
        (lambda metadata: metadata.update(dimension=99), "dimension"),
        (
            lambda metadata: metadata["records"][0].update(id=str(uuid4())),
            "record IDs",
        ),
    ],
)
def test_incompatible_dimension_or_id_mapping_rebuilds_cache(
    tmp_path: Path, mutate, expected_reason: str
) -> None:
    pytest.importorskip("usearch.index")
    storage = InMemoryStorage()
    storage.put(record([1.0, 0.0], "stored"))
    index_path = tmp_path / "records.usearch"
    persistent_client(storage, index_path)
    metadata_path = Path(f"{index_path}.meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutate(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    recovered = persistent_client(storage, index_path)

    assert recovered.ann_index_stats.rebuild_count == 1
    assert expected_reason in recovered.ann_persistence_recovery_reason


def test_interrupted_temp_files_do_not_replace_valid_generation(tmp_path: Path) -> None:
    pytest.importorskip("usearch.index")
    storage = InMemoryStorage()
    storage.put(record([1.0, 0.0], "stored"))
    index_path = tmp_path / "records.usearch"
    persistent_client(storage, index_path)
    Path(f"{index_path}.tmp").write_bytes(b"partial-index")
    Path(f"{index_path}.meta.json.tmp").write_text("{partial", encoding="utf-8")

    reloaded = persistent_client(storage, index_path)

    assert reloaded.similarity._index.load_count == 1
    assert reloaded.similarity._index.rebuild_count == 0


def test_persistence_write_failure_rolls_back_storage_and_index(
    tmp_path: Path, monkeypatch
) -> None:
    pytest.importorskip("usearch.index")
    storage = InMemoryStorage()
    index_path = tmp_path / "records.usearch"
    client = persistent_client(storage, index_path)
    stored = record([1.0, 0.0], "failing")
    real_replace = os.replace
    failed = False

    def fail_once(source, destination):
        nonlocal failed
        if not failed and Path(destination) == index_path:
            failed = True
            raise OSError("simulated atomic replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_once)
    with pytest.raises(AnnMutationError, match="rolled back"):
        client.store(stored)

    assert storage.get(stored.id) is None
    reloaded = persistent_client(storage, index_path)
    assert reloaded.similarity._index.load_count == 1
    assert reloaded.check([1.0, 0.0]).matched_record_id is None


def test_exact_mode_ignores_ann_persistence_configuration(tmp_path: Path) -> None:
    client = Client(
        storage_backend=InMemoryStorage(),
        search_mode="exact_cosine",
        ann_config=AnnConfig(persistence_path=tmp_path / "unused.usearch"),
    )

    client.store(record([1.0, 0.0], "exact"))

    assert not list(tmp_path.iterdir())
    assert client.ann_persistence_recovery_reason is None
    assert client.ann_index_stats is None


def test_metadata_contains_no_serialized_response_payload(tmp_path: Path) -> None:
    pytest.importorskip("usearch.index")
    storage = InMemoryStorage()
    storage.put(record([1.0, 0.0], "secret-response"))
    index_path = tmp_path / "records.usearch"
    persistent_client(storage, index_path)

    metadata = json.loads(Path(f"{index_path}.meta.json").read_text(encoding="utf-8"))

    assert "secret-response" not in json.dumps(metadata)
