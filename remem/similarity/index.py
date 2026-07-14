"""Search-index implementations used by :mod:`remem.similarity`.

Indexes deliberately store only derived vector data.  Execution records remain
the source of truth in the configured storage backend, so an ANN index can be
rebuilt safely after a process restart.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Mapping, Optional
from uuid import UUID

import numpy as np

from remem.models.execution_record import ExecutionRecord
from remem.similarity.metrics import cosine_similarity


@dataclass(frozen=True)
class AnnConfig:
    """Configuration for the optional HNSW approximate-neighbor index.

    Higher ``ef_search`` values generally improve recall at the cost of query
    latency.  The defaults favor accurate semantic-cache decisions.
    """

    m: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    candidate_count: int = 50
    persistence_path: Optional[str | Path] = None

    def __post_init__(self) -> None:
        if self.m <= 0:
            raise ValueError("m must be a positive integer.")
        if self.ef_construction <= 0:
            raise ValueError("ef_construction must be a positive integer.")
        if self.ef_search <= 0:
            raise ValueError("ef_search must be a positive integer.")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be a positive integer.")
        if self.persistence_path is not None and not str(self.persistence_path).strip():
            raise ValueError("persistence_path must not be empty when provided.")


@dataclass(frozen=True)
class AnnIndexStats:
    """Read-only lifecycle telemetry for the derived ANN index."""

    record_count: int
    rebuild_count: int
    load_count: int
    persistence_enabled: bool


class AnnIndexStateError(RuntimeError):
    """Raised when ANN candidate identifiers cannot be resolved safely."""


class AnnMutationError(AnnIndexStateError):
    """Raised when storage and ANN mutation cannot complete atomically."""


class AnnPersistenceError(AnnIndexStateError):
    """Raised when persistent ANN state cannot be saved or validated."""


def rerank_candidates(
    query_embedding: Sequence[float],
    candidate_ids: Sequence[UUID],
    records_by_id: Mapping[UUID, ExecutionRecord],
    threshold: float,
    top_k: Optional[int],
) -> list[tuple[ExecutionRecord, float]]:
    """Deduplicate ANN candidates and rank them using exact cosine scores."""

    unique_ids = list(dict.fromkeys(candidate_ids))
    missing_ids = [
        record_id for record_id in unique_ids if record_id not in records_by_id
    ]
    if missing_ids:
        missing = ", ".join(str(record_id) for record_id in missing_ids)
        raise AnnIndexStateError(
            f"ANN candidates reference unavailable records: {missing}. "
            "Rebuild the ANN index from authoritative storage."
        )

    matches = [
        (
            records_by_id[record_id],
            cosine_similarity(query_embedding, records_by_id[record_id].embedding),
        )
        for record_id in unique_ids
    ]
    matches = [match for match in matches if match[1] >= threshold]
    matches.sort(key=lambda item: (-item[1], str(item[0].id)))
    return matches if top_k is None else matches[:top_k]


class ExactSimilarityIndex:
    """Dependency-free exhaustive cosine search preserving legacy behavior."""

    def search(
        self,
        query_embedding: Sequence[float],
        entries: Sequence[ExecutionRecord],
        threshold: float,
        top_k: Optional[int],
    ) -> list[tuple[ExecutionRecord, float]]:
        matches = [
            (entry, cosine_similarity(query_embedding, entry.embedding))
            for entry in entries
        ]
        matches = [match for match in matches if match[1] >= threshold]
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches if top_k is None else matches[:top_k]


class HnswSimilarityIndex:
    """Optional USearch/HNSW cosine index synchronized from storage records.

    Persistent native state is an optional derived cache. Storage remains
    authoritative, and stale or corrupt artifacts are rebuilt automatically.
    """

    def __init__(self, config: Optional[AnnConfig] = None) -> None:
        try:
            from usearch.index import Index
        except ImportError as exc:  # pragma: no cover - exercised without extra
            raise ImportError(
                "ANN search requires the optional 'usearch' dependency. "
                "Install it with: pip install remem-ai[ann]"
            ) from exc

        self._index_type = Index
        self.config = config or AnnConfig()
        self._index: Any = None
        self._records: list[ExecutionRecord] = []
        self._key_by_record_id: dict[UUID, int] = {}
        self._record_id_by_key: dict[int, UUID] = {}
        self._embedding_by_record_id: dict[UUID, tuple[float, ...]] = {}
        self._next_key = 0
        self._fingerprint: tuple[tuple[str, tuple[float, ...]], ...] = ()
        self._dimension: Optional[int] = None
        self._lock = RLock()
        self.rebuild_count = 0
        self.load_count = 0
        self.persistence_recovery_reason: Optional[str] = None

    @property
    def persistence_path(self) -> Optional[Path]:
        if self.config.persistence_path is None:
            return None
        return Path(self.config.persistence_path)

    @property
    def metadata_path(self) -> Optional[Path]:
        if self.persistence_path is None:
            return None
        return Path(f"{self.persistence_path}.meta.json")

    @property
    def stats(self) -> AnnIndexStats:
        return AnnIndexStats(
            record_count=len(self._records),
            rebuild_count=self.rebuild_count,
            load_count=self.load_count,
            persistence_enabled=self.persistence_path is not None,
        )

    def initialize(self, entries: Sequence[ExecutionRecord]) -> None:
        """Fast-load valid persistent state or rebuild it from storage."""

        with self._lock:
            if self.persistence_path is None:
                self._synchronize(entries)
                return
            try:
                self._load_persistent(entries)
                self.persistence_recovery_reason = None
            except Exception as exc:
                self.persistence_recovery_reason = f"{type(exc).__name__}: {exc}"
                self._synchronize(entries, force=True)
                self._persist()

    def search(
        self,
        query_embedding: Sequence[float],
        entries: Sequence[ExecutionRecord],
        threshold: float,
        top_k: Optional[int],
    ) -> list[tuple[ExecutionRecord, float]]:
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer when provided.")
        if not entries:
            return []

        self.rebuild(entries)
        candidate_ids = self.candidate_ids(query_embedding, top_k)
        records_by_id = {record.id: record for record in self._records}
        return rerank_candidates(
            query_embedding,
            candidate_ids,
            records_by_id,
            threshold,
            top_k,
        )

    def rebuild(self, entries: Sequence[ExecutionRecord]) -> None:
        """Synchronize the derived index from authoritative records."""

        with self._lock:
            if self._synchronize(entries):
                self._persist()

    def upsert(self, record: ExecutionRecord) -> str:
        """Insert or replace one record without rebuilding the full graph."""

        vector = self._validate_vector(record.embedding, f"embedding for {record.id}")
        embedding = tuple(float(value) for value in record.embedding)
        with self._lock:
            if self._dimension is not None and vector.size != self._dimension:
                raise ValueError(
                    f"Embedding dimension {vector.size} does not match index "
                    f"dimension {self._dimension}."
                )
            existing_key = self._key_by_record_id.get(record.id)
            if existing_key is not None:
                if self._embedding_by_record_id[record.id] == embedding:
                    self._replace_cached_record(record)
                    return "unchanged"
                self._index.remove(existing_key, compact=True)
                self._index.add(existing_key, vector)
                self._embedding_by_record_id[record.id] = embedding
                self._replace_cached_record(record)
                self._refresh_fingerprint()
                self._persist()
                return "updated"

            if self._index is None:
                self._dimension = int(vector.size)
                self._index = self._create_index(self._dimension)
            key = self._next_key
            self._next_key += 1
            self._index.add(key, vector)
            self._key_by_record_id[record.id] = key
            self._record_id_by_key[key] = record.id
            self._embedding_by_record_id[record.id] = embedding
            self._records.append(record)
            self._refresh_fingerprint()
            self._persist()
            return "inserted"

    def delete(self, record_id: UUID) -> bool:
        """Remove one record and compact native graph state."""

        with self._lock:
            key = self._key_by_record_id.get(record_id)
            if key is None:
                return False
            removed = bool(self._index.remove(key, compact=True))
            if not removed:
                raise AnnIndexStateError(
                    f"ANN index could not remove key {key} for record {record_id}."
                )
            del self._key_by_record_id[record_id]
            del self._record_id_by_key[key]
            del self._embedding_by_record_id[record_id]
            self._records = [
                record for record in self._records if record.id != record_id
            ]
            self._refresh_fingerprint()
            if not self._records:
                self._clear_state()
            self._persist()
            return True

    def clear(self) -> None:
        """Reset native index data, mappings, and consistency state."""

        with self._lock:
            self._clear_state()
            self._persist()

    def candidate_ids(
        self,
        query_embedding: Sequence[float],
        top_k: Optional[int],
        eligible_ids: Optional[set[UUID]] = None,
    ) -> list[UUID]:
        """Return approximate candidate IDs without resolving stored records."""

        with self._lock:
            if top_k is not None and top_k <= 0:
                raise ValueError("top_k must be a positive integer when provided.")
            if not self._records:
                return []
            if eligible_ids is not None and not eligible_ids:
                return []

            query = self._validate_vector(query_embedding, "query embedding")
            if query.size != self._dimension:
                raise ValueError(
                    f"Query embedding dimension {query.size} does not match index "
                    f"dimension {self._dimension}."
                )

            requested = self.config.candidate_count
            if top_k is not None:
                requested = max(requested, top_k)
            target = requested
            if eligible_ids is not None:
                target = min(target, len(eligible_ids))
            count = min(requested, len(self._records))
            while True:
                matches = self._index.search(query, count=count)
                candidate_ids: list[UUID] = []
                for label in matches.keys:
                    key = int(label)
                    record_id = self._record_id_by_key.get(key)
                    if record_id is None:
                        raise AnnIndexStateError(
                            f"ANN index returned unknown internal label {key}. "
                            "Rebuild the ANN index from authoritative storage."
                        )
                    if eligible_ids is None or record_id in eligible_ids:
                        candidate_ids.append(record_id)

                candidate_ids = list(dict.fromkeys(candidate_ids))
                if len(candidate_ids) >= target or count == len(self._records):
                    return candidate_ids
                count = min(len(self._records), max(count + 1, count * 2))

    def _synchronize(
        self, entries: Sequence[ExecutionRecord], *, force: bool = False
    ) -> bool:
        fingerprint = tuple(
            (str(entry.id), tuple(float(value) for value in entry.embedding))
            for entry in entries
        )
        if not force and fingerprint == self._fingerprint:
            return False

        if not entries:
            self._clear_state()
            return True

        vectors = [
            self._validate_vector(entry.embedding, f"embedding for {entry.id}")
            for entry in entries
        ]
        dimension = int(vectors[0].size)
        if any(vector.size != dimension for vector in vectors):
            raise ValueError(f"All index embeddings must have dimension {dimension}.")
        if len({entry.id for entry in entries}) != len(entries):
            raise ValueError("ANN index entries must have unique record IDs.")

        index = self._create_index(dimension)
        key_by_record_id = {}
        record_id_by_key = {}
        embedding_by_record_id = {}
        for key, (record, vector) in enumerate(zip(entries, vectors)):
            index.add(key, vector)
            key_by_record_id[record.id] = key
            record_id_by_key[key] = record.id
            embedding_by_record_id[record.id] = tuple(
                float(value) for value in record.embedding
            )

        self._index = index
        self._records = list(entries)
        self._key_by_record_id = key_by_record_id
        self._record_id_by_key = record_id_by_key
        self._embedding_by_record_id = embedding_by_record_id
        self._next_key = len(entries)
        self._fingerprint = fingerprint
        self._dimension = dimension
        self.rebuild_count += 1
        return True

    def _create_index(self, dimension: int):
        return self._index_type(
            ndim=dimension,
            metric="cos",
            dtype="f32",
            connectivity=self.config.m,
            expansion_add=self.config.ef_construction,
            expansion_search=self.config.ef_search,
        )

    def _replace_cached_record(self, record: ExecutionRecord) -> None:
        self._records = [
            record if cached.id == record.id else cached for cached in self._records
        ]

    def _refresh_fingerprint(self) -> None:
        self._fingerprint = tuple(
            (str(record.id), self._embedding_by_record_id[record.id])
            for record in self._records
        )

    def _clear_state(self) -> None:
        if self._index is not None:
            self._index.reset()
        self._index = None
        self._records = []
        self._key_by_record_id = {}
        self._record_id_by_key = {}
        self._embedding_by_record_id = {}
        self._next_key = 0
        self._fingerprint = ()
        self._dimension = None

    def _persist(self) -> None:
        path = self.persistence_path
        metadata_path = self.metadata_path
        if path is None or metadata_path is None:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        index_temp = Path(f"{path}.tmp")
        metadata_temp = Path(f"{metadata_path}.tmp")
        try:
            index_checksum = None
            if self._index is not None:
                self._index.save(str(index_temp))
                index_checksum = self._file_checksum(index_temp)

            metadata = {
                "format_version": 1,
                "engine": "usearch-hnsw-cosine-f32",
                "config": self._persistence_config(),
                "dimension": self._dimension,
                "next_key": self._next_key,
                "native_size": len(self._records),
                "storage_fingerprint": self._storage_fingerprint(self._records),
                "index_sha256": index_checksum,
                "records": [
                    {"id": str(record.id), "key": self._key_by_record_id[record.id]}
                    for record in self._records
                ],
            }
            with metadata_temp.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())

            if self._index is None:
                if path.exists():
                    path.unlink()
            else:
                os.replace(index_temp, path)
            os.replace(metadata_temp, metadata_path)
        except Exception as exc:
            for temporary_path in (index_temp, metadata_temp):
                if temporary_path.exists():
                    temporary_path.unlink()
            raise AnnPersistenceError(
                f"Failed to save persistent ANN index at '{path}': {exc}"
            ) from exc

    def _load_persistent(self, entries: Sequence[ExecutionRecord]) -> None:
        path = self.persistence_path
        metadata_path = self.metadata_path
        if path is None or metadata_path is None:
            raise AnnPersistenceError("ANN persistence is not configured.")
        if not metadata_path.is_file():
            raise AnnPersistenceError(f"Metadata file '{metadata_path}' is missing.")

        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise AnnPersistenceError(
                f"Metadata file '{metadata_path}' is unreadable: {exc}"
            ) from exc

        if metadata.get("format_version") != 1:
            raise AnnPersistenceError("Unsupported ANN persistence format version.")
        if metadata.get("engine") != "usearch-hnsw-cosine-f32":
            raise AnnPersistenceError("Persistent ANN engine identity is incompatible.")
        if metadata.get("config") != self._persistence_config():
            raise AnnPersistenceError("Persistent ANN configuration is incompatible.")
        if metadata.get("storage_fingerprint") != self._storage_fingerprint(entries):
            raise AnnPersistenceError(
                "Persistent ANN state is stale for current storage."
            )

        raw_records = metadata.get("records")
        if not isinstance(raw_records, list) or len(raw_records) != len(entries):
            raise AnnPersistenceError("Persistent ANN record mapping is invalid.")
        try:
            key_by_record_id = {
                UUID(item["id"]): int(item["key"]) for item in raw_records
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise AnnPersistenceError(
                "Persistent ANN record mapping is invalid."
            ) from exc
        if len(key_by_record_id) != len(entries):
            raise AnnPersistenceError("Persistent ANN record IDs are not unique.")
        keys = list(key_by_record_id.values())
        if any(key < 0 for key in keys) or len(set(keys)) != len(keys):
            raise AnnPersistenceError("Persistent ANN keys are invalid or duplicated.")
        if set(key_by_record_id) != {record.id for record in entries}:
            raise AnnPersistenceError("Persistent ANN record IDs do not match storage.")

        next_key = metadata.get("next_key")
        if not isinstance(next_key, int) or next_key < 0:
            raise AnnPersistenceError("Persistent ANN next-key state is invalid.")
        if keys and next_key <= max(keys):
            raise AnnPersistenceError(
                "Persistent ANN next key would reuse an active key."
            )

        if not entries:
            if metadata.get("dimension") is not None:
                raise AnnPersistenceError("Empty persistent ANN state has a dimension.")
            if metadata.get("native_size") != 0:
                raise AnnPersistenceError(
                    "Empty persistent ANN state has native entries."
                )
            if metadata.get("index_sha256") is not None:
                raise AnnPersistenceError(
                    "Empty persistent ANN state references an index."
                )
            self._clear_state()
            self._next_key = next_key
            self.load_count += 1
            return

        vectors = [
            self._validate_vector(record.embedding, f"embedding for {record.id}")
            for record in entries
        ]
        dimension = int(vectors[0].size)
        if any(vector.size != dimension for vector in vectors):
            raise AnnPersistenceError(
                f"Storage embeddings do not share dimension {dimension}."
            )
        if metadata.get("dimension") != dimension:
            raise AnnPersistenceError("Persistent ANN dimension is incompatible.")
        if metadata.get("native_size") != len(entries):
            raise AnnPersistenceError("Persistent ANN native size is invalid.")
        if not path.is_file():
            raise AnnPersistenceError(f"Native index file '{path}' is missing.")
        if metadata.get("index_sha256") != self._file_checksum(path):
            raise AnnPersistenceError("Persistent ANN index checksum does not match.")

        index = self._create_index(dimension)
        index.load(str(path))
        if int(index.size) != len(entries):
            raise AnnPersistenceError("Loaded ANN index size does not match storage.")

        self._index = index
        self._records = list(entries)
        self._key_by_record_id = key_by_record_id
        self._record_id_by_key = {
            key: record_id for record_id, key in key_by_record_id.items()
        }
        self._embedding_by_record_id = {
            record.id: tuple(float(value) for value in record.embedding)
            for record in entries
        }
        self._next_key = next_key
        self._refresh_fingerprint()
        self._dimension = dimension
        self.load_count += 1

    def _persistence_config(self) -> dict[str, int]:
        return {
            "m": self.config.m,
            "ef_construction": self.config.ef_construction,
            "ef_search": self.config.ef_search,
            "candidate_count": self.config.candidate_count,
        }

    @staticmethod
    def _storage_fingerprint(entries: Sequence[ExecutionRecord]) -> str:
        canonical = [
            [str(record.id), [float(value) for value in record.embedding]]
            for record in sorted(entries, key=lambda item: str(item.id))
        ]
        payload = json.dumps(canonical, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _file_checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _validate_vector(vector: Sequence[float], name: str) -> np.ndarray:
        if isinstance(vector, (str, bytes)):
            raise TypeError(f"{name} must be a sequence of finite numbers.")
        try:
            array = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a sequence of finite numbers.") from exc
        if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a non-empty sequence of finite numbers.")
        return array


class PartitionedHnswSimilarityIndex:
    """Namespace-partitioned HNSW indexes with conservative policy filtering."""

    def __init__(self, config: Optional[AnnConfig] = None) -> None:
        try:
            from usearch.index import Index  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised without extra
            raise ImportError(
                "ANN search requires the optional 'usearch' dependency. "
                "Install it with: pip install remem-ai[ann]"
            ) from exc
        self.config = config or AnnConfig()
        self._partitions: dict[str, HnswSimilarityIndex] = {}
        self._namespace_by_record_id: dict[UUID, str] = {}
        self._lock = RLock()

    @property
    def stats(self) -> AnnIndexStats:
        child_stats = [partition.stats for partition in self._partitions.values()]
        return AnnIndexStats(
            record_count=sum(stats.record_count for stats in child_stats),
            rebuild_count=sum(stats.rebuild_count for stats in child_stats),
            load_count=sum(stats.load_count for stats in child_stats),
            persistence_enabled=self.config.persistence_path is not None,
        )

    @property
    def persistence_recovery_reason(self) -> Optional[str]:
        reasons = [
            f"namespace {namespace!r}: {partition.persistence_recovery_reason}"
            for namespace, partition in self._partitions.items()
            if partition.persistence_recovery_reason is not None
        ]
        return "; ".join(reasons) or None

    @property
    def rebuild_count(self) -> int:
        return self.stats.rebuild_count

    @property
    def load_count(self) -> int:
        return self.stats.load_count

    @property
    def _index(self):
        return self._single_partition()._index

    @_index.setter
    def _index(self, value) -> None:
        self._single_partition()._index = value

    @property
    def _key_by_record_id(self) -> dict[UUID, int]:
        return {
            record_id: key
            for partition in self._partitions.values()
            for record_id, key in partition._key_by_record_id.items()
        }

    def initialize(self, entries: Sequence[ExecutionRecord]) -> None:
        """Load or rebuild each namespace partition from authoritative records."""

        with self._lock:
            self._partitions = {}
            self._namespace_by_record_id = {}
            groups = self._group_by_namespace(entries)
            if not groups and self.config.persistence_path is not None:
                groups[""] = []
            for namespace, records in groups.items():
                partition = self._new_partition(namespace)
                partition.initialize(records)
                self._partitions[namespace] = partition
                for record in records:
                    self._namespace_by_record_id[record.id] = namespace

    def search(
        self,
        query_embedding: Sequence[float],
        entries: Sequence[ExecutionRecord],
        threshold: float,
        top_k: Optional[int],
    ) -> list[tuple[ExecutionRecord, float]]:
        """Compatibility search across partitions with exact reranking."""

        self.rebuild(entries)
        candidate_ids = self.candidate_ids(query_embedding, top_k)
        records_by_id = {record.id: record for record in entries}
        return rerank_candidates(
            query_embedding,
            candidate_ids,
            records_by_id,
            threshold,
            top_k,
        )

    def rebuild(self, entries: Sequence[ExecutionRecord]) -> None:
        """Reconcile all partitions with authoritative storage."""

        with self._lock:
            groups = self._group_by_namespace(entries)
            namespaces = set(groups) | set(self._partitions)
            if not namespaces and self.config.persistence_path is not None:
                namespaces.add("")
            for namespace in namespaces:
                partition = self._partitions.get(namespace)
                if partition is None:
                    partition = self._new_partition(namespace)
                    self._partitions[namespace] = partition
                partition.rebuild(groups.get(namespace, []))
            self._namespace_by_record_id = {
                record.id: namespace
                for namespace, records in groups.items()
                for record in records
            }

    def upsert(self, record: ExecutionRecord) -> str:
        """Upsert a record, moving it atomically between namespace partitions."""

        with self._lock:
            namespace = record.context.namespace
            previous_namespace = self._namespace_by_record_id.get(record.id)
            if previous_namespace is not None and previous_namespace != namespace:
                previous = self._partitions[previous_namespace]
                previous.delete(record.id)
                result = self._partition(namespace).upsert(record)
                self._namespace_by_record_id[record.id] = namespace
                return "updated"

            result = self._partition(namespace).upsert(record)
            self._namespace_by_record_id[record.id] = namespace
            return result

    def delete(self, record_id: UUID) -> bool:
        """Delete a record from its namespace partition."""

        with self._lock:
            namespace = self._namespace_by_record_id.get(record_id)
            if namespace is None:
                return False
            removed = self._partitions[namespace].delete(record_id)
            if removed:
                del self._namespace_by_record_id[record_id]
            return removed

    def clear(self) -> None:
        """Clear all namespace partitions and their persistent state."""

        with self._lock:
            for partition in self._partitions.values():
                partition.clear()
            self._namespace_by_record_id = {}

    def candidate_ids(
        self,
        query_embedding: Sequence[float],
        top_k: Optional[int],
        *,
        namespace: Optional[str] = None,
        predicate: Optional[Callable[[ExecutionRecord], bool]] = None,
    ) -> list[UUID]:
        """Search selected namespaces and exclude incompatible records safely."""

        with self._lock:
            if namespace is None:
                partitions = list(self._partitions.values())
            else:
                partition = self._partitions.get(namespace)
                partitions = [] if partition is None else [partition]

            candidate_ids: list[UUID] = []
            for partition in partitions:
                eligible_ids = None
                if predicate is not None:
                    eligible_ids = {
                        record.id for record in partition._records if predicate(record)
                    }
                candidate_ids.extend(
                    partition.candidate_ids(
                        query_embedding,
                        top_k,
                        eligible_ids=eligible_ids,
                    )
                )
            return list(dict.fromkeys(candidate_ids))

    def _partition(self, namespace: str) -> HnswSimilarityIndex:
        partition = self._partitions.get(namespace)
        if partition is None:
            partition = self._new_partition(namespace)
            self._partitions[namespace] = partition
        return partition

    def _new_partition(self, namespace: str) -> HnswSimilarityIndex:
        persistence_path = self._partition_path(namespace)
        config = replace(self.config, persistence_path=persistence_path)
        return HnswSimilarityIndex(config)

    def _partition_path(self, namespace: str) -> Optional[Path]:
        if self.config.persistence_path is None:
            return None
        base = Path(self.config.persistence_path)
        if namespace == "":
            return base
        digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        return base.parent / f"{base.name}.partitions" / f"{digest}.usearch"

    def _single_partition(self) -> HnswSimilarityIndex:
        if len(self._partitions) != 1:
            raise AnnIndexStateError(
                "A single native index is unavailable for multiple namespaces."
            )
        return next(iter(self._partitions.values()))

    @staticmethod
    def _group_by_namespace(
        entries: Sequence[ExecutionRecord],
    ) -> dict[str, list[ExecutionRecord]]:
        groups: dict[str, list[ExecutionRecord]] = {}
        for record in entries:
            groups.setdefault(record.context.namespace, []).append(record)
        return groups
