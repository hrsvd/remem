"""Search-index implementations used by :mod:`remem.similarity`.

Indexes deliberately store only derived vector data.  Execution records remain
the source of truth in the configured storage backend, so an ANN index can be
rebuilt safely after a process restart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

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

    def __post_init__(self) -> None:
        if self.m <= 0:
            raise ValueError("m must be a positive integer.")
        if self.ef_construction <= 0:
            raise ValueError("ef_construction must be a positive integer.")
        if self.ef_search <= 0:
            raise ValueError("ef_search must be a positive integer.")


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

    The index is intentionally rebuildable rather than persisted separately:
    JSON storage remains authoritative, avoiding stale index files and keeping
    index updates and deletes correct across existing storage backends.
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
        self._index = None
        self._records: list[ExecutionRecord] = []
        self._fingerprint: tuple[tuple[str, tuple[float, ...]], ...] = ()
        self._dimension: Optional[int] = None

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

        self._synchronize(entries)
        query = self._validate_vector(query_embedding, "query embedding")
        if query.size != self._dimension:
            raise ValueError(
                f"Query embedding dimension {query.size} does not match index "
                f"dimension {self._dimension}."
            )

        count = len(self._records) if top_k is None else min(top_k, len(self._records))
        matches = self._index.search(query, count=count)
        matches = [
            (self._records[int(label)], float(1.0 - distance))
            for label, distance in zip(matches.keys, matches.distances)
            if float(1.0 - distance) >= threshold
        ]
        return matches

    def _synchronize(self, entries: Sequence[ExecutionRecord]) -> None:
        fingerprint = tuple(
            (str(entry.id), tuple(float(value) for value in entry.embedding))
            for entry in entries
        )
        if fingerprint == self._fingerprint:
            return

        vectors = [
            self._validate_vector(entry.embedding, f"embedding for {entry.id}")
            for entry in entries
        ]
        dimension = int(vectors[0].size)
        if any(vector.size != dimension for vector in vectors):
            raise ValueError(f"All index embeddings must have dimension {dimension}.")
        if len({entry.id for entry in entries}) != len(entries):
            raise ValueError("ANN index entries must have unique record IDs.")

        index = self._index_type(
            ndim=dimension,
            metric="cos",
            dtype="f32",
            connectivity=self.config.m,
            expansion_add=self.config.ef_construction,
            expansion_search=self.config.ef_search,
        )
        for label, vector in enumerate(vectors):
            index.add(label, vector)

        self._index = index
        self._records = list(entries)
        self._fingerprint = fingerprint
        self._dimension = dimension

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
