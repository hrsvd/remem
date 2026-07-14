from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Literal, Optional
from uuid import UUID

from remem.models.execution_record import ExecutionRecord
from remem.similarity.index import (
    AnnConfig,
    AnnIndexStats,
    ExactSimilarityIndex,
    PartitionedHnswSimilarityIndex,
    rerank_candidates,
)


@dataclass
class SimilarityMatch:
    """Explicit response container holding both the found entry and distance calculations."""

    entry: ExecutionRecord
    score: float


class SimilarityEngine:
    """Finds semantically similar execution records using an exact or ANN index."""

    def __init__(
        self,
        backend: Literal["exact", "hnsw"] = "exact",
        ann_config: Optional[AnnConfig] = None,
    ) -> None:
        self._index: ExactSimilarityIndex | PartitionedHnswSimilarityIndex
        if backend == "exact":
            self._index = ExactSimilarityIndex()
        elif backend == "hnsw":
            self._index = PartitionedHnswSimilarityIndex(ann_config)
        else:
            raise ValueError("backend must be either 'exact' or 'hnsw'.")
        self.backend = backend

    def rebuild(self, entries: Sequence[ExecutionRecord]) -> None:
        """Rebuild derived ANN state; exact search has no derived state."""

        if isinstance(self._index, PartitionedHnswSimilarityIndex):
            self._index.rebuild(entries)

    def initialize(self, entries: Sequence[ExecutionRecord]) -> None:
        """Load or derive initial ANN state; exact search has no state."""

        if isinstance(self._index, PartitionedHnswSimilarityIndex):
            self._index.initialize(entries)

    @property
    def persistence_recovery_reason(self) -> Optional[str]:
        """Explain why persistent ANN state was rebuilt, when applicable."""

        if not isinstance(self._index, PartitionedHnswSimilarityIndex):
            return None
        return self._index.persistence_recovery_reason

    @property
    def ann_index_stats(self) -> Optional[AnnIndexStats]:
        """Return ANN lifecycle counters, or ``None`` for exact search."""

        if not isinstance(self._index, PartitionedHnswSimilarityIndex):
            return None
        return self._index.stats

    def upsert(self, record: ExecutionRecord) -> str:
        """Incrementally insert or replace one ANN record."""

        if not isinstance(self._index, PartitionedHnswSimilarityIndex):
            return "unchanged"
        return self._index.upsert(record)

    def delete(self, record_id: UUID) -> bool:
        """Incrementally remove one ANN record."""

        if not isinstance(self._index, PartitionedHnswSimilarityIndex):
            return False
        return self._index.delete(record_id)

    def clear(self) -> None:
        """Clear all derived ANN state."""

        if isinstance(self._index, PartitionedHnswSimilarityIndex):
            self._index.clear()

    def find_candidate_ids(
        self,
        query_embedding: Sequence[float],
        top_k: Optional[int] = None,
        *,
        namespace: Optional[str] = None,
        predicate: Optional[Callable[[ExecutionRecord], bool]] = None,
    ) -> list[UUID]:
        """Discover ANN candidate IDs without resolving storage records."""

        if not isinstance(self._index, PartitionedHnswSimilarityIndex):
            raise RuntimeError("Candidate ID discovery is available only for HNSW.")
        return self._index.candidate_ids(
            query_embedding,
            top_k,
            namespace=namespace,
            predicate=predicate,
        )

    @staticmethod
    def rerank_candidate_records(
        query_embedding: Sequence[float],
        candidate_ids: Sequence[UUID],
        records: Sequence[ExecutionRecord],
        threshold: float,
        top_k: Optional[int],
    ) -> list[tuple[ExecutionRecord, float]]:
        """Apply exact cosine reranking to directly resolved ANN records."""

        records_by_id = {record.id: record for record in records}
        return rerank_candidates(
            query_embedding,
            candidate_ids,
            records_by_id,
            threshold,
            top_k,
        )

    def find_best_match(
        self,
        query_embedding: Sequence[float],
        entries: Sequence[ExecutionRecord],
        threshold: float = 0.0,
    ) -> Optional[SimilarityMatch]:

        if not query_embedding or not entries:
            return None

        matches = self.find_all_matches(query_embedding, entries, threshold, top_k=1)
        if matches:
            entry, score = matches[0]
            return SimilarityMatch(entry=entry, score=score)
        return None

    def find_all_matches(
        self,
        query_embedding: Sequence[float],
        entries: Sequence[ExecutionRecord],
        threshold: float = 0.0,
        top_k: Optional[int] = None,
    ) -> list[tuple[ExecutionRecord, float]]:

        if not query_embedding or not entries:
            return []

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer when provided.")

        return self._index.search(query_embedding, entries, threshold, top_k)
