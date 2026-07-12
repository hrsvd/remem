from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Optional

from remem.models.execution_record import ExecutionRecord
from remem.similarity.index import AnnConfig, ExactSimilarityIndex, HnswSimilarityIndex


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
        if backend == "exact":
            self._index = ExactSimilarityIndex()
        elif backend == "hnsw":
            self._index = HnswSimilarityIndex(ann_config)
        else:
            raise ValueError("backend must be either 'exact' or 'hnsw'.")
        self.backend = backend

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
