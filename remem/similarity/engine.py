from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from remem.models.execution_record import ExecutionRecord
from remem.similarity.metrics import cosine_similarity


@dataclass
class SimilarityMatch:
    """Explicit response container holding both the found entry and distance calculations."""

    entry: ExecutionRecord
    score: float


class SimilarityEngine:
    """Finds semantically similar execution records."""

    def find_best_match(
        self,
        query_embedding: Sequence[float],
        entries: Sequence[ExecutionRecord],
        threshold: float = 0.0,
    ) -> Optional[SimilarityMatch]:

        if not query_embedding or not entries:
            return None

        best_match = None
        highest_score = float("-inf")

        for entry in entries:
            score = cosine_similarity(query_embedding, entry.embedding)

            if score > highest_score and score >= threshold:
                highest_score = score
                best_match = entry

        if best_match:
            return SimilarityMatch(entry=best_match, score=highest_score)
        return None

    def find_all_matches(
        self,
        query_embedding: Sequence[float],
        entries: Sequence[ExecutionRecord],
        threshold: float = 0.0,
    ) -> list[tuple[ExecutionRecord, float]]:

        if not query_embedding or not entries:
            return []

        matches = []

        for entry in entries:
            score = cosine_similarity(query_embedding, entry.embedding)

            if score >= threshold:
                matches.append((entry, score))

        matches.sort(key=lambda item: item[1], reverse=True)

        return matches