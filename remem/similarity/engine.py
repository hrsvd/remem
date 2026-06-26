from collections.abc import Sequence
from typing import Optional

from remem.models.retrieval_entry import RetrievalEntry
from remem.similarity.metrics import cosine_similarity


class SimilarityEngine:
    """Finds semantically similar retrieval entries."""

    def find_best_match(
        self,
        query_embedding: Sequence[float],
        entries: Sequence[RetrievalEntry],
        threshold: float = 0.0,
    ) -> Optional[RetrievalEntry]:

        if not query_embedding or not entries:
            return None

        best_match = None
        highest_score = float("-inf")

        for entry in entries:
            score = cosine_similarity(query_embedding, entry.embedding)

            if score > highest_score and score >= threshold:
                highest_score = score
                best_match = entry

        return best_match

    def find_all_matches(
        self,
        query_embedding: Sequence[float],
        entries: Sequence[RetrievalEntry],
        threshold: float = 0.0,
    ) -> list[tuple[RetrievalEntry, float]]:

        if not query_embedding or not entries:
            return []

        matches = []

        for entry in entries:
            score = cosine_similarity(query_embedding, entry.embedding)

            if score >= threshold:
                matches.append((entry, score))

        matches.sort(key=lambda item: item[1], reverse=True)

        return matches