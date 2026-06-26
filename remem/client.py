from collections.abc import Sequence
from typing import List, Optional
from uuid import UUID

from remem.models.retrieval_entry import RetrievalEntry
from remem.similarity.engine import SimilarityEngine
from remem.storage.in_memory_storage import InMemoryStorage
from remem.storage.storage import StorageInterface


class Client:
    """Public entry point into the Remem engine orchestrating storage

    and dependency wiring.
    """

    def __init__(self, storage_backend: Optional[StorageInterface] = None):
        self.storage: StorageInterface = storage_backend or InMemoryStorage()
        self.similarity = SimilarityEngine()
        self._hits = 0
        self._misses = 0

    def store(
        self,
        embedding: Sequence[float],
        references: List[str],
        namespace: str = "",
    ) -> None:
        """Saves a reusable execution entry."""
        entry = RetrievalEntry(
            embedding=embedding, references=references, namespace=namespace
        )
        self.storage.put(entry)

    def lookup(
        self, query_embedding: Sequence[float], threshold: float = 0.0
    ) -> Optional[RetrievalEntry]:
        """Performs a semantic search lookup over stored entries tracking hits/misses."""
        entries = self.storage.all()
        best_match = self.similarity.find_best_match(
            query_embedding, entries, threshold=threshold
        )

        if best_match:
            self._hits += 1
            return best_match

        self._misses += 1
        return None

    def delete(self, entry_id: UUID) -> bool:
        """Removes an entry by ID from underlying storage."""
        return self.storage.delete(entry_id)

    def all(self) -> List[RetrievalEntry]:
        """Returns all stored entries (primarily for early debugging)."""
        return self.storage.all()

    @property
    def stats(self) -> dict:
        """Exposes observability metrics."""
        total_lookups = self._hits + self._misses
        hit_rate = (
            float(self._hits / total_lookups) if total_lookups > 0 else 0.0
        )

        return {
            "entries": len(self.storage.all()),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
        }