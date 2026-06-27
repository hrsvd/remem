from typing import Callable, Optional
from uuid import UUID

from remem.models.execution_record import ExecutionRecord
from remem.models.execution_context import ExecutionContext
from remem.models.execution_result import ExecutionResult
from remem.reuse.policy import ReusePolicy
from remem.reuse.engine import ReuseDecision, ReuseEngine, ReuseOutcome
from remem.similarity.engine import SimilarityEngine
from remem.storage.in_memory_storage import InMemoryStorage
from remem.storage.storage import StorageInterface


class Client:
    """Public entry point into the Remem engine orchestrating clean functional boundaries."""

    def __init__(
        self,
        storage_backend: Optional[StorageInterface] = None,
        policy: Optional[ReusePolicy] = None,
    ):
        self.storage: StorageInterface = storage_backend or InMemoryStorage()
        self.similarity = SimilarityEngine()
        self.policy = policy or ReusePolicy()
        self.reuse_planner = ReuseEngine(
            self.storage, self.similarity, self.policy
        )
        self._misses = 0

    def store(self, record: ExecutionRecord) -> None:
        """Stores a rich execution record directly."""
        self.storage.put(record)

    def get_or_compute(
        self,
        query_embedding: list[float],
        compute_callback: Callable[[], ExecutionResult],
        context: Optional[ExecutionContext] = None,
    ) -> ReuseOutcome:
        """Flagship reuse planner endpoint accepting structured ExecutionContext."""
        exec_context = context or ExecutionContext()

        outcome = self.reuse_planner.get_or_compute(
            query_embedding=query_embedding,
            compute_callback=compute_callback,
            context=exec_context,
        )

        if outcome.decision is ReuseDecision.MISS:
            self._misses += 1

        return outcome

    def delete(self, entry_id: UUID) -> bool:
        return self.storage.delete(entry_id)

    def all(self) -> list[ExecutionRecord]:
        return self.storage.all()

    @property
    def stats(self) -> dict:
        entries_list = self.storage.all()
        total_hits = sum(e.hit_count for e in entries_list)

        return {
            "entries": len(entries_list),
            "hits": total_hits,
            "misses": self._misses,
        }