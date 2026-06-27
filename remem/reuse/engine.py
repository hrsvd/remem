from enum import Enum
from typing import Any, Callable, Optional
from uuid import uuid4

from remem.models.execution_record import ExecutionRecord
from remem.models.execution_result import ExecutionResult
from remem.models.execution_context import ExecutionContext
from remem.reuse.matcher import MetadataMatcher
from remem.reuse.policy import ReusePolicy
from remem.similarity.engine import SimilarityEngine
from remem.storage.storage import StorageInterface


class ReuseDecision(Enum):
    RESPONSE_REUSED = "RESPONSE_REUSED"
    RETRIEVAL_REUSED = "RETRIEVAL_USED"
    COMPUTED = "COMPUTED"
    MISS = "MISS"


class ReuseOutcome:
    """Encapsulates structural reuse planner decisions."""

    def __init__(
        self,
        result: Any,
        decision: ReuseDecision,
        similarity_score: float,
        references: Optional[list[str]] = None,
    ):
        self.result = result
        self.decision = decision
        self.similarity_score = similarity_score
        self.references = references or []


class ReuseEngine:
    """Core engine orchestrating decision boundaries around work reuse."""

    def __init__(
        self,
        storage: StorageInterface,
        similarity: SimilarityEngine,
        policy: ReusePolicy,
    ):
        self.storage = storage
        self.similarity = similarity
        self.policy = policy

    def get_or_compute(
        self,
        query_embedding: list[float],
        compute_callback: Callable[[], ExecutionResult],
        context: ExecutionContext,
    ) -> ReuseOutcome:

        # 1. Apply metadata compatibility filtering before vector similarity scan
        all_entries = self.storage.all()
        compatible_candidates = MetadataMatcher.filter_candidates(
            all_entries, context, self.policy
        )

        # 2. Query vector scan abstracted purely over compatible records
        best_match = self.similarity.find_best_match(
            query_embedding, compatible_candidates, threshold=self.policy.retrieval_threshold
        )

        # Cache MISS: Execute, persist record with execution context, and return structural payload
        if not best_match:
            exec_result = compute_callback()
            new_record = ExecutionRecord(
                id=uuid4(),
                embedding=query_embedding,
                references=exec_result.references,
                response=exec_result.response,
                context=context,
            )
            self.storage.put(new_record)
            
            return ReuseOutcome(
                result=exec_result.response,
                decision=ReuseDecision.MISS,
                similarity_score=0.0,
                references=exec_result.references,
            )

        matched_entry = best_match.entry
        self.storage.increment_hit(matched_entry.id)
        score = best_match.score

        # Cache HIT: High Confidence -> Reuse computed response payload directly
        if score >= self.policy.response_threshold and matched_entry.response is not None:
            return ReuseOutcome(
                result=matched_entry.response,
                decision=ReuseDecision.RESPONSE_REUSED,
                similarity_score=score,
                references=matched_entry.references,
            )

        # Partial Hit (Retrieve Reuse): Re-run computation using found references
        computed_exec = compute_callback()

        return ReuseOutcome(
            result=computed_exec.response,
            decision=ReuseDecision.RETRIEVAL_REUSED,
            similarity_score=score,
            references=matched_entry.references,
        )