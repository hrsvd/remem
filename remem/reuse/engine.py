from typing import Callable
from uuid import uuid4

from remem.metrics.collector import MetricsCollector
from remem.metrics.events import MetricEvent
from remem.models.execution_context import ExecutionContext
from remem.models.execution_record import ExecutionRecord
from remem.models.execution_result import ExecutionResult
from remem.reuse.decision import ReuseDecision, ReuseOutcome
from remem.reuse.matcher import MetadataMatcher
from remem.reuse.policy import ReusePolicy
from remem.similarity.engine import SimilarityEngine
from remem.storage.storage import StorageInterface


class ReuseEngine:
    """Core engine evaluating reusability boundaries and emitting metric events."""

    def __init__(
        self,
        storage: StorageInterface,
        similarity: SimilarityEngine,
        policy: ReusePolicy,
        metrics: MetricsCollector,
    ):
        self.storage = storage
        self.similarity = similarity
        self.policy = policy
        self.metrics = metrics

    def _find_best_compatible(
        self,
        query_embedding: list[float],
        context: ExecutionContext,
        threshold: float,
    ):
        """Shared metadata-filter + similarity scan used by both public methods."""
        all_entries = self.storage.all()
        compatible = MetadataMatcher.filter_candidates(
            all_entries, context, self.policy
        )
        return self.similarity.find_best_match(
            query_embedding, compatible, threshold=threshold
        )

    def check(
        self,
        query_embedding: list[float],
        context: ExecutionContext,
    ) -> ReuseDecision:
        """Returns the reuse decision and any cached artifacts without running any callback.

        The caller inspects the decision and routes accordingly:

        * ``RESPONSE_REUSED`` – ``outcome.result`` holds the cached LLM response.
          No pipeline work needed.
        * ``RETRIEVAL_REUSED`` – ``outcome.references`` holds the cached documents.
          Skip the vector-DB search; pass those docs to your LLM, then call
          ``remember()`` to store the result.
        * ``MISS`` – no usable previous work.  Run the full pipeline and call
          ``remember()`` to store the result.
        """
        self.metrics.record(MetricEvent.REQUEST)
        best_match = self._find_best_compatible(
            query_embedding, context, self.policy.retrieval_threshold
        )

        if not best_match:
            self.metrics.record(MetricEvent.MISS)
            return ReuseOutcome(
                result=None,
                decision=ReuseDecision.MISS,
                similarity_score=0.0,
                reason="No compatible execution found.",
            )

        matched = best_match.entry
        self.storage.increment_hit(matched.id)
        score = best_match.score

        if score >= self.policy.response_threshold and matched.response is not None:
            self.metrics.record(MetricEvent.HIT, similarity=score)
            self.metrics.record(MetricEvent.RESPONSE_REUSED)
            return ReuseOutcome(
                result=matched.response,
                decision=ReuseDecision.RESPONSE_REUSED,
                similarity_score=score,
                reason=f"Vector similarity {score:.2f} met response threshold.",
                matched_record_id=matched.id,
                references=matched.references,
            )

        self.metrics.record(MetricEvent.HIT, similarity=score)
        self.metrics.record(MetricEvent.RETRIEVAL_REUSED)
        return ReuseOutcome(
            result=None,
            decision=ReuseDecision.RETRIEVAL_REUSED,
            similarity_score=score,
            reason=f"Vector similarity {score:.2f} met retrieval threshold but fell below response threshold.",
            matched_record_id=matched.id,
            references=matched.references,
        )

    def get_or_compute(
        self,
        query_embedding: list[float],
        compute_callback: Callable[[], ExecutionResult],
        context: ExecutionContext,
    ) -> ReuseOutcome:
        """All-in-one reuse planner: runs ``compute_callback`` only when necessary.

        For the RETRIEVAL_REUSED branch the callback is still invoked (because
        it owns the full pipeline), but ``outcome.references`` carries the
        cached documents so callers that inspect the decision can skip their
        own vector-DB search.  Prefer ``check()`` + ``remember()`` when you
        want explicit control over each pipeline stage.
        """
        self.metrics.record(MetricEvent.REQUEST)
        best_match = self._find_best_compatible(
            query_embedding, context, self.policy.retrieval_threshold
        )

        # MISS: run full pipeline and store the result
        if not best_match:
            self.metrics.record(MetricEvent.MISS)
            exec_result = compute_callback()
            self.storage.put(
                ExecutionRecord(
                    id=uuid4(),
                    embedding=query_embedding,
                    references=exec_result.references,
                    response=exec_result.response,
                    context=context,
                )
            )
            return ReuseOutcome(
                result=exec_result.response,
                decision=ReuseDecision.MISS,
                similarity_score=0.0,
                reason="No compatible execution found.",
                references=exec_result.references,
            )

        matched_entry = best_match.entry
        self.storage.increment_hit(matched_entry.id)
        score = best_match.score

        # Full hit: return cached LLM response, skip pipeline entirely
        if (
            score >= self.policy.response_threshold
            and matched_entry.response is not None
        ):
            self.metrics.record(MetricEvent.HIT, similarity=score)
            self.metrics.record(MetricEvent.RESPONSE_REUSED)
            return ReuseOutcome(
                result=matched_entry.response,
                decision=ReuseDecision.RESPONSE_REUSED,
                similarity_score=score,
                reason=f"Vector similarity {score:.2f} met response threshold.",
                matched_record_id=matched_entry.id,
                references=matched_entry.references,
            )

        # Partial hit: retrieval can be reused, re-run computation, store new result
        self.metrics.record(MetricEvent.HIT, similarity=score)
        self.metrics.record(MetricEvent.RETRIEVAL_REUSED)
        computed_exec = compute_callback()
        self.storage.put(
            ExecutionRecord(
                id=uuid4(),
                embedding=query_embedding,
                references=computed_exec.references,
                response=computed_exec.response,
                context=context,
            )
        )
        return ReuseOutcome(
            result=computed_exec.response,
            decision=ReuseDecision.RETRIEVAL_REUSED,
            similarity_score=score,
            reason=f"Vector similarity {score:.2f} met retrieval threshold but fell below response threshold.",
            matched_record_id=matched_entry.id,
            references=matched_entry.references,
        )
