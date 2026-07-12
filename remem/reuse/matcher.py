from typing import List

from remem.models.execution_context import ExecutionContext
from remem.models.execution_record import ExecutionRecord
from remem.reuse.policy import ReusePolicy


class MetadataMatcher:
    """Filters stored execution candidates based on explicit ReusePolicy rules

    before performing vector distance checks.
    """

    @staticmethod
    def filter_candidates(
        entries: List[ExecutionRecord],
        current_context: ExecutionContext,
        policy: ReusePolicy,
    ) -> List[ExecutionRecord]:
        valid_candidates = []
        for entry in entries:
            if policy.is_compatible(current_context, entry.context):
                valid_candidates.append(entry)
        return valid_candidates
