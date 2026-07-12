from dataclasses import dataclass

from remem.models.execution_context import ExecutionContext


@dataclass
class ReusePolicy:
    """Configurable policy dictating strict or fuzzy constraints for AI work reuse."""

    retrieval_threshold: float = 0.80
    response_threshold: float = 0.95
    require_same_namespace: bool = True
    require_same_kb_version: bool = True
    require_same_prompt_version: bool = True
    require_same_model: bool = True

    def is_compatible(
        self, current: ExecutionContext, cached: ExecutionContext
    ) -> bool:
        """Evaluates strict metadata compatibility constraints before reuse."""
        if self.require_same_namespace and current.namespace != cached.namespace:
            return False
        if self.require_same_kb_version and current.kb_version != cached.kb_version:
            return False
        if (
            self.require_same_prompt_version
            and current.prompt_version != cached.prompt_version
        ):
            return False
        if self.require_same_model and current.model != cached.model:
            return False
        return True
