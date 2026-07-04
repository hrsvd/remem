"""Remem — an AI Work Reuse Engine.

Remember expensive AI work. Reuse it intelligently.

Quickstart::

    from remem import Client, ExecutionResult

    client = Client()

    outcome = client.get_or_compute(
        query_embedding=embed("What is our refund policy?"),
        compute_callback=lambda: ExecutionResult(
            response=run_expensive_pipeline(),
            references=["doc_42"],
        ),
    )

    print(outcome.decision, outcome.result)
"""

from remem.client import Client
from remem.models.execution_context import ExecutionContext
from remem.models.execution_record import ExecutionRecord
from remem.models.execution_result import ExecutionResult
from remem.reuse.decision import ReuseDecision, ReuseOutcome
from remem.reuse.policy import ReusePolicy
from remem.storage.json_storage import JsonStorage
from remem.storage.memory_storage import InMemoryStorage
from remem.storage.storage import StorageInterface

__version__ = "1.0.0b1"

__all__ = [
    # Core facade
    "Client",
    # Models
    "ExecutionContext",
    "ExecutionRecord",
    "ExecutionResult",
    # Reuse policy & outcomes
    "ReusePolicy",
    "ReuseOutcome",
    "ReuseDecision",
    # Storage backends
    "StorageInterface",
    "JsonStorage",
    "InMemoryStorage",
    # Metadata
    "__version__",
]
