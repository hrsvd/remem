from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from remem.models.execution_context import ExecutionContext


@dataclass
class ExecutionRecord:
    """Represents a rich execution record capable of intelligent work reuse."""

    embedding: list[float]
    references: list[str]
    context: ExecutionContext = field(default_factory=ExecutionContext)
    id: UUID = field(default_factory=uuid4)
    response: Optional[Any] = None
    hit_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)