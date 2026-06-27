from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4


@dataclass
class ExecutionRecord:
    """Represents a rich execution record capable of intelligent work reuse."""

    embedding: list[float]
    references: list[str]
    id: UUID = field(default_factory=uuid4)
    response: Optional[Any] = None
    namespace: str = ""
    kb_version: str = "1.0"
    prompt_version: str = "1.0"
    hit_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)