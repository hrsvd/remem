from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4


@dataclass(slots=True)
class RetrievalEntry:
    """Represents a reusable retrieval result.

    This class is intentionally a pure data model and should never
    contain business logic.
    """

    id: UUID = field(default_factory=uuid4)

    # Embedding produced by the application.
    embedding: Sequence[float] = field(default_factory=list)

    # References to retrieved artifacts (chunk IDs, document IDs, etc.)
    references: list[str] = field(default_factory=list)

    namespace: str = ""

    kb_version: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.utcnow)

    hit_count: int = 0