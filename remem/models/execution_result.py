from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ExecutionResult:
    """Encapsulates flexible execution payloads returned by computation callbacks."""

    response: Any  # Supports strings, DataFrames, dicts, byte arrays, etc.
    references: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)