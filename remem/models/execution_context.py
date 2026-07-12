from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ExecutionContext:
    """Encapsulates all contextual metadata for a single request execution."""

    namespace: str = ""
    kb_version: str = "1.0"
    prompt_version: str = "1.0"
    model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
