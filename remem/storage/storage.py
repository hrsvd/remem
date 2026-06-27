from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from remem.models.execution_record import ExecutionRecord


class StorageInterface(ABC):
    """Abstract Base Class (Contract) for storage operations."""

    @abstractmethod
    def put(self, entry: ExecutionRecord) -> None:
        """Saves an ExecutionRecord into storage."""

    @abstractmethod
    def get(self, entry_id: UUID) -> Optional[ExecutionRecord]:
        """Retrieves an ExecutionRecord by its unique ID."""

    @abstractmethod
    def delete(self, entry_id: UUID) -> bool:
        """Deletes an ExecutionRecord by its unique ID."""

    @abstractmethod
    def update(self, entry: ExecutionRecord) -> None:
        """Updates an existing ExecutionRecord."""

    @abstractmethod
    def increment_hit(self, entry_id: UUID) -> None:
        """Atomically increments a record's hit counter."""

    @abstractmethod
    def all(self) -> List[ExecutionRecord]:
        """Returns all stored ExecutionRecords."""