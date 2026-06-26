from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID
from remem.models.retrieval_entry import RetrievalEntry


class StorageInterface(ABC):
    """Abstract Base Class (Contract) for storage operations.

    Defines the interface for storing, retrieving, updating, and deleting
    RetrievalEntries. Follows the Dependency Inversion Principle.
    """

    @abstractmethod
    def put(self, entry: RetrievalEntry) -> None:
        """Saves a RetrievalEntry into storage."""

    @abstractmethod
    def get(self, entry_id: UUID) -> Optional[RetrievalEntry]:
        """Retrieves a RetrievalEntry by its unique ID."""

    @abstractmethod
    def delete(self, entry_id: UUID) -> bool:
        """Deletes a RetrievalEntry by its unique ID.

        Returns True if deleted, False if not found.
        """

    @abstractmethod
    def update(self, entry: RetrievalEntry) -> None:
        """Updates an existing RetrievalEntry."""

    @abstractmethod
    def all(self) -> List[RetrievalEntry]:
        """Returns all stored RetrievalEntries."""