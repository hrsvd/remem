from typing import List, Optional
from uuid import UUID
from remem.models.retrieval_entry import RetrievalEntry
from remem.storage.storage import StorageInterface


class InMemoryStorage(StorageInterface):
    """Simplest storage implementation using an in-memory Python dictionary.

    Keys are entry IDs (UUID), values are RetrievalEntry objects.
    """

    def __init__(self):
        self._store: dict[UUID, RetrievalEntry] = {}

    def put(self, entry: RetrievalEntry) -> None:
        """Saves a RetrievalEntry into the dictionary."""
        self._store[entry.id] = entry

    def get(self, entry_id: UUID) -> Optional[RetrievalEntry]:
        """Retrieves a RetrievalEntry by its unique ID.

        Returns None if the ID does not exist.
        """
        return self._store.get(entry_id)

    def delete(self, entry_id: UUID) -> bool:
        """Deletes a RetrievalEntry by its unique ID.

        Returns True if successfully deleted, or False if the ID was not found.
        """
        if entry_id in self._store:
            del self._store[entry_id]
            return True
        return False

    def update(self, entry: RetrievalEntry) -> None:
        """Updates an existing RetrievalEntry.

        Overwrites the current value matching the entry ID.
        """
        self._store[entry.id] = entry

    def all(self) -> List[RetrievalEntry]:
        """Returns an iteration of all stored RetrievalEntries."""
        return list(self._store.values())