from typing import List, Optional
from uuid import UUID

from remem.models.execution_record import ExecutionRecord
from remem.storage.storage import StorageInterface


class InMemoryStorage(StorageInterface):
    """Thread-safe transient memory mapper storage."""

    def __init__(self):
        self._data: dict[UUID, ExecutionRecord] = {}

    def put(self, entry: ExecutionRecord) -> None:
        self._data[entry.id] = entry

    def get(self, entry_id: UUID) -> Optional[ExecutionRecord]:
        return self._data.get(entry_id)

    def delete(self, entry_id: UUID) -> bool:
        if entry_id in self._data:
            del self._data[entry_id]
            return True
        return False

    def update(self, entry: ExecutionRecord) -> None:
        if entry.id in self._data:
            self._data[entry.id] = entry

    def increment_hit(self, entry_id: UUID) -> None:
        if entry_id in self._data:
            self._data[entry_id].hit_count += 1

    def all(self) -> List[ExecutionRecord]:
        return list(self._data.values())