import unittest
from uuid import uuid4
from remem.models.retrieval_entry import RetrievalEntry
from remem.storage.in_memory_storage import InMemoryStorage


class TestInMemoryStorage(unittest.TestCase):

    def setUp(self):
        self.storage = InMemoryStorage()

    def test_put_and_get(self):
        """Store and retrieve elements."""
        entry = RetrievalEntry(namespace="test_space")
        self.storage.put(entry)

        retrieved = self.storage.get(entry.id)
        self.assertEqual(retrieved, entry)
        self.assertEqual(retrieved.namespace, "test_space")

    def test_get_nonexistent(self):
        """Get missing entry IDs returns None."""
        non_existent_id = uuid4()
        result = self.storage.get(non_existent_id)
        self.assertIsNone(result)

    def test_delete(self):
        """Delete removes existing elements."""
        entry = RetrievalEntry()
        self.storage.put(entry)
        
        deleted = self.storage.delete(entry.id)
        self.assertTrue(deleted)
        self.assertIsNone(self.storage.get(entry.id))
        
        deleted_again = self.storage.delete(entry.id)
        self.assertFalse(deleted_again)

    def test_update(self):
        """Update mutations persist."""
        entry = RetrievalEntry(namespace="v1")
        self.storage.put(entry)

        entry.namespace = "v2"
        self.storage.update(entry)

        updated_entry = self.storage.get(entry.id)
        self.assertEqual(updated_entry.namespace, "v2")

    def test_all_iteration(self):
        """All items return within list iterables."""
        entry1 = RetrievalEntry()
        entry2 = RetrievalEntry()
        
        self.storage.put(entry1)
        self.storage.put(entry2)

        all_entries = self.storage.all()
        self.assertEqual(len(all_entries), 2)
        self.assertIn(entry1, all_entries)
        self.assertIn(entry2, all_entries)


if __name__ == "__main__":
    unittest.main()