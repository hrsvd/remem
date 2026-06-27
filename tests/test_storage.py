import unittest
from uuid import uuid4
from remem.models.execution_record import ExecutionRecord
from remem.storage.in_memory_storage import InMemoryStorage

class TestInMemoryStorage(unittest.TestCase):
    def setUp(self):
        self.storage = InMemoryStorage()

    def test_put_and_increment_hit(self):
        record_id = uuid4()
        record = ExecutionRecord(id=record_id, embedding=[1.0], references=[])
        self.storage.put(record)
        
        self.assertEqual(self.storage.get(record_id).hit_count, 0)
        self.storage.increment_hit(record_id)
        self.assertEqual(self.storage.get(record_id).hit_count, 1)

    def test_delete_record(self):
        record_id = uuid4()
        record = ExecutionRecord(id=record_id, embedding=[1.0], references=[])
        self.storage.put(record)
        self.assertTrue(self.storage.delete(record_id))
        self.assertIsNone(self.storage.get(record_id))

if __name__ == "__main__":
    unittest.main()