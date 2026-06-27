import unittest
from remem.client import Client
from remem.models.execution_record import ExecutionRecord

class TestModernClient(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_store_and_all_records(self):
        """Verify storing a rich ExecutionRecord works and persists."""
        record = ExecutionRecord(
            embedding=[0.1, 0.9],
            references=["doc_weather.txt"],
            namespace="general"
        )
        self.client.store(record)
        
        all_records = self.client.all()
        self.assertEqual(len(all_records), 1)
        self.assertEqual(all_records[0].references, ["doc_weather.txt"])
        self.assertEqual(self.client.stats["entries"], 1)

    def test_delete_record(self):
        """Verify record deletion reduces inventory accurately."""
        record = ExecutionRecord(
            embedding=[0.5, 0.5],
            references=["deletable.txt"]
        )
        self.client.store(record)
        self.assertEqual(self.client.stats["entries"], 1)

        deletion_success = self.client.delete(record.id)
        self.assertTrue(deletion_success)
        self.assertEqual(self.client.stats["entries"], 0)

if __name__ == "__main__":
    unittest.main()