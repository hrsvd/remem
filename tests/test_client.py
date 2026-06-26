import unittest
from remem.client import Client


class TestRemehClientIntegration(unittest.TestCase):

    def setUp(self):
        """Initialize the public facade client for testing."""
        self.client = Client()

    def test_store_and_lookup_integration(self):
        """Verify storing and looking up data via the Client works end-to-end."""
        # 1. Store an entry using the public API
        self.client.store(
            embedding=[1.0, 0.0],
            references=["doc_test.txt"],
            namespace="test_space"
        )

        # 2. Verify it shows up in stats
        self.assertEqual(self.client.stats["entries"], 1)

        # 3. Lookup the entry via the Client facade
        query_vector = [0.9, 0.1]
        match = self.client.lookup(query_vector, threshold=0.5)

        # 4. Validate output
        self.assertIsNotNone(match)
        self.assertEqual(match.references, ["doc_test.txt"])
        self.assertEqual(match.namespace, "test_space")
        self.assertEqual(self.client.stats["hits"], 1)
        self.assertEqual(self.client.stats["misses"], 0)

    def test_lookup_misses_threshold(self):
        """Verify that low-similarity lookups are ignored and register as a miss."""
        self.client.store(
            embedding=[1.0, 0.0],
            references=["doc_exact.txt"]
        )

        # Query completely orthogonal vector with an overly strict threshold
        query_vector = [0.0, 1.0]
        match = self.client.lookup(query_vector, threshold=0.99)

        self.assertIsNone(match)
        # Update assertions to reflect isolated client state (0 hits, 1 miss)
        self.assertEqual(self.client.stats["hits"], 0)
        self.assertEqual(self.client.stats["misses"], 1)

    def test_delete_entry(self):
        """Verify entry deletion reduces stored inventory accurately."""
        self.client.store(embedding=[0.5, 0.5], references=["deletable.txt"])
        self.assertEqual(self.client.stats["entries"], 1)

        # Grab entry ID to delete
        all_entries = self.client.all()
        entry_id = all_entries[0].id

        deletion_success = self.client.delete(entry_id)
        self.assertTrue(deletion_success)
        self.assertEqual(self.client.stats["entries"], 0)


if __name__ == "__main__":
    unittest.main()