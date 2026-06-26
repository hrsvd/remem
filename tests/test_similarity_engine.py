import unittest
from remem.models.retrieval_entry import RetrievalEntry
from remem.similarity.engine import SimilarityEngine


class TestSimilarityEngine(unittest.TestCase):

    def setUp(self):
        self.engine = SimilarityEngine()

    def test_no_entries(self):
        """No entries returns None or empty arrays."""
        self.assertIsNone(self.engine.find_best_match([0.1, 0.2], []))
        self.assertEqual(self.engine.find_all_matches([0.1, 0.2], []), [])

    def test_one_entry(self):
        """Exact match behavior with 1 entry."""
        entry = RetrievalEntry(embedding=[1.0, 0.0])
        best = self.engine.find_best_match([1.0, 0.0], [entry])
        self.assertEqual(best, entry)

    def test_two_entries(self):
        """Query closer to entry 2 selects entry 2."""
        entry1 = RetrievalEntry(embedding=[0.5, 0.5])
        entry2 = RetrievalEntry(embedding=[0.9, 0.1])
        query = [0.8, 0.2]
        
        best = self.engine.find_best_match(query, [entry1, entry2])
        self.assertEqual(best, entry2)

    def test_100_entries(self):
        """Linear search reliability over scale arrays."""
        base_entries = [
            RetrievalEntry(embedding=[float(i % 10) / 10, float((i + 5) % 10) / 10]) 
            for i in range(100)
        ]
        
        target_entry = RetrievalEntry(embedding=[0.9, 0.9])
        base_entries.append(target_entry)
        
        best = self.engine.find_best_match([0.85, 0.85], base_entries)
        self.assertEqual(best, target_entry)

    def test_threshold_respected(self):
        """Entries below thresholds are ignored."""
        entry_low = RetrievalEntry(embedding=[0.0, 1.0])
        entry_high = RetrievalEntry(embedding=[1.0, 0.0])

        query = [1.0, 0.0]
        threshold = 0.8

        best = self.engine.find_best_match(query, [entry_low, entry_high], threshold=threshold)
        self.assertEqual(best, entry_high)

        all_matches = self.engine.find_all_matches(query, [entry_low, entry_high], threshold=threshold)
        self.assertEqual(len(all_matches), 1)
        self.assertEqual(all_matches[0][0], entry_high)


if __name__ == "__main__":
    unittest.main()