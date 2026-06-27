import unittest
from remem.models.execution_record import ExecutionRecord
from remem.similarity.engine import SimilarityEngine

class TestSimilarityEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SimilarityEngine()

    def test_find_best_match(self):
        entry1 = ExecutionRecord(embedding=[1.0, 0.0], references=[])
        entry2 = ExecutionRecord(embedding=[0.0, 1.0], references=[])
        
        match = self.engine.find_best_match([0.9, 0.1], [entry1, entry2])
        self.assertIsNotNone(match)
        self.assertEqual(match.entry, entry1)
        self.assertTrue(match.score > 0.8)

if __name__ == "__main__":
    unittest.main()