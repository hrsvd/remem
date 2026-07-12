import unittest
from uuid import uuid4

from remem.models.execution_record import ExecutionRecord
from remem.similarity.engine import SimilarityEngine
from remem.similarity.metrics import cosine_similarity


class TestSimilarityMetrics(unittest.TestCase):
    def test_cosine_similarity_identical_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 2.0], [1.0, 2.0]), 1.0)

    def test_cosine_similarity_orthogonal_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_cosine_similarity_zero_vector_returns_zero(self):
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)

    def test_cosine_similarity_rejects_empty_vectors(self):
        with self.assertRaises(ValueError):
            cosine_similarity([], [])

    def test_cosine_similarity_rejects_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1.0, 2.0], [1.0])


class TestSimilarityEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SimilarityEngine()
        self.low_match = ExecutionRecord(
            id=uuid4(),
            embedding=[0.0, 1.0],
            references=["doc_low"],
            response="low",
        )
        self.high_match = ExecutionRecord(
            id=uuid4(),
            embedding=[1.0, 0.0],
            references=["doc_high"],
            response="high",
        )

    def test_find_best_match_returns_highest_scoring_entry_above_threshold(self):
        match = self.engine.find_best_match(
            query_embedding=[1.0, 0.0],
            entries=[self.low_match, self.high_match],
            threshold=0.5,
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.entry.id, self.high_match.id)
        self.assertAlmostEqual(match.score, 1.0)

    def test_find_best_match_returns_none_when_no_entry_meets_threshold(self):
        match = self.engine.find_best_match(
            query_embedding=[1.0, 0.0],
            entries=[self.low_match],
            threshold=0.5,
        )

        self.assertIsNone(match)

    def test_find_best_match_returns_none_for_empty_inputs(self):
        self.assertIsNone(self.engine.find_best_match([], [self.high_match]))
        self.assertIsNone(self.engine.find_best_match([1.0, 0.0], []))

    def test_find_all_matches_returns_matches_sorted_by_score(self):
        matches = self.engine.find_all_matches(
            query_embedding=[1.0, 0.0],
            entries=[self.low_match, self.high_match],
            threshold=0.0,
        )

        self.assertEqual(
            [entry.id for entry, _ in matches], [self.high_match.id, self.low_match.id]
        )
        self.assertGreaterEqual(matches[0][1], matches[1][1])

    def test_find_all_matches_filters_by_threshold(self):
        matches = self.engine.find_all_matches(
            query_embedding=[1.0, 0.0],
            entries=[self.low_match, self.high_match],
            threshold=0.5,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0].id, self.high_match.id)


if __name__ == "__main__":
    unittest.main()
