import unittest
from remem.models.execution_record import ExecutionRecord
from remem.models.execution_context import ExecutionContext
from remem.reuse.matcher import MetadataMatcher
from remem.reuse.policy import ReusePolicy


class TestMetadataMatcher(unittest.TestCase):

    def test_filter_matching_records(self):
        policy = ReusePolicy(require_same_namespace=True)
        record = ExecutionRecord(
            embedding=[1.0, 0.0],
            references=[],
            context=ExecutionContext(namespace="finance"),
        )
        candidates = [record]

        filtered = MetadataMatcher.filter_candidates(
            candidates, ExecutionContext(namespace="finance"), policy
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], record)

    def test_filter_mismatches(self):
        policy = ReusePolicy(require_same_namespace=True)
        record = ExecutionRecord(
            embedding=[1.0, 0.0],
            references=[],
            context=ExecutionContext(namespace="finance"),
        )
        candidates = [record]

        filtered = MetadataMatcher.filter_candidates(
            candidates, ExecutionContext(namespace="weather"), policy
        )
        self.assertEqual(len(filtered), 0)


if __name__ == "__main__":
    unittest.main()