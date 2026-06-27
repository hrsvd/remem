import unittest
from remem.client import Client
from remem.models.execution_result import ExecutionResult
from remem.reuse.engine import ReuseDecision

class TestReusePlannerIntegration(unittest.TestCase):
    def setUp(self):
        self.client = Client()

    def test_cache_miss_and_response_reuse(self):
        # 1. First execution - Cache MISS
        def callback_one():
            return ExecutionResult(response="computed-1", references=["doc1.txt"])

        outcome1 = self.client.get_or_compute(
            query_embedding=[0.1, 0.9],
            compute_callback=callback_one
        )
        self.assertEqual(outcome1.decision, ReuseDecision.MISS)
        self.assertEqual(outcome1.result, "computed-1")
        self.assertEqual(self.client.stats["misses"], 1)

        # 2. Second execution - Cache HIT (Response Reused)
        def callback_two():
            return ExecutionResult(response="computed-2", references=["doc2.txt"])

        outcome2 = self.client.get_or_compute(
            query_embedding=[0.11, 0.89], # High similarity
            compute_callback=callback_two
        )
        self.assertEqual(outcome2.decision, ReuseDecision.RESPONSE_REUSED)
        self.assertEqual(outcome2.result, "computed-1") # Reused old payload
        self.assertEqual(self.client.stats["hits"], 1)

if __name__ == "__main__":
    unittest.main()