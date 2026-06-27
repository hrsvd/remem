import unittest
from remem.models.execution_record import ExecutionRecord
from remem.models.execution_result import ExecutionResult

class TestExecutionModels(unittest.TestCase):
    def test_execution_record_init(self):
        record = ExecutionRecord(
            embedding=[0.1, 0.2],
            references=["doc.txt"],
            response="cached data",
            namespace="test"
        )
        self.assertEqual(record.embedding, [0.1, 0.2])
        self.assertEqual(record.response, "cached data")
        self.assertEqual(record.hit_count, 0)

    def test_execution_result_init(self):
        result = ExecutionResult(
            response="computed data",
            references=["source.pdf"],
            metadata={"model": "gpt-4"}
        )
        self.assertEqual(result.response, "computed data")
        self.assertEqual(result.metadata["model"], "gpt-4")

if __name__ == "__main__":
    unittest.main()