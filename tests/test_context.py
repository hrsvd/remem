import unittest
from remem.models.execution_context import ExecutionContext
from remem.models.execution_record import ExecutionRecord


class TestExecutionContextModel(unittest.TestCase):

    def test_execution_context_initialization(self):
        context = ExecutionContext(
            namespace="agent",
            kb_version="v3",
            prompt_version="v2",
            model="text-embedding-3",
        )
        self.assertEqual(context.namespace, "agent")
        self.assertEqual(context.kb_version, "v3")

    def test_execution_record_contains_context(self):
        context = ExecutionContext(namespace="test")
        record = ExecutionRecord(
            embedding=[0.1, 0.2], references=[], context=context
        )
        self.assertEqual(record.context.namespace, "test")


if __name__ == "__main__":
    unittest.main()