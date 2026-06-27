import unittest
from remem.models.execution_context import ExecutionContext
from remem.reuse.policy import ReusePolicy


class TestReusePolicy(unittest.TestCase):

    def test_compatible_contexts(self):
        policy = ReusePolicy(
            require_same_namespace=True, require_same_kb_version=True
        )
        ctx_a = ExecutionContext(namespace="chat", kb_version="1.0")
        ctx_b = ExecutionContext(namespace="chat", kb_version="1.0")
        self.assertTrue(policy.is_compatible(ctx_a, ctx_b))

    def test_incompatible_namespaces(self):
        policy = ReusePolicy(require_same_namespace=True)
        ctx_a = ExecutionContext(namespace="chat")
        ctx_b = ExecutionContext(namespace="search")
        self.assertFalse(policy.is_compatible(ctx_a, ctx_b))


if __name__ == "__main__":
    unittest.main()