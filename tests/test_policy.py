import unittest

from remem.models.execution_context import ExecutionContext
from remem.models.execution_record import ExecutionRecord
from remem.reuse.matcher import MetadataMatcher
from remem.reuse.policy import ReusePolicy


class TestReusePolicy(unittest.TestCase):
    def setUp(self):
        self.current = ExecutionContext(
            namespace="support",
            kb_version="2026-07",
            prompt_version="v1",
            model="gpt-4o",
        )

    def test_matching_contexts_are_compatible(self):
        cached = ExecutionContext(
            namespace="support",
            kb_version="2026-07",
            prompt_version="v1",
            model="gpt-4o",
        )

        self.assertTrue(ReusePolicy().is_compatible(self.current, cached))

    def test_namespace_mismatch_is_rejected_by_default(self):
        cached = ExecutionContext(
            namespace="sales",
            kb_version="2026-07",
            prompt_version="v1",
            model="gpt-4o",
        )

        self.assertFalse(ReusePolicy().is_compatible(self.current, cached))

    def test_kb_version_mismatch_is_rejected_by_default(self):
        cached = ExecutionContext(
            namespace="support",
            kb_version="2026-06",
            prompt_version="v1",
            model="gpt-4o",
        )

        self.assertFalse(ReusePolicy().is_compatible(self.current, cached))

    def test_prompt_version_mismatch_is_rejected_by_default(self):
        cached = ExecutionContext(
            namespace="support",
            kb_version="2026-07",
            prompt_version="v2",
            model="gpt-4o",
        )

        self.assertFalse(ReusePolicy().is_compatible(self.current, cached))

    def test_model_mismatch_is_rejected_by_default(self):
        cached = ExecutionContext(
            namespace="support",
            kb_version="2026-07",
            prompt_version="v1",
            model="gpt-4.1",
        )

        self.assertFalse(ReusePolicy().is_compatible(self.current, cached))

    def test_relaxed_policy_allows_selected_mismatch(self):
        cached = ExecutionContext(
            namespace="support",
            kb_version="2026-07",
            prompt_version="v2",
            model="gpt-4o",
        )
        policy = ReusePolicy(require_same_prompt_version=False)

        self.assertTrue(policy.is_compatible(self.current, cached))


class TestMetadataMatcher(unittest.TestCase):
    def test_filter_candidates_returns_only_policy_compatible_records(self):
        current = ExecutionContext(namespace="tenant-a", kb_version="v1")
        matching = ExecutionRecord(
            embedding=[1.0, 0.0],
            references=["doc_a"],
            context=ExecutionContext(namespace="tenant-a", kb_version="v1"),
        )
        rejected = ExecutionRecord(
            embedding=[0.0, 1.0],
            references=["doc_b"],
            context=ExecutionContext(namespace="tenant-b", kb_version="v1"),
        )

        candidates = MetadataMatcher.filter_candidates(
            entries=[matching, rejected],
            current_context=current,
            policy=ReusePolicy(),
        )

        self.assertEqual(candidates, [matching])


if __name__ == "__main__":
    unittest.main()
