from datetime import datetime
import unittest
from uuid import uuid4
from remem.models.retrieval_entry import RetrievalEntry


class TestRetrievalEntry(unittest.TestCase):

    def test_create_and_verify_object(self):
        """Test simple dataclass instantiation and slot attributes."""
        entry_id = uuid4()
        created_time = datetime.utcnow()

        entry = RetrievalEntry(
            id=entry_id,
            embedding=[0.1, 0.2, 0.3],
            references=["doc1.txt"],
            namespace="default",
            kb_version="1.0",
            created_at=created_time,
            hit_count=0,
        )

        self.assertEqual(entry.id, entry_id)
        self.assertEqual(list(entry.embedding), [0.1, 0.2, 0.3])
        self.assertEqual(entry.references, ["doc1.txt"])
        self.assertEqual(entry.namespace, "default")
        self.assertEqual(entry.kb_version, "1.0")
        self.assertEqual(entry.created_at, created_time)
        self.assertEqual(entry.hit_count, 0)


if __name__ == "__main__":
    unittest.main()