from remem.cache.remem import Remem
from remem.models.retrieval_entry import RetrievalEntry
from remem.storage.in_memory_storage import InMemoryStorage


def main():
    print("🚀 Starting Remem Modern Engine Demo...\n")

    # Initialize public interface client facade
    client = Remem()

    # 1. Create storage
    storage = InMemoryStorage()

    # 2. Insert entries using new Sequence typing models
    entry1 = RetrievalEntry(
        embedding=[0.1, 0.9],
        references=["doc_weather.txt"],
        namespace="general",
    )
    entry2 = RetrievalEntry(
        embedding=[0.8, 0.2], references=["doc_coding.txt"], namespace="tech"
    )
    entry3 = RetrievalEntry(
        embedding=[0.5, 0.5], references=["doc_random.txt"], namespace="misc"
    )

    storage.put(entry1)
    storage.put(entry2)
    storage.put(entry3)

    # 3. Run Similarity via the new Remem API entry point facade
    query_vector = [0.75, 0.25]
    print(f"Query vector: {query_vector}")

    best_match = client.similarity.find_best_match(
        query_vector, storage.all()
    )

    print("\n✨ Result:")
    if best_match:
        print(f"Found best match reference: {best_match.references}")
        print(f"Best match embedding: {best_match.embedding}")
        print(f"Namespace: {best_match.namespace}")
    else:
        print("No match found.")


if __name__ == "__main__":
    main()