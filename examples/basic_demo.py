from remem import Client


def main():
    print("🚀 Starting Remem Modern Engine Demo...\n")

    client = Client()

    # 1. Store API
    print("Storing entries via client.store(...) API...")
    client.store(
        embedding=[0.1, 0.9],
        references=["doc_weather.txt"],
        namespace="general",
    )
    client.store(
        embedding=[0.8, 0.2], references=["doc_coding.txt"], namespace="tech"
    )

    # 2. Lookup API
    query_vector = [0.75, 0.25]
    print(
        f"\nLooking up semantically similar entry for query: {query_vector}..."
    )

    match = client.lookup(query_vector, threshold=0.0)

    print("\n✨ Lookup Result:")
    if match:
        print(f"Found match reference: {match.references}")
        print(f"Match embedding: {match.embedding}")
        print(f"Namespace: {match.namespace}")
    else:
        print("No match found.")

    # 3. Observability Statistics
    print("\n📊 Engine Statistics:")
    print(client.stats)


if __name__ == "__main__":
    main()