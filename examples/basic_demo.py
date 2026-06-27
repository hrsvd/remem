from remem import Client
from remem.models.execution_record import ExecutionRecord
from remem.models.execution_result import ExecutionResult


def main():
    print("🚀 Starting Remem Reuse Planner Demo v0.3.0...\n")

    client = Client()

    # 1. Seed the system with a past execution record (Simulating a prior run)
    print("Seeding storage with an existing execution record...")
    past_record = ExecutionRecord(
        embedding=[0.1, 0.9],
        references=["weather_doc_v1.txt"],
        response="Sunny and 72 degrees in the valley today.",
        namespace="weather",
    )
    client.store(past_record)
    print("Seeded successfully.\n")

    # Define an expensive callback that simulates generating new work
    def expensive_llm_callback() -> ExecutionResult:
        print("⚡ [Callback Executing]: Running heavy computation/LLM query...")
        return ExecutionResult(
            response="Dynamic computation result.",
            references=["fresh_doc.txt"],
        )

    # --- SCENARIO A: Cache HIT (Response Reused) ---
    print("--- Scenario A: Query is highly similar (>= 0.95) ---")
    # Query vector is extremely close to [0.1, 0.9]
    outcome_response = client.get_or_compute(
        query_embedding=[0.11, 0.89],
        compute_callback=expensive_llm_callback,
        similarity_threshold=0.85,
        response_reuse_threshold=0.95,
    )
    print(f"Decision: {outcome_response.decision.name}")
    print(f"Payload Result: {outcome_response.result}\n")

    # --- SCENARIO B: Partial HIT (Retrieval Reused, Computation runs) ---
    print("--- Scenario B: Query is moderately similar (>= 0.85 but < 0.95) ---")
    # Query vector is reasonably close, but not perfectly identical
    outcome_retrieval = client.get_or_compute(
        query_embedding=[-0.2, 0.8],
        compute_callback=expensive_llm_callback,
        similarity_threshold=0.85,
        response_reuse_threshold=0.95,
    )
    print(f"Decision: {outcome_retrieval.decision.name}")
    print(f"Payload Result (Recomputed): {outcome_retrieval.result}\n")

    # --- SCENARIO C: Cache MISS (Completely new work) ---
    print("--- Scenario C: Query is dissimilar (< 0.85) ---")
    # Query vector points in a completely different direction
    outcome_miss = client.get_or_compute(
        query_embedding=[0.9, 0.1],
        compute_callback=expensive_llm_callback,
        similarity_threshold=0.85,
        response_reuse_threshold=0.95,
    )
    print(f"Decision: {outcome_miss.decision.name}")
    print(f"Payload Result (Newly computed): {outcome_miss.result}\n")

    # Observability Statistics
    print("📊 Final Engine Statistics:")
    print(client.stats)


if __name__ == "__main__":
    main()