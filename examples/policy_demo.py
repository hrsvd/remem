from remem import Client
from remem.models.execution_record import ExecutionRecord
from remem.models.execution_context import ExecutionContext
from remem.models.execution_result import ExecutionResult
from remem.reuse.policy import ReusePolicy


def main():
    print("🚀 Starting Remem Policy-Driven Demo v0.4.0...\n")

    # Initialize Client with strict knowledge base version checks
    policy = ReusePolicy(
        retrieval_threshold=0.80,
        response_threshold=0.95,
        require_same_namespace=True,
        require_same_kb_version=True,
        require_same_prompt_version=True,
    )
    client = Client(policy=policy)

    # Seed storage with a prior execution record
    print("Seeding storage with an existing execution record (Namespace: 'weather', KB: 'v1')...")
    past_record = ExecutionRecord(
        embedding=[0.1, 0.9],
        references=["weather_doc_v1.txt"],
        response="Precomputed Sunny Forecast.",
        context=ExecutionContext(namespace="weather", kb_version="v1", prompt_version="v1"),
    )
    client.store(past_record)

    def expensive_llm_callback() -> ExecutionResult:
        print("⚡ [Callback Executing]: Running heavy computation...")
        return ExecutionResult(
            response="Fresh dynamic computation.",
            references=["fresh_doc.txt"],
        )

    # --- SCENARIO 1: Full Response Reuse ---
    print("\n--- Scenario 1: Query matches perfectly (Same namespace, same versions) ---")
    outcome_reuse = client.get_or_compute(
        query_embedding=[0.11, 0.89],
        compute_callback=expensive_llm_callback,
        context=ExecutionContext(namespace="weather", kb_version="v1", prompt_version="v1"),
    )
    print(f"Decision: {outcome_reuse.decision.name}")
    print(f"Result Payload: {outcome_reuse.result}")

    # --- SCENARIO 2: Knowledge Base Version Mismatch (Cache Miss) ---
    print("\n--- Scenario 2: KB version mismatch (v1 cached vs v2 queried) ---")
    outcome_mismatch = client.get_or_compute(
        query_embedding=[0.11, 0.89],
        compute_callback=expensive_llm_callback,
        context=ExecutionContext(namespace="weather", kb_version="v2", prompt_version="v1"),
    )
    print(f"Decision: {outcome_mismatch.decision.name}")
    print(f"Result Payload (Newly computed): {outcome_mismatch.result}")

    print("\n📊 Final Engine Statistics:")
    print(client.stats)


if __name__ == "__main__":
    main()