"""
GridFlow AI: Interactive CLI Live Demo Runner
Executes all three hackathon demonstration scenarios in the terminal and outputs
verified JSON transition payloads, Tier-2 SQLite checkpoint counts, and generated files.
"""

import json
from orchestrator import GridFlowOrchestrator


def run_cli_demo() -> None:
    orch = GridFlowOrchestrator()

    scenarios = [
        (
            "SCENARIO 1: Standard Deterministic Execution (Zero-Error Pass)",
            "Build a thread-safe Token Bucket Rate Limiter API service in Python",
            "standard",
        ),
        (
            "SCENARIO 2: Self-Healing Retry Loop (Reviewer Rejects Draft 1 -> Executor Fixes Draft 2)",
            "Build a Smart Grid Substation Telemetry Anomaly Detector in Python",
            "self_heal",
        ),
        (
            "SCENARIO 3: Circuit Breaker Trip (max_iterations == 3 -> Fallback_Handler)",
            "Stress test Reviewer_Node Circuit Breaker max_iterations=3 and Fallback_Handler recovery",
            "circuit_breaker",
        ),
    ]

    for title, prompt, mode in scenarios:
        print("\n" + "=" * 78)
        print(f" {title}")
        print(f" Prompt : {prompt}")
        print(f" Mode   : {mode}")
        print("=" * 78)

        result = orch.run_workflow(user_query=prompt, mode=mode, max_iterations=3)
        for idx, t in enumerate(result["transitions"], start=1):
            print(
                f" [Step {idx}] {t['current_node']:16s} | status={t['status']:14s} "
                f"| next={t['next_node']:16s} | iter={t['iteration_count']}"
            )
            print(f"          Trace: {t['reasoning_trace']}")

        print("-" * 78)
        print(f" Session ID          : {result['session_id']}")
        print(f" SQLite Checkpoints  : {result['checkpoints_saved']} saved in ./workspace/checkpoints.sqlite")
        print(f" Generated Artifacts : {list(result['final_state']['artifacts'].keys())}")
        print("\n Sample Final Transition JSON Payload (Mandatory Output Schema):")
        print(json.dumps(result["transitions"][-1], indent=2))


if __name__ == "__main__":
    run_cli_demo()
