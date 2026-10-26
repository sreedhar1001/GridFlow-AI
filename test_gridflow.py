"""
GridFlow AI: Automated Verification & Compliance Test Suite
Verifies 100% adherence to:
1. System Instructions & Orchestration Protocol (Mandatory JSON Schema, Routing, Loop Prevention)
2. State Management & Memory Architecture (memory.md: Tier 1 GridState, Tier 2 SQLite, Tier 3 ChromaDB)
3. Knowledge Base & RAG Specification (knowledge_base.md: ./chroma_db_store, cosine >= 0.72)
4. Agent Skill Registry & Tool Implementations (skill.md: RBAC, Pydantic validation, AST security, Sandbox)
"""

import os
import unittest
from state import merge_artifacts, GridState, EpisodicSQLiteCheckpointer
from knowledge_base import knowledge_collection
from tools import (
    chromadb_semantic_search,
    web_scraper_extracter,
    python_ast_validator,
    artifact_file_writer,
    invoke_tool_with_rbac,
)
from orchestrator import GridFlowOrchestrator


REQUIRED_TRANSITION_KEYS = {
    "current_node",
    "task_id",
    "status",
    "reasoning_trace",
    "payload",
    "next_node",
    "iteration_count",
}


class TestGridFlowArchitecture(unittest.TestCase):

    def test_01_chromadb_semantic_search_and_threshold(self) -> None:
        res = chromadb_semantic_search(
            query="GridFlow multi-agent cyclic topology Planner Researcher Executor Reviewer",
            top_k=4,
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreaterEqual(res["retrieved_count"], 1)
        for chunk in res["results"]:
            self.assertGreaterEqual(chunk["similarity_score"], 0.72)
            self.assertIn("doc_id", chunk)

        # Validation error on out-of-range top_k
        bad_res = chromadb_semantic_search(query="test", top_k=99)
        self.assertEqual(bad_res["status"], "VALIDATION_ERROR")

    def test_02_python_ast_validator_syntax_and_security(self) -> None:
        valid_res = python_ast_validator("def add(a: int, b: int) -> int:\n    return a + b\n")
        self.assertEqual(valid_res["status"], "VALID")
        self.assertTrue(valid_res["is_syntax_valid"])
        self.assertEqual(valid_res["security_warnings"], [])

        flagged_res = python_ast_validator("import subprocess\nx = eval('1+1')\n")
        self.assertEqual(flagged_res["status"], "SECURITY_FLAGGED")
        self.assertTrue(flagged_res["is_syntax_valid"])
        self.assertEqual(len(flagged_res["security_warnings"]), 2)

        syntax_res = python_ast_validator("def broken(:\n    pass")
        self.assertEqual(syntax_res["status"], "SYNTAX_ERROR")
        self.assertFalse(syntax_res["is_syntax_valid"])

    def test_03_artifact_file_writer_sandboxing(self) -> None:
        ok = artifact_file_writer("unit_test_artifact.py", "print('GridFlow Verified')\n")
        self.assertEqual(ok["status"], "SUCCESS")
        self.assertTrue(os.path.exists(ok["saved_path"]))

        traversal = artifact_file_writer("../hack.py", "print('bad')")
        self.assertEqual(traversal["status"], "ERROR")

    def test_04_rbac_enforcement(self) -> None:
        # Researcher_Node cannot invoke artifact_file_writer
        denied = invoke_tool_with_rbac(
            "Researcher_Node",
            "artifact_file_writer",
            {"filename": "illegal.py", "content": "x=1"},
        )
        self.assertEqual(denied["status"], "RBAC_VIOLATION")

        # Reviewer_Node can invoke python_ast_validator
        allowed = invoke_tool_with_rbac(
            "Reviewer_Node",
            "python_ast_validator",
            {"code_string": "x = 42\n"},
        )
        self.assertEqual(allowed["status"], "VALID")

    def test_05_standard_workflow_and_mandatory_schema(self) -> None:
        orch = GridFlowOrchestrator()
        result = orch.run_workflow(
            user_query="Build a thread-safe Token Bucket Rate Limiter API service in Python",
            mode="standard",
        )
        final_state = result["final_state"]
        transitions = result["transitions"]

        # Path: Planner_Node -> Researcher_Node -> Executor_Node -> Reviewer_Node -> END
        self.assertEqual(len(transitions), 4)
        for t in transitions:
            self.assertEqual(set(t.keys()), REQUIRED_TRANSITION_KEYS)
            self.assertIn("content", t["payload"])
            self.assertIn("citations", t["payload"])

        self.assertEqual(final_state["next_node"], "END")
        self.assertIsNotNone(final_state["final_response"])
        self.assertGreaterEqual(len(final_state["artifacts"]), 2)

        # Verify Tier 2 SQLite checkpoints match step count
        history = orch.checkpointer.get_thread_history(result["session_id"])
        self.assertEqual(len(history), 4)

    def test_06_self_healing_retry_loop(self) -> None:
        orch = GridFlowOrchestrator()
        result = orch.run_workflow(
            user_query="Build a smart grid substation telemetry anomaly detector in Python",
            mode="self_heal",
        )
        transitions = result["transitions"]
        # Planner -> Researcher -> Executor (flawed) -> Reviewer (NEEDS_REVISION) -> Executor (fixed) -> Reviewer (COMPLETED)
        self.assertEqual(len(transitions), 6)
        self.assertEqual(transitions[3]["status"], "NEEDS_REVISION")
        self.assertEqual(transitions[3]["next_node"], "Executor_Node")
        self.assertEqual(transitions[5]["status"], "COMPLETED")
        self.assertEqual(transitions[5]["next_node"], "END")

    def test_07_circuit_breaker_max_iterations_fallback(self) -> None:
        orch = GridFlowOrchestrator()
        result = orch.run_workflow(
            user_query="Stress test circuit breaker max_iterations=3 fallback routing",
            mode="circuit_breaker",
            max_iterations=3,
        )
        transitions = result["transitions"]
        # Planner(1) -> Researcher(2) -> Ex1(3) -> Rev1(4) -> Ex2(5) -> Rev2(6) -> Ex3(7) -> Rev3(8: FAILED->Fallback_Handler) -> Fallback(9: END)
        self.assertEqual(len(transitions), 9)
        self.assertEqual(transitions[7]["status"], "FAILED")
        self.assertEqual(transitions[7]["next_node"], "Fallback_Handler")
        self.assertEqual(transitions[7]["iteration_count"], 3)
        self.assertEqual(result["final_state"]["execution_logs"][-1]["node_name"], "Fallback_Handler")


if __name__ == "__main__":
    unittest.main(verbosity=2)
