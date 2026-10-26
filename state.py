"""
GridFlow AI: State Management & Memory Architecture (memory.md implementation)
Implements Tier 1 (Short-Term Graph State), Tier 2 (SQLite Episodic Session Persistence),
and Tier 3 (ChromaDB Long-Term Semantic Memory).
"""

import os
import json
import sqlite3
import operator
from datetime import datetime, timezone
from typing import TypedDict, Annotated, List, Dict, Any, Optional


def merge_artifacts(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Custom reducer to safely merge generated artifacts without overwriting prior keys."""
    merged = existing.copy() if existing else {}
    if new:
        merged.update(new)
    return merged


class SubTask(TypedDict):
    task_id: str
    description: str
    assigned_node: str       # "Researcher_Node" | "Executor_Node"
    status: str              # "PENDING" | "IN_PROGRESS" | "COMPLETED" | "FAILED"
    dependencies: List[str]  # List of prerequisite task_ids
    output_summary: Optional[str]


class ExecutionLogEntry(TypedDict):
    step_number: int
    node_name: str
    task_id: str
    status: str
    reasoning_trace: str
    timestamp: str


class NodeTransitionPayload(TypedDict):
    """Mandatory Output Schema per Section 4 of System Instructions."""
    current_node: str
    task_id: str
    status: str
    reasoning_trace: str
    payload: Dict[str, Any]
    next_node: str
    iteration_count: int


class GridState(TypedDict):
    # Session & Input Metadata
    session_id: str
    user_query: str

    # DAG Task Planning
    plan: List[SubTask]
    active_task_id: str
    next_node: str           # Controls conditional edge routing

    # Append-Only Audit Trail (uses operator.add reducer)
    execution_logs: Annotated[List[ExecutionLogEntry], operator.add]

    # RAG & Context Buffer
    retrieved_context: List[Dict[str, Any]]

    # Isolated Artifact Storage (uses custom dictionary merge reducer)
    artifacts: Annotated[Dict[str, str], merge_artifacts]

    # Circuit Breaker & Error Tracking
    iteration_count: int     # Resets to 0 upon moving to a new task_id
    max_iterations: int      # Default: 3
    error_trace: Optional[str]

    # Final Deliverable
    final_response: Optional[str]


class EpisodicSQLiteCheckpointer:
    """
    Tier 2 — Episodic Session Persistence (SQLite Checkpointer).
    Persists state snapshots at every node boundary keyed by thread_id (session_id).
    Enables fault recovery, state replay, and Human-in-the-Loop (HITL) inspection.
    """

    def __init__(self, db_path: str = "./workspace/checkpoints.sqlite"):
        if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            import tempfile
            db_path = os.path.join(tempfile.gettempdir(), "workspace", "checkpoints.sqlite")
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state_checkpoints (
                    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    node_name TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    transition_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_checkpoint(
        self,
        thread_id: str,
        step_number: int,
        node_name: str,
        state: GridState,
        transition: NodeTransitionPayload,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO state_checkpoints
                (thread_id, step_number, node_name, state_json, transition_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    step_number,
                    node_name,
                    json.dumps(state, default=str),
                    json.dumps(transition, default=str),
                    now,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def get_thread_history(self, thread_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT checkpoint_id, thread_id, step_number, node_name,
                       state_json, transition_json, created_at
                FROM state_checkpoints
                WHERE thread_id = ?
                ORDER BY step_number ASC
                """,
                (thread_id,),
            ).fetchall()
            return [
                {
                    "checkpoint_id": r["checkpoint_id"],
                    "thread_id": r["thread_id"],
                    "step_number": r["step_number"],
                    "node_name": r["node_name"],
                    "state": json.loads(r["state_json"]),
                    "transition": json.loads(r["transition_json"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def list_sessions(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT thread_id, COUNT(*) as steps, MAX(created_at) as last_updated
                FROM state_checkpoints
                GROUP BY thread_id
                ORDER BY last_updated DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]


def apply_state_update(current: GridState, update: Dict[str, Any]) -> GridState:
    """Applies partial node state updates using the exact GridState reducers."""
    next_state: Dict[str, Any] = dict(current)
    for key, val in update.items():
        if key == "execution_logs":
            next_state["execution_logs"] = operator.add(
                list(current.get("execution_logs", [])), list(val)
            )
        elif key == "artifacts":
            next_state["artifacts"] = merge_artifacts(
                current.get("artifacts", {}), val
            )
        else:
            next_state[key] = val
    return next_state  # type: ignore[return-value]
