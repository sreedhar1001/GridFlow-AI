"""
GridFlow AI: Live Hackathon Demo Server & REST API
Serves the interactive GridFlow AI Orchestration Control Center UI and provides
REST endpoints for:
- POST /api/run          : Execute full cyclic multi-agent workflow (standard, self_heal, circuit_breaker)
- POST /api/tool         : Test individual skills with RBAC & Pydantic validation
- POST /api/kb/upsert    : Add custom documents into ChromaDB ('gridflow_core_knowledge')
- GET  /api/status       : Inspect Tier 2 SQLite sessions, Tier 3 ChromaDB docs, and ./workspace/outputs/ artifacts
- GET  /api/session/<id> : Retrieve full Tier 2 SQLite checkpoint history for any thread_id
"""

import os
import sys
import json
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from knowledge_base import knowledge_collection, SEED_DOCUMENTS
from tools import (
    chromadb_semantic_search,
    web_scraper_extracter,
    python_ast_validator,
    artifact_file_writer,
    invoke_tool_with_rbac,
    NODE_RBAC_REGISTRY,
)
from orchestrator import GridFlowOrchestrator

import tempfile

ORCHESTRATOR = GridFlowOrchestrator()
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    OUTPUTS_DIR = os.path.join(tempfile.gettempdir(), "workspace", "outputs")
else:
    OUTPUTS_DIR = os.path.abspath("./workspace/outputs")


class GridFlowRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler serving both the static dashboard and JSON API endpoints."""

    def log_message(self, format: str, *args: object) -> None:
        sys.stdout.write(f"[GridFlow-Server] {self.address_string()} - {format % args}\n")
        sys.stdout.flush()

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def do_OPTIONS(self) -> None:
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        endpoint = qs.get("endpoint", [""])[0]

        if path == "/api/status" or endpoint == "status":
            os.makedirs(OUTPUTS_DIR, exist_ok=True)
            output_files = []
            for fname in sorted(os.listdir(OUTPUTS_DIR)):
                fpath = os.path.join(OUTPUTS_DIR, fname)
                if os.path.isfile(fpath):
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    output_files.append({
                        "filename": fname,
                        "bytes": os.path.getsize(fpath),
                        "content": content,
                    })

            kb_data = knowledge_collection.get()
            kb_docs = [
                {
                    "id": doc_id,
                    "document": doc,
                    "metadata": meta,
                }
                for doc_id, doc, meta in zip(
                    kb_data.get("ids", []),
                    kb_data.get("documents", []),
                    kb_data.get("metadatas", []),
                )
            ]

            sessions = ORCHESTRATOR.checkpointer.list_sessions()
            rbac_matrix = {
                node: list(tools.keys()) for node, tools in NODE_RBAC_REGISTRY.items()
            }

            self._send_json(200, {
                "status": "ONLINE",
                "collection_name": "gridflow_core_knowledge",
                "chroma_path": "./chroma_db_store",
                "kb_count": knowledge_collection.count(),
                "kb_documents": kb_docs,
                "sessions": sessions,
                "output_files": output_files,
                "rbac_matrix": rbac_matrix,
            })
            return

        if path.startswith("/api/session/") or endpoint == "session":
            thread_id = qs.get("thread_id", [""])[0] or path.split("/api/session/", 1)[-1].strip("/")
            history = ORCHESTRATOR.checkpointer.get_thread_history(thread_id)
            self._send_json(200, {
                "thread_id": thread_id,
                "checkpoint_count": len(history),
                "checkpoints": history,
            })
            return

        # Serve static UI files for root or rewritten index routes
        if path in ("/", "/index.html", "/api/index.py", "/api/index"):
            index_path = os.path.join(STATIC_DIR, "index.html")
            if os.path.exists(index_path):
                with open(index_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        self._send_json(404, {"error": f"Route not found: {path}"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        endpoint = qs.get("endpoint", [""])[0]

        try:
            body = self._read_json_body()
        except Exception as e:
            self._send_json(400, {"status": "ERROR", "error": f"Invalid JSON payload: {e}"})
            return

        if path == "/api/run" or endpoint == "run":
            user_query = body.get(
                "user_query",
                "Build a thread-safe Token Bucket Rate Limiter API service in Python",
            )
            mode = body.get("mode", "standard")
            session_id = body.get("session_id") or None
            max_iterations = int(body.get("max_iterations", 3))

            result = ORCHESTRATOR.run_workflow(
                user_query=user_query,
                session_id=session_id,
                mode=mode,
                max_iterations=max_iterations,
            )
            self._send_json(200, result)
            return

        if path == "/api/tool" or endpoint == "tool":
            node_name = body.get("node_name", "Researcher_Node")
            tool_name = body.get("tool_name", "chromadb_semantic_search")
            kwargs = body.get("kwargs", {})
            enforce_rbac = body.get("enforce_rbac", True)

            if enforce_rbac:
                res = invoke_tool_with_rbac(node_name=node_name, tool_name=tool_name, kwargs=kwargs)
            else:
                tool_map = {
                    "chromadb_semantic_search": chromadb_semantic_search,
                    "web_scraper_extracter": web_scraper_extracter,
                    "python_ast_validator": python_ast_validator,
                    "artifact_file_writer": artifact_file_writer,
                }
                fn = tool_map.get(tool_name)
                if not fn:
                    res = {"status": "ERROR", "error": f"Unknown tool: {tool_name}"}
                else:
                    res = fn(**kwargs)

            self._send_json(200, {
                "node_name": node_name,
                "tool_name": tool_name,
                "result": res,
            })
            return

        if path == "/api/kb/upsert" or endpoint == "kb_upsert":
            doc_id = body.get("doc_id", "DOC-USER-001").strip()
            source_type = body.get("source_type", "user_upload").strip()
            title = body.get("title", "User Uploaded Specification").strip()
            content = body.get("content", "").strip()
            if not doc_id or not content:
                self._send_json(400, {"status": "ERROR", "error": "Both doc_id and content are required."})
                return

            knowledge_collection.upsert(
                ids=[doc_id],
                documents=[content],
                metadatas=[{"doc_id": doc_id, "source_type": source_type, "title": title}],
            )
            self._send_json(200, {
                "status": "SUCCESS",
                "doc_id": doc_id,
                "total_documents": knowledge_collection.count(),
            })
            return

        self._send_json(404, {"error": f"POST route not found: {path}"})


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    # Pre-seed an initial demonstration run so the dashboard immediately displays rich state
    if not ORCHESTRATOR.checkpointer.list_sessions():
        ORCHESTRATOR.run_workflow(
            user_query="Build a thread-safe Token Bucket Rate Limiter API service in Python",
            session_id="demo-initial-run",
            mode="standard",
        )
    server = ThreadingHTTPServer((host, port), GridFlowRequestHandler)
    print(f"============================================================")
    print(f" GridFlow AI Live Control Center running at:")
    print(f" http://{host}:{port}")
    print(f"============================================================")
    sys.stdout.flush()
    server.serve_forever()


if __name__ == "__main__":
    port_arg = int(os.environ.get("GRIDFLOW_PORT", "8765"))
    run_server(port=port_arg)
