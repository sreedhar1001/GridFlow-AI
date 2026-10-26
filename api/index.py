"""
Vercel Serverless Function Entrypoint for GridFlow AI
Exposes `handler` (inheriting from BaseHTTPRequestHandler via SimpleHTTPRequestHandler)
so Vercel natively serves both `/` (`static/index.html`) and all `/api/*` endpoints.
"""

import os
import sys

# Ensure project root is on sys.path in Vercel serverless environment
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from server import GridFlowRequestHandler, ORCHESTRATOR

# Ensure at least one initial session checkpoint exists on cold start
try:
    if not ORCHESTRATOR.checkpointer.list_sessions():
        ORCHESTRATOR.run_workflow(
            user_query="Build a thread-safe Token Bucket Rate Limiter API service in Python",
            session_id="demo-initial-run",
            mode="standard",
        )
except Exception:
    pass


class handler(GridFlowRequestHandler):
    """Vercel Python Runtime HTTP Handler."""
    pass
