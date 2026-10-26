# GridFlow AI — Stateful Cyclic Multi-Agent Execution Environment

Built with 100% adherence to the four GridFlow AI specifications:
1. **System Instructions & Orchestration Protocol** ([gemini-code-1793002808914.md](file:///C:/Users/sridh/Desktop/GridFlow%20AI/gemini-code-1793002808914.md) / [orchestrator.py](file:///C:/Users/sridh/Desktop/GridFlow%20AI/orchestrator.py))
2. **State Management & Memory Architecture (`memory.md`)** ([gemini-code-1793002907443.md](file:///C:/Users/sridh/Desktop/GridFlow%20AI/gemini-code-1793002907443.md) / [state.py](file:///C:/Users/sridh/Desktop/GridFlow%20AI/state.py))
3. **Knowledge Base & RAG Specification (`knowledge_base.md`)** ([gemini-code-1793002998209.md](file:///C:/Users/sridh/Desktop/GridFlow%20AI/gemini-code-1793002998209.md) / [knowledge_base.py](file:///C:/Users/sridh/Desktop/GridFlow%20AI/knowledge_base.py))
4. **Agent Skill Registry & Tool Implementations (`skill.md` / `tools.py`)** ([gemini-code-1793003085385.md](file:///C:/Users/sridh/Desktop/GridFlow%20AI/gemini-code-1793003085385.md) / [tools.py](file:///C:/Users/sridh/Desktop/GridFlow%20AI/tools.py))

---

## Core Architecture

- **4 Specialized Autonomous Nodes + Circuit Breaker Fallback:**
  1. `Planner_Node`: Decomposes high-level user goals into a topological Directed Acyclic Graph (DAG) of `SubTask`s.
  2. `Researcher_Node`: Queries `./chroma_db_store` (`gridflow_core_knowledge`) using cosine similarity filtering ($s \ge 0.72$) and `web_scraper_extracter` fallback. Never fabricates citations.
  3. `Executor_Node`: Synthesizes production Python code and reports, validates via `python_ast_validator`, and writes artifacts to `./workspace/outputs/` via `artifact_file_writer`.
  4. `Reviewer_Node` (Circuit Breaker): Validates `Executor_Node` outputs for syntax errors (`ast.parse`), banned imports/calls (`subprocess`, `shutil`, `eval`, `exec`, `compile`), and citation alignment. Enforces `max_iterations = 3` loop prevention, routing to `Fallback_Handler` if `iteration_count == 3`.
- **Three-Tier Memory Hierarchy:**
  - **Tier 1:** In-Memory `GridState` `TypedDict` with `operator.add` reducer on `execution_logs` and `merge_artifacts` custom reducer on `artifacts`.
  - **Tier 2:** Episodic Session Persistence via `EpisodicSQLiteCheckpointer` (`./workspace/checkpoints.sqlite`) saving state snapshots at every node boundary keyed by `thread_id`.
  - **Tier 3:** Long-Term Semantic Memory in persistent ChromaDB (`./chroma_db_store`).

---

## Running the Live Demo & Tests

### 1. Launch Interactive Web Control Center (Live Demo UI)
```powershell
& "C:\Users\sridh\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe" server.py
```
Then open **http://127.0.0.1:8765** in your browser.

### 2. Run Interactive CLI Demo (All 3 Scenarios)
```powershell
& "C:\Users\sridh\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe" demo_cli.py
```

### 3. Run Automated Verification Suite (7 Unit & Integration Tests)
```powershell
& "C:\Users\sridh\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe" test_gridflow.py
```
