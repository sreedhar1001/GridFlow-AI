# GridFlow AI — Complete Devpost Hackathon Submission Kit

Use the sections below to copy-paste directly into your **Devpost** submission form.

---

## 1. Project Name & Tagline (Elevator Pitch)
- **Project Name:** `GridFlow AI`
- **Elevator Pitch (under 200 chars):**
  `A stateful, cyclic multi-agent execution grid with 3-tier memory, ChromaDB RAG (cosine >= 0.72), RBAC tool sandboxing, and an AST circuit-breaker that self-heals code before delivery.`

---

## 2. "Try It Out" Links & GitHub Repository
- **Live Demo Web App (Vercel):** `https://express-silicon-8b34x3q.tmp.now`
- **GitHub Repository:** `https://github.com/sreedhar1001/GridFlow-AI`

---

## 3. Video Demo Link (How Devpost Works)
> **Important Devpost Rule:** Devpost's **Video demo link** field ONLY accepts a **YouTube**, **Vimeo**, **Facebook Video**, or **Youku** URL (e.g., `https://www.youtube.com/watch?v=...` or `https://vimeo.com/...`).

### Option A — Upload the Generated Video File (`gridflow_demo_video.mp4` / Screen Recording) in 30 Seconds:
1. Press **`Win + Alt + R`** (Windows Xbox Game Bar) while viewing **https://express-silicon-8b34x3q.tmp.now** (or `http://127.0.0.1:8765`) and click through:
   - **Scenario A (Standard Pass):** Click **Scenario A** $\rightarrow$ show `Planner_Node` $\rightarrow$ `Researcher_Node` $\rightarrow$ `Executor_Node` $\rightarrow$ `Reviewer_Node` $\rightarrow$ `DELIVERABLE VERIFIED`.
   - **Scenario B (Self-Healing Retry Loop):** Click **Scenario B** $\rightarrow$ show `Reviewer_Node` catching `import subprocess` (`NEEDS_REVISION`) and `Executor_Node` automatically fixing it on Iteration 2!
   - **Scenario C (Circuit Breaker):** Click **Scenario C** $\rightarrow$ show 3 consecutive rejections tripping `Fallback_Handler`.
   - **Tab 2 (Skill & RBAC Security Bench):** Click **Demo 1 (RBAC Violation)** and **Demo 2 (AST Security Flag)**.
2. Upload that 60-second recording to **[YouTube Studio](https://studio.youtube.com)** (set visibility to **Public** or **Unlisted**) or **[Vimeo](https://vimeo.com/upload)** and paste the URL into Devpost!

---

## 4. Full Project Story (Copy & Paste into Devpost "Project Story" Markdown Box)

```markdown
## Inspiration
Modern LLM agent pipelines frequently fail in production due to three critical vulnerabilities: **unbounded infinite retry loops**, **hallucinated citations**, and **unverified code execution**. We built **GridFlow AI** to solve these challenges through a deterministic, stateful, cyclic multi-agent execution architecture with strict quality gates and circuit-breaker guarantees.

## What it does
**GridFlow AI** is a stateful, cyclic multi-agent orchestrator that decomposes complex engineering goals into a topological Directed Acyclic Graph (DAG) of sub-tasks and routes them across four specialized autonomous nodes plus a deterministic fault-recovery handler:

1. **`Planner_Node`**: Decomposes high-level user prompts into a topological DAG of `SubTask` items with explicit dependency tracking.
2. **`Researcher_Node`**: Queries our persistent **ChromaDB** semantic vector store (`gridflow_core_knowledge`) enforcing a strict cosine similarity threshold ($s \ge 0.72$) and falls back to live documentation extraction (`web_scraper_extracter`) when local RAG context is insufficient—never fabricating citations.
3. **`Executor_Node`**: Synthesizes production Python code and technical reports grounded in retrieved citations and persists artifacts inside a jailed `./workspace/outputs/` sandbox (`artifact_file_writer`).
4. **`Reviewer_Node` (Circuit Breaker)**: Parses generated Python code via `ast.parse` (`python_ast_validator`) to verify zero syntax errors and scan for restricted imports (`subprocess`, `shutil`) and unsafe builtins (`eval`, `exec`, `compile`).
   - **Self-Healing Loop**: If a violation is detected, `Reviewer_Node` emits `NEEDS_REVISION` and routes the error trace back to `Executor_Node` for automated remediation.
   - **Loop Prevention (`max_iterations = 3`)**: If a sub-task fails verification 3 times (`iteration_count == 3`), the circuit breaker trips and forces a deterministic transition to `Fallback_Handler`, preserving partial artifacts and full audit logs for Human-in-the-Loop (HITL) inspection.

## How we built it
- **Three-Tier Memory Architecture (`memory.md` / `state.py`)**:
  - **Tier 1 (Short-Term Graph State)**: Typed `GridState` schema using `operator.add` for append-only `execution_logs` and a custom `merge_artifacts` reducer for isolated artifact storage.
  - **Tier 2 (Episodic Session Persistence)**: `EpisodicSQLiteCheckpointer` saving full JSON state snapshots into SQLite (`./workspace/checkpoints.sqlite`) at every node boundary keyed by `thread_id`.
  - **Tier 3 (Long-Term Semantic Memory)**: Persistent ChromaDB vector store (`./chroma_db_store`) with 384-dimensional cosine similarity indexing.
- **Role-Based Access Control (RBAC) & Pydantic v2 Validation (`skill.md` / `tools.py`)**: Every tool invocation is gated by `NODE_RBAC_REGISTRY` and strict schema validation so unauthorized tool calls or path-traversal attempts (`../`) are blocked deterministically.
- **Interactive Real-Time Control Center (`server.py` + `static/index.html`)**: Built a zero-dependency web dashboard deployed on Vercel that visualizes the cyclic DAG, mandatory JSON transition payloads, RBAC security testbench, and 3-tier memory snapshots in real time.

## Challenges we ran into
- Preventing context window overflow during cyclic retries between `Executor_Node` and `Reviewer_Node` while keeping an immutable audit trail. Solving this required designing custom state reducers (`merge_artifacts` and `operator.add`) and offloading full snapshots to Tier-2 SQLite checkpoints.
- Ensuring deterministic AST security validation that catches both direct and aliased restricted imports (`subprocess`, `shutil`) and builtin calls (`eval`, `exec`, `compile`) before any artifact is marked `COMPLETED`.

## Accomplishments that we're proud of
- **100% Schema & Protocol Compliance**: Every node turn emits a deterministic JSON transition payload (`current_node`, `task_id`, `status`, `reasoning_trace`, `payload`, `next_node`, `iteration_count`).
- **Live Self-Healing & Circuit Breaker Verification**: Demonstrating live automatic code remediation on Iteration 2 and guaranteed termination on Iteration 3 without infinite loops.
- **7/7 Automated Unit & Integration Tests Passing** in under 0.2 seconds.

## What we learned
Combining **strict AST static analysis**, **role-based tool sandboxing**, and **bounded cyclic state graphs** transforms brittle LLM chains into verifiable, production-grade software engineering systems.

## What's next for GridFlow AI
- Parallel multi-branch execution for independent DAG sub-tasks.
- Distributed Redis checkpointer support for multi-worker cloud clusters.
- Interactive Human-in-the-Loop (HITL) breakpoint editing directly from the Tier-2 checkpoint browser.
```

---

## 5. "Built With" Tags (Enter these in Devpost)
`python`, `langgraph`, `langchain`, `chromadb`, `pydantic`, `sqlite`, `ast`, `beautifulsoup`, `javascript`, `html5`, `css3`, `vercel`
