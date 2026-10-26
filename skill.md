# GridFlow AI: Agent Skill Registry & Tool Implementations (`skill.md`)

## 1. Tool Execution & Security Policy
All autonomous nodes in the GridFlow execution grid must adhere to the following runtime constraints when invoking external tools:
1. **Role-Based Access Control (RBAC):** Nodes may only invoke tools explicitly bound to their role (`Researcher_Node`, `Executor_Node`, or `Reviewer_Node`).
2. **Strict Schema Validation:** All tool arguments are validated via **Pydantic v2** schemas before execution. Malformed tool calls return a structured `VALIDATION_ERROR` dictionary rather than crashing the LangGraph state.
3. **Execution Timeouts & Sandboxing:** Network calls enforce a hard timeout of `10 seconds`. File system writes are strictly jailed to the `./workspace/outputs/` directory.

---

## 2. Node-to-Skill Binding Matrix

| Skill Name | Assigned Node(s) | Primary Purpose |
| :--- | :--- | :--- |
| `chromadb_semantic_search` | `Researcher_Node` | Queries local ChromaDB vector store with cosine similarity filtering ($s \ge 0.72$). |
| `web_scraper_extracter` | `Researcher_Node` | Fallback scraper using `BeautifulSoup` when local RAG context is insufficient. |
| `python_ast_validator` | `Executor_Node`, `Reviewer_Node` | Validates generated Python syntax via `ast.parse` and scans for unsafe imports. |
| `artifact_file_writer` | `Executor_Node` | Safely writes verified code, reports, or configs to `./workspace/outputs/`. |
