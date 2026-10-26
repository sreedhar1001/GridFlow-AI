"""
GridFlow AI: Agent Skill Registry & Tool Implementations (skill.md / tools.py)
Implements all 4 registered skills with strict Role-Based Access Control (RBAC),
Pydantic v2 schema validation (returning VALIDATION_ERROR on malformed inputs),
10-second network timeouts, and ./workspace/outputs/ filesystem sandboxing.
"""

import os
import ast
import re
import urllib.request
import urllib.error
from html.parser import HTMLParser
from typing import Optional, List, Dict, Any, Callable

try:
    from pydantic import BaseModel, Field, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

    class ValidationError(Exception):  # type: ignore[no-redef]
        pass

    class BaseModel:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def Field(default: Any = ..., **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return default

try:
    from langchain_core.tools import tool as lc_tool
except ImportError:
    lc_tool = None

from knowledge_base import chroma_client, default_ef, knowledge_collection


def tool(name: str, args_schema: Any = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that wraps functions as LangChain-compatible tools with schema metadata."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func.tool_name = name  # type: ignore[attr-defined]
        func.name = name  # type: ignore[attr-defined]
        func.args_schema = args_schema  # type: ignore[attr-defined]
        func.invoke = lambda kwargs: func(**kwargs) if isinstance(kwargs, dict) else func(kwargs)  # type: ignore[attr-defined]
        return func
    return decorator


class _CleanHTMLTextExtractor(HTMLParser):
    """Fallback HTML parser when BeautifulSoup is not installed."""
    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag.lower() in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self.chunks.append(stripped)


# =====================================================================
# SKILL 01: ChromaDB Semantic Search
# =====================================================================

class ChromaSearchInput(BaseModel):
    query: str = Field(..., description="Concise semantic search query for retrieving technical context.")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of candidate chunks to retrieve.")
    source_filter: Optional[str] = Field(
        default=None,
        description="Optional filter: 'architecture_spec', 'api_reference', 'code_template', or 'user_upload'."
    )

    @classmethod
    def validate_args(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Field 'query' must be a non-empty string.")
        top_k = kwargs.get("top_k", 5)
        if not isinstance(top_k, int) or not (1 <= top_k <= 10):
            raise ValueError("Field 'top_k' must be an integer between 1 and 10.")
        source_filter = kwargs.get("source_filter", None)
        allowed_filters = {None, "architecture_spec", "api_reference", "code_template", "user_upload"}
        if source_filter not in allowed_filters:
            raise ValueError(f"Field 'source_filter' must be one of {allowed_filters}.")
        return {"query": query.strip(), "top_k": top_k, "source_filter": source_filter}


@tool("chromadb_semantic_search", args_schema=ChromaSearchInput)
def chromadb_semantic_search(query: str, top_k: int = 5, source_filter: Optional[str] = None) -> Dict[str, Any]:
    """Queries the persistent ChromaDB knowledge base and filters chunks by cosine similarity >= 0.72."""
    try:
        validated = ChromaSearchInput.validate_args(
            {"query": query, "top_k": top_k, "source_filter": source_filter}
        )
        query = validated["query"]
        top_k = validated["top_k"]
        source_filter = validated["source_filter"]
    except Exception as ve:
        return {"status": "VALIDATION_ERROR", "error": str(ve), "results": []}

    try:
        where_clause = {"source_type": source_filter} if source_filter else None
        results = knowledge_collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        valid_chunks = []
        if results and results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for doc, meta, dist in zip(docs, metas, distances):
                similarity = round(1.0 - float(dist), 4)
                if similarity >= 0.72:
                    valid_chunks.append({
                        "doc_id": meta.get("doc_id", "unknown"),
                        "source_type": meta.get("source_type", "general"),
                        "similarity_score": similarity,
                        "content": doc
                    })

        if not valid_chunks:
            return {
                "status": "INSUFFICIENT_LOCAL_CONTEXT",
                "message": "No chunks passed the 0.72 similarity threshold. Trigger web_scraper_extracter if needed.",
                "results": []
            }

        return {
            "status": "SUCCESS",
            "retrieved_count": len(valid_chunks),
            "results": valid_chunks
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e), "results": []}


# =====================================================================
# SKILL 02: Web Documentation Scraper & Extractor
# =====================================================================

class WebScraperInput(BaseModel):
    url: str = Field(..., description="Valid HTTP/HTTPS documentation URL to extract text from.")
    max_chars: int = Field(default=4000, ge=500, le=10000, description="Maximum character limit for extracted text.")

    @classmethod
    def validate_args(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        url = kwargs.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("Field 'url' must be a non-empty string.")
        max_chars = kwargs.get("max_chars", 4000)
        if not isinstance(max_chars, int) or not (500 <= max_chars <= 10000):
            raise ValueError("Field 'max_chars' must be an integer between 500 and 10000.")
        return {"url": url.strip(), "max_chars": max_chars}


@tool("web_scraper_extracter", args_schema=WebScraperInput)
def web_scraper_extracter(url: str, max_chars: int = 4000) -> Dict[str, Any]:
    """Fetches and extracts clean text from a live webpage when local RAG knowledge is insufficient."""
    try:
        validated = WebScraperInput.validate_args({"url": url, "max_chars": max_chars})
        url = validated["url"]
        max_chars = validated["max_chars"]
    except Exception as ve:
        return {"status": "VALIDATION_ERROR", "error": str(ve)}

    if not url.startswith(("http://", "https://")):
        return {"status": "ERROR", "error": "Invalid URL scheme. Must start with http:// or https://"}

    try:
        headers = {"User-Agent": "GridFlowAI-Researcher/1.0"}
        try:
            import requests  # type: ignore[import-untyped]
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            html_text = response.text
        except ImportError:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                html_text = resp.read().decode("utf-8", errors="replace")

        try:
            from bs4 import BeautifulSoup  # type: ignore[import-untyped]
            soup = BeautifulSoup(html_text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                tag.decompose()
            main_content = soup.find("main") or soup.find("article") or soup.body
            raw_text = main_content.get_text(separator="\n", strip=True) if main_content else ""
        except ImportError:
            parser = _CleanHTMLTextExtractor()
            parser.feed(html_text)
            raw_text = "\n".join(parser.chunks)

        cleaned_text = re.sub(r"\n{3,}", "\n\n", raw_text)[:max_chars]

        return {
            "status": "SUCCESS",
            "source_url": url,
            "extracted_length": len(cleaned_text),
            "content": cleaned_text
        }
    except Exception as e:
        return {"status": "ERROR", "source_url": url, "error": str(e)}


# =====================================================================
# SKILL 03: Python AST Syntax & Security Validator
# =====================================================================

class PythonASTInput(BaseModel):
    code_string: str = Field(..., description="Raw Python code string to validate for syntax and security.")

    @classmethod
    def validate_args(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        code_string = kwargs.get("code_string")
        if not isinstance(code_string, str) or not code_string.strip():
            raise ValueError("Field 'code_string' must be a non-empty Python code string.")
        return {"code_string": code_string}


BANNED_CALLS = {"eval", "exec", "compile"}
BANNED_MODULES = {"subprocess", "shutil"}


@tool("python_ast_validator", args_schema=PythonASTInput)
def python_ast_validator(code_string: str) -> Dict[str, Any]:
    """Parses Python code via AST to verify zero syntax errors and check for restricted operations."""
    try:
        validated = PythonASTInput.validate_args({"code_string": code_string})
        code_string = validated["code_string"]
    except Exception as ve:
        return {
            "status": "VALIDATION_ERROR",
            "is_syntax_valid": False,
            "syntax_error": str(ve),
            "security_warnings": []
        }

    clean_code = re.sub(r"^```python\n|```$", "", code_string.strip(), flags=re.MULTILINE)
    security_warnings: List[str] = []

    try:
        tree = ast.parse(clean_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in BANNED_MODULES:
                        security_warnings.append(f"Restricted module import detected: '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in BANNED_MODULES:
                    security_warnings.append(f"Restricted module import detected: '{node.module}'")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in BANNED_CALLS:
                    security_warnings.append(f"Restricted builtin call detected: '{node.func.id}()'")

        return {
            "status": "VALID" if not security_warnings else "SECURITY_FLAGGED",
            "is_syntax_valid": True,
            "syntax_error": None,
            "security_warnings": security_warnings
        }
    except SyntaxError as se:
        return {
            "status": "SYNTAX_ERROR",
            "is_syntax_valid": False,
            "syntax_error": f"Line {se.lineno}, Col {se.offset}: {se.msg}",
            "security_warnings": []
        }
    except Exception as e:
        return {"status": "ERROR", "is_syntax_valid": False, "syntax_error": str(e), "security_warnings": []}


# =====================================================================
# SKILL 04: Sandboxed Artifact File Writer
# =====================================================================

class ArtifactWriterInput(BaseModel):
    filename: str = Field(..., description="Target filename (e.g., 'solution.py' or 'report.md'). No directory paths.")
    content: str = Field(..., description="Complete file content to write to disk.")

    @classmethod
    def validate_args(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        filename = kwargs.get("filename")
        content = kwargs.get("content")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("Field 'filename' must be a non-empty string.")
        if not isinstance(content, str):
            raise ValueError("Field 'content' must be a string.")
        return {"filename": filename, "content": content}


@tool("artifact_file_writer", args_schema=ArtifactWriterInput)
def artifact_file_writer(filename: str, content: str) -> Dict[str, Any]:
    """Safely writes generated code or markdown artifacts into './workspace/outputs/'."""
    try:
        validated = ArtifactWriterInput.validate_args({"filename": filename, "content": content})
        filename = validated["filename"]
        content = validated["content"]
    except Exception as ve:
        return {"status": "VALIDATION_ERROR", "error": str(ve)}

    safe_filename = os.path.basename(filename)
    if not re.match(r"^[\w\-. ]+$", safe_filename) or ".." in filename or safe_filename != filename:
        return {"status": "ERROR", "error": "Invalid filename. Path traversal is strictly prohibited."}

    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        import tempfile
        output_dir = os.path.join(tempfile.gettempdir(), "workspace", "outputs")
    else:
        output_dir = os.path.abspath("./workspace/outputs")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, safe_filename)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return {
            "status": "SUCCESS",
            "artifact_id": safe_filename,
            "saved_path": file_path,
            "bytes_written": len(content.encode("utf-8"))
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


# =====================================================================
# Node Tool Registries & RBAC Enforcement
# =====================================================================
RESEARCHER_TOOLS = [chromadb_semantic_search, web_scraper_extracter]
EXECUTOR_TOOLS = [python_ast_validator, artifact_file_writer]
REVIEWER_TOOLS = [python_ast_validator]

NODE_RBAC_REGISTRY: Dict[str, Dict[str, Callable[..., Any]]] = {
    "Planner_Node": {},
    "Researcher_Node": {t.tool_name: t for t in RESEARCHER_TOOLS},  # type: ignore[attr-defined]
    "Executor_Node": {t.tool_name: t for t in EXECUTOR_TOOLS},      # type: ignore[attr-defined]
    "Reviewer_Node": {t.tool_name: t for t in REVIEWER_TOOLS},      # type: ignore[attr-defined]
}


def invoke_tool_with_rbac(node_name: str, tool_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforces Role-Based Access Control (RBAC) per skill.md Section 1.1:
    Nodes may only invoke tools explicitly bound to their role.
    """
    allowed_tools = NODE_RBAC_REGISTRY.get(node_name, {})
    if tool_name not in allowed_tools:
        return {
            "status": "RBAC_VIOLATION",
            "error": (
                f"Access Denied: Node '{node_name}' is not authorized to invoke '{tool_name}'. "
                f"Permitted tools: {list(allowed_tools.keys())}"
            ),
        }
    target_tool = allowed_tools[tool_name]
    return target_tool(**kwargs)
