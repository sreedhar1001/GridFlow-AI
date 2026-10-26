# GridFlow AI: Knowledge Base & RAG Specification (`knowledge_base.md`)

## 1. Vector Store Architecture & Configuration
GridFlow AI utilizes **ChromaDB** as its persistent semantic knowledge engine, enabling the `Researcher_Node` to retrieve verified domain context, architectural blueprints, and code documentation with zero hallucination.

* **Storage Mode:** Persistent Local Client (`./chroma_db_store`)
* **Primary Collection:** `gridflow_core_knowledge`
* **Embedding Function:** `all-MiniLM-L6-v2` (384 dimensions via `sentence-transformers`) or `text-embedding-3-small` (1536 dimensions via OpenAI)
* **Distance Metric:** Cosine Similarity (`{"hnsw:space": "cosine"}`)

### Initialization Reference Implementation:
```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize persistent ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db_store")

# Configure deterministic embedding function
default_ef = embedding_functions.DefaultEmbeddingFunction()

# Create or load the primary knowledge collection
knowledge_collection = chroma_client.get_or_create_collection(
    name="gridflow_core_knowledge",
    embedding_function=default_ef,
    metadata={"hnsw:space": "cosine"}
)
```
