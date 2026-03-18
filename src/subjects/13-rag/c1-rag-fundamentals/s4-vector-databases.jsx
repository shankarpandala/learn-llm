import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function VectorDatabases() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Vector Databases: Chroma, Pinecone, Weaviate, pgvector</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Vector databases are specialized storage systems optimized for storing, indexing, and
        querying high-dimensional embedding vectors. They use approximate nearest neighbor (ANN)
        algorithms to achieve sub-linear search time over millions or billions of vectors.
      </p>

      <DefinitionBlock
        title="Approximate Nearest Neighbor (ANN) Search"
        definition="ANN search finds vectors approximately closest to a query vector $q$ in time $O(\log n)$ instead of exact search's $O(n)$, using index structures like HNSW (Hierarchical Navigable Small World graphs) or IVF (Inverted File Index). The trade-off is controlled by a recall parameter."
        id="def-ann"
      />

      <h2 className="text-2xl font-semibold">Index Structures</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The efficiency of vector search depends on the indexing algorithm. HNSW builds a
        multi-layer graph where each node connects to its nearest neighbors, enabling
        greedy traversal that approximates exact search:
      </p>
      <BlockMath math="\text{HNSW complexity: } O(\log n) \text{ query time, } O(n \log n) \text{ build time}" />

      <ExampleBlock
        title="Vector Database Selection"
        problem="Choose a vector database based on requirements for a RAG application."
        steps={[
          { formula: '\\text{Chroma: embedded, } < 1M \\text{ vectors}', explanation: 'Ideal for prototyping, local development, and small-scale applications. Runs in-process.' },
          { formula: '\\text{Pinecone: managed, } > 1M \\text{ vectors}', explanation: 'Fully managed cloud service with automatic scaling. Best for production without infra overhead.' },
          { formula: '\\text{Weaviate: self-hosted, hybrid search}', explanation: 'Supports both vector and keyword search natively. Good for complex filtering requirements.' },
          { formula: '\\text{pgvector: existing Postgres, } < 10M \\text{ vectors}', explanation: 'Extension for PostgreSQL. Best when you already use Postgres and want unified storage.' },
        ]}
        id="example-db-selection"
      />

      <PythonCode
        title="vector_databases.py"
        code={`# Chroma - lightweight, embedded vector database
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.Client()  # In-memory; use PersistentClient for disk
collection = client.create_collection("rag_docs", embedding_function=ef)

# Add documents
collection.add(
    documents=[
        "Transformers use self-attention mechanisms.",
        "BERT is a bidirectional transformer model.",
        "GPT generates text autoregressively.",
    ],
    ids=["doc1", "doc2", "doc3"],
    metadatas=[
        {"source": "textbook", "chapter": 4},
        {"source": "paper", "year": 2018},
        {"source": "paper", "year": 2018},
    ],
)

# Query with metadata filtering
results = collection.query(
    query_texts=["How does attention work?"],
    n_results=2,
    where={"source": "textbook"},
)
print("Chroma results:", results["documents"])

# pgvector - PostgreSQL extension
# pip install pgvector psycopg2-binary
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect("postgresql://localhost/ragdb")
register_vector(conn)
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding vector(384),
        metadata JSONB
    )
""")
cur.execute("""
    CREATE INDEX IF NOT EXISTS docs_embedding_idx
    ON documents USING hnsw (embedding vector_cosine_ops)
""")

# Query nearest neighbors
cur.execute("""
    SELECT content, 1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT 5
""", (query_embedding, query_embedding))
results = cur.fetchall()`}
        id="code-vector-dbs"
      />

      <NoteBlock
        type="tip"
        title="Start Simple, Scale Later"
        content="Begin with Chroma for prototyping - it requires zero configuration and runs in-process. Move to pgvector if you already use PostgreSQL, or to a managed service like Pinecone when you need production-grade scaling. Premature infrastructure complexity is a common pitfall in RAG projects."
        id="note-start-simple"
      />

      <h2 className="text-2xl font-semibold">Distance Metrics</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Vector databases support multiple distance metrics. The choice depends on your embedding
        model and use case:
      </p>
      <BlockMath math="\text{Cosine: } 1 - \frac{a \cdot b}{\|a\|\|b\|} \quad \text{L2: } \|a - b\|_2 \quad \text{Dot: } -a \cdot b" />

      <WarningBlock
        title="Index Building Takes Time"
        content="HNSW index construction is O(n log n) and can take hours for millions of vectors. Plan for index build time in your deployment pipeline. Some databases like Pinecone handle this transparently, but self-hosted solutions like pgvector or Weaviate require explicit index management and tuning of parameters like ef_construction and M."
        id="warning-index-time"
      />

      <NoteBlock
        type="note"
        title="Metadata Filtering"
        content="All major vector databases support metadata filtering alongside vector search. This enables queries like 'find similar documents from 2024 in the finance domain.' Pre-filtering (before ANN search) is more efficient but may miss results; post-filtering (after ANN search) is more accurate but may return fewer results than requested."
        id="note-metadata"
      />
    </div>
  )
}
