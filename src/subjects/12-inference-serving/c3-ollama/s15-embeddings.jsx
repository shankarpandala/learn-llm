import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Embeddings() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Ollama for Embeddings</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Beyond text generation, Ollama can serve embedding models that convert text into dense
        vector representations. These embeddings power similarity search, RAG systems, clustering,
        and classification, all running locally with no API costs.
      </p>

      <DefinitionBlock
        title="Embedding Models in Ollama"
        definition="Embedding models map text to fixed-dimensional vectors where semantically similar texts produce similar vectors. Ollama serves embedding models via the /api/embeddings endpoint. Popular choices include nomic-embed-text (768 dims) and mxbai-embed-large (1024 dims)."
        id="def-embeddings"
      />

      <PythonCode
        title="Terminal"
        code={`# Pull embedding models
ollama pull nomic-embed-text      # 274MB, 768 dimensions
ollama pull mxbai-embed-large     # 670MB, 1024 dimensions
ollama pull all-minilm            # 45MB, 384 dimensions (tiny but fast)

# Generate an embedding via API
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "Ollama makes embeddings easy"
}'
# Returns: {"embedding": [0.0123, -0.0456, ...]}  (768 floats)`}
        id="code-setup"
      />

      <PythonCode
        title="embeddings_demo.py"
        code={`import ollama
import numpy as np

def get_embedding(text, model="nomic-embed-text"):
    """Get embedding vector for a text string."""
    resp = ollama.embeddings(model=model, prompt=text)
    return np.array(resp["embedding"])

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Demonstrate semantic similarity
texts = [
    "The cat sat on the mat",
    "A feline rested on the rug",      # semantically similar
    "Quantum physics is fascinating",   # unrelated
    "The dog lay on the carpet",        # somewhat similar
]

embeddings = [get_embedding(t) for t in texts]

print("Pairwise cosine similarities:")
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"  {sim:.3f}: '{texts[i][:40]}' <-> '{texts[j][:40]}'")

# Simple semantic search
query = "A pet resting on furniture"
query_emb = get_embedding(query)

scores = [(cosine_similarity(query_emb, emb), text)
          for emb, text in zip(embeddings, texts)]
scores.sort(reverse=True)

print(f"\\nSearch: '{query}'")
for score, text in scores:
    print(f"  {score:.3f}: {text}")`}
        id="code-demo"
      />

      <PythonCode
        title="local_vector_store.py"
        code={`import ollama
import numpy as np
import json

class LocalVectorStore:
    """Minimal vector store using Ollama embeddings."""

    def __init__(self, model="nomic-embed-text"):
        self.model = model
        self.documents = []
        self.embeddings = []

    def add(self, text, metadata=None):
        resp = ollama.embeddings(model=self.model, prompt=text)
        self.documents.append({"text": text, "metadata": metadata or {}})
        self.embeddings.append(np.array(resp["embedding"]))

    def search(self, query, top_k=3):
        query_emb = np.array(
            ollama.embeddings(model=self.model, prompt=query)["embedding"]
        )
        scores = [
            np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            for emb in self.embeddings
        ]
        indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], scores[i]) for i in indices]

# Build a knowledge base
store = LocalVectorStore()
store.add("Python is a high-level programming language", {"topic": "python"})
store.add("JavaScript runs in web browsers", {"topic": "js"})
store.add("Docker containers package applications", {"topic": "devops"})
store.add("Neural networks are inspired by the brain", {"topic": "ml"})
store.add("SQL is used for database queries", {"topic": "databases"})

results = store.search("How do I build a web app?", top_k=2)
for doc, score in results:
    print(f"  [{score:.3f}] {doc['text']}")`}
        id="code-vector-store"
      />

      <NoteBlock
        type="tip"
        title="Embedding Speed"
        content="Embedding models are much faster than generative models because they process input in a single forward pass with no autoregressive loop. nomic-embed-text can embed hundreds of documents per second on a modern GPU. Batch your embedding requests for maximum throughput."
        id="note-speed"
      />

      <WarningBlock
        title="Choose the Right Model for Your Use Case"
        content="nomic-embed-text is optimized for retrieval (RAG). all-minilm is tiny but lower quality. For production RAG systems, test your embedding model on your specific domain -- a model that excels on general benchmarks may underperform on specialized text like medical or legal documents."
        id="warning-model-choice"
      />
    </div>
  )
}
