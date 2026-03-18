import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function EmbeddingModels() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Embedding Models for Retrieval</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The quality of a RAG system depends heavily on the embedding model used to convert
        text into dense vectors. These models map semantically similar text to nearby points
        in a high-dimensional vector space, enabling efficient similarity search over large
        document collections.
      </p>

      <DefinitionBlock
        title="Text Embedding for Retrieval"
        definition="A text embedding model $f: \mathcal{T} \to \mathbb{R}^d$ maps a text sequence to a dense vector such that semantically similar texts have high cosine similarity: $\text{sim}(f(t_1), f(t_2)) \approx \text{semantic\_similarity}(t_1, t_2)$."
        id="def-embedding-retrieval"
      />

      <h2 className="text-2xl font-semibold">Cosine Similarity</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The core metric for comparing embeddings is cosine similarity, which measures the angle
        between two vectors regardless of their magnitude:
      </p>
      <BlockMath math="\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \frac{\sum_{i=1}^{d} a_i b_i}{\sqrt{\sum_{i=1}^{d} a_i^2} \cdot \sqrt{\sum_{i=1}^{d} b_i^2}}" />
      <p className="text-gray-700 dark:text-gray-300">
        Values range from <InlineMath math="-1" /> (opposite) to <InlineMath math="1" /> (identical),
        with <InlineMath math="0" /> indicating orthogonality.
      </p>

      <ExampleBlock
        title="Comparing Embedding Models"
        problem="Evaluate how different embedding models capture semantic similarity for a RAG query."
        steps={[
          { formula: 'e_q = f(\\text{"How does attention work?"})', explanation: 'Embed the query using the chosen model.' },
          { formula: 'e_{d_1} = f(\\text{"Self-attention computes weighted sums."})', explanation: 'Embed a relevant document chunk.' },
          { formula: 'e_{d_2} = f(\\text{"The weather is sunny today."})', explanation: 'Embed an irrelevant document chunk.' },
          { formula: '\\text{sim}(e_q, e_{d_1}) \\gg \\text{sim}(e_q, e_{d_2})', explanation: 'A good embedding model produces much higher similarity for the relevant chunk.' },
        ]}
        id="example-embedding-comparison"
      />

      <PythonCode
        title="embedding_models_comparison.py"
        code={`# Comparing popular embedding models for RAG
import numpy as np
from sentence_transformers import SentenceTransformer

# Load different embedding models
models = {
    "all-MiniLM-L6-v2": SentenceTransformer("all-MiniLM-L6-v2"),      # 384d, fast
    "bge-large-en-v1.5": SentenceTransformer("BAAI/bge-large-en-v1.5"), # 1024d, strong
}

query = "How does the transformer attention mechanism work?"
docs = [
    "Self-attention computes a weighted sum of value vectors using query-key dot products.",
    "The transformer uses multi-head attention to attend to different subspaces.",
    "Python is a popular programming language for data science.",
]

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

for name, model in models.items():
    q_emb = model.encode(query)
    d_embs = model.encode(docs)
    print(f"\\n{name} (dim={len(q_emb)}):")
    for i, doc in enumerate(docs):
        sim = cosine_sim(q_emb, d_embs[i])
        print(f"  sim={sim:.4f} | {doc[:60]}...")

# Using OpenAI embeddings (API-based)
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",  # 1536d, cost-effective
    input=[query] + docs
)
embeddings = [e.embedding for e in response.data]
q_emb = embeddings[0]
print("\\ntext-embedding-3-small:")
for i, doc in enumerate(docs):
    sim = cosine_sim(q_emb, embeddings[i + 1])
    print(f"  sim={sim:.4f} | {doc[:60]}...")`}
        id="code-embedding-models"
      />

      <NoteBlock
        type="tip"
        title="Choosing an Embedding Model"
        content="For most RAG applications, start with a strong open-source model like BGE-large or E5-large-v2 (1024 dimensions). If latency matters, use smaller models like all-MiniLM-L6-v2 (384d). OpenAI's text-embedding-3-small offers a good balance of quality and cost for API-based systems. Check the MTEB leaderboard for current benchmarks."
        id="note-choosing-model"
      />

      <h2 className="text-2xl font-semibold">Bi-Encoder vs. Cross-Encoder</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Bi-encoders embed query and document independently, enabling pre-computation and fast
        retrieval. Cross-encoders process the query-document pair jointly for higher accuracy
        but cannot pre-compute document embeddings. In practice, RAG systems use bi-encoders
        for initial retrieval and cross-encoders for reranking.
      </p>
      <BlockMath math="\text{Bi-encoder: } s(q, d) = f(q)^T f(d) \quad \text{Cross-encoder: } s(q, d) = g([q; d])" />

      <WarningBlock
        title="Embedding Model and Query Mismatch"
        content="Always use the same embedding model for both indexing and querying. Mixing models (e.g., indexing with OpenAI embeddings but querying with Sentence Transformers) produces incompatible vector spaces and will return meaningless results. If you switch models, you must re-embed your entire corpus."
        id="warning-model-mismatch"
      />

      <NoteBlock
        type="note"
        title="Matryoshka Embeddings"
        content="Modern embedding models like text-embedding-3-small support Matryoshka representation learning, where the first k dimensions of the embedding are themselves a valid (lower-quality) embedding. This lets you trade quality for storage and speed by truncating embeddings to fewer dimensions."
        id="note-matryoshka"
      />
    </div>
  )
}
