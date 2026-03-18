import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function HybridSearch() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Hybrid Search: BM25 + Vector Search</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Vector search excels at semantic matching but can miss exact keyword matches. BM25,
        the classical information retrieval algorithm, excels at keyword matching but misses
        semantic relationships. Hybrid search combines both approaches for more robust retrieval,
        capturing both exact term matches and semantic similarity.
      </p>

      <DefinitionBlock
        title="BM25 (Best Matching 25)"
        definition="BM25 is a probabilistic ranking function that scores document relevance based on term frequency, inverse document frequency, and document length normalization: $\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{avgdl})}$"
        id="def-bm25"
      />

      <h2 className="text-2xl font-semibold">The BM25 Formula</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The BM25 scoring function has three key components:
      </p>
      <BlockMath math="\text{IDF}(t) = \ln\left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\right)" />
      <p className="text-gray-700 dark:text-gray-300">
        where <InlineMath math="N" /> is the total number of documents, <InlineMath math="n(t)" /> is
        the number of documents containing term <InlineMath math="t" />, <InlineMath math="f(t,d)" /> is
        the frequency of <InlineMath math="t" /> in document <InlineMath math="d" />,
        <InlineMath math="k_1 = 1.2" /> controls term frequency saturation,
        and <InlineMath math="b = 0.75" /> controls document length normalization.
      </p>

      <TheoremBlock
        title="Reciprocal Rank Fusion (RRF)"
        statement="Given ranked lists from multiple retrieval methods, RRF combines them using: $\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$ where $r(d)$ is the rank of document $d$ in ranked list $r$, and $k = 60$ is a constant. This produces a unified ranking without requiring score normalization."
        id="theorem-rrf"
      />

      <ExampleBlock
        title="Why Hybrid Search Outperforms Either Alone"
        problem="Show cases where BM25 and vector search each fail independently."
        steps={[
          { formula: '\\text{Query: "HNSW algorithm complexity"}', explanation: 'BM25 finds documents with exact match "HNSW". Vector search may rank general "algorithm" docs higher.' },
          { formula: '\\text{Query: "fast approximate nearest neighbor"}', explanation: 'Vector search understands this means HNSW/FAISS. BM25 may miss docs using "ANN" abbreviation.' },
          { formula: '\\text{Hybrid: } s = \\alpha \\cdot s_{\\text{vector}} + (1-\\alpha) \\cdot s_{\\text{BM25}}', explanation: 'Linear combination captures both exact and semantic matches. Alpha typically 0.5-0.7.' },
          { formula: '\\text{RRF}(d) = \\frac{1}{60 + r_{\\text{vec}}(d)} + \\frac{1}{60 + r_{\\text{BM25}}(d)}', explanation: 'Alternatively, RRF combines by rank position, avoiding score normalization.' },
        ]}
        id="example-hybrid-advantage"
      />

      <PythonCode
        title="hybrid_search.py"
        code={`# Hybrid search: BM25 + vector search with rank fusion
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document

# Sample documents
docs = [
    Document(page_content="HNSW builds a hierarchical navigable small world graph for ANN search."),
    Document(page_content="Approximate nearest neighbor search enables sub-linear query time."),
    Document(page_content="Vector databases use HNSW or IVF indexes for fast similarity search."),
    Document(page_content="BM25 is a bag-of-words retrieval function based on term frequency."),
    Document(page_content="Cosine similarity measures the angle between embedding vectors."),
]

# BM25 retriever (keyword-based)
bm25_retriever = BM25Retriever.from_documents(docs, k=3)

# Vector retriever (semantic)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(docs, embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Ensemble: combines BM25 + vector with configurable weights
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],  # 40% BM25, 60% vector
)

# Compare results
query = "HNSW algorithm for fast search"
print("BM25 results:")
for doc in bm25_retriever.invoke(query):
    print(f"  {doc.page_content[:70]}...")

print("\\nVector results:")
for doc in vector_retriever.invoke(query):
    print(f"  {doc.page_content[:70]}...")

print("\\nHybrid results:")
for doc in hybrid_retriever.invoke(query):
    print(f"  {doc.page_content[:70]}...")`}
        id="code-hybrid-search"
      />

      <NoteBlock
        type="tip"
        title="Weaviate Native Hybrid Search"
        content="Weaviate supports hybrid search natively with a single API call, combining BM25 and vector search with a configurable alpha parameter. Set alpha=0 for pure BM25, alpha=1 for pure vector, or any value in between. This avoids the overhead of running two separate retrievers."
        id="note-weaviate-hybrid"
      />

      <WarningBlock
        title="Score Normalization Challenge"
        content="BM25 scores and cosine similarities are on fundamentally different scales. BM25 scores can range from 0 to unbounded positive values, while cosine similarity is bounded in [-1, 1]. Use rank-based fusion (RRF) instead of score-based combination unless you carefully normalize scores first."
        id="warning-score-normalization"
      />

      <NoteBlock
        type="note"
        title="When to Use Hybrid Search"
        content="Hybrid search is most valuable when your corpus contains domain-specific terminology (medical codes, legal citations, product IDs) that embeddings may not capture well. For purely conversational queries over general text, vector-only search often suffices. Profile your query patterns to decide."
        id="note-when-hybrid"
      />
    </div>
  )
}
