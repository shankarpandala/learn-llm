import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ChunkingStrategies() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Chunking Strategies: Fixed, Recursive, and Semantic</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Chunking is the process of breaking documents into smaller pieces for embedding and
        retrieval. The chunking strategy directly affects retrieval quality: chunks that are
        too large dilute relevance, while chunks that are too small lose context. Finding the
        right granularity is one of the most impactful decisions in RAG system design.
      </p>

      <DefinitionBlock
        title="Document Chunking"
        definition="Chunking partitions a document $D$ into a sequence of chunks $\{c_1, c_2, \dots, c_n\}$ where each chunk $c_i$ has size $|c_i| \leq S$ tokens, with optional overlap $O$ such that $c_i \cap c_{i+1} = O$ tokens. The goal is to maximize semantic coherence within each chunk."
        id="def-chunking"
      />

      <h2 className="text-2xl font-semibold">Fixed-Size Chunking</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The simplest approach splits text into chunks of a fixed number of characters or tokens,
        with a specified overlap to preserve context across boundaries. Fast and deterministic,
        but may split sentences or paragraphs mid-thought.
      </p>

      <h2 className="text-2xl font-semibold">Recursive Character Splitting</h2>
      <p className="text-gray-700 dark:text-gray-300">
        LangChain's default strategy attempts to split on natural boundaries (paragraphs, then
        sentences, then words) before falling back to character-level splits. This produces
        more semantically coherent chunks than fixed-size splitting.
      </p>

      <ExampleBlock
        title="Chunk Size Trade-offs"
        problem="Analyze how chunk size affects retrieval precision and recall."
        steps={[
          { formula: '\\text{Small chunks (}|c| = 128\\text{ tokens)}', explanation: 'High precision: retrieved chunks are highly relevant. Low recall: may miss surrounding context needed for generation.' },
          { formula: '\\text{Large chunks (}|c| = 1024\\text{ tokens)}', explanation: 'High recall: captures full context. Low precision: irrelevant content dilutes the embedding.' },
          { formula: '\\text{Overlap } O = 0.1 \\times S', explanation: 'A 10-20% overlap helps preserve context across chunk boundaries.' },
          { formula: '\\text{Optimal: } |c| \\approx 256-512 \\text{ tokens}', explanation: 'Empirically, 256-512 tokens balances precision and recall for most use cases.' },
        ]}
        id="example-chunk-size"
      />

      <PythonCode
        title="chunking_strategies.py"
        code={`# Comparing chunking strategies
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

document = """
# Transformer Architecture

The transformer architecture was introduced in "Attention Is All You Need" (2017).
It relies entirely on self-attention mechanisms, dispensing with recurrence and
convolutions. The model consists of an encoder and decoder, each with multiple layers.

## Self-Attention

Self-attention computes a weighted sum of all positions in the input sequence.
For each position, it generates query, key, and value vectors. The attention
weights are computed as softmax(QK^T / sqrt(d_k)).

## Multi-Head Attention

Rather than computing a single attention function, multi-head attention projects
queries, keys, and values h times with different learned linear projections.
This allows the model to attend to information from different representation
subspaces at different positions.
"""

# 1. Fixed-size character splitting
fixed_splitter = CharacterTextSplitter(
    chunk_size=200, chunk_overlap=50, separator="\\n"
)
fixed_chunks = fixed_splitter.split_text(document)
print(f"Fixed: {len(fixed_chunks)} chunks")
for i, c in enumerate(fixed_chunks):
    print(f"  Chunk {i}: {len(c)} chars | {c[:50]}...")

# 2. Recursive splitting (respects structure)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\\n## ", "\\n# ", "\\n\\n", "\\n", ". ", " ", ""],
)
recursive_chunks = recursive_splitter.split_text(document)
print(f"\\nRecursive: {len(recursive_chunks)} chunks")
for i, c in enumerate(recursive_chunks):
    print(f"  Chunk {i}: {len(c)} chars | {c[:50]}...")

# 3. Token-based splitting (for precise token budgets)
token_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20)
token_chunks = token_splitter.split_text(document)
print(f"\\nToken-based: {len(token_chunks)} chunks")`}
        id="code-chunking-strategies"
      />

      <h2 className="text-2xl font-semibold">Semantic Chunking</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Semantic chunking uses embeddings to detect topic boundaries. It computes the similarity
        between consecutive sentences and splits where similarity drops below a threshold,
        producing chunks that are semantically self-contained.
      </p>

      <PythonCode
        title="semantic_chunking.py"
        code={`# Semantic chunking using embedding similarity
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Splits where embedding similarity between sentences drops
semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=70,  # split at 70th percentile of dissimilarity
)

chunks = semantic_splitter.split_text(document)
print(f"Semantic: {len(chunks)} chunks")
for i, c in enumerate(chunks):
    print(f"  Chunk {i}: {len(c)} chars")`}
        id="code-semantic-chunking"
      />

      <NoteBlock
        type="intuition"
        title="Think of Chunks as Index Cards"
        content="Imagine creating index cards for a textbook. Each card should contain one complete idea with enough context to be useful on its own. If a card has half a sentence, it is useless. If it has three pages, it is too unfocused. Chunking is the art of creating index cards that are just right."
        id="note-index-cards"
      />

      <WarningBlock
        title="Chunk Size Must Match Embedding Model"
        content="Embedding models have maximum input lengths (e.g., 512 tokens for many sentence-transformers, 8192 for text-embedding-3-small). Chunks exceeding this limit will be silently truncated, losing information. Always verify your chunk size is within your embedding model's context window."
        id="warning-chunk-model-match"
      />
    </div>
  )
}
