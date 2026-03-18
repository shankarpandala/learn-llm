import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Reranking() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Cross-Encoder Reranking</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Initial retrieval using bi-encoders is fast but approximate. Reranking applies a more
        powerful cross-encoder model to re-score and reorder the top-k retrieved documents,
        significantly improving the quality of the final context passed to the LLM. This
        two-stage approach combines the efficiency of bi-encoders with the accuracy of
        cross-encoders.
      </p>

      <DefinitionBlock
        title="Cross-Encoder Reranking"
        definition="A cross-encoder processes the query-document pair jointly through a transformer: $s(q, d) = \sigma(W \cdot \text{BERT}([q; \text{SEP}; d]) + b)$, producing a relevance score in $[0, 1]$. Unlike bi-encoders, cross-encoders capture fine-grained query-document interactions but cannot pre-compute document representations."
        id="def-reranking"
      />

      <h2 className="text-2xl font-semibold">Two-Stage Retrieval</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The two-stage retrieve-then-rerank pipeline first casts a wide net with fast bi-encoder
        retrieval, then refines with expensive but accurate cross-encoder scoring:
      </p>
      <BlockMath math="\text{Stage 1: } D_{20} = \text{top-}20(\text{bi-encoder}(q, \mathcal{D}))" />
      <BlockMath math="\text{Stage 2: } D_5 = \text{top-}5(\text{cross-encoder}(q, D_{20}))" />

      <ExampleBlock
        title="Reranking Improves Precision"
        problem="Show how reranking reorders retrieved documents for better relevance."
        steps={[
          { formula: '\\text{Retrieve top-20 with bi-encoder}', explanation: 'Fast initial retrieval ensures high recall but may include irrelevant results.' },
          { formula: 's_i = \\text{CrossEncoder}(q, d_i) \\quad \\forall d_i \\in D_{20}', explanation: 'Score each candidate with the cross-encoder for fine-grained relevance.' },
          { formula: 'D_5 = \\text{sort}(D_{20}, \\text{key}=s_i)[:5]', explanation: 'Take only the top 5 after reranking - these are much more relevant.' },
          { formula: '\\text{Precision@5 increases from 0.4 to 0.8 typically}', explanation: 'Reranking commonly doubles precision at small k values.' },
        ]}
        id="example-reranking"
      />

      <PythonCode
        title="cross_encoder_reranking.py"
        code={`# Two-stage retrieval with cross-encoder reranking
from sentence_transformers import CrossEncoder
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Stage 1: Bi-encoder retrieval (fast, approximate)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents, embeddings)
initial_results = vectorstore.similarity_search("How does LoRA work?", k=20)

# Stage 2: Cross-encoder reranking (slow, accurate)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "How does LoRA work?"
pairs = [(query, doc.page_content) for doc in initial_results]
scores = reranker.predict(pairs)

# Sort by cross-encoder score
ranked = sorted(
    zip(initial_results, scores), key=lambda x: x[1], reverse=True
)

print("After reranking:")
for doc, score in ranked[:5]:
    print(f"  Score: {score:.4f} | {doc.page_content[:60]}...")

# Using Cohere Rerank API (production-grade)
import cohere

co = cohere.Client("your-api-key")

rerank_response = co.rerank(
    model="rerank-english-v3.0",
    query="How does LoRA work?",
    documents=[doc.page_content for doc in initial_results],
    top_n=5,
)

print("\\nCohere reranked:")
for result in rerank_response.results:
    print(f"  Score: {result.relevance_score:.4f} | Index: {result.index}")
    print(f"  {initial_results[result.index].page_content[:60]}...")`}
        id="code-reranking"
      />

      <PythonCode
        title="langchain_reranking.py"
        code={`# Reranking integrated into a LangChain RAG pipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Base retriever: cast a wide net
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Cross-encoder compressor
model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=model, top_n=5)

# Compressed retriever: retrieve 20, rerank to top 5
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# Use in a RAG chain
results = reranking_retriever.invoke("How does LoRA reduce parameters?")
for doc in results:
    print(f"  {doc.page_content[:80]}...")`}
        id="code-langchain-rerank"
      />

      <NoteBlock
        type="tip"
        title="Reranking Budget"
        content="Cross-encoders process each query-document pair independently, so reranking 20 documents requires 20 forward passes. Keep the initial retrieval set to 20-50 documents. Beyond that, the latency cost outweighs the relevance gains. Cohere's rerank API can process up to 1000 documents efficiently due to batching optimizations."
        id="note-budget"
      />

      <WarningBlock
        title="Reranker Training Domain Matters"
        content="Most open-source cross-encoders are trained on web search data (MS MARCO). They may underperform on specialized domains like legal, medical, or code. For domain-specific applications, consider fine-tuning a cross-encoder on your domain data or using a general-purpose LLM as a reranker with appropriate prompting."
        id="warning-domain"
      />
    </div>
  )
}
