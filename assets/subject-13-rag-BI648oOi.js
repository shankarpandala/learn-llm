import{j as e}from"./vendor-DWbzdFaj.js";import{r as t}from"./vendor-katex-BYl39Yo6.js";import{D as a,E as i,P as r,N as n,W as o,T as s}from"./subject-01-text-fundamentals-DG6tAvii.js";function l(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Why RAG: Motivation and Hallucination Reduction"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Large language models store knowledge implicitly in their parameters, but this knowledge can be outdated, incomplete, or simply wrong. Retrieval-Augmented Generation (RAG) addresses these limitations by grounding model outputs in retrieved evidence, dramatically reducing hallucinations and enabling up-to-date, verifiable responses."}),e.jsx(a,{title:"Retrieval-Augmented Generation (RAG)",definition:"RAG is a technique that augments an LLM's input prompt with relevant documents retrieved from an external knowledge base, allowing the model to generate responses grounded in specific, verifiable sources rather than relying solely on parametric memory.",id:"def-rag"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The Hallucination Problem"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"LLMs generate plausible-sounding text even when they lack factual knowledge. This happens because the training objective optimizes for next-token probability, not factual accuracy. The probability of a hallucinated output can be modeled as:"}),e.jsx(t.BlockMath,{math:"P(\\text{hallucination}) = 1 - P(\\text{fact} \\in \\theta)"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["where ",e.jsx(t.InlineMath,{math:"\\theta"})," represents the model's parametric knowledge. RAG reduces this by conditioning on retrieved context ",e.jsx(t.InlineMath,{math:"C"}),":"]}),e.jsx(t.BlockMath,{math:"P(y \\mid x) = \\sum_{d \\in \\mathcal{D}} P(y \\mid x, d) \\cdot P(d \\mid x)"}),e.jsx(i,{title:"Parametric vs. RAG-Augmented Response",problem:"Ask an LLM about a company's Q3 2025 earnings without and with RAG.",steps:[{formula:"\\text{Without RAG: } P(y \\mid x) = P(y \\mid x; \\theta)",explanation:"The model relies on training data, which may predate Q3 2025."},{formula:"\\text{With RAG: } P(y \\mid x, C) = P(y \\mid x, \\{d_1, d_2, \\dots\\}; \\theta)",explanation:"Retrieved earnings reports provide factual grounding."},{formula:"\\text{Faithfulness} = \\frac{|\\text{claims supported by } C|}{|\\text{total claims}|}",explanation:"RAG increases faithfulness by providing verifiable source material."}],id:"example-parametric-vs-rag"}),e.jsx(r,{title:"basic_rag_motivation.py",code:`# Demonstrating why RAG matters: grounding LLM responses
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Without RAG - model may hallucinate recent facts
question = "What were Acme Corp's Q3 2025 revenue figures?"
naive_response = llm.invoke(question)
print("Without RAG:", naive_response.content)
# Likely hallucinates specific numbers

# With RAG - ground the response in retrieved context
context = """
Acme Corp Q3 2025 Earnings Report:
- Revenue: $4.2B (up 15% YoY)
- Net Income: $890M
- Guidance: $4.5B for Q4 2025
"""

rag_prompt = ChatPromptTemplate.from_template(
    """Answer based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't know."

Context: {context}
Question: {question}
Answer:"""
)

chain = rag_prompt | llm
rag_response = chain.invoke({"context": context, "question": question})
print("With RAG:", rag_response.content)
# Accurate, grounded in the provided document`,id:"code-rag-motivation"}),e.jsx(n,{type:"intuition",title:"RAG as Open-Book vs. Closed-Book Exam",content:"Think of a vanilla LLM as taking a closed-book exam: it can only rely on what it memorized during training. RAG transforms this into an open-book exam where the model can look up relevant reference material before answering. The model still needs reasoning ability, but it no longer needs to memorize every fact.",id:"note-open-book"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Key Benefits of RAG"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"RAG provides several advantages over pure parametric approaches: knowledge can be updated without retraining, responses are traceable to source documents, and domain-specific knowledge can be injected without fine-tuning. The cost of updating a vector store is orders of magnitude lower than retraining a model."}),e.jsx(o,{title:"RAG Is Not a Silver Bullet",content:"RAG only helps when the answer exists in the knowledge base and is retrievable. Poor chunking, bad embeddings, or irrelevant retrieval can actually degrade performance compared to a well-trained base model. Always evaluate whether retrieved context improves or confuses the generation.",id:"warning-not-silver-bullet"}),e.jsx(n,{type:"historical",title:"Origins of RAG",content:"The RAG framework was introduced by Lewis et al. at Facebook AI Research in 2020. The original paper combined a pre-trained seq2seq model (BART) with a dense retriever (DPR) and showed significant improvements on knowledge-intensive tasks like open-domain QA. The approach has since been adopted industry-wide as a standard pattern for production LLM applications.",id:"note-rag-history"})]})}const j=Object.freeze(Object.defineProperty({__proto__:null,default:l},Symbol.toStringTag,{value:"Module"}));function c(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"The Retrieve-Augment-Generate Pipeline"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"A RAG system consists of three core stages: retrieving relevant documents from a knowledge base, augmenting the user query with this context, and generating a grounded response. Understanding each stage and how they connect is essential for building effective RAG applications."}),e.jsx(a,{title:"RAG Pipeline",definition:"The RAG pipeline transforms a user query $q$ through three stages: (1) Retrieve: find documents $D = \\{d_1, \\dots, d_k\\}$ where $d_i = \\arg\\max_{d \\in \\mathcal{D}} \\text{sim}(e_q, e_d)$, (2) Augment: construct prompt $p = f(q, D)$, (3) Generate: produce answer $a = \\text{LLM}(p)$.",id:"def-pipeline"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Stage 1: Retrieval"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The retrieval stage converts the query into an embedding and searches a vector database for the most similar document chunks. The similarity is typically measured using cosine similarity or dot product:"}),e.jsx(t.BlockMath,{math:"\\text{sim}(q, d) = \\frac{e_q \\cdot e_d}{\\|e_q\\| \\|e_d\\|}"}),e.jsx(i,{title:"End-to-End RAG Pipeline",problem:"Build a RAG pipeline that answers questions about a technical document.",steps:[{formula:"\\text{Chunk: } \\mathcal{D} \\to \\{c_1, c_2, \\dots, c_n\\}",explanation:"Split documents into overlapping chunks of manageable size."},{formula:"e_{c_i} = \\text{Embed}(c_i) \\in \\mathbb{R}^d",explanation:"Compute dense embeddings for each chunk."},{formula:"D_k = \\text{top-}k(\\text{sim}(e_q, e_{c_i}))",explanation:"Retrieve the k most similar chunks to the query embedding."},{formula:"a = \\text{LLM}(\\text{prompt}(q, D_k))",explanation:"Generate the answer conditioned on query and retrieved context."}],id:"example-pipeline"}),e.jsx(r,{title:"full_rag_pipeline.py",code:`# Complete RAG pipeline with LangChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Step 1: Prepare documents
documents = [
    "Transformers use self-attention to process sequences in parallel.",
    "The attention mechanism computes weighted sums over all positions.",
    "Layer normalization stabilizes training of deep transformer networks.",
    "Positional encodings provide sequence order information to the model.",
    "Multi-head attention allows attending to different representation subspaces.",
]

# Step 2: Chunk and embed (here docs are already small)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50
)
splits = text_splitter.create_documents(documents)

# Step 3: Store in vector database
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 4: Build generation chain
prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the provided context.

Context: {context}
Question: {question}
Answer:"""
)

def format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

# Step 5: Query
answer = rag_chain.invoke("How does self-attention work?")
print(answer)`,id:"code-full-pipeline"}),e.jsx(n,{type:"tip",title:"Choosing the Right k",content:"The number of retrieved documents k is a critical hyperparameter. Too few and you miss relevant context; too many and you overwhelm the LLM with noise or exceed context limits. Start with k=3-5 for most applications and tune based on retrieval quality metrics.",id:"note-choosing-k"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Stage 2: Augmentation"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The augmentation stage constructs the final prompt by combining the user query with retrieved context. Prompt engineering here is crucial: the template must instruct the model to use the context faithfully, cite sources when possible, and acknowledge when the context is insufficient."}),e.jsx(o,{title:"Context Window Limits",content:"Retrieved chunks must fit within the model's context window along with the system prompt, query, and space for the response. A model with 8K context can practically use about 4-5K tokens for retrieved context. Always calculate your token budget before designing the pipeline.",id:"warning-context-limits"}),e.jsx(n,{type:"note",title:"Indexing Pipeline vs. Query Pipeline",content:"RAG has two distinct pipelines: the indexing pipeline (offline) handles document loading, chunking, embedding, and storage. The query pipeline (online) handles query embedding, retrieval, augmentation, and generation. Separating these concerns allows independent optimization of each stage.",id:"note-two-pipelines"})]})}const S=Object.freeze(Object.defineProperty({__proto__:null,default:c},Symbol.toStringTag,{value:"Module"}));function d(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Embedding Models for Retrieval"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The quality of a RAG system depends heavily on the embedding model used to convert text into dense vectors. These models map semantically similar text to nearby points in a high-dimensional vector space, enabling efficient similarity search over large document collections."}),e.jsx(a,{title:"Text Embedding for Retrieval",definition:"A text embedding model $f: \\mathcal{T} \\to \\mathbb{R}^d$ maps a text sequence to a dense vector such that semantically similar texts have high cosine similarity: $\\text{sim}(f(t_1), f(t_2)) \\approx \\text{semantic\\_similarity}(t_1, t_2)$.",id:"def-embedding-retrieval"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Cosine Similarity"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The core metric for comparing embeddings is cosine similarity, which measures the angle between two vectors regardless of their magnitude:"}),e.jsx(t.BlockMath,{math:"\\cos(\\theta) = \\frac{\\mathbf{a} \\cdot \\mathbf{b}}{\\|\\mathbf{a}\\| \\|\\mathbf{b}\\|} = \\frac{\\sum_{i=1}^{d} a_i b_i}{\\sqrt{\\sum_{i=1}^{d} a_i^2} \\cdot \\sqrt{\\sum_{i=1}^{d} b_i^2}}"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["Values range from ",e.jsx(t.InlineMath,{math:"-1"})," (opposite) to ",e.jsx(t.InlineMath,{math:"1"})," (identical), with ",e.jsx(t.InlineMath,{math:"0"})," indicating orthogonality."]}),e.jsx(i,{title:"Comparing Embedding Models",problem:"Evaluate how different embedding models capture semantic similarity for a RAG query.",steps:[{formula:'e_q = f(\\text{"How does attention work?"})',explanation:"Embed the query using the chosen model."},{formula:'e_{d_1} = f(\\text{"Self-attention computes weighted sums."})',explanation:"Embed a relevant document chunk."},{formula:'e_{d_2} = f(\\text{"The weather is sunny today."})',explanation:"Embed an irrelevant document chunk."},{formula:"\\text{sim}(e_q, e_{d_1}) \\gg \\text{sim}(e_q, e_{d_2})",explanation:"A good embedding model produces much higher similarity for the relevant chunk."}],id:"example-embedding-comparison"}),e.jsx(r,{title:"embedding_models_comparison.py",code:`# Comparing popular embedding models for RAG
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
    print(f"  sim={sim:.4f} | {doc[:60]}...")`,id:"code-embedding-models"}),e.jsx(n,{type:"tip",title:"Choosing an Embedding Model",content:"For most RAG applications, start with a strong open-source model like BGE-large or E5-large-v2 (1024 dimensions). If latency matters, use smaller models like all-MiniLM-L6-v2 (384d). OpenAI's text-embedding-3-small offers a good balance of quality and cost for API-based systems. Check the MTEB leaderboard for current benchmarks.",id:"note-choosing-model"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Bi-Encoder vs. Cross-Encoder"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Bi-encoders embed query and document independently, enabling pre-computation and fast retrieval. Cross-encoders process the query-document pair jointly for higher accuracy but cannot pre-compute document embeddings. In practice, RAG systems use bi-encoders for initial retrieval and cross-encoders for reranking."}),e.jsx(t.BlockMath,{math:"\\text{Bi-encoder: } s(q, d) = f(q)^T f(d) \\quad \\text{Cross-encoder: } s(q, d) = g([q; d])"}),e.jsx(o,{title:"Embedding Model and Query Mismatch",content:"Always use the same embedding model for both indexing and querying. Mixing models (e.g., indexing with OpenAI embeddings but querying with Sentence Transformers) produces incompatible vector spaces and will return meaningless results. If you switch models, you must re-embed your entire corpus.",id:"warning-model-mismatch"}),e.jsx(n,{type:"note",title:"Matryoshka Embeddings",content:"Modern embedding models like text-embedding-3-small support Matryoshka representation learning, where the first k dimensions of the embedding are themselves a valid (lower-quality) embedding. This lets you trade quality for storage and speed by truncating embeddings to fewer dimensions.",id:"note-matryoshka"})]})}const T=Object.freeze(Object.defineProperty({__proto__:null,default:d},Symbol.toStringTag,{value:"Module"}));function m(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Vector Databases: Chroma, Pinecone, Weaviate, pgvector"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Vector databases are specialized storage systems optimized for storing, indexing, and querying high-dimensional embedding vectors. They use approximate nearest neighbor (ANN) algorithms to achieve sub-linear search time over millions or billions of vectors."}),e.jsx(a,{title:"Approximate Nearest Neighbor (ANN) Search",definition:"ANN search finds vectors approximately closest to a query vector $q$ in time $O(\\log n)$ instead of exact search's $O(n)$, using index structures like HNSW (Hierarchical Navigable Small World graphs) or IVF (Inverted File Index). The trade-off is controlled by a recall parameter.",id:"def-ann"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Index Structures"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The efficiency of vector search depends on the indexing algorithm. HNSW builds a multi-layer graph where each node connects to its nearest neighbors, enabling greedy traversal that approximates exact search:"}),e.jsx(t.BlockMath,{math:"\\text{HNSW complexity: } O(\\log n) \\text{ query time, } O(n \\log n) \\text{ build time}"}),e.jsx(i,{title:"Vector Database Selection",problem:"Choose a vector database based on requirements for a RAG application.",steps:[{formula:"\\text{Chroma: embedded, } < 1M \\text{ vectors}",explanation:"Ideal for prototyping, local development, and small-scale applications. Runs in-process."},{formula:"\\text{Pinecone: managed, } > 1M \\text{ vectors}",explanation:"Fully managed cloud service with automatic scaling. Best for production without infra overhead."},{formula:"\\text{Weaviate: self-hosted, hybrid search}",explanation:"Supports both vector and keyword search natively. Good for complex filtering requirements."},{formula:"\\text{pgvector: existing Postgres, } < 10M \\text{ vectors}",explanation:"Extension for PostgreSQL. Best when you already use Postgres and want unified storage."}],id:"example-db-selection"}),e.jsx(r,{title:"vector_databases.py",code:`# Chroma - lightweight, embedded vector database
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
results = cur.fetchall()`,id:"code-vector-dbs"}),e.jsx(n,{type:"tip",title:"Start Simple, Scale Later",content:"Begin with Chroma for prototyping - it requires zero configuration and runs in-process. Move to pgvector if you already use PostgreSQL, or to a managed service like Pinecone when you need production-grade scaling. Premature infrastructure complexity is a common pitfall in RAG projects.",id:"note-start-simple"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Distance Metrics"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Vector databases support multiple distance metrics. The choice depends on your embedding model and use case:"}),e.jsx(t.BlockMath,{math:"\\text{Cosine: } 1 - \\frac{a \\cdot b}{\\|a\\|\\|b\\|} \\quad \\text{L2: } \\|a - b\\|_2 \\quad \\text{Dot: } -a \\cdot b"}),e.jsx(o,{title:"Index Building Takes Time",content:"HNSW index construction is O(n log n) and can take hours for millions of vectors. Plan for index build time in your deployment pipeline. Some databases like Pinecone handle this transparently, but self-hosted solutions like pgvector or Weaviate require explicit index management and tuning of parameters like ef_construction and M.",id:"warning-index-time"}),e.jsx(n,{type:"note",title:"Metadata Filtering",content:"All major vector databases support metadata filtering alongside vector search. This enables queries like 'find similar documents from 2024 in the finance domain.' Pre-filtering (before ANN search) is more efficient but may miss results; post-filtering (after ANN search) is more accurate but may return fewer results than requested.",id:"note-metadata"})]})}const M=Object.freeze(Object.defineProperty({__proto__:null,default:m},Symbol.toStringTag,{value:"Module"}));function u(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Chunking Strategies: Fixed, Recursive, and Semantic"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Chunking is the process of breaking documents into smaller pieces for embedding and retrieval. The chunking strategy directly affects retrieval quality: chunks that are too large dilute relevance, while chunks that are too small lose context. Finding the right granularity is one of the most impactful decisions in RAG system design."}),e.jsx(a,{title:"Document Chunking",definition:"Chunking partitions a document $D$ into a sequence of chunks $\\{c_1, c_2, \\dots, c_n\\}$ where each chunk $c_i$ has size $|c_i| \\leq S$ tokens, with optional overlap $O$ such that $c_i \\cap c_{i+1} = O$ tokens. The goal is to maximize semantic coherence within each chunk.",id:"def-chunking"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Fixed-Size Chunking"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The simplest approach splits text into chunks of a fixed number of characters or tokens, with a specified overlap to preserve context across boundaries. Fast and deterministic, but may split sentences or paragraphs mid-thought."}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Recursive Character Splitting"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"LangChain's default strategy attempts to split on natural boundaries (paragraphs, then sentences, then words) before falling back to character-level splits. This produces more semantically coherent chunks than fixed-size splitting."}),e.jsx(i,{title:"Chunk Size Trade-offs",problem:"Analyze how chunk size affects retrieval precision and recall.",steps:[{formula:"\\text{Small chunks (}|c| = 128\\text{ tokens)}",explanation:"High precision: retrieved chunks are highly relevant. Low recall: may miss surrounding context needed for generation."},{formula:"\\text{Large chunks (}|c| = 1024\\text{ tokens)}",explanation:"High recall: captures full context. Low precision: irrelevant content dilutes the embedding."},{formula:"\\text{Overlap } O = 0.1 \\times S",explanation:"A 10-20% overlap helps preserve context across chunk boundaries."},{formula:"\\text{Optimal: } |c| \\approx 256-512 \\text{ tokens}",explanation:"Empirically, 256-512 tokens balances precision and recall for most use cases."}],id:"example-chunk-size"}),e.jsx(r,{title:"chunking_strategies.py",code:`# Comparing chunking strategies
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
print(f"\\nToken-based: {len(token_chunks)} chunks")`,id:"code-chunking-strategies"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Semantic Chunking"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Semantic chunking uses embeddings to detect topic boundaries. It computes the similarity between consecutive sentences and splits where similarity drops below a threshold, producing chunks that are semantically self-contained."}),e.jsx(r,{title:"semantic_chunking.py",code:`# Semantic chunking using embedding similarity
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
    print(f"  Chunk {i}: {len(c)} chars")`,id:"code-semantic-chunking"}),e.jsx(n,{type:"intuition",title:"Think of Chunks as Index Cards",content:"Imagine creating index cards for a textbook. Each card should contain one complete idea with enough context to be useful on its own. If a card has half a sentence, it is useless. If it has three pages, it is too unfocused. Chunking is the art of creating index cards that are just right.",id:"note-index-cards"}),e.jsx(o,{title:"Chunk Size Must Match Embedding Model",content:"Embedding models have maximum input lengths (e.g., 512 tokens for many sentence-transformers, 8192 for text-embedding-3-small). Chunks exceeding this limit will be silently truncated, losing information. Always verify your chunk size is within your embedding model's context window.",id:"warning-chunk-model-match"})]})}const C=Object.freeze(Object.defineProperty({__proto__:null,default:u},Symbol.toStringTag,{value:"Module"}));function p(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Hierarchical Indexing: Parent-Child and Summary Indexing"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Flat chunking loses the hierarchical structure of documents. Hierarchical indexing preserves parent-child relationships between chunks, enabling retrieval of fine-grained passages while providing broader context. Summary indexing creates high-level summaries that enable top-down navigation of large document collections."}),e.jsx(a,{title:"Parent-Child Indexing",definition:"Parent-child indexing creates two levels of chunks: small child chunks $c_i$ used for precise retrieval, and larger parent chunks $P(c_i)$ returned as context. The retrieval matches on children but returns the parent: $\\text{context} = P(\\arg\\max_{c_i} \\text{sim}(q, c_i))$.",id:"def-parent-child"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The Small-to-Big Retrieval Pattern"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Small chunks embed more precisely, but the LLM needs broader context to generate good answers. The small-to-big pattern resolves this tension: embed small chunks for retrieval accuracy, then expand to their parent chunk for generation context."}),e.jsx(i,{title:"Parent-Child Retrieval Flow",problem:"Retrieve precise matches but return full-context parent chunks.",steps:[{formula:"D \\to \\{P_1, P_2, \\dots\\} \\to \\{c_{1,1}, c_{1,2}, \\dots, c_{2,1}, \\dots\\}",explanation:"Split document into parent chunks (e.g., 1024 tokens), then child chunks (e.g., 256 tokens)."},{formula:"e_{c_{i,j}} = \\text{Embed}(c_{i,j})",explanation:"Embed only the child chunks for precise similarity matching."},{formula:"c^* = \\arg\\max_{c_{i,j}} \\text{sim}(e_q, e_{c_{i,j}})",explanation:"Find the best-matching child chunk."},{formula:"\\text{Return } P(c^*)",explanation:"Return the parent chunk containing the matched child for richer context."}],id:"example-parent-child"}),e.jsx(r,{title:"parent_child_indexing.py",code:`# Parent-child retrieval with LlamaIndex
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    HierarchicalNodeParser,
)
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore

# Create hierarchical chunks at multiple levels
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]  # Parent -> Child -> Leaf
)

documents = [Document(text="...long document text...")]
nodes = node_parser.get_nodes_from_documents(documents)

# Leaf nodes are embedded; parent nodes stored for retrieval
leaf_nodes = [n for n in nodes if n.child_nodes is None or len(n.child_nodes) == 0]

# Build index on leaf nodes only
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

index = VectorStoreIndex(leaf_nodes)
base_retriever = index.as_retriever(similarity_top_k=6)

# AutoMergingRetriever: if enough child nodes match,
# merge up to parent for broader context
retriever = AutoMergingRetriever(
    base_retriever, docstore, simple_ratio_thresh=0.4
)

results = retriever.retrieve("How does self-attention work?")
for r in results:
    print(f"Score: {r.score:.4f} | Size: {len(r.text)} chars")
    print(f"  {r.text[:100]}...\\n")`,id:"code-parent-child"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Summary Indexing"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Summary indexing generates an LLM-created summary of each document or section, then embeds these summaries as a top-level index. Queries first match against summaries to identify relevant documents, then drill into the document chunks for detailed retrieval."}),e.jsx(r,{title:"summary_indexing.py",code:`# Summary index for top-down document navigation
from llama_index.core import (
    SummaryIndex,
    VectorStoreIndex,
    Document,
)
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

documents = [
    Document(text="Chapter 1: Self-attention allows each position..."),
    Document(text="Chapter 2: Training transformers requires..."),
    Document(text="Chapter 3: Inference optimization techniques..."),
]

# Summary index: LLM generates summaries for routing
summary_index = SummaryIndex.from_documents(documents)
summary_engine = summary_index.as_query_engine(
    response_mode="tree_summarize"
)

# Vector index: for precise retrieval within chapters
vector_index = VectorStoreIndex.from_documents(documents)
vector_engine = vector_index.as_query_engine(similarity_top_k=3)

# Router decides which index to use based on query type
tools = [
    QueryEngineTool.from_defaults(
        query_engine=summary_engine,
        description="Useful for high-level questions about the overall content.",
    ),
    QueryEngineTool.from_defaults(
        query_engine=vector_engine,
        description="Useful for specific technical questions.",
    ),
]

router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=tools,
)
response = router_engine.query("What topics are covered?")
print(response)`,id:"code-summary-index"}),e.jsx(n,{type:"intuition",title:"Think of a Library Catalog",content:"Summary indexing works like a library catalog. You first check the catalog (summaries) to find which books (documents) are relevant, then go to those specific books and look through their pages (chunks). Without the catalog, you would need to flip through every page of every book.",id:"note-library-catalog"}),e.jsx(o,{title:"Summary Quality Matters",content:"Summary indexing is only as good as the generated summaries. If a summary misses a key topic from the document, queries about that topic will fail to route correctly. Always validate summary coverage against the source documents, especially for technical content where nuances matter.",id:"warning-summary-quality"})]})}const L=Object.freeze(Object.defineProperty({__proto__:null,default:p},Symbol.toStringTag,{value:"Module"}));function h(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Metadata Enrichment and Filtering"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Pure vector similarity often retrieves semantically similar but contextually wrong documents. Metadata filtering adds structured constraints to vector search, allowing you to narrow results by source, date, category, or any custom attribute. Enriching chunks with metadata during indexing is crucial for production RAG systems."}),e.jsx(a,{title:"Metadata-Filtered Retrieval",definition:"Metadata-filtered retrieval combines vector similarity with structured constraints: $D_k = \\text{top-}k\\{d \\in \\mathcal{D} : \\text{filter}(d.\\text{metadata}) = \\text{true}\\}$ ranked by $\\text{sim}(e_q, e_d)$. This restricts the search space before or after ANN lookup.",id:"def-metadata-filtering"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Types of Metadata"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Effective metadata falls into several categories: structural (source file, section, page number), temporal (creation date, last modified), categorical (topic, department, document type), and derived (LLM-generated summaries, keywords, entities)."}),e.jsx(i,{title:"Metadata Enrichment Pipeline",problem:"Enrich document chunks with metadata for filtered retrieval.",steps:[{formula:"\\text{Structural: } \\{\\text{source, section, page}\\}",explanation:"Extract from document structure: file path, headers, page numbers."},{formula:"\\text{Temporal: } \\{\\text{created, modified}\\}",explanation:"Capture timestamps for recency filtering."},{formula:"\\text{LLM-derived: } \\{\\text{summary, entities, keywords}\\}",explanation:"Use an LLM to extract topics, named entities, and key phrases from each chunk."},{formula:"\\text{Query: } \\text{sim}(q, d) \\text{ WHERE } d.\\text{year} \\geq 2024",explanation:"Combine vector search with metadata filters at query time."}],id:"example-metadata-enrichment"}),e.jsx(r,{title:"metadata_enrichment.py",code:`# Metadata enrichment and filtered retrieval with Chroma
import chromadb
from datetime import datetime
from langchain_openai import ChatOpenAI

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("enriched_docs")

# Enrich chunks with metadata during indexing
def enrich_chunk(text, source_file, page_num):
    """Add structural and LLM-derived metadata to a chunk."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # LLM-derived metadata
    response = llm.invoke(
        f"Extract 3-5 keywords and a one-sentence summary from:\\n{text}"
    )
    keywords = response.content  # Parse as needed

    return {
        "source": source_file,
        "page": page_num,
        "indexed_at": datetime.now().isoformat(),
        "char_count": len(text),
        "keywords": keywords,
    }

# Index with rich metadata
chunks = [
    {"text": "Self-attention computes Q, K, V matrices...", "source": "transformer.pdf", "page": 5},
    {"text": "LoRA reduces trainable parameters by...", "source": "finetuning.pdf", "page": 12},
    {"text": "The 2024 scaling laws paper shows...", "source": "scaling.pdf", "page": 1},
]

for i, chunk in enumerate(chunks):
    metadata = enrich_chunk(chunk["text"], chunk["source"], chunk["page"])
    collection.add(
        documents=[chunk["text"]],
        ids=[f"chunk_{i}"],
        metadatas=[metadata],
    )

# Filtered queries
results = collection.query(
    query_texts=["How does attention work?"],
    n_results=5,
    where={"source": "transformer.pdf"},  # Only search transformer docs
)

# Complex filters with $and / $or
results = collection.query(
    query_texts=["recent scaling research"],
    n_results=5,
    where={
        "$and": [
            {"source": {"$ne": "finetuning.pdf"}},
            {"page": {"$lte": 10}},
        ]
    },
)
print("Filtered results:", results["documents"])`,id:"code-metadata-enrichment"}),e.jsx(n,{type:"tip",title:"Self-Query Retrieval",content:"LangChain's SelfQueryRetriever uses an LLM to automatically extract metadata filters from natural language queries. A query like 'What did the 2024 paper say about scaling?' is decomposed into a vector search for 'scaling' with a metadata filter for year=2024. This eliminates the need for users to specify filters manually.",id:"note-self-query"}),e.jsx(r,{title:"self_query_retriever.py",code:`# Self-query: LLM automatically extracts filters from queries
from langchain.retrievers import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Define filterable metadata fields
metadata_field_info = [
    AttributeInfo(name="source", description="The source document name", type="string"),
    AttributeInfo(name="year", description="Publication year", type="integer"),
    AttributeInfo(name="topic", description="Main topic", type="string"),
]

vectorstore = Chroma(
    collection_name="papers",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
)

retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    vectorstore=vectorstore,
    document_contents="Research papers about machine learning",
    metadata_field_info=metadata_field_info,
)

# The LLM parses "2024 papers about scaling" into:
# query="scaling" + filter={year: 2024}
docs = retriever.invoke("What do the 2024 papers say about scaling laws?")
for doc in docs:
    print(f"[{doc.metadata}] {doc.page_content[:80]}...")`,id:"code-self-query"}),e.jsx(o,{title:"Metadata Cardinality and Index Performance",content:"High-cardinality metadata fields (like unique IDs or timestamps) can degrade filter performance in some vector databases. Group continuous values into buckets (e.g., year instead of full timestamp) and keep the number of distinct filter values manageable. Test filter performance at scale during development.",id:"warning-cardinality"})]})}const G=Object.freeze(Object.defineProperty({__proto__:null,default:h},Symbol.toStringTag,{value:"Module"}));function g(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Hybrid Search: BM25 + Vector Search"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Vector search excels at semantic matching but can miss exact keyword matches. BM25, the classical information retrieval algorithm, excels at keyword matching but misses semantic relationships. Hybrid search combines both approaches for more robust retrieval, capturing both exact term matches and semantic similarity."}),e.jsx(a,{title:"BM25 (Best Matching 25)",definition:"BM25 is a probabilistic ranking function that scores document relevance based on term frequency, inverse document frequency, and document length normalization: $\\text{BM25}(q, d) = \\sum_{t \\in q} \\text{IDF}(t) \\cdot \\frac{f(t, d) \\cdot (k_1 + 1)}{f(t, d) + k_1 \\cdot (1 - b + b \\cdot \\frac{|d|}{avgdl})}$",id:"def-bm25"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The BM25 Formula"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The BM25 scoring function has three key components:"}),e.jsx(t.BlockMath,{math:"\\text{IDF}(t) = \\ln\\left(\\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\\right)"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["where ",e.jsx(t.InlineMath,{math:"N"})," is the total number of documents, ",e.jsx(t.InlineMath,{math:"n(t)"})," is the number of documents containing term ",e.jsx(t.InlineMath,{math:"t"}),", ",e.jsx(t.InlineMath,{math:"f(t,d)"})," is the frequency of ",e.jsx(t.InlineMath,{math:"t"})," in document ",e.jsx(t.InlineMath,{math:"d"}),",",e.jsx(t.InlineMath,{math:"k_1 = 1.2"})," controls term frequency saturation, and ",e.jsx(t.InlineMath,{math:"b = 0.75"})," controls document length normalization."]}),e.jsx(s,{title:"Reciprocal Rank Fusion (RRF)",statement:"Given ranked lists from multiple retrieval methods, RRF combines them using: $\\text{RRF}(d) = \\sum_{r \\in R} \\frac{1}{k + r(d)}$ where $r(d)$ is the rank of document $d$ in ranked list $r$, and $k = 60$ is a constant. This produces a unified ranking without requiring score normalization.",id:"theorem-rrf"}),e.jsx(i,{title:"Why Hybrid Search Outperforms Either Alone",problem:"Show cases where BM25 and vector search each fail independently.",steps:[{formula:'\\text{Query: "HNSW algorithm complexity"}',explanation:'BM25 finds documents with exact match "HNSW". Vector search may rank general "algorithm" docs higher.'},{formula:'\\text{Query: "fast approximate nearest neighbor"}',explanation:'Vector search understands this means HNSW/FAISS. BM25 may miss docs using "ANN" abbreviation.'},{formula:"\\text{Hybrid: } s = \\alpha \\cdot s_{\\text{vector}} + (1-\\alpha) \\cdot s_{\\text{BM25}}",explanation:"Linear combination captures both exact and semantic matches. Alpha typically 0.5-0.7."},{formula:"\\text{RRF}(d) = \\frac{1}{60 + r_{\\text{vec}}(d)} + \\frac{1}{60 + r_{\\text{BM25}}(d)}",explanation:"Alternatively, RRF combines by rank position, avoiding score normalization."}],id:"example-hybrid-advantage"}),e.jsx(r,{title:"hybrid_search.py",code:`# Hybrid search: BM25 + vector search with rank fusion
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
    print(f"  {doc.page_content[:70]}...")`,id:"code-hybrid-search"}),e.jsx(n,{type:"tip",title:"Weaviate Native Hybrid Search",content:"Weaviate supports hybrid search natively with a single API call, combining BM25 and vector search with a configurable alpha parameter. Set alpha=0 for pure BM25, alpha=1 for pure vector, or any value in between. This avoids the overhead of running two separate retrievers.",id:"note-weaviate-hybrid"}),e.jsx(o,{title:"Score Normalization Challenge",content:"BM25 scores and cosine similarities are on fundamentally different scales. BM25 scores can range from 0 to unbounded positive values, while cosine similarity is bounded in [-1, 1]. Use rank-based fusion (RRF) instead of score-based combination unless you carefully normalize scores first.",id:"warning-score-normalization"}),e.jsx(n,{type:"note",title:"When to Use Hybrid Search",content:"Hybrid search is most valuable when your corpus contains domain-specific terminology (medical codes, legal citations, product IDs) that embeddings may not capture well. For purely conversational queries over general text, vector-only search often suffices. Profile your query patterns to decide.",id:"note-when-hybrid"})]})}const N=Object.freeze(Object.defineProperty({__proto__:null,default:g},Symbol.toStringTag,{value:"Module"}));function f(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Query Transformation: HyDE, Expansion, and Decomposition"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Raw user queries are often poorly suited for retrieval: they may be vague, use different vocabulary than the documents, or require information from multiple sources. Query transformation techniques rewrite or expand the query before retrieval to dramatically improve recall and relevance."}),e.jsx(a,{title:"Hypothetical Document Embeddings (HyDE)",definition:"HyDE uses an LLM to generate a hypothetical answer to the query, then embeds that hypothetical answer instead of the query itself. Since the hypothetical answer is closer in vocabulary and structure to actual documents, it retrieves more relevant results: $e_{\\text{HyDE}} = \\text{Embed}(\\text{LLM}(q))$ where $\\text{sim}(e_{\\text{HyDE}}, e_d) > \\text{sim}(e_q, e_d)$.",id:"def-hyde"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"HyDE: Hypothetical Document Embeddings"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:'The key insight behind HyDE is that queries and documents occupy different distributions in embedding space. A question like "How does attention work?" is syntactically different from the answer "Attention computes weighted sums using query-key dot products." By generating a hypothetical answer, we bridge this distribution gap.'}),e.jsx(i,{title:"HyDE Retrieval Flow",problem:"Use HyDE to improve retrieval for an ambiguous query.",steps:[{formula:'q = \\text{"What is RLHF?"}',explanation:"The raw query is short and may not match detailed technical documents."},{formula:'h = \\text{LLM}(q) = \\text{"RLHF (Reinforcement Learning from Human Feedback) is a technique..."}',explanation:"LLM generates a hypothetical answer using its parametric knowledge."},{formula:"e_h = \\text{Embed}(h)",explanation:"Embed the hypothetical document, not the original query."},{formula:"D_k = \\text{top-}k(\\text{sim}(e_h, e_d))",explanation:"Retrieve using the hypothetical embedding, which is closer to document distribution."}],id:"example-hyde"}),e.jsx(r,{title:"query_transformations.py",code:`# Query transformation techniques for better retrieval
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 1. HyDE - Hypothetical Document Embeddings
hyde_prompt = ChatPromptTemplate.from_template(
    """Write a short, detailed passage that would answer this question.
Do not say "I don't know" - generate a plausible answer.

Question: {question}
Passage:"""
)

def hyde_retrieve(query, vectorstore, k=5):
    """Generate hypothetical doc, embed it, retrieve real docs."""
    hypothetical = (hyde_prompt | llm).invoke({"question": query})
    hyde_embedding = embeddings.embed_query(hypothetical.content)
    return vectorstore.similarity_search_by_vector(hyde_embedding, k=k)

# 2. Multi-Query Expansion
expansion_prompt = ChatPromptTemplate.from_template(
    """Generate 3 different versions of this question to improve search.
Each version should use different wording but seek the same information.

Original: {question}
Versions (one per line):"""
)

def multi_query_retrieve(query, retriever, k=5):
    """Generate multiple query versions and merge results."""
    response = (expansion_prompt | llm).invoke({"question": query})
    queries = [query] + response.content.strip().split("\\n")

    all_docs = []
    seen = set()
    for q in queries:
        for doc in retriever.invoke(q.strip()):
            if doc.page_content not in seen:
                all_docs.append(doc)
                seen.add(doc.page_content)
    return all_docs[:k]

# 3. Query Decomposition (for complex questions)
decompose_prompt = ChatPromptTemplate.from_template(
    """Break this complex question into 2-4 simpler sub-questions.

Question: {question}
Sub-questions (one per line):"""
)

def decompose_and_retrieve(query, retriever, k=3):
    """Decompose complex query into sub-queries, retrieve for each."""
    response = (decompose_prompt | llm).invoke({"question": query})
    sub_queries = response.content.strip().split("\\n")

    results = {}
    for sq in sub_queries:
        sq = sq.strip().lstrip("0123456789.- ")
        results[sq] = retriever.invoke(sq)[:k]
    return results

# Example usage
query = "How does LoRA compare to full fine-tuning in terms of cost and quality?"
sub_results = decompose_and_retrieve(query, retriever, k=3)
for sq, docs in sub_results.items():
    print(f"\\nSub-query: {sq}")
    for d in docs:
        print(f"  -> {d.page_content[:60]}...")`,id:"code-query-transformations"}),e.jsx(n,{type:"intuition",title:"Why HyDE Works Despite Hallucinations",content:"HyDE's hypothetical answer may contain factual errors, but that does not matter. The goal is not factual accuracy but vocabulary and structure matching. A hallucinated passage about 'RLHF uses reward models trained on human preferences' will be embedded near real documents about RLHF, even if the details are wrong. The real documents provide the factual grounding.",id:"note-hyde-intuition"}),e.jsx(o,{title:"Query Transformation Adds Latency",content:"Each transformation requires an LLM call, adding 200-1000ms of latency. Multi-query expansion multiplies retrieval calls. For latency-sensitive applications, consider caching common transformations or using a smaller, faster model for the transformation step.",id:"warning-latency"}),e.jsx(n,{type:"tip",title:"Step-Back Prompting",content:"Step-back prompting asks the LLM to generate a more abstract, higher-level question before retrieval. For 'What was the GDP of France in 2023?', the step-back question might be 'What are the economic indicators of France?' This retrieves broader context that is more likely to contain the specific answer.",id:"note-step-back"})]})}const E=Object.freeze(Object.defineProperty({__proto__:null,default:f},Symbol.toStringTag,{value:"Module"}));function x(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Cross-Encoder Reranking"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Initial retrieval using bi-encoders is fast but approximate. Reranking applies a more powerful cross-encoder model to re-score and reorder the top-k retrieved documents, significantly improving the quality of the final context passed to the LLM. This two-stage approach combines the efficiency of bi-encoders with the accuracy of cross-encoders."}),e.jsx(a,{title:"Cross-Encoder Reranking",definition:"A cross-encoder processes the query-document pair jointly through a transformer: $s(q, d) = \\sigma(W \\cdot \\text{BERT}([q; \\text{SEP}; d]) + b)$, producing a relevance score in $[0, 1]$. Unlike bi-encoders, cross-encoders capture fine-grained query-document interactions but cannot pre-compute document representations.",id:"def-reranking"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Two-Stage Retrieval"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The two-stage retrieve-then-rerank pipeline first casts a wide net with fast bi-encoder retrieval, then refines with expensive but accurate cross-encoder scoring:"}),e.jsx(t.BlockMath,{math:"\\text{Stage 1: } D_{20} = \\text{top-}20(\\text{bi-encoder}(q, \\mathcal{D}))"}),e.jsx(t.BlockMath,{math:"\\text{Stage 2: } D_5 = \\text{top-}5(\\text{cross-encoder}(q, D_{20}))"}),e.jsx(i,{title:"Reranking Improves Precision",problem:"Show how reranking reorders retrieved documents for better relevance.",steps:[{formula:"\\text{Retrieve top-20 with bi-encoder}",explanation:"Fast initial retrieval ensures high recall but may include irrelevant results."},{formula:"s_i = \\text{CrossEncoder}(q, d_i) \\quad \\forall d_i \\in D_{20}",explanation:"Score each candidate with the cross-encoder for fine-grained relevance."},{formula:"D_5 = \\text{sort}(D_{20}, \\text{key}=s_i)[:5]",explanation:"Take only the top 5 after reranking - these are much more relevant."},{formula:"\\text{Precision@5 increases from 0.4 to 0.8 typically}",explanation:"Reranking commonly doubles precision at small k values."}],id:"example-reranking"}),e.jsx(r,{title:"cross_encoder_reranking.py",code:`# Two-stage retrieval with cross-encoder reranking
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
    print(f"  {initial_results[result.index].page_content[:60]}...")`,id:"code-reranking"}),e.jsx(r,{title:"langchain_reranking.py",code:`# Reranking integrated into a LangChain RAG pipeline
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
    print(f"  {doc.page_content[:80]}...")`,id:"code-langchain-rerank"}),e.jsx(n,{type:"tip",title:"Reranking Budget",content:"Cross-encoders process each query-document pair independently, so reranking 20 documents requires 20 forward passes. Keep the initial retrieval set to 20-50 documents. Beyond that, the latency cost outweighs the relevance gains. Cohere's rerank API can process up to 1000 documents efficiently due to batching optimizations.",id:"note-budget"}),e.jsx(o,{title:"Reranker Training Domain Matters",content:"Most open-source cross-encoders are trained on web search data (MS MARCO). They may underperform on specialized domains like legal, medical, or code. For domain-specific applications, consider fine-tuning a cross-encoder on your domain data or using a general-purpose LLM as a reranker with appropriate prompting.",id:"warning-domain"})]})}const I=Object.freeze(Object.defineProperty({__proto__:null,default:x},Symbol.toStringTag,{value:"Module"}));function y(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Multi-Hop Reasoning and Iterative Retrieval"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Many real-world questions cannot be answered from a single retrieval step. Multi-hop reasoning requires the system to retrieve information, reason about it, and then retrieve again based on what it learned. This iterative process mirrors how humans research complex topics by following chains of evidence."}),e.jsx(a,{title:"Multi-Hop Retrieval",definition:"Multi-hop retrieval iteratively refines the query and retrieves new documents at each step: $q_0 \\to D_1 \\to q_1 = f(q_0, D_1) \\to D_2 \\to q_2 = f(q_1, D_2) \\to \\dots \\to a = g(q_0, D_1, D_2, \\dots)$. Each hop's query is informed by the documents retrieved in previous hops.",id:"def-multi-hop"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"When Single-Hop Fails"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:`Consider the question: "Which company acquired the startup that developed the embedding model used by LangChain's default retriever?" Answering this requires multiple lookups: first finding which embedding model LangChain uses, then finding which startup built it, then finding the acquiring company.`}),e.jsx(i,{title:"Multi-Hop Question Decomposition",problem:"Answer: 'Did the inventor of the transformer architecture work at the same company as the creator of Word2Vec?'",steps:[{formula:'q_1 = \\text{"Who invented the transformer architecture?"}',explanation:"First hop: retrieve documents about transformer origins."},{formula:"D_1 \\to \\text{Vaswani et al. at Google Brain}",explanation:"Extract the answer: Ashish Vaswani et al. at Google."},{formula:'q_2 = \\text{"Who created Word2Vec and where?"}',explanation:"Second hop: retrieve documents about Word2Vec origins."},{formula:"D_2 \\to \\text{Tomas Mikolov at Google}",explanation:"Extract: Mikolov at Google. Both were at Google, so the answer is yes."}],id:"example-multi-hop"}),e.jsx(r,{title:"iterative_retrieval.py",code:`# Multi-hop iterative retrieval with LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 1: Decompose complex query into sub-questions
decompose_prompt = ChatPromptTemplate.from_template(
    """Break this question into sequential sub-questions where each may
depend on answers from previous sub-questions.

Question: {question}
Sub-questions (one per line, in order):"""
)

# Step 2: Iteratively retrieve and answer
answer_prompt = ChatPromptTemplate.from_template(
    """Answer this sub-question using the context and any previous findings.

Previous findings: {previous}
Context: {context}
Sub-question: {sub_question}
Answer:"""
)

def multi_hop_retrieve(question, max_hops=4):
    """Iterative multi-hop retrieval and reasoning."""
    # Decompose
    sub_qs = (decompose_prompt | llm | StrOutputParser()).invoke(
        {"question": question}
    ).strip().split("\\n")

    findings = []
    for i, sq in enumerate(sub_qs[:max_hops]):
        sq = sq.strip().lstrip("0123456789.- ")

        # Use previous findings to augment retrieval query
        search_query = sq
        if findings:
            search_query = f"{sq} Context: {findings[-1]}"

        # Retrieve
        docs = retriever.invoke(search_query)
        context = "\\n".join(d.page_content for d in docs)

        # Answer sub-question
        answer = (answer_prompt | llm | StrOutputParser()).invoke({
            "previous": "\\n".join(findings) if findings else "None",
            "context": context,
            "sub_question": sq,
        })

        findings.append(f"Q: {sq} -> A: {answer}")
        print(f"Hop {i+1}: {sq}")
        print(f"  Answer: {answer}\\n")

    # Final synthesis
    synthesis_prompt = ChatPromptTemplate.from_template(
        """Synthesize a final answer from these findings.
Original question: {question}
Findings:
{findings}
Final answer:"""
    )

    final = (synthesis_prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "findings": "\\n".join(findings),
    })
    return final

result = multi_hop_retrieve(
    "How does the attention mechanism in transformers relate to "
    "the concept of memory in LSTMs?"
)
print("Final:", result)`,id:"code-multi-hop"}),e.jsx(n,{type:"intuition",title:"Multi-Hop as Research",content:"Multi-hop RAG mimics how a researcher works: read a paper, note a reference, look up that reference, find another lead, follow it. Each step builds on prior knowledge. The key is knowing when to stop iterating - a fixed maximum of 3-4 hops prevents runaway chains while handling most real questions.",id:"note-research-analogy"}),e.jsx(o,{title:"Error Propagation in Multi-Hop",content:"Each hop introduces the possibility of error. If hop 1 retrieves the wrong document, all subsequent hops build on incorrect context. Multi-hop systems need robust error handling: verify intermediate answers, use multiple retrieval paths, and consider backtracking when confidence is low.",id:"warning-error-propagation"}),e.jsx(n,{type:"note",title:"IRCoT: Interleaving Retrieval with Chain-of-Thought",content:"IRCoT (Trivedi et al., 2023) interleaves chain-of-thought reasoning with retrieval. After each reasoning step, the model generates a retrieval query based on what information it still needs. This is more flexible than pre-decomposition because the retrieval strategy adapts dynamically to what the model discovers at each step.",id:"note-ircot"})]})}const P=Object.freeze(Object.defineProperty({__proto__:null,default:y},Symbol.toStringTag,{value:"Module"}));function v(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Agentic RAG: Agent-Driven Retrieval and Routing"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Standard RAG pipelines follow a fixed retrieve-then-generate pattern. Agentic RAG gives an LLM agent control over the retrieval process, allowing it to decide which knowledge sources to query, when to retrieve, whether results are sufficient, and when to retry with a different strategy. This creates adaptive, self-correcting retrieval systems."}),e.jsx(a,{title:"Agentic RAG",definition:"Agentic RAG uses an LLM agent with access to retrieval tools to dynamically orchestrate the retrieval process. The agent decides at each step whether to retrieve, which source to query, and whether to generate an answer or gather more information: $a_t = \\pi(s_t)$ where $s_t$ includes the query, retrieved documents, and reasoning history.",id:"def-agentic-rag"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Router-Based RAG"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The simplest form of agentic RAG routes queries to different retrieval backends based on query type. A coding question routes to a code documentation index, a policy question routes to the compliance database, and a general question may skip retrieval entirely."}),e.jsx(i,{title:"Agentic RAG Decision Flow",problem:"Design an agent that routes queries and self-corrects retrieval.",steps:[{formula:"\\text{Route: } q \\to \\{\\text{vectorDB, SQL, API, none}\\}",explanation:"Agent classifies the query and selects the appropriate retrieval tool."},{formula:"\\text{Retrieve: } D = \\text{tool}(q)",explanation:"Execute retrieval using the selected tool."},{formula:"\\text{Grade: } \\text{relevant}(q, D) \\geq \\tau",explanation:"Agent evaluates whether retrieved documents are relevant enough."},{formula:"\\text{If irrelevant: retry with transformed } q'",explanation:"Self-correct by rephrasing the query or trying a different source."}],id:"example-agentic-flow"}),e.jsx(r,{title:"agentic_rag.py",code:`# Agentic RAG with LangGraph - self-correcting retrieval
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class RAGState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    retries: int

# Node: Route query to appropriate source
def route_query(state: RAGState):
    """Decide whether to retrieve or answer directly."""
    router_prompt = ChatPromptTemplate.from_template(
        """Classify this question:
- "retrieve" if it needs external knowledge
- "direct" if you can answer from general knowledge

Question: {question}
Classification:"""
    )
    result = (router_prompt | llm | StrOutputParser()).invoke(
        {"question": state["question"]}
    )
    return "retrieve" if "retrieve" in result.lower() else "direct"

# Node: Retrieve documents
def retrieve(state: RAGState):
    vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
    docs = vectorstore.similarity_search(state["question"], k=4)
    return {"documents": [d.page_content for d in docs]}

# Node: Grade document relevance
def grade_documents(state: RAGState):
    grade_prompt = ChatPromptTemplate.from_template(
        """Is this document relevant to the question?
Document: {document}
Question: {question}
Answer "yes" or "no":"""
    )
    relevant = []
    for doc in state["documents"]:
        result = (grade_prompt | llm | StrOutputParser()).invoke(
            {"document": doc, "question": state["question"]}
        )
        if "yes" in result.lower():
            relevant.append(doc)
    return {"documents": relevant}

# Node: Generate answer
def generate(state: RAGState):
    gen_prompt = ChatPromptTemplate.from_template(
        """Answer using only the provided context.
Context: {context}
Question: {question}
Answer:"""
    )
    context = "\\n\\n".join(state["documents"])
    answer = (gen_prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": state["question"]}
    )
    return {"generation": answer}

# Node: Transform query for retry
def transform_query(state: RAGState):
    rewrite_prompt = ChatPromptTemplate.from_template(
        """Rewrite this question to be more specific for search.
Original: {question}
Rewritten:"""
    )
    new_q = (rewrite_prompt | llm | StrOutputParser()).invoke(
        {"question": state["question"]}
    )
    return {"question": new_q, "retries": state["retries"] + 1}

# Build the graph
def should_retry(state: RAGState):
    if not state["documents"] and state["retries"] < 2:
        return "transform"
    return "generate"

graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade_documents)
graph.add_node("generate", generate)
graph.add_node("transform", transform_query)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", should_retry)
graph.add_edge("transform", "retrieve")
graph.add_edge("generate", END)

app = graph.compile()
result = app.invoke({"question": "How does LoRA work?", "retries": 0})
print(result["generation"])`,id:"code-agentic-rag"}),e.jsx(n,{type:"tip",title:"Corrective RAG (CRAG)",content:"CRAG (Yan et al., 2024) is a specific agentic RAG pattern where the agent evaluates retrieval quality and takes corrective action. If documents are irrelevant, the agent can rewrite the query, search the web as a fallback, or refine the knowledge base. This self-correcting loop significantly improves robustness.",id:"note-crag"}),e.jsx(o,{title:"Agent Loops and Cost",content:"Agentic RAG can enter retry loops that consume many LLM calls. Always set a maximum number of retries (2-3 is typical) and implement cost tracking. A single complex query could trigger 10+ LLM calls across routing, grading, transformation, and generation steps.",id:"warning-cost"}),e.jsx(n,{type:"note",title:"Tool-Use RAG",content:"Modern agentic RAG systems expose retrieval as tools that the LLM can call. The agent might have tools for vector search, SQL queries, web search, and API calls. The LLM's function-calling capability naturally handles routing and multi-step retrieval without explicit graph construction.",id:"note-tool-use"})]})}const O=Object.freeze(Object.defineProperty({__proto__:null,default:v},Symbol.toStringTag,{value:"Module"}));function b(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"RAG Evaluation Metrics: Faithfulness, Relevance, Precision@k"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Evaluating RAG systems requires measuring both retrieval quality and generation quality. Unlike simple accuracy metrics, RAG evaluation must assess whether the right documents were retrieved, whether the generated answer is faithful to those documents, and whether the answer is actually relevant to the user's question."}),e.jsx(a,{title:"RAG Evaluation Dimensions",definition:"RAG evaluation spans three dimensions: (1) Retrieval quality - are the right documents retrieved? Measured by Precision@k and Recall@k. (2) Faithfulness - is the answer grounded in retrieved context? (3) Answer relevance - does the answer address the question? Each can be computed with ground truth labels or via LLM-as-judge.",id:"def-rag-eval"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Retrieval Metrics"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Retrieval metrics evaluate how well the system finds relevant documents:"}),e.jsx(t.BlockMath,{math:"\\text{Precision@k} = \\frac{|\\text{relevant docs in top-}k|}{k}"}),e.jsx(t.BlockMath,{math:"\\text{Recall@k} = \\frac{|\\text{relevant docs in top-}k|}{|\\text{total relevant docs}|}"}),e.jsx(t.BlockMath,{math:"\\text{MRR} = \\frac{1}{|Q|} \\sum_{i=1}^{|Q|} \\frac{1}{\\text{rank}_i}"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Mean Reciprocal Rank (MRR) rewards finding the first relevant document early. Normalized Discounted Cumulative Gain (nDCG) extends this to graded relevance:"}),e.jsx(t.BlockMath,{math:"\\text{nDCG@k} = \\frac{\\text{DCG@k}}{\\text{IDCG@k}} = \\frac{\\sum_{i=1}^{k} \\frac{2^{r_i} - 1}{\\log_2(i+1)}}{\\text{IDCG@k}}"}),e.jsx(i,{title:"Computing Precision@k and MRR",problem:"Given 5 retrieved documents where relevance labels are [1, 0, 1, 0, 1], compute metrics.",steps:[{formula:"\\text{Precision@3} = \\frac{|\\{d_1, d_3\\}|}{3} = \\frac{2}{3} \\approx 0.667",explanation:"Two of the top 3 retrieved documents are relevant."},{formula:"\\text{Precision@5} = \\frac{|\\{d_1, d_3, d_5\\}|}{5} = \\frac{3}{5} = 0.6",explanation:"Three of all 5 retrieved documents are relevant."},{formula:"\\text{MRR} = \\frac{1}{\\text{rank}_1} = \\frac{1}{1} = 1.0",explanation:"The first relevant document is at rank 1, so MRR is perfect."},{formula:"\\text{Recall@5} = \\frac{3}{|\\text{total relevant}|}",explanation:"Recall depends on how many relevant documents exist in the entire corpus."}],id:"example-retrieval-metrics"}),e.jsx(r,{title:"rag_metrics.py",code:`# Computing RAG evaluation metrics
import numpy as np

def precision_at_k(relevant, k):
    """Precision@k: fraction of top-k that are relevant."""
    return sum(relevant[:k]) / k

def recall_at_k(relevant, total_relevant, k):
    """Recall@k: fraction of all relevant docs found in top-k."""
    return sum(relevant[:k]) / total_relevant

def mrr(relevant_lists):
    """Mean Reciprocal Rank across multiple queries."""
    rr_scores = []
    for relevance in relevant_lists:
        for i, r in enumerate(relevance):
            if r == 1:
                rr_scores.append(1.0 / (i + 1))
                break
        else:
            rr_scores.append(0.0)
    return np.mean(rr_scores)

def ndcg_at_k(relevance, k):
    """Normalized Discounted Cumulative Gain."""
    dcg = sum(
        (2**r - 1) / np.log2(i + 2)
        for i, r in enumerate(relevance[:k])
    )
    ideal = sorted(relevance, reverse=True)[:k]
    idcg = sum(
        (2**r - 1) / np.log2(i + 2)
        for i, r in enumerate(ideal)
    )
    return dcg / idcg if idcg > 0 else 0.0

# Example evaluation
relevance = [1, 0, 1, 0, 1]  # Binary relevance of retrieved docs
print(f"Precision@3: {precision_at_k(relevance, 3):.3f}")
print(f"Precision@5: {precision_at_k(relevance, 5):.3f}")
print(f"Recall@5 (assuming 4 total relevant): {recall_at_k(relevance, 4, 5):.3f}")
print(f"nDCG@5: {ndcg_at_k(relevance, 5):.3f}")

# Faithfulness via LLM-as-judge
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

faithfulness_prompt = ChatPromptTemplate.from_template(
    """Given the context and the answer, extract all claims from the answer.
For each claim, determine if it is supported by the context.

Context: {context}
Answer: {answer}

Return: number of supported claims / total claims as a score 0-1."""
)

score = (faithfulness_prompt | llm).invoke({
    "context": "Transformers use self-attention. Introduced in 2017.",
    "answer": "Transformers, introduced in 2017, use self-attention and were invented at OpenAI.",
})
print(f"Faithfulness: {score.content}")`,id:"code-rag-metrics"}),e.jsx(n,{type:"note",title:"LLM-as-Judge for Generation Quality",content:"Since generation quality (faithfulness, relevance) is hard to measure with traditional metrics, LLM-as-judge approaches use a powerful LLM to evaluate the output. The judge LLM checks if claims are supported by context (faithfulness), if the answer addresses the question (relevance), and if the answer is complete (recall). GPT-4 judgments correlate well with human ratings.",id:"note-llm-judge"}),e.jsx(o,{title:"Metric Gaming",content:"Optimizing for one metric can degrade others. Increasing k improves retrieval recall but may decrease precision and faithfulness (by introducing noisy context). Always evaluate multiple metrics together and prioritize faithfulness for production systems - a wrong answer is worse than no answer.",id:"warning-metric-gaming"})]})}const D=Object.freeze(Object.defineProperty({__proto__:null,default:b},Symbol.toStringTag,{value:"Module"}));function _(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"The RAGAS Evaluation Framework"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"RAGAS (Retrieval Augmented Generation Assessment) is a framework for automated evaluation of RAG pipelines. It provides reference-free metrics that use LLMs to assess faithfulness, answer relevance, and context quality without requiring manually labeled ground truth data."}),e.jsx(a,{title:"RAGAS Framework",definition:"RAGAS evaluates RAG systems across four core metrics: (1) Faithfulness $= \\frac{|\\text{supported claims}|}{|\\text{total claims}|}$, (2) Answer Relevance $= \\text{mean}(\\text{sim}(q, q_i^{\\text{gen}}))$, (3) Context Precision $= \\frac{\\sum_{k} \\text{Precision@}k \\cdot v_k}{\\text{total relevant}}$, and (4) Context Recall $= \\frac{|\\text{GT sentences attributable to context}|}{|\\text{GT sentences}|}$.",id:"def-ragas"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"RAGAS Metrics in Detail"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Each RAGAS metric targets a different failure mode in the RAG pipeline:"}),e.jsx(t.BlockMath,{math:"\\text{Faithfulness} = \\frac{|V_s|}{|V|} \\text{ where } V = \\text{claims}(a), \\; V_s = \\{v \\in V : v \\text{ supported by } C\\}"}),e.jsx(t.BlockMath,{math:"\\text{Answer Relevance} = \\frac{1}{N} \\sum_{i=1}^{N} \\text{sim}(e_q, e_{q_i})"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["Answer relevance generates ",e.jsx(t.InlineMath,{math:"N"})," questions from the answer and measures their similarity to the original question. High similarity means the answer addresses what was asked."]}),e.jsx(i,{title:"RAGAS Evaluation Workflow",problem:"Evaluate a RAG pipeline using all four RAGAS metrics.",steps:[{formula:"\\text{Input: } (q, C, a, a^*)",explanation:"Each sample needs: question, retrieved context, generated answer, and optionally ground truth."},{formula:"\\text{Faithfulness: extract claims from } a, \\text{ verify against } C",explanation:"LLM extracts factual claims from the answer and checks each against the context."},{formula:"\\text{Relevance: generate questions from } a, \\text{ compare to } q",explanation:"If the answer is relevant, questions generated from it should resemble the original query."},{formula:"\\text{Context Precision: rank relevant contexts higher}",explanation:"Measures whether the most relevant context chunks appear at the top of the retrieval results."}],id:"example-ragas-workflow"}),e.jsx(r,{title:"ragas_evaluation.py",code:`# Evaluating a RAG pipeline with RAGAS
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": [
        "What is self-attention?",
        "How does LoRA reduce parameters?",
        "What is the transformer architecture?",
    ],
    "answer": [
        "Self-attention computes weighted sums of all positions using QKV.",
        "LoRA adds low-rank matrices A and B where W = W0 + BA.",
        "Transformers use attention mechanisms for sequence processing.",
    ],
    "contexts": [
        ["Self-attention computes attention weights using query-key dot products."],
        ["LoRA decomposes weight updates into low-rank matrices A and B."],
        ["The transformer was introduced in Attention Is All You Need (2017)."],
    ],
    "ground_truth": [
        "Self-attention allows each position to attend to all other positions.",
        "LoRA freezes pretrained weights and adds trainable low-rank decomposition.",
        "The transformer is an architecture using self-attention, introduced in 2017.",
    ],
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
)

print("RAGAS Scores:")
print(f"  Faithfulness:       {results['faithfulness']:.3f}")
print(f"  Answer Relevancy:   {results['answer_relevancy']:.3f}")
print(f"  Context Precision:  {results['context_precision']:.3f}")
print(f"  Context Recall:     {results['context_recall']:.3f}")

# Per-sample analysis
df = results.to_pandas()
print("\\nPer-sample scores:")
print(df[["question", "faithfulness", "answer_relevancy"]].to_string())`,id:"code-ragas"}),e.jsx(r,{title:"ragas_test_generation.py",code:`# Generate synthetic test data with RAGAS
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document

# Your RAG documents
documents = [
    Document(page_content="Self-attention computes Q, K, V matrices..."),
    Document(page_content="LoRA freezes base weights and trains low-rank..."),
    Document(page_content="Transformers replaced RNNs for sequence modeling..."),
]

# Generate diverse test questions automatically
generator = TestsetGenerator.from_langchain(
    generator_llm=ChatOpenAI(model="gpt-4o-mini"),
    critic_llm=ChatOpenAI(model="gpt-4o-mini"),
    embeddings=OpenAIEmbeddings(),
)

testset = generator.generate_with_langchain_docs(
    documents,
    test_size=20,
    distributions={
        simple: 0.5,       # Simple factual questions
        reasoning: 0.3,    # Multi-step reasoning
        multi_context: 0.2, # Require multiple chunks
    },
)

test_df = testset.to_pandas()
print(f"Generated {len(test_df)} test questions:")
print(test_df[["question", "evolution_type"]].head(10))`,id:"code-ragas-testgen"}),e.jsx(n,{type:"tip",title:"Continuous Evaluation",content:"Run RAGAS evaluation as part of your CI/CD pipeline. Track metrics over time to catch regressions when you change chunking strategies, embedding models, or prompts. Set minimum thresholds (e.g., faithfulness > 0.8) as quality gates before deploying RAG pipeline changes.",id:"note-continuous-eval"}),e.jsx(o,{title:"RAGAS Limitations",content:"RAGAS metrics rely on LLM judgments, which can be inconsistent and biased. The faithfulness metric may miss subtle hallucinations, and answer relevancy can be fooled by paraphrasing. Always supplement automated RAGAS evaluation with periodic human evaluation on a representative sample.",id:"warning-ragas-limitations"})]})}const B=Object.freeze(Object.defineProperty({__proto__:null,default:_},Symbol.toStringTag,{value:"Module"}));function k(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Context Window Optimization"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The context window is a finite and precious resource. Stuffing it with too many retrieved documents dilutes relevance and can degrade generation quality. Context optimization techniques maximize the signal-to-noise ratio of the information passed to the LLM, ensuring every token in the context contributes to a better answer."}),e.jsx(a,{title:"Context Window Budget",definition:"The context budget $B$ is the maximum tokens available for retrieved context: $B = W - T_{\\text{system}} - T_{\\text{query}} - T_{\\text{output}}$ where $W$ is the model's context window, and the other terms are the system prompt, query, and reserved output tokens respectively.",id:"def-context-budget"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The Lost in the Middle Problem"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Research by Liu et al. (2023) showed that LLMs pay less attention to information in the middle of long contexts. Relevant information placed at the beginning or end of the context is used more effectively than information buried in the middle."}),e.jsx(t.BlockMath,{math:"P(\\text{use info at position } i) \\propto \\begin{cases} \\text{high} & i \\text{ near start or end} \\\\ \\text{low} & i \\text{ in middle} \\end{cases}"}),e.jsx(i,{title:"Context Window Optimization Strategies",problem:"Optimize context usage for a model with 8K context window.",steps:[{formula:"B = 8192 - 500 - 100 - 1000 = 6592 \\text{ tokens}",explanation:"Calculate available budget after system prompt, query, and output reservation."},{formula:"\\text{With 512-token chunks: } \\lfloor 6592/512 \\rfloor = 12 \\text{ chunks max}",explanation:"Maximum number of full chunks that fit in the budget."},{formula:"\\text{Order: most relevant at start and end}",explanation:"Place the highest-relevance chunks at positions 1 and n, not in the middle."},{formula:"\\text{Compress: remove redundant information}",explanation:"Use LLM-based compression to extract only the relevant sentences from each chunk."}],id:"example-context-optimization"}),e.jsx(r,{title:"context_optimization.py",code:`# Context window optimization techniques
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(enc.encode(text))

# 1. Token budget calculation
MODEL_CONTEXT = 8192
SYSTEM_TOKENS = 500
QUERY_TOKENS = 100
OUTPUT_RESERVE = 1000
BUDGET = MODEL_CONTEXT - SYSTEM_TOKENS - QUERY_TOKENS - OUTPUT_RESERVE
print(f"Context budget: {BUDGET} tokens")

# 2. Smart context stuffing with budget awareness
def stuff_context(docs, max_tokens):
    """Add documents until budget is exhausted."""
    context_parts = []
    total_tokens = 0
    for doc in docs:
        doc_tokens = count_tokens(doc.page_content)
        if total_tokens + doc_tokens > max_tokens:
            break
        context_parts.append(doc.page_content)
        total_tokens += doc_tokens
    return "\\n\\n".join(context_parts), total_tokens

# 3. Lost-in-the-middle reordering
def reorder_for_attention(docs):
    """Place most relevant docs at start and end (avoid middle)."""
    if len(docs) <= 2:
        return docs
    # docs assumed sorted by relevance (most relevant first)
    reordered = []
    for i, doc in enumerate(docs):
        if i % 2 == 0:
            reordered.insert(0, doc)   # Prepend (start)
        else:
            reordered.append(doc)      # Append (end)
    return reordered

# 4. Context compression with LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

compress_prompt = ChatPromptTemplate.from_template(
    """Extract ONLY the sentences relevant to the question from this text.
Remove all irrelevant information. Keep extracted sentences verbatim.

Question: {question}
Text: {text}
Relevant sentences:"""
)

def compress_context(question, docs, max_tokens):
    """Compress each document to keep only relevant sentences."""
    compressed = []
    total = 0
    for doc in docs:
        result = (compress_prompt | llm).invoke({
            "question": question,
            "text": doc.page_content,
        })
        tokens = count_tokens(result.content)
        if total + tokens <= max_tokens:
            compressed.append(result.content)
            total += tokens
    return "\\n\\n".join(compressed)

# Example: compress 5 chunks into budget
original_tokens = sum(count_tokens(d.page_content) for d in docs)
compressed = compress_context("How does attention work?", docs, BUDGET)
print(f"Original: {original_tokens} tokens")
print(f"Compressed: {count_tokens(compressed)} tokens")`,id:"code-context-optimization"}),e.jsx(n,{type:"intuition",title:"Quality Over Quantity",content:"Passing 10 mediocre chunks is often worse than passing 3 highly relevant ones. The LLM must parse all provided context, and irrelevant information can confuse it or cause it to hallucinate by mixing up details from different chunks. Think of it as providing a research assistant with a focused brief rather than a stack of loosely related papers.",id:"note-quality-over-quantity"}),e.jsx(o,{title:"Compression Can Lose Information",content:"LLM-based context compression adds latency and cost, and the compressor LLM may accidentally remove information that turns out to be important for answering the question. Always evaluate compressed vs. uncompressed performance on your specific use case before deploying compression in production.",id:"warning-compression"}),e.jsx(n,{type:"tip",title:"Long-Context Models Are Not a Free Pass",content:"Models with 128K or 1M token context windows can fit more documents, but retrieval quality still matters. Studies show that even long-context models degrade when filled with irrelevant information. Use the extra context for more relevant documents, not more total documents.",id:"note-long-context"})]})}const $=Object.freeze(Object.defineProperty({__proto__:null,default:k},Symbol.toStringTag,{value:"Module"}));function w(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"When RAG Is Not the Answer"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"RAG has become the default pattern for adding knowledge to LLMs, but it is not always the right approach. Understanding when RAG fails, when simpler solutions suffice, and when fine-tuning or other techniques are more appropriate saves engineering effort and produces better systems. The best RAG system is sometimes no RAG at all."}),e.jsx(a,{title:"RAG vs. Fine-Tuning Decision",definition:"Choose RAG when knowledge is dynamic, factual, and retrievable. Choose fine-tuning when the task requires learning a style, format, or behavioral pattern. Choose both (RAG + fine-tuned model) for domain-specific applications requiring specialized knowledge and adapted behavior.",id:"def-rag-vs-ft"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"When RAG Fails"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"RAG assumes the answer exists in the knowledge base and can be found via embedding similarity. These assumptions break down in several important cases."}),e.jsx(i,{title:"RAG Failure Modes",problem:"Identify scenarios where RAG is the wrong approach.",steps:[{formula:"\\text{Reasoning tasks: } 2x + 3 = 7, \\text{ solve for } x",explanation:"Math problems require computation, not retrieval. No document contains the answer to this specific equation."},{formula:'\\text{Style/format tasks: "Write like Shakespeare"}',explanation:"Writing style is a behavioral pattern best learned through fine-tuning, not retrieval."},{formula:'\\text{Aggregation: "What is the average salary?"}',explanation:"Requires computation over many records. RAG retrieves individual documents, not aggregates."},{formula:'\\text{Implicit knowledge: "Is this code secure?"}',explanation:"Security assessment requires reasoning about patterns, not retrieving specific vulnerabilities."}],id:"example-failure-modes"}),e.jsx(r,{title:"rag_decision_framework.py",code:`# Framework for deciding whether to use RAG
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def should_use_rag(task_description):
    """Analyze whether a task benefits from RAG."""
    analysis_prompt = ChatPromptTemplate.from_template(
        """Analyze whether RAG is appropriate for this task.

Task: {task}

Score each criterion (0-1):
1. Knowledge is factual and specific (not reasoning)
2. Knowledge changes over time or is domain-specific
3. Answers exist verbatim or near-verbatim in documents
4. Task requires citing sources
5. Knowledge base is available and well-structured

Recommend: RAG / Fine-tuning / Prompt Engineering / Hybrid"""
    )

    result = (analysis_prompt | llm).invoke({"task": task_description})
    return result.content

# Decision matrix examples
tasks = {
    "Answer customer questions about product features":
        "RAG - factual, specific, exists in docs",
    "Generate marketing copy in brand voice":
        "Fine-tuning - style/behavioral pattern",
    "Calculate tax liability from financial data":
        "Tool use - requires computation, not retrieval",
    "Summarize company policies for new employees":
        "RAG - factual, citable, in knowledge base",
    "Translate text between languages":
        "Base model - general capability, no retrieval needed",
    "Debug a Python error message":
        "Hybrid - retrieve docs + reason about code",
}

print("RAG Decision Framework:")
print("=" * 60)
for task, recommendation in tasks.items():
    print(f"\\nTask: {task}")
    print(f"  -> {recommendation}")

# Cost comparison
def estimate_costs(n_queries_per_day, approach):
    """Rough cost comparison between approaches."""
    if approach == "rag":
        embedding_cost = n_queries_per_day * 0.0001  # per query embedding
        llm_cost = n_queries_per_day * 0.003         # with context
        infra_cost = 50 / 30                          # vector DB monthly / days
        return embedding_cost + llm_cost + infra_cost
    elif approach == "finetuned":
        llm_cost = n_queries_per_day * 0.002  # cheaper without context
        training_cost = 100 / 30               # amortized training
        return llm_cost + training_cost
    elif approach == "base":
        return n_queries_per_day * 0.001  # no context, no training

for approach in ["base", "rag", "finetuned"]:
    cost = estimate_costs(1000, approach)
    print(f"\\n{approach}: ~USD {cost:.2f}/day for 1000 queries")`,id:"code-decision-framework"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Alternatives to RAG"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Before building a RAG pipeline, consider simpler alternatives that may solve your problem with less complexity:"}),e.jsx(n,{type:"note",title:"The Alternatives Ladder",content:"(1) Prompt engineering: include knowledge directly in the system prompt for small, static knowledge bases. (2) Few-shot examples: show the model what you want rather than telling it. (3) Fine-tuning: bake domain knowledge or behavioral patterns into the model weights. (4) Long-context models: for knowledge bases under 200K tokens, just put everything in the context. (5) RAG: for large, dynamic knowledge bases where retrieval is necessary.",id:"note-alternatives"}),e.jsx(o,{title:"RAG Complexity Is Real",content:"A production RAG system requires: document processing pipeline, embedding infrastructure, vector database operations, retrieval tuning, prompt engineering, evaluation framework, and monitoring. This is significant operational overhead. If your knowledge base fits in a long-context window or rarely changes, simpler approaches may deliver 90% of the benefit at 10% of the complexity.",id:"warning-complexity"}),e.jsx(n,{type:"tip",title:"The Hybrid Approach",content:"For many production systems, the best answer is RAG combined with fine-tuning. Fine-tune the model to understand your domain's terminology and follow your output format, then use RAG to inject specific, up-to-date facts. This gives you the behavioral consistency of fine-tuning with the knowledge freshness of RAG.",id:"note-hybrid"}),e.jsx(n,{type:"intuition",title:"The Librarian Analogy",content:"RAG is like hiring a librarian to find relevant books before asking someone a question. But if the question is 'what is 2+2?', the librarian adds no value. If the question is about the librarian's own writing style, they need practice (fine-tuning), not books. RAG is powerful for the right problems, but not every problem is a retrieval problem.",id:"note-librarian"})]})}const H=Object.freeze(Object.defineProperty({__proto__:null,default:w},Symbol.toStringTag,{value:"Module"}));export{S as a,T as b,M as c,C as d,L as e,G as f,N as g,E as h,I as i,P as j,O as k,D as l,B as m,$ as n,H as o,j as s};
