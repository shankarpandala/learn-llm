import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function MetadataFiltering() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Metadata Enrichment and Filtering</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Pure vector similarity often retrieves semantically similar but contextually wrong
        documents. Metadata filtering adds structured constraints to vector search, allowing
        you to narrow results by source, date, category, or any custom attribute. Enriching
        chunks with metadata during indexing is crucial for production RAG systems.
      </p>

      <DefinitionBlock
        title="Metadata-Filtered Retrieval"
        definition="Metadata-filtered retrieval combines vector similarity with structured constraints: $D_k = \text{top-}k\{d \in \mathcal{D} : \text{filter}(d.\text{metadata}) = \text{true}\}$ ranked by $\text{sim}(e_q, e_d)$. This restricts the search space before or after ANN lookup."
        id="def-metadata-filtering"
      />

      <h2 className="text-2xl font-semibold">Types of Metadata</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Effective metadata falls into several categories: structural (source file, section,
        page number), temporal (creation date, last modified), categorical (topic, department,
        document type), and derived (LLM-generated summaries, keywords, entities).
      </p>

      <ExampleBlock
        title="Metadata Enrichment Pipeline"
        problem="Enrich document chunks with metadata for filtered retrieval."
        steps={[
          { formula: '\\text{Structural: } \\{\\text{source, section, page}\\}', explanation: 'Extract from document structure: file path, headers, page numbers.' },
          { formula: '\\text{Temporal: } \\{\\text{created, modified}\\}', explanation: 'Capture timestamps for recency filtering.' },
          { formula: '\\text{LLM-derived: } \\{\\text{summary, entities, keywords}\\}', explanation: 'Use an LLM to extract topics, named entities, and key phrases from each chunk.' },
          { formula: '\\text{Query: } \\text{sim}(q, d) \\text{ WHERE } d.\\text{year} \\geq 2024', explanation: 'Combine vector search with metadata filters at query time.' },
        ]}
        id="example-metadata-enrichment"
      />

      <PythonCode
        title="metadata_enrichment.py"
        code={`# Metadata enrichment and filtered retrieval with Chroma
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
print("Filtered results:", results["documents"])`}
        id="code-metadata-enrichment"
      />

      <NoteBlock
        type="tip"
        title="Self-Query Retrieval"
        content="LangChain's SelfQueryRetriever uses an LLM to automatically extract metadata filters from natural language queries. A query like 'What did the 2024 paper say about scaling?' is decomposed into a vector search for 'scaling' with a metadata filter for year=2024. This eliminates the need for users to specify filters manually."
        id="note-self-query"
      />

      <PythonCode
        title="self_query_retriever.py"
        code={`# Self-query: LLM automatically extracts filters from queries
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
    print(f"[{doc.metadata}] {doc.page_content[:80]}...")`}
        id="code-self-query"
      />

      <WarningBlock
        title="Metadata Cardinality and Index Performance"
        content="High-cardinality metadata fields (like unique IDs or timestamps) can degrade filter performance in some vector databases. Group continuous values into buckets (e.g., year instead of full timestamp) and keep the number of distinct filter values manageable. Test filter performance at scale during development."
        id="warning-cardinality"
      />
    </div>
  )
}
