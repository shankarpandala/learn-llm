import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function HierarchicalIndexing() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Hierarchical Indexing: Parent-Child and Summary Indexing</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Flat chunking loses the hierarchical structure of documents. Hierarchical indexing
        preserves parent-child relationships between chunks, enabling retrieval of fine-grained
        passages while providing broader context. Summary indexing creates high-level summaries
        that enable top-down navigation of large document collections.
      </p>

      <DefinitionBlock
        title="Parent-Child Indexing"
        definition="Parent-child indexing creates two levels of chunks: small child chunks $c_i$ used for precise retrieval, and larger parent chunks $P(c_i)$ returned as context. The retrieval matches on children but returns the parent: $\text{context} = P(\arg\max_{c_i} \text{sim}(q, c_i))$."
        id="def-parent-child"
      />

      <h2 className="text-2xl font-semibold">The Small-to-Big Retrieval Pattern</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Small chunks embed more precisely, but the LLM needs broader context to generate
        good answers. The small-to-big pattern resolves this tension: embed small chunks for
        retrieval accuracy, then expand to their parent chunk for generation context.
      </p>

      <ExampleBlock
        title="Parent-Child Retrieval Flow"
        problem="Retrieve precise matches but return full-context parent chunks."
        steps={[
          { formula: 'D \\to \\{P_1, P_2, \\dots\\} \\to \\{c_{1,1}, c_{1,2}, \\dots, c_{2,1}, \\dots\\}', explanation: 'Split document into parent chunks (e.g., 1024 tokens), then child chunks (e.g., 256 tokens).' },
          { formula: 'e_{c_{i,j}} = \\text{Embed}(c_{i,j})', explanation: 'Embed only the child chunks for precise similarity matching.' },
          { formula: 'c^* = \\arg\\max_{c_{i,j}} \\text{sim}(e_q, e_{c_{i,j}})', explanation: 'Find the best-matching child chunk.' },
          { formula: '\\text{Return } P(c^*)', explanation: 'Return the parent chunk containing the matched child for richer context.' },
        ]}
        id="example-parent-child"
      />

      <PythonCode
        title="parent_child_indexing.py"
        code={`# Parent-child retrieval with LlamaIndex
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
    print(f"  {r.text[:100]}...\\n")`}
        id="code-parent-child"
      />

      <h2 className="text-2xl font-semibold">Summary Indexing</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Summary indexing generates an LLM-created summary of each document or section, then
        embeds these summaries as a top-level index. Queries first match against summaries
        to identify relevant documents, then drill into the document chunks for detailed retrieval.
      </p>

      <PythonCode
        title="summary_indexing.py"
        code={`# Summary index for top-down document navigation
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
print(response)`}
        id="code-summary-index"
      />

      <NoteBlock
        type="intuition"
        title="Think of a Library Catalog"
        content="Summary indexing works like a library catalog. You first check the catalog (summaries) to find which books (documents) are relevant, then go to those specific books and look through their pages (chunks). Without the catalog, you would need to flip through every page of every book."
        id="note-library-catalog"
      />

      <WarningBlock
        title="Summary Quality Matters"
        content="Summary indexing is only as good as the generated summaries. If a summary misses a key topic from the document, queries about that topic will fail to route correctly. Always validate summary coverage against the source documents, especially for technical content where nuances matter."
        id="warning-summary-quality"
      />
    </div>
  )
}
