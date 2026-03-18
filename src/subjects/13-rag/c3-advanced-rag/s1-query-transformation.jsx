import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function QueryTransformation() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Query Transformation: HyDE, Expansion, and Decomposition</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Raw user queries are often poorly suited for retrieval: they may be vague, use different
        vocabulary than the documents, or require information from multiple sources. Query
        transformation techniques rewrite or expand the query before retrieval to dramatically
        improve recall and relevance.
      </p>

      <DefinitionBlock
        title="Hypothetical Document Embeddings (HyDE)"
        definition="HyDE uses an LLM to generate a hypothetical answer to the query, then embeds that hypothetical answer instead of the query itself. Since the hypothetical answer is closer in vocabulary and structure to actual documents, it retrieves more relevant results: $e_{\text{HyDE}} = \text{Embed}(\text{LLM}(q))$ where $\text{sim}(e_{\text{HyDE}}, e_d) > \text{sim}(e_q, e_d)$."
        id="def-hyde"
      />

      <h2 className="text-2xl font-semibold">HyDE: Hypothetical Document Embeddings</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The key insight behind HyDE is that queries and documents occupy different distributions
        in embedding space. A question like "How does attention work?" is syntactically different
        from the answer "Attention computes weighted sums using query-key dot products."
        By generating a hypothetical answer, we bridge this distribution gap.
      </p>

      <ExampleBlock
        title="HyDE Retrieval Flow"
        problem="Use HyDE to improve retrieval for an ambiguous query."
        steps={[
          { formula: 'q = \\text{"What is RLHF?"}', explanation: 'The raw query is short and may not match detailed technical documents.' },
          { formula: 'h = \\text{LLM}(q) = \\text{"RLHF (Reinforcement Learning from Human Feedback) is a technique..."}', explanation: 'LLM generates a hypothetical answer using its parametric knowledge.' },
          { formula: 'e_h = \\text{Embed}(h)', explanation: 'Embed the hypothetical document, not the original query.' },
          { formula: 'D_k = \\text{top-}k(\\text{sim}(e_h, e_d))', explanation: 'Retrieve using the hypothetical embedding, which is closer to document distribution.' },
        ]}
        id="example-hyde"
      />

      <PythonCode
        title="query_transformations.py"
        code={`# Query transformation techniques for better retrieval
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
        print(f"  -> {d.page_content[:60]}...")`}
        id="code-query-transformations"
      />

      <NoteBlock
        type="intuition"
        title="Why HyDE Works Despite Hallucinations"
        content="HyDE's hypothetical answer may contain factual errors, but that does not matter. The goal is not factual accuracy but vocabulary and structure matching. A hallucinated passage about 'RLHF uses reward models trained on human preferences' will be embedded near real documents about RLHF, even if the details are wrong. The real documents provide the factual grounding."
        id="note-hyde-intuition"
      />

      <WarningBlock
        title="Query Transformation Adds Latency"
        content="Each transformation requires an LLM call, adding 200-1000ms of latency. Multi-query expansion multiplies retrieval calls. For latency-sensitive applications, consider caching common transformations or using a smaller, faster model for the transformation step."
        id="warning-latency"
      />

      <NoteBlock
        type="tip"
        title="Step-Back Prompting"
        content="Step-back prompting asks the LLM to generate a more abstract, higher-level question before retrieval. For 'What was the GDP of France in 2023?', the step-back question might be 'What are the economic indicators of France?' This retrieves broader context that is more likely to contain the specific answer."
        id="note-step-back"
      />
    </div>
  )
}
