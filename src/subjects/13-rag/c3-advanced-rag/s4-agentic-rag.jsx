import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function AgenticRAG() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Agentic RAG: Agent-Driven Retrieval and Routing</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Standard RAG pipelines follow a fixed retrieve-then-generate pattern. Agentic RAG
        gives an LLM agent control over the retrieval process, allowing it to decide which
        knowledge sources to query, when to retrieve, whether results are sufficient, and
        when to retry with a different strategy. This creates adaptive, self-correcting
        retrieval systems.
      </p>

      <DefinitionBlock
        title="Agentic RAG"
        definition="Agentic RAG uses an LLM agent with access to retrieval tools to dynamically orchestrate the retrieval process. The agent decides at each step whether to retrieve, which source to query, and whether to generate an answer or gather more information: $a_t = \pi(s_t)$ where $s_t$ includes the query, retrieved documents, and reasoning history."
        id="def-agentic-rag"
      />

      <h2 className="text-2xl font-semibold">Router-Based RAG</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The simplest form of agentic RAG routes queries to different retrieval backends
        based on query type. A coding question routes to a code documentation index, a
        policy question routes to the compliance database, and a general question may
        skip retrieval entirely.
      </p>

      <ExampleBlock
        title="Agentic RAG Decision Flow"
        problem="Design an agent that routes queries and self-corrects retrieval."
        steps={[
          { formula: '\\text{Route: } q \\to \\{\\text{vectorDB, SQL, API, none}\\}', explanation: 'Agent classifies the query and selects the appropriate retrieval tool.' },
          { formula: '\\text{Retrieve: } D = \\text{tool}(q)', explanation: 'Execute retrieval using the selected tool.' },
          { formula: '\\text{Grade: } \\text{relevant}(q, D) \\geq \\tau', explanation: 'Agent evaluates whether retrieved documents are relevant enough.' },
          { formula: '\\text{If irrelevant: retry with transformed } q\'', explanation: 'Self-correct by rephrasing the query or trying a different source.' },
        ]}
        id="example-agentic-flow"
      />

      <PythonCode
        title="agentic_rag.py"
        code={`# Agentic RAG with LangGraph - self-correcting retrieval
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
print(result["generation"])`}
        id="code-agentic-rag"
      />

      <NoteBlock
        type="tip"
        title="Corrective RAG (CRAG)"
        content="CRAG (Yan et al., 2024) is a specific agentic RAG pattern where the agent evaluates retrieval quality and takes corrective action. If documents are irrelevant, the agent can rewrite the query, search the web as a fallback, or refine the knowledge base. This self-correcting loop significantly improves robustness."
        id="note-crag"
      />

      <WarningBlock
        title="Agent Loops and Cost"
        content="Agentic RAG can enter retry loops that consume many LLM calls. Always set a maximum number of retries (2-3 is typical) and implement cost tracking. A single complex query could trigger 10+ LLM calls across routing, grading, transformation, and generation steps."
        id="warning-cost"
      />

      <NoteBlock
        type="note"
        title="Tool-Use RAG"
        content="Modern agentic RAG systems expose retrieval as tools that the LLM can call. The agent might have tools for vector search, SQL queries, web search, and API calls. The LLM's function-calling capability naturally handles routing and multi-step retrieval without explicit graph construction."
        id="note-tool-use"
      />
    </div>
  )
}
