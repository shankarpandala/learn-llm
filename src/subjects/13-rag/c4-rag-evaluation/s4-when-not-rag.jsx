import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function WhenNotRAG() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">When RAG Is Not the Answer</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        RAG has become the default pattern for adding knowledge to LLMs, but it is not always
        the right approach. Understanding when RAG fails, when simpler solutions suffice, and
        when fine-tuning or other techniques are more appropriate saves engineering effort and
        produces better systems. The best RAG system is sometimes no RAG at all.
      </p>

      <DefinitionBlock
        title="RAG vs. Fine-Tuning Decision"
        definition="Choose RAG when knowledge is dynamic, factual, and retrievable. Choose fine-tuning when the task requires learning a style, format, or behavioral pattern. Choose both (RAG + fine-tuned model) for domain-specific applications requiring specialized knowledge and adapted behavior."
        id="def-rag-vs-ft"
      />

      <h2 className="text-2xl font-semibold">When RAG Fails</h2>
      <p className="text-gray-700 dark:text-gray-300">
        RAG assumes the answer exists in the knowledge base and can be found via embedding
        similarity. These assumptions break down in several important cases.
      </p>

      <ExampleBlock
        title="RAG Failure Modes"
        problem="Identify scenarios where RAG is the wrong approach."
        steps={[
          { formula: '\\text{Reasoning tasks: } 2x + 3 = 7, \\text{ solve for } x', explanation: 'Math problems require computation, not retrieval. No document contains the answer to this specific equation.' },
          { formula: '\\text{Style/format tasks: "Write like Shakespeare"}', explanation: 'Writing style is a behavioral pattern best learned through fine-tuning, not retrieval.' },
          { formula: '\\text{Aggregation: "What is the average salary?"}', explanation: 'Requires computation over many records. RAG retrieves individual documents, not aggregates.' },
          { formula: '\\text{Implicit knowledge: "Is this code secure?"}', explanation: 'Security assessment requires reasoning about patterns, not retrieving specific vulnerabilities.' },
        ]}
        id="example-failure-modes"
      />

      <PythonCode
        title="rag_decision_framework.py"
        code={`# Framework for deciding whether to use RAG
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
    print(f"\\n{approach}: ~USD {cost:.2f}/day for 1000 queries")`}
        id="code-decision-framework"
      />

      <h2 className="text-2xl font-semibold">Alternatives to RAG</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Before building a RAG pipeline, consider simpler alternatives that may solve your
        problem with less complexity:
      </p>

      <NoteBlock
        type="note"
        title="The Alternatives Ladder"
        content="(1) Prompt engineering: include knowledge directly in the system prompt for small, static knowledge bases. (2) Few-shot examples: show the model what you want rather than telling it. (3) Fine-tuning: bake domain knowledge or behavioral patterns into the model weights. (4) Long-context models: for knowledge bases under 200K tokens, just put everything in the context. (5) RAG: for large, dynamic knowledge bases where retrieval is necessary."
        id="note-alternatives"
      />

      <WarningBlock
        title="RAG Complexity Is Real"
        content="A production RAG system requires: document processing pipeline, embedding infrastructure, vector database operations, retrieval tuning, prompt engineering, evaluation framework, and monitoring. This is significant operational overhead. If your knowledge base fits in a long-context window or rarely changes, simpler approaches may deliver 90% of the benefit at 10% of the complexity."
        id="warning-complexity"
      />

      <NoteBlock
        type="tip"
        title="The Hybrid Approach"
        content="For many production systems, the best answer is RAG combined with fine-tuning. Fine-tune the model to understand your domain's terminology and follow your output format, then use RAG to inject specific, up-to-date facts. This gives you the behavioral consistency of fine-tuning with the knowledge freshness of RAG."
        id="note-hybrid"
      />

      <NoteBlock
        type="intuition"
        title="The Librarian Analogy"
        content="RAG is like hiring a librarian to find relevant books before asking someone a question. But if the question is 'what is 2+2?', the librarian adds no value. If the question is about the librarian's own writing style, they need practice (fine-tuning), not books. RAG is powerful for the right problems, but not every problem is a retrieval problem."
        id="note-librarian"
      />
    </div>
  )
}
