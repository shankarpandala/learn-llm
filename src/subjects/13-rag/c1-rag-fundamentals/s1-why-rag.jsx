import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function WhyRAG() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Why RAG: Motivation and Hallucination Reduction</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Large language models store knowledge implicitly in their parameters, but this knowledge
        can be outdated, incomplete, or simply wrong. Retrieval-Augmented Generation (RAG)
        addresses these limitations by grounding model outputs in retrieved evidence, dramatically
        reducing hallucinations and enabling up-to-date, verifiable responses.
      </p>

      <DefinitionBlock
        title="Retrieval-Augmented Generation (RAG)"
        definition="RAG is a technique that augments an LLM's input prompt with relevant documents retrieved from an external knowledge base, allowing the model to generate responses grounded in specific, verifiable sources rather than relying solely on parametric memory."
        id="def-rag"
      />

      <h2 className="text-2xl font-semibold">The Hallucination Problem</h2>
      <p className="text-gray-700 dark:text-gray-300">
        LLMs generate plausible-sounding text even when they lack factual knowledge. This happens
        because the training objective optimizes for next-token probability, not factual accuracy.
        The probability of a hallucinated output can be modeled as:
      </p>
      <BlockMath math="P(\text{hallucination}) = 1 - P(\text{fact} \in \theta)" />
      <p className="text-gray-700 dark:text-gray-300">
        where <InlineMath math="\theta" /> represents the model's parametric knowledge. RAG
        reduces this by conditioning on retrieved context <InlineMath math="C" />:
      </p>
      <BlockMath math="P(y \mid x) = \sum_{d \in \mathcal{D}} P(y \mid x, d) \cdot P(d \mid x)" />

      <ExampleBlock
        title="Parametric vs. RAG-Augmented Response"
        problem="Ask an LLM about a company's Q3 2025 earnings without and with RAG."
        steps={[
          { formula: '\\text{Without RAG: } P(y \\mid x) = P(y \\mid x; \\theta)', explanation: 'The model relies on training data, which may predate Q3 2025.' },
          { formula: '\\text{With RAG: } P(y \\mid x, C) = P(y \\mid x, \\{d_1, d_2, \\dots\\}; \\theta)', explanation: 'Retrieved earnings reports provide factual grounding.' },
          { formula: '\\text{Faithfulness} = \\frac{|\\text{claims supported by } C|}{|\\text{total claims}|}', explanation: 'RAG increases faithfulness by providing verifiable source material.' },
        ]}
        id="example-parametric-vs-rag"
      />

      <PythonCode
        title="basic_rag_motivation.py"
        code={`# Demonstrating why RAG matters: grounding LLM responses
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
# Accurate, grounded in the provided document`}
        id="code-rag-motivation"
      />

      <NoteBlock
        type="intuition"
        title="RAG as Open-Book vs. Closed-Book Exam"
        content="Think of a vanilla LLM as taking a closed-book exam: it can only rely on what it memorized during training. RAG transforms this into an open-book exam where the model can look up relevant reference material before answering. The model still needs reasoning ability, but it no longer needs to memorize every fact."
        id="note-open-book"
      />

      <h2 className="text-2xl font-semibold">Key Benefits of RAG</h2>
      <p className="text-gray-700 dark:text-gray-300">
        RAG provides several advantages over pure parametric approaches: knowledge can be updated
        without retraining, responses are traceable to source documents, and domain-specific
        knowledge can be injected without fine-tuning. The cost of updating a vector store is
        orders of magnitude lower than retraining a model.
      </p>

      <WarningBlock
        title="RAG Is Not a Silver Bullet"
        content="RAG only helps when the answer exists in the knowledge base and is retrievable. Poor chunking, bad embeddings, or irrelevant retrieval can actually degrade performance compared to a well-trained base model. Always evaluate whether retrieved context improves or confuses the generation."
        id="warning-not-silver-bullet"
      />

      <NoteBlock
        type="historical"
        title="Origins of RAG"
        content="The RAG framework was introduced by Lewis et al. at Facebook AI Research in 2020. The original paper combined a pre-trained seq2seq model (BART) with a dense retriever (DPR) and showed significant improvements on knowledge-intensive tasks like open-domain QA. The approach has since been adopted industry-wide as a standard pattern for production LLM applications."
        id="note-rag-history"
      />
    </div>
  )
}
