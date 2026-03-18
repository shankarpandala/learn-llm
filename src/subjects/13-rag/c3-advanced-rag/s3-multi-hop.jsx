import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function MultiHop() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Multi-Hop Reasoning and Iterative Retrieval</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Many real-world questions cannot be answered from a single retrieval step. Multi-hop
        reasoning requires the system to retrieve information, reason about it, and then
        retrieve again based on what it learned. This iterative process mirrors how humans
        research complex topics by following chains of evidence.
      </p>

      <DefinitionBlock
        title="Multi-Hop Retrieval"
        definition="Multi-hop retrieval iteratively refines the query and retrieves new documents at each step: $q_0 \to D_1 \to q_1 = f(q_0, D_1) \to D_2 \to q_2 = f(q_1, D_2) \to \dots \to a = g(q_0, D_1, D_2, \dots)$. Each hop's query is informed by the documents retrieved in previous hops."
        id="def-multi-hop"
      />

      <h2 className="text-2xl font-semibold">When Single-Hop Fails</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Consider the question: "Which company acquired the startup that developed the embedding
        model used by LangChain's default retriever?" Answering this requires multiple lookups:
        first finding which embedding model LangChain uses, then finding which startup built it,
        then finding the acquiring company.
      </p>

      <ExampleBlock
        title="Multi-Hop Question Decomposition"
        problem="Answer: 'Did the inventor of the transformer architecture work at the same company as the creator of Word2Vec?'"
        steps={[
          { formula: 'q_1 = \\text{"Who invented the transformer architecture?"}', explanation: 'First hop: retrieve documents about transformer origins.' },
          { formula: 'D_1 \\to \\text{Vaswani et al. at Google Brain}', explanation: 'Extract the answer: Ashish Vaswani et al. at Google.' },
          { formula: 'q_2 = \\text{"Who created Word2Vec and where?"}', explanation: 'Second hop: retrieve documents about Word2Vec origins.' },
          { formula: 'D_2 \\to \\text{Tomas Mikolov at Google}', explanation: 'Extract: Mikolov at Google. Both were at Google, so the answer is yes.' },
        ]}
        id="example-multi-hop"
      />

      <PythonCode
        title="iterative_retrieval.py"
        code={`# Multi-hop iterative retrieval with LangChain
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
print("Final:", result)`}
        id="code-multi-hop"
      />

      <NoteBlock
        type="intuition"
        title="Multi-Hop as Research"
        content="Multi-hop RAG mimics how a researcher works: read a paper, note a reference, look up that reference, find another lead, follow it. Each step builds on prior knowledge. The key is knowing when to stop iterating - a fixed maximum of 3-4 hops prevents runaway chains while handling most real questions."
        id="note-research-analogy"
      />

      <WarningBlock
        title="Error Propagation in Multi-Hop"
        content="Each hop introduces the possibility of error. If hop 1 retrieves the wrong document, all subsequent hops build on incorrect context. Multi-hop systems need robust error handling: verify intermediate answers, use multiple retrieval paths, and consider backtracking when confidence is low."
        id="warning-error-propagation"
      />

      <NoteBlock
        type="note"
        title="IRCoT: Interleaving Retrieval with Chain-of-Thought"
        content="IRCoT (Trivedi et al., 2023) interleaves chain-of-thought reasoning with retrieval. After each reasoning step, the model generates a retrieval query based on what information it still needs. This is more flexible than pre-decomposition because the retrieval strategy adapts dynamically to what the model discovers at each step."
        id="note-ircot"
      />
    </div>
  )
}
