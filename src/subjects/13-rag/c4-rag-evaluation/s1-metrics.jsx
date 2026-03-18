import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function Metrics() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">RAG Evaluation Metrics: Faithfulness, Relevance, Precision@k</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Evaluating RAG systems requires measuring both retrieval quality and generation quality.
        Unlike simple accuracy metrics, RAG evaluation must assess whether the right documents
        were retrieved, whether the generated answer is faithful to those documents, and whether
        the answer is actually relevant to the user's question.
      </p>

      <DefinitionBlock
        title="RAG Evaluation Dimensions"
        definition="RAG evaluation spans three dimensions: (1) Retrieval quality - are the right documents retrieved? Measured by Precision@k and Recall@k. (2) Faithfulness - is the answer grounded in retrieved context? (3) Answer relevance - does the answer address the question? Each can be computed with ground truth labels or via LLM-as-judge."
        id="def-rag-eval"
      />

      <h2 className="text-2xl font-semibold">Retrieval Metrics</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Retrieval metrics evaluate how well the system finds relevant documents:
      </p>
      <BlockMath math="\text{Precision@k} = \frac{|\text{relevant docs in top-}k|}{k}" />
      <BlockMath math="\text{Recall@k} = \frac{|\text{relevant docs in top-}k|}{|\text{total relevant docs}|}" />
      <BlockMath math="\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}" />
      <p className="text-gray-700 dark:text-gray-300">
        Mean Reciprocal Rank (MRR) rewards finding the first relevant document early.
        Normalized Discounted Cumulative Gain (nDCG) extends this to graded relevance:
      </p>
      <BlockMath math="\text{nDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}} = \frac{\sum_{i=1}^{k} \frac{2^{r_i} - 1}{\log_2(i+1)}}{\text{IDCG@k}}" />

      <ExampleBlock
        title="Computing Precision@k and MRR"
        problem="Given 5 retrieved documents where relevance labels are [1, 0, 1, 0, 1], compute metrics."
        steps={[
          { formula: '\\text{Precision@3} = \\frac{|\\{d_1, d_3\\}|}{3} = \\frac{2}{3} \\approx 0.667', explanation: 'Two of the top 3 retrieved documents are relevant.' },
          { formula: '\\text{Precision@5} = \\frac{|\\{d_1, d_3, d_5\\}|}{5} = \\frac{3}{5} = 0.6', explanation: 'Three of all 5 retrieved documents are relevant.' },
          { formula: '\\text{MRR} = \\frac{1}{\\text{rank}_1} = \\frac{1}{1} = 1.0', explanation: 'The first relevant document is at rank 1, so MRR is perfect.' },
          { formula: '\\text{Recall@5} = \\frac{3}{|\\text{total relevant}|}', explanation: 'Recall depends on how many relevant documents exist in the entire corpus.' },
        ]}
        id="example-retrieval-metrics"
      />

      <PythonCode
        title="rag_metrics.py"
        code={`# Computing RAG evaluation metrics
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
print(f"Faithfulness: {score.content}")`}
        id="code-rag-metrics"
      />

      <NoteBlock
        type="note"
        title="LLM-as-Judge for Generation Quality"
        content="Since generation quality (faithfulness, relevance) is hard to measure with traditional metrics, LLM-as-judge approaches use a powerful LLM to evaluate the output. The judge LLM checks if claims are supported by context (faithfulness), if the answer addresses the question (relevance), and if the answer is complete (recall). GPT-4 judgments correlate well with human ratings."
        id="note-llm-judge"
      />

      <WarningBlock
        title="Metric Gaming"
        content="Optimizing for one metric can degrade others. Increasing k improves retrieval recall but may decrease precision and faithfulness (by introducing noisy context). Always evaluate multiple metrics together and prioritize faithfulness for production systems - a wrong answer is worse than no answer."
        id="warning-metric-gaming"
      />
    </div>
  )
}
