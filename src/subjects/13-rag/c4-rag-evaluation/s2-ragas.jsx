import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function RAGAS() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">The RAGAS Evaluation Framework</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        RAGAS (Retrieval Augmented Generation Assessment) is a framework for automated
        evaluation of RAG pipelines. It provides reference-free metrics that use LLMs to
        assess faithfulness, answer relevance, and context quality without requiring
        manually labeled ground truth data.
      </p>

      <DefinitionBlock
        title="RAGAS Framework"
        definition="RAGAS evaluates RAG systems across four core metrics: (1) Faithfulness $= \frac{|\text{supported claims}|}{|\text{total claims}|}$, (2) Answer Relevance $= \text{mean}(\text{sim}(q, q_i^{\text{gen}}))$, (3) Context Precision $= \frac{\sum_{k} \text{Precision@}k \cdot v_k}{\text{total relevant}}$, and (4) Context Recall $= \frac{|\text{GT sentences attributable to context}|}{|\text{GT sentences}|}$."
        id="def-ragas"
      />

      <h2 className="text-2xl font-semibold">RAGAS Metrics in Detail</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Each RAGAS metric targets a different failure mode in the RAG pipeline:
      </p>
      <BlockMath math="\text{Faithfulness} = \frac{|V_s|}{|V|} \text{ where } V = \text{claims}(a), \; V_s = \{v \in V : v \text{ supported by } C\}" />
      <BlockMath math="\text{Answer Relevance} = \frac{1}{N} \sum_{i=1}^{N} \text{sim}(e_q, e_{q_i})" />
      <p className="text-gray-700 dark:text-gray-300">
        Answer relevance generates <InlineMath math="N" /> questions from the answer and measures
        their similarity to the original question. High similarity means the answer addresses
        what was asked.
      </p>

      <ExampleBlock
        title="RAGAS Evaluation Workflow"
        problem="Evaluate a RAG pipeline using all four RAGAS metrics."
        steps={[
          { formula: '\\text{Input: } (q, C, a, a^*)', explanation: 'Each sample needs: question, retrieved context, generated answer, and optionally ground truth.' },
          { formula: '\\text{Faithfulness: extract claims from } a, \\text{ verify against } C', explanation: 'LLM extracts factual claims from the answer and checks each against the context.' },
          { formula: '\\text{Relevance: generate questions from } a, \\text{ compare to } q', explanation: 'If the answer is relevant, questions generated from it should resemble the original query.' },
          { formula: '\\text{Context Precision: rank relevant contexts higher}', explanation: 'Measures whether the most relevant context chunks appear at the top of the retrieval results.' },
        ]}
        id="example-ragas-workflow"
      />

      <PythonCode
        title="ragas_evaluation.py"
        code={`# Evaluating a RAG pipeline with RAGAS
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
print(df[["question", "faithfulness", "answer_relevancy"]].to_string())`}
        id="code-ragas"
      />

      <PythonCode
        title="ragas_test_generation.py"
        code={`# Generate synthetic test data with RAGAS
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
print(test_df[["question", "evolution_type"]].head(10))`}
        id="code-ragas-testgen"
      />

      <NoteBlock
        type="tip"
        title="Continuous Evaluation"
        content="Run RAGAS evaluation as part of your CI/CD pipeline. Track metrics over time to catch regressions when you change chunking strategies, embedding models, or prompts. Set minimum thresholds (e.g., faithfulness > 0.8) as quality gates before deploying RAG pipeline changes."
        id="note-continuous-eval"
      />

      <WarningBlock
        title="RAGAS Limitations"
        content="RAGAS metrics rely on LLM judgments, which can be inconsistent and biased. The faithfulness metric may miss subtle hallucinations, and answer relevancy can be fooled by paraphrasing. Always supplement automated RAGAS evaluation with periodic human evaluation on a representative sample."
        id="warning-ragas-limitations"
      />
    </div>
  )
}
