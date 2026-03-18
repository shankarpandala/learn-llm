import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function TaskEval() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Task-Specific Evaluation</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        General benchmarks measure broad capabilities, but production applications require
        task-specific evaluation tailored to your domain. Building custom evaluation pipelines
        ensures models actually perform well on the tasks that matter.
      </p>

      <DefinitionBlock
        title="Task-Specific Evaluation"
        definition="The practice of designing evaluation datasets, metrics, and procedures targeting a specific use case such as summarization, question answering, classification, or retrieval-augmented generation. Metrics like ROUGE, BLEU, F1, and custom rubrics replace generic accuracy."
        id="def-task-eval"
      />

      <h2 className="text-2xl font-semibold">Common Task Metrics</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Different tasks require different metrics. For text generation tasks:
      </p>
      <BlockMath math="\text{ROUGE-L} = \frac{(1 + \beta^2) \cdot R_{lcs} \cdot P_{lcs}}{R_{lcs} + \beta^2 \cdot P_{lcs}}" />
      <p className="text-gray-700 dark:text-gray-300">
        where <InlineMath math="R_{lcs}" /> and <InlineMath math="P_{lcs}" /> are recall and
        precision based on the longest common subsequence. For classification:
      </p>
      <BlockMath math="F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}" />

      <ExampleBlock
        title="Designing a RAG Evaluation"
        problem="Evaluate a retrieval-augmented generation system for a customer support chatbot."
        steps={[
          { formula: '\\text{Retrieval: Recall@k, MRR}', explanation: 'Measure whether the retriever finds relevant documents in the top-k results.' },
          { formula: '\\text{Faithfulness} = \\frac{\\text{claims supported by context}}{\\text{total claims}}', explanation: 'Check that generated answers are grounded in retrieved documents, not hallucinated.' },
          { formula: '\\text{Answer Relevance} = \\cos(\\mathbf{q}, \\mathbf{a})', explanation: 'Measure semantic similarity between the question and the generated answer.' },
          { formula: '\\text{End-to-end: Human + LLM judge}', explanation: 'Combine automated metrics with human evaluation and LLM judge scores.' },
        ]}
        id="example-rag-eval"
      />

      <PythonCode
        title="task_specific_eval.py"
        code={`# Task-specific evaluation framework
from dataclasses import dataclass
from typing import Callable
import json
import numpy as np

@dataclass
class EvalCase:
    input_text: str
    expected: str
    metadata: dict = None

class TaskEvaluator:
    """Custom task evaluation pipeline."""

    def __init__(self, model_fn: Callable, metrics: dict[str, Callable]):
        self.model_fn = model_fn
        self.metrics = metrics
        self.results = []

    def evaluate(self, test_cases: list[EvalCase]) -> dict:
        for case in test_cases:
            prediction = self.model_fn(case.input_text)
            scores = {}
            for name, metric_fn in self.metrics.items():
                scores[name] = metric_fn(prediction, case.expected)
            self.results.append({
                "input": case.input_text,
                "expected": case.expected,
                "prediction": prediction,
                "scores": scores,
            })

        # Aggregate metrics
        agg = {}
        for name in self.metrics:
            values = [r["scores"][name] for r in self.results]
            agg[name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
        return agg

# Example: Summarization evaluation with ROUGE
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def rouge_l(prediction, reference):
    scores = scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure

def length_ratio(prediction, reference):
    return len(prediction.split()) / max(len(reference.split()), 1)

# Build evaluator
evaluator = TaskEvaluator(
    model_fn=lambda x: "model response placeholder",
    metrics={"rouge_l": rouge_l, "length_ratio": length_ratio}
)

# Example: Classification evaluation
from sklearn.metrics import classification_report

def evaluate_classifier(model_fn, test_data):
    """Evaluate LLM as classifier."""
    y_true, y_pred = [], []
    for item in test_data:
        prompt = f"Classify the sentiment: '{item['text']}'\\nAnswer: positive or negative"
        pred = model_fn(prompt).strip().lower()
        y_pred.append(pred)
        y_true.append(item["label"])

    report = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred))
    return report`}
        id="code-task-eval"
      />

      <NoteBlock
        type="tip"
        title="Building Evaluation Datasets"
        content="Start with 50-100 hand-curated examples covering edge cases and common scenarios. Use stratified sampling across categories. Include adversarial examples that test failure modes. Version your eval sets alongside your code, and never let evaluation data leak into training."
        id="note-building-eval"
      />

      <WarningBlock
        title="Goodhart's Law in LLM Evaluation"
        content="When a metric becomes a target, it ceases to be a good metric. Models optimized for ROUGE may produce outputs that game n-gram overlap without being genuinely good summaries. Always pair automated metrics with human evaluation for critical applications."
        id="warning-goodhart"
      />

      <NoteBlock
        type="note"
        title="Evaluation Frameworks"
        content="Tools like RAGAS (for RAG evaluation), DeepEval, and OpenAI Evals provide structured frameworks for task-specific evaluation. They include pre-built metrics for faithfulness, answer relevance, context precision, and more."
        id="note-frameworks"
      />
    </div>
  )
}
