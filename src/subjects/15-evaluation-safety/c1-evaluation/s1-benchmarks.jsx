import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function Benchmarks() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LLM Benchmarks: MMLU, HellaSwag, HumanEval, GPQA</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Benchmarks provide standardized ways to measure LLM capabilities across knowledge,
        reasoning, code generation, and expert-level problem solving. Understanding what each
        benchmark measures and its limitations is essential for interpreting model comparisons.
      </p>

      <DefinitionBlock
        title="MMLU (Massive Multitask Language Understanding)"
        definition="A benchmark consisting of 57 subjects across STEM, humanities, social sciences, and more, with 14,042 multiple-choice questions testing knowledge and reasoning from elementary to professional level."
        id="def-mmlu"
      />

      <DefinitionBlock
        title="HellaSwag"
        definition="A commonsense reasoning benchmark where models must select the most plausible continuation of a scenario. Despite appearing simple to humans (95%+ accuracy), it tests grounded commonsense knowledge that challenges language models."
        id="def-hellaswag"
      />

      <DefinitionBlock
        title="HumanEval"
        definition="A code generation benchmark with 164 hand-written Python programming problems. Each problem includes a function signature, docstring, and unit tests. The metric pass@k measures whether at least one of k generated samples passes all tests."
        id="def-humaneval"
      />

      <DefinitionBlock
        title="GPQA (Graduate-Level Google-Proof Q&A)"
        definition="An expert-level benchmark containing 448 multiple-choice questions in biology, physics, and chemistry designed so that even with internet access, non-experts struggle to answer correctly."
        id="def-gpqa"
      />

      <h2 className="text-2xl font-semibold">Benchmark Metrics</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Each benchmark uses specific metrics. MMLU and GPQA report accuracy on multiple-choice
        questions. HumanEval uses the <InlineMath math="pass@k" /> metric:
      </p>
      <BlockMath math="pass@k = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]" />
      <p className="text-gray-700 dark:text-gray-300">
        where <InlineMath math="n" /> is the total number of generated samples and{' '}
        <InlineMath math="c" /> is the number that pass all tests.
      </p>

      <ExampleBlock
        title="Computing pass@k"
        problem="A model generates n=10 code samples for a problem and c=3 pass all tests. What is pass@1?"
        steps={[
          { formula: 'pass@1 = 1 - \\frac{\\binom{10-3}{1}}{\\binom{10}{1}}', explanation: 'Plug in n=10, c=3, k=1.' },
          { formula: 'pass@1 = 1 - \\frac{7}{10} = 0.3', explanation: 'There is a 30% chance a single random sample passes.' },
        ]}
        id="example-passk"
      />

      <PythonCode
        title="running_lm_eval_harness.py"
        code={`# Using lm-eval-harness (EleutherAI) to evaluate models on benchmarks
# Install: pip install lm-eval

# Command-line usage:
# lm_eval --model hf \\
#     --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \\
#     --tasks mmlu,hellaswag,humaneval \\
#     --batch_size 8 \\
#     --output_path results/

# Python API usage:
import lm_eval

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Llama-3.1-8B-Instruct",
    tasks=["mmlu", "hellaswag"],
    batch_size=8,
    num_fewshot=5,  # 5-shot for MMLU standard
)

# Extract scores
for task_name, task_result in results["results"].items():
    acc = task_result.get("acc,none", task_result.get("acc_norm,none", "N/A"))
    print(f"{task_name}: {acc:.4f}")

# Custom task evaluation with GPQA
results_gpqa = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Llama-3.1-8B-Instruct",
    tasks=["gpqa_main"],
    batch_size=4,
)
print(f"GPQA: {results_gpqa['results']['gpqa_main']['acc,none']:.4f}")`}
        id="code-lm-eval"
      />

      <NoteBlock
        type="historical"
        title="Evolution of LLM Benchmarks"
        content="MMLU (Hendrycks et al., 2021) became the standard knowledge benchmark. HellaSwag (Zellers et al., 2019) pioneered adversarial dataset construction via Adversarial Filtering. HumanEval (Chen et al., 2021) established code generation evaluation alongside Codex. GPQA (Rein et al., 2023) pushed toward expert-level evaluation as frontier models saturated earlier benchmarks."
        id="note-benchmark-history"
      />

      <WarningBlock
        title="Benchmark Saturation and Contamination"
        content="As models improve, benchmarks saturate: MMLU scores now exceed 90% for frontier models. Additionally, data contamination (benchmark questions appearing in training data) inflates scores. Always consider whether a benchmark still discriminates between model capabilities, and look for contamination analyses in model reports."
        id="warning-saturation"
      />

      <NoteBlock
        type="tip"
        title="Choosing the Right Benchmark"
        content="Use MMLU for broad knowledge assessment, HellaSwag for commonsense reasoning, HumanEval/MBPP for code generation, and GPQA for expert-level science reasoning. For production use cases, always supplement standard benchmarks with domain-specific evaluations."
        id="note-choosing"
      />
    </div>
  )
}
