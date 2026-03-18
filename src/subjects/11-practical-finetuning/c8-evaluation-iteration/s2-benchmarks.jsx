import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Benchmarks() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Task-Specific Benchmarks (MMLU, HellaSwag)</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Standardized benchmarks provide comparable evaluation across models. MMLU tests
        knowledge and reasoning across 57 subjects, HellaSwag tests commonsense reasoning,
        and many other benchmarks target specific capabilities. Running these on your
        fine-tuned model helps detect regressions and measure improvement.
      </p>

      <DefinitionBlock
        title="MMLU (Massive Multitask Language Understanding)"
        definition="MMLU is a benchmark consisting of 14,042 multiple-choice questions spanning 57 subjects from elementary to professional level. Performance is measured as accuracy: $\text{Acc} = \frac{\text{correct answers}}{\text{total questions}}$. State-of-the-art models score above 85%, while random chance yields 25%."
        id="def-mmlu"
      />

      <DefinitionBlock
        title="HellaSwag"
        definition="HellaSwag is a commonsense natural language inference benchmark where the model must select the most plausible continuation of a scenario from four choices. It tests grounded commonsense reasoning with adversarially filtered wrong answers."
        id="def-hellaswag"
      />

      <ExampleBlock
        title="Common Benchmark Suite"
        problem="Which benchmarks should you run after fine-tuning?"
        steps={[
          { formula: '\\text{MMLU: knowledge breadth}', explanation: 'Tests whether fine-tuning preserved general knowledge (watch for regression).' },
          { formula: '\\text{HellaSwag: commonsense}', explanation: 'Tests commonsense reasoning -- should remain stable after domain fine-tuning.' },
          { formula: '\\text{TruthfulQA: hallucination}', explanation: 'Tests whether the model generates truthful answers vs plausible-sounding falsehoods.' },
          { formula: '\\text{HumanEval / MBPP: coding}', explanation: 'For code-focused fine-tunes, measure pass@k on code generation benchmarks.' },
          { formula: '\\text{MT-Bench: conversation}', explanation: 'Multi-turn benchmark scored by GPT-4 to evaluate chat quality (1-10 scale).' },
        ]}
        id="example-benchmark-suite"
      />

      <PythonCode
        title="run_lm_eval.py"
        code={`# lm-evaluation-harness is the standard tool for running benchmarks
# Install: pip install lm-eval

# Run MMLU (5-shot) from command line
# lm_eval --model hf \\
#     --model_args pretrained=./my-finetuned-model \\
#     --tasks mmlu \\
#     --num_fewshot 5 \\
#     --batch_size 4 \\
#     --output_path ./eval_results/mmlu

# Run multiple benchmarks at once
# lm_eval --model hf \\
#     --model_args pretrained=./my-finetuned-model,dtype=float16 \\
#     --tasks mmlu,hellaswag,truthfulqa_mc2,winogrande,arc_challenge \\
#     --batch_size auto \\
#     --output_path ./eval_results/full_suite

# Python API usage
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

model = HFLM(
    pretrained="./my-finetuned-model",
    dtype="float16",
    batch_size=4,
)

results = evaluator.simple_evaluate(
    model=model,
    tasks=["mmlu", "hellaswag", "truthfulqa_mc2"],
    num_fewshot=5,
    batch_size=4,
)

# Print results
for task, metrics in results["results"].items():
    acc = metrics.get("acc,none", metrics.get("acc_norm,none", "N/A"))
    print(f"{task}: {acc:.4f}")

# Save results to JSON
import json
with open("eval_results.json", "w") as f:
    json.dump(results["results"], f, indent=2)`}
        id="code-lm-eval"
      />

      <PythonCode
        title="compare_base_vs_finetuned.py"
        code={`import json
import os

def compare_results(base_path, ft_path):
    """Compare benchmark results between base and finetuned models."""
    with open(base_path) as f:
        base = json.load(f)
    with open(ft_path) as f:
        ft = json.load(f)

    print(f"{'Benchmark':<25} {'Base':>8} {'Finetuned':>10} {'Delta':>8}")
    print("-" * 55)

    for task in base:
        if task not in ft:
            continue
        base_acc = base[task].get("acc,none", base[task].get("acc_norm,none", 0))
        ft_acc = ft[task].get("acc,none", ft[task].get("acc_norm,none", 0))
        delta = ft_acc - base_acc
        marker = "+" if delta > 0 else ""
        print(f"{task:<25} {base_acc:>8.4f} {ft_acc:>10.4f} {marker}{delta:>7.4f}")

        if delta < -0.02:
            print(f"  ** WARNING: regression of {abs(delta)*100:.1f}% on {task}")

# Example output:
# Benchmark                     Base  Finetuned    Delta
# -------------------------------------------------------
# mmlu                        0.6820     0.6910  +0.0090
# hellaswag                   0.8210     0.8250  +0.0040
# truthfulqa_mc2              0.5100     0.5350  +0.0250

compare_results("eval_results_base.json", "eval_results_ft.json")`}
        id="code-compare"
      />

      <NoteBlock
        type="tip"
        title="Use the Open LLM Leaderboard Format"
        content="If you want to submit your model to the Hugging Face Open LLM Leaderboard, run the exact tasks and settings they specify: MMLU (5-shot), HellaSwag (10-shot), TruthfulQA (0-shot), Winogrande (5-shot), GSM8K (5-shot), and ARC-Challenge (25-shot)."
        id="note-leaderboard"
      />

      <WarningBlock
        title="Benchmark Contamination"
        content="If your training data contains benchmark questions or answers, your scores will be inflated and meaningless. Always check for data contamination by searching your training set for benchmark examples. The lm-evaluation-harness has decontamination tools to help with this."
        id="warning-contamination"
      />

      <NoteBlock
        type="note"
        title="Benchmark Limitations"
        content="Multiple-choice benchmarks like MMLU test recognition rather than generation. A model might score well on MMLU but still generate poor free-form text. Use benchmarks as one signal among many, not as the sole evaluation criterion."
        id="note-limitations"
      />
    </div>
  )
}
