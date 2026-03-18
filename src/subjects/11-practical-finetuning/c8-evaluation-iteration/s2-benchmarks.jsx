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
      <h1 className="text-3xl font-bold">Benchmarks: MMLU, HellaSwag, and More</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Standardized benchmarks provide quantitative measures of model capability across reasoning,
        knowledge, and language understanding. Running these before and after finetuning reveals
        whether your changes improved or degraded the model's general abilities.
      </p>

      <DefinitionBlock
        title="LLM Benchmarks"
        definition="LLM benchmarks are standardized test suites that evaluate model capabilities through multiple-choice questions, completion tasks, or open-ended generation. Common benchmarks include MMLU (knowledge), HellaSwag (commonsense), ARC (reasoning), TruthfulQA (factuality), and GSM8K (math)."
        id="def-benchmarks"
      />

      <ExampleBlock
        title="Key Benchmark Suite"
        problem="What does each benchmark measure?"
        steps={[
          { formula: '\\text{MMLU (57 subjects): knowledge breadth}', explanation: 'Multiple-choice questions across STEM, humanities, social science. Tests factual knowledge.' },
          { formula: '\\text{HellaSwag: commonsense reasoning}', explanation: 'Sentence completion requiring physical and social commonsense.' },
          { formula: '\\text{ARC: scientific reasoning}', explanation: 'Grade-school science questions. ARC-Challenge is the harder subset.' },
          { formula: '\\text{GSM8K: math reasoning}', explanation: 'Grade-school math word problems requiring multi-step reasoning.' },
          { formula: '\\text{TruthfulQA: factual accuracy}', explanation: 'Questions designed to test whether models repeat common misconceptions.' },
        ]}
        id="example-benchmarks"
      />

      <PythonCode
        title="run_lm_eval.sh"
        code={`# lm-evaluation-harness is the standard tool for LLM benchmarks
pip install lm-eval

# Run a single benchmark
lm_eval --model hf \\
    --model_args pretrained=./your-model,dtype=bfloat16 \\
    --tasks mmlu \\
    --batch_size 8 \\
    --output_path ./eval-results

# Run the Open LLM Leaderboard benchmark suite
lm_eval --model hf \\
    --model_args pretrained=./your-model,dtype=bfloat16 \\
    --tasks mmlu,hellaswag,arc_challenge,truthfulqa_mc2,gsm8k \\
    --batch_size 4 \\
    --num_fewshot 5 \\
    --output_path ./eval-results

# For QLoRA/LoRA models (load adapter on top of base)
lm_eval --model hf \\
    --model_args pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,\\
peft=./your-lora-adapter,dtype=bfloat16,load_in_4bit=True \\
    --tasks mmlu,hellaswag \\
    --batch_size 4

# View results
cat ./eval-results/results.json | python -m json.tool`}
        id="code-lm-eval"
      />

      <PythonCode
        title="quick_benchmark.py"
        code={`# Quick programmatic benchmark evaluation
import lm_eval

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=./your-model,dtype=bfloat16",
    tasks=["mmlu", "hellaswag"],
    batch_size=8,
    num_fewshot=5,
)

# Print results
for task, metrics in results["results"].items():
    acc = metrics.get("acc,none", metrics.get("acc_norm,none", "N/A"))
    print(f"{task}: {acc:.4f}")

# Compare base vs finetuned
def compare_models(base_path, finetuned_path, tasks):
    """Compare two models on the same benchmarks."""
    base_results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={base_path},dtype=bfloat16",
        tasks=tasks, batch_size=8,
    )
    ft_results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={finetuned_path},dtype=bfloat16",
        tasks=tasks, batch_size=8,
    )

    print(f"{'Task':<20} {'Base':>8} {'Finetuned':>10} {'Delta':>8}")
    print("-" * 50)
    for task in tasks:
        base_acc = base_results["results"][task].get("acc,none", 0)
        ft_acc = ft_results["results"][task].get("acc,none", 0)
        delta = ft_acc - base_acc
        symbol = "+" if delta > 0 else ""
        print(f"{task:<20} {base_acc:>8.4f} {ft_acc:>10.4f} {symbol}{delta:>7.4f}")`}
        id="code-quick-benchmark"
      />

      <NoteBlock
        type="tip"
        title="Benchmark Before and After"
        content="Always run benchmarks on the base model BEFORE finetuning to establish a baseline. Then run the same benchmarks after finetuning. If general benchmarks drop significantly (>2%), you may be experiencing catastrophic forgetting -- the model is losing general knowledge while learning your task."
        id="note-before-after"
      />

      <WarningBlock
        title="Benchmarks Are Not Everything"
        content="High benchmark scores do not guarantee a good model for your use case. Benchmarks test general knowledge and reasoning, but your specific task may require domain expertise, a particular output format, or a specific tone that benchmarks do not measure. Always combine benchmarks with task-specific evaluation."
        id="warning-benchmarks-limit"
      />

      <NoteBlock
        type="note"
        title="Evaluation Time"
        content="Running the full benchmark suite takes 2-8 hours on a single GPU depending on model size and benchmarks selected. For quick iteration, run only MMLU and HellaSwag (~30 minutes). Save full evaluation for final model candidates."
        id="note-eval-time"
      />
    </div>
  )
}
