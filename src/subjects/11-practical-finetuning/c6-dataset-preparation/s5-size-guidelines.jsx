import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function SizeGuidelines() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">How Much Data Do You Need?</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        One of the most common questions in finetuning is about dataset size. The answer depends on
        your goal, data quality, model size, and the complexity of the target behavior. This section
        provides practical guidelines based on empirical results from the community.
      </p>

      <DefinitionBlock
        title="Scaling Laws for Finetuning"
        definition="Unlike pretraining where more data almost always helps, finetuning exhibits diminishing returns and risks overfitting. For instruction tuning, the LIMA paper showed that 1,000 high-quality examples can match models trained on 52K lower-quality examples. The relationship is roughly $\\text{quality} \\propto \\sqrt{n \\cdot q}$ where $n$ is dataset size and $q$ is quality."
        id="def-scaling"
      />

      <ExampleBlock
        title="Dataset Size by Use Case"
        problem="How many examples do you need for different finetuning goals?"
        steps={[
          { formula: '\\text{Style/tone adaptation: } 50\\text{-}200', explanation: 'Teaching a model to write in a specific voice or persona. Very few examples needed.' },
          { formula: '\\text{Single task (classification, extraction): } 200\\text{-}2\\text{K}', explanation: 'Narrow task with clear input/output patterns.' },
          { formula: '\\text{Domain instruction following: } 1\\text{K-}10\\text{K}', explanation: 'Teaching domain expertise (legal, medical, coding in specific frameworks).' },
          { formula: '\\text{General instruction improvement: } 10\\text{K-}100\\text{K}', explanation: 'Broadly improving helpfulness, reasoning, and instruction following.' },
          { formula: '\\text{Continued pretraining: } 1\\text{M-}10\\text{B tokens}', explanation: 'Adapting to a new language or domain requires much more data.' },
        ]}
        id="example-size-by-use"
      />

      <PythonCode
        title="dataset_size_experiment.py"
        code={`import numpy as np

def estimate_training_time(
    dataset_size,
    avg_tokens_per_example=512,
    batch_size=8,
    epochs=2,
    tokens_per_second=5000,  # Typical for QLoRA on RTX 4090
):
    """Estimate training time for a given dataset size."""
    total_tokens = dataset_size * avg_tokens_per_example * epochs
    total_steps = (dataset_size * epochs) / batch_size
    time_seconds = total_tokens / tokens_per_second
    time_hours = time_seconds / 3600
    return {
        "total_tokens": total_tokens,
        "total_steps": int(total_steps),
        "time_hours": time_hours,
    }

# Compare different dataset sizes
print(f"{'Size':>10} {'Tokens':>12} {'Steps':>8} {'Time':>10}")
print("-" * 45)
for size in [100, 500, 1000, 5000, 10000, 50000, 100000]:
    result = estimate_training_time(size)
    print(f"{size:>10,} {result['total_tokens']:>12,} "
          f"{result['total_steps']:>8,} {result['time_hours']:>9.1f}h")

# Output:
#       100      102,400      25      0.0h
#       500      512,000     125      0.0h
#     1,000    1,024,000     250      0.1h
#     5,000    5,120,000   1,250      0.3h
#    10,000   10,240,000   2,500      0.6h
#    50,000   51,200,000  12,500      2.8h
#   100,000  102,400,000  25,000      5.7h`}
        id="code-estimate-time"
      />

      <PythonCode
        title="optimal_epochs.py"
        code={`# How many epochs to train for different dataset sizes

def recommend_epochs(dataset_size, task_type="instruction"):
    """Recommend number of training epochs."""
    if task_type == "instruction":
        if dataset_size < 500:
            return 3, "Small dataset: more epochs to learn patterns"
        elif dataset_size < 5000:
            return 2, "Medium dataset: 2 epochs balances learning and overfitting"
        elif dataset_size < 50000:
            return 1, "Large dataset: 1 epoch usually sufficient"
        else:
            return 1, "Very large: may not even need full epoch"
    elif task_type == "continued_pretrain":
        return 1, "Continued pretraining: always 1 epoch to avoid repetition"
    elif task_type == "classification":
        if dataset_size < 1000:
            return 5, "Small classification: more epochs with early stopping"
        else:
            return 3, "Classification: 3 epochs with validation monitoring"

for size in [100, 500, 2000, 10000, 50000]:
    epochs, reason = recommend_epochs(size)
    print(f"Dataset {size:>6,}: {epochs} epoch(s) - {reason}")

# Key insight: number of gradient updates matters more than epochs
# target_steps = dataset_size * epochs / batch_size
# Aim for 500-5000 total steps for most instruction tuning tasks`}
        id="code-epochs"
      />

      <NoteBlock
        type="intuition"
        title="The LIMA Insight"
        content="The LIMA paper (2023) demonstrated that a model finetuned on just 1,000 carefully curated examples could match models trained on 52,000 examples. The key was extreme quality: each example was handwritten by researchers to be clear, complete, and diverse. This suggests that data curation effort yields higher returns than data collection volume."
        id="note-lima"
      />

      <NoteBlock
        type="tip"
        title="Practical Starting Point"
        content="Start with 1,000-2,000 high-quality examples and train for 2 epochs. Evaluate results. If the model performs well on most tasks but struggles with specific ones, add 200-500 targeted examples for those areas. This iterative approach is more effective than training on a massive dataset from the start."
        id="note-starting-point"
      />

      <WarningBlock
        title="Overfitting Signs"
        content="With small datasets, overfitting is the primary risk. Signs: training loss drops below 0.3, model outputs memorized responses verbatim, model performs well on training-like prompts but poorly on novel ones. Mitigations: fewer epochs, higher dropout, lower rank, data augmentation."
        id="warning-overfitting"
      />
    </div>
  )
}
