import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function FailureModes() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Common Failure Modes & Debugging</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Fine-tuning often fails in subtle ways. The model trains without errors but produces
        bad outputs. This section catalogs the most common failure modes, their symptoms,
        and practical debugging strategies to identify and fix them.
      </p>

      <DefinitionBlock
        title="Catastrophic Forgetting"
        definition="Catastrophic forgetting occurs when fine-tuning overwrites the model's pre-trained knowledge. The model performs well on the fine-tuning task but loses general capabilities. Formally, if $\mathcal{L}_{\text{general}}$ increases significantly while $\mathcal{L}_{\text{task}}$ decreases, the model has forgotten."
        id="def-catastrophic-forgetting"
      />

      <ExampleBlock
        title="Failure Mode Checklist"
        problem="What are the most common fine-tuning failures and their symptoms?"
        steps={[
          { formula: '\\text{Repetition loops: model repeats phrases endlessly}', explanation: 'Caused by too high a learning rate or too many epochs. Reduce LR or add repetition penalty.' },
          { formula: '\\text{Format collapse: wrong output format}', explanation: 'Model ignores chat template or outputs raw tokens. Check tokenizer chat_template and training format.' },
          { formula: '\\text{Hallucination increase: more confident lies}', explanation: 'Fine-tuning on noisy data teaches the model to generate plausible-sounding but incorrect text.' },
          { formula: '\\text{Catastrophic forgetting: lost general knowledge}', explanation: 'Too many epochs on narrow data. Use LoRA, lower LR, or mix in general data.' },
          { formula: '\\text{Mode collapse: same response to different inputs}', explanation: 'Training data too homogeneous. Increase dataset diversity or reduce epochs.' },
        ]}
        id="example-failure-checklist"
      />

      <PythonCode
        title="debug_model_outputs.py"
        code={`from transformers import pipeline
import torch

def diagnose_model(model_path, test_prompts):
    """Run diagnostic tests on a fine-tuned model."""
    pipe = pipeline(
        "text-generation", model=model_path,
        torch_dtype=torch.float16, device_map="auto"
    )

    issues = []

    for prompt in test_prompts:
        output = pipe(
            prompt, max_new_tokens=256, temperature=0.7,
            do_sample=True, return_full_text=False
        )[0]["generated_text"]

        # Check for repetition
        words = output.split()
        if len(words) > 20:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            unique_ratio = len(set(trigrams)) / len(trigrams) if trigrams else 1
            if unique_ratio < 0.5:
                issues.append(
                    f"REPETITION detected (unique trigram ratio: {unique_ratio:.2f})")
                issues.append(f"  Prompt: {prompt[:80]}...")

        # Check for empty or very short responses
        if len(output.strip()) < 10:
            issues.append(f"EMPTY/SHORT response: '{output.strip()}'")
            issues.append(f"  Prompt: {prompt[:80]}...")

        # Check for leaked special tokens
        special_tokens = ["<|", "</s>", "[INST]", "<s>"]
        for token in special_tokens:
            if token in output:
                issues.append(f"SPECIAL TOKEN leaked: '{token}' in output")
                break

        # Check for mode collapse
        outputs_set = set()
        for _ in range(3):
            o = pipe(prompt, max_new_tokens=100, temperature=0.7,
                     do_sample=True, return_full_text=False)[0]["generated_text"]
            outputs_set.add(o.strip())
        if len(outputs_set) == 1:
            issues.append(f"MODE COLLAPSE: identical outputs for: {prompt[:60]}...")

    if not issues:
        print("No obvious issues detected.")
    else:
        print(f"Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")

    return issues

test_prompts = [
    "What is the capital of France?",
    "Write a haiku about the ocean.",
    "Explain quantum entanglement simply.",
    "def fibonacci(n):",
    "Translate to Spanish: The weather is nice today.",
]

diagnose_model("./my-finetuned-model", test_prompts)`}
        id="code-diagnose"
      />

      <PythonCode
        title="check_training_data.py"
        code={`import json
from collections import Counter

def audit_training_data(data_path):
    """Check training data for common issues."""
    with open(data_path) as f:
        data = [json.loads(line) for line in f]

    print(f"Total samples: {len(data)}")

    # Check for duplicates
    texts = [json.dumps(d, sort_keys=True) for d in data]
    dupes = len(texts) - len(set(texts))
    if dupes > 0:
        print(f"WARNING: {dupes} duplicate samples ({dupes/len(data)*100:.1f}%)")

    # Check response lengths
    lengths = []
    for d in data:
        resp = d.get("output", d.get("response", d.get("completion", "")))
        lengths.append(len(resp.split()))

    print(f"Response length: min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.0f}")

    empty = sum(1 for l in lengths if l == 0)
    if empty > 0:
        print(f"WARNING: {empty} samples with empty responses")

    if "label" in data[0]:
        labels = Counter(d["label"] for d in data)
        print(f"Label distribution: {dict(labels)}")
        max_ratio = max(labels.values()) / min(labels.values())
        if max_ratio > 10:
            print(f"WARNING: Imbalanced labels (ratio {max_ratio:.1f}:1)")

    short = sum(
        1 for d in data
        if len(str(d.get("input", d.get("instruction", "")))) < 10
    )
    if short > 0:
        print(f"WARNING: {short} samples with very short inputs")

audit_training_data("training_data.jsonl")`}
        id="code-audit-data"
      />

      <WarningBlock
        title="The Training Data Is Almost Always the Problem"
        content="When a fine-tuned model produces bad outputs, the root cause is almost always the training data -- not the hyperparameters. Before tuning learning rates or LoRA rank, manually inspect 50-100 training examples for quality issues: formatting errors, incorrect labels, low-quality responses, or mismatched instruction-response pairs."
        id="warning-data-first"
      />

      <NoteBlock
        type="tip"
        title="Quick Debugging Checklist"
        content="1) Check a few training examples manually. 2) Compare base model vs fine-tuned outputs on the same prompts. 3) Look at the loss curve -- is it still decreasing or has it plateaued? 4) Test with temperature=0 to see deterministic output. 5) Check if the chat template matches between training and inference."
        id="note-debug-checklist"
      />

      <NoteBlock
        type="note"
        title="Loss Can Be Misleading"
        content="A low training loss does not guarantee good model quality. The model could be memorizing training data verbatim (overfitting) or learning superficial patterns. Always evaluate on held-out examples that were not in the training set."
        id="note-loss-misleading"
      />
    </div>
  )
}
