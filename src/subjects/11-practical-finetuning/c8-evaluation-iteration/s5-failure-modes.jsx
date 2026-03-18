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
      <h1 className="text-3xl font-bold">Common Finetuning Failure Modes</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Finetuning can go wrong in many ways. Understanding common failure modes and their symptoms
        helps you diagnose issues quickly and apply the right fix. This section catalogs the most
        frequent problems practitioners encounter.
      </p>

      <DefinitionBlock
        title="Catastrophic Forgetting"
        definition="Catastrophic forgetting occurs when finetuning on a narrow dataset causes the model to lose previously learned general knowledge. The model becomes good at the finetuning task but worse at everything else. PEFT methods like LoRA mitigate this by only updating a small fraction of parameters."
        id="def-catastrophic-forgetting"
      />

      <ExampleBlock
        title="Failure Mode Diagnosis"
        problem="How to identify and fix common finetuning failures?"
        steps={[
          { formula: '\\text{Repeating text / looping}', explanation: 'Cause: overtrained or wrong generation params. Fix: reduce epochs, lower temperature, use repetition_penalty=1.1.' },
          { formula: '\\text{Ignoring instructions}', explanation: 'Cause: wrong chat template or poor data formatting. Fix: verify template matches model, check data formatting.' },
          { formula: '\\text{Generic/bland responses}', explanation: 'Cause: too many similar examples, low diversity. Fix: diversify dataset, reduce epochs.' },
          { formula: '\\text{Hallucinating facts}', explanation: 'Cause: training on inaccurate data. Fix: clean dataset, add factual verification step.' },
          { formula: '\\text{Loss not decreasing}', explanation: 'Cause: LR too low, data issue, or wrong template. Fix: increase LR, verify data format.' },
        ]}
        id="example-diagnosis"
      />

      <PythonCode
        title="diagnose_failures.py"
        code={`import torch
from collections import Counter

def diagnose_model_output(model, tokenizer, test_prompts, max_tokens=256):
    """Run diagnostic tests on a finetuned model."""
    issues = []

    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids=inputs, max_new_tokens=max_tokens,
            temperature=0.7, do_sample=True,
        )
        response = tokenizer.decode(outputs[0][inputs.shape[1]:],
                                     skip_special_tokens=True)

        # Check for repetition
        words = response.split()
        if len(words) > 10:
            ngrams = [" ".join(words[i:i+5]) for i in range(len(words)-4)]
            ngram_counts = Counter(ngrams)
            max_repeat = max(ngram_counts.values()) if ngram_counts else 0
            if max_repeat > 3:
                issues.append(f"REPETITION: '{prompt[:40]}...' - 5-gram repeated {max_repeat}x")

        # Check for empty/very short response
        if len(response.strip()) < 10:
            issues.append(f"EMPTY: '{prompt[:40]}...' - response only {len(response)} chars")

        # Check for prompt regurgitation
        overlap = len(set(prompt.lower().split()) & set(response.lower().split()[:20]))
        if overlap > len(prompt.split()) * 0.8:
            issues.append(f"ECHO: '{prompt[:40]}...' - response echoes the prompt")

        # Check for template leakage
        template_tokens = ["<|", "|>", "[INST]", "[/INST]", "<s>", "</s>"]
        for tok in template_tokens:
            if tok in response:
                issues.append(f"TEMPLATE LEAK: '{prompt[:40]}...' - found '{tok}' in response")

    if issues:
        print(f"Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No obvious issues detected")

    return issues

# Diagnostic test prompts
diagnostic_prompts = [
    "Hello, how are you?",
    "List 3 colors.",
    "What is 2 + 2?",
    "Write a haiku about rain.",
    "Explain photosynthesis in one sentence.",
]

# issues = diagnose_model_output(model, tokenizer, diagnostic_prompts)`}
        id="code-diagnose"
      />

      <PythonCode
        title="fix_common_issues.py"
        code={`# Fixes for common finetuning failures

# Issue 1: Model outputs garbage / wrong language
# -> Wrong tokenizer or chat template
# Fix: ensure tokenizer matches the base model exactly
# tokenizer = AutoTokenizer.from_pretrained(SAME_MODEL_AS_BASE)

# Issue 2: Training loss goes to NaN
# -> Numerical instability
# Fix:
training_fixes_nan = {
    "bf16": True,           # Use bf16 instead of fp16
    "max_grad_norm": 0.3,   # Aggressive gradient clipping
    "learning_rate": 1e-4,  # Reduce learning rate
    "warmup_steps": 20,     # Add warmup
}

# Issue 3: Model repeats itself endlessly
# -> Overtrained or high temperature
generation_fixes_repeat = {
    "temperature": 0.6,        # Lower temperature
    "repetition_penalty": 1.15, # Penalize repetition
    "top_p": 0.9,
    "max_new_tokens": 256,     # Limit output length
}

# Issue 4: Catastrophic forgetting (lost general knowledge)
# -> Too many epochs or too aggressive finetuning
training_fixes_forgetting = {
    "num_epochs": 1,         # Reduce epochs
    "learning_rate": 1e-5,   # Lower learning rate
    "lora_r": 8,             # Lower rank (less capacity to overwrite)
    "lora_dropout": 0.1,     # More regularization
}

# Issue 5: Model refuses everything (over-aligned)
# -> Too much safety data in training mix
# Fix: reduce safety/refusal examples, balance with helpful examples

for name, fixes in [("NaN fix", training_fixes_nan),
                     ("Repeat fix", generation_fixes_repeat),
                     ("Forgetting fix", training_fixes_forgetting)]:
    print(f"\\n{name}:")
    for k, v in fixes.items():
        print(f"  {k}: {v}")`}
        id="code-fixes"
      />

      <NoteBlock
        type="tip"
        title="The 80/20 Rule of Debugging"
        content="80% of finetuning issues are caused by: (1) wrong chat template, (2) bad data formatting, (3) too many epochs, or (4) learning rate too high/low. Check these four things first before investigating more exotic causes."
        id="note-80-20"
      />

      <WarningBlock
        title="Silent Failures"
        content="The most dangerous failures are silent: the model trains successfully, loss looks good, but outputs are subtly wrong (confident hallucinations, biased responses, inconsistent behavior). This is why qualitative evaluation on diverse prompts is essential -- you cannot rely on training loss alone."
        id="warning-silent"
      />
    </div>
  )
}
