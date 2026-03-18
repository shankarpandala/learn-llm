import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Temperature() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Temperature & Repetition Penalties</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Temperature scaling and repetition penalties are two critical knobs for controlling LLM output
        quality. Temperature adjusts the sharpness of the probability distribution, while repetition
        penalties discourage the model from repeating itself.
      </p>

      <DefinitionBlock
        title="Temperature Scaling"
        definition="Temperature divides logits before the softmax: $P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$. As $T \to 0$, the distribution becomes a one-hot (greedy). As $T \to \infty$, it becomes uniform (random)."
        notation="$T$ = temperature. $T=1$ is the default (unmodified distribution). $T<1$ sharpens. $T>1$ flattens."
        id="def-temperature"
      />

      <ExampleBlock
        title="Temperature Effects on a Distribution"
        problem="Given logits [2.0, 1.0, 0.5, 0.1], compute probabilities at T=0.5, T=1.0, and T=2.0."
        steps={[
          { formula: 'T=0.5: softmax([4.0, 2.0, 1.0, 0.2]) \\approx [0.84, 0.11, 0.04, 0.01]', explanation: 'Low temperature concentrates mass on the top token.' },
          { formula: 'T=1.0: softmax([2.0, 1.0, 0.5, 0.1]) \\approx [0.47, 0.17, 0.10, 0.07]', explanation: 'Default temperature preserves the original distribution shape.' },
          { formula: 'T=2.0: softmax([1.0, 0.5, 0.25, 0.05]) \\approx [0.33, 0.20, 0.16, 0.13]', explanation: 'High temperature makes unlikely tokens more probable.' },
        ]}
        id="example-temperature"
      />

      <PythonCode
        title="temperature_demo.py"
        code={`import torch
import torch.nn.functional as F
import numpy as np

logits = torch.tensor([[2.0, 1.0, 0.5, 0.1, -0.5]])
tokens = ["the", "a", "an", "one", "some"]

print("Temperature effects on probability distribution:")
print(f"{'Token':<8} {'T=0.3':<10} {'T=0.7':<10} {'T=1.0':<10} {'T=1.5':<10} {'T=2.0':<10}")
print("-" * 58)

for temp in [0.3, 0.7, 1.0, 1.5, 2.0]:
    probs = F.softmax(logits / temp, dim=-1)[0]
    row = f""
    for i, tok in enumerate(tokens):
        row = f"{tok:<8}" if i == 0 else row
    vals = "  ".join(f"{p:.4f}" for p in probs.tolist())
    print(f"         {vals}  (T={temp})")

# Entropy as a measure of randomness
for temp in [0.3, 0.7, 1.0, 1.5, 2.0]:
    probs = F.softmax(logits / temp, dim=-1)[0]
    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    print(f"T={temp}: entropy = {entropy:.3f}")`}
        id="code-temperature"
      />

      <DefinitionBlock
        title="Repetition Penalty"
        definition="Repetition penalty reduces the logit of previously generated tokens: if token $w$ appeared before, $z_w' = z_w / \\theta$ when $z_w > 0$, or $z_w' = z_w \\cdot \\theta$ when $z_w < 0$, where $\\theta > 1$ is the penalty factor."
        notation="$\\theta$ = repetition penalty. $\\theta=1.0$ means no penalty. Typical range: $1.1$ to $1.3$."
        id="def-repetition-penalty"
      />

      <PythonCode
        title="repetition_penalty.py"
        code={`import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "The cat sat on the mat. The cat"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Without repetition penalty -- likely to repeat
output_no_penalty = model.generate(
    input_ids, max_new_tokens=40,
    do_sample=True, temperature=0.7, top_p=0.9,
    repetition_penalty=1.0,
)

# With repetition penalty
output_with_penalty = model.generate(
    input_ids, max_new_tokens=40,
    do_sample=True, temperature=0.7, top_p=0.9,
    repetition_penalty=1.2,
)

# With no_repeat_ngram_size (prevents exact n-gram repeats)
output_ngram = model.generate(
    input_ids, max_new_tokens=40,
    do_sample=True, temperature=0.7, top_p=0.9,
    no_repeat_ngram_size=3,
)

print("No penalty:", tokenizer.decode(output_no_penalty[0]))
print("Rep penalty 1.2:", tokenizer.decode(output_with_penalty[0]))
print("No 3-gram repeat:", tokenizer.decode(output_ngram[0]))`}
        id="code-repetition"
      />

      <NoteBlock
        type="tip"
        title="Practical Temperature Guidelines"
        content="For factual Q&A and code: T=0.0-0.3. For general conversation: T=0.5-0.8. For creative writing and brainstorming: T=0.8-1.2. Never go above T=2.0 in production -- output becomes incoherent."
        id="note-temperature-guidelines"
      />

      <NoteBlock
        type="note"
        title="Frequency vs Presence Penalty"
        content="OpenAI's API offers two distinct penalties: frequency_penalty scales with how many times a token appeared, while presence_penalty applies a flat penalty if a token appeared at all. They serve different purposes -- frequency prevents overuse while presence encourages topic diversity."
        id="note-freq-presence"
      />

      <WarningBlock
        title="Temperature Zero Is Not Deterministic Everywhere"
        content="While temperature=0 should be deterministic, floating-point rounding in GPU operations can cause slight non-determinism. For truly reproducible outputs, also set random seeds and use deterministic CUDA operations."
        id="warning-determinism"
      />
    </div>
  )
}
