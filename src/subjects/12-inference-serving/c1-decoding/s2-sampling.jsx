import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Sampling() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Top-k, Top-p & Nucleus Sampling</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Sampling introduces randomness into text generation, producing more diverse and creative
        outputs. Rather than always picking the most likely token, we sample from the probability
        distribution, optionally restricting which tokens are eligible.
      </p>

      <DefinitionBlock
        title="Pure Random Sampling"
        definition="Sample the next token directly from the full vocabulary distribution: $y_t \sim P(w \mid y_{<t}, x)$. This can produce incoherent text because low-probability tokens have a non-zero chance of being selected."
        id="def-random-sampling"
      />

      <DefinitionBlock
        title="Top-k Sampling"
        definition="Restrict sampling to the $k$ most probable tokens, redistributing probability mass among them. Formally, let $V^{(k)}$ be the top-$k$ tokens, then $P'(w) = P(w) / \sum_{w' \in V^{(k)}} P(w')$ for $w \in V^{(k)}$, and $0$ otherwise."
        notation="$k$ = number of tokens to keep. GPT-2 used $k = 40$ by default."
        id="def-top-k"
      />

      <DefinitionBlock
        title="Top-p (Nucleus) Sampling"
        definition="Select the smallest set of tokens whose cumulative probability exceeds $p$: $V^{(p)} = \min\{V' \subseteq V : \sum_{w \in V'} P(w) \geq p\}$. This adapts the number of candidates dynamically based on the distribution shape."
        notation="$p \in (0, 1]$. Typical values: $p = 0.9$ or $p = 0.95$."
        id="def-top-p"
      />

      <ExampleBlock
        title="Top-k vs Top-p Behavior"
        problem="Compare top-k=3 and top-p=0.9 on a peaked vs flat distribution."
        steps={[
          { formula: 'Peaked: P = [0.7, 0.15, 0.08, 0.04, 0.03]', explanation: 'Top-k=3 keeps [0.7, 0.15, 0.08]. Top-p=0.9 keeps [0.7, 0.15, 0.08] (sum=0.93). Similar result.' },
          { formula: 'Flat: P = [0.22, 0.20, 0.19, 0.18, 0.11, 0.10]', explanation: 'Top-k=3 keeps only [0.22, 0.20, 0.19]. Top-p=0.9 keeps [0.22, 0.20, 0.19, 0.18, 0.11] (sum=0.90). More tokens.' },
          { formula: 'Top-p adapts to uncertainty', explanation: 'When the model is uncertain (flat distribution), top-p allows more diversity. When confident, it restricts choices.' },
        ]}
        id="example-topk-topp"
      />

      <PythonCode
        title="sampling_strategies.py"
        code={`import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def sample_next_token(logits, top_k=0, top_p=1.0, temperature=1.0):
    """Apply top-k and/or top-p filtering then sample."""
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k_vals, _ = torch.topk(logits, top_k)
        min_top_k = top_k_vals[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_top_k,
                             torch.full_like(logits, float('-inf')),
                             logits)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative prob above threshold
        remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[remove_mask] = float('-inf')
        # Scatter back to original ordering
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# Generate with different strategies
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

for name, kwargs in [
    ("Top-k=50", {"top_k": 50}),
    ("Top-p=0.9", {"top_p": 0.9}),
    ("Top-k=50 + Top-p=0.95", {"top_k": 50, "top_p": 0.95}),
]:
    gen = input_ids.clone()
    for _ in range(30):
        with torch.no_grad():
            logits = model(gen).logits[:, -1, :]
        next_id = sample_next_token(logits, **kwargs)
        gen = torch.cat([gen, next_id], dim=-1)
    print(f"{name}: {tokenizer.decode(gen[0])}")`}
        id="code-sampling"
      />

      <PythonCode
        title="hf_sampling.py"
        code={`from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

input_ids = tokenizer.encode("The best way to learn is", return_tensors="pt")

# HuggingFace generate with sampling parameters
output = model.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=True,
    top_k=50,
    top_p=0.92,
    temperature=0.8,
    no_repeat_ngram_size=3,  # prevent 3-gram repetition
)

print(tokenizer.decode(output[0], skip_special_tokens=True))`}
        id="code-hf-sampling"
      />

      <NoteBlock
        type="tip"
        title="Combining Top-k and Top-p"
        content="In practice, top-k and top-p are often used together. Top-k provides a hard ceiling on candidates, while top-p further refines within that set. Most APIs (OpenAI, Anthropic) expose top_p as the primary sampling parameter."
        id="note-combining"
      />

      <WarningBlock
        title="Reproducibility with Sampling"
        content="Sampling is stochastic -- the same prompt produces different outputs each run. Set a random seed (torch.manual_seed) for reproducible results during development. In production, embrace the variability or use temperature=0 for deterministic behavior."
        id="warning-reproducibility"
      />
    </div>
  )
}
