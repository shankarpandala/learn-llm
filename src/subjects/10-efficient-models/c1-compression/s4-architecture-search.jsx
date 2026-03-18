import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ArchitectureSearch() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Neural Architecture Search for LLMs</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Neural Architecture Search (NAS) automates the design of efficient model architectures
        by searching over configurations of depth, width, attention heads, and feed-forward
        dimensions. For LLMs, NAS finds Pareto-optimal trade-offs between accuracy and
        latency/memory that human designers often miss.
      </p>

      <DefinitionBlock
        title="Neural Architecture Search"
        definition="NAS formulates architecture design as an optimization problem: $a^* = \arg\max_{a \in \mathcal{A}} \text{Acc}(a)$ subject to $\text{Cost}(a) \leq B$, where $\mathcal{A}$ is the search space, and $B$ is a resource budget (FLOPs, latency, or parameters)."
        notation="Search space for Transformer LLMs typically includes: number of layers $L \in [6, 48]$, hidden dimension $d \in [256, 4096]$, heads $h \in [4, 32]$, FFN ratio $r \in [2, 8]$, yielding $|\mathcal{A}| > 10^{12}$ candidate architectures."
        id="def-nas"
      />

      <ExampleBlock
        title="Search Space Design for Efficient LLMs"
        problem="Define a NAS search space for a Transformer targeting < 1B parameters and < 100ms latency on mobile."
        steps={[
          {
            formula: 'L \\in \\{6, 8, 12, 16, 20, 24\\}',
            explanation: 'Search over different depths. Deeper is more capable but slower.'
          },
          {
            formula: 'd \\in \\{384, 512, 640, 768, 1024\\}',
            explanation: 'Hidden dimension affects parameter count quadratically via FFN layers.'
          },
          {
            formula: '\\text{Params} \\approx 12 L d^2 + 2Vd',
            explanation: 'Rough parameter estimate: 12Ld^2 for transformer layers + 2Vd for embeddings.'
          },
          {
            formula: '\\text{Filter: } 12 \\times 24 \\times 1024^2 + 2 \\times 32000 \\times 1024 \\approx 367M',
            explanation: 'Even the largest candidate fits within the 1B budget.'
          }
        ]}
        id="example-nas-search-space"
      />

      <PythonCode
        title="nas_for_transformers.py"
        code={`import itertools
import math

def estimate_params(n_layers, d_model, ffn_ratio=4, vocab_size=32000):
    """Estimate total parameters in a Transformer."""
    # Self-attention: Q, K, V, O projections
    attn_params = 4 * d_model * d_model * n_layers
    # FFN: two linear layers with expansion ratio
    ffn_params = 2 * d_model * (ffn_ratio * d_model) * n_layers
    # Layer norms (2 per layer)
    norm_params = 4 * d_model * n_layers
    # Embeddings (tied input/output)
    embed_params = vocab_size * d_model
    return attn_params + ffn_params + norm_params + embed_params

def estimate_flops(n_layers, d_model, seq_len=2048, ffn_ratio=4):
    """Estimate FLOPs for a single forward pass."""
    # Attention: 2 * seq * d^2 * 4 (QKV + O) + 2 * seq^2 * d (scores)
    attn_flops = n_layers * (8 * seq_len * d_model**2 + 2 * seq_len**2 * d_model)
    # FFN: 2 * seq * d * (r*d) * 2
    ffn_flops = n_layers * 4 * seq_len * d_model * (ffn_ratio * d_model)
    return attn_flops + ffn_flops

# Search space
layers_options = [6, 8, 12, 16, 20, 24]
dim_options = [384, 512, 640, 768, 1024, 1280]
ffn_ratios = [2, 4, 8]

# Brute-force search with constraints
budget_params = 1_000_000_000   # 1B parameters
budget_flops = 5e12             # 5 TFLOPs

candidates = []
for L, d, r in itertools.product(layers_options, dim_options, ffn_ratios):
    params = estimate_params(L, d, r)
    flops = estimate_flops(L, d, ffn_ratio=r)
    if params <= budget_params and flops <= budget_flops:
        # Proxy score: deeper + wider is generally better
        score = L * math.log(d)
        candidates.append((score, L, d, r, params, flops))

# Top-5 architectures by proxy score
candidates.sort(reverse=True)
print("Top-5 architectures under 1B params, 5 TFLOPs:")
print(f"{'Layers':>7} {'Dim':>6} {'FFN_r':>6} {'Params':>12} {'TFLOPs':>8}")
for score, L, d, r, params, flops in candidates[:5]:
    print(f"{L:>7} {d:>6} {r:>6} {params:>12,} {flops/1e12:>8.2f}")

# In practice, each candidate would be trained briefly (or use
# a supernet with weight-sharing) and evaluated on a validation set.`}
        id="code-nas"
      />

      <NoteBlock
        type="note"
        title="Efficient NAS Methods"
        content="Early NAS required training thousands of models from scratch (Zoph & Le, 2017). Modern approaches use weight-sharing supernets (one-shot NAS), predictor-based methods that estimate accuracy without training, or hardware-aware search that directly optimizes for latency on target devices."
        id="note-efficient-nas"
      />

      <NoteBlock
        type="historical"
        title="NAS-Designed LLMs"
        content="AutoTinyBERT (2021) used NAS to find optimal BERT architectures for each downstream task. LiteTransformer (2020) discovered that splitting attention between local and global heads is more efficient. Primer (2022) by Google used NAS to find architectural modifications that improved training efficiency by 4x over vanilla Transformers."
        id="note-nas-history"
      />

      <WarningBlock
        title="Search Cost vs. Benefit"
        content="NAS itself is computationally expensive. A single NAS run can cost thousands of GPU hours. For one-off models, hand-designed scaling laws (Chinchilla, LLaMA configurations) may be more practical. NAS shines when deploying across many hardware targets where one-size-fits-all architectures are suboptimal."
        id="warning-nas-cost"
      />
    </div>
  )
}
