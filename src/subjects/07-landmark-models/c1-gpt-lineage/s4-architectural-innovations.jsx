import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ArchitecturalInnovations() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Architectural Innovations Across GPT Versions</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Each GPT generation incorporated improvements to the base transformer decoder architecture.
        These innovations in normalization, positional encoding, attention patterns, and numerical
        precision became standard across the field.
      </p>

      <DefinitionBlock
        title="Pre-Layer Normalization"
        definition="A reordering of the transformer block where layer normalization is applied before the attention and FFN sublayers rather than after (post-norm). The output becomes $x + \text{Attn}(\text{LN}(x))$ instead of $\text{LN}(x + \text{Attn}(x))$. This improves training stability and enables deeper networks without careful learning rate warmup."
        id="def-pre-ln"
      />

      <h2 className="text-2xl font-semibold">Normalization Strategies</h2>
      <p className="text-gray-700 dark:text-gray-300">
        GPT-1 used post-layer normalization (original transformer style). GPT-2 switched to
        pre-layer normalization which became the standard for all subsequent large models. Modern
        models like LLaMA further adopted RMSNorm, which drops the mean-centering of LayerNorm
        for faster computation with comparable performance.
      </p>

      <ExampleBlock
        title="Normalization Comparison"
        problem="Compare LayerNorm vs RMSNorm computation for a hidden state vector h."
        steps={[
          { formula: '\\text{LayerNorm}(h) = \\gamma \\cdot \\frac{h - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta', explanation: 'Standard LayerNorm: centers by mean, scales by variance, applies learned affine transform.' },
          { formula: '\\text{RMSNorm}(h) = \\gamma \\cdot \\frac{h}{\\sqrt{\\frac{1}{d}\\sum_{i=1}^{d} h_i^2 + \\epsilon}}', explanation: 'RMSNorm: only normalizes by root mean square. No mean centering, no bias term. ~10-15% faster.' },
        ]}
        id="example-norm-comparison"
      />

      <h2 className="text-2xl font-semibold">Positional Encoding Evolution</h2>
      <p className="text-gray-700 dark:text-gray-300">
        GPT-1/2/3 used learned absolute positional embeddings, which limited generalization beyond
        the training sequence length. Modern models adopted Rotary Position Embeddings (RoPE),
        which encode relative positions through rotation matrices applied to query and key vectors.
      </p>

      <DefinitionBlock
        title="Rotary Position Embeddings (RoPE)"
        definition="A positional encoding that applies a rotation matrix $R_\theta^{(m)}$ to query and key vectors at position $m$. The attention score between positions $m$ and $n$ depends only on the relative distance: $q_m^T k_n = (R_\theta^{(m)} q)^T (R_\theta^{(n)} k) = q^T R_\theta^{(n-m)} k$. This naturally encodes relative position without explicit relative attention computation."
        id="def-rope"
      />

      <h2 className="text-2xl font-semibold">Activation Functions</h2>
      <p className="text-gray-700 dark:text-gray-300">
        GPT-1/2 used GELU activation in the feed-forward layers. Modern architectures switched
        to SwiGLU (Shazeer, 2020), a gated variant that combines Swish activation with a gating
        mechanism, improving performance at similar parameter counts.
      </p>

      <PythonCode
        title="architectural_components.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in LLaMA, Mistral)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class SwiGLU(nn.Module):
    """SwiGLU activation for feed-forward network (Shazeer 2020)."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # gate projection

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# Compare parameter counts
dim, hidden = 4096, 11008  # LLaMA-7B dimensions
swiglu = SwiGLU(dim, hidden)
standard_ffn = nn.Sequential(nn.Linear(dim, hidden * 2 // 3 * 4), nn.GELU(), nn.Linear(hidden * 2 // 3 * 4, dim))

print(f"SwiGLU params: {sum(p.numel() for p in swiglu.parameters()):,}")
print(f"Standard FFN params: {sum(p.numel() for p in standard_ffn.parameters()):,}")

# Test RMSNorm vs LayerNorm speed
import time
x = torch.randn(32, 2048, 4096)
rmsnorm = RMSNorm(4096)
layernorm = nn.LayerNorm(4096)

for name, norm in [("LayerNorm", layernorm), ("RMSNorm", rmsnorm)]:
    start = time.perf_counter()
    for _ in range(100):
        _ = norm(x)
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.3f}s for 100 iterations")`}
        id="code-arch-components"
      />

      <NoteBlock
        type="tip"
        title="Architecture Cheat Sheet"
        content="GPT-1/2/3: Post/Pre-LN, learned positional, GELU, dense FFN. GPT-4 (rumored): Pre-LN, possibly RoPE, SwiGLU, MoE. LLaMA/Mistral: Pre-RMSNorm, RoPE, SwiGLU, GQA. The trend is clear: RMSNorm + RoPE + SwiGLU + GQA has become the default modern recipe."
        id="note-arch-cheatsheet"
      />

      <WarningBlock
        title="Hidden Dimension Scaling with SwiGLU"
        content="SwiGLU uses three weight matrices instead of two, so for a fair parameter comparison the hidden dimension is typically set to 2/3 of what a standard FFN would use. If you see a model config with hidden_dim = 11008 for dim = 4096, this is the 2/3 adjustment (4 * 4096 * 2/3 rounded up)."
        id="warning-swiglu-dims"
      />
    </div>
  )
}
