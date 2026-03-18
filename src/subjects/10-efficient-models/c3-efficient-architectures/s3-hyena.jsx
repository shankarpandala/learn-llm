import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function HyenaHierarchy() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Hyena Hierarchy: Long Convolutions for Attention-Free Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Hyena replaces self-attention with a hierarchy of long convolutions and element-wise
        gating. It achieves sub-quadratic complexity while maintaining the expressiveness
        needed for language modeling. The key insight is that interleaving data-controlled
        gating with long convolution filters can implicitly learn attention-like patterns.
      </p>

      <DefinitionBlock
        title="Hyena Operator"
        definition="The Hyena operator of order $N$ applies $N$ stages of gated long convolution: $y = h_N * (x_N \odot (h_{N-1} * (x_{N-1} \odot \cdots (h_1 * (x_1 \odot v)) \cdots)))$ where $h_i$ are implicitly parameterized long convolution filters, $x_i$ are data-dependent projections of the input, $*$ is convolution, and $\odot$ is element-wise multiplication."
        notation="Complexity: $O(N \cdot L \log L)$ using FFT-based convolution, compared to $O(L^2 d)$ for attention."
        id="def-hyena"
      />

      <ExampleBlock
        title="Hyena Order-2 Computation"
        problem="Trace through a Hyena-2 block for a sequence of length 4 with d_model=2."
        steps={[
          {
            formula: 'v, x_1, x_2 = \\text{Linear}(\\text{input}) \\quad \\text{(3 projections)}',
            explanation: 'Project input into value v and two gating signals, analogous to Q/K/V.'
          },
          {
            formula: 'z_1 = h_1 * (x_1 \\odot v)',
            explanation: 'First stage: gate v by x_1, then apply long convolution filter h_1.'
          },
          {
            formula: 'z_2 = h_2 * (x_2 \\odot z_1)',
            explanation: 'Second stage: gate first output by x_2, apply another long convolution.'
          },
          {
            formula: 'y = \\text{Linear}(z_2)',
            explanation: 'Output projection. The nested gating creates data-dependent mixing of tokens.'
          }
        ]}
        id="example-hyena"
      />

      <PythonCode
        title="hyena_operator.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ImplicitFilter(nn.Module):
    """Parameterize convolution filter implicitly with a small MLP."""
    def __init__(self, d_model, seq_len, n_hidden=64):
        super().__init__()
        self.seq_len = seq_len
        # Small MLP maps position -> filter value
        self.mlp = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, d_model),
        )
        # Exponential decay for positional bias
        self.decay = nn.Parameter(torch.linspace(-1, -5, d_model))

    def forward(self):
        t = torch.linspace(0, 1, self.seq_len).unsqueeze(-1).to(self.decay.device)
        h = self.mlp(t)  # (seq_len, d_model)
        # Apply exponential decay window
        window = torch.exp(self.decay.unsqueeze(0) * t * self.seq_len)
        return h * window

def fft_conv(u, h):
    """Efficient convolution via FFT: O(L log L) instead of O(L^2)."""
    L = u.shape[-2]
    # Pad to avoid circular convolution artifacts
    fft_len = 2 * L
    U = torch.fft.rfft(u, n=fft_len, dim=-2)
    H = torch.fft.rfft(h, n=fft_len, dim=-2)
    Y = U * H
    y = torch.fft.irfft(Y, n=fft_len, dim=-2)[:, :L, :]
    return y

class HyenaBlock(nn.Module):
    """Simplified Hyena operator (order 2)."""
    def __init__(self, d_model, seq_len, order=2):
        super().__init__()
        self.order = order
        # Input projections: value + order gating signals
        self.in_proj = nn.Linear(d_model, d_model * (order + 1))
        # Implicit convolution filters
        self.filters = nn.ModuleList([
            ImplicitFilter(d_model, seq_len) for _ in range(order)
        ])
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        # Project to value + gating signals
        projections = self.in_proj(x).chunk(self.order + 1, dim=-1)
        v = projections[0]
        gates = projections[1:]

        # Iterative gated convolution
        z = v
        for i in range(self.order):
            z = gates[i] * z  # Element-wise gating
            h = self.filters[i]()  # (seq_len, d_model)
            z = fft_conv(z, h)     # Long convolution via FFT

        return self.out_proj(z)

# Benchmark: compare FLOPs scaling
d_model = 256
for L in [256, 1024, 4096, 16384]:
    # Attention FLOPs: O(L^2 * d)
    attn_flops = 2 * L * L * d_model
    # Hyena FLOPs: O(order * L * log(L) * d) for FFT convolution
    hyena_flops = 2 * 2 * L * math.log2(L) * d_model
    ratio = attn_flops / hyena_flops
    print(f"L={L:>6}: Attention={attn_flops/1e6:.0f}M, "
          f"Hyena={hyena_flops/1e6:.0f}M, speedup={ratio:.1f}x")

# Test forward pass
block = HyenaBlock(d_model=256, seq_len=1024, order=2)
x = torch.randn(2, 1024, 256)
y = block(x)
print(f"\\nInput: {x.shape} -> Output: {y.shape}")`}
        id="code-hyena"
      />

      <NoteBlock
        type="intuition"
        title="Convolution as Implicit Attention"
        content="A long convolution filter h can learn to weight nearby tokens heavily and distant tokens weakly — similar to a fixed attention pattern. By stacking multiple gated convolutions, Hyena builds data-dependent mixing that approximates the flexibility of attention. The gating signals act like learned soft masks that select which information the convolution should propagate."
        id="note-hyena-intuition"
      />

      <NoteBlock
        type="historical"
        title="From S4 to Hyena"
        content="Hyena (Poli et al., 2023) builds on the S4 line of work (Gu et al., 2022). While S4 used structured state spaces for long convolutions, Hyena uses implicitly parameterized filters via small MLPs. StripedHyena scaled this to 7B parameters, showing competitive results with similarly sized Transformers on language tasks."
        id="note-hyena-history"
      />

      <WarningBlock
        title="Fixed Sequence Length at Init"
        content="The implicit filter is parameterized for a fixed maximum sequence length. Handling variable-length sequences requires either padding (wasteful) or filter interpolation. This is less flexible than attention, which naturally handles any sequence length. Newer variants address this with position-independent filter parameterizations."
        id="warning-hyena-seqlen"
      />
    </div>
  )
}
