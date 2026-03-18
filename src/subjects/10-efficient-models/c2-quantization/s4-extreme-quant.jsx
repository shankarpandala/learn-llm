import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function ExtremeQuantization() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Extreme Quantization: 1-Bit and Ternary Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Extreme quantization pushes model compression to the limit: ternary weights
        use only {'{-1, 0, +1}'} and binary weights use {'{-1, +1}'}. These enable
        replacing multiplications with additions and sign flips, offering dramatic
        speedups on specialized hardware. BitNet and related work show this is viable
        even for billion-parameter language models.
      </p>

      <DefinitionBlock
        title="Ternary Quantization"
        definition="Ternary quantization maps each weight to one of three values: $w_q \in \{-\alpha, 0, +\alpha\}$ where $\alpha$ is a learned or computed scaling factor. The quantization function is $w_q = \alpha \cdot \text{sign}(w) \cdot \mathbb{1}(|w| > \Delta)$ where $\Delta$ is a threshold."
        notation="For a weight matrix $W$: $\hat{W} = \alpha \cdot T$ where $T \in \{-1, 0, +1\}^{m \times n}$ and $\alpha = \frac{\sum_{|w| > \Delta} |w|}{|\{w : |w| > \Delta\}|}$ (mean of non-zero magnitudes)."
        id="def-ternary"
      />

      <DefinitionBlock
        title="BitNet b1.58"
        definition="BitNet b1.58 (Ma et al., 2024) constrains every weight to $\{-1, 0, +1\}$, achieving 1.58 bits per weight ($\log_2 3 \approx 1.58$). Matrix multiplication $Y = WX$ becomes $Y = \alpha \cdot (T \cdot X)$ where the ternary multiply reduces to additions and subtractions only."
        id="def-bitnet"
      />

      <ExampleBlock
        title="Ternary Matrix Multiplication"
        problem="Compute Y = W * x where W is ternary and x = [0.5, -0.3, 0.8, 0.1]."
        steps={[
          {
            formula: 'T = [[1, 0, -1, 1], [-1, 1, 0, -1]], \\quad \\alpha = 0.45',
            explanation: 'Weight matrix decomposed into ternary values and scale.'
          },
          {
            formula: 'T \\cdot x = [0.5 + 0 - 0.8 + 0.1, \\; -0.5 - 0.3 + 0 - 0.1]',
            explanation: 'No multiplications! Just additions/subtractions based on sign.'
          },
          {
            formula: 'T \\cdot x = [-0.2, -0.9]',
            explanation: 'Raw ternary output computed with zero multiplies.'
          },
          {
            formula: 'Y = 0.45 \\times [-0.2, -0.9] = [-0.09, -0.405]',
            explanation: 'Apply scale factor alpha. Only one multiply per output element.'
          }
        ]}
        id="example-ternary-matmul"
      />

      <TheoremBlock
        title="Compute Savings from Ternary Weights"
        statement="For a matrix multiplication $Y = WX$ with $W \in \mathbb{R}^{m \times n}$, standard FP16 requires $2mn$ FLOPs. With ternary weights, this reduces to $mn$ additions plus $m$ scalar multiplications (for the scale factor), a theoretical $2\times$ compute reduction per layer."
        proof="Each element of $T \cdot x$ involves $n$ operations, each of which is either +x_j, -x_j, or 0 (no-op for zeros). With typical 30% sparsity in ternary matrices, the effective operation count is $\sim 0.7mn$ additions + $m$ multiplies, compared to $2mn$ FLOPs for FP16 matmul."
        id="thm-ternary-compute"
      />

      <PythonCode
        title="extreme_quantization.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

def ternarize(W, threshold_ratio=0.7):
    """Quantize weights to {-alpha, 0, +alpha}."""
    abs_W = W.abs()
    threshold = threshold_ratio * abs_W.mean()

    # Create ternary mask
    T = torch.zeros_like(W)
    T[W > threshold] = 1.0
    T[W < -threshold] = -1.0

    # Compute optimal scale factor
    mask = T != 0
    alpha = abs_W[mask].mean() if mask.any() else torch.tensor(1.0)

    return T, alpha

def binary_quantize(W):
    """Quantize weights to {-alpha, +alpha} (1-bit)."""
    alpha = W.abs().mean()
    B = torch.sign(W)
    B[B == 0] = 1.0  # No zeros in binary
    return B, alpha

# Compare quantization levels
torch.manual_seed(42)
W = torch.randn(1024, 1024) * 0.02
x = torch.randn(1024, 128)

# Ground truth
Y_true = W @ x

# Ternary (1.58-bit)
T, alpha_t = ternarize(W)
Y_ternary = alpha_t * (T @ x)
sparsity = (T == 0).float().mean()
print(f"Ternary: alpha={alpha_t:.4f}, sparsity={sparsity:.1%}")

# Binary (1-bit)
B, alpha_b = binary_quantize(W)
Y_binary = alpha_b * (B @ x)

# INT4 baseline
scale_4 = W.abs().max() / 7
W_int4 = torch.clamp(torch.round(W / scale_4), -8, 7) * scale_4
Y_int4 = W_int4 @ x

# Compare errors and sizes
for name, Y_q, bits in [("INT4", Y_int4, 4), ("Ternary", Y_ternary, 1.58),
                          ("Binary", Y_binary, 1)]:
    mse = ((Y_true - Y_q)**2).mean().item()
    size_mb = 1024 * 1024 * bits / 8 / 1e6
    print(f"{name:>8}: MSE={mse:.6f}, size={size_mb:.2f} MB "
          f"(vs {1024*1024*16/8/1e6:.2f} MB FP16)")

# BitNet-style linear layer
class BitLinear(nn.Module):
    """1.58-bit linear layer (BitNet b1.58 style)."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def forward(self, x):
        # Activation quantization: absmax to INT8
        x_scale = x.abs().max()
        x_q = (x / x_scale * 127).round().clamp(-128, 127)

        # Weight ternarization
        T, alpha = ternarize(self.weight)

        # Compute: only additions and subtractions
        y = alpha * (T @ (x_q.float())) * (x_scale / 127)
        return y

layer = BitLinear(4096, 4096)
test_input = torch.randn(1, 128, 4096)
output = layer(test_input)
print(f"\\nBitLinear output shape: {output.shape}")
print(f"Weight memory: {4096*4096*1.58/8/1e6:.1f} MB (vs {4096*4096*2/1e6:.1f} MB FP16)")`}
        id="code-extreme-quant"
      />

      <NoteBlock
        type="historical"
        title="The Path to 1-Bit LLMs"
        content="BinaryConnect (2015) first showed binary weights could work for small networks. TWN (2016) introduced ternary weights. BitNet (2023) scaled binary quantization to Transformer LLMs. BitNet b1.58 (2024) demonstrated that 1.58-bit LLMs can match FP16 performance starting at 3B parameters, with dramatic energy and latency improvements."
        id="note-extreme-history"
      />

      <WarningBlock
        title="Hardware Support Required"
        content="Extreme quantization only delivers speedups with specialized kernels or hardware. Standard GPU CUDA cores cannot efficiently exploit ternary arithmetic. Custom kernels (like those in llama.cpp for 2-4 bit) and upcoming hardware with native low-bit support are necessary to realize the theoretical gains."
        id="warning-hardware-support"
      />
    </div>
  )
}
