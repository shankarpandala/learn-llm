import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function ScaledDotProduct() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Scaled Dot-Product Attention</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Scaled dot-product attention is the core computation inside every transformer. It computes
        a weighted sum of values where the weights come from the compatibility between queries
        and keys, scaled by the square root of the key dimension.
      </p>

      <DefinitionBlock
        title="Scaled Dot-Product Attention"
        definition="$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ where $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$, and $d_k$ is the key dimension."
        notation="n = query length, m = key/value length, d_k = key dim, d_v = value dim"
        id="def-scaled-dot-product"
      />

      <h2 className="text-2xl font-semibold">Why Scale by <InlineMath math="\sqrt{d_k}" />?</h2>
      <p className="text-gray-700 dark:text-gray-300">
        When <InlineMath math="d_k" /> is large, the dot products <InlineMath math="QK^T" /> tend
        to grow in magnitude, pushing the softmax into regions with extremely small gradients.
        Dividing by <InlineMath math="\sqrt{d_k}" /> keeps the variance of the dot products at
        approximately 1, ensuring healthy gradient flow.
      </p>

      <TheoremBlock
        title="Variance of Dot Products"
        statement="If the components of Q and K are independent random variables with mean 0 and variance 1, then $\text{Var}(q \cdot k) = d_k$. Scaling by $\frac{1}{\sqrt{d_k}}$ normalizes the variance back to 1."
        proofSteps={[
          'Let q_i, k_i be i.i.d. with mean 0 and variance 1.',
          'q \\cdot k = \\sum_{i=1}^{d_k} q_i k_i',
          '\\text{Var}(q_i k_i) = E[q_i^2 k_i^2] - (E[q_i k_i])^2 = 1 \\cdot 1 - 0 = 1',
          '\\text{Var}(q \\cdot k) = \\sum_{i=1}^{d_k} \\text{Var}(q_i k_i) = d_k',
          '\\text{Var}\\left(\\frac{q \\cdot k}{\\sqrt{d_k}}\\right) = \\frac{d_k}{d_k} = 1',
        ]}
        id="thm-scaling"
      />

      <ExampleBlock
        title="Softmax Saturation Without Scaling"
        problem="Show how large dot products cause softmax to saturate for d_k = 512."
        steps={[
          { formula: '\\text{Var}(q \\cdot k) = 512', explanation: 'Without scaling, dot products have standard deviation ≈ 22.6.' },
          { formula: '\\text{softmax}([22, -18, 20, -15]) \\approx [0.88, 0.0, 0.12, 0.0]', explanation: 'Large magnitude differences make softmax nearly one-hot, killing gradients.' },
          { formula: '\\text{After scaling: } [0.97, -0.80, 0.88, -0.66]', explanation: 'Dividing by √512 ≈ 22.6 keeps values small, producing smoother attention weights.' },
        ]}
        id="example-saturation"
      />

      <PythonCode
        title="scaled_dot_product_attention.py"
        code={`import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, n, d_k)
    K: (batch, m, d_k)
    V: (batch, m, d_v)
    Returns: (batch, n, d_v), attention_weights (batch, n, m)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights

# Demo
batch, n, d_k, d_v = 1, 4, 64, 64
Q = torch.randn(batch, n, d_k)
K = torch.randn(batch, n, d_k)
V = torch.randn(batch, n, d_v)

out, weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {out.shape}")        # [1, 4, 64]
print(f"Attention weights shape: {weights.shape}")  # [1, 4, 4]
print(f"Weights sum to 1: {weights.sum(dim=-1)}")   # All ones

# Show effect of scaling on weight entropy
scores_unscaled = torch.matmul(Q, K.transpose(-2, -1))
scores_scaled = scores_unscaled / math.sqrt(d_k)
w_unscaled = F.softmax(scores_unscaled, dim=-1)
w_scaled = F.softmax(scores_scaled, dim=-1)
entropy_un = -(w_unscaled * w_unscaled.log()).sum(-1).mean()
entropy_sc = -(w_scaled * w_scaled.log()).sum(-1).mean()
print(f"Entropy unscaled: {entropy_un:.3f}, scaled: {entropy_sc:.3f}")`}
        id="code-sdpa"
      />

      <NoteBlock
        type="tip"
        title="PyTorch Built-in"
        content="Since PyTorch 2.0, use torch.nn.functional.scaled_dot_product_attention which automatically selects the most efficient backend (FlashAttention, Memory-Efficient Attention, or the math fallback) depending on hardware and input sizes."
        id="note-pytorch-sdpa"
      />

      <WarningBlock
        title="Numerical Stability"
        content="In practice, softmax is computed as softmax(x - max(x)) to avoid overflow. When using masked attention, positions set to -inf are handled correctly because exp(-inf) = 0. However, if all positions in a row are masked, you get NaN — always ensure at least one position is unmasked per query."
        id="warning-numerical"
      />
    </div>
  )
}
