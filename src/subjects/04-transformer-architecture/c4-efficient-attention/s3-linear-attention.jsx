import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LinearAttention() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Linear Attention via Kernel Methods</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Linear attention replaces the softmax with a decomposable kernel function, enabling
        the attention computation to be rewritten with matrix multiplications in a different
        order — reducing complexity from <InlineMath math="O(n^2 d)" /> to{' '}
        <InlineMath math="O(n d^2)" />, which is linear in sequence length.
      </p>

      <DefinitionBlock
        title="Linear Attention"
        definition="Replace $\text{softmax}(QK^T)V$ with $\phi(Q)(\phi(K)^T V)$ where $\phi$ is a feature map. By computing $\phi(K)^T V \in \mathbb{R}^{d \times d}$ first, we avoid materializing the $n \times n$ attention matrix."
        notation="φ = kernel feature map, typically φ(x) = elu(x) + 1 or φ(x) = exp(x) (approximate softmax)"
        id="def-linear-attention"
      />

      <h2 className="text-2xl font-semibold">The Kernel Trick for Attention</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Standard attention: compute <InlineMath math="(QK^T)V" /> left-to-right, creating
        the <InlineMath math="n \times n" /> matrix first. Linear attention: compute{' '}
        <InlineMath math="Q(K^T V)" /> right-to-left, creating a <InlineMath math="d \times d" />{' '}
        matrix instead. When <InlineMath math="d \ll n" />, this is dramatically cheaper.
      </p>
      <BlockMath math="\text{Standard: } \underbrace{(QK^T)}_{n \times n} V \quad \text{vs.} \quad \text{Linear: } \phi(Q) \underbrace{(\phi(K)^T V)}_{d \times d_v}" />

      <ExampleBlock
        title="Complexity Comparison"
        problem="Compare FLOPs for standard vs. linear attention with n=8192, d=128."
        steps={[
          { formula: '\\text{Standard: } O(n^2 d) = O(8192^2 \\times 128) \\approx 8.6 \\times 10^9', explanation: 'Quadratic in n: must compute all pairwise interactions.' },
          { formula: '\\text{Linear: } O(n d^2) = O(8192 \\times 128^2) \\approx 1.3 \\times 10^8', explanation: 'Linear in n: compute KV first, then multiply by Q.' },
          { formula: '\\text{Speedup: } \\frac{n}{d} = \\frac{8192}{128} = 64\\times', explanation: 'Linear attention is faster when n >> d (long sequences).' },
        ]}
        id="example-complexity"
      />

      <PythonCode
        title="linear_attention.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

def standard_attention(Q, K, V):
    """O(n^2 d) standard attention."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)

def linear_attention(Q, K, V, feature_map='elu'):
    """O(n d^2) linear attention using kernel feature maps."""
    if feature_map == 'elu':
        phi_Q = F.elu(Q) + 1    # Ensure non-negative
        phi_K = F.elu(K) + 1
    elif feature_map == 'relu':
        phi_Q = F.relu(Q)
        phi_K = F.relu(K)
    else:
        raise ValueError(f"Unknown feature map: {feature_map}")

    # Key insight: compute K^T V first (d x d), then multiply by Q
    # Standard: (n x n) @ (n x d) = O(n^2 d)
    # Linear:   (n x d) @ ((d x n) @ (n x d)) = (n x d) @ (d x d) = O(n d^2)
    KV = torch.matmul(phi_K.transpose(-2, -1), V)    # (d_k, d_v)
    Z = phi_K.transpose(-2, -1).sum(dim=-1, keepdim=True)  # normalizer
    out = torch.matmul(phi_Q, KV)                      # (n, d_v)
    normalizer = torch.matmul(phi_Q, Z) + 1e-6         # (n, 1)
    return out / normalizer

def causal_linear_attention(Q, K, V, feature_map='elu'):
    """Causal linear attention using cumulative sums (RNN-like)."""
    if feature_map == 'elu':
        phi_Q = F.elu(Q) + 1
        phi_K = F.elu(K) + 1
    else:
        phi_Q = F.relu(Q) + 1e-6
        phi_K = F.relu(K) + 1e-6

    B, n, d = phi_Q.shape
    d_v = V.size(-1)

    # Running state: accumulate K^T V incrementally
    S = torch.zeros(B, d, d_v, device=Q.device)  # Running KV state
    Z = torch.zeros(B, d, 1, device=Q.device)    # Running normalizer
    outputs = []

    for t in range(n):
        k_t = phi_K[:, t:t+1, :]     # (B, 1, d)
        v_t = V[:, t:t+1, :]          # (B, 1, d_v)
        q_t = phi_Q[:, t:t+1, :]      # (B, 1, d)

        S = S + torch.matmul(k_t.transpose(-2, -1), v_t)
        Z = Z + k_t.transpose(-2, -1)

        out_t = torch.matmul(q_t, S) / (torch.matmul(q_t, Z) + 1e-6)
        outputs.append(out_t)

    return torch.cat(outputs, dim=1)

# Compare outputs
B, n, d = 2, 64, 32
Q = torch.randn(B, n, d)
K = torch.randn(B, n, d)
V = torch.randn(B, n, d)

std_out = standard_attention(Q, K, V)
lin_out = linear_attention(Q, K, V)
print(f"Standard output: {std_out.shape}")
print(f"Linear output:   {lin_out.shape}")
print(f"Max difference:  {(std_out - lin_out).abs().max():.4f}")
print("(Outputs differ because kernel is approximate, not exact softmax)")`}
        id="code-linear"
      />

      <NoteBlock
        type="intuition"
        title="Linear Attention as an RNN"
        content="Causal linear attention can be computed as a recurrence: maintain a running state S = Σ φ(k_t) v_t^T and normalizer Z = Σ φ(k_t). Each new output is q_t^T S / q_t^T Z. This makes it equivalent to a linear RNN, enabling O(1) per-token generation without a KV cache."
        id="note-rnn-connection"
      />

      <WarningBlock
        title="Quality Degradation"
        content="Linear attention approximates softmax attention but does not replicate it exactly. The softmax's ability to create sharp, peaked distributions is lost with most kernel feature maps. Models trained with linear attention typically underperform standard attention on language modeling, which is why it has not replaced softmax in state-of-the-art LLMs."
        id="warning-quality"
      />

      <NoteBlock
        type="note"
        title="Modern Linear-Time Models"
        content="The idea of linear-time sequence modeling has evolved beyond kernel attention into state-space models (Mamba, S4) and linear RNNs (RWKV, RetNet). These achieve competitive quality with O(n) complexity by using structured recurrences rather than attention approximations."
        id="note-modern-linear"
      />
    </div>
  )
}
