import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ALiBi() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">ALiBi: Attention with Linear Biases</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        ALiBi (Attention with Linear Biases) takes a radically simple approach to position
        encoding: instead of adding position embeddings to token representations, it adds a
        linear distance-based bias directly to the attention scores. This requires zero
        learned parameters and enables strong length extrapolation.
      </p>

      <DefinitionBlock
        title="ALiBi (Press et al., 2022)"
        definition="ALiBi adds a static bias to attention scores: $\text{score}_{ij} = q_i^T k_j - m \cdot |i - j|$ where $m$ is a head-specific slope. No positional embeddings are added to the token representations."
        notation="m = head slope (geometric sequence), |i-j| = distance between positions"
        id="def-alibi"
      />

      <h2 className="text-2xl font-semibold">The Linear Bias</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Each head uses a different slope <InlineMath math="m" /> from a geometric sequence.
        For <InlineMath math="h" /> heads, the slopes are{' '}
        <InlineMath math="2^{-8/h}, 2^{-16/h}, \ldots, 2^{-8}" />.
        Heads with small slopes attend broadly; heads with large slopes attend locally.
      </p>
      <BlockMath math="\text{ALiBi bias}_{ij}^{(k)} = -m_k \cdot |i - j|, \quad m_k = 2^{-8k/h}, \; k = 1, \ldots, h" />

      <ExampleBlock
        title="ALiBi Slopes for 8 Heads"
        problem="Calculate the slopes and their effect on attention range."
        steps={[
          { formula: 'm_1 = 2^{-1} = 0.5', explanation: 'Steepest slope — strongly penalizes distant tokens.' },
          { formula: 'm_4 = 2^{-4} = 0.0625', explanation: 'Medium slope — moderate distance penalty.' },
          { formula: 'm_8 = 2^{-8/8 \\cdot 8} = 2^{-8} \\approx 0.0039', explanation: 'Shallowest slope — nearly uniform attention across positions.' },
          { formula: '\\text{At distance 100: bias}_1 = -50, \\text{ bias}_8 = -0.39', explanation: 'Head 1 ignores distant tokens; head 8 can attend to the whole context.' },
        ]}
        id="example-slopes"
      />

      <PythonCode
        title="alibi_attention.py"
        code={`import torch
import torch.nn.functional as F
import math

def get_alibi_slopes(num_heads):
    """Compute ALiBi slopes for each head."""
    # Geometric sequence: 2^(-8/h), 2^(-16/h), ..., 2^(-8)
    ratio = 2 ** (-8 / num_heads)
    slopes = torch.tensor([ratio ** (i + 1) for i in range(num_heads)])
    return slopes

def build_alibi_bias(seq_len, num_heads):
    """Build the full ALiBi bias matrix."""
    slopes = get_alibi_slopes(num_heads)  # (num_heads,)

    # Distance matrix |i - j| for causal attention (j <= i)
    positions = torch.arange(seq_len)
    # For causal: distance is i - j (always >= 0)
    distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq, seq)

    # For causal masking, set future positions to -inf later
    bias = -slopes.unsqueeze(1).unsqueeze(2) * distance.abs().unsqueeze(0)
    return bias  # (num_heads, seq_len, seq_len)

def alibi_attention(Q, K, V, alibi_bias, causal=True):
    """Attention with ALiBi bias (no positional embeddings on Q, K, V)."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Add ALiBi bias
    scores = scores + alibi_bias[:, :scores.size(-2), :scores.size(-1)]

    if causal:
        mask = torch.triu(torch.ones_like(scores[0]), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights

# Demo
num_heads = 8
seq_len = 32
d_head = 64

slopes = get_alibi_slopes(num_heads)
print("ALiBi slopes:", [f"{s:.6f}" for s in slopes.tolist()])

alibi_bias = build_alibi_bias(seq_len, num_heads)
print(f"\\nALiBi bias shape: {alibi_bias.shape}")  # [8, 32, 32]

# No positional embeddings — just raw token embeddings
Q = torch.randn(num_heads, seq_len, d_head)
K = torch.randn(num_heads, seq_len, d_head)
V = torch.randn(num_heads, seq_len, d_head)

out, weights = alibi_attention(Q, K, V, alibi_bias)
print(f"Output shape: {out.shape}")

# Length extrapolation: bias extends naturally to longer sequences
long_bias = build_alibi_bias(seq_len * 4, num_heads)
print(f"\\nExtrapolated bias shape: {long_bias.shape}")  # [8, 128, 128]
print("No retraining needed — bias is a fixed function of distance")`}
        id="code-alibi"
      />

      <NoteBlock
        type="intuition"
        title="Why ALiBi Extrapolates"
        content="ALiBi's bias is a simple linear function of distance with no learned parameters. At inference, extending to longer sequences just means computing the bias for larger distances — the penalty function is the same regardless of sequence length. The model learns to use content-based attention with a distance prior, and this prior generalizes naturally."
        id="note-extrapolation"
      />

      <WarningBlock
        title="ALiBi vs. RoPE Tradeoffs"
        content="ALiBi is simpler and extrapolates better out-of-the-box, but RoPE generally achieves higher quality within the training context length. Most state-of-the-art LLMs (LLaMA, Mistral, Qwen) chose RoPE with frequency scaling rather than ALiBi. ALiBi is used in BLOOM and MPT but has fallen out of favor for the largest models."
        id="warning-vs-rope"
      />

      <NoteBlock
        type="note"
        title="No Positional Embeddings Needed"
        content="ALiBi's key simplification is that it removes positional embeddings entirely. Token representations are pure content vectors. Position information exists only in the attention bias. This reduces model parameters (no position embedding table) and simplifies the architecture."
        id="note-no-embeddings"
      />

      <NoteBlock
        type="historical"
        title="ALiBi's Influence"
        content="Press et al. (2022) showed that ALiBi trained on 1024 tokens could extrapolate to 2048+ at inference with minimal degradation. This was groundbreaking at the time. While RoPE ultimately won the adoption race, ALiBi's insight — that position need not be in the embeddings — influenced later work on position extrapolation."
        id="note-history"
      />
    </div>
  )
}
