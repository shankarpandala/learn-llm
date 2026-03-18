import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function RotaryPositionEmbeddings() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">RoPE: Rotary Position Embeddings</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Rotary Position Embeddings (RoPE) encode position by rotating query and key vectors in
        2D subspaces. This elegant method naturally encodes relative positions in the attention
        dot product and has become the dominant positional encoding in modern LLMs including
        LLaMA, Mistral, Qwen, and GPT-NeoX.
      </p>

      <DefinitionBlock
        title="Rotary Position Embedding (RoPE)"
        definition="For position $m$, RoPE applies a rotation to each consecutive pair of dimensions: $f(x_m, m) = R_{\Theta,m} x_m$ where $R_{\Theta,m}$ is a block-diagonal matrix of 2D rotations. The key property: $\langle f(q_m, m), f(k_n, n) \rangle = g(q_m, k_n, m-n)$, making the dot product depend only on relative position $m - n$."
        notation="Θ = {θ_i = 10000^{-2i/d}} are the rotation frequencies (same as sinusoidal PE)"
        id="def-rope"
      />

      <h2 className="text-2xl font-semibold">The Rotation Mechanism</h2>
      <p className="text-gray-700 dark:text-gray-300">
        For each pair of dimensions <InlineMath math="(2i, 2i+1)" />, RoPE rotates the query
        and key vectors by angle <InlineMath math="m\theta_i" /> at position{' '}
        <InlineMath math="m" />. When computing <InlineMath math="q_m^T k_n" />, the absolute
        rotations cancel and only the relative rotation{' '}
        <InlineMath math="(m-n)\theta_i" /> remains.
      </p>
      <BlockMath math="R_{\Theta,m} = \begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 & & \\ \sin m\theta_1 & \cos m\theta_1 & & \\ & & \ddots & \\ & & & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\ & & & \sin m\theta_{d/2} & \cos m\theta_{d/2} \end{pmatrix}" />

      <ExampleBlock
        title="RoPE Relative Position Property"
        problem="Show that q_m^T k_n depends only on m - n for a single dimension pair."
        steps={[
          { formula: 'q_m^{\\text{rot}} = (q_1 \\cos m\\theta - q_2 \\sin m\\theta,\\; q_1 \\sin m\\theta + q_2 \\cos m\\theta)', explanation: 'Rotate the query 2D vector by angle mθ.' },
          { formula: 'k_n^{\\text{rot}} = (k_1 \\cos n\\theta - k_2 \\sin n\\theta,\\; k_1 \\sin n\\theta + k_2 \\cos n\\theta)', explanation: 'Rotate the key 2D vector by angle nθ.' },
          { formula: 'q_m^{\\text{rot}} \\cdot k_n^{\\text{rot}} = (q_1 k_1 + q_2 k_2)\\cos(m-n)\\theta + (q_1 k_2 - q_2 k_1)\\sin(m-n)\\theta', explanation: 'The dot product depends on (m-n)θ — only relative position matters.' },
        ]}
        id="example-rope-relative"
      />

      <PythonCode
        title="rotary_position_embeddings.py"
        code={`import torch
import torch.nn as nn

def precompute_rope_freqs(dim, max_len, theta=10000.0):
    """Precompute RoPE rotation frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_len).float()
    # Outer product: (max_len, dim/2)
    angles = torch.outer(positions, freqs)
    # Stack cos and sin
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    """Apply rotary embeddings to input tensor.
    x: (batch, seq_len, num_heads, d_head) or (batch, num_heads, seq_len, d_head)
    """
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]

    # Reshape cos/sin for broadcasting
    cos = cos[:x1.shape[-2]].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, d/2)
    sin = sin[:x1.shape[-2]].unsqueeze(0).unsqueeze(0)

    # Apply rotation
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)

# Setup
d_head = 64
max_len = 2048
cos_cached, sin_cached = precompute_rope_freqs(d_head, max_len)

# Verify relative position property
B, n, h = 1, 32, 8
Q = torch.randn(B, h, n, d_head)
K = torch.randn(B, h, n, d_head)

Q_rot = apply_rope(Q, cos_cached, sin_cached)
K_rot = apply_rope(K, cos_cached, sin_cached)

# Dot product Q[pos_m] · K[pos_n] should depend on m-n
scores = torch.matmul(Q_rot, K_rot.transpose(-2, -1))
print(f"Attention scores shape: {scores.shape}")  # [1, 8, 32, 32]

# RoPE preserves vector norms (rotation doesn't change length)
norm_before = Q[0, 0, 0].norm().item()
norm_after = Q_rot[0, 0, 0].norm().item()
print(f"Q norm before RoPE: {norm_before:.4f}")
print(f"Q norm after RoPE:  {norm_after:.4f}")
print(f"Norm preserved: {abs(norm_before - norm_after) < 1e-5}")

# RoPE is only applied to Q and K, NOT to V
print("\\nRoPE applied to: Q (queries), K (keys)")
print("RoPE NOT applied to: V (values) — values carry content, not position")`}
        id="code-rope"
      />

      <NoteBlock
        type="intuition"
        title="Why RoPE Works So Well"
        content="RoPE combines the best of absolute and relative position encoding. It injects absolute position into Q and K (each position gets a unique rotation), but the attention score naturally depends only on relative position (rotations compose). Unlike additive position encodings, RoPE modifies the dot product geometry directly, making position information inseparable from content."
        id="note-why-rope-works"
      />

      <WarningBlock
        title="RoPE Base Frequency and Context Length"
        content="The base frequency θ=10000 determines the longest wavelength and thus the effective context window. For longer contexts, models increase θ (e.g., Code Llama uses θ=1M for 100K context). NTK-aware scaling and YaRN further modify the frequency schedule to extrapolate beyond training lengths without fine-tuning."
        id="warning-base-freq"
      />

      <NoteBlock
        type="note"
        title="RoPE Adoption"
        content="RoPE (Su et al., 2021) was first adopted by GPT-NeoX and PaLM. LLaMA's success cemented RoPE as the standard. Nearly all modern open-weight LLMs use RoPE: LLaMA 2/3, Mistral, Qwen, Gemma, DeepSeek, Yi, and Phi. The main exception is models using ALiBi (like BLOOM and MPT)."
        id="note-adoption"
      />
    </div>
  )
}
