import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function AttentionMasking() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Attention Masking: Causal and Padding Masks</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Masking controls which positions each token can attend to. Causal masks prevent tokens
        from looking ahead (essential for autoregressive generation), while padding masks
        ensure that padding tokens do not participate in attention computations.
      </p>

      <DefinitionBlock
        title="Causal (Look-Ahead) Mask"
        definition="A causal mask is a lower-triangular matrix $M \in \{0, -\infty\}^{n \times n}$ where $M_{ij} = 0$ if $i \geq j$ and $M_{ij} = -\infty$ otherwise. Applied as $\text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$."
        notation="Position i can attend to positions 0 through i, but not to positions i+1 through n-1."
        id="def-causal-mask"
      />

      <DefinitionBlock
        title="Padding Mask"
        definition="A padding mask is a binary vector $p \in \{0, 1\}^n$ where $p_j = 0$ for padding positions. It is broadcast so that all queries ignore padding keys: $\text{scores}_{ij} = -\infty$ wherever $p_j = 0$."
        id="def-padding-mask"
      />

      <h2 className="text-2xl font-semibold">Causal Masking for Autoregressive Models</h2>
      <p className="text-gray-700 dark:text-gray-300">
        In decoder-only models like GPT, each token must only attend to itself and previous tokens.
        Without the causal mask, the model would "cheat" during training by reading future tokens.
      </p>

      <ExampleBlock
        title="Causal Mask for 4 Tokens"
        problem="Construct and apply a causal mask for the sentence 'The cat sat down'."
        steps={[
          { formula: 'M = \\begin{pmatrix} 0 & -\\infty & -\\infty & -\\infty \\\\ 0 & 0 & -\\infty & -\\infty \\\\ 0 & 0 & 0 & -\\infty \\\\ 0 & 0 & 0 & 0 \\end{pmatrix}', explanation: 'Lower-triangular: token i attends only to tokens 0..i.' },
          { formula: '\\text{scores} + M', explanation: 'Adding -inf makes those positions 0 after softmax.' },
          { formula: '\\text{softmax row 2} = \\text{softmax}([s_{20}, s_{21}, s_{22}, -\\infty])', explanation: '"sat" attends to "The", "cat", and "sat" only.' },
        ]}
        id="example-causal"
      />

      <PythonCode
        title="attention_masks.py"
        code={`import torch
import torch.nn.functional as F
import math

def create_causal_mask(seq_len, device='cpu'):
    """Lower-triangular causal mask."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask  # 1 = attend, 0 = block

def create_padding_mask(lengths, max_len, device='cpu'):
    """Mask out padding positions given actual sequence lengths."""
    arange = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
    mask = arange < lengths.unsqueeze(1)  # (batch, max_len)
    return mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, max_len) for broadcasting

def masked_attention(Q, K, V, causal=True, padding_mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if causal:
        n = Q.size(-2)
        causal_mask = create_causal_mask(n, Q.device)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

    if padding_mask is not None:
        scores = scores.masked_fill(padding_mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights

# Demo: Causal mask
B, n, d = 1, 6, 32
Q = K = V = torch.randn(B, n, d)
out, w = masked_attention(Q, K, V, causal=True)
print("Causal attention weights (should be lower-triangular):")
print(w[0].round(decimals=2))

# Demo: Padding mask for variable-length sequences
lengths = torch.tensor([4, 6])  # First sequence has 4 real tokens, second has 6
pad_mask = create_padding_mask(lengths, max_len=6)
Q2 = K2 = V2 = torch.randn(2, n, d)
out2, w2 = masked_attention(Q2, K2, V2, causal=False, padding_mask=pad_mask)
print(f"\\nPadding mask shape: {pad_mask.shape}")
print(f"Seq 1 attends to positions: {pad_mask[0, 0, 0]}")`}
        id="code-masks"
      />

      <NoteBlock
        type="tip"
        title="Combining Masks"
        content="In practice, causal and padding masks are combined with a logical AND (or equivalently, both are added to the scores as additive -inf terms). Decoder models need both: the causal mask for autoregressive ordering, and the padding mask for batched sequences of different lengths."
        id="note-combining"
      />

      <WarningBlock
        title="Prefix LM vs. Causal LM Masking"
        content="Some architectures (e.g., T5, UL2) use a prefix LM mask where a prefix portion of the sequence has full bidirectional attention while the rest uses causal masking. This is different from a pure causal mask and allows bidirectional encoding of the input while generating the output autoregressively."
        id="warning-prefix-lm"
      />

      <NoteBlock
        type="historical"
        title="Masked Self-Attention in the Original Paper"
        content="Vaswani et al. (2017) introduced masked self-attention specifically for the decoder. The encoder uses full bidirectional attention (no causal mask). Modern decoder-only models (GPT series) use causal masking throughout, eliminating the encoder entirely."
        id="note-history"
      />
    </div>
  )
}
