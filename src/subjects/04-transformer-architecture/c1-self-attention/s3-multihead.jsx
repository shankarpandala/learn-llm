import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function MultiHeadAttention() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Multi-Head Attention</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Rather than performing a single attention function, multi-head attention runs several
        attention operations in parallel, each with different learned projections. This allows
        the model to jointly attend to information from different representation subspaces at
        different positions.
      </p>

      <DefinitionBlock
        title="Multi-Head Attention"
        definition="$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$ where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ and $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$."
        notation="h = number of heads, d_k = d_model / h (typically), W^O = output projection"
        id="def-multihead"
      />

      <NoteBlock
        type="intuition"
        title="Why Multiple Heads?"
        content="A single attention head can only compute one set of attention weights per position. With 8 heads, one head might attend to syntactic structure (subject-verb agreement), another to semantic similarity, another to positional neighbors, etc. Multiple heads give the model a richer set of information-routing patterns."
        id="note-why-heads"
      />

      <h2 className="text-2xl font-semibold">Head Splitting and Concatenation</h2>
      <p className="text-gray-700 dark:text-gray-300">
        In practice, instead of creating <InlineMath math="h" /> separate weight matrices, we
        project to the full <InlineMath math="d_{\text{model}}" /> dimension and then reshape
        into <InlineMath math="h" /> heads of size <InlineMath math="d_k = d_{\text{model}} / h" />.
        After attention, heads are concatenated and projected back.
      </p>
      <BlockMath math="d_k = d_v = \frac{d_{\text{model}}}{h}" />

      <ExampleBlock
        title="Multi-Head Dimensions in GPT-2 Small"
        problem="GPT-2 Small has d_model=768 and h=12 heads. Trace the tensor shapes."
        steps={[
          { formula: 'X \\in \\mathbb{R}^{B \\times n \\times 768}', explanation: 'Input: batch of sequences with 768-dim embeddings.' },
          { formula: 'QKV \\in \\mathbb{R}^{B \\times n \\times 768} \\text{ each}', explanation: 'Full projection, then reshaped to (B, 12, n, 64).' },
          { formula: '\\text{head}_i \\in \\mathbb{R}^{B \\times n \\times 64}', explanation: 'Each head operates on d_k=768/12=64 dimensional slices.' },
          { formula: '\\text{Concat} \\in \\mathbb{R}^{B \\times n \\times 768}', explanation: 'Concatenating 12 heads of dim 64 gives back 768.' },
          { formula: '\\text{Output} = \\text{Concat} \\cdot W^O \\in \\mathbb{R}^{B \\times n \\times 768}', explanation: 'Final linear projection mixes information across heads.' },
        ]}
        id="example-gpt2-shapes"
      />

      <PythonCode
        title="multi_head_attention.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B, n, _ = Q.shape

        # Project and reshape: (B, n, d_model) -> (B, h, n, d_k)
        Q = self.W_q(Q).view(B, n, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention per head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # (B, h, n, d_k)

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(B, n, self.d_model)
        return self.W_o(context)

# Usage
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)
out = mha(x, x, x)  # Self-attention: Q=K=V=x
print(f"Output: {out.shape}")  # [2, 10, 512]
print(f"Parameters: {sum(p.numel() for p in mha.parameters()):,}")`}
        id="code-mha"
      />

      <WarningBlock
        title="Head Count Must Divide d_model"
        content="If d_model is not evenly divisible by the number of heads, you cannot cleanly split the dimensions. This is why transformer model dimensions are almost always multiples of common head counts (64, 128). GPT-3's d_model=12288 with 96 heads gives d_k=128 exactly."
        id="warning-divisibility"
      />

      <NoteBlock
        type="note"
        title="Computational Cost Is Unchanged"
        content="Multi-head attention with h heads of dimension d_k = d_model/h has the same total computation as single-head attention with full d_model. The cost is O(n^2 * d_model) either way. The benefit is purely representational — multiple independent attention patterns at no extra FLOPs."
        id="note-cost"
      />
    </div>
  )
}
