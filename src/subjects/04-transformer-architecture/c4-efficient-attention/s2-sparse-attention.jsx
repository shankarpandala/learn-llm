import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function SparseAttention() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Sparse Attention: Local and Strided Patterns</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Sparse attention restricts which token pairs can attend to each other, replacing the
        full <InlineMath math="n \times n" /> attention matrix with a sparse pattern that
        computes only a subset of entries. This reduces complexity from{' '}
        <InlineMath math="O(n^2)" /> to <InlineMath math="O(n\sqrt{n})" /> or{' '}
        <InlineMath math="O(n \log n)" />.
      </p>

      <DefinitionBlock
        title="Sparse Attention"
        definition="Instead of computing attention over all $n^2$ pairs, each query attends to a fixed subset $S(i) \subset \{1, \ldots, n\}$ of keys. The attention output becomes $\text{out}_i = \sum_{j \in S(i)} \alpha_{ij} v_j$ where $|S(i)| \ll n$."
        notation="S(i) = sparse connectivity set for query position i"
        id="def-sparse-attention"
      />

      <h2 className="text-2xl font-semibold">Common Sparsity Patterns</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The two most common patterns are local (sliding window) attention and strided (dilated)
        attention. Combining them gives each token both fine-grained local context and
        long-range global reach.
      </p>

      <ExampleBlock
        title="Local + Strided Pattern (Sparse Transformer)"
        problem="For a sequence of 16 tokens with window=4 and stride=4, show the connectivity."
        steps={[
          { formula: '\\text{Local: token 8 attends to } \\{5, 6, 7, 8\\}', explanation: 'Sliding window of size 4 — captures nearby context.' },
          { formula: '\\text{Strided: token 8 attends to } \\{0, 4, 8, 12\\}', explanation: 'Every 4th token — captures distant context with stride.' },
          { formula: '\\text{Combined: } \\{0, 4, 5, 6, 7, 8, 12\\}', explanation: 'Union of local and strided — 7 keys instead of 16.' },
          { formula: '\\text{Complexity: } O(n\\sqrt{n}) \\text{ with } w = \\sqrt{n}', explanation: 'Each token attends to O(√n) others, total O(n√n).' },
        ]}
        id="example-sparse-patterns"
      />

      <PythonCode
        title="sparse_attention_patterns.py"
        code={`import torch
import torch.nn.functional as F
import math

def create_local_mask(seq_len, window_size):
    """Sliding window attention mask."""
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = True
    return mask

def create_strided_mask(seq_len, stride):
    """Strided (dilated) attention mask."""
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        # Attend to positions that are multiples of stride, up to position i
        for j in range(0, i + 1, stride):
            mask[i, j] = True
        mask[i, i] = True  # Always attend to self
    return mask

def create_combined_mask(seq_len, window_size, stride):
    """Combine local and strided patterns."""
    local = create_local_mask(seq_len, window_size)
    strided = create_strided_mask(seq_len, stride)
    return local | strided  # Union

def sparse_attention(Q, K, V, mask):
    """Attention with arbitrary sparse mask."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))
    weights = F.softmax(scores, dim=-1)
    weights = weights.masked_fill(weights.isnan(), 0.0)
    return torch.matmul(weights, V)

seq_len = 16
local_mask = create_local_mask(seq_len, window_size=4)
strided_mask = create_strided_mask(seq_len, stride=4)
combined_mask = create_combined_mask(seq_len, window_size=4, stride=4)

# Count non-zero entries
full_entries = seq_len * seq_len
local_entries = local_mask.sum().item()
strided_entries = strided_mask.sum().item()
combined_entries = combined_mask.sum().item()

print(f"Full attention entries:     {full_entries}")
print(f"Local (w=4) entries:        {local_entries} ({local_entries/full_entries:.1%})")
print(f"Strided (s=4) entries:      {strided_entries} ({strided_entries/full_entries:.1%})")
print(f"Combined entries:           {combined_entries} ({combined_entries/full_entries:.1%})")

# Run sparse attention
Q = K = V = torch.randn(1, seq_len, 64)
out = sparse_attention(Q, K, V, combined_mask)
print(f"\\nSparse attention output: {out.shape}")`}
        id="code-sparse"
      />

      <NoteBlock
        type="note"
        title="Longformer's Sliding Window + Global Tokens"
        content="Longformer (Beltagy et al., 2020) uses sliding window attention for most tokens but designates certain positions (e.g., [CLS]) as global tokens that attend to and are attended by all positions. This hybrid achieves O(n) complexity while maintaining the ability to aggregate global information."
        id="note-longformer"
      />

      <WarningBlock
        title="Sparse Patterns Require Custom Kernels"
        content="Naive implementation of sparse attention using masks on a dense matrix still computes O(n²) entries — it just zeros some out. True efficiency requires custom CUDA kernels that only compute the non-zero entries. Libraries like xformers and Triton provide optimized sparse attention implementations."
        id="warning-custom-kernels"
      />

      <NoteBlock
        type="historical"
        title="Evolution of Sparse Attention"
        content="Sparse Transformer (Child et al., 2019) introduced factored sparse patterns. BigBird (2020) added random attention connections for theoretical completeness. Longformer (2020) popularized sliding window + global. Mistral (2023) showed that sliding window attention works well even for large-scale LLMs."
        id="note-history"
      />
    </div>
  )
}
