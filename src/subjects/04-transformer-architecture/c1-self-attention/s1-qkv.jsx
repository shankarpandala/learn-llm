import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function QueryKeyValue() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Query, Key, and Value: The Intuition Behind Attention</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Self-attention allows every token in a sequence to attend to every other token. The mechanism
        revolves around three learned projections — Query, Key, and Value — that together determine
        how information flows between positions.
      </p>

      <DefinitionBlock
        title="Query, Key, Value Projections"
        definition="Given an input matrix $X \in \mathbb{R}^{n \times d}$, we compute $Q = XW^Q$, $K = XW^K$, $V = XW^V$ where $W^Q, W^K \in \mathbb{R}^{d \times d_k}$ and $W^V \in \mathbb{R}^{d \times d_v}$ are learned parameter matrices."
        notation="Q = query, K = key, V = value, n = sequence length, d = model dimension, d_k = key dimension"
        id="def-qkv"
      />

      <NoteBlock
        type="intuition"
        title="The Library Analogy"
        content="Think of attention like a library search. The Query is your search question, the Keys are book titles on the shelf, and the Values are the actual book contents. You compare your query to each key, then retrieve a weighted combination of the values based on how well each key matched."
        id="note-library-analogy"
      />

      <h2 className="text-2xl font-semibold">How QKV Projections Work</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Each token's embedding is projected into three different vector spaces. The query vector
        asks "what am I looking for?", the key vector says "what do I contain?", and the value
        vector holds "what information do I provide if selected?"
      </p>
      <BlockMath math="Q = XW^Q, \quad K = XW^K, \quad V = XW^V" />

      <ExampleBlock
        title="QKV Computation for a Short Sequence"
        problem="Given a 3-token sequence with d=4 and d_k=d_v=2, show how Q, K, V are computed."
        steps={[
          { formula: 'X \\in \\mathbb{R}^{3 \\times 4}', explanation: 'Input: 3 tokens, each with a 4-dimensional embedding.' },
          { formula: 'W^Q \\in \\mathbb{R}^{4 \\times 2}', explanation: 'The query weight matrix projects from d=4 to d_k=2.' },
          { formula: 'Q = XW^Q \\in \\mathbb{R}^{3 \\times 2}', explanation: 'Each token now has a 2-dimensional query vector.' },
          { formula: '\\text{Same for } K = XW^K,\\; V = XW^V', explanation: 'Key and value projections follow the identical pattern with their own weight matrices.' },
        ]}
        id="example-qkv-shapes"
      />

      <PythonCode
        title="qkv_projections.py"
        code={`import torch
import torch.nn as nn

# Dimensions
batch_size, seq_len, d_model, d_k = 2, 5, 16, 8

# Input embeddings
X = torch.randn(batch_size, seq_len, d_model)

# Learned projection matrices
W_Q = nn.Linear(d_model, d_k, bias=False)
W_K = nn.Linear(d_model, d_k, bias=False)
W_V = nn.Linear(d_model, d_k, bias=False)

# Compute Q, K, V
Q = W_Q(X)  # (batch, seq_len, d_k)
K = W_K(X)  # (batch, seq_len, d_k)
V = W_V(X)  # (batch, seq_len, d_k)

print(f"Input shape:  {X.shape}")   # [2, 5, 16]
print(f"Query shape:  {Q.shape}")   # [2, 5, 8]
print(f"Key shape:    {K.shape}")   # [2, 5, 8]
print(f"Value shape:  {V.shape}")   # [2, 5, 8]

# Attention scores: how much each query attends to each key
scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_len, seq_len)
print(f"Score shape:  {scores.shape}")  # [2, 5, 5]`}
        id="code-qkv"
      />

      <WarningBlock
        title="QKV Are Not Interchangeable"
        content="Although Q, K, and V are all linear projections of the same input X, they serve fundamentally different roles. Swapping Q and K transposes the attention matrix (reversing who attends to whom). The value matrix V determines what information is actually retrieved — it is never used in the compatibility computation."
        id="warning-qkv-roles"
      />

      <NoteBlock
        type="note"
        title="Parameter Count"
        content="The QKV projections account for 3 * d_model * d_k parameters per attention head (ignoring biases). In GPT-3 with d_model=12288 and 96 heads (d_k=128), that is 3 * 12288 * 12288 ≈ 453M parameters just for QKV across all heads."
        id="note-param-count"
      />
    </div>
  )
}
