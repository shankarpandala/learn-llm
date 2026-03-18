import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function OneHotToDense() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">One-Hot to Dense Vectors</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The simplest way to represent words numerically is the one-hot encoding: assign each word
        in a vocabulary of size <InlineMath math="V" /> a unique index and represent it as a binary
        vector with a single 1 at that index and 0s everywhere else. While straightforward, this
        approach has fundamental limitations that motivate the move to dense, distributed representations.
      </p>

      <DefinitionBlock
        title="One-Hot Encoding"
        definition="Given a vocabulary $\mathcal{V}$ of size $V$, the one-hot representation of the $i$-th word is a vector $\mathbf{e}_i \in \{0,1\}^V$ where $e_{ij} = 1$ if $j = i$ and $e_{ij} = 0$ otherwise."
        notation="$\mathbf{e}_i$ denotes the one-hot vector for word $i$; the inner product $\mathbf{e}_i^\top \mathbf{e}_j = 0$ for all $i \neq j$."
        id="def-one-hot"
      />

      <WarningBlock
        title="The Curse of Orthogonality"
        content="Because every pair of one-hot vectors is orthogonal, the cosine similarity between any two distinct words is exactly zero. This means 'cat' is as different from 'dog' as it is from 'quantum' -- the representation captures no semantic relationships whatsoever."
        id="warn-orthogonality"
      />

      <PythonCode
        title="one_hot_demo.py"
        id="code-one-hot"
        code={`import numpy as np

# Build a tiny vocabulary
vocab = ["king", "queen", "man", "woman", "apple"]
word_to_idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# One-hot encode
def one_hot(word):
    vec = np.zeros(V)
    vec[word_to_idx[word]] = 1.0
    return vec

# Demonstrate the problem: all similarities are 0
king = one_hot("king")
queen = one_hot("queen")
apple = one_hot("apple")

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

print(f"sim(king, queen) = {cosine_sim(king, queen):.2f}")   # 0.00
print(f"sim(king, apple) = {cosine_sim(king, apple):.2f}")   # 0.00
print(f"Memory for V=50,000: {50_000 * 50_000 * 4 / 1e9:.1f} GB (float32)")
# For a real vocabulary, this is extremely wasteful`}
      />

      <DefinitionBlock
        title="Distributional Hypothesis"
        definition="Words that occur in similar linguistic contexts tend to have similar meanings (Harris, 1954; Firth, 1957). Formally, if two words $w_i$ and $w_j$ share similar context distributions $P(\text{context} \mid w_i) \approx P(\text{context} \mid w_j)$, then $w_i$ and $w_j$ are semantically related."
        id="def-distributional"
      />

      <NoteBlock
        type="historical"
        title="Firth's Famous Quote"
        content="'You shall know a word by the company it keeps' -- J.R. Firth (1957). This insight underpins virtually all modern embedding methods. Rather than defining meaning through symbolic rules, we learn meaning from statistical patterns of co-occurrence in large corpora."
        id="note-firth"
      />

      <ExampleBlock
        title="From Sparse to Dense: Embedding Lookup"
        problem="Show that multiplying a one-hot vector by an embedding matrix $\mathbf{W} \in \mathbb{R}^{V \times d}$ simply selects the corresponding row, yielding a dense vector in $\mathbb{R}^d$."
        steps={[
          { formula: '$\\mathbf{e}_i^\\top \\mathbf{W} = \\mathbf{W}[i, :]$', explanation: 'The one-hot vector acts as a row selector.' },
          { formula: '$\\mathbf{W}[i, :] \\in \\mathbb{R}^d$ where $d \\ll V$', explanation: 'The result is a dense vector of dimension d (typically 50-300), much smaller than V (often 50,000+).' },
          { formula: '$\\cos(\\mathbf{W}[i,:],\\, \\mathbf{W}[j,:]) \\neq 0$ in general', explanation: 'Dense vectors can express graded similarity between words, unlike one-hot vectors.' },
        ]}
        id="ex-embedding-lookup"
      />

      <PythonCode
        title="embedding_lookup.py"
        id="code-embedding-lookup"
        code={`import numpy as np

V, d = 5, 3  # tiny vocab, 3-dim embeddings
vocab = ["king", "queen", "man", "woman", "apple"]

# Random embedding matrix (in practice, learned from data)
np.random.seed(42)
W = np.random.randn(V, d)

# One-hot lookup is equivalent to indexing
word_idx = 0  # "king"
e_king = np.zeros(V)
e_king[word_idx] = 1.0

# These two are identical:
via_matmul = e_king @ W
via_index  = W[word_idx]

print("Via matmul:", via_matmul)
print("Via index: ", via_index)
print("Equal?", np.allclose(via_matmul, via_index))

# Now similarity is meaningful (once W is learned)
print(f"\\nDense sim(king, queen) = {np.dot(W[0], W[1]) / (np.linalg.norm(W[0]) * np.linalg.norm(W[1])):.3f}")`}
      />

      <NoteBlock
        type="intuition"
        title="Why Dense Representations Work"
        content="Dense embeddings compress word identity into a low-dimensional space where geometric relationships encode semantic ones. Each dimension can be thought of as capturing a latent feature -- formality, animacy, gender -- that is shared across many words. This sharing is what makes embeddings generalize: even words seen rarely can be placed near semantically similar neighbors."
        id="note-why-dense"
      />
    </div>
  )
}
