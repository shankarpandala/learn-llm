import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function GloVe() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">GloVe: Global Vectors for Word Representation</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        GloVe (Pennington et al., 2014) bridges the gap between count-based methods (like LSA) and
        prediction-based methods (like Word2Vec). It directly factorizes the log of the global
        word-word co-occurrence matrix using a weighted least squares objective, producing embeddings
        that capture both local context patterns and global corpus statistics.
      </p>

      <DefinitionBlock
        title="Co-occurrence Matrix"
        definition="The co-occurrence matrix $\mathbf{X}$ has entries $X_{ij}$ counting the number of times word $j$ appears in the context of word $i$ across the entire corpus. Context is defined by a symmetric window of size $c$. Closer context words may be given higher weight via a decaying function $1/d$ where $d$ is the distance."
        notation="$X_{ij}$ = co-occurrence count; $X_i = \sum_k X_{ik}$ = total co-occurrences for word $i$; $P_{ij} = X_{ij}/X_i$ = co-occurrence probability."
        id="def-cooccurrence"
      />

      <p className="text-gray-700 dark:text-gray-300">
        The key insight of GloVe is that ratios of co-occurrence probabilities encode meaning.
        For words <InlineMath math="i" /> (ice) and <InlineMath math="j" /> (steam), the
        ratio <InlineMath math="P_{ik}/P_{jk}" /> is large when <InlineMath math="k" /> (solid)
        relates to ice but not steam, and small when <InlineMath math="k" /> (gas) relates to
        steam but not ice.
      </p>

      <DefinitionBlock
        title="GloVe Objective Function"
        definition="GloVe minimizes the weighted least squares cost: $J = \\sum_{i,j=1}^{V} f(X_{ij})\\left(\\mathbf{w}_i^\\top \\tilde{\\mathbf{w}}_j + b_i + \\tilde{b}_j - \\log X_{ij}\\right)^2$ where $\\mathbf{w}_i$ and $\\tilde{\\mathbf{w}}_j$ are word and context vectors, $b_i$ and $\\tilde{b}_j$ are biases, and $f$ is a weighting function."
        id="def-glove-obj"
      />

      <p className="text-gray-700 dark:text-gray-300">
        The weighting function <InlineMath math="f(x)" /> prevents very frequent co-occurrences
        from dominating the objective:
      </p>
      <BlockMath math={String.raw`f(x) = \begin{cases} (x/x_{\max})^{\alpha} & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}`} />
      <p className="text-gray-700 dark:text-gray-300">
        with <InlineMath math="x_{\max} = 100" /> and <InlineMath math="\alpha = 3/4" /> as
        recommended defaults.
      </p>

      <ExampleBlock
        title="Co-occurrence Probability Ratios"
        problem="Given co-occurrence data: $P(\text{solid}|\text{ice}) = 1.9 \times 10^{-4}$, $P(\text{solid}|\text{steam}) = 2.2 \times 10^{-5}$, $P(\text{gas}|\text{ice}) = 6.6 \times 10^{-5}$, $P(\text{gas}|\text{steam}) = 7.8 \times 10^{-4}$. Compute the ratios."
        steps={[
          { formula: '$P(\\text{solid}|\\text{ice}) / P(\\text{solid}|\\text{steam}) = 8.9$', explanation: '"solid" is strongly associated with ice over steam.' },
          { formula: '$P(\\text{gas}|\\text{ice}) / P(\\text{gas}|\\text{steam}) = 0.085$', explanation: '"gas" is strongly associated with steam over ice.' },
          { formula: '$P(\\text{water}|\\text{ice}) / P(\\text{water}|\\text{steam}) \\approx 1.0$', explanation: 'Neutral words that relate equally to both yield ratios near 1.' },
        ]}
        id="ex-ratios"
      />

      <PythonCode
        title="glove_cooccurrence.py"
        id="code-glove"
        code={`import numpy as np
from collections import Counter, defaultdict

# Build co-occurrence matrix from a small corpus
corpus = [
    "the king sat on the throne".split(),
    "the queen wore the crown".split(),
    "the king and queen ruled the kingdom".split(),
    "a man and a woman walked to the throne".split(),
]

# Build vocabulary
word_counts = Counter(w for sent in corpus for w in sent)
vocab = sorted(word_counts.keys())
word_to_idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# Co-occurrence matrix with window size 2
window = 2
cooccur = np.zeros((V, V))

for sent in corpus:
    for i, word in enumerate(sent):
        wi = word_to_idx[word]
        for j in range(max(0, i - window), min(len(sent), i + window + 1)):
            if i != j:
                wj = word_to_idx[sent[j]]
                distance = abs(i - j)
                cooccur[wi][wj] += 1.0 / distance  # distance weighting

print(f"Vocabulary: {vocab}")
print(f"Co-occurrence matrix shape: {cooccur.shape}")

# Show co-occurrences for 'king'
king_idx = word_to_idx["king"]
print(f"\\nCo-occurrences with 'king':")
for w, idx in sorted(word_to_idx.items()):
    if cooccur[king_idx][idx] > 0:
        print(f"  {w:10s}: {cooccur[king_idx][idx]:.2f}")

# GloVe weighting function
def glove_weight(x, x_max=100, alpha=0.75):
    return np.where(x < x_max, (x / x_max) ** alpha, 1.0)

# Show weights for different co-occurrence counts
counts = [1, 5, 10, 50, 100, 500]
print(f"\\nGloVe weights f(x) for x_max=100:")
for c in counts:
    print(f"  f({c:3d}) = {glove_weight(c):.4f}")`}
      />

      <NoteBlock
        type="note"
        title="GloVe vs Word2Vec in Practice"
        content="Both GloVe and Word2Vec produce high-quality embeddings and often perform comparably on downstream tasks. GloVe's main advantage is interpretability: its objective explicitly connects to co-occurrence statistics. Word2Vec (SGNS) is often easier to train incrementally on streaming data. Pre-trained GloVe vectors (6B, 42B, 840B tokens) are freely available from the Stanford NLP group."
        id="note-comparison"
      />

      <WarningBlock
        title="Memory Requirements"
        content="The full co-occurrence matrix is V x V, which for a 400k vocabulary requires 640 GB in float32. In practice, the matrix is very sparse, and only non-zero entries are stored. Still, building the matrix for large corpora requires careful engineering and can be the main memory bottleneck."
        id="warn-memory"
      />
    </div>
  )
}
