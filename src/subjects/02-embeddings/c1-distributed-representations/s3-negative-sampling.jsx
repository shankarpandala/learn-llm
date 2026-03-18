import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function NegativeSampling() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Training Objectives & Negative Sampling</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Computing the full softmax over a vocabulary of size <InlineMath math="V" /> at every
        training step is computationally prohibitive. Negative sampling (NEG) and Noise Contrastive
        Estimation (NCE) replace this expensive normalization with a much cheaper binary classification
        task, reducing the cost from <InlineMath math="O(V)" /> to <InlineMath math="O(k)" /> per
        training example, where <InlineMath math="k" /> is the number of negative samples.
      </p>

      <DefinitionBlock
        title="Noise Contrastive Estimation (NCE)"
        definition="NCE reformulates density estimation as a binary classification problem: distinguish real data samples from noise samples drawn from a known distribution $P_n(w)$. For each positive (center, context) pair, we draw $k$ negative samples from $P_n$ and train a logistic classifier to separate them."
        id="def-nce"
      />

      <p className="text-gray-700 dark:text-gray-300">
        The Skip-gram with negative sampling (SGNS) objective for a center word{' '}
        <InlineMath math="w" /> and true context word <InlineMath math="c" /> is:
      </p>
      <BlockMath math={String.raw`\mathcal{L}_{\text{NEG}} = \log \sigma(\mathbf{v}_c'{}^\top \mathbf{v}_w) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)}\!\left[\log \sigma(-\mathbf{v}_{w_i}'{}^\top \mathbf{v}_w)\right]`} />
      <p className="text-gray-700 dark:text-gray-300">
        where <InlineMath math="\sigma(x) = 1/(1 + e^{-x})" /> is the sigmoid function and{' '}
        <InlineMath math="P_n(w) \propto f(w)^{3/4}" /> is the noise distribution (unigram
        raised to the 3/4 power).
      </p>

      <DefinitionBlock
        title="Negative Sampling Distribution"
        definition="The noise distribution is the unigram distribution raised to the 3/4 power: $P_n(w) = \\frac{f(w)^{3/4}}{\\sum_{w'} f(w')^{3/4}}$, where $f(w)$ is the frequency of word $w$ in the corpus. The 3/4 exponent smooths the distribution, giving rare words a higher sampling probability than their raw frequency would suggest."
        id="def-noise-dist"
      />

      <ExampleBlock
        title="Effect of the 3/4 Power"
        problem="Compare sampling probabilities for a frequent word (f=0.01) and a rare word (f=0.0001) under raw unigram vs. smoothed distribution."
        steps={[
          { formula: '$\\text{Raw ratio} = 0.01 / 0.0001 = 100$', explanation: 'The frequent word is 100x more likely to be sampled.' },
          { formula: '$0.01^{0.75} = 0.01778$, $0.0001^{0.75} = 0.000562$', explanation: 'Apply the 3/4 power to both frequencies.' },
          { formula: '$\\text{Smoothed ratio} = 0.01778 / 0.000562 \\approx 31.6$', explanation: 'The ratio drops from 100 to ~32, giving rare words more negative samples and better gradient signal.' },
        ]}
        id="ex-smoothing"
      />

      <TheoremBlock
        title="SGNS Implicitly Factorizes the PMI Matrix"
        statement="Levy & Goldberg (2014) showed that Skip-gram with negative sampling, when fully converged, implicitly factorizes a shifted pointwise mutual information (PMI) matrix: $\mathbf{v}_w \cdot \mathbf{v}_c' = \text{PMI}(w, c) - \log k$, where $k$ is the number of negative samples."
        corollaries={[
          'This connects neural embedding methods to classical count-based distributional semantics.',
          'The number of negative samples k acts as an implicit regularizer on the PMI values.',
        ]}
        id="thm-pmi"
      />

      <PythonCode
        title="negative_sampling_from_scratch.py"
        id="code-neg-sampling"
        code={`import numpy as np

np.random.seed(42)

# Simulate a tiny vocabulary with word frequencies
vocab = ["the", "king", "queen", "man", "woman", "throne"]
freqs = np.array([100, 20, 18, 25, 22, 5], dtype=float)

# Compute noise distribution: f(w)^(3/4) / Z
smoothed = freqs ** 0.75
noise_dist = smoothed / smoothed.sum()

print("Noise distribution (3/4 smoothing):")
for w, p in zip(vocab, noise_dist):
    print(f"  {w:8s}: {p:.4f}")

# Draw negative samples for a training pair
k = 5  # number of negative samples
center_idx = 1   # "king"
context_idx = 2  # "queen" (positive pair)

neg_indices = np.random.choice(len(vocab), size=k, p=noise_dist)
print(f"\\nPositive pair: ({vocab[center_idx]}, {vocab[context_idx]})")
print(f"Negative samples: {[vocab[i] for i in neg_indices]}")

# Simple gradient step (illustrative)
d = 10  # embedding dim
W_center = np.random.randn(len(vocab), d) * 0.1   # center embeddings
W_context = np.random.randn(len(vocab), d) * 0.1  # context embeddings
lr = 0.01

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

# Forward: positive pair
v_c = W_center[center_idx]
v_ctx = W_context[context_idx]
pos_score = sigmoid(np.dot(v_c, v_ctx))
loss = -np.log(pos_score + 1e-8)

# Forward: negative pairs
for ni in neg_indices:
    v_neg = W_context[ni]
    neg_score = sigmoid(-np.dot(v_c, v_neg))
    loss -= np.log(neg_score + 1e-8)

print(f"\\nLoss (1 step): {loss:.4f}")`}
      />

      <NoteBlock
        type="tip"
        title="Choosing k (Number of Negatives)"
        content="Mikolov et al. recommend k=5-20 for small datasets and k=2-5 for large datasets. More negatives provide a better approximation of the full softmax gradient but increase computation. In practice, k=5 is a robust default."
        id="note-choosing-k"
      />

      <WarningBlock
        title="Self-Negatives"
        content="When sampling negatives, you may accidentally sample the true context word as a negative. For large vocabularies this is rare and can be safely ignored. For very small vocabularies, you should explicitly filter out the positive word from the negative sample set."
        id="warn-self-neg"
      />
    </div>
  )
}
