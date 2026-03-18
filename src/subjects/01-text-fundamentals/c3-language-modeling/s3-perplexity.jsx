import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function Perplexity() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Perplexity and Evaluation</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        How do we measure whether one language model is better than another? Perplexity is the
        standard intrinsic evaluation metric. It quantifies how "surprised" a model is by
        test data -- lower perplexity means the model predicts the data better.
      </p>

      <DefinitionBlock
        title="Cross-Entropy"
        definition="The cross-entropy of a language model $P_\theta$ on a test sequence $w_1, \ldots, w_N$ is: $H(P_\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log_2 P_\theta(w_i \mid w_1, \ldots, w_{i-1})$. It measures the average number of bits needed to encode each token under the model."
        notation="$H(P_\theta)$ is measured in bits when using $\log_2$, or in nats when using $\ln$."
        id="def-cross-entropy"
      />

      <DefinitionBlock
        title="Perplexity"
        definition="Perplexity is the exponentiated cross-entropy: $\text{PPL}(P_\theta) = 2^{H(P_\theta)}$. Equivalently, it is the inverse geometric mean probability assigned to each token."
        notation="$\text{PPL} = 2^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 P_\theta(w_i \mid w_{<i})}$"
        id="def-perplexity"
      />

      <TheoremBlock
        title="Perplexity as Weighted Branching Factor"
        statement="Perplexity can be interpreted as the effective vocabulary size the model is choosing from at each step. A perplexity of 100 means the model is, on average, as uncertain as if it were choosing uniformly among 100 equally likely tokens."
        id="theorem-ppl-interp"
      />

      <ExampleBlock
        title="Computing Perplexity"
        problem="A bigram model assigns these probabilities to a test sentence 'the cat sat': $P(\\text{the}) = 0.1$, $P(\\text{cat}|\\text{the}) = 0.05$, $P(\\text{sat}|\\text{cat}) = 0.2$. What is the perplexity?"
        steps={[
          { formula: '$\\log_2 P = \\log_2(0.1) + \\log_2(0.05) + \\log_2(0.2)$', explanation: 'Sum the log probabilities of each token.' },
          { formula: '$= -3.322 + (-4.322) + (-2.322) = -9.966$', explanation: 'Compute each log base 2 value.' },
          { formula: '$H = -\\frac{1}{3}(-9.966) = 3.322$ bits', explanation: 'Divide by the number of tokens N=3.' },
          { formula: '$\\text{PPL} = 2^{3.322} = 10.0$', explanation: 'Exponentiate to get perplexity. The model is as uncertain as choosing from 10 options.' },
        ]}
        id="example-ppl"
      />

      <PythonCode
        title="perplexity_computation.py"
        code={`import numpy as np
from collections import Counter

def compute_perplexity(log_probs):
    """
    Compute perplexity from a list of log2 probabilities.
    PPL = 2^(-1/N * sum(log2(P)))
    """
    N = len(log_probs)
    cross_entropy = -np.sum(log_probs) / N
    perplexity = 2 ** cross_entropy
    return perplexity, cross_entropy

# Example: compare two models on the same test data
# Model A: assigns higher probabilities (better model)
model_a_probs = [0.1, 0.05, 0.2, 0.15, 0.3]
# Model B: assigns lower probabilities (worse model)
model_b_probs = [0.02, 0.01, 0.05, 0.03, 0.08]

log_probs_a = np.log2(model_a_probs)
log_probs_b = np.log2(model_b_probs)

ppl_a, ce_a = compute_perplexity(log_probs_a)
ppl_b, ce_b = compute_perplexity(log_probs_b)

print(f"Model A: perplexity = {ppl_a:.1f}, cross-entropy = {ce_a:.3f} bits")
print(f"Model B: perplexity = {ppl_b:.1f}, cross-entropy = {ce_b:.3f} bits")
print(f"Model A is {ppl_b/ppl_a:.1f}x better (lower PPL = better)")

# Real-world perplexity benchmarks (approximate)
benchmarks = {
    "Trigram model (1990s)":       220,
    "LSTM (2016)":                 82,
    "GPT-2 (2019)":               35,
    "GPT-3 (2020)":               20,
    "GPT-4 class (2023)":         8,
    "Uniform random (50k vocab)":  50000,
}

print("\\nHistorical perplexity on Penn Treebank (approx):")
for model, ppl in sorted(benchmarks.items(), key=lambda x: -x[1]):
    bar = "#" * int(np.log2(ppl))
    print(f"  {model:<30s} PPL={ppl:<8} {bar}")`}
        id="code-perplexity"
      />

      <WarningBlock
        title="Perplexity Comparisons Require Same Tokenization"
        content="You can only compare perplexity between models that use the same vocabulary and tokenization. A character-level model will have lower perplexity per character but higher perplexity per word than a word-level model. Always specify the tokenization when reporting perplexity."
        id="warning-ppl-comparison"
      />

      <NoteBlock
        type="tip"
        title="Bits-Per-Character (BPC)"
        content="To compare models with different tokenizations, normalize by the number of characters instead of tokens. Bits-per-character (BPC) = total cross-entropy / number of characters. This gives a tokenization-independent measure of model quality."
        id="note-bpc"
      />

      <NoteBlock
        type="note"
        title="Beyond Perplexity"
        content="Perplexity measures how well a model predicts text, but it does not directly measure downstream task performance. A model with lower perplexity is not always better at summarization, translation, or reasoning. Modern LLM evaluation increasingly relies on benchmarks like MMLU, HumanEval, and human preference ratings rather than perplexity alone."
        id="note-beyond-ppl"
      />
    </div>
  )
}
