import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function WhatIsLM() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">What is a Language Model?</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        A language model assigns a probability to a sequence of words. This simple idea is
        the mathematical foundation of all modern LLMs: GPT, LLaMA, Claude, and every other
        generative text model. Understanding what language models compute is essential to
        understanding how they work.
      </p>

      <DefinitionBlock
        title="Language Model"
        definition="A language model is a probability distribution over sequences of tokens. Given a sequence $w_1, w_2, \ldots, w_n$, a language model assigns a probability $P(w_1, w_2, \ldots, w_n)$ that measures how likely this sequence is under the model's learned distribution."
        notation="$P(w_1, w_2, \ldots, w_n)$ or equivalently $P(\mathbf{w})$ where $\mathbf{w}$ is a token sequence."
        id="def-lm"
      />

      <TheoremBlock
        title="Chain Rule of Probability"
        statement="Any joint probability over a sequence can be decomposed exactly using the chain rule: $P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, w_2, \ldots, w_{i-1})$. This is not an approximation; it is an identity from probability theory."
        id="theorem-chain-rule"
      />

      <p className="text-gray-700 dark:text-gray-300">
        The chain rule tells us that a language model can be equivalently viewed as a
        next-token predictor. At each step, the model predicts the distribution over the next
        token given all preceding tokens:
      </p>

      <div className="my-4">
        <BlockMath math="P(w_t \mid w_1, \ldots, w_{t-1})" />
      </div>

      <p className="text-gray-700 dark:text-gray-300">
        This is precisely what GPT and other autoregressive models learn to do.
      </p>

      <ExampleBlock
        title="Computing Sequence Probability"
        problem="A language model gives these next-token probabilities. What is $P(\text{'the cat sat'})$?"
        steps={[
          { formula: '$P(\\text{the}) = 0.05$', explanation: 'Probability of "the" as the first word.' },
          { formula: '$P(\\text{cat} \\mid \\text{the}) = 0.02$', explanation: 'Probability of "cat" following "the".' },
          { formula: '$P(\\text{sat} \\mid \\text{the cat}) = 0.08$', explanation: 'Probability of "sat" given "the cat".' },
          { formula: '$P(\\text{the cat sat}) = 0.05 \\times 0.02 \\times 0.08 = 0.00008$', explanation: 'Multiply all conditional probabilities by the chain rule.' },
        ]}
        id="example-seq-prob"
      />

      <PythonCode
        title="language_model_basics.py"
        code={`import numpy as np

# A simple demonstration of language modeling
# A language model defines P(next_token | context)

# Vocabulary
vocab = ["the", "cat", "dog", "sat", "ran", "on", "mat", "<end>"]
V = len(vocab)
word_to_idx = {w: i for i, w in enumerate(vocab)}

# A toy conditional probability table: P(next | prev)
# Each row sums to 1.0
cond_probs = {
    "<start>": {"the": 0.6, "cat": 0.1, "dog": 0.1, "sat": 0.05,
                "ran": 0.05, "on": 0.02, "mat": 0.02, "<end>": 0.06},
    "the":     {"cat": 0.3, "dog": 0.3, "mat": 0.2, "the": 0.01,
                "sat": 0.05, "ran": 0.05, "on": 0.05, "<end>": 0.04},
    "cat":     {"sat": 0.4, "ran": 0.3, "on": 0.1, "the": 0.05,
                "cat": 0.01, "dog": 0.01, "mat": 0.03, "<end>": 0.1},
    "sat":     {"on": 0.6, "the": 0.1, "<end>": 0.2, "cat": 0.02,
                "dog": 0.02, "sat": 0.01, "ran": 0.02, "mat": 0.03},
}

def sequence_probability(tokens):
    """Compute P(w1, w2, ..., wn) using the chain rule."""
    prob = 1.0
    prev = "<start>"
    for token in tokens:
        if prev in cond_probs and token in cond_probs[prev]:
            p = cond_probs[prev][token]
        else:
            p = 1e-6  # Smoothing for unseen transitions
        prob *= p
        prev = token
    return prob

# Compare probabilities of different sequences
sequences = [
    ["the", "cat", "sat", "on"],
    ["the", "dog", "ran"],
    ["cat", "the", "mat", "sat"],  # Ungrammatical
    ["sat", "cat", "on", "the"],   # Ungrammatical
]

print("Sequence probabilities:")
for seq in sequences:
    p = sequence_probability(seq)
    print(f"  P({' '.join(seq)}) = {p:.8f}")

# The model assigns higher probability to grammatical sequences!`}
        id="code-lm-basics"
      />

      <NoteBlock
        type="intuition"
        title="Language Models as World Models"
        content="A sufficiently good language model must implicitly capture facts, reasoning, and world knowledge. To correctly predict that 'The capital of France is ___' should be completed with 'Paris', the model must 'know' geography. This is why scaling language models has led to emergent capabilities far beyond simple text prediction."
        id="note-lm-world-model"
      />

      <NoteBlock
        type="note"
        title="Three Uses of Language Models"
        content="Language models serve three main purposes: (1) Scoring - evaluating how likely or fluent a sentence is (used in speech recognition, machine translation); (2) Generation - sampling new text by repeatedly predicting the next token; (3) Representation - using internal states as features for downstream tasks (like BERT embeddings)."
        id="note-lm-uses"
      />

      <NoteBlock
        type="historical"
        title="The Language Modeling Hypothesis"
        content="Shannon (1948) first framed language as a stochastic process and proposed measuring its entropy. Jelinek at IBM (1970s-80s) built n-gram language models for speech recognition, famously stating 'Every time I fire a linguist, the performance of the speech recognizer goes up.' The modern LLM era began with GPT (2018), showing that large-scale language modeling alone can produce powerful general-purpose AI systems."
        id="note-lm-history"
      />
    </div>
  )
}
