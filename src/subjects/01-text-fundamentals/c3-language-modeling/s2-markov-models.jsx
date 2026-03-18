import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function MarkovModels() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Markov Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The chain rule decomposes sequence probabilities exactly, but estimating the full
        conditional <InlineMath math="P(w_t \mid w_1, \ldots, w_{t-1})" /> is intractable for long
        sequences. Markov models make this feasible by assuming that only recent context matters.
      </p>

      <DefinitionBlock
        title="Markov Assumption"
        definition="The $k$-th order Markov assumption states that the probability of a token depends only on the preceding $k$ tokens: $P(w_t \mid w_1, \ldots, w_{t-1}) \approx P(w_t \mid w_{t-k}, \ldots, w_{t-1})$. A bigram model uses $k=1$, a trigram model uses $k=2$."
        id="def-markov"
      />

      <h2 className="text-2xl font-semibold">Bigram Model</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The simplest Markov language model conditions only on the immediately preceding word:
      </p>
      <div className="my-4">
        <BlockMath math="P(w_1, \ldots, w_n) \approx \prod_{i=1}^{n} P(w_i \mid w_{i-1})" />
      </div>

      <p className="text-gray-700 dark:text-gray-300">
        Bigram probabilities are estimated from counts using maximum likelihood estimation (MLE):
      </p>
      <div className="my-4">
        <BlockMath math="P_{\text{MLE}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}" />
      </div>

      <ExampleBlock
        title="Bigram MLE"
        problem="Given the corpus: 'the cat sat . the cat ate . the dog sat .', compute $P(\\text{sat} \\mid \\text{cat})$."
        steps={[
          { formula: '$C(\\text{cat}, \\text{sat}) = 1$', explanation: 'The bigram "cat sat" appears once.' },
          { formula: '$C(\\text{cat}) = 2$', explanation: '"cat" appears twice as a context word (before "sat" and "ate").' },
          { formula: '$P(\\text{sat} \\mid \\text{cat}) = 1/2 = 0.5$', explanation: 'Divide bigram count by unigram count of the context word.' },
        ]}
        id="example-bigram-mle"
      />

      <h2 className="text-2xl font-semibold">Smoothing</h2>
      <p className="text-gray-700 dark:text-gray-300">
        MLE assigns zero probability to unseen n-grams, which is catastrophic: a single
        unseen bigram makes the entire sequence probability zero. Smoothing techniques
        redistribute probability mass to unseen events.
      </p>

      <TheoremBlock
        title="Laplace (Add-1) Smoothing"
        statement="Add-1 smoothing adds 1 to every n-gram count: $P_{\text{Laplace}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + V}$ where $V$ is the vocabulary size. This ensures no probability is ever zero."
        id="theorem-laplace"
      />

      <PythonCode
        title="markov_language_model.py"
        code={`from collections import Counter, defaultdict
import random

# Training corpus
corpus = """the cat sat on the mat . the cat ate the fish .
the dog sat on the rug . the dog chased the cat .
the bird flew over the house . the cat watched the bird ."""

# Tokenize
sentences = [s.strip().split() for s in corpus.split('.') if s.strip()]
# Add start/end tokens
sentences = [['<s>'] + s + ['</s>'] for s in sentences]

# Count unigrams and bigrams
unigram_counts = Counter()
bigram_counts = Counter()
for sent in sentences:
    for i in range(len(sent)):
        unigram_counts[sent[i]] += 1
        if i > 0:
            bigram_counts[(sent[i-1], sent[i])] += 1

vocab = set(unigram_counts.keys())
V = len(vocab)

def bigram_prob(w2, w1, alpha=1.0):
    """P(w2 | w1) with Laplace smoothing."""
    return (bigram_counts[(w1, w2)] + alpha) / (unigram_counts[w1] + alpha * V)

# Show some bigram probabilities
print("Bigram probabilities:")
contexts = ["<s>", "the", "cat", "dog"]
for ctx in contexts:
    top = sorted(vocab, key=lambda w: bigram_prob(w, ctx), reverse=True)[:3]
    probs = [f"{w}:{bigram_prob(w, ctx):.3f}" for w in top]
    print(f"  P(? | {ctx}): {', '.join(probs)}")

# Generate text
def generate_bigram(max_len=15):
    tokens = ['<s>']
    for _ in range(max_len):
        prev = tokens[-1]
        candidates = list(vocab)
        probs = [bigram_prob(w, prev) for w in candidates]
        next_word = random.choices(candidates, weights=probs, k=1)[0]
        if next_word == '</s>':
            break
        tokens.append(next_word)
    return ' '.join(tokens[1:])

print("\\nGenerated sentences (bigram model):")
for i in range(5):
    print(f"  {i+1}. {generate_bigram()}")`}
        id="code-markov"
      />

      <h2 className="text-2xl font-semibold">Trigram and Higher-Order Models</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Trigram models condition on two preceding words, capturing more context:
      </p>
      <div className="my-4">
        <BlockMath math="P(w_i \mid w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i)}{C(w_{i-2}, w_{i-1})}" />
      </div>

      <WarningBlock
        title="The Sparsity Problem"
        content="Higher-order n-grams are exponentially sparser. With a vocabulary of 50,000 words, there are 50,000^3 = 1.25 x 10^14 possible trigrams. Even massive corpora will only observe a tiny fraction. This is the fundamental limitation that neural language models overcome by learning continuous representations."
        id="warning-sparsity"
      />

      <NoteBlock
        type="tip"
        title="Interpolation and Backoff"
        content="Practical n-gram models use interpolation (a weighted mix of unigram, bigram, and trigram probabilities) or backoff (use the trigram if seen, otherwise fall back to bigram, then unigram). Kneser-Ney smoothing, which uses a sophisticated backoff distribution, is considered the best classical smoothing method."
        id="note-interpolation"
      />

      <NoteBlock
        type="intuition"
        title="Markov Models vs. Transformers"
        content="An n-gram model with context k can only 'see' k tokens back. A Transformer with context window C can attend to all C previous tokens. GPT-4 with a 128k context window is like a 128,000-gram model, except it generalizes through learned parameters instead of memorizing counts."
        id="note-markov-vs-transformer"
      />
    </div>
  )
}
