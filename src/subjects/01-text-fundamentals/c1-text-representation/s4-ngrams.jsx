import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Ngrams() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">N-grams</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        N-grams are contiguous sequences of <InlineMath math="n" /> items from a text. They capture
        local word order and co-occurrence patterns, forming the basis of classical language models
        and remaining useful as features in modern NLP systems.
      </p>

      <DefinitionBlock
        title="N-gram"
        definition="An n-gram is a contiguous sequence of $n$ tokens from a given text. A unigram ($n=1$) is a single token, a bigram ($n=2$) is a pair of consecutive tokens, and a trigram ($n=3$) is a triple."
        notation="For a sentence $w_1, w_2, \ldots, w_m$, the set of n-grams is $\{(w_i, w_{i+1}, \ldots, w_{i+n-1}) : 1 \leq i \leq m - n + 1\}$."
        id="def-ngram"
      />

      <ExampleBlock
        title="Extracting N-grams"
        problem="Extract all bigrams and trigrams from the sentence: 'the cat sat on the mat'."
        steps={[
          { formula: 'Tokens: [the, cat, sat, on, the, mat]', explanation: 'First tokenize the sentence into words.' },
          { formula: 'Bigrams: [(the, cat), (cat, sat), (sat, on), (on, the), (the, mat)]', explanation: 'Slide a window of size 2 across the tokens.' },
          { formula: 'Trigrams: [(the, cat, sat), (cat, sat, on), (sat, on, the), (on, the, mat)]', explanation: 'Slide a window of size 3 across the tokens.' },
        ]}
        id="example-ngrams"
      />

      <PythonCode
        title="ngram_extraction.py"
        code={`from collections import Counter
from nltk.util import ngrams
import nltk

text = "the cat sat on the mat the cat ate the food"
tokens = text.split()

# Extract n-grams using NLTK
unigrams = list(ngrams(tokens, 1))
bigrams = list(ngrams(tokens, 2))
trigrams = list(ngrams(tokens, 3))

print("Bigrams:", bigrams[:5])
print("Trigrams:", trigrams[:3])

# Count n-gram frequencies
bigram_freq = Counter(bigrams)
print("\\nMost common bigrams:")
for gram, count in bigram_freq.most_common(5):
    print(f"  {gram}: {count}")

# Build a simple bigram probability table
# P(w2 | w1) = count(w1, w2) / count(w1)
unigram_freq = Counter(tokens)
print("\\nBigram probabilities P(w2 | w1):")
for (w1, w2), count in bigram_freq.most_common(5):
    prob = count / unigram_freq[w1]
    print(f"  P({w2} | {w1}) = {count}/{unigram_freq[w1]} = {prob:.3f}")`}
        id="code-ngrams"
      />

      <h2 className="text-2xl font-semibold">N-gram Language Models</h2>
      <p className="text-gray-700 dark:text-gray-300">
        An n-gram language model estimates the probability of a word given
        the <InlineMath math="n-1" /> preceding words. The probability of an entire sentence is
        decomposed using the chain rule with the Markov assumption:
      </p>

      <div className="my-4">
        <BlockMath math="P(w_1, \ldots, w_m) \approx \prod_{i=1}^{m} P(w_i \mid w_{i-n+1}, \ldots, w_{i-1})" />
      </div>

      <PythonCode
        title="ngram_language_model.py"
        code={`from collections import Counter, defaultdict
import random

# Training corpus
corpus = [
    "the cat sat on the mat",
    "the cat ate the fish",
    "the dog sat on the rug",
    "the dog chased the cat",
]

# Build bigram model with Laplace smoothing
bigram_counts = Counter()
unigram_counts = Counter()
vocab = set()

for sentence in corpus:
    tokens = ['<s>'] + sentence.split() + ['</s>']
    vocab.update(tokens)
    for i in range(len(tokens) - 1):
        bigram_counts[(tokens[i], tokens[i+1])] += 1
        unigram_counts[tokens[i]] += 1

V = len(vocab)

def bigram_prob(w2, w1, alpha=1.0):
    """Compute P(w2|w1) with Laplace smoothing."""
    return (bigram_counts[(w1, w2)] + alpha) / (unigram_counts[w1] + alpha * V)

# Generate text using the bigram model
def generate(max_len=10):
    tokens = ['<s>']
    for _ in range(max_len):
        prev = tokens[-1]
        candidates = list(vocab)
        probs = [bigram_prob(w, prev) for w in candidates]
        total = sum(probs)
        probs = [p / total for p in probs]
        next_word = random.choices(candidates, weights=probs, k=1)[0]
        if next_word == '</s>':
            break
        tokens.append(next_word)
    return ' '.join(tokens[1:])

print("Generated sentences:")
for _ in range(5):
    print(f"  {generate()}")`}
        id="code-ngram-lm"
      />

      <WarningBlock
        title="The Curse of Dimensionality"
        content="As n grows, the number of possible n-grams explodes exponentially. For a vocabulary of size V, there are V^n possible n-grams. Most will never appear in the training data, making probability estimation unreliable. This is why smoothing techniques (Laplace, Kneser-Ney) are essential."
        id="warning-sparsity"
      />

      <NoteBlock
        type="note"
        title="Character N-grams"
        content="N-grams need not be words. Character n-grams are sequences of n characters and are useful for language identification, spelling correction, and handling morphologically rich languages. The fastText model uses character n-grams to build word representations."
        id="note-char-ngrams"
      />

      <NoteBlock
        type="tip"
        title="N-grams in Modern NLP"
        content="While neural models have largely replaced n-gram language models, n-gram features remain useful as baseline features in text classification, as part of BM25 scoring in search, and in fast language identification tools like Google's CLD3."
        id="note-modern-ngrams"
      />
    </div>
  )
}
