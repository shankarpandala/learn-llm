import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Stopwords() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Stop Words and Vocabulary</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Not all words carry equal information. Function words like "the", "is", and "of" appear
        frequently but contribute little to document meaning. Managing vocabulary size and
        filtering uninformative tokens is a key preprocessing decision that affects both
        model performance and computational cost.
      </p>

      <DefinitionBlock
        title="Stop Words"
        definition="Stop words are high-frequency, low-information words that are often removed during text preprocessing. They include articles (the, a), prepositions (in, on, at), conjunctions (and, but, or), and common verbs (is, are, was). Different stop word lists exist for different languages and tasks."
        id="def-stopwords"
      />

      <PythonCode
        title="stopwords_exploration.py"
        code={`import nltk
from collections import Counter

# nltk.download('stopwords')  # Run once
from nltk.corpus import stopwords

# English stop words
en_stops = set(stopwords.words('english'))
print(f"NLTK English stop words: {len(en_stops)}")
print(f"Examples: {sorted(list(en_stops))[:20]}")

# Demonstrate the impact of stop word removal
text = """The transformer architecture has fundamentally changed the field
of natural language processing. It uses self-attention mechanisms
to process all tokens in a sequence simultaneously, which is much
more efficient than the recurrent approaches that came before it."""

tokens = text.lower().split()
filtered = [t for t in tokens if t not in en_stops]

print(f"\\nOriginal:  {len(tokens)} tokens")
print(f"Filtered:  {len(filtered)} tokens ({len(tokens)-len(filtered)} removed)")
print(f"\\nOriginal first 15:  {tokens[:15]}")
print(f"Filtered first 15:  {filtered[:15]}")

# Zipf's law: word frequency follows a power law
all_tokens = text.lower().split()
freq = Counter(all_tokens)
print("\\nTop 10 words by frequency:")
for word, count in freq.most_common(10):
    is_stop = "STOP" if word in en_stops else ""
    print(f"  {word:<15} {count:>3}  {is_stop}")
# Most frequent words are stop words!`}
        id="code-stopwords"
      />

      <h2 className="text-2xl font-semibold">Zipf's Law and Vocabulary</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Word frequencies in natural language follow Zipf's law: the frequency of a word is
        inversely proportional to its rank in the frequency table.
      </p>

      <div className="my-4">
        <BlockMath math="f(r) \propto \frac{1}{r^s}" />
        <p className="text-center text-sm text-gray-500 dark:text-gray-400">
          where <InlineMath math="r" /> is the rank and <InlineMath math="s \approx 1" /> for English text.
        </p>
      </div>

      <p className="text-gray-700 dark:text-gray-300">
        This means a small number of words account for most of the text (stop words), while
        the vast majority of unique words are rare. This long tail creates a fundamental
        tension in vocabulary design.
      </p>

      <DefinitionBlock
        title="Vocabulary"
        definition="A vocabulary $V$ is the set of unique tokens a model can represent. Tokens outside the vocabulary are unknown (OOV). Vocabulary size is a key hyperparameter: too small and information is lost, too large and the model becomes inefficient."
        id="def-vocabulary"
      />

      <PythonCode
        title="vocabulary_management.py"
        code={`from collections import Counter
import numpy as np

# Simulating vocabulary statistics on a corpus
# (Using word frequencies from a hypothetical corpus)
np.random.seed(42)
# Generate Zipfian word frequencies
vocab_size = 10000
ranks = np.arange(1, vocab_size + 1)
frequencies = (1.0 / ranks) * 100000  # Zipf's law
frequencies = frequencies.astype(int)

total_tokens = frequencies.sum()
cumulative = np.cumsum(frequencies) / total_tokens

# How many words cover X% of all tokens?
for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
    n_words = np.searchsorted(cumulative, threshold) + 1
    print(f"  {threshold*100:.0f}% coverage: {n_words:,} words "
          f"({n_words/vocab_size*100:.1f}% of vocabulary)")

# Vocabulary pruning strategies
print("\\nVocabulary reduction strategies:")
strategies = {
    "Min frequency = 2":  sum(1 for f in frequencies if f >= 2),
    "Min frequency = 5":  sum(1 for f in frequencies if f >= 5),
    "Min frequency = 10": sum(1 for f in frequencies if f >= 10),
    "Top 5,000 words":    5000,
    "Top 1,000 words":    1000,
}
for name, size in strategies.items():
    print(f"  {name:<25} -> vocab size: {size:,}")

# Special tokens in modern LLM vocabularies
special_tokens = {
    "<pad>":   "Padding for batch processing",
    "<unk>":   "Unknown/OOV token",
    "<bos>":   "Beginning of sequence",
    "<eos>":   "End of sequence",
    "<mask>":  "Masked position (BERT-style)",
    "<sep>":   "Separator between segments",
}
print("\\nSpecial tokens in LLM vocabularies:")
for token, desc in special_tokens.items():
    print(f"  {token:<10} {desc}")`}
        id="code-vocabulary"
      />

      <ExampleBlock
        title="When to Remove Stop Words"
        problem="Should you remove stop words for a sentiment analysis task on product reviews?"
        steps={[
          { formula: 'Consider: "This is not good" vs "This is good"', explanation: '"not" is a stop word, but removing it flips the sentiment entirely!' },
          { formula: 'Consider: "I could not be happier"', explanation: '"could not be" are all stop words, but they carry critical sentiment information.' },
          { formula: 'Decision: Do NOT remove stop words for sentiment analysis', explanation: 'Negation words and function words carry grammatical meaning essential for sentiment.' },
        ]}
        id="example-stopword-decision"
      />

      <WarningBlock
        title="Stop Word Removal Is Task-Dependent"
        content="Stop word removal helps for topic modeling, keyword extraction, and search (TF-IDF). But it hurts for tasks that depend on syntax: sentiment analysis, question answering, and machine translation. For LLM training, stop words are never removed because the model needs to generate fluent text with proper grammar."
        id="warning-task-dependent"
      />

      <NoteBlock
        type="tip"
        title="Frequency-Based vs. List-Based Filtering"
        content="Instead of using a fixed stop word list, consider frequency-based filtering: remove words that appear in more than X% of documents (too common) or fewer than Y documents (too rare). scikit-learn's TfidfVectorizer supports this with max_df and min_df parameters, providing data-driven vocabulary selection."
        id="note-frequency-filtering"
      />
    </div>
  )
}
