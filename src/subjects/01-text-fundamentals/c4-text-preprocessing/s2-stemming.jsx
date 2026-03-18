import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Stemming() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Stemming and Lemmatization</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Words appear in many inflected forms: "running", "runs", "ran" all relate to the concept
        of "run". Stemming and lemmatization reduce words to a base form, shrinking the
        effective vocabulary and grouping related words together. These techniques are
        fundamental to information retrieval and classical NLP.
      </p>

      <DefinitionBlock
        title="Stemming"
        definition="Stemming is a heuristic process that chops off word suffixes to produce a stem. The stem may not be a valid word. For example, the Porter stemmer reduces 'running' to 'run', 'studies' to 'studi', and 'university' to 'univers'."
        id="def-stemming"
      />

      <DefinitionBlock
        title="Lemmatization"
        definition="Lemmatization uses morphological analysis and vocabulary lookup to reduce a word to its dictionary form (lemma). Unlike stemming, the result is always a valid word. For example, 'better' lemmatizes to 'good', 'mice' to 'mouse', and 'ran' to 'run'."
        id="def-lemmatization"
      />

      <ExampleBlock
        title="Stemming vs Lemmatization"
        problem="Compare stemming and lemmatization for the words: 'studies', 'studying', 'better', 'wolves'"
        steps={[
          { formula: "'studies' -> stem: 'studi', lemma: 'study'", explanation: 'The stemmer applies a crude suffix rule; the lemmatizer finds the dictionary form.' },
          { formula: "'studying' -> stem: 'studi', lemma: 'study'", explanation: 'Both reduce to the base, but the stem is not a real word.' },
          { formula: "'better' -> stem: 'better', lemma: 'good'", explanation: 'Stemmers cannot handle irregular forms; lemmatizers use morphological rules.' },
          { formula: "'wolves' -> stem: 'wolv', lemma: 'wolf'", explanation: 'Lemmatization correctly handles irregular plurals.' },
        ]}
        id="example-stem-vs-lemma"
      />

      <PythonCode
        title="stemming_lemmatization.py"
        code={`import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

# Download required data (run once)
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

porter = PorterStemmer()
snowball = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

words = [
    'running', 'runs', 'ran',
    'studies', 'studying', 'studied',
    'better', 'best', 'good',
    'wolves', 'mice', 'geese',
    'happily', 'happiness', 'unhappy',
    'organization', 'organizing', 'organized',
]

print(f"{'Word':<16} {'Porter':<16} {'Snowball':<16} {'Lemma (v)':<16} {'Lemma (n)':<16}")
print("-" * 80)
for word in words:
    p = porter.stem(word)
    s = snowball.stem(word)
    lv = lemmatizer.lemmatize(word, pos='v')  # As verb
    ln = lemmatizer.lemmatize(word, pos='n')  # As noun
    print(f"{word:<16} {p:<16} {s:<16} {lv:<16} {ln:<16}")

# Why POS matters for lemmatization
print("\\nPOS-aware lemmatization:")
print(f"  'saw' as noun: {lemmatizer.lemmatize('saw', 'n')}")   # saw (tool)
print(f"  'saw' as verb: {lemmatizer.lemmatize('saw', 'v')}")   # see`}
        id="code-stemming"
      />

      <h2 className="text-2xl font-semibold">Stemming Algorithms</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The most common stemmers apply cascading rules to strip suffixes:
      </p>
      <ul className="ml-6 list-disc space-y-2 text-gray-700 dark:text-gray-300">
        <li><strong>Porter Stemmer (1980)</strong> - The classic algorithm with 5 phases of suffix-stripping rules. Fast but aggressive.</li>
        <li><strong>Snowball Stemmer</strong> - Martin Porter's improved version with better rules and multi-language support.</li>
        <li><strong>Lancaster Stemmer</strong> - More aggressive than Porter, producing shorter stems.</li>
      </ul>

      <PythonCode
        title="stemming_in_search.py"
        code={`from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stemmer = PorterStemmer()

def stem_tokenizer(text):
    """Tokenize and stem for use in TF-IDF."""
    tokens = text.lower().split()
    return [stemmer.stem(t) for t in tokens]

documents = [
    "The dogs are running in the park",
    "A dog runs quickly through parks",
    "Cats sleep on comfortable beds",
    "The cat is sleeping peacefully on a bed",
]

# TF-IDF with stemming
tfidf_stem = TfidfVectorizer(tokenizer=stem_tokenizer)
X_stem = tfidf_stem.fit_transform(documents)

# TF-IDF without stemming
tfidf_raw = TfidfVectorizer()
X_raw = tfidf_raw.fit_transform(documents)

print("Cosine similarity D1 vs D2 (related: dogs running/runs):")
print(f"  Without stemming: {cosine_similarity(X_raw[0:1], X_raw[1:2])[0][0]:.3f}")
print(f"  With stemming:    {cosine_similarity(X_stem[0:1], X_stem[1:2])[0][0]:.3f}")

print("\\nCosine similarity D3 vs D4 (related: cats sleeping):")
print(f"  Without stemming: {cosine_similarity(X_raw[2:3], X_raw[3:4])[0][0]:.3f}")
print(f"  With stemming:    {cosine_similarity(X_stem[2:3], X_stem[3:4])[0][0]:.3f}")
# Stemming improves similarity for semantically related documents!`}
        id="code-stem-search"
      />

      <WarningBlock
        title="Stemming Errors"
        content="Stemmers can both over-stem (merging unrelated words: 'university' and 'universe' both stem to 'univers') and under-stem (failing to merge related words: 'alumnus' and 'alumni' produce different stems). These errors can hurt precision in search and classification tasks."
        id="warning-stem-errors"
      />

      <NoteBlock
        type="note"
        title="Modern LLMs and Stemming"
        content="Subword tokenizers (BPE, WordPiece) largely eliminate the need for explicit stemming or lemmatization. They naturally capture morphological structure: 'running' might be tokenized as 'run' + 'ning', letting the model learn the relationship implicitly. However, stemming remains valuable for lightweight search indices, traditional IR systems, and low-resource languages."
        id="note-modern-stemming"
      />
    </div>
  )
}
