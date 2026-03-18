import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function FastText() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">FastText: Subword Embeddings</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        FastText (Bojanowski et al., 2017) extends Word2Vec by representing each word as a bag of
        character n-grams. This simple modification yields two major benefits: it can generate
        embeddings for out-of-vocabulary (OOV) words, and it naturally captures morphological
        patterns (e.g., the shared structure in "teach", "teacher", "teaching").
      </p>

      <DefinitionBlock
        title="Character N-gram Representation"
        definition="A word $w$ is represented as the set of its character n-grams $\mathcal{G}_w$ (for $n$ between $n_{\min}$ and $n_{\max}$), plus the whole word itself. The word embedding is the sum of its n-gram vectors: $\mathbf{v}_w = \mathbf{z}_w + \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$ where $\mathbf{z}_w$ is the whole-word vector and $\mathbf{z}_g$ are n-gram vectors."
        notation="Special boundary markers < and > are added: 'where' becomes '<where>'. Default n-gram range: $n_{\min}=3$, $n_{\max}=6$."
        id="def-subword"
      />

      <ExampleBlock
        title="Character N-grams for 'where'"
        problem="Compute the character n-grams for the word 'where' with $n_{\min}=3$ and $n_{\max}=5$, using boundary markers."
        steps={[
          { formula: 'Marked form: <where>', explanation: 'Add boundary markers < and >.' },
          { formula: 'n=3: <wh, whe, her, ere, re>', explanation: 'All character trigrams.' },
          { formula: 'n=4: <whe, wher, here, ere>', explanation: 'All character 4-grams.' },
          { formula: 'n=5: <wher, where, here>', explanation: 'All character 5-grams.' },
          { formula: 'Total: 13 n-grams + whole word', explanation: 'The word vector is the sum of all 14 vectors.' },
        ]}
        id="ex-ngrams"
      />

      <NoteBlock
        type="intuition"
        title="Why Subwords Help with OOV Words"
        content="When encountering an unseen word like 'unforgettably', FastText decomposes it into known n-grams (e.g., 'unf', 'for', 'get', 'tab', 'bly') that overlap with training words like 'unfair', 'forget', 'table', and 'ably'. The resulting embedding is a meaningful composition of these shared morphological fragments, rather than a zero vector or random initialization."
        id="note-oov"
      />

      <PythonCode
        title="fasttext_demo.py"
        id="code-fasttext"
        code={`import numpy as np

# Demonstrate character n-gram extraction
def get_ngrams(word, n_min=3, n_max=6):
    """Extract character n-grams with boundary markers."""
    marked = f"<{word}>"
    ngrams = []
    for n in range(n_min, n_max + 1):
        for i in range(len(marked) - n + 1):
            ngrams.append(marked[i:i+n])
    return ngrams

# Show n-grams for different words
words = ["where", "teacher", "teaching", "unteachable"]
for w in words:
    ng = get_ngrams(w)
    print(f"{w:15s} -> {len(ng)} n-grams")
    print(f"  Sample: {ng[:5]}...")

# Show shared n-grams between related words
def shared_ngrams(w1, w2):
    ng1 = set(get_ngrams(w1))
    ng2 = set(get_ngrams(w2))
    shared = ng1 & ng2
    return shared

shared = shared_ngrams("teacher", "teaching")
print(f"\\nShared n-grams between 'teacher' and 'teaching':")
print(f"  {sorted(shared)}")
print(f"  Count: {len(shared)}")

# Simulate FastText embedding computation
np.random.seed(42)
d = 10
ngram_vectors = {}  # pretend these are learned

def get_fasttext_embedding(word):
    """Compute word embedding as sum of n-gram vectors."""
    ngrams = get_ngrams(word) + [word]  # n-grams + whole word
    total = np.zeros(d)
    for ng in ngrams:
        if ng not in ngram_vectors:
            # Hash-based bucket (FastText uses hashing)
            h = hash(ng) % 2_000_000
            np.random.seed(h)
            ngram_vectors[ng] = np.random.randn(d) * 0.1
        total += ngram_vectors[ng]
    return total / len(ngrams)  # average for stability

# Even an OOV word gets a reasonable embedding
emb_teach = get_fasttext_embedding("teach")
emb_teacher = get_fasttext_embedding("teacher")
emb_quantum = get_fasttext_embedding("quantum")

cos = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"\\nsim(teach, teacher) = {cos(emb_teach, emb_teacher):.3f}")
print(f"sim(teach, quantum)  = {cos(emb_teach, emb_quantum):.3f}")`}
      />

      <WarningBlock
        title="Hashing Collisions"
        content="FastText uses a hash function to map n-grams to a fixed number of buckets (default 2 million). Different n-grams may collide into the same bucket, sharing a vector. This is usually benign for large bucket counts but can degrade quality if the vocabulary is very large relative to the number of buckets."
        id="warn-hashing"
      />

      <NoteBlock
        type="note"
        title="Pre-trained FastText Models"
        content="Facebook AI Research released pre-trained FastText vectors for 157 languages, trained on Common Crawl and Wikipedia. These models are especially valuable for morphologically rich languages (Finnish, Turkish, Arabic) where a single lemma can have dozens of surface forms. The subword approach handles these naturally."
        id="note-pretrained"
      />

      <NoteBlock
        type="tip"
        title="Using FastText in Python"
        content="Install the official fasttext library (pip install fasttext) for training, or use gensim's FastText wrapper. For just loading pre-trained vectors, gensim.models.fasttext.load_facebook_model() handles the .bin format which preserves subword information for OOV inference."
        id="note-usage"
      />
    </div>
  )
}
