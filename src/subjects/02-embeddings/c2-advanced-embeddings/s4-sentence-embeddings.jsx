import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function SentenceEmbeddings() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Sentence & Document Embeddings</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        While word embeddings capture the meaning of individual tokens, many applications require
        fixed-size representations of entire sentences, paragraphs, or documents. The challenge
        is to compose variable-length sequences of word vectors into a single vector that preserves
        semantic meaning for tasks like semantic search, clustering, and similarity comparison.
      </p>

      <DefinitionBlock
        title="Sentence Embedding"
        definition="A sentence embedding is a function $f: \mathcal{S} \to \mathbb{R}^d$ mapping a variable-length sequence of tokens to a fixed-dimensional vector such that semantically similar sentences are mapped to nearby points: $\text{sim}(s_1, s_2) \text{ high} \implies \cos(f(s_1), f(s_2)) \text{ high}$."
        id="def-sent-emb"
      />

      <DefinitionBlock
        title="Average Pooling Baseline"
        definition="The simplest sentence embedding averages the word embeddings: $\mathbf{s} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{v}_{w_i}$. Despite its simplicity, this often works surprisingly well, especially with weighted variants like SIF (Smooth Inverse Frequency) that down-weight common words: $\mathbf{s} = \frac{1}{n}\sum_{i=1}^{n} \frac{a}{a + p(w_i)} \mathbf{v}_{w_i}$ where $a$ is a hyperparameter and $p(w_i)$ is word frequency."
        id="def-avg-pool"
      />

      <PythonCode
        title="sentence_embedding_basics.py"
        id="code-avg"
        code={`import numpy as np
from collections import Counter

# Simulated word embeddings (50-dim)
np.random.seed(42)
d = 50
word_vecs = {
    "the": np.random.randn(d) * 0.1,
    "cat": np.random.randn(d) + np.array([1]*25 + [0]*25),
    "sat": np.random.randn(d) * 0.5,
    "on": np.random.randn(d) * 0.1,
    "mat": np.random.randn(d) + np.array([0]*25 + [1]*25),
    "dog": np.random.randn(d) + np.array([1]*25 + [0.1]*25),
    "lay": np.random.randn(d) * 0.5,
    "rug": np.random.randn(d) + np.array([0]*25 + [0.9]*25),
}

# Word frequencies for SIF weighting
word_freq = {"the": 0.07, "cat": 0.001, "sat": 0.002, "on": 0.03,
             "mat": 0.0005, "dog": 0.002, "lay": 0.001, "rug": 0.0004}

def avg_embedding(sentence, word_vecs):
    """Simple average of word vectors."""
    words = sentence.lower().split()
    vecs = [word_vecs[w] for w in words if w in word_vecs]
    return np.mean(vecs, axis=0) if vecs else np.zeros(d)

def sif_embedding(sentence, word_vecs, word_freq, a=1e-3):
    """SIF-weighted average (Arora et al., 2017)."""
    words = sentence.lower().split()
    vecs = []
    for w in words:
        if w in word_vecs:
            weight = a / (a + word_freq.get(w, 1e-4))
            vecs.append(weight * word_vecs[w])
    return np.mean(vecs, axis=0) if vecs else np.zeros(d)

# Compare three sentences
s1 = "the cat sat on the mat"
s2 = "the dog lay on the rug"
s3 = "the mat sat on the cat"  # same words, different meaning!

cos = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

print("=== Simple Average ===")
e1, e2, e3 = avg_embedding(s1, word_vecs), avg_embedding(s2, word_vecs), avg_embedding(s3, word_vecs)
print(f"sim(s1, s2) = {cos(e1, e2):.3f}  (similar meaning)")
print(f"sim(s1, s3) = {cos(e1, e3):.3f}  (same words, different meaning)")

print("\\n=== SIF Weighted ===")
e1, e2, e3 = sif_embedding(s1, word_vecs, word_freq), sif_embedding(s2, word_vecs, word_freq), sif_embedding(s3, word_vecs, word_freq)
print(f"sim(s1, s2) = {cos(e1, e2):.3f}")
print(f"sim(s1, s3) = {cos(e1, e3):.3f}")
print("\\nNote: averaging ignores word order, so s1 and s3 are identical!")`}
      />

      <WarningBlock
        title="Bag-of-Words Loses Word Order"
        content="Averaging word embeddings creates a bag-of-words representation that is invariant to word order. 'The cat sat on the mat' and 'The mat sat on the cat' produce identical embeddings. For tasks where order matters, use encoder-based models like Sentence-BERT."
        id="warn-order"
      />

      <DefinitionBlock
        title="Sentence-BERT (SBERT)"
        definition="Sentence-BERT (Reimers & Gurevych, 2019) fine-tunes a pre-trained BERT model using siamese and triplet networks to produce semantically meaningful sentence embeddings. It uses mean pooling over BERT token outputs and trains with a combination of classification (NLI) and regression (STS) objectives, making cosine similarity directly meaningful."
        id="def-sbert"
      />

      <PythonCode
        title="sentence_transformers_demo.py"
        id="code-sbert"
        code={`# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pre-trained sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode sentences
sentences = [
    "The cat sat on the mat.",
    "A dog was lying on the rug.",
    "Machine learning is a subfield of AI.",
    "Neural networks learn from data.",
    "The feline rested upon the carpet.",  # paraphrase of sentence 1
]

embeddings = model.encode(sentences)
print(f"Embedding shape: {embeddings.shape}")  # (5, 384)

# Compute pairwise cosine similarities
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)

print("\\nPairwise cosine similarities:")
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        print(f"  [{i}] vs [{j}]: {sim_matrix[i][j]:.3f}  "
              f"({sentences[i][:30]}... vs {sentences[j][:30]}...)")

# Semantic search: find most similar to a query
query = "A pet was resting on the floor"
query_emb = model.encode([query])
scores = cosine_similarity(query_emb, embeddings)[0]
best = np.argmax(scores)
print(f"\\nQuery: '{query}'")
print(f"Best match: '{sentences[best]}' (score: {scores[best]:.3f})")`}
      />

      <NoteBlock
        type="tip"
        title="Choosing a Sentence Embedding Model"
        content="For general-purpose English tasks, 'all-MiniLM-L6-v2' offers an excellent speed/quality trade-off (384 dims, 80MB). For maximum quality, use 'all-mpnet-base-v2' (768 dims). For multilingual support, use 'paraphrase-multilingual-MiniLM-L12-v2'. For instruction-following embeddings (where you describe the task), consider Instructor or E5 models."
        id="note-choosing"
      />

      <NoteBlock
        type="note"
        title="Beyond Sentence-BERT"
        content="Recent developments include instruction-tuned embeddings (Instructor, E5) that condition on task descriptions, and Matryoshka embeddings that support flexible dimensionality -- you can truncate the vector to any prefix length (e.g., 64, 128, 256 dims) while maintaining quality, enabling cost-quality trade-offs at query time."
        id="note-beyond"
      />
    </div>
  )
}
