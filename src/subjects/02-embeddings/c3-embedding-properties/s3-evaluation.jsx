import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function Evaluation() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Evaluation Methods for Word Embeddings</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        How do we know if one set of word embeddings is better than another? Evaluation methods
        fall into two categories: intrinsic evaluations that test embedding properties directly
        (similarity, analogies) and extrinsic evaluations that measure performance on downstream
        NLP tasks. A good embedding should perform well on both, though they do not always agree.
      </p>

      <DefinitionBlock
        title="Intrinsic Evaluation"
        definition="Intrinsic evaluation measures the quality of embeddings directly by testing geometric properties of the vector space. Common intrinsic tasks include word similarity (correlating cosine similarity with human judgments), analogy completion ($a:b :: c:?$), and categorization (clustering words into semantic categories)."
        id="def-intrinsic"
      />

      <DefinitionBlock
        title="Extrinsic Evaluation"
        definition="Extrinsic evaluation measures embedding quality indirectly by using embeddings as features in a downstream task (e.g., sentiment analysis, NER, parsing) and comparing task performance. Better embeddings should yield better downstream results, though the relationship is not always monotonic."
        id="def-extrinsic"
      />

      <ExampleBlock
        title="Word Similarity Benchmarks"
        problem="Key benchmarks for measuring word similarity correlation:"
        steps={[
          { formula: 'WordSim-353 (Finkelstein et al., 2001)', explanation: '353 word pairs rated 0-10 by humans. Mixes similarity and relatedness (e.g., "car"-"gasoline" rated high).' },
          { formula: 'SimLex-999 (Hill et al., 2015)', explanation: '999 pairs rated for genuine similarity (not just relatedness). "Car"-"gasoline" scores low; "car"-"automobile" scores high.' },
          { formula: 'MEN (Bruni et al., 2014)', explanation: '3,000 pairs with crowd-sourced ratings. Large size gives more statistical power.' },
          { formula: 'Metric: Spearman $\\rho$ correlation', explanation: 'Rank correlation between human scores and cosine similarities. Higher is better.' },
        ]}
        id="ex-benchmarks"
      />

      <NoteBlock
        type="intuition"
        title="Similarity vs Relatedness"
        content="WordSim-353 conflates two distinct notions: similarity (car/automobile -- same concept) and relatedness (car/gasoline -- associated but different). SimLex-999 was specifically designed to test only similarity. This distinction matters because different embeddings may excel at one but not the other, and downstream tasks may need either."
        id="note-sim-rel"
      />

      <PythonCode
        title="embedding_evaluation.py"
        id="code-eval"
        code={`import numpy as np
from scipy.stats import spearmanr

# Simulated word similarity dataset (like SimLex-999)
# Format: (word1, word2, human_score)
similarity_pairs = [
    ("happy", "joyful", 9.2),
    ("happy", "sad", 1.5),
    ("car", "automobile", 8.8),
    ("car", "gasoline", 3.1),  # related but not similar
    ("dog", "cat", 6.5),
    ("dog", "computer", 0.8),
    ("king", "queen", 7.0),
    ("king", "throne", 4.2),   # related but not similar
    ("big", "large", 8.5),
    ("big", "small", 2.0),
]

# Simulated embeddings (in practice, load real ones)
np.random.seed(42)
d = 50

def make_similar_vecs(base_seed, similarity):
    """Create two vectors with approximately given cosine similarity."""
    np.random.seed(base_seed)
    v1 = np.random.randn(d)
    v1 /= np.linalg.norm(v1)
    # Mix v1 with random vector to control similarity
    noise = np.random.randn(d)
    noise /= np.linalg.norm(noise)
    v2 = similarity * v1 + (1 - abs(similarity)) * noise
    v2 /= np.linalg.norm(v2)
    return v1, v2

# Generate embeddings that roughly correlate with human scores
word_vecs = {}
for i, (w1, w2, score) in enumerate(similarity_pairs):
    target_sim = (score / 10.0) * 0.8 + np.random.randn() * 0.1
    v1, v2 = make_similar_vecs(i * 100, target_sim)
    word_vecs[w1] = v1
    word_vecs[w2] = v2

# Compute cosine similarities
cos = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

human_scores = []
model_scores = []
print(f"{'Pair':30s} {'Human':>6s} {'Cosine':>7s}")
print("-" * 48)
for w1, w2, score in similarity_pairs:
    model_sim = cos(word_vecs[w1], word_vecs[w2])
    human_scores.append(score)
    model_scores.append(model_sim)
    print(f"{w1 + ' / ' + w2:30s} {score:6.1f} {model_sim:7.3f}")

# Spearman rank correlation
rho, pvalue = spearmanr(human_scores, model_scores)
print(f"\\nSpearman rho: {rho:.3f} (p={pvalue:.4f})")
print(f"Interpretation: {'strong' if rho > 0.6 else 'moderate' if rho > 0.4 else 'weak'} correlation")

# Analogy evaluation (accuracy metric)
print("\\n--- Analogy Accuracy ---")
print("Standard benchmarks: Google Analogy (19,544 pairs)")
print("Categories: semantic (capital-country, gender) + syntactic (tense, plural)")
print("Typical accuracy: Word2Vec ~75%, GloVe ~75%, FastText ~78%")`}
      />

      <WarningBlock
        title="Intrinsic-Extrinsic Disconnect"
        content="Higher intrinsic scores do not always predict better downstream performance. Embeddings optimized for word similarity may not be optimal for NER or parsing. Chiu et al. (2016) showed that intrinsic metrics explain only a fraction of the variance in extrinsic task performance. Always evaluate on your actual downstream task."
        id="warn-disconnect"
      />

      <NoteBlock
        type="note"
        title="The MTEB Benchmark"
        content="The Massive Text Embedding Benchmark (MTEB, Muennighoff et al., 2023) provides a comprehensive evaluation framework covering 8 task types (classification, clustering, pair classification, reranking, retrieval, STS, summarization, and zero-shot classification) across 58 datasets and 112 languages. It is the current standard for comparing embedding models."
        id="note-mteb"
      />

      <NoteBlock
        type="tip"
        title="Practical Evaluation Advice"
        content="Start with MTEB scores to shortlist candidate models. Then evaluate on a held-out sample from your specific domain. Key metrics to check: retrieval recall@k for search applications, Spearman correlation for similarity tasks, and clustering quality (V-measure) for topic modeling. Always compare against a simple TF-IDF baseline."
        id="note-practical"
      />
    </div>
  )
}
