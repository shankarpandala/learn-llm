import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function DataQuality() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Deduplication and Data Filtering</h1>
      <p className="text-lg text-gray-300">
        Web-scale datasets contain massive amounts of duplicate and near-duplicate content.
        Deduplication is critical: training on duplicated data wastes compute, degrades model
        quality, and increases memorization risk. Filtering removes toxic, low-quality, or
        personally identifiable content.
      </p>

      <DefinitionBlock
        title="Exact Deduplication"
        definition="Exact deduplication removes documents with identical content. This is typically implemented using hash-based methods: compute a hash $h(d)$ (e.g., SHA-256) for each document $d$, then remove all but one document per hash value."
        notation="Two documents are exact duplicates iff $h(d_1) = h(d_2)$. Exact dedup typically removes 10-30% of raw web data."
        id="exact-dedup-def"
      />

      <DefinitionBlock
        title="Fuzzy (Near) Deduplication"
        definition="Near-deduplication identifies documents that are highly similar but not identical. MinHash with Locality-Sensitive Hashing (LSH) approximates Jaccard similarity between document n-gram sets. Documents with $J(d_1, d_2) > \\theta$ (typically $\\theta = 0.8$) are considered near-duplicates."
        notation="Jaccard similarity: $J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}$ where $A, B$ are n-gram sets."
        id="fuzzy-dedup-def"
      />

      <ExampleBlock
        title="MinHash Deduplication"
        problem="Use MinHash-LSH to find near-duplicate documents in a corpus."
        steps={[
          {
            formula: '\\text{Step 1: Extract n-grams (e.g., 5-grams) from each document}',
            explanation: 'Convert "the cat sat on the" to {(the,cat,sat,on,the), (cat,sat,on,the,...)}.'
          },
          {
            formula: '\\text{Step 2: Compute } k \\text{ MinHash signatures per document}',
            explanation: 'Apply k random hash functions to the n-gram set. Each signature is the minimum hash value. Typical k=128.'
          },
          {
            formula: 'P(\\text{MinHash}(A) = \\text{MinHash}(B)) = J(A, B)',
            explanation: 'MinHash probability equals Jaccard similarity, enabling fast approximation.'
          },
          {
            formula: '\\text{Step 3: LSH bands to find candidate pairs efficiently}',
            explanation: 'Split signatures into b bands of r rows. Documents matching in any band are candidates. Tune b,r to control precision/recall.'
          }
        ]}
        id="minhash-example"
      />

      <NoteBlock
        type="tip"
        title="Deduplication at Scale"
        content="For trillion-token datasets, exact dedup uses Bloom filters (space-efficient probabilistic set). MinHash-LSH for near-dedup is parallelizable with MapReduce. The FineWeb dataset used 5-gram MinHash with 128 hash functions and found ~40% near-duplicates in Common Crawl. Removing these significantly improved downstream benchmarks."
        id="dedup-scale-note"
      />

      <PythonCode
        title="deduplication.py"
        code={`import hashlib
from collections import defaultdict
import random

# 1. Exact deduplication with hashing
def exact_dedup(documents):
    """Remove exact duplicate documents using SHA-256."""
    seen = set()
    unique = []
    for doc in documents:
        doc_hash = hashlib.sha256(doc.encode()).hexdigest()
        if doc_hash not in seen:
            seen.add(doc_hash)
            unique.append(doc)
    removed = len(documents) - len(unique)
    print(f"Exact dedup: {len(documents)} -> {len(unique)} ({removed} removed)")
    return unique

# 2. MinHash for near-deduplication
class MinHashDedup:
    def __init__(self, num_perm=128, ngram_size=5, threshold=0.8):
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.threshold = threshold
        self.max_hash = (1 << 32) - 1
        # Random hash parameters
        self.a = [random.randint(1, self.max_hash) for _ in range(num_perm)]
        self.b = [random.randint(0, self.max_hash) for _ in range(num_perm)]
        self.prime = (1 << 61) - 1

    def get_ngrams(self, text):
        words = text.lower().split()
        return set(tuple(words[i:i+self.ngram_size])
                   for i in range(len(words) - self.ngram_size + 1))

    def minhash_signature(self, text):
        ngrams = self.get_ngrams(text)
        sig = []
        for i in range(self.num_perm):
            min_val = float('inf')
            for ng in ngrams:
                h = hash(ng)
                val = (self.a[i] * h + self.b[i]) % self.prime
                min_val = min(min_val, val)
            sig.append(min_val)
        return sig

    def jaccard_estimate(self, sig1, sig2):
        return sum(a == b for a, b in zip(sig1, sig2)) / len(sig1)

    def deduplicate(self, documents):
        sigs = [self.minhash_signature(doc) for doc in documents]
        to_remove = set()
        for i in range(len(documents)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(documents)):
                if j in to_remove:
                    continue
                sim = self.jaccard_estimate(sigs[i], sigs[j])
                if sim >= self.threshold:
                    to_remove.add(j)
        result = [d for i, d in enumerate(documents) if i not in to_remove]
        print(f"MinHash dedup: {len(documents)} -> {len(result)}")
        return result

# Demo
docs = [
    "The cat sat on the mat and looked around the room quietly",
    "The cat sat on the mat and looked around the room quietly",  # exact
    "The cat sat on the mat and looked around the room softly",   # near-dup
    "Quantum computing uses qubits for parallel computation",
    "Machine learning models require large training datasets",
]
clean = exact_dedup(docs)
deduper = MinHashDedup(num_perm=64, ngram_size=3, threshold=0.7)
clean = deduper.deduplicate(clean)
for d in clean:
    print(f"  {d[:60]}...")`}
        id="dedup-code"
      />

      <WarningBlock
        title="Over-Deduplication Can Hurt"
        content="Aggressive deduplication can remove legitimate repeated content like common phrases, templates, or code patterns that the model should learn. Some studies show that moderate duplication (2-3x) of high-quality data can be beneficial. The goal is removing noise and spam duplication, not eliminating all repetition."
        id="over-dedup-warning"
      />
    </div>
  )
}
