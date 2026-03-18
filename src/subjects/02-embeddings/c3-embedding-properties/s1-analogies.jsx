import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function Analogies() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Analogies & Vector Arithmetic</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        One of the most striking properties of word embeddings is that semantic relationships
        are encoded as consistent vector offsets. The famous example{' '}
        <InlineMath math="\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}" />{' '}
        demonstrates that the direction from "man" to "woman" captures a gender relationship
        that can be transferred to other word pairs. This section explores the mathematics
        and limitations of embedding analogies.
      </p>

      <DefinitionBlock
        title="Vector Analogy"
        definition="An analogy 'a is to b as c is to ?' is solved by finding the word $d$ whose embedding is closest to $\mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c$: $d^* = \arg\max_{d \in \mathcal{V} \setminus \{a,b,c\}} \cos(\mathbf{v}_d, \mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c)$."
        notation="The offset $\mathbf{v}_b - \mathbf{v}_a$ encodes the relationship between $a$ and $b$. Adding this offset to $\mathbf{v}_c$ yields a target point near $\mathbf{v}_d$."
        id="def-analogy"
      />

      <ExampleBlock
        title="Classic Analogy Examples"
        problem="What relationships do these vector offsets encode?"
        steps={[
          { formula: '$\\vec{\\text{king}} - \\vec{\\text{man}} + \\vec{\\text{woman}} \\approx \\vec{\\text{queen}}$', explanation: 'Gender relationship: the male-female offset transfers from common nouns to royalty.' },
          { formula: '$\\vec{\\text{Paris}} - \\vec{\\text{France}} + \\vec{\\text{Italy}} \\approx \\vec{\\text{Rome}}$', explanation: 'Capital-country relationship: a geographic relational offset.' },
          { formula: '$\\vec{\\text{walking}} - \\vec{\\text{walk}} + \\vec{\\text{swim}} \\approx \\vec{\\text{swimming}}$', explanation: 'Morphological relationship: the tense offset generalizes across verbs.' },
        ]}
        id="ex-analogies"
      />

      <p className="text-gray-700 dark:text-gray-300">
        The analogy works because the cosine similarity objective can be decomposed:
      </p>
      <BlockMath math={String.raw`\cos(\mathbf{v}_d, \mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c) \propto \mathbf{v}_d^\top\mathbf{v}_b - \mathbf{v}_d^\top\mathbf{v}_a + \mathbf{v}_d^\top\mathbf{v}_c`} />
      <p className="text-gray-700 dark:text-gray-300">
        This means we seek a word <InlineMath math="d" /> that is similar to <InlineMath math="b" /> and{' '}
        <InlineMath math="c" /> but dissimilar from <InlineMath math="a" />.
      </p>

      <PythonCode
        title="analogy_demo.py"
        id="code-analogy"
        code={`import numpy as np
from gensim.models import KeyedVectors

# Load pre-trained Word2Vec (Google News, 300-dim)
# Download: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM
# wv = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# For demo: simulate with small random embeddings
np.random.seed(42)
d = 100
words = ["king", "queen", "man", "woman", "prince", "princess",
         "paris", "france", "rome", "italy", "berlin", "germany",
         "walk", "walking", "swim", "swimming", "run", "running"]

# Create embeddings with structure
vecs = {}
gender_dir = np.random.randn(d)
gender_dir /= np.linalg.norm(gender_dir)
royal_dir = np.random.randn(d)
royal_dir /= np.linalg.norm(royal_dir)

base = np.random.randn(d) * 0.3
vecs["man"]   = base + 0.0 * gender_dir + 0.0 * royal_dir
vecs["woman"] = base + 1.0 * gender_dir + 0.0 * royal_dir
vecs["king"]  = base + 0.0 * gender_dir + 1.0 * royal_dir
vecs["queen"] = base + 1.0 * gender_dir + 1.0 * royal_dir
# Add noise to all
for w in vecs:
    vecs[w] += np.random.randn(d) * 0.1

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def solve_analogy(a, b, c, vecs, exclude=None):
    """Solve: a is to b as c is to ?"""
    target = vecs[b] - vecs[a] + vecs[c]
    exclude = exclude or {a, b, c}
    best_word, best_sim = None, -1
    for w, v in vecs.items():
        if w in exclude:
            continue
        sim = cosine_sim(target, v)
        if sim > best_sim:
            best_word, best_sim = w, sim
    return best_word, best_sim

# Test: man -> woman :: king -> ?
answer, score = solve_analogy("man", "woman", "king", vecs)
print(f"man : woman :: king : {answer} (sim={score:.3f})")

# Show the offset vectors
gender_offset = vecs["woman"] - vecs["man"]
royal_offset = vecs["queen"] - vecs["king"]
print(f"\\nCosine(woman-man, queen-king) = {cosine_sim(gender_offset, royal_offset):.3f}")
print("(High similarity = consistent gender direction)")`}
      />

      <WarningBlock
        title="Analogies Are Not Always Reliable"
        content="The analogy task has a success rate of only about 60-75% on standard benchmarks, even with good embeddings. Many failures stem from polysemy (words with multiple meanings), frequency imbalances, or relationships that are not well-captured by linear offsets. Do not treat analogy accuracy as the sole measure of embedding quality."
        id="warn-reliability"
      />

      <NoteBlock
        type="note"
        title="Alternative: 3CosAdd vs 3CosMul"
        content="The standard additive method (3CosAdd) can be improved by the multiplicative method (3CosMul) proposed by Levy & Goldberg (2014): d* = argmax cos(d,b)*cos(d,c) / (cos(d,a) + epsilon). This avoids the issue where one large similarity term can dominate the additive formulation."
        id="note-cosmul"
      />

      <NoteBlock
        type="intuition"
        title="Why Linear Offsets Emerge"
        content="Levy & Goldberg (2014) showed that Word2Vec implicitly factorizes a PMI matrix, and PMI differences correspond to log-probability ratios. When a consistent relationship (like gender) shifts co-occurrence patterns by a constant factor across word pairs, it manifests as a constant vector offset in the embedding space. The linearity is a consequence of the log-bilinear structure of the training objective."
        id="note-why-linear"
      />
    </div>
  )
}
