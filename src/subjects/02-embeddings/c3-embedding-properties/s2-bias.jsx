import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function Bias() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Bias in Word Embeddings</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Word embeddings trained on large corpora absorb and amplify societal biases present in
        the training text. Bolukbasi et al. (2016) demonstrated that embeddings encode harmful
        stereotypes: "man is to computer programmer as woman is to homemaker." Understanding
        and mitigating these biases is critical when embeddings are used in real-world applications
        like hiring, search, and recommendation systems.
      </p>

      <DefinitionBlock
        title="Embedding Bias"
        definition="Bias in embeddings manifests as systematic geometric relationships between word vectors that reflect societal stereotypes rather than definitional facts. Formally, a gender bias exists when $\cos(\vec{w}, \vec{he} - \vec{she})$ is large for stereotypically male words (e.g., 'surgeon') and negative for stereotypically female words (e.g., 'nurse'), even though these associations are not definitional."
        id="def-bias"
      />

      <ExampleBlock
        title="Detecting Gender Bias via Analogies"
        problem="The analogy framework reveals bias when it completes stereotypical associations."
        steps={[
          { formula: '$\\vec{\\text{man}} - \\vec{\\text{woman}} + \\vec{\\text{nurse}} \\approx \\vec{\\text{doctor}}$', explanation: 'Implies nurse is to woman as doctor is to man -- a harmful stereotype.' },
          { formula: '$\\vec{\\text{he}} - \\vec{\\text{she}} + \\vec{\\text{receptionist}} \\approx \\vec{\\text{boss}}$', explanation: 'Encodes a gendered occupational hierarchy.' },
          { formula: '$\\cos(\\vec{\\text{engineer}}, \\vec{\\text{he}}) > \\cos(\\vec{\\text{engineer}}, \\vec{\\text{she}})$', explanation: 'Direct similarity measurement shows "engineer" is closer to male pronouns.' },
        ]}
        id="ex-bias-analogies"
      />

      <DefinitionBlock
        title="WEAT (Word Embedding Association Test)"
        definition="WEAT (Caliskan et al., 2017) measures bias by computing the differential association between two sets of target words (e.g., male vs female names) and two sets of attribute words (e.g., career vs family terms): $s(X,Y,A,B) = \\sum_{x \\in X} s(x,A,B) - \\sum_{y \\in Y} s(y,A,B)$ where $s(w,A,B) = \\text{mean}_{a \\in A} \\cos(w,a) - \\text{mean}_{b \\in B} \\cos(w,b)$."
        id="def-weat"
      />

      <PythonCode
        title="bias_detection.py"
        id="code-bias"
        code={`import numpy as np

np.random.seed(42)
d = 100

# Simulate embeddings with an embedded gender direction
gender_dir = np.random.randn(d)
gender_dir /= np.linalg.norm(gender_dir)

def make_vec(base_seed, gender_component=0.0):
    np.random.seed(base_seed)
    v = np.random.randn(d) * 0.5
    v += gender_component * gender_dir
    return v / np.linalg.norm(v)

# Words with stereotypical gender associations (from training data)
word_vecs = {
    "he":           make_vec(1, gender_component=-1.0),
    "she":          make_vec(2, gender_component=1.0),
    "engineer":     make_vec(3, gender_component=-0.5),
    "nurse":        make_vec(4, gender_component=0.6),
    "doctor":       make_vec(5, gender_component=-0.3),
    "teacher":      make_vec(6, gender_component=0.3),
    "programmer":   make_vec(7, gender_component=-0.6),
    "receptionist": make_vec(8, gender_component=0.5),
    "scientist":    make_vec(9, gender_component=-0.4),
    "librarian":    make_vec(10, gender_component=0.4),
}

cos = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Measure gender direction projection for each occupation
gender_direction = word_vecs["he"] - word_vecs["she"]
gender_direction /= np.linalg.norm(gender_direction)

print("Gender bias scores (projection onto he-she direction):")
print("(Negative = male-biased, Positive = female-biased)")
print("-" * 50)
occupations = ["engineer", "nurse", "doctor", "teacher",
               "programmer", "receptionist", "scientist", "librarian"]
for occ in occupations:
    proj = np.dot(word_vecs[occ], gender_direction)
    bar = "+" * int(abs(proj) * 30)
    direction = "M" if proj < 0 else "F"
    print(f"  {occ:15s}: {proj:+.3f} [{direction}] {bar}")

# Simple debiasing: remove gender component from occupation words
print("\\n--- After Hard Debiasing ---")
for occ in occupations:
    v = word_vecs[occ]
    # Remove projection onto gender direction
    v_debiased = v - np.dot(v, gender_direction) * gender_direction
    proj = np.dot(v_debiased, gender_direction)
    print(f"  {occ:15s}: {proj:+.6f}")`}
      />

      <DefinitionBlock
        title="Hard Debiasing (Bolukbasi et al., 2016)"
        definition="Hard debiasing neutralizes bias by (1) identifying a gender subspace via PCA on gendered word pairs, (2) zeroing out the gender component for 'neutral' words (occupations, adjectives), and (3) equalizing definitional pairs (e.g., he/she, king/queen) to be equidistant from neutral words. For word $w$ and gender direction $g$: $w_{\text{debiased}} = w - (w \cdot g)\, g$."
        id="def-hard-debias"
      />

      <WarningBlock
        title="Debiasing Has Limitations"
        content="Gonen & Goldberg (2019) showed that hard debiasing is 'lipstick on a pig': while direct bias measures decrease, the debiased embeddings still cluster words by gender. Neighborhood-based metrics reveal that most of the bias information remains encoded in indirect geometric relationships. More sophisticated approaches like contextual debiasing and data-level interventions are needed."
        id="warn-lipstick"
      />

      <NoteBlock
        type="note"
        title="Types of Bias"
        content="Embedding bias extends well beyond gender. Studies have documented racial bias (African-American names associated with negative attributes), religious bias, age bias, and disability bias in standard embeddings. The WEAT framework can measure any of these by choosing appropriate target and attribute word sets."
        id="note-types"
      />

      <NoteBlock
        type="tip"
        title="Best Practices"
        content="Always audit embeddings for bias before deployment. Use multiple measurement methods (WEAT, analogy tests, cluster analysis). Consider training on more balanced corpora, applying post-hoc debiasing, and evaluating downstream task fairness rather than just intrinsic embedding metrics. Document known biases in model cards."
        id="note-practices"
      />
    </div>
  )
}
