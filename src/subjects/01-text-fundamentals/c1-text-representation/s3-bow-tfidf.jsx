import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function BowTfidf() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Bag-of-Words and TF-IDF</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Before neural embeddings, the standard way to represent documents as numerical vectors
        was through count-based methods. Bag-of-Words and TF-IDF remain important baselines and
        are still used in information retrieval, search engines, and lightweight classifiers.
      </p>

      <DefinitionBlock
        title="Bag-of-Words (BoW)"
        definition="A Bag-of-Words representation converts a document into a vector of word counts (or binary indicators), discarding word order. Each dimension corresponds to a unique word in the vocabulary."
        notation="For a vocabulary of size $V$, each document $d$ becomes a vector $\mathbf{x} \in \mathbb{R}^V$ where $x_i = \text{count}(w_i, d)$."
        id="def-bow"
      />

      <DefinitionBlock
        title="Term Frequency (TF)"
        definition="Term frequency measures how often a term $t$ appears in a document $d$. The raw count is often normalized to prevent bias toward longer documents."
        id="def-tf"
      />

      <div className="my-4">
        <BlockMath math="\text{tf}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}" />
        <p className="text-center text-sm text-gray-500 dark:text-gray-400">
          where <InlineMath math="f_{t,d}" /> is the raw count of term <InlineMath math="t" /> in document <InlineMath math="d" />.
        </p>
      </div>

      <DefinitionBlock
        title="Inverse Document Frequency (IDF)"
        definition="IDF measures how informative a term is across the entire corpus. Rare terms get higher IDF scores, while common terms (like 'the') get lower scores."
        id="def-idf"
      />

      <div className="my-4">
        <BlockMath math="\text{idf}(t, D) = \log \frac{|D|}{|\{d \in D : t \in d\}|}" />
        <p className="text-center text-sm text-gray-500 dark:text-gray-400">
          where <InlineMath math="|D|" /> is the total number of documents and the denominator counts documents containing term <InlineMath math="t" />.
        </p>
      </div>

      <TheoremBlock
        title="TF-IDF Score"
        statement="The TF-IDF weight of a term $t$ in document $d$ within corpus $D$ is the product: $\text{tfidf}(t, d, D) = \text{tf}(t, d) \times \text{idf}(t, D)$. This balances local importance (term frequency) with global rarity (inverse document frequency)."
        id="theorem-tfidf"
      />

      <ExampleBlock
        title="Computing TF-IDF by Hand"
        problem="Given 3 documents: D1='the cat sat', D2='the dog sat', D3='the cat played'. Compute TF-IDF for 'cat' in D1."
        steps={[
          { formula: '$\\text{tf}(\\text{cat}, D_1) = 1/3 \\approx 0.333$', explanation: '"cat" appears once out of 3 total words in D1.' },
          { formula: '$\\text{idf}(\\text{cat}, D) = \\log(3/2) \\approx 0.405$', explanation: '"cat" appears in 2 out of 3 documents.' },
          { formula: '$\\text{tfidf} = 0.333 \\times 0.405 \\approx 0.135$', explanation: 'Multiply TF by IDF to get the final weight.' },
        ]}
        id="example-tfidf"
      />

      <PythonCode
        title="bow_tfidf_sklearn.py"
        code={`from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

documents = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "The cat chased the dog",
]

# Bag-of-Words
bow = CountVectorizer()
X_bow = bow.fit_transform(documents)
print("Vocabulary:", bow.get_feature_names_out())
print("BoW matrix (dense):")
print(X_bow.toarray())
# Each row = document, each column = word count

# TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(documents)
print("\\nTF-IDF matrix:")
print(X_tfidf.toarray().round(3))

# Document similarity using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(X_tfidf)
print("\\nCosine similarity between documents:")
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        print(f"  D{i+1} vs D{j+1}: {sim_matrix[i][j]:.3f}")`}
        id="code-tfidf"
      />

      <NoteBlock
        type="intuition"
        title="Why TF-IDF Works for Search"
        content="When you search for 'transformer architecture', TF-IDF naturally boosts documents that mention these specific terms frequently (high TF) while downweighting documents that merely contain common words like 'the' or 'is' (low IDF). This is exactly what makes it the backbone of traditional search engines."
        id="note-tfidf-intuition"
      />

      <NoteBlock
        type="note"
        title="Limitations of Bag-of-Words"
        content="BoW and TF-IDF ignore word order entirely: 'dog bites man' and 'man bites dog' produce identical vectors. They also cannot capture synonymy (different words, same meaning) or polysemy (same word, different meanings). Neural embeddings like Word2Vec and BERT address these shortcomings."
        id="note-bow-limitations"
      />
    </div>
  )
}
