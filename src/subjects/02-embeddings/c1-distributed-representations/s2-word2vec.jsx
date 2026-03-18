import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function Word2Vec() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Word2Vec: CBOW & Skip-gram</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Word2Vec (Mikolov et al., 2013) introduced two efficient architectures for learning word
        embeddings from raw text: Continuous Bag-of-Words (CBOW) and Skip-gram. Both exploit the
        distributional hypothesis by training a shallow neural network to predict words from their
        context or vice versa. The learned weight matrices become the word embeddings.
      </p>

      <DefinitionBlock
        title="Continuous Bag-of-Words (CBOW)"
        definition="CBOW predicts a target word $w_t$ given its surrounding context words $\{w_{t-c}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+c}\}$. The context vectors are averaged and passed through a linear layer with softmax to produce a probability distribution over the vocabulary."
        notation="Context window size $c$; input embeddings $\mathbf{W} \in \mathbb{R}^{V \times d}$; output weights $\mathbf{W}' \in \mathbb{R}^{d \times V}$."
        id="def-cbow"
      />

      <p className="text-gray-700 dark:text-gray-300">
        The CBOW objective maximizes:
      </p>
      <BlockMath math={String.raw`\mathcal{L}_{\text{CBOW}} = \frac{1}{T}\sum_{t=1}^{T} \log P(w_t \mid w_{t-c}, \dots, w_{t+c})`} />
      <p className="text-gray-700 dark:text-gray-300">
        where the conditional probability uses softmax over the vocabulary:
      </p>
      <BlockMath math={String.raw`P(w_t \mid \text{ctx}) = \frac{\exp(\mathbf{v}'_{w_t}{}^\top \bar{\mathbf{v}}_{\text{ctx}})}{\sum_{w=1}^{V} \exp(\mathbf{v}'_{w}{}^\top \bar{\mathbf{v}}_{\text{ctx}})}`} />

      <DefinitionBlock
        title="Skip-gram"
        definition="Skip-gram reverses CBOW: given a center word $w_t$, it predicts each context word $w_{t+j}$ (for $-c \\leq j \\leq c$, $j \\neq 0$) independently. This makes Skip-gram especially effective for rare words because each word appears as a center word in multiple training examples."
        id="def-skipgram"
      />

      <p className="text-gray-700 dark:text-gray-300">
        The Skip-gram objective maximizes:
      </p>
      <BlockMath math={String.raw`\mathcal{L}_{\text{SG}} = \frac{1}{T}\sum_{t=1}^{T} \sum_{\substack{-c \leq j \leq c \\ j \neq 0}} \log P(w_{t+j} \mid w_t)`} />

      <NoteBlock
        type="intuition"
        title="CBOW vs Skip-gram Trade-offs"
        content="CBOW is faster to train and performs slightly better on frequent words because it averages context (smoothing noise). Skip-gram is slower but excels on rare words and small datasets because each center-context pair creates a separate training example. In practice, Skip-gram with negative sampling is the most widely used variant."
        id="note-tradeoffs"
      />

      <ExampleBlock
        title="Training Pairs from a Sentence"
        problem="Given the sentence 'the cat sat on the mat' and window size $c=2$, generate the Skip-gram training pairs for center word 'sat'."
        steps={[
          { formula: 'Center: sat, Context: the', explanation: 'Two positions to the left (t-2).' },
          { formula: 'Center: sat, Context: cat', explanation: 'One position to the left (t-1).' },
          { formula: 'Center: sat, Context: on', explanation: 'One position to the right (t+1).' },
          { formula: 'Center: sat, Context: the', explanation: 'Two positions to the right (t+2).' },
        ]}
        id="ex-training-pairs"
      />

      <PythonCode
        title="word2vec_gensim.py"
        id="code-word2vec"
        code={`from gensim.models import Word2Vec

# Sample corpus (list of tokenized sentences)
corpus = [
    ["the", "king", "rules", "the", "kingdom"],
    ["the", "queen", "rules", "with", "wisdom"],
    ["a", "man", "and", "a", "woman", "walked"],
    ["the", "prince", "is", "son", "of", "the", "king"],
    ["the", "princess", "is", "daughter", "of", "the", "queen"],
]

# Train Skip-gram model (sg=1); use sg=0 for CBOW
model = Word2Vec(
    sentences=corpus,
    vector_size=50,    # embedding dimension
    window=3,          # context window size
    min_count=1,       # include all words
    sg=1,              # 1 = Skip-gram, 0 = CBOW
    epochs=100,        # more epochs for tiny corpus
    seed=42,
)

# Access the learned embedding
king_vec = model.wv["king"]
print(f"Embedding shape: {king_vec.shape}")  # (50,)

# Find most similar words
similar = model.wv.most_similar("king", topn=3)
for word, score in similar:
    print(f"  {word}: {score:.3f}")

# Vector arithmetic (may need larger corpus for good results)
result = model.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1
)
print(f"\\nking - man + woman = {result[0][0]} ({result[0][1]:.3f})")`}
      />

      <WarningBlock
        title="Softmax Bottleneck"
        content="The full softmax requires computing a dot product with every word in the vocabulary for each training step, making it O(V) per example. For vocabularies of 100k+ words, this is prohibitively expensive. This motivates the negative sampling and hierarchical softmax approximations covered in the next section."
        id="warn-softmax"
      />

      <NoteBlock
        type="historical"
        title="Impact of Word2Vec"
        content="Word2Vec's 2013 release was a watershed moment for NLP. It demonstrated that simple, shallow models trained on large corpora could produce embeddings capturing complex semantic relationships. The resulting 'word vectors' became a standard component in virtually all NLP pipelines until the rise of contextual embeddings with ELMo and BERT."
        id="note-history"
      />
    </div>
  )
}
