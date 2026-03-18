import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function ELMo() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">ELMo & Contextual Embeddings</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        ELMo (Embeddings from Language Models, Peters et al., 2018) marked a paradigm shift from
        static word embeddings to contextual representations. Unlike Word2Vec or GloVe, which
        assign a single vector per word type, ELMo produces different vectors for each word token
        depending on its surrounding context. The word "bank" gets different representations in
        "river bank" versus "bank account."
      </p>

      <DefinitionBlock
        title="Contextual Word Embedding"
        definition="A contextual embedding is a function $f: (w, C) \\to \\mathbb{R}^d$ that maps a word $w$ together with its context $C$ (the surrounding sentence or passage) to a vector. Unlike static embeddings where $f(w) = \\mathbf{v}_w$ is fixed, contextual embeddings produce different vectors for the same word in different contexts."
        id="def-contextual"
      />

      <DefinitionBlock
        title="ELMo Architecture"
        definition="ELMo uses a 2-layer bidirectional LSTM trained as a language model. The forward LSTM models $P(w_t | w_1, \\dots, w_{t-1})$ and the backward LSTM models $P(w_t | w_{t+1}, \\dots, w_T)$. The final ELMo representation for word $t$ is a task-specific weighted combination of all layers: $\\text{ELMo}_t^{\\text{task}} = \\gamma^{\\text{task}} \\sum_{j=0}^{L} s_j^{\\text{task}} \\mathbf{h}_{t,j}$."
        notation="$\\mathbf{h}_{t,0}$ = character CNN layer; $\\mathbf{h}_{t,1}, \\mathbf{h}_{t,2}$ = biLSTM layers; $s_j$ = softmax-normalized weights; $\\gamma$ = scalar."
        id="def-elmo"
      />

      <p className="text-gray-700 dark:text-gray-300">
        The biLM objective maximizes the joint log-likelihood of both directions:
      </p>
      <BlockMath math={String.raw`\mathcal{L} = \sum_{t=1}^{T} \left( \log P(w_t \mid w_1, \dots, w_{t-1};\, \Theta_{\text{fwd}}) + \log P(w_t \mid w_{t+1}, \dots, w_T;\, \Theta_{\text{bwd}}) \right)`} />

      <NoteBlock
        type="intuition"
        title="What Each Layer Captures"
        content="Peters et al. showed that different biLSTM layers capture different types of information. Layer 0 (character CNN) captures morphology and character patterns. Layer 1 captures syntax -- POS tagging benefits most from this layer. Layer 2 captures semantics -- word sense disambiguation benefits most from this layer. The learned task-specific weights allow downstream models to mix these information types optimally."
        id="note-layers"
      />

      <ExampleBlock
        title="Context-Dependent Representations"
        problem="Show how ELMo produces different vectors for the word 'bank' in two sentences: (A) 'He sat by the river bank' and (B) 'She went to the bank to deposit money'."
        steps={[
          { formula: 'Sentence A context: river, sat, by', explanation: 'The biLSTM processes left-to-right and right-to-left, incorporating the nature-related context.' },
          { formula: 'Sentence B context: deposit, money, went', explanation: 'The financial context words push the hidden states in a different direction.' },
          { formula: '$\\mathbf{h}^A_{\\text{bank}} \\neq \\mathbf{h}^B_{\\text{bank}}$', explanation: 'The resulting ELMo vectors are different despite being the same word, resolving the ambiguity.' },
        ]}
        id="ex-context"
      />

      <PythonCode
        title="elmo_conceptual.py"
        id="code-elmo"
        code={`import numpy as np

# Conceptual demonstration of ELMo layer combination
# (Using allennlp for real ELMo is heavyweight; we illustrate the key idea)

np.random.seed(42)
d = 256  # hidden dimension

# Simulate ELMo layer outputs for "bank" in two contexts
# Layer 0: character-level (similar for same word)
h0_river = np.random.randn(d)
h0_money = h0_river + np.random.randn(d) * 0.05  # nearly identical

# Layer 1: syntactic (somewhat different)
h1_river = np.random.randn(d) * 0.8
h1_money = np.random.randn(d) * 0.8  # different syntax context

# Layer 2: semantic (very different for polysemous words)
h2_river = np.random.randn(d) * 0.6
h2_money = np.random.randn(d) * 0.6  # very different meaning

# Task-specific weights (learned during fine-tuning)
# For sentiment analysis, semantics matters most
sentiment_weights = np.array([0.1, 0.2, 0.7])  # favor layer 2
sentiment_weights = np.exp(sentiment_weights) / np.exp(sentiment_weights).sum()
gamma = 1.2  # task scalar

def elmo_combine(h0, h1, h2, weights, gamma):
    """Compute task-specific ELMo representation."""
    return gamma * (weights[0] * h0 + weights[1] * h1 + weights[2] * h2)

elmo_river = elmo_combine(h0_river, h1_river, h2_river, sentiment_weights, gamma)
elmo_money = elmo_combine(h0_money, h1_money, h2_money, sentiment_weights, gamma)

cos = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("ELMo representations for 'bank':")
print(f"  Cosine similarity (river vs money context): {cos(elmo_river, elmo_money):.3f}")
print(f"  Layer 0 similarity (char-level): {cos(h0_river, h0_money):.3f}")
print(f"  Layer 2 similarity (semantic):   {cos(h2_river, h2_money):.3f}")

# For POS tagging, syntax matters more
pos_weights = np.array([0.1, 0.7, 0.2])  # favor layer 1
pos_weights = np.exp(pos_weights) / np.exp(pos_weights).sum()

elmo_river_pos = elmo_combine(h0_river, h1_river, h2_river, pos_weights, gamma)
elmo_money_pos = elmo_combine(h0_money, h1_money, h2_money, pos_weights, gamma)

print(f"\\nWith POS-tagging weights:")
print(f"  Cosine similarity: {cos(elmo_river_pos, elmo_money_pos):.3f}")
print("  (Higher because both are nouns -- syntax is more similar than semantics)")`}
      />

      <WarningBlock
        title="ELMo is Not an Encoder"
        content="ELMo is a feature extractor, not a fine-tunable encoder. The biLSTM weights are frozen after pre-training; only the layer mixing weights and scalar are learned per task. This makes it fast to adapt but limits its expressiveness compared to fully fine-tuned models like BERT."
        id="warn-frozen"
      />

      <NoteBlock
        type="historical"
        title="From ELMo to BERT"
        content="ELMo's success in 2018 (improving state-of-the-art on 6 NLP benchmarks by simply concatenating ELMo vectors to existing model inputs) demonstrated the power of pre-trained contextual representations. This directly inspired BERT (2018), which replaced LSTMs with Transformers and introduced full fine-tuning, further advancing the paradigm."
        id="note-to-bert"
      />
    </div>
  )
}
