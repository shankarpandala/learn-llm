import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function StatisticalVsNeural() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Statistical vs Neural Language Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Language modeling has undergone a dramatic transformation from count-based statistical
        methods to neural networks. Understanding this evolution explains why modern LLMs are
        so much more capable than their predecessors and what fundamental problems neural
        approaches solve.
      </p>

      <h2 className="text-2xl font-semibold">Statistical Language Models</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Statistical LMs estimate probabilities directly from corpus counts. N-gram models are
        the canonical example. Their key limitation is that they treat each word as an
        independent atomic symbol with no notion of similarity between words.
      </p>

      <DefinitionBlock
        title="The Sparsity Problem"
        definition="In n-gram models, the number of possible n-grams grows as $V^n$ where $V$ is vocabulary size. For a 50,000-word vocabulary, there are $2.5 \times 10^9$ possible bigrams, most of which will never appear in training data. This means most probability estimates are based on zero counts."
        id="def-sparsity"
      />

      <h2 className="text-2xl font-semibold">Neural Language Models</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Neural LMs represent words as dense vectors (embeddings) in a continuous space. Similar
        words have similar vectors, enabling the model to generalize: if it learns that
        "the cat sat on the mat" is likely, it can infer that "the dog sat on the rug" is
        also likely, because "cat" and "dog" have similar embeddings.
      </p>

      <DefinitionBlock
        title="Neural Language Model"
        definition="A neural language model parameterizes $P(w_t \mid w_{t-k}, \ldots, w_{t-1})$ using a neural network with parameters $\theta$. Words are embedded in $\mathbb{R}^d$, and the network learns a smooth function from context embeddings to a probability distribution over the vocabulary."
        notation="$P_\theta(w_t \mid w_{<t}) = \text{softmax}(f_\theta(\mathbf{e}_{w_{<t}}))$ where $\mathbf{e}$ are learned embeddings."
        id="def-neural-lm"
      />

      <ExampleBlock
        title="Generalization Through Embeddings"
        problem="A statistical model trained on 'The cat sits on the mat' assigns zero probability to 'The dog sits on the rug'. How does a neural model handle this?"
        steps={[
          { formula: 'Embeddings: cat ~ dog (both animals)', explanation: 'The model learns that "cat" and "dog" have similar vector representations.' },
          { formula: 'Embeddings: mat ~ rug (both floor coverings)', explanation: 'Similarly, "mat" and "rug" are nearby in embedding space.' },
          { formula: '$P(\\text{dog sits on the rug}) > 0$', explanation: 'Because the context embeddings are similar, the model assigns non-zero probability to the unseen but analogous sentence.' },
        ]}
        id="example-generalization"
      />

      <PythonCode
        title="lm_evolution_comparison.py"
        code={`import numpy as np

# Simulating the key difference: discrete vs continuous representations

# === Statistical LM: words are indices ===
vocab = {"the": 0, "cat": 1, "dog": 2, "sat": 3, "ran": 4, "mat": 5}

# Bigram counts (sparse matrix)
bigram_counts = np.zeros((len(vocab), len(vocab)))
bigram_counts[0, 1] = 5   # the -> cat
bigram_counts[0, 2] = 3   # the -> dog
bigram_counts[1, 3] = 4   # cat -> sat
bigram_counts[2, 4] = 2   # dog -> ran

# "cat -> ran" is ZERO! No generalization possible.
print("Statistical LM:")
print(f"  P(sat|cat) = {bigram_counts[1,3] / bigram_counts[1,:].sum():.3f}")
print(f"  P(ran|cat) = {bigram_counts[1,4] / max(bigram_counts[1,:].sum(), 1):.3f}")
print(f"  P(sat|dog) = {bigram_counts[2,3] / max(bigram_counts[2,:].sum(), 1):.3f}")

# === Neural LM: words are vectors ===
# Learned embeddings (2D for illustration)
embeddings = {
    "cat": np.array([0.8, 0.2]),   # Similar to dog
    "dog": np.array([0.7, 0.3]),   # Similar to cat
    "sat": np.array([-0.5, 0.9]),  # Similar to ran
    "ran": np.array([-0.4, 0.8]),  # Similar to sat
}

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\\nNeural LM (embedding similarities):")
print(f"  sim(cat, dog) = {cosine_sim(embeddings['cat'], embeddings['dog']):.3f}")
print(f"  sim(sat, ran) = {cosine_sim(embeddings['sat'], embeddings['ran']):.3f}")
print("  -> Because cat ~ dog and sat ~ ran,")
print("     P(ran|cat) > 0 even if never seen in training!")

# Evolution of LM architectures
print("\\nLM Architecture Evolution:")
timeline = [
    ("1980s", "N-gram models",           "Count-based, smoothing"),
    ("2003",  "Bengio's NNLM",           "First neural LM, feedforward"),
    ("2013",  "Word2Vec",                 "Efficient word embeddings"),
    ("2015",  "LSTM/GRU LMs",            "Recurrent, variable context"),
    ("2017",  "Transformer",             "Self-attention, parallel"),
    ("2018",  "GPT / BERT",              "Pre-training + fine-tuning"),
    ("2020",  "GPT-3 (175B)",            "In-context learning emerges"),
    ("2023",  "GPT-4, LLaMA, Claude",    "Reasoning, instruction-following"),
]
for year, model, desc in timeline:
    print(f"  {year}: {model:<25s} {desc}")`}
        id="code-lm-comparison"
      />

      <NoteBlock
        type="intuition"
        title="The Curse of Dimensionality, Solved"
        content="Bengio (2003) identified the core insight: n-gram models suffer from the curse of dimensionality because they need to see every possible context. Neural models map words to a continuous space where similar contexts produce similar predictions. A model that learns from 'the cat sat' can generalize to 'the kitten sat' because the representations are close."
        id="note-curse"
      />

      <h2 className="text-2xl font-semibold">Key Differences</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-gray-700 dark:text-gray-300">
          <thead>
            <tr className="border-b border-gray-300 dark:border-gray-600">
              <th className="px-4 py-2 text-left">Aspect</th>
              <th className="px-4 py-2 text-left">Statistical LM</th>
              <th className="px-4 py-2 text-left">Neural LM</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            <tr><td className="px-4 py-2 font-medium">Representation</td><td className="px-4 py-2">Discrete counts</td><td className="px-4 py-2">Dense embeddings</td></tr>
            <tr><td className="px-4 py-2 font-medium">Generalization</td><td className="px-4 py-2">No similarity notion</td><td className="px-4 py-2">Similar words share parameters</td></tr>
            <tr><td className="px-4 py-2 font-medium">Context</td><td className="px-4 py-2">Fixed n-gram window</td><td className="px-4 py-2">Variable (up to context length)</td></tr>
            <tr><td className="px-4 py-2 font-medium">Parameters</td><td className="px-4 py-2">Millions of counts</td><td className="px-4 py-2">Millions to trillions of weights</td></tr>
            <tr><td className="px-4 py-2 font-medium">Training</td><td className="px-4 py-2">Simple counting</td><td className="px-4 py-2">Gradient descent (GPU-intensive)</td></tr>
          </tbody>
        </table>
      </div>

      <NoteBlock
        type="historical"
        title="Bengio's 2003 Paper"
        content="'A Neural Probabilistic Language Model' by Bengio et al. (2003) is one of the most influential NLP papers. It introduced the idea of learning distributed word representations jointly with a language model. Though it took over a decade for the ideas to fully mature (through Word2Vec, LSTMs, and finally Transformers), this paper laid the conceptual foundation for all modern LLMs."
        id="note-bengio"
      />
    </div>
  )
}
