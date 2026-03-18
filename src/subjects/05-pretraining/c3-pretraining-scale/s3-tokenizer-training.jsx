import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function TokenizerTraining() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">BPE Training and Vocabulary Design</h1>
      <p className="text-lg text-gray-300">
        The tokenizer is the first component of any LLM and its design critically impacts model
        performance. Byte-Pair Encoding (BPE) is the dominant tokenization algorithm, learned
        from the pretraining corpus to create an efficient vocabulary of subword tokens.
      </p>

      <DefinitionBlock
        title="Byte-Pair Encoding (BPE)"
        definition="BPE starts with a base vocabulary of individual bytes (or characters) and iteratively merges the most frequent adjacent pair of tokens into a new token. After $k$ merge operations, the vocabulary has $|V_{\text{base}}| + k$ tokens. The merge order defines the tokenization rules."
        notation="At each step: find $(a, b) = \arg\max_{(x,y)} \text{freq}(xy)$, add $ab$ to vocabulary, replace all occurrences of $a \, b$ with $ab$."
        id="bpe-def"
      />

      <ExampleBlock
        title="BPE Merge Process"
        problem="Apply BPE merges to the corpus: 'low lower lowest low lower'."
        steps={[
          {
            formula: '\\text{Initial tokens: l o w, l o w e r, l o w e s t, l o w, l o w e r}',
            explanation: 'Start with character-level tokens. Count all adjacent pairs.'
          },
          {
            formula: '\\text{Most frequent pair: (l, o) = 5. Merge: \"lo\"}',
            explanation: 'Create new token "lo". Corpus becomes: lo w, lo w e r, lo w e s t, lo w, lo w e r.'
          },
          {
            formula: '\\text{Next pair: (lo, w) = 5. Merge: \"low\"}',
            explanation: 'Corpus becomes: low, low e r, low e s t, low, low e r.'
          },
          {
            formula: '\\text{Next pair: (e, r) = 2. Merge: \"er\"}',
            explanation: 'Corpus becomes: low, low er, low e s t, low, low er. Continue until vocab size reached.'
          }
        ]}
        id="bpe-merge-example"
      />

      <NoteBlock
        type="intuition"
        title="Why BPE Works Well"
        content="BPE naturally discovers meaningful subword units: common words become single tokens, rare words split into known subwords. This balances vocabulary size (typically 32K-128K) against sequence length. Larger vocabularies mean shorter sequences but more parameters in the embedding layer. The sweet spot depends on the data and languages covered."
        id="bpe-intuition"
      />

      <TheoremBlock
        title="Vocabulary Size Trade-off"
        statement="For a fixed compute budget $C$, vocabulary size $|V|$ creates a trade-off: increasing $|V|$ decreases average sequence length by factor $\rho(|V|)$ (compression ratio) but increases embedding parameters by $|V| \times d$. The optimal vocabulary size satisfies $\frac{\partial}{\partial |V|}[\text{training cost}(|V|)] = 0$."
        proof="Total compute is proportional to $C \propto 6ND$ where $N$ is parameters and $D$ is tokens. Larger $|V|$ means fewer tokens (shorter sequences) but more parameters in embeddings. Recent work suggests optimal $|V|$ scales as roughly $|V|^* \propto N^{0.5}$ for model size $N$."
        id="vocab-tradeoff-thm"
      />

      <PythonCode
        title="tokenizer_training.py"
        code={`from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers import normalizers, decoders
from transformers import AutoTokenizer

# Train a BPE tokenizer from scratch
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|padding|>", "<|unknown|>"],
    show_progress=True,
)

# Train on sample text (in practice, train on billions of tokens)
sample_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models learn representations from data.",
    "Natural language processing enables computers to understand text.",
    "Transformers use self-attention to process sequences in parallel.",
    "Large language models are pretrained on vast amounts of text data.",
] * 100  # Repeat for more frequent pairs

tokenizer.train_from_iterator(sample_texts, trainer=trainer)
print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

# Test tokenization
test = "Transformers revolutionized natural language processing"
encoded = tokenizer.encode(test)
print(f"Text: {test}")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Token count: {len(encoded.ids)}")

# Compare real tokenizers
print("\\n--- Comparing Real Tokenizers ---")
for name in ["bert-base-uncased", "gpt2", "meta-llama/Llama-2-7b-hf"]:
    try:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        ids = tok.encode(test)
        tokens = tok.tokenize(test)
        print(f"\\n{name}:")
        print(f"  Vocab size: {tok.vocab_size:,}")
        print(f"  Tokens ({len(tokens)}): {tokens[:10]}")
    except Exception as e:
        print(f"\\n{name}: (requires auth) vocab ~32K-128K")

# Compute compression ratio
import math
sample = "The Transformer architecture has become the backbone of modern NLP."
for name, tok in [("GPT-2", AutoTokenizer.from_pretrained("gpt2"))]:
    ids = tok.encode(sample)
    chars = len(sample)
    ratio = chars / len(ids)
    bits_per_char = math.log2(tok.vocab_size) / ratio
    print(f"\\n{name} compression:")
    print(f"  Characters: {chars}, Tokens: {len(ids)}")
    print(f"  Chars/token: {ratio:.1f}")
    print(f"  Bits/char (vocab): {bits_per_char:.1f}")`}
        id="tokenizer-code"
      />

      <WarningBlock
        title="Tokenizer Fertility Across Languages"
        content="BPE tokenizers trained primarily on English text produce far more tokens for non-English text (sometimes 3-10x more). This means the model sees fewer words per context window in other languages, effectively reducing its capacity for multilingual understanding. Modern tokenizers like those in LLaMA-3 use larger vocabularies (128K) and train on more balanced multilingual data to reduce this disparity."
        id="fertility-warning"
      />
    </div>
  )
}
