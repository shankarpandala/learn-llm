import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Tokenization() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Tokenization: Word, Subword, and Character</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Tokenization is the first step in any NLP pipeline. It converts raw text into discrete
        units (tokens) that a model can process. The choice of tokenization strategy profoundly
        affects vocabulary size, the ability to handle rare words, and overall model performance.
      </p>

      <DefinitionBlock
        title="Tokenization"
        definition="Tokenization is the process of splitting a sequence of text into smaller units called tokens. Tokens may be words, subwords, or individual characters depending on the strategy used."
        id="def-tokenization"
      />

      <h2 className="text-2xl font-semibold">Word Tokenization</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The simplest approach splits text on whitespace and punctuation. While intuitive, word-level
        tokenization creates enormous vocabularies and cannot handle out-of-vocabulary (OOV) words.
      </p>

      <PythonCode
        title="word_tokenization.py"
        code={`# Simple word tokenization approaches
text = "The transformer architecture hasn't been surpassed yet."

# Naive whitespace split
tokens_naive = text.split()
print("Whitespace:", tokens_naive)
# ['The', 'transformer', 'architecture', "hasn't", 'been', 'surpassed', 'yet.']

# NLTK word tokenizer handles punctuation and contractions
import nltk
tokens_nltk = nltk.word_tokenize(text)
print("NLTK:", tokens_nltk)
# ['The', 'transformer', 'architecture', 'has', "n't", 'been', 'surpassed', 'yet', '.']

# Problem: OOV words in a fixed vocabulary
vocab = set(tokens_nltk)
new_text = "Transformers revolutionized NLP"
new_tokens = nltk.word_tokenize(new_text)
oov = [t for t in new_tokens if t not in vocab]
print(f"Out-of-vocabulary: {oov}")  # All tokens are OOV`}
        id="code-word-tokenization"
      />

      <h2 className="text-2xl font-semibold">Byte Pair Encoding (BPE)</h2>
      <p className="text-gray-700 dark:text-gray-300">
        BPE starts with individual characters and iteratively merges the most frequent adjacent pair
        of tokens. This produces a subword vocabulary that balances between character-level and
        word-level granularity. GPT-2, GPT-3, and GPT-4 all use variants of BPE.
      </p>

      <ExampleBlock
        title="BPE Merge Example"
        problem="Given the corpus with word frequencies: {'low': 5, 'lower': 2, 'newest': 6, 'widest': 3}, show the first BPE merge."
        steps={[
          { formula: 'Initial vocabulary: {l, o, w, e, r, n, s, t, i, d}', explanation: 'Start with all individual characters.' },
          { formula: 'Count pairs: (e, s) appears 9 times (6+3)', explanation: 'The pair "e"+"s" is the most frequent across "newest" and "widest".' },
          { formula: 'Merge: es -> new token "es"', explanation: 'Replace all occurrences of "e" followed by "s" with the merged token "es".' },
          { formula: 'Next: (es, t) appears 9 times', explanation: 'Continue merging the most frequent pair in the updated corpus.' },
        ]}
        id="example-bpe"
      />

      <PythonCode
        title="bpe_with_tiktoken.py"
        code={`# Using tiktoken (OpenAI's fast BPE tokenizer)
import tiktoken

# GPT-4 uses cl100k_base encoding
enc = tiktoken.get_encoding("cl100k_base")

text = "Tokenization is fundamental to LLMs."
tokens = enc.encode(text)
print(f"Token IDs: {tokens}")
print(f"Number of tokens: {len(tokens)}")

# Decode individual tokens to see subwords
for tid in tokens:
    print(f"  {tid} -> '{enc.decode([tid])}'")

# Compare token counts for different texts
examples = [
    "Hello world",
    "antidisestablishmentarianism",  # Long word gets split
    "こんにちは世界",  # Non-English text
]
for ex in examples:
    toks = enc.encode(ex)
    print(f"'{ex}' -> {len(toks)} tokens")`}
        id="code-bpe"
      />

      <h2 className="text-2xl font-semibold">WordPiece and SentencePiece</h2>
      <p className="text-gray-700 dark:text-gray-300">
        WordPiece (used by BERT) is similar to BPE but selects merges based on likelihood
        rather than frequency. SentencePiece treats the input as a raw byte stream, making it
        language-agnostic and able to handle any Unicode text without pre-tokenization.
      </p>

      <NoteBlock
        type="intuition"
        title="Why Subword Tokenization Works"
        content="Subword methods capture morphological structure. The word 'unhappiness' might be split into 'un', 'happiness' or 'un', 'happy', 'ness'. This lets models generalize: if they know 'happy' and see 'un' + 'happy', they can infer meaning compositionally."
        id="note-subword-intuition"
      />

      <WarningBlock
        title="Tokenization Affects Everything Downstream"
        content="The choice of tokenizer determines the model's effective context length. A sentence that takes 10 tokens with one tokenizer might take 20 with another. This is why token counts, not word counts, determine cost and context limits in LLM APIs."
        id="warning-tokenization"
      />

      <NoteBlock
        type="historical"
        title="Evolution of Tokenization"
        content="Early NLP used word-level tokens (Word2Vec, 2013). BPE was adapted from data compression to NLP by Sennrich et al. (2016). Google introduced WordPiece for BERT (2018). SentencePiece (Kudo & Richardson, 2018) unified subword tokenization into a language-independent framework used by T5, LLaMA, and many modern models."
        id="note-history"
      />
    </div>
  )
}
