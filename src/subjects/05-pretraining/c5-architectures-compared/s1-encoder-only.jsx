import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function EncoderOnly() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Encoder-Only Models: The BERT Family</h1>
      <p className="text-lg text-gray-300">
        Encoder-only architectures use bidirectional self-attention where every token attends
        to every other token. They excel at understanding tasks (classification, NER, question
        answering) but cannot generate text autoregressively.
      </p>

      <DefinitionBlock
        title="Encoder-Only Architecture"
        definition="An encoder-only Transformer processes an input sequence $x = (x_1, \ldots, x_n)$ through $L$ layers of bidirectional self-attention and feed-forward networks. The attention mask is fully visible: $M_{ij} = 1$ for all $i, j$. Output is a contextualized representation $h_i \in \mathbb{R}^d$ for each position."
        notation="$H = \text{TransformerEncoder}(X + P)$ where $H \in \mathbb{R}^{n \times d}$ and attention is unrestricted."
        id="encoder-only-def"
      />

      <ExampleBlock
        title="BERT Family Evolution"
        problem="Trace the evolution of encoder-only models from BERT to modern variants."
        steps={[
          {
            formula: '\\text{BERT (2018): 110M/340M, MLM+NSP, WordPiece}',
            explanation: 'Original bidirectional pretraining. Revolutionized NLP benchmarks.'
          },
          {
            formula: '\\text{RoBERTa (2019): Same arch, no NSP, dynamic masking, 10x data}',
            explanation: 'Showed BERT was undertrained. Longer training, bigger batches, more data.'
          },
          {
            formula: '\\text{ALBERT (2019): Factorized embeddings, cross-layer sharing}',
            explanation: 'Reduced parameters 18x. Replaced NSP with Sentence Order Prediction.'
          },
          {
            formula: '\\text{DeBERTa (2020): Disentangled attention, enhanced decoding}',
            explanation: 'Separate content and position embeddings. SOTA on SuperGLUE. DeBERTa-v3 uses ELECTRA-style training.'
          },
          {
            formula: '\\text{ELECTRA (2020): Replaced token detection, not MLM}',
            explanation: 'Generator creates corrupted tokens, discriminator detects them. Trains on ALL tokens, not just 15%.'
          }
        ]}
        id="bert-family-example"
      />

      <NoteBlock
        type="intuition"
        title="Why Bidirectional Attention Helps Understanding"
        content="Consider disambiguating 'bank' in 'I went to the bank to deposit money' vs 'I sat by the river bank'. A left-to-right model processing 'bank' hasn't seen 'deposit' or 'river' yet. A bidirectional model sees the full context simultaneously, making disambiguation trivial. This is why encoder-only models dominate classification and extraction tasks."
        id="bidirectional-intuition"
      />

      <PythonCode
        title="encoder_only_models.py"
        code={`from transformers import (
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, pipeline
)
import torch

# Compare BERT family models
models_info = {
    "bert-base-uncased":          {"params": "110M", "vocab": 30522},
    "roberta-base":               {"params": "125M", "vocab": 50265},
    "microsoft/deberta-v3-base":  {"params": "184M", "vocab": 128100},
}

text = "The quick brown fox jumps over the lazy dog."
for name, info in models_info.items():
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        inputs = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
        hidden = out.last_hidden_state
        print(f"{name}:")
        print(f"  Params: {info['params']}, Vocab: {info['vocab']}")
        print(f"  Hidden shape: {hidden.shape}")
        print(f"  CLS repr norm: {hidden[0, 0].norm():.2f}")
    except Exception as e:
        print(f"{name}: {e}")

# Task: Sentiment classification with BERT
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
results = classifier([
    "This movie was absolutely fantastic!",
    "I was disappointed by the poor acting.",
    "The weather is nice today.",
])
for r in results:
    print(f"  {r['label']}: {r['score']:.4f}")

# Task: Named Entity Recognition
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
entities = ner("Elon Musk founded SpaceX in Hawthorne, California.")
for e in entities:
    print(f"  {e['entity_group']}: {e['word']} ({e['score']:.3f})")

# ELECTRA: Replaced Token Detection
from transformers import ElectraForPreTraining, ElectraTokenizer

tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
model = ElectraForPreTraining.from_pretrained("google/electra-small-discriminator")

sentence = "The chef cooked a delicious meal"
fake = "The chef cooked a electric meal"  # 'electric' is a replaced token
inputs = tokenizer(fake, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predictions = (logits.squeeze() > 0).int()
tokens = tokenizer.tokenize(fake)
print("\\nELECTRA replaced token detection:")
for tok, pred in zip(tokens, predictions[1:-1]):
    label = "REPLACED" if pred == 1 else "original"
    print(f"  {tok:>12s}: {label}")`}
        id="encoder-only-code"
      />

      <WarningBlock
        title="Encoder-Only Models Are Declining in Popularity"
        content="Since 2022, decoder-only models have dominated both understanding and generation tasks. Models like GPT-4, LLaMA, and Claude use decoder-only architectures yet match or exceed encoder-only models on classification and NLU benchmarks. Encoder-only models remain valuable for efficient embedding, retrieval, and token-level tasks where bidirectional context and smaller model size are advantages."
        id="encoder-decline-warning"
      />
    </div>
  )
}
