import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function EncoderDecoderFamily() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Encoder-Decoder Models: T5 and BART</h1>
      <p className="text-lg text-gray-300">
        Encoder-decoder architectures combine a bidirectional encoder with an autoregressive
        decoder connected via cross-attention. This design naturally handles sequence-to-sequence
        tasks: the encoder processes the input and the decoder generates the output.
      </p>

      <DefinitionBlock
        title="Encoder-Decoder Architecture"
        definition="An encoder-decoder Transformer consists of: (1) an encoder with bidirectional self-attention producing representations $H_{\text{enc}} = \text{Encoder}(X)$, and (2) a decoder with causal self-attention plus cross-attention to $H_{\text{enc}}$. The decoder generates output: $P(y_t \mid y_{<t}, X) = \text{Decoder}(y_{<t}, H_{\text{enc}})_t$."
        notation="Cross-attention: $Q$ from decoder, $K, V$ from encoder. $\text{CrossAttn}(Q, K_{\text{enc}}, V_{\text{enc}}) = \text{softmax}(QK_{\text{enc}}^T / \sqrt{d})V_{\text{enc}}$."
        id="enc-dec-def"
      />

      <ExampleBlock
        title="T5: Text-to-Text Transfer Transformer"
        problem="Understand how T5 frames all NLP tasks as text-to-text problems."
        steps={[
          {
            formula: '\\text{Classification: } \\text{\"sentiment: This movie is great\"} \\rightarrow \\text{\"positive\"}',
            explanation: 'T5 generates the label as text. Unified framework for all tasks.'
          },
          {
            formula: '\\text{Translation: } \\text{\"translate English to French: Hello\"} \\rightarrow \\text{\"Bonjour\"}',
            explanation: 'Task prefix tells the model what operation to perform.'
          },
          {
            formula: '\\text{Summarization: } \\text{\"summarize: [long article]\"} \\rightarrow \\text{\"[summary]\"}',
            explanation: 'Natural fit for encoder-decoder: encode long input, decode short output.'
          },
          {
            formula: '\\text{Pretraining: span corruption with sentinel tokens}',
            explanation: 'T5 masks random spans and replaces with <extra_id_N>. Model generates the missing spans.'
          }
        ]}
        id="t5-tasks-example"
      />

      <NoteBlock
        type="historical"
        title="BART: Denoising Sequence-to-Sequence"
        content="BART (Lewis et al., 2019) uses a different pretraining strategy: it corrupts text with various noise functions (token masking, deletion, infilling, permutation, rotation) and trains the decoder to reconstruct the original. This denoising objective is more flexible than T5's span corruption and gives BART strong performance on both generation and comprehension tasks."
        id="bart-note"
      />

      <TheoremBlock
        title="Span Corruption Objective (T5)"
        statement="Given input $x$, sample spans to corrupt with total masked tokens $\approx 15\%$. Replace each span with a unique sentinel token $\langle s_i \rangle$. The target is the concatenation of sentinel tokens followed by the masked tokens: $y = \langle s_1 \rangle t_{1,1} \ldots t_{1,k_1} \langle s_2 \rangle t_{2,1} \ldots$. The loss is $\mathcal{L} = -\sum_t \log P(y_t \mid y_{<t}, \tilde{x})$."
        proof="Span corruption is more compute-efficient than token-level MLM because the target sequence is much shorter than the input (only the corrupted spans). T5 found mean span length 3 with 15% corruption rate optimal. The encoder processes the corrupted input bidirectionally, and the decoder generates only the missing spans."
        id="span-corruption-thm"
      />

      <PythonCode
        title="encoder_decoder_models.py"
        code={`from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer
)
import torch

# T5: Text-to-Text model
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# T5 for multiple tasks using text prefixes
tasks = [
    "translate English to German: The house is wonderful.",
    "summarize: State authorities dispatched emergency crews Tuesday to "
    "fight wildfires that have burned across the state.",
    "stsb sentence1: The cat sat on the mat. sentence2: A cat is sitting on a mat.",
]

print("=== T5 Multi-Task ===")
for task_input in tasks:
    input_ids = t5_tokenizer(task_input, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = t5_model.generate(input_ids, max_new_tokens=50)
    result = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input:  {task_input[:60]}...")
    print(f"Output: {result}\\n")

# T5 Span Corruption (pretraining objective)
def simulate_span_corruption(text, tokenizer, mask_ratio=0.15, mean_span=3):
    """Simulate T5's span corruption pretraining objective."""
    tokens = tokenizer.tokenize(text)
    n = len(tokens)
    n_mask = max(1, int(n * mask_ratio))

    # Select spans
    masked = set()
    sentinel_id = 0
    spans = []
    while len(masked) < n_mask and sentinel_id < 100:
        import random
        start = random.randint(0, n - 1)
        length = min(random.randint(1, mean_span * 2), n - start)
        span_tokens = []
        for i in range(start, start + length):
            if i not in masked:
                masked.add(i)
                span_tokens.append(tokens[i])
        if span_tokens:
            spans.append((sentinel_id, span_tokens))
            sentinel_id += 1

    # Build corrupted input and target
    corrupted = []
    in_span = False
    current_sentinel = 0
    for i, tok in enumerate(tokens):
        if i in masked:
            if not in_span:
                corrupted.append(f"<extra_id_{current_sentinel}>")
                current_sentinel += 1
                in_span = True
        else:
            in_span = False
            corrupted.append(tok)

    target = []
    for sid, stoks in spans:
        target.append(f"<extra_id_{sid}>")
        target.extend(stoks)

    return " ".join(corrupted), " ".join(target)

text = "The quick brown fox jumps over the lazy dog in the park"
corrupted, target = simulate_span_corruption(text, t5_tokenizer)
print("=== Span Corruption ===")
print(f"Original:  {text}")
print(f"Corrupted: {corrupted}")
print(f"Target:    {target}")

# BART for summarization
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
print(f"\\n=== Architecture Comparison ===")
print(f"T5-small:   enc={t5_model.config.num_layers}L, "
      f"dec={t5_model.config.num_decoder_layers}L, "
      f"d={t5_model.config.d_model}, "
      f"params={sum(p.numel() for p in t5_model.parameters())/1e6:.0f}M")
print(f"BART-base:  enc={bart_model.config.encoder_layers}L, "
      f"dec={bart_model.config.decoder_layers}L, "
      f"d={bart_model.config.d_model}, "
      f"params={sum(p.numel() for p in bart_model.parameters())/1e6:.0f}M")`}
        id="enc-dec-code"
      />

      <WarningBlock
        title="Encoder-Decoder Has Higher Inference Cost for Long Inputs"
        content="The encoder must process the full input before any decoding begins. For interactive use cases (chatbots), this creates higher time-to-first-token latency. Additionally, encoder-decoder models have roughly 2x the parameters of a decoder-only model with equivalent decoder capacity. This is why modern LLMs have largely shifted to decoder-only architectures despite the theoretical elegance of encoder-decoder designs."
        id="enc-dec-cost-warning"
      />
    </div>
  )
}
