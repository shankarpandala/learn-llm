import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function AttentionIsAllYouNeed() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">"Attention Is All You Need" — Paper Walkthrough</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Published by Vaswani et al. at NeurIPS 2017, this paper introduced the transformer
        architecture, replacing recurrence entirely with self-attention. It remains one of
        the most influential papers in deep learning, with every modern LLM tracing its
        lineage to this work.
      </p>

      <NoteBlock
        type="historical"
        title="Context: The State of NLP in 2017"
        content="Before transformers, sequence-to-sequence models relied on LSTMs and GRUs with attention (Bahdanau 2014, Luong 2015). These models processed tokens sequentially, preventing parallelization and struggling with long-range dependencies. The transformer eliminated recurrence entirely, enabling massive parallelism during training."
        id="note-context"
      />

      <h2 className="text-2xl font-semibold">Key Contributions</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The paper introduced several ideas that became standard: scaled dot-product attention,
        multi-head attention, positional encoding, and the specific combination of residual
        connections with layer normalization in an encoder-decoder structure.
      </p>

      <DefinitionBlock
        title="The Transformer (Vaswani et al., 2017)"
        definition="A sequence-to-sequence model using stacked self-attention and feed-forward layers. The core attention formula: $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ with multi-head extension: $\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$."
        notation="Original config: d_model=512, h=8, d_ff=2048, N=6 layers (encoder & decoder)"
        id="def-original-transformer"
      />

      <ExampleBlock
        title="Original Paper Configuration"
        problem="Calculate key numbers for the base and big transformer models from the paper."
        steps={[
          { formula: '\\text{Base: } d=512, h=8, d_k=64, d_{ff}=2048, N=6', explanation: 'Base model: 65M parameters, trained on 8 P100 GPUs for 12 hours.' },
          { formula: '\\text{Big: } d=1024, h=16, d_k=64, d_{ff}=4096, N=6', explanation: 'Big model: 213M parameters, trained on 8 P100 GPUs for 3.5 days.' },
          { formula: '\\text{WMT14 EN-DE: 28.4 BLEU (big)}', explanation: 'State-of-the-art machine translation, surpassing all existing models.' },
          { formula: '\\text{Training cost: } \\$150{-}\\$1500 \\text{ (2017 prices)}', explanation: 'Remarkably cheap by modern standards — GPT-4 cost ~$100M+.' },
        ]}
        id="example-paper-config"
      />

      <PythonCode
        title="original_transformer_config.py"
        code={`import torch
import torch.nn as nn

def count_transformer_params(d_model, num_heads, d_ff, num_layers,
                              src_vocab, tgt_vocab):
    """Estimate parameter count for the original transformer."""
    d_k = d_model // num_heads

    # Per encoder layer
    attn_params = 4 * d_model * d_model  # Q, K, V, O projections
    ffn_params = 2 * d_model * d_ff + d_model + d_ff  # 2 linears + biases
    ln_params = 4 * d_model  # 2 layer norms * (gamma + beta)
    enc_layer = attn_params + ffn_params + ln_params

    # Per decoder layer (extra cross-attention)
    dec_layer = enc_layer + attn_params + 2 * d_model  # + cross-attn + LN

    # Embeddings
    src_emb = src_vocab * d_model
    tgt_emb = tgt_vocab * d_model

    total = (num_layers * enc_layer +
             num_layers * dec_layer +
             src_emb + tgt_emb)
    return total

# Base model
base = count_transformer_params(
    d_model=512, num_heads=8, d_ff=2048,
    num_layers=6, src_vocab=37000, tgt_vocab=37000
)
print(f"Base model: {base:,} params ({base/1e6:.0f}M)")

# Big model
big = count_transformer_params(
    d_model=1024, num_heads=16, d_ff=4096,
    num_layers=6, src_vocab=37000, tgt_vocab=37000
)
print(f"Big model:  {big:,} params ({big/1e6:.0f}M)")

# Build the actual base model using PyTorch
model = nn.Transformer(
    d_model=512, nhead=8, num_encoder_layers=6,
    num_decoder_layers=6, dim_feedforward=2048,
    dropout=0.1, batch_first=True
)
actual_params = sum(p.numel() for p in model.parameters())
print(f"\\nPyTorch base model: {actual_params:,} params")

# The paper's key insight: training speed
print("\\nTraining parallelism comparison:")
print("  LSTM: O(n) sequential steps — cannot parallelize across time")
print("  Transformer: O(1) sequential steps — full sequence in parallel")
print("  Self-attention: O(n^2) compute but all in parallel")`}
        id="code-paper-config"
      />

      <h2 className="text-2xl font-semibold">Paper Structure Summary</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Section 1 motivates replacing recurrence. Section 2 reviews prior work. Section 3
        presents the architecture (the famous Figure 1). Section 4 explains why self-attention
        is preferable to recurrence and convolutions. Sections 5-6 show training details and
        results on WMT translation benchmarks.
      </p>

      <NoteBlock
        type="tip"
        title="The Most Important Figure in Deep Learning"
        content="Figure 1 of the paper (the architecture diagram) is arguably the most reproduced figure in AI research. When reading it, note three types of attention: encoder self-attention (bidirectional), decoder self-attention (causal), and encoder-decoder cross-attention. These three patterns cover nearly all attention variants used today."
        id="note-figure1"
      />

      <WarningBlock
        title="What the Paper Got Wrong (or Didn't Predict)"
        content="The paper focused on machine translation and encoder-decoder models. It did not predict that (1) decoder-only models would dominate, (2) scale would matter more than architecture, (3) the same architecture would work for vision, audio, and code, or (4) emergent abilities would appear at scale. The architecture was far more general than the authors realized."
        id="warning-predictions"
      />

      <NoteBlock
        type="note"
        title="Citation Impact"
        content="'Attention Is All You Need' has over 130,000 citations, making it one of the most cited papers in computer science history. The 8 authors went on to found or join leading AI companies (Google Brain/DeepMind, Cohere, Character.AI, Adept, Essential AI, Near), demonstrating the paper's outsized impact on both research and industry."
        id="note-citations"
      />
    </div>
  )
}
