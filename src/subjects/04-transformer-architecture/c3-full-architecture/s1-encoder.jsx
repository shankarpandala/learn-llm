import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function EncoderStack() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">The Transformer Encoder Stack</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The encoder processes an input sequence with full bidirectional attention, producing
        contextualized representations where every token can attend to every other token.
        Encoder-only models like BERT are built by stacking N identical encoder layers.
      </p>

      <DefinitionBlock
        title="Encoder Layer"
        definition="Each encoder layer consists of two sublayers: (1) multi-head self-attention, and (2) a position-wise feed-forward network. Each sublayer is wrapped with a residual connection and layer normalization: $\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$."
        notation="N = number of stacked layers (6 in original, 12 in BERT-base, 24 in BERT-large)"
        id="def-encoder-layer"
      />

      <h2 className="text-2xl font-semibold">Encoder Architecture</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The encoder takes token embeddings plus positional encodings as input and passes them
        through N identical layers. The output is a sequence of contextualized vectors, one per
        input token, that encode rich bidirectional context.
      </p>
      <BlockMath math="h^{(l)} = \text{EncoderLayer}(h^{(l-1)}), \quad l = 1, \ldots, N" />

      <ExampleBlock
        title="Information Flow in a 6-Layer Encoder"
        problem="Trace how the word 'bank' gets contextualized in 'The bank of the river was muddy'."
        steps={[
          { formula: 'h^{(0)}_{\\text{bank}} = e_{\\text{bank}} + PE_1', explanation: 'Initial embedding is context-free — "bank" could mean financial or river.' },
          { formula: 'h^{(1)}: \\text{bank attends to river, muddy}', explanation: 'Layer 1 attention starts incorporating nearby context.' },
          { formula: 'h^{(3)}: \\text{captures "bank of the river" phrase}', explanation: 'Middle layers build phrase-level representations.' },
          { formula: 'h^{(6)}: \\text{fully disambiguated to riverbank}', explanation: 'Final representation encodes complete sentence context.' },
        ]}
        id="example-contextualization"
      />

      <PythonCode
        title="transformer_encoder.py"
        code={`import torch
import torch.nn as nn
import math

class TransformerEncoder(nn.Module):
    """Full transformer encoder stack."""
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_emb(positions)
        x = self.dropout(x)

        # Convert padding mask: True = ignore
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.final_norm(x)

# Build a BERT-base-sized encoder
encoder = TransformerEncoder(
    vocab_size=30522, d_model=768, num_heads=12,
    d_ff=3072, num_layers=12, max_len=512
)

# Forward pass
input_ids = torch.randint(0, 30522, (2, 128))
mask = torch.ones(2, 128)
mask[0, 100:] = 0  # First sequence padded after position 100

output = encoder(input_ids, attention_mask=mask)
print(f"Encoder output: {output.shape}")  # [2, 128, 768]

# Count parameters
total = sum(p.numel() for p in encoder.parameters())
print(f"Total parameters: {total:,}")  # ~86M (BERT-base scale)`}
        id="code-encoder"
      />

      <NoteBlock
        type="note"
        title="Bidirectional Context"
        content="The key property of the encoder is bidirectional attention. Every token sees every other token, including future tokens. This makes encoders ideal for understanding tasks (classification, NER, similarity) but unsuitable for text generation — they cannot be used autoregressively without modification."
        id="note-bidirectional"
      />

      <WarningBlock
        title="Encoder-Only Models Are Not Generative"
        content="Although BERT-style encoders can be used for masked language modeling (predicting [MASK] tokens), they cannot generate text left-to-right. For generation, you need either a decoder (GPT) or an encoder-decoder (T5). Encoder-only models excel at classification, retrieval, and extraction tasks."
        id="warning-not-generative"
      />

      <NoteBlock
        type="historical"
        title="Notable Encoder-Only Models"
        content="BERT (Devlin et al., 2018) popularized encoder-only transformers. RoBERTa (2019) improved training. DeBERTa (2021) added disentangled attention. Modern embedding models like E5, GTE, and BGE are all encoder-based, showing encoders remain vital for retrieval and understanding tasks."
        id="note-history"
      />
    </div>
  )
}
