import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function DecoderStack() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">The Transformer Decoder and Cross-Attention</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The decoder generates output tokens autoregressively, one at a time. Each decoder layer
        contains three sublayers: masked self-attention (causal), cross-attention to encoder
        outputs (in encoder-decoder models), and a feed-forward network.
      </p>

      <DefinitionBlock
        title="Decoder Layer (Encoder-Decoder)"
        definition="Each decoder layer applies: (1) causal self-attention on decoder tokens, (2) cross-attention where Q comes from the decoder and K, V come from the encoder output, and (3) a position-wise FFN. Each sublayer uses residual connections and layer normalization."
        id="def-decoder-layer"
      />

      <DefinitionBlock
        title="Cross-Attention"
        definition="Cross-attention computes $\text{Attention}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}})$ where queries are from the decoder layer and keys/values are from the encoder output. This allows the decoder to selectively read from the input representation."
        notation="Q = decoder hidden states, K = V = encoder output"
        id="def-cross-attention"
      />

      <h2 className="text-2xl font-semibold">Decoder-Only vs. Encoder-Decoder</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Decoder-only models (GPT series) drop the cross-attention sublayer entirely, using only
        causal self-attention and FFN. The input and output are concatenated into a single
        sequence with causal masking.
      </p>

      <ExampleBlock
        title="Three Sublayers in a Full Decoder"
        problem="Trace tensor flow through one decoder layer in a translation model."
        steps={[
          { formula: 'h^{\\text{dec}} \\in \\mathbb{R}^{B \\times m \\times d}', explanation: 'Decoder input: m target tokens generated so far.' },
          { formula: '\\text{CausalAttn}(h^{\\text{dec}})', explanation: 'Masked self-attention — each target token attends only to previous target tokens.' },
          { formula: '\\text{CrossAttn}(Q=h^{\\text{dec}}, K=h^{\\text{enc}}, V=h^{\\text{enc}})', explanation: 'Cross-attention lets each target token read from all encoder (source) positions.' },
          { formula: '\\text{FFN}(\\cdot)', explanation: 'Feed-forward network processes each position independently.' },
        ]}
        id="example-decoder-flow"
      />

      <PythonCode
        title="transformer_decoder.py"
        code={`import torch
import torch.nn as nn
import math

class CausalDecoderOnly(nn.Module):
    """GPT-style decoder-only transformer."""
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerEncoderLayer(  # No cross-attn needed
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_ff, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) * math.sqrt(self.d_model)
        x = self.dropout(x + self.pos_emb(positions))

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T,
            device=input_ids.device)

        x = self.decoder(x, mask=causal_mask, is_causal=True)
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

# Build a small GPT-like model
model = CausalDecoderOnly(
    vocab_size=50257, d_model=768, num_heads=12,
    d_ff=3072, num_layers=12
)

tokens = torch.randint(0, 50257, (2, 64))
logits = model(tokens)
print(f"Logits shape: {logits.shape}")  # [2, 64, 50257]
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Autoregressive generation (greedy)
@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=20):
    model.eval()
    ids = prompt_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(ids)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids

prompt = torch.randint(0, 50257, (1, 5))
generated = generate(model, prompt, max_new_tokens=10)
print(f"Generated sequence length: {generated.shape[1]}")`}
        id="code-decoder"
      />

      <NoteBlock
        type="intuition"
        title="Why Decoder-Only Models Dominate"
        content="Decoder-only models are simpler (no cross-attention), naturally handle any task as text-to-text completion, and scale more predictably. They can replicate encoder-decoder behavior by processing the input and output as a single concatenated sequence with appropriate attention masking."
        id="note-decoder-dominance"
      />

      <WarningBlock
        title="KV Cache for Efficient Generation"
        content="During autoregressive generation, recomputing attention over all previous tokens at each step is O(n²) total. The KV cache stores previously computed key and value vectors, reducing each step to O(n). Without caching, generation speed degrades dramatically for long sequences."
        id="warning-kv-cache"
      />

      <NoteBlock
        type="note"
        title="Cross-Attention Lives On"
        content="While pure LLMs are decoder-only, cross-attention remains crucial in multimodal models. Vision-language models like Flamingo use cross-attention to let the language decoder attend to visual features. Whisper uses cross-attention for the audio encoder-to-text decoder connection."
        id="note-cross-attn-lives"
      />
    </div>
  )
}
