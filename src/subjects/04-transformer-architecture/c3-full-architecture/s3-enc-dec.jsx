import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function EncoderDecoder() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">The Full Encoder-Decoder Architecture</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The original transformer is an encoder-decoder model designed for sequence-to-sequence
        tasks. The encoder processes the input bidirectionally, and the decoder generates the
        output autoregressively while attending to the encoder's representations via cross-attention.
      </p>

      <DefinitionBlock
        title="Encoder-Decoder Transformer"
        definition="The full architecture maps an input sequence $(x_1, \ldots, x_n)$ to an output sequence $(y_1, \ldots, y_m)$ via: $h = \text{Encoder}(x_1, \ldots, x_n)$, then $P(y_t | y_{<t}, h) = \text{Decoder}(y_{<t}, h)$ for each output step $t$."
        notation="h = encoder hidden states, y_{<t} = previously generated tokens"
        id="def-enc-dec"
      />

      <h2 className="text-2xl font-semibold">Information Flow</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The encoder runs once on the full input. The decoder then runs autoregressively, generating
        one token at a time. At each step, the decoder's cross-attention layers read from the
        encoder output, allowing it to selectively focus on relevant parts of the input.
      </p>

      <ExampleBlock
        title="Machine Translation: English to French"
        problem="Show the encoder-decoder flow for translating 'The cat sat' to 'Le chat assis'."
        steps={[
          { formula: '\\text{Encoder}(\\text{"The", "cat", "sat"}) \\rightarrow h \\in \\mathbb{R}^{3 \\times d}', explanation: 'Encoder produces contextualized representations for all 3 source tokens.' },
          { formula: '\\text{Decoder step 1: } P(y_1 | \\text{<bos>}, h) \\rightarrow \\text{"Le"}', explanation: 'Decoder generates first token, attending to all encoder positions.' },
          { formula: '\\text{Decoder step 2: } P(y_2 | \\text{"Le"}, h) \\rightarrow \\text{"chat"}', explanation: 'Cross-attention focuses on "cat" in the encoder output.' },
          { formula: '\\text{Decoder step 3: } P(y_3 | \\text{"Le", "chat"}, h) \\rightarrow \\text{"assis"}', explanation: 'Cross-attention focuses on "sat" — note the word alignment is learned.' },
        ]}
        id="example-translation"
      />

      <PythonCode
        title="encoder_decoder_transformer.py"
        code={`import torch
import torch.nn as nn
import math

class EncoderDecoderTransformer(nn.Module):
    """Full encoder-decoder transformer (T5-style)."""
    def __init__(self, src_vocab, tgt_vocab, d_model=512, num_heads=8,
                 d_ff=2048, num_layers=6, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_ff, dropout=dropout,
            activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.output_proj = nn.Linear(d_model, tgt_vocab)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src_ids, src_mask=None):
        T = src_ids.size(1)
        pos = torch.arange(T, device=src_ids.device).unsqueeze(0)
        x = self.dropout(self.src_emb(src_ids) * math.sqrt(self.d_model)
                         + self.pos_emb(pos))
        padding_mask = (src_mask == 0) if src_mask is not None else None
        return self.transformer.encoder(x, src_key_padding_mask=padding_mask)

    def decode(self, tgt_ids, memory, tgt_mask=None, memory_mask=None):
        T = tgt_ids.size(1)
        pos = torch.arange(T, device=tgt_ids.device).unsqueeze(0)
        x = self.dropout(self.tgt_emb(tgt_ids) * math.sqrt(self.d_model)
                         + self.pos_emb(pos))
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        mem_pad = (memory_mask == 0) if memory_mask is not None else None
        out = self.transformer.decoder(x, memory, tgt_mask=causal,
                                       memory_key_padding_mask=mem_pad)
        return self.output_proj(out)

    def forward(self, src_ids, tgt_ids, src_mask=None):
        memory = self.encode(src_ids, src_mask)
        logits = self.decode(tgt_ids, memory, memory_mask=src_mask)
        return logits

# Build and test
model = EncoderDecoderTransformer(src_vocab=32000, tgt_vocab=32000)
src = torch.randint(0, 32000, (2, 20))
tgt = torch.randint(0, 32000, (2, 15))
logits = model(src, tgt)
print(f"Output logits: {logits.shape}")  # [2, 15, 32000]
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")`}
        id="code-enc-dec"
      />

      <NoteBlock
        type="note"
        title="Encoder-Decoder Use Cases"
        content="Encoder-decoder models excel at tasks with distinct input/output: translation, summarization, and question answering. T5 frames all NLP tasks as text-to-text, using an encoder-decoder. BART uses a similar architecture for denoising pretraining. Whisper uses encoder-decoder for speech-to-text."
        id="note-use-cases"
      />

      <WarningBlock
        title="Encoder-Decoder vs. Decoder-Only Scaling"
        content="Encoder-decoder models have roughly 2x the parameters of a decoder-only model with the same layer size, since both encoder and decoder have separate stacks. When comparing, a 6B encoder-decoder (like Flan-T5-XL) has comparable parameter count to a 6B decoder-only model but splits capacity between understanding and generation."
        id="warning-param-comparison"
      />

      <NoteBlock
        type="historical"
        title="The Rise and Decline of Encoder-Decoder"
        content="The original transformer (2017) was encoder-decoder. T5 (2019) and BART (2019) refined the approach. However, GPT-3 (2020) demonstrated that decoder-only models with sufficient scale could match or surpass encoder-decoder models on most tasks, leading to the current dominance of decoder-only architectures in large-scale LLMs."
        id="note-history"
      />
    </div>
  )
}
