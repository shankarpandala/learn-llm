import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function BahdanauAttention() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Bahdanau Attention</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Bahdanau attention (2015) was the breakthrough that eliminated the information bottleneck
        in seq2seq models. Instead of compressing the entire input into a single vector, attention
        allows the decoder to look back at all encoder hidden states and dynamically focus on
        the most relevant parts for each output token. This is the direct ancestor of the
        attention mechanism in transformers.
      </p>

      <DefinitionBlock
        title="Bahdanau (Additive) Attention"
        definition="At each decoder step $t$, compute alignment scores $e_{t,j} = v^T \tanh(W_1 h_j^{\text{enc}} + W_2 s_{t-1}^{\text{dec}})$ for all encoder positions $j$. Normalize via softmax to get attention weights $\alpha_{t,j}$, then compute the context vector as a weighted sum $c_t = \sum_j \alpha_{t,j} h_j^{\text{enc}}$."
        notation="$e_{t,j} \in \mathbb{R}$ is the alignment score, $\alpha_{t,j} \in [0,1]$ are attention weights, $c_t \in \mathbb{R}^d$ is the context vector."
        id="def-bahdanau"
      />

      <h2 className="text-2xl font-semibold">Attention Equations</h2>
      <BlockMath math="e_{t,j} = v^T \tanh(W_1 h_j^{\text{enc}} + W_2 s_{t-1}^{\text{dec}})" />
      <BlockMath math="\alpha_{t,j} = \frac{\exp(e_{t,j})}{\sum_{k=1}^{S} \exp(e_{t,k})}" />
      <BlockMath math="c_t = \sum_{j=1}^{S} \alpha_{t,j} \, h_j^{\text{enc}}" />
      <p className="text-gray-700 dark:text-gray-300">
        The context vector <InlineMath math="c_t" /> is concatenated with the decoder input
        and fed into the decoder RNN:
      </p>
      <BlockMath math="s_t^{\text{dec}} = f(s_{t-1}^{\text{dec}}, [y_{t-1}; c_t])" />

      <ExampleBlock
        title="Attention Alignment"
        problem="For translating 'the black cat' to 'le chat noir', what should the attention weights look like?"
        steps={[
          { formula: '\\alpha_{1,:} \\approx [0.9, 0.05, 0.05]', explanation: 'When generating "le", attention focuses on "the" (position 1).' },
          { formula: '\\alpha_{2,:} \\approx [0.05, 0.1, 0.85]', explanation: 'When generating "chat", attention focuses on "cat" (position 3), not "black".' },
          { formula: '\\alpha_{3,:} \\approx [0.05, 0.85, 0.1]', explanation: 'When generating "noir", attention focuses on "black" (position 2). Note the word-order reversal handled naturally by attention.' },
        ]}
        id="example-alignment"
      />

      <PythonCode
        title="bahdanau_attention.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention mechanism."""
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W1 = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W2 = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (batch, src_len, enc_dim)
        # decoder_hidden: (batch, dec_dim)

        # Expand decoder hidden to match src_len dimension
        dec_expanded = decoder_hidden.unsqueeze(1)  # (batch, 1, dec_dim)

        # Compute alignment scores
        energy = self.v(
            torch.tanh(self.W1(encoder_outputs) + self.W2(dec_expanded))
        )  # (batch, src_len, 1)

        attention_weights = F.softmax(energy.squeeze(-1), dim=-1)
        # (batch, src_len)

        # Weighted sum of encoder outputs
        context = torch.bmm(
            attention_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)  # (batch, enc_dim)

        return context, attention_weights

# Test
attn = BahdanauAttention(enc_dim=512, dec_dim=512, attn_dim=256)
enc_out = torch.randn(4, 20, 512)    # 4 sentences, 20 tokens
dec_h = torch.randn(4, 512)          # decoder state
ctx, weights = attn(enc_out, dec_h)
print(f"Context: {ctx.shape}")        # (4, 512)
print(f"Weights: {weights.shape}")    # (4, 20)
print(f"Weights sum: {weights.sum(-1)}")  # [1, 1, 1, 1]`}
        id="code-bahdanau"
      />

      <PythonCode
        title="attention_decoder.py"
        code={`import torch
import torch.nn as nn

class AttentionDecoder(nn.Module):
    """Decoder with Bahdanau attention."""
    def __init__(self, vocab_size, embed_dim, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(enc_dim, dec_dim, attn_dim)
        # Input: embedding + context vector
        self.rnn = nn.GRU(embed_dim + enc_dim, dec_dim, batch_first=True)
        self.fc_out = nn.Linear(dec_dim + enc_dim + embed_dim, vocab_size)

    def forward(self, tgt_token, decoder_hidden, encoder_outputs):
        # tgt_token: (batch,) single token ids
        embedded = self.embedding(tgt_token)  # (batch, embed_dim)

        context, attn_weights = self.attention(
            encoder_outputs, decoder_hidden
        )

        # Concatenate embedding and context as RNN input
        rnn_input = torch.cat([embedded, context], dim=-1)
        rnn_input = rnn_input.unsqueeze(1)  # (batch, 1, embed+enc)

        output, hidden = self.rnn(
            rnn_input, decoder_hidden.unsqueeze(0)
        )
        hidden = hidden.squeeze(0)  # (batch, dec_dim)

        # Prediction from hidden + context + embedding
        prediction = self.fc_out(
            torch.cat([hidden, context, embedded], dim=-1)
        )
        return prediction, hidden, attn_weights`}
        id="code-attn-decoder"
      />

      <NoteBlock
        type="intuition"
        title="Attention as Soft Addressing"
        content="Think of encoder hidden states as memory slots and the decoder state as a query. Attention computes a similarity between the query and each memory slot, then retrieves a weighted combination. This is essentially a differentiable dictionary lookup -- the foundation of the Query-Key-Value framework in transformers."
        id="note-soft-addressing"
      />

      <WarningBlock
        title="Attention Complexity"
        content="Bahdanau attention has O(S * T) time complexity where S is the source length and T is the target length. For each of the T decoder steps, we compute scores against all S encoder positions. This is acceptable for short sequences but becomes costly for very long inputs (thousands of tokens), motivating efficient attention variants."
        id="warning-attention-cost"
      />

      <NoteBlock
        type="historical"
        title="The Paper That Changed NLP"
        content="'Neural Machine Translation by Jointly Learning to Align and Translate' (Bahdanau, Cho, Bengio, 2015) introduced attention to NLP. The alignment visualization -- showing which source words the model attends to for each target word -- was groundbreaking because it made the model interpretable. This paper has over 30,000 citations and directly inspired the transformer's attention mechanism."
        id="note-bahdanau-history"
      />
    </div>
  )
}
