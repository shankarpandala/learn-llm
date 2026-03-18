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
      <h1 className="text-3xl font-bold">Encoder-Decoder Architecture</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The encoder-decoder (seq2seq) architecture maps variable-length input sequences to
        variable-length output sequences. The encoder compresses the input into a fixed-size
        context vector, and the decoder generates the output one token at a time, conditioned
        on this context. This architecture powered the first neural machine translation systems.
      </p>

      <DefinitionBlock
        title="Sequence-to-Sequence Model"
        definition="A seq2seq model consists of an encoder RNN that reads the input sequence $x_1, \ldots, x_S$ and produces a context vector $c = h_S^{\text{enc}}$, and a decoder RNN that generates the output sequence $y_1, \ldots, y_T$ conditioned on $c$."
        notation="$P(y_1, \ldots, y_T | x_1, \ldots, x_S) = \prod_{t=1}^{T} P(y_t | y_{<t}, c)$"
        id="def-seq2seq"
      />

      <h2 className="text-2xl font-semibold">The Information Bottleneck</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The context vector <InlineMath math="c \in \mathbb{R}^d" /> must encode all information
        from the source sequence into a single fixed-size vector. For a 50-word sentence compressed
        into a 512-dimensional vector, each dimension must encode roughly 0.1 words worth of
        information. This bottleneck degrades performance on long sequences.
      </p>
      <BlockMath math="c = h_S^{\text{enc}} = f_{\text{enc}}(x_1, x_2, \ldots, x_S)" />
      <BlockMath math="h_t^{\text{dec}} = f_{\text{dec}}(y_{t-1}, h_{t-1}^{\text{dec}}, c)" />
      <BlockMath math="P(y_t | y_{<t}, c) = \text{softmax}(W_o h_t^{\text{dec}})" />

      <ExampleBlock
        title="Seq2Seq for Translation"
        problem="Trace through a seq2seq model translating 'I love cats' to 'J'aime les chats'."
        steps={[
          { formula: 'h_1 = \\text{enc}(\\text{emb}(\\text{\"I\"}), h_0)', explanation: 'Encoder processes first token, updating hidden state.' },
          { formula: 'h_2 = \\text{enc}(\\text{emb}(\\text{\"love\"}), h_1)', explanation: 'Second token encoded, hidden state accumulates meaning.' },
          { formula: 'c = h_3 = \\text{enc}(\\text{emb}(\\text{\"cats\"}), h_2)', explanation: 'Final encoder hidden state becomes the context vector.' },
          { formula: 'P(y_1|c) \\to \\text{\"J\'aime\"}', explanation: 'Decoder generates first target token from context vector.' },
          { formula: 'P(y_2|y_1, c) \\to \\text{\"les\"}', explanation: 'Each subsequent token is conditioned on previous outputs and context.' },
        ]}
        id="example-translation"
      />

      <PythonCode
        title="seq2seq_model.py"
        code={`import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2,
                 dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout)

    def forward(self, src):
        # src: (batch, src_len)
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        # hidden: (num_layers, batch, hidden_dim) -- context vector
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2,
                 dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_token, hidden, cell):
        # tgt_token: (batch, 1) -- single token
        embedded = self.embedding(tgt_token)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, tgt_len, vocab_size,
                              device=self.device)

        hidden, cell = self.encoder(src)
        input_tok = tgt[:, 0:1]  # <SOS> token

        for t in range(1, tgt_len):
            pred, hidden, cell = self.decoder(input_tok, hidden, cell)
            outputs[:, t] = pred
            # Teacher forcing: use ground truth or model prediction
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_tok = tgt[:, t:t+1]
            else:
                input_tok = pred.argmax(dim=-1, keepdim=True)
        return outputs

# Build model
device = torch.device('cpu')
enc = Encoder(vocab_size=10000, embed_dim=256, hidden_dim=512)
dec = Decoder(vocab_size=8000, embed_dim=256, hidden_dim=512)
model = Seq2Seq(enc, dec, device)
src = torch.randint(0, 10000, (4, 20))
tgt = torch.randint(0, 8000, (4, 15))
out = model(src, tgt)
print(f"Output: {out.shape}")  # (4, 15, 8000)`}
        id="code-seq2seq"
      />

      <WarningBlock
        title="The Bottleneck Problem Is Real"
        content="Cho et al. (2014) showed that seq2seq performance degrades sharply for sentences longer than ~20 tokens. The fixed-size context vector simply cannot retain all necessary information from long inputs. This was the primary motivation for introducing attention mechanisms (Bahdanau et al., 2015)."
        id="warning-bottleneck"
      />

      <NoteBlock
        type="historical"
        title="Birth of Neural Machine Translation"
        content="The seq2seq architecture was independently proposed by Sutskever et al. (2014) at Google and Cho et al. (2014). Sutskever's key trick was reversing the source sentence order, which shortened the distance between corresponding words and improved BLEU scores. Within two years, Google deployed a production NMT system (Wu et al., 2016) based on these ideas."
        id="note-nmt-history"
      />

      <NoteBlock
        type="tip"
        title="Tricks for Better Seq2Seq"
        content="Common improvements include: (1) using bidirectional encoder, (2) multi-layer LSTMs with residual connections, (3) reversing the source sequence, (4) beam search decoding instead of greedy, (5) input feeding (concatenating attention output to decoder input). Each provides incremental gains, but attention was the transformative addition."
        id="note-seq2seq-tricks"
      />
    </div>
  )
}
