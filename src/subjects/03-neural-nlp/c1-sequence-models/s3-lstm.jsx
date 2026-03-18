import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LSTM() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LSTM Architecture</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Long Short-Term Memory (LSTM) networks solve the vanishing gradient problem by
        introducing a gating mechanism that controls information flow. The cell state acts
        as a conveyor belt, allowing gradients to flow across many time steps with minimal
        decay. LSTMs were the dominant sequence model from 1997 until transformers emerged in 2017.
      </p>

      <DefinitionBlock
        title="LSTM Cell"
        definition="An LSTM cell maintains two state vectors: the hidden state $h_t$ and the cell state $c_t$. Three gates (forget, input, output) regulate information flow. The cell state update is additive rather than multiplicative, creating a gradient highway."
        notation="$f_t, i_t, o_t \in (0,1)^d$ are the forget, input, and output gates; $c_t \in \mathbb{R}^d$ is the cell state."
        id="def-lstm"
      />

      <h2 className="text-2xl font-semibold">LSTM Gate Equations</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The LSTM computes four quantities at each time step:
      </p>
      <BlockMath math="f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}" />
      <BlockMath math="i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(input gate)}" />
      <BlockMath math="\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \quad \text{(candidate cell)}" />
      <BlockMath math="c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell update)}" />
      <BlockMath math="o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(output gate)}" />
      <BlockMath math="h_t = o_t \odot \tanh(c_t) \quad \text{(hidden state)}" />

      <ExampleBlock
        title="Gate Roles in Practice"
        problem="Explain what each gate does when processing the sentence: 'The cat sat on the mat. It was happy.'"
        steps={[
          { formula: 'f_t \\approx 1 \\text{ (keep)}, f_t \\approx 0 \\text{ (forget)}', explanation: 'The forget gate decides to keep "cat" info through "sat on the mat" (f near 1) and might clear some info at the period (f near 0).' },
          { formula: 'i_t \\approx 1 \\text{ when new info is relevant}', explanation: 'The input gate opens when processing "happy" to write the sentiment into the cell state.' },
          { formula: 'o_t \\text{ controls what is exposed}', explanation: 'The output gate selects which cell dimensions are relevant for the current prediction. At "It", it exposes the subject info to resolve the pronoun.' },
        ]}
        id="example-gates"
      />

      <NoteBlock
        type="intuition"
        title="Why the Cell State Fixes Vanishing Gradients"
        content="The key insight is the cell state update: c_t = f_t * c_{t-1} + i_t * c_tilde. This is an additive update, not a multiplicative one. When the forget gate is close to 1, the gradient flows through the cell state almost unchanged, like a skip connection. The gradient of c_T with respect to c_t is the product of forget gates, which can stay close to 1."
        id="note-gradient-highway"
      />

      <PythonCode
        title="lstm_from_scratch.py"
        code={`import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    """LSTM cell implemented from scratch."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Combined linear for all 4 gates (efficiency)
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x_t, state):
        h_prev, c_prev = state
        combined = torch.cat([h_prev, x_t], dim=-1)
        gates = self.gates(combined)

        # Split into 4 gate activations
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        g = torch.tanh(g)     # candidate cell
        o = torch.sigmoid(o)  # output gate

        c_t = f * c_prev + i * g   # cell state update
        h_t = o * torch.tanh(c_t)  # hidden state
        return h_t, c_t

# Test
cell = LSTMCell(input_size=100, hidden_size=256)
h = torch.zeros(32, 256)
c = torch.zeros(32, 256)
x = torch.randn(32, 100)
h_new, c_new = cell(x, (h, c))
print(f"h: {h_new.shape}, c: {c_new.shape}")  # (32, 256), (32, 256)`}
        id="code-lstm-scratch"
      />

      <PythonCode
        title="lstm_text_classifier.py"
        code={`import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        # output: (batch, seq_len, hidden_dim)
        # h_n: (num_layers, batch, hidden_dim)
        output, (h_n, c_n) = self.lstm(emb)
        # Use final layer's hidden state
        logits = self.fc(self.dropout(h_n[-1]))
        return logits

model = LSTMClassifier(
    vocab_size=30000, embed_dim=128,
    hidden_dim=256, num_classes=5
)
x = torch.randint(0, 30000, (16, 100))
print(f"Output: {model(x).shape}")  # (16, 5)

# Parameter count comparison with vanilla RNN
rnn_params = 128*256 + 256*256 + 256      # ~98K
lstm_params = 4 * (128*256 + 256*256 + 256)  # ~394K
print(f"RNN params: {rnn_params:,}")
print(f"LSTM params: {lstm_params:,}")  # 4x more due to 4 gates`}
        id="code-lstm-classifier"
      />

      <WarningBlock
        title="LSTM Parameter Count"
        content="LSTMs have 4x the parameters of a vanilla RNN with the same hidden size because they compute four gate/candidate values. For hidden_size=d and input_size=n, an LSTM cell has 4(n*d + d*d + d) parameters. This makes them slower to train but far more capable at capturing long-range dependencies."
        id="warning-lstm-params"
      />

      <NoteBlock
        type="historical"
        title="LSTM Timeline"
        content="LSTMs were introduced by Hochreiter and Schmidhuber in 1997. The forget gate was added by Gers et al. in 2000 (the original had no forget gate). Peephole connections (letting gates see the cell state directly) were introduced in 2002 but are rarely used today. LSTMs dominated NLP from roughly 2014-2017, powering Google Translate, speech recognition, and state-of-the-art language models."
        id="note-lstm-history"
      />
    </div>
  )
}
