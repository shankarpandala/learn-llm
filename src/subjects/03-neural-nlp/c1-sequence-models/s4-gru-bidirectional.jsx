import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function GRUBidirectional() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">GRU and Bidirectional RNNs</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The Gated Recurrent Unit (GRU) simplifies the LSTM by merging the forget and input gates
        into a single update gate, and combining the cell and hidden states. Bidirectional RNNs
        process sequences in both directions to capture context from both past and future tokens.
      </p>

      <DefinitionBlock
        title="Gated Recurrent Unit (GRU)"
        definition="A GRU uses two gates: an update gate $z_t$ and a reset gate $r_t$. The update gate interpolates between the previous hidden state and a candidate, while the reset gate controls how much past state enters the candidate computation."
        notation="$z_t = \sigma(W_z [h_{t-1}, x_t])$, $r_t = \sigma(W_r [h_{t-1}, x_t])$, $\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t])$, $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$"
        id="def-gru"
      />

      <h2 className="text-2xl font-semibold">GRU Equations</h2>
      <BlockMath math="z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z) \quad \text{(update gate)}" />
      <BlockMath math="r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r) \quad \text{(reset gate)}" />
      <BlockMath math="\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)" />
      <BlockMath math="h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t" />
      <p className="text-gray-700 dark:text-gray-300">
        When <InlineMath math="z_t \approx 0" />, the hidden state is copied forward unchanged
        (like a skip connection). When <InlineMath math="z_t \approx 1" />, the state is fully
        replaced by the candidate. The reset gate <InlineMath math="r_t" /> controls how much
        history enters the candidate computation.
      </p>

      <ExampleBlock
        title="GRU vs LSTM Parameter Count"
        problem="Compare parameter counts for GRU and LSTM with input_size=128, hidden_size=256."
        steps={[
          { formula: '\\text{LSTM: } 4 \\times (128 \\times 256 + 256 \\times 256 + 256) = 394{,}240', explanation: 'LSTM has 4 weight matrices (forget, input, candidate, output).' },
          { formula: '\\text{GRU: } 3 \\times (128 \\times 256 + 256 \\times 256 + 256) = 295{,}680', explanation: 'GRU has 3 weight matrices (update, reset, candidate) -- 25% fewer parameters.' },
          { formula: '\\text{Ratio: } 295{,}680 / 394{,}240 = 0.75', explanation: 'GRU uses 75% of the parameters of an LSTM, leading to faster training with comparable performance on many tasks.' },
        ]}
        id="example-gru-params"
      />

      <PythonCode
        title="gru_implementation.py"
        code={`import torch
import torch.nn as nn

class GRUCell(nn.Module):
    """GRU cell from scratch."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t, h_prev):
        combined = torch.cat([x_t, h_prev], dim=-1)
        z = torch.sigmoid(self.W_z(combined))  # update gate
        r = torch.sigmoid(self.W_r(combined))  # reset gate

        combined_r = torch.cat([x_t, r * h_prev], dim=-1)
        h_candidate = torch.tanh(self.W_h(combined_r))

        h_t = (1 - z) * h_prev + z * h_candidate
        return h_t

# Quick test
cell = GRUCell(128, 256)
h = torch.zeros(32, 256)
x = torch.randn(32, 128)
h_new = cell(x, h)
print(f"New hidden state: {h_new.shape}")  # (32, 256)`}
        id="code-gru-scratch"
      />

      <h2 className="text-2xl font-semibold">Bidirectional RNNs</h2>
      <p className="text-gray-700 dark:text-gray-300">
        A unidirectional RNN only sees past context when computing <InlineMath math="h_t" />.
        For many NLP tasks (named entity recognition, sentiment analysis, machine comprehension),
        future context is equally informative. Bidirectional RNNs run two separate RNNs: one
        forward and one backward, then concatenate their hidden states.
      </p>
      <BlockMath math="\overrightarrow{h_t} = \text{RNN}_{\text{fwd}}(x_t, \overrightarrow{h_{t-1}})" />
      <BlockMath math="\overleftarrow{h_t} = \text{RNN}_{\text{bwd}}(x_t, \overleftarrow{h_{t+1}})" />
      <BlockMath math="h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}] \in \mathbb{R}^{2d}" />

      <PythonCode
        title="bidirectional_lstm.py"
        code={`import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,  # <-- key change
            dropout=0.3,
        )
        # hidden_dim * 2 because bidirectional concatenates
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        output, (h_n, _) = self.lstm(emb)
        # h_n shape: (num_layers*2, batch, hidden_dim)
        # Concatenate final forward and backward hidden states
        fwd = h_n[-2]  # last forward layer
        bwd = h_n[-1]  # last backward layer
        combined = torch.cat([fwd, bwd], dim=-1)
        return self.fc(combined)

model = BiLSTMClassifier(20000, 128, 256, 3)
x = torch.randint(0, 20000, (16, 80))
print(f"Output: {model(x).shape}")  # (16, 3)

# Bidirectional doubles the representation size at each step
output, _ = model.lstm(model.embedding(x))
print(f"BiLSTM output: {output.shape}")  # (16, 80, 512)`}
        id="code-bilstm"
      />

      <NoteBlock
        type="tip"
        title="When to Use Bidirectional"
        content="Use bidirectional RNNs for encoding tasks where the full sequence is available (classification, tagging, question answering). Do NOT use them for autoregressive generation (language modeling, machine translation decoding), since the model would be cheating by looking at future tokens it is supposed to predict."
        id="note-when-bidir"
      />

      <WarningBlock
        title="Bidirectional Doubles Memory and Compute"
        content="A bidirectional LSTM with hidden_size=256 produces 512-dimensional representations and has twice the parameters of a unidirectional one. For very long sequences, this can be a significant memory burden. Also, bidirectional models cannot be used for streaming/online inference since they require the complete input."
        id="warning-bidir-cost"
      />

      <NoteBlock
        type="historical"
        title="GRU and Bidirectional Origins"
        content="The GRU was proposed by Cho et al. (2014) in the context of neural machine translation. Empirical studies (Chung et al., 2014; Jozefowicz et al., 2015) found GRU and LSTM perform comparably, with GRU sometimes better on smaller datasets. Bidirectional RNNs were introduced by Schuster and Paliwal (1997) and became standard in NLP with ELMo (Peters et al., 2018)."
        id="note-gru-history"
      />
    </div>
  )
}
