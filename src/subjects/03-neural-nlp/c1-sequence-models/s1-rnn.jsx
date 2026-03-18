import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function RNN() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Recurrent Neural Networks (RNN)</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Recurrent Neural Networks introduced the idea of maintaining a hidden state that gets
        updated at each time step, allowing neural networks to process sequences of arbitrary
        length. This section covers the core RNN architecture, its forward equations, and
        practical implementation in PyTorch.
      </p>

      <DefinitionBlock
        title="Recurrent Neural Network"
        definition="An RNN is a neural network that processes sequential input $x_1, x_2, \ldots, x_T$ by maintaining a hidden state $h_t$ that is updated at each time step according to $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$."
        notation="$h_t \in \mathbb{R}^d$ is the hidden state, $W_{hh} \in \mathbb{R}^{d \times d}$ is the recurrent weight matrix, $W_{xh} \in \mathbb{R}^{d \times n}$ maps input to hidden space."
        id="def-rnn"
      />

      <h2 className="text-2xl font-semibold">RNN Forward Equations</h2>
      <p className="text-gray-700 dark:text-gray-300">
        At each time step <InlineMath math="t" />, the RNN computes:
      </p>
      <BlockMath math="h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)" />
      <BlockMath math="y_t = W_{hy} h_t + b_y" />
      <p className="text-gray-700 dark:text-gray-300">
        The hidden state <InlineMath math="h_t" /> serves as the network's memory. It encodes
        information about all previous inputs <InlineMath math="x_1, \ldots, x_t" /> seen so far.
        The output <InlineMath math="y_t" /> is a linear projection used for the task at hand
        (classification, next-token prediction, etc.).
      </p>

      <ExampleBlock
        title="Hidden State Dimensions"
        problem="An RNN has hidden size 256 and processes word embeddings of dimension 100. What are the shapes of the weight matrices?"
        steps={[
          { formula: 'W_{xh} \\in \\mathbb{R}^{256 \\times 100}', explanation: 'Maps 100-dim input embeddings to 256-dim hidden space.' },
          { formula: 'W_{hh} \\in \\mathbb{R}^{256 \\times 256}', explanation: 'Recurrent weights connecting previous hidden state to current.' },
          { formula: 'b_h \\in \\mathbb{R}^{256}', explanation: 'Bias vector for the hidden state update.' },
          { formula: '\\text{Total params} = 256 \\times 100 + 256 \\times 256 + 256 = 91{,}392', explanation: 'Parameter count for the recurrent cell alone (excluding output layer).' },
        ]}
        id="example-rnn-dims"
      />

      <PythonCode
        title="rnn_from_scratch.py"
        code={`import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """Vanilla RNN implemented from scratch."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hy = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_0=None):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=x.device)

        h_t = h_0
        outputs = []
        for t in range(seq_len):
            h_t = torch.tanh(self.W_xh(x[:, t]) + self.W_hh(h_t))
            outputs.append(h_t)

        # Stack all hidden states: (batch, seq_len, hidden_size)
        hidden_states = torch.stack(outputs, dim=1)
        # Output projection on final hidden state
        out = self.W_hy(h_t)
        return out, hidden_states

# Usage
model = SimpleRNN(input_size=100, hidden_size=256, output_size=10)
x = torch.randn(32, 20, 100)  # batch=32, seq_len=20, embed=100
logits, all_h = model(x)
print(f"Output shape: {logits.shape}")       # (32, 10)
print(f"Hidden states: {all_h.shape}")       # (32, 20, 256)`}
        id="code-rnn-scratch"
      />

      <PythonCode
        title="rnn_pytorch_builtin.py"
        code={`import torch
import torch.nn as nn

# PyTorch built-in RNN for text classification
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) of token IDs
        emb = self.embedding(x)          # (batch, seq_len, embed_dim)
        output, h_n = self.rnn(emb)      # h_n: (1, batch, hidden_dim)
        logits = self.fc(h_n.squeeze(0)) # (batch, num_classes)
        return logits

model = RNNClassifier(vocab_size=10000, embed_dim=128,
                      hidden_dim=256, num_classes=4)
tokens = torch.randint(0, 10000, (16, 50))  # batch=16, seq_len=50
print(model(tokens).shape)  # (16, 4)`}
        id="code-rnn-pytorch"
      />

      <NoteBlock
        type="intuition"
        title="The Hidden State as Memory"
        content="Think of the hidden state as a fixed-size summary of everything the network has read so far. At each step, the RNN must decide what to keep from its current memory and what to incorporate from the new input. This compression into a fixed-size vector is both the RNN's strength (constant memory) and its weakness (information bottleneck)."
        id="note-hidden-state"
      />

      <WarningBlock
        title="Vanilla RNNs Struggle with Long Sequences"
        content="In practice, vanilla RNNs have difficulty learning dependencies that span more than 10-20 time steps. The repeated matrix multiplication in the recurrence causes gradients to either vanish or explode during backpropagation through time (BPTT). This is addressed by LSTM and GRU architectures."
        id="warning-rnn-limits"
      />

      <NoteBlock
        type="historical"
        title="Origins of Recurrent Networks"
        content="The Elman network (1990) introduced the simple recurrent architecture with a hidden state fed back as input. Jordan networks (1986) instead fed the output back. Backpropagation Through Time (BPTT), the algorithm for training RNNs, was formalized by Werbos (1990), though the idea dates to the 1980s."
        id="note-rnn-history"
      />
    </div>
  )
}
