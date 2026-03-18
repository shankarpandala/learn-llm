import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function FeedForwardNetwork() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Feed-Forward Networks in Transformers</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Every transformer layer contains a position-wise feed-forward network (FFN) applied
        independently to each token. This two-layer MLP is where the majority of the model's
        parameters reside and where much of the "knowledge storage" happens.
      </p>

      <DefinitionBlock
        title="Position-Wise Feed-Forward Network"
        definition="$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$ where $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$, and $\sigma$ is a nonlinear activation function."
        notation="d_ff is typically 4 × d_model. σ is ReLU in the original transformer, GELU in modern variants."
        id="def-ffn"
      />

      <h2 className="text-2xl font-semibold">ReLU vs. GELU Activation</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The original transformer uses ReLU: <InlineMath math="\text{ReLU}(x) = \max(0, x)" />.
        Modern models (BERT, GPT-2+) prefer GELU, which provides a smooth approximation that
        allows small negative values to pass through.
      </p>
      <BlockMath math="\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}(x + 0.044715x^3)\right]\right)" />

      <ExampleBlock
        title="FFN Parameter Count in GPT-3"
        problem="Calculate the FFN parameters per layer in GPT-3 (d_model=12288, d_ff=49152)."
        steps={[
          { formula: 'W_1: 12288 \\times 49152 = 603,979,776', explanation: 'First linear layer expands from d_model to d_ff = 4 * d_model.' },
          { formula: 'W_2: 49152 \\times 12288 = 603,979,776', explanation: 'Second linear layer contracts back to d_model.' },
          { formula: 'b_1 + b_2 = 49152 + 12288 = 61,440', explanation: 'Bias terms (small relative to weights).' },
          { formula: '\\text{Total} \\approx 1.208\\text{B parameters per layer}', explanation: 'The FFN has ~2/3 of each layer\'s total parameters.' },
        ]}
        id="example-ffn-params"
      />

      <NoteBlock
        type="intuition"
        title="FFN as Key-Value Memory"
        content="Research by Geva et al. (2021) showed that FFN layers function as key-value memories. The first layer's rows act as keys that pattern-match on input features, while the second layer's columns act as values that contribute to the output distribution. Individual neurons can encode specific facts."
        id="note-kv-memory"
      />

      <PythonCode
        title="feed_forward_network.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerFFN(nn.Module):
    """Position-wise FFN with configurable activation."""
    def __init__(self, d_model, d_ff, activation='gelu', dropout=0.1):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        if self.activation == 'relu':
            hidden = F.relu(self.W1(x))
        elif self.activation == 'gelu':
            hidden = F.gelu(self.W1(x))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        return self.W2(self.dropout(hidden))

# SwiGLU variant used in LLaMA, PaLM, etc.
class SwiGLUFFN(nn.Module):
    """Gated FFN: SwiGLU activation (Shazeer 2020)."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.W_gate(x))      # SiLU = x * sigmoid(x)
        up = self.W_up(x)
        return self.W_down(self.dropout(gate * up))

# Compare parameter counts
d_model, d_ff = 768, 3072
ffn_standard = TransformerFFN(d_model, d_ff)
ffn_swiglu = SwiGLUFFN(d_model, d_ff)

params_std = sum(p.numel() for p in ffn_standard.parameters())
params_glu = sum(p.numel() for p in ffn_swiglu.parameters())
print(f"Standard FFN params: {params_std:,}")   # ~4.7M
print(f"SwiGLU FFN params:   {params_glu:,}")   # ~7.1M (3 matrices)

# Forward pass
x = torch.randn(2, 10, d_model)
print(f"Standard output: {ffn_standard(x).shape}")
print(f"SwiGLU output:   {ffn_swiglu(x).shape}")`}
        id="code-ffn"
      />

      <WarningBlock
        title="SwiGLU Changes the Parameter Budget"
        content="SwiGLU uses three weight matrices instead of two, increasing FFN parameters by 50%. To keep the total parameter count constant, LLaMA reduces d_ff from 4*d_model to roughly 2.67*d_model (specifically 8/3*d_model rounded to multiples of 256). Always account for this when comparing architectures."
        id="warning-swiglu-params"
      />

      <NoteBlock
        type="note"
        title="Position-Wise Means Per-Token"
        content="The FFN is applied identically and independently to every position in the sequence. There is no interaction between tokens in the FFN — all cross-position communication happens in the attention layer. This makes FFNs embarrassingly parallel across the sequence dimension."
        id="note-position-wise"
      />
    </div>
  )
}
