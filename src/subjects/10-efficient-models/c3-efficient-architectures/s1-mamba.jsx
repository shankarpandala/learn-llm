import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function MambaSSM() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Mamba: Selective State Space Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Mamba introduces a selective state space model (SSM) that achieves Transformer-quality
        language modeling with linear-time complexity in sequence length. Unlike attention's
        quadratic cost, Mamba processes sequences in O(n) time and O(1) memory per step during
        generation, enabling efficient handling of very long contexts.
      </p>

      <DefinitionBlock
        title="Continuous State Space Model"
        definition="A state space model maps an input sequence $x(t)$ to output $y(t)$ through a latent state $h(t) \in \mathbb{R}^N$: $h'(t) = A h(t) + B x(t)$ and $y(t) = C h(t) + D x(t)$, where $A \in \mathbb{R}^{N \times N}$ is the state matrix, $B \in \mathbb{R}^{N \times 1}$ the input matrix, and $C \in \mathbb{R}^{1 \times N}$ the output matrix."
        notation="Discretized with step size $\Delta$: $\bar{A} = \exp(\Delta A)$, $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$. Recurrence: $h_k = \bar{A} h_{k-1} + \bar{B} x_k$, $y_k = C h_k$."
        id="def-ssm"
      />

      <DefinitionBlock
        title="Selective State Space (Mamba)"
        definition="Mamba makes SSM parameters input-dependent: $B_k = s_B(x_k)$, $C_k = s_C(x_k)$, and $\Delta_k = \text{softplus}(s_\Delta(x_k))$, where $s_B$, $s_C$, $s_\Delta$ are learned linear projections. This selectivity allows the model to filter information based on content, analogous to attention's content-based routing."
        id="def-mamba-selective"
      />

      <ExampleBlock
        title="SSM Recurrence Step"
        problem="Given state h_0 = [0.5, -0.3], A_bar = [[0.9, 0.1], [0, 0.8]], B_bar = [0.2, 0.1], C = [1, 1], compute output for input x_1 = 3.0."
        steps={[
          {
            formula: 'h_1 = \\bar{A} h_0 + \\bar{B} x_1 = \\begin{bmatrix} 0.9(0.5)+0.1(-0.3) \\\\ 0(0.5)+0.8(-0.3) \\end{bmatrix} + 3.0 \\begin{bmatrix} 0.2 \\\\ 0.1 \\end{bmatrix}',
            explanation: 'Apply state transition and input injection.'
          },
          {
            formula: 'h_1 = \\begin{bmatrix} 0.42 \\\\ -0.24 \\end{bmatrix} + \\begin{bmatrix} 0.6 \\\\ 0.3 \\end{bmatrix} = \\begin{bmatrix} 1.02 \\\\ 0.06 \\end{bmatrix}',
            explanation: 'New hidden state combines memory and new input.'
          },
          {
            formula: 'y_1 = C h_1 = [1, 1] \\cdot [1.02, 0.06] = 1.08',
            explanation: 'Output is a linear readout of the state.'
          }
        ]}
        id="example-ssm-recurrence"
      />

      <TheoremBlock
        title="Linear-Time Sequence Processing"
        statement="An SSM processes a sequence of length $L$ in $O(L \cdot N)$ time and $O(N)$ memory during autoregressive generation, compared to $O(L^2 \cdot d)$ time for self-attention. For training, Mamba uses a parallel scan algorithm achieving $O(L \cdot N \cdot \log L)$ time."
        proof="Each recurrence step $h_k = \bar{A}_k h_{k-1} + \bar{B}_k x_k$ is $O(N)$ for state dimension $N$. Over $L$ steps: $O(LN)$. The parallel scan exploits associativity of the linear recurrence, enabling GPU-efficient parallel computation in $O(L \log L)$ parallel time."
        id="thm-ssm-complexity"
      />

      <PythonCode
        title="mamba_selective_ssm.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    """Simplified Mamba selective state space model block."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner
        )

        # SSM parameters: input-dependent (selective)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # B, C, delta
        self.dt_proj = nn.Linear(1, d_inner, bias=True)  # Expand delta

        # State matrix A (structured, not input-dependent)
        A = torch.arange(1, d_state + 1).float()
        self.A_log = nn.Parameter(torch.log(A))  # Learn in log space

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.d_state = d_state

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        batch, seq_len, _ = x.shape

        # Project and split into two paths
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        # 1D convolution for local context
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Compute input-dependent SSM parameters (SELECTIVE)
        x_ssm = self.x_proj(x_conv)
        B = x_ssm[:, :, :self.d_state]           # (batch, seq, d_state)
        C = x_ssm[:, :, self.d_state:2*self.d_state]
        delta = F.softplus(x_ssm[:, :, -1:])      # (batch, seq, 1)

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_state,) - negative for stability
        # Simplified: run SSM recurrence
        h = torch.zeros(batch, x_conv.shape[-1], self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            dt = delta[:, t, :]  # (batch, 1)
            A_bar = torch.exp(dt * A)  # (batch, d_state)
            B_bar = dt * B[:, t, :]
            h = A_bar.unsqueeze(1) * h + B_bar.unsqueeze(1) * x_conv[:, t:t+1, :].transpose(1, 2)
            y_t = (C[:, t, :].unsqueeze(2) * h).sum(dim=1)  # (batch, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)

        # Gate and project output
        y = y * F.silu(z)
        return self.out_proj(y)

# Benchmark: Mamba vs Attention scaling
model = SelectiveSSM(d_model=256, d_state=16)
for seq_len in [128, 512, 1024, 2048]:
    x = torch.randn(2, seq_len, 256)
    import time
    start = time.time()
    y = model(x)
    elapsed = (time.time() - start) * 1000
    print(f"Seq={seq_len:>5}: output={y.shape}, time={elapsed:.1f}ms")
    # Time grows linearly, not quadratically!`}
        id="code-mamba"
      />

      <NoteBlock
        type="intuition"
        title="Selectivity as Learned Gating"
        content="In standard SSMs, B and C are fixed, so the model treats all inputs identically. Mamba's selectivity makes these input-dependent — the model can 'choose' what to store in its state and what to read out. This is conceptually similar to how attention selects which tokens to attend to, but achieved through recurrent dynamics instead of pairwise comparisons."
        id="note-selectivity-intuition"
      />

      <WarningBlock
        title="Trade-offs vs. Attention"
        content="While Mamba is faster for long sequences, it cannot perform exact token-to-token lookback like attention. Tasks requiring precise copying or retrieval from distant context may still favor Transformers. Hybrid architectures (like Jamba) combine both mechanisms to get the best of both worlds."
        id="warning-mamba-tradeoffs"
      />
    </div>
  )
}
