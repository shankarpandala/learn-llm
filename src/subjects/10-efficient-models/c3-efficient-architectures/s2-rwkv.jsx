import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function RWKV() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">RWKV: Linear Attention with RNN Efficiency</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        RWKV (Receptance Weighted Key Value) combines the training parallelism of Transformers
        with the inference efficiency of RNNs. It replaces quadratic attention with a linear
        mechanism that can be computed either as a parallel scan during training or as a
        constant-memory recurrence during inference.
      </p>

      <DefinitionBlock
        title="RWKV Time-Mixing"
        definition="RWKV's time-mixing block computes a weighted combination of current and previous token features. For token at position $t$: $r_t = W_r \cdot (\mu_r x_t + (1 - \mu_r) x_{t-1})$, $k_t = W_k \cdot (\mu_k x_t + (1 - \mu_k) x_{t-1})$, $v_t = W_v \cdot (\mu_v x_t + (1 - \mu_v) x_{t-1})$, where $\mu$ are learned interpolation parameters."
        notation="Output: $o_t = \sigma(r_t) \odot \frac{\sum_{i=1}^{t} e^{-(t-i)w + k_i} v_i}{\sum_{i=1}^{t} e^{-(t-i)w + k_i}}$ where $w$ is a channel-wise decay factor."
        id="def-rwkv-time-mixing"
      />

      <DefinitionBlock
        title="RWKV Channel-Mixing"
        definition="The channel-mixing block (replacing FFN) uses gated linear units: $r_t = W_r \cdot (\mu_r x_t + (1-\mu_r) x_{t-1})$, $k_t = W_k \cdot (\mu_k x_t + (1-\mu_k) x_{t-1})$, output $= \sigma(r_t) \odot (W_v \cdot \max(k_t, 0)^2)$. The squared ReLU activation provides nonlinearity without traditional MLP structure."
        id="def-rwkv-channel-mixing"
      />

      <ExampleBlock
        title="RWKV Recurrent Inference"
        problem="Show how RWKV maintains constant memory during autoregressive generation."
        steps={[
          {
            formula: 'a_t = e^{-w} \\cdot a_{t-1} + e^{k_t} \\cdot v_t',
            explanation: 'Running numerator: exponentially decayed sum of weighted values.'
          },
          {
            formula: 'b_t = e^{-w} \\cdot b_{t-1} + e^{k_t}',
            explanation: 'Running denominator: exponentially decayed sum of weights.'
          },
          {
            formula: 'wkv_t = a_t / b_t',
            explanation: 'Weighted key-value output, computed from two scalar states per channel.'
          },
          {
            formula: 'o_t = \\sigma(r_t) \\odot wkv_t',
            explanation: 'Gate the output with receptance. Memory: O(d) regardless of sequence length.'
          }
        ]}
        id="example-rwkv-recurrence"
      />

      <PythonCode
        title="rwkv_mechanism.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class RWKVTimeMixing(nn.Module):
    """RWKV time-mixing block with linear attention."""
    def __init__(self, d_model, layer_id=0, n_layers=12):
        super().__init__()
        self.d_model = d_model

        # Learnable interpolation factors (token shift mixing)
        ratio = layer_id / max(n_layers - 1, 1)
        self.time_mix_r = nn.Parameter(torch.ones(d_model) * (1 - ratio))
        self.time_mix_k = nn.Parameter(torch.ones(d_model) * (1 - ratio))
        self.time_mix_v = nn.Parameter(torch.ones(d_model) * (1 - ratio))

        # Channel-wise decay (learned)
        self.time_decay = nn.Parameter(torch.randn(d_model) * 0.1 - 5.0)
        self.time_first = nn.Parameter(torch.randn(d_model) * 0.1)

        # Projections
        self.W_r = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, state=None):
        """x: (batch, seq_len, d_model)"""
        B, T, C = x.shape

        # Token shift: mix current and previous token
        x_prev = torch.zeros_like(x[:, :1, :]) if state is None else state
        x_shifted = torch.cat([x_prev, x[:, :-1, :]], dim=1)

        r = self.W_r(x * self.time_mix_r + x_shifted * (1 - self.time_mix_r))
        k = self.W_k(x * self.time_mix_k + x_shifted * (1 - self.time_mix_k))
        v = self.W_v(x * self.time_mix_v + x_shifted * (1 - self.time_mix_v))

        # WKV computation (parallel mode for training)
        w = -torch.exp(self.time_decay)  # Negative decay
        u = self.time_first

        # Recurrent WKV (simple version)
        wkv = torch.zeros(B, C, device=x.device)
        a = torch.zeros(B, C, device=x.device)
        b = torch.zeros(B, C, device=x.device)
        outputs = []

        for t in range(T):
            kt, vt = k[:, t], v[:, t]
            # Numerically stable computation
            ww = w + kt
            p = torch.max(a, ww)
            e1 = torch.exp(a - p)
            e2 = torch.exp(ww - p)
            wkv = (e1 * a + e2 * vt) / (e1 * b + e2)
            # For next step: include bonus for first token
            qq = torch.max(w + a, u + kt)
            e1 = torch.exp(w + a - qq)
            e2 = torch.exp(u + kt - qq)
            a = qq + torch.log(e1 + e2)
            b = e1 * b + e2
            outputs.append(wkv)

        wkv_out = torch.stack(outputs, dim=1)

        # Receptance gating
        out = torch.sigmoid(r) * wkv_out
        return self.W_o(out), x[:, -1:, :]  # Return state for next call

# Compare memory usage: RWKV vs Attention
d = 512
seq_lengths = [256, 1024, 4096, 16384]
print("Memory comparison (relative to d_model):")
print(f"{'Seq Len':>8} {'Attention KV':>14} {'RWKV State':>12}")
for L in seq_lengths:
    attn_memory = L * d * 2  # K and V cache
    rwkv_memory = d * 2      # Just a and b states (constant!)
    print(f"{L:>8} {attn_memory:>14,} {rwkv_memory:>12,}")`}
        id="code-rwkv"
      />

      <NoteBlock
        type="historical"
        title="RWKV Evolution"
        content="RWKV was created by Bo Peng as an open-source project. RWKV-4 (2023) demonstrated competitive performance with Transformers up to 14B parameters. RWKV-5 (Eagle) and RWKV-6 (Finch) introduced multi-headed variants and improved the WKV mechanism. The model is fully open-source and has an active community."
        id="note-rwkv-history"
      />

      <NoteBlock
        type="tip"
        title="When to Use RWKV"
        content="RWKV excels in scenarios with very long sequences and streaming inference (chatbots, real-time processing). Its constant memory footprint during generation means a 14B RWKV model uses the same memory generating the 100th token as the 100,000th. Choose attention-based models when precise recall over long contexts is critical."
        id="note-rwkv-when"
      />

      <WarningBlock
        title="Finite State Capacity"
        content="RWKV's recurrent state has fixed capacity (d_model floats). As sequences grow very long, earlier information is exponentially decayed. Unlike attention's KV cache which grows with context, RWKV must compress all history into a fixed-size state, which can lose fine-grained details from far back."
        id="warning-rwkv-state"
      />
    </div>
  )
}
