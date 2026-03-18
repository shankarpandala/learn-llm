import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function ResidualConnections() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Residual Connections and Gradient Flow</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Residual (skip) connections are essential for training deep transformers. By adding the
        input of each sublayer directly to its output, they create a gradient highway that
        prevents vanishing gradients and allows information to flow unchanged through many layers.
      </p>

      <DefinitionBlock
        title="Residual Connection"
        definition="Given a sublayer function $F(x)$, a residual connection computes $\text{output} = x + F(x)$. In a transformer layer with Pre-Norm: $\text{output} = x + \text{Attention}(\text{LayerNorm}(x))$."
        notation="x = input, F(x) = sublayer transformation (attention or FFN)"
        id="def-residual"
      />

      <TheoremBlock
        title="Gradient Flow Through Residual Connections"
        statement="For a network with L residual blocks, the gradient of the loss with respect to an early layer's output $x_l$ includes a direct identity term: $\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \left(1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1} F_i(x_i)\right)$. The '1' term ensures gradients can flow directly from the loss to any layer."
        id="thm-gradient-flow"
      />

      <h2 className="text-2xl font-semibold">Why Residuals Matter</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Without residual connections, gradients must pass through every layer's nonlinearity.
        In a 96-layer GPT-3, this would cause severe vanishing gradients. The skip connection
        provides a direct path, ensuring the gradient magnitude stays bounded.
      </p>

      <ExampleBlock
        title="Gradient Magnitude With and Without Residuals"
        problem="Compare gradient norms after 10 layers with and without residual connections."
        steps={[
          { formula: '\\text{Without: } \\|\\nabla\\| \\approx \\prod_{i=1}^{10} \\|J_i\\|', explanation: 'Gradient is a product of Jacobians — exponential decay if ||J_i|| < 1.' },
          { formula: '\\text{If } \\|J_i\\| \\approx 0.9: \\; 0.9^{10} \\approx 0.35', explanation: '65% gradient signal lost after just 10 layers.' },
          { formula: '\\text{With residuals: } \\nabla = I + \\text{higher-order terms}', explanation: 'The identity component ensures gradient magnitude ≥ 1.' },
          { formula: '\\text{At 96 layers: } 0.9^{96} \\approx 0.00003 \\text{ vs. } \\approx 1.0', explanation: 'Residual connections prevent catastrophic gradient collapse.' },
        ]}
        id="example-gradient-decay"
      />

      <PythonCode
        title="residual_connections.py"
        code={`import torch
import torch.nn as nn

class TransformerLayerPreNorm(nn.Module):
    """Single transformer layer with Pre-Norm residual connections."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Residual 1: x + Attention(LayerNorm(x))
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_out)

        # Residual 2: x + FFN(LayerNorm(x))
        x = x + self.ffn(self.norm2(x))
        return x

# Build a deep model and verify gradient flow
d_model, num_heads, d_ff, num_layers = 256, 8, 1024, 12
layers = nn.ModuleList([
    TransformerLayerPreNorm(d_model, num_heads, d_ff)
    for _ in range(num_layers)
])
final_norm = nn.LayerNorm(d_model)

x = torch.randn(1, 20, d_model, requires_grad=True)
h = x
for layer in layers:
    h = layer(h)
h = final_norm(h)

loss = h.sum()
loss.backward()

print(f"Input gradient norm: {x.grad.norm():.4f}")
print(f"Input gradient mean: {x.grad.mean():.6f}")
print(f"Gradient is finite: {x.grad.isfinite().all()}")

# Compare: without residuals, gradient would vanish
class NoResidualLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
    def forward(self, x):
        return torch.tanh(self.linear(x))  # No skip connection

no_res = nn.Sequential(*[NoResidualLayer(d_model) for _ in range(num_layers)])
x2 = torch.randn(1, 20, d_model, requires_grad=True)
loss2 = no_res(x2).sum()
loss2.backward()
print(f"\\nNo-residual gradient norm: {x2.grad.norm():.6f}")  # Much smaller`}
        id="code-residual"
      />

      <NoteBlock
        type="intuition"
        title="Residuals as an Ensemble"
        content="Veit et al. (2016) showed that residual networks can be viewed as an ensemble of many shallow networks. Each layer adds a small refinement to the representation. Deleting a single layer has minimal impact, unlike in a plain network where it would be catastrophic."
        id="note-ensemble"
      />

      <WarningBlock
        title="Residual Connection Scale"
        content="In very deep transformers (100+ layers), even with residual connections, the activations can grow unboundedly because each layer adds to the residual stream. Techniques like DeepNorm (scaling the residual by a factor α > 1) or fixup initialization address this by carefully controlling the scale of each sublayer's contribution."
        id="warning-scale"
      />
    </div>
  )
}
