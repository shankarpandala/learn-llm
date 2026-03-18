import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function QuantizationAwareTraining() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Quantization-Aware Training (QAT)</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Quantization-aware training simulates quantization during the forward pass while
        maintaining full-precision weights for gradient updates. This allows the model to
        adapt to quantization noise during training, achieving significantly better accuracy
        at low bit-widths than post-training quantization alone.
      </p>

      <DefinitionBlock
        title="Quantization-Aware Training"
        definition="QAT inserts fake quantization operators into the computation graph: $\hat{w} = s \cdot \text{clamp}\left(\left\lfloor w/s \right\rceil, -2^{b-1}, 2^{b-1}-1\right)$. The forward pass uses quantized weights $\hat{w}$, but backpropagation updates the full-precision weights $w$ using the Straight-Through Estimator (STE)."
        notation="STE: $\frac{\partial \hat{w}}{\partial w} \approx 1$ when $w$ is within the clamp range, $0$ otherwise. This allows gradients to flow through the non-differentiable rounding operation."
        id="def-qat"
      />

      <ExampleBlock
        title="STE Gradient Flow"
        problem="Show how gradients pass through quantization using the Straight-Through Estimator."
        steps={[
          {
            formula: '\\text{Forward: } \\hat{w} = Q(w) = s \\cdot \\text{round}(w/s)',
            explanation: 'Quantize weight w to discrete value using scale s.'
          },
          {
            formula: '\\frac{\\partial Q}{\\partial w} = 0 \\text{ (true gradient of rounding)}',
            explanation: 'The rounding function has zero gradient almost everywhere — training would stall.'
          },
          {
            formula: '\\text{STE: } \\frac{\\partial Q}{\\partial w} \\approx \\mathbb{1}_{|w| \\leq c}',
            explanation: 'STE approximates the gradient as 1 within the clamp range, enabling learning.'
          },
          {
            formula: 'w \\leftarrow w - \\eta \\cdot \\frac{\\partial L}{\\partial \\hat{w}}',
            explanation: 'Full-precision weights are updated using gradients computed at quantized values.'
          }
        ]}
        id="example-ste"
      />

      <TheoremBlock
        title="QAT vs PTQ Accuracy Gap"
        statement="For a model with quantization noise $\epsilon_q \sim \mathcal{U}(-s/2, s/2)$ per weight, QAT reduces the effective noise by learning weight distributions that are robust to discretization. Empirically, QAT at INT4 achieves accuracy comparable to PTQ at INT8."
        proof="QAT optimizes the loss landscape including quantization noise: $\min_w L(Q(w); \mathcal{D})$. The model learns to place weights near quantization grid points and to reduce sensitivity to quantization in critical layers. This is analogous to training with weight noise regularization."
        id="thm-qat-vs-ptq"
      />

      <PythonCode
        title="quantization_aware_training.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class FakeQuantize(torch.autograd.Function):
    """Simulates quantization in forward, passes gradients via STE."""
    @staticmethod
    def forward(ctx, x, n_bits=8):
        qmin = -(2 ** (n_bits - 1))
        qmax = 2 ** (n_bits - 1) - 1
        scale = x.abs().max() / qmax
        # Quantize and dequantize
        x_q = torch.clamp(torch.round(x / scale), qmin, qmax)
        x_hat = x_q * scale
        # Save mask for STE
        ctx.save_for_backward((x >= qmin * scale) & (x <= qmax * scale))
        return x_hat

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        # STE: pass gradient through where within clamp range
        return grad_output * mask.float(), None

fake_quantize = FakeQuantize.apply

class QATLinear(nn.Module):
    """Linear layer with fake quantization for QAT."""
    def __init__(self, in_features, out_features, n_bits=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.n_bits = n_bits

    def forward(self, x):
        # Quantize weights during forward pass
        w_q = fake_quantize(self.linear.weight, self.n_bits)
        return F.linear(x, w_q, self.linear.bias)

# Training loop comparison: QAT vs standard
model_qat = nn.Sequential(
    QATLinear(784, 256, n_bits=4),
    nn.ReLU(),
    QATLinear(256, 10, n_bits=4),
)

# Simulate one training step
x = torch.randn(32, 784)
y = torch.randint(0, 10, (32,))
optimizer = torch.optim.Adam(model_qat.parameters(), lr=1e-3)

logits = model_qat(x)
loss = F.cross_entropy(logits, y)
loss.backward()
optimizer.step()
print(f"QAT training loss: {loss.item():.4f}")

# After training, export to actual INT4 weights
for name, module in model_qat.named_modules():
    if isinstance(module, QATLinear):
        w = module.linear.weight.data
        scale = w.abs().max() / 7  # INT4: -8 to 7
        w_int = torch.clamp(torch.round(w / scale), -8, 7).to(torch.int8)
        print(f"{name}: scale={scale:.6f}, unique values={w_int.unique().numel()}")`}
        id="code-qat"
      />

      <NoteBlock
        type="intuition"
        title="Why QAT Outperforms PTQ"
        content="Think of PTQ as forcing a dancer to perform in a straitjacket after learning to dance freely. QAT is like training the dancer in the straitjacket from the start — they learn movements that work within the constraints. The model learns weight distributions that are naturally quantization-friendly."
        id="note-qat-intuition"
      />

      <NoteBlock
        type="tip"
        title="QAT for LLMs in Practice"
        content="Full QAT of a 70B model is prohibitively expensive. Practical approaches include: (1) QLoRA-style QAT that only trains adapters, (2) applying QAT only to the most quantization-sensitive layers, and (3) short QAT fine-tuning (a few hundred steps) after initial PTQ to recover lost accuracy."
        id="note-qat-practical"
      />

      <WarningBlock
        title="STE Approximation Degrades at Low Bits"
        content="The STE becomes a poor gradient approximation below 4 bits because the rounding error is too large relative to the step size. At 2-bit and below, more sophisticated gradient estimators or alternative training strategies (like learned step sizes via LSQ) are needed."
        id="warning-ste-limits"
      />
    </div>
  )
}
