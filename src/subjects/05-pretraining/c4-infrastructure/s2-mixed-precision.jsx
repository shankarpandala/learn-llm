import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function MixedPrecision() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Mixed Precision Training: FP16 and BF16</h1>
      <p className="text-lg text-gray-300">
        Mixed precision training uses lower-precision floating-point formats (FP16 or BF16)
        for most operations while maintaining FP32 for critical accumulations. This halves
        memory usage and dramatically increases throughput on modern GPUs with tensor cores.
      </p>

      <DefinitionBlock
        title="Floating Point Formats"
        definition="FP32 uses 1 sign, 8 exponent, 23 mantissa bits (range $\pm 3.4 \times 10^{38}$). FP16 uses 1 sign, 5 exponent, 10 mantissa bits (range $\pm 65504$). BF16 uses 1 sign, 8 exponent, 7 mantissa bits (range $\pm 3.4 \times 10^{38}$, same as FP32)."
        notation="BF16 has the same exponent range as FP32 but less precision. FP16 has more precision but much smaller range. BF16 is preferred for LLM training due to reduced overflow risk."
        id="fp-formats-def"
      />

      <ExampleBlock
        title="Memory Savings with Mixed Precision"
        problem="Calculate memory savings for a 7B parameter model with mixed precision."
        steps={[
          {
            formula: '\\text{FP32 model: } 7\\text{B} \\times 4 \\text{ bytes} = 28 \\text{ GB}',
            explanation: 'Full precision model weights alone take 28 GB.'
          },
          {
            formula: '\\text{FP16/BF16 model: } 7\\text{B} \\times 2 \\text{ bytes} = 14 \\text{ GB}',
            explanation: 'Half precision cuts model weight memory in half.'
          },
          {
            formula: '\\text{Master weights (FP32): } 28 \\text{ GB (kept for optimizer step)}',
            explanation: 'FP32 master copy maintained for accurate parameter updates.'
          },
          {
            formula: '\\text{Total: } 14 \\text{ (forward/backward)} + 28 \\text{ (master)} = 42 \\text{ GB vs } 28 \\text{ GB (FP32 only)}',
            explanation: 'But activations and gradients are also in FP16, giving net savings plus 2x faster matmuls on tensor cores.'
          }
        ]}
        id="memory-savings-example"
      />

      <TheoremBlock
        title="Loss Scaling for FP16"
        statement="FP16 has minimum subnormal value $\approx 5.96 \times 10^{-8}$. Gradients smaller than this underflow to zero. Loss scaling multiplies the loss by a scale factor $S$ before backpropagation: $\tilde{g} = S \cdot \nabla_\theta \mathcal{L}$. After backprop, gradients are divided by $S$ before the optimizer step."
        proof="The chain rule preserves the scale factor: $\frac{\partial (S \cdot \mathcal{L})}{\partial \theta} = S \cdot \frac{\partial \mathcal{L}}{\partial \theta}$. By scaling up, small gradients move into FP16 representable range. Dynamic loss scaling starts with large $S$ and halves it when overflow (NaN/Inf) is detected, doubles it after $N$ successful steps."
        id="loss-scaling-thm"
      />

      <NoteBlock
        type="tip"
        title="BF16 Simplifies Training"
        content="BF16 (Brain Float 16) shares FP32's exponent range, so loss scaling is typically unnecessary. This eliminates the complexity of dynamic loss scaling and the risk of NaN gradients from overflow. BF16 is the default for modern LLM training on A100, H100, and newer GPUs. The trade-off is slightly less mantissa precision than FP16 (7 vs 10 bits)."
        id="bf16-note"
      />

      <PythonCode
        title="mixed_precision.py"
        code={`import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# Demonstrate precision formats
def show_precision():
    x = torch.tensor(3.141592653589793)
    print(f"FP32: {x.float().item():.10f}")
    print(f"FP16: {x.half().float().item():.10f}")
    print(f"BF16: {x.bfloat16().float().item():.10f}")

    # Overflow example
    big = torch.tensor(70000.0)
    print(f"\\nFP16 max: {torch.finfo(torch.float16).max}")
    print(f"70000 in FP16: {big.half()}")  # inf!
    print(f"70000 in BF16: {big.bfloat16()}")  # fine

    # Small gradient example
    tiny = torch.tensor(1e-8)
    print(f"\\n1e-8 in FP16: {tiny.half()}")  # 0!
    print(f"1e-8 in BF16: {tiny.bfloat16()}")
    scaled = tiny * 1024  # Loss scaling
    print(f"1e-8 * 1024 in FP16: {scaled.half()}")

show_precision()

# Mixed precision training with PyTorch AMP
class SimpleModel(nn.Module):
    def __init__(self, d=1024):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d),
            nn.LayerNorm(d),
        )

    def forward(self, x):
        return self.layers(x)

def train_mixed_precision():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()  # Dynamic loss scaling for FP16

    x = torch.randn(32, 1024, device=device)
    target = torch.randn(32, 1024, device=device)

    for step in range(5):
        optimizer.zero_grad()

        # Forward pass in FP16/BF16
        with autocast(device_type=device, dtype=torch.float16):
            output = model(x)
            loss = nn.functional.mse_loss(output, target)

        # Backward with loss scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # Unscale before grad clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        print(f"Step {step}: loss={loss.item():.4f}, scale={scaler.get_scale()}")

# HuggingFace Trainer with mixed precision
training_args_example = {
    "fp16": False,           # Set True for FP16
    "bf16": True,            # Preferred on A100/H100
    "bf16_full_eval": True,
    "tf32": True,            # TF32 for FP32 ops on Ampere+
    "half_precision_backend": "auto",
}
print(f"\\nRecommended HF training args: {training_args_example}")

if torch.cuda.is_available():
    train_mixed_precision()
else:
    print("\\n(CUDA required for mixed precision demo)")`}
        id="mixed-precision-code"
      />

      <WarningBlock
        title="Numerical Instability with FP16"
        content="FP16 training can suffer from overflow in attention logits (softmax of large values), underflow in gradients, and loss of precision in layer norm statistics. Always use loss scaling with FP16. Consider BF16 if your hardware supports it. For very large models, some operations (softmax, layer norm, loss computation) should remain in FP32 even in mixed precision mode."
        id="fp16-instability-warning"
      />
    </div>
  )
}
