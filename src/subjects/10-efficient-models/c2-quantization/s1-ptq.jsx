import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function PostTrainingQuantization() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Post-Training Quantization (PTQ)</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Post-training quantization converts a pre-trained model's weights and activations from
        high-precision floating point (FP16/FP32) to lower-bit integer representations (INT8/INT4)
        without retraining. This is the fastest path to model compression, reducing memory by
        2-4x and enabling inference on consumer hardware.
      </p>

      <DefinitionBlock
        title="Uniform Affine Quantization"
        definition="Quantization maps a floating-point value $x$ to an integer $x_q$ using scale $s$ and zero-point $z$: $x_q = \text{clamp}\left(\left\lfloor \frac{x}{s} \right\rceil + z, \; 0, \; 2^b - 1\right)$ where $b$ is the bit-width, $s = \frac{x_{\max} - x_{\min}}{2^b - 1}$, and $z = \text{round}(-x_{\min}/s)$."
        notation="Dequantization recovers the approximate value: $\hat{x} = s \cdot (x_q - z)$. The quantization error is $|x - \hat{x}| \leq s/2$."
        id="def-ptq"
      />

      <ExampleBlock
        title="INT8 Quantization of a Weight Tensor"
        problem="Quantize weights [-0.8, -0.2, 0.0, 0.3, 1.2] to INT8 (0-255 range)."
        steps={[
          {
            formula: 's = \\frac{1.2 - (-0.8)}{255} = \\frac{2.0}{255} \\approx 0.00784',
            explanation: 'Scale maps the full float range to 256 integer levels.'
          },
          {
            formula: 'z = \\text{round}\\left(\\frac{0.8}{0.00784}\\right) = \\text{round}(102.04) = 102',
            explanation: 'Zero-point ensures 0.0 maps to an exact integer.'
          },
          {
            formula: 'x_q = [0, 77, 102, 140, 255]',
            explanation: 'Each value: round(x / 0.00784) + 102, clamped to [0, 255].'
          },
          {
            formula: '\\hat{x} = [-0.800, -0.196, 0.000, 0.298, 1.200]',
            explanation: 'Dequantized values. Maximum error is ~0.004 — negligible for neural networks.'
          }
        ]}
        id="example-int8-quant"
      />

      <TheoremBlock
        title="Model Size Under Quantization"
        statement="A model with $N$ parameters stored at $b$ bits per parameter occupies $\frac{N \cdot b}{8}$ bytes. Quantizing from FP16 (16 bits) to INT4 (4 bits) yields a $4\times$ size reduction."
        proof="FP16 model: $N \times 16 / 8 = 2N$ bytes. INT4 model: $N \times 4 / 8 = N/2$ bytes. Ratio: $2N / (N/2) = 4\times$. A 7B-parameter model goes from 14 GB (FP16) to 3.5 GB (INT4)."
        corollaries={[
          'A 70B model at INT4 (35 GB) fits on a single 48GB GPU.',
          'INT8 quantization halves memory: 7B model goes from 14 GB to 7 GB.'
        ]}
        id="thm-model-size"
      />

      <PythonCode
        title="post_training_quantization.py"
        code={`import torch
import numpy as np

def quantize_tensor(tensor, n_bits=8):
    """Symmetric quantization of a float tensor to n_bits."""
    qmin = -(2 ** (n_bits - 1))
    qmax = 2 ** (n_bits - 1) - 1

    # Compute scale from max absolute value
    abs_max = tensor.abs().max()
    scale = abs_max / qmax

    # Quantize
    quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax).to(torch.int8)

    return quantized, scale

def dequantize_tensor(quantized, scale):
    """Recover float values from quantized tensor."""
    return quantized.float() * scale

# Example: quantize a weight matrix
torch.manual_seed(42)
W = torch.randn(4096, 4096) * 0.02  # Typical LLM weight scale

W_q, scale = quantize_tensor(W, n_bits=8)
W_hat = dequantize_tensor(W_q, scale)

# Measure quantization error
mse = ((W - W_hat) ** 2).mean().item()
rel_error = (W - W_hat).abs().mean().item() / W.abs().mean().item()
print(f"Scale: {scale:.6f}")
print(f"MSE: {mse:.2e}")
print(f"Relative error: {rel_error:.4%}")

# Memory comparison
fp16_bytes = W.numel() * 2
int8_bytes = W_q.numel() * 1 + 4  # +4 bytes for scale
int4_bytes = W_q.numel() // 2 + 4  # 4 bits per weight
print(f"\\nFP16: {fp16_bytes / 1e6:.1f} MB")
print(f"INT8: {int8_bytes / 1e6:.1f} MB ({fp16_bytes/int8_bytes:.1f}x smaller)")
print(f"INT4: {int4_bytes / 1e6:.1f} MB ({fp16_bytes/int4_bytes:.1f}x smaller)")

# Full model size estimates
for params_b in [7, 13, 70]:
    fp16_gb = params_b * 2
    int8_gb = params_b * 1
    int4_gb = params_b * 0.5
    print(f"\\n{params_b}B model: FP16={fp16_gb}GB, INT8={int8_gb}GB, INT4={int4_gb}GB")`}
        id="code-ptq"
      />

      <NoteBlock
        type="tip"
        title="Calibration Data Matters"
        content="PTQ methods like GPTQ and AWQ use a small calibration dataset (128-1024 samples) to determine optimal quantization parameters per layer. Using representative data for calibration is critical — the calibration set should match your deployment distribution."
        id="note-calibration"
      />

      <WarningBlock
        title="Accuracy Degradation at Low Bits"
        content="INT8 PTQ is nearly lossless for most LLMs. At INT4, naive round-to-nearest quantization can degrade perplexity significantly (5-15% increase). This is why advanced methods like GPTQ and AWQ were developed — they minimize the layer-wise reconstruction error during quantization."
        id="warning-low-bit-accuracy"
      />
    </div>
  )
}
