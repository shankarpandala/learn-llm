import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function GptqAwq() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">GPTQ and AWQ: Advanced Weight Quantization</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        GPTQ and AWQ are state-of-the-art post-training quantization methods specifically
        designed for large language models. They achieve INT4/INT3 quantization with minimal
        accuracy loss by solving layer-wise optimization problems using calibration data.
      </p>

      <DefinitionBlock
        title="GPTQ (Generative Pre-Trained Quantization)"
        definition="GPTQ quantizes weights column-by-column, minimizing the squared error $\| W X - \hat{W} X \|_2^2$ where $X$ is the layer input from calibration data. It uses the inverse Hessian $H^{-1} = (X X^\top)^{-1}$ to optimally adjust remaining weights after quantizing each column, compensating for quantization error."
        notation="For each column $j$: quantize $w_j$, compute error $\delta_j = (w_j - \hat{w}_j) / [H^{-1}]_{jj}$, update remaining weights: $W_{:, j+1:} \leftarrow W_{:, j+1:} - \delta_j \cdot H^{-1}_{j, j+1:} / H^{-1}_{jj}$."
        id="def-gptq"
      />

      <DefinitionBlock
        title="AWQ (Activation-Aware Weight Quantization)"
        definition="AWQ observes that a small fraction of weights (0.1-1%) are critical because they correspond to large activation magnitudes. Instead of mixed-precision, AWQ applies per-channel scaling: $s_j = \left(\frac{\max|X_j|}{q_{\max}}\right)^\alpha$ to protect salient channels before quantization, where $\alpha \in [0, 1]$ balances protection and quantization range."
        id="def-awq"
      />

      <ExampleBlock
        title="GPTQ Column-wise Quantization"
        problem="Quantize a 2x3 weight matrix W using GPTQ with calibration data."
        steps={[
          {
            formula: 'H = X X^\\top + \\lambda I \\quad \\text{(Hessian with damping)}',
            explanation: 'Compute the Hessian from calibration inputs. Damping prevents numerical issues.'
          },
          {
            formula: '\\text{Quantize column 0: } \\hat{w}_0 = Q(w_0)',
            explanation: 'Quantize the first column using round-to-nearest.'
          },
          {
            formula: '\\delta_0 = w_0 - \\hat{w}_0 \\quad \\text{(quantization error)}',
            explanation: 'Measure the error introduced by quantizing column 0.'
          },
          {
            formula: 'W_{:, 1:} \\leftarrow W_{:, 1:} - \\delta_0 \\cdot (H^{-1}_{0, 1:} / H^{-1}_{00})',
            explanation: 'Compensate: adjust remaining columns to minimize overall output error.'
          }
        ]}
        id="example-gptq"
      />

      <PythonCode
        title="gptq_awq_quantization.py"
        code={`import torch
import torch.nn.functional as F

def gptq_quantize(W, X, n_bits=4, block_size=128):
    """Simplified GPTQ quantization of weight matrix W given input X."""
    rows, cols = W.shape
    qmin, qmax = -(2**(n_bits-1)), 2**(n_bits-1) - 1

    # Compute Hessian
    H = X @ X.T / X.shape[1]
    H += 1e-4 * torch.eye(cols)  # Damping
    H_inv = torch.linalg.inv(H)

    W_q = W.clone()
    scales = torch.zeros(rows)

    # Process in blocks for efficiency
    for j in range(0, cols, block_size):
        end = min(j + block_size, cols)
        for k in range(j, end):
            # Quantize column k
            w_col = W_q[:, k]
            scale = w_col.abs().max() / qmax
            scales_col = scale
            w_int = torch.clamp(torch.round(w_col / scale), qmin, qmax)
            w_hat = w_int * scale

            # Error compensation for remaining columns
            error = (w_col - w_hat) / H_inv[k, k]
            W_q[:, k] = w_hat
            if k + 1 < cols:
                W_q[:, k+1:] -= error.unsqueeze(1) * H_inv[k, k+1:].unsqueeze(0)

    return W_q

def awq_quantize(W, X, n_bits=4, alpha=0.5):
    """Simplified AWQ: scale salient channels before quantization."""
    qmax = 2**(n_bits-1) - 1

    # Find salient channels based on activation magnitudes
    act_scales = X.abs().mean(dim=1)  # Per-channel activation magnitude
    s = (act_scales / act_scales.max()).pow(alpha)
    s = s.clamp(min=1e-5)

    # Scale weights to protect salient channels
    W_scaled = W * s.unsqueeze(0)

    # Standard quantization on scaled weights
    scale = W_scaled.abs().max(dim=0).values / qmax
    W_int = torch.clamp(torch.round(W_scaled / scale), -qmax, qmax)
    W_hat = (W_int * scale) / s.unsqueeze(0)  # Undo scaling

    return W_hat

# Compare methods
torch.manual_seed(42)
W = torch.randn(256, 256) * 0.02
X = torch.randn(256, 1024) * 0.5  # Calibration activations

# Naive round-to-nearest
scale_naive = W.abs().max() / 7
W_naive = torch.clamp(torch.round(W / scale_naive), -8, 7) * scale_naive

# GPTQ
W_gptq = gptq_quantize(W, X, n_bits=4)

# AWQ
W_awq = awq_quantize(W, X, n_bits=4)

# Measure output error (what matters for accuracy)
Y_true = W @ X
for name, W_q in [("Naive", W_naive), ("GPTQ", W_gptq), ("AWQ", W_awq)]:
    Y_q = W_q @ X
    mse = ((Y_true - Y_q) ** 2).mean().item()
    print(f"{name:>6} INT4 output MSE: {mse:.6f}")`}
        id="code-gptq-awq"
      />

      <TheoremBlock
        title="Optimality of GPTQ Error Compensation"
        statement="GPTQ's column-wise update rule minimizes the layer output error $\|WX - \hat{W}X\|_F^2$ greedily. The compensation step is the optimal least-squares adjustment for remaining weights given the Hessian of the layer output."
        proof="After quantizing column $j$, the residual error is $\delta_j \cdot X_j$. The optimal adjustment to column $k > j$ minimizes $\|\delta_j X_j - \Delta w_k X_k\|^2$, giving $\Delta w_k = \delta_j \cdot \text{Cov}(X_j, X_k) / \text{Var}(X_k) = \delta_j \cdot H^{-1}_{jk} / H^{-1}_{jj}$."
        id="thm-gptq-optimality"
      />

      <NoteBlock
        type="tip"
        title="Practical Quantization Choice"
        content="GPTQ is better when you need the absolute best INT4 quality and have calibration data. AWQ is faster to apply and more robust across different input distributions. Both methods quantize a 70B model in under an hour on a single GPU, producing models that fit in 35GB at INT4."
        id="note-practical-choice"
      />

      <WarningBlock
        title="Calibration Data Sensitivity"
        content="Both GPTQ and AWQ assume the calibration data is representative. If calibration uses English Wikipedia but deployment involves code generation, the quantization parameters may be suboptimal. Use diverse, domain-appropriate calibration sets for best results."
        id="warning-calibration-sensitivity"
      />
    </div>
  )
}
