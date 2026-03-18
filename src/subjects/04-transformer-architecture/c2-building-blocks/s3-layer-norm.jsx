import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LayerNorm() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Layer Normalization vs. Batch Normalization</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Normalization stabilizes training by controlling the scale of activations. Transformers
        universally use Layer Normalization rather than Batch Normalization because it normalizes
        across features independently per sample, making it compatible with variable-length
        sequences and small batch sizes.
      </p>

      <DefinitionBlock
        title="Layer Normalization"
        definition="For an input vector $x \in \mathbb{R}^d$: $\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ where $\mu = \frac{1}{d}\sum_{i=1}^d x_i$, $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$, and $\gamma, \beta \in \mathbb{R}^d$ are learned parameters."
        notation="μ = mean across features, σ² = variance across features, γ = scale, β = shift, ε ≈ 1e-5"
        id="def-layernorm"
      />

      <h2 className="text-2xl font-semibold">Layer Norm vs. Batch Norm</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Batch Normalization computes statistics across the batch dimension for each feature,
        while Layer Normalization computes statistics across the feature dimension for each sample.
        This distinction is critical for sequence models.
      </p>

      <ExampleBlock
        title="Normalization Axis Comparison"
        problem="For a tensor of shape (batch=4, seq_len=10, d_model=512), compare which dimensions BatchNorm and LayerNorm normalize over."
        steps={[
          { formula: '\\text{BatchNorm: mean over } (4, 10) \\text{ for each of 512 features}', explanation: 'Computes one mean and variance per feature across all samples and positions.' },
          { formula: '\\text{LayerNorm: mean over } (512,) \\text{ for each position}', explanation: 'Computes one mean and variance per (sample, position) pair across all 512 features.' },
          { formula: '\\text{BatchNorm stats: 512 means, 512 variances}', explanation: 'Statistics depend on batch — problematic at inference with batch size 1.' },
          { formula: '\\text{LayerNorm stats: 4 \\times 10 = 40 means, 40 variances}', explanation: 'Statistics are independent of batch size, computed per token.' },
        ]}
        id="example-norm-axes"
      />

      <PythonCode
        title="layer_norm_comparison.py"
        code={`import torch
import torch.nn as nn

batch, seq_len, d_model = 4, 10, 512
x = torch.randn(batch, seq_len, d_model) * 5 + 3  # Shifted and scaled

# --- Layer Normalization (standard in transformers) ---
layer_norm = nn.LayerNorm(d_model)
ln_out = layer_norm(x)

# Check: each position has mean≈0, std≈1
print("LayerNorm per-position stats:")
print(f"  Mean: {ln_out[0, 0].mean():.4f}")  # ≈ 0
print(f"  Std:  {ln_out[0, 0].std():.4f}")   # ≈ 1

# --- RMSNorm (used in LLaMA, Gemma) ---
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019)."""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight

rms_norm = RMSNorm(d_model)
rms_out = rms_norm(x)
print(f"\\nRMSNorm output shape: {rms_out.shape}")
print(f"  RMS of output: {rms_out[0, 0].pow(2).mean().sqrt():.4f}")  # ≈ 1

# --- Pre-Norm vs Post-Norm placement ---
class PreNormBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, d_model)

    def forward(self, x):
        return x + self.ff(self.norm(x))  # Norm BEFORE sublayer

class PostNormBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.norm(x + self.ff(x))  # Norm AFTER sublayer

print(f"\\nPreNorm output:  {PreNormBlock(d_model)(x).shape}")
print(f"PostNorm output: {PostNormBlock(d_model)(x).shape}")`}
        id="code-layernorm"
      />

      <NoteBlock
        type="note"
        title="Pre-Norm vs. Post-Norm"
        content="The original transformer uses Post-Norm (normalize after the residual addition). GPT-2 and most modern models switched to Pre-Norm (normalize before the sublayer), which makes training more stable and eliminates the need for learning rate warmup. Pre-Norm ensures the residual path has an unimpeded gradient highway."
        id="note-pre-post"
      />

      <WarningBlock
        title="RMSNorm Drops the Mean Centering"
        content="RMSNorm (used in LLaMA, Gemma, Mistral) simplifies LayerNorm by removing the mean subtraction and learned bias. This saves compute and works well in practice, but the output is not zero-centered. When porting weights between architectures, do not interchange LayerNorm and RMSNorm without retraining."
        id="warning-rmsnorm"
      />

      <NoteBlock
        type="historical"
        title="Evolution of Normalization"
        content="BatchNorm (Ioffe & Szegedy, 2015) was designed for CNNs. LayerNorm (Ba et al., 2016) was proposed specifically for RNNs and later adopted by transformers. RMSNorm (Zhang & Sennrich, 2019) simplified LayerNorm with minimal quality loss. Modern LLMs almost universally use Pre-RMSNorm."
        id="note-history"
      />
    </div>
  )
}
