import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function LoraDeepDive() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LoRA Deep Dive: Hyperparameters and Rank Selection</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        LoRA (Low-Rank Adaptation) decomposes weight updates into low-rank matrices, drastically
        reducing trainable parameters. Understanding its hyperparameters -- rank, alpha, target
        modules, and dropout -- is critical for getting good results.
      </p>

      <DefinitionBlock
        title="LoRA Decomposition"
        definition="For a pretrained weight matrix $W_0 \\in \\mathbb{R}^{d \\times k}$, LoRA adds a low-rank update: $W = W_0 + \\frac{\\alpha}{r} B A$ where $B \\in \\mathbb{R}^{d \\times r}$, $A \\in \\mathbb{R}^{r \\times k}$, and $r \\ll \\min(d, k)$. Matrix $A$ is initialized with random Gaussian values and $B$ is initialized to zero, so the update starts at zero."
        notation="W = W_0 + \frac{\alpha}{r} BA"
        id="def-lora"
      />

      <TheoremBlock
        title="LoRA Parameter Count"
        statement="For a single linear layer of shape $d \times k$ with LoRA rank $r$, the number of trainable parameters is $r(d + k)$, compared to $dk$ for full finetuning."
        proof="Matrix A has shape $r \times k$ (rk parameters) and matrix B has shape $d \times r$ (dr parameters). Total: $dr + rk = r(d+k)$."
        id="thm-lora-params"
      />

      <h2 className="text-2xl font-semibold">Key Hyperparameters</h2>

      <ExampleBlock
        title="Rank Selection Guidelines"
        problem="How to choose the LoRA rank r for different scenarios?"
        steps={[
          { formula: 'r = 8', explanation: 'Good starting point for simple tasks (classification, style transfer). Very memory efficient.' },
          { formula: 'r = 16\\text{-}32', explanation: 'Recommended default for instruction tuning and chat finetuning. Good quality-efficiency balance.' },
          { formula: 'r = 64\\text{-}128', explanation: 'For complex domain adaptation or when you have large datasets (50K+ examples).' },
          { formula: 'r = 256', explanation: 'Approaches full finetuning expressiveness. Rarely needed; consider full FT at this point.' },
        ]}
        id="example-rank-selection"
      />

      <PythonCode
        title="lora_hyperparameters.py"
        code={`from peft import LoraConfig, get_peft_model

# Recommended configuration for instruction tuning
lora_config = LoraConfig(
    r=16,                    # Rank: controls capacity of adaptation
    lora_alpha=32,           # Scaling: effective lr multiplier = alpha/r = 2
    lora_dropout=0.05,       # Regularization: 0.05-0.1 for small datasets
    target_modules=[         # Which modules to apply LoRA to
        "q_proj",            # Query projection (attention)
        "k_proj",            # Key projection (attention)
        "v_proj",            # Value projection (attention)
        "o_proj",            # Output projection (attention)
        "gate_proj",         # MLP gate projection
        "up_proj",           # MLP up projection
        "down_proj",         # MLP down projection
    ],
    task_type="CAUSAL_LM",
    bias="none",             # Don't train biases (saves memory)
)

# Apply and inspect
# model_lora = get_peft_model(model, lora_config)
# model_lora.print_trainable_parameters()

# --- Experiment with different ranks ---
import torch

d, k = 4096, 4096  # Typical hidden dim for 7B model
for r in [8, 16, 32, 64, 128, 256]:
    lora_params = r * (d + k)
    full_params = d * k
    ratio = lora_params / full_params * 100
    print(f"r={r:>3d}: {lora_params:>10,} params ({ratio:.2f}% of full)")

# r=  8:      65,536 params (0.39% of full)
# r= 16:     131,072 params (0.78% of full)
# r= 32:     262,144 params (1.56% of full)
# r= 64:     524,288 params (3.12% of full)
# r=128:   1,048,576 params (6.25% of full)
# r=256:   2,097,152 params (12.50% of full)`}
        id="code-lora-hyperparams"
      />

      <h2 className="text-2xl font-semibold">The Alpha/Rank Ratio</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The effective scaling factor is <InlineMath math="\frac{\alpha}{r}" />. When you change the
        rank, keep <InlineMath math="\frac{\alpha}{r}" /> constant (typically 1 or 2) to maintain
        similar learning dynamics. For example, if <InlineMath math="r=16, \alpha=32" />, then
        for <InlineMath math="r=64" />, use <InlineMath math="\alpha=128" />.
      </p>

      <PythonCode
        title="alpha_rank_experiment.py"
        code={`# Impact of alpha/r ratio on training dynamics
import numpy as np

def simulate_lora_update(d, r, alpha, lr=1e-4):
    """Simulate the magnitude of a LoRA weight update."""
    # Random initialization (simplified)
    A = np.random.randn(r, d) * 0.01  # Small init
    B = np.zeros((d, r))               # Zero init

    # After one gradient step (simplified)
    grad_A = np.random.randn(r, d) * 0.1
    grad_B = np.random.randn(d, r) * 0.1
    A -= lr * grad_A
    B -= lr * grad_B

    # Effective weight update
    delta_W = (alpha / r) * (B @ A)
    return np.linalg.norm(delta_W)

d = 4096
for r in [8, 16, 32, 64]:
    # Fixed alpha
    norm_fixed = simulate_lora_update(d, r, alpha=16)
    # Scaled alpha (alpha/r = 2)
    norm_scaled = simulate_lora_update(d, r, alpha=2*r)
    print(f"r={r:>2d}: fixed alpha=16 -> |dW|={norm_fixed:.4f}, "
          f"scaled alpha={2*r} -> |dW|={norm_scaled:.4f}")`}
        id="code-alpha-experiment"
      />

      <NoteBlock
        type="tip"
        title="Target Module Selection"
        content="Targeting all linear layers (q, k, v, o, gate, up, down) gives better results than only q_proj and v_proj, at the cost of more trainable parameters. For 7B models with r=16, targeting all 7 modules adds ~42M trainable params (~0.5%). This is the recommended default."
        id="note-target-modules"
      />

      <WarningBlock
        title="Overfitting with High Rank"
        content="Higher rank does not always mean better results. With small datasets (<1000 examples), rank 8-16 with dropout 0.1 often outperforms rank 64+ which may overfit. Monitor validation loss and use early stopping."
        id="warning-overfit"
      />
    </div>
  )
}
