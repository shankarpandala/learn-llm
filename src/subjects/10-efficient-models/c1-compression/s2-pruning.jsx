import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function Pruning() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Pruning: Magnitude and Structured Approaches</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Pruning removes redundant parameters from neural networks, reducing model size and
        computation. The key insight is that large networks are over-parameterized: many weights
        contribute negligibly to the output and can be zeroed out with minimal accuracy loss.
      </p>

      <DefinitionBlock
        title="Weight Pruning"
        definition="Pruning sets a subset of model weights to zero based on an importance criterion. Unstructured pruning removes individual weights: $w_{ij} = 0$ if $|w_{ij}| < \theta$. Structured pruning removes entire neurons, attention heads, or layers, yielding direct speedups on hardware."
        notation="Sparsity $s = \frac{\text{number of zero weights}}{\text{total weights}}$. A model with 90% sparsity has only 10% non-zero parameters."
        id="def-pruning"
      />

      <ExampleBlock
        title="Magnitude Pruning"
        problem="Given weight matrix W = [[0.8, -0.02, 0.5], [0.01, -0.7, 0.03]], prune to 50% sparsity using magnitude pruning."
        steps={[
          {
            formula: '|W| = [[0.8, 0.02, 0.5], [0.01, 0.7, 0.03]]',
            explanation: 'Compute absolute values of all weights.'
          },
          {
            formula: '\\text{sorted} = [0.01, 0.02, 0.03, 0.5, 0.7, 0.8]',
            explanation: 'Sort all magnitudes. For 50% sparsity, threshold is the 3rd value = 0.03.'
          },
          {
            formula: 'W_{\\text{pruned}} = [[0.8, 0, 0.5], [0, -0.7, 0]]',
            explanation: 'Zero out all weights with magnitude <= 0.03. Three of six weights removed.'
          }
        ]}
        id="example-magnitude-pruning"
      />

      <PythonCode
        title="pruning_methods.py"
        code={`import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Create a simple linear layer
layer = nn.Linear(768, 768, bias=False)
total_params = layer.weight.numel()
print(f"Total parameters: {total_params:,}")  # 589,824

# --- Unstructured magnitude pruning ---
prune.l1_unstructured(layer, name='weight', amount=0.5)
sparsity = (layer.weight == 0).sum().item() / total_params
print(f"Unstructured sparsity: {sparsity:.1%}")  # 50.0%

# Remove pruning reparameterization (make permanent)
prune.remove(layer, 'weight')

# --- Structured pruning: remove entire output neurons ---
layer2 = nn.Linear(768, 768, bias=False)
prune.ln_structured(layer2, name='weight', amount=0.3, n=2, dim=0)
# 30% of output neurons (rows) are zeroed out
zero_rows = (layer2.weight.sum(dim=1) == 0).sum().item()
print(f"Pruned neurons: {zero_rows}/{layer2.weight.shape[0]}")

# --- Iterative magnitude pruning (IMP) ---
# Key idea: prune, retrain, prune more, retrain...
def iterative_prune(model, target_sparsity=0.9, steps=10):
    """Gradually increase sparsity over multiple rounds."""
    per_step = 1 - (1 - target_sparsity) ** (1 / steps)
    for step in range(steps):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, 'weight', amount=per_step)
        # In practice: retrain for several epochs here
        current = sum((p == 0).sum().item() for p in model.parameters())
        total = sum(p.numel() for p in model.parameters())
        print(f"Step {step+1}: sparsity = {current/total:.1%}")
    return model

# Model size calculation at different sparsity levels
base_size_gb = 7.0  # 7B params at FP16 = ~14GB, but ~7GB with sparse storage
for s in [0.5, 0.8, 0.9, 0.95]:
    effective_gb = base_size_gb * (1 - s)
    print(f"Sparsity {s:.0%}: effective size = {effective_gb:.2f} GB")`}
        id="code-pruning"
      />

      <TheoremBlock
        title="Lottery Ticket Hypothesis"
        statement="A randomly initialized dense network contains a subnetwork (the 'winning ticket') that, when trained in isolation with its original initialization, can match the full network's accuracy. Formally, there exists a mask $m$ such that training $f(x; m \odot \theta_0)$ achieves test accuracy comparable to training $f(x; \theta_0)$."
        proof="Frankle & Carlin (2019) demonstrated this empirically: train a network, prune the smallest-magnitude weights, rewind remaining weights to their initial values, and retrain. The resulting sparse network matches or exceeds the dense network's performance at 10-20% of the original size."
        corollaries={[
          'Over-parameterization during training is beneficial even if the final model is sparse.',
          'The specific initialization of surviving weights matters — random re-initialization fails.'
        ]}
        id="thm-lottery-ticket"
      />

      <NoteBlock
        type="tip"
        title="Structured vs. Unstructured Trade-offs"
        content="Unstructured pruning achieves higher sparsity at the same accuracy but requires sparse matrix libraries for speedup. Structured pruning (removing heads, layers, or channels) gives immediate speedup on standard hardware. For LLMs, removing 30-50% of attention heads often preserves 95%+ of performance."
        id="note-structured-vs-unstructured"
      />

      <WarningBlock
        title="Pruning Without Retraining"
        content="One-shot pruning without fine-tuning degrades accuracy rapidly above 40-50% sparsity. SparseGPT (2023) addresses this by solving a layer-wise reconstruction problem, enabling 50-60% unstructured sparsity on GPT-scale models without any retraining."
        id="warning-pruning-retraining"
      />
    </div>
  )
}
