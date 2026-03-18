import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function Checkpointing() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Gradient Checkpointing and Model Saving</h1>
      <p className="text-lg text-gray-300">
        Training large models requires careful memory management and fault tolerance. Gradient
        checkpointing trades compute for memory by recomputing activations during the backward
        pass. Model checkpointing saves training state periodically to enable recovery from
        hardware failures during multi-week training runs.
      </p>

      <DefinitionBlock
        title="Gradient Checkpointing (Activation Checkpointing)"
        definition="Instead of storing all intermediate activations for backpropagation, gradient checkpointing stores activations only at selected checkpoint boundaries. During the backward pass, activations between checkpoints are recomputed from the nearest checkpoint. This reduces memory from $O(L)$ to $O(\sqrt{L})$ at the cost of ~33% more compute."
        notation="For $L$ layers with checkpoints every $\sqrt{L}$ layers: memory = $O(\sqrt{L})$ activations, compute = one extra forward pass per segment."
        id="grad-checkpoint-def"
      />

      <TheoremBlock
        title="Optimal Checkpoint Placement"
        statement="For a sequential model with $L$ layers, placing checkpoints every $k$ layers gives memory $O(L/k + k)$ for activations. The optimal spacing is $k^* = \sqrt{L}$, yielding minimum memory $O(\sqrt{L})$ with compute overhead factor of at most 2x (one additional forward pass per segment)."
        proof="Memory consists of: $L/k$ stored checkpoints plus $k$ recomputed activations in the longest segment. Total: $f(k) = L/k + k$. Setting $f'(k) = -L/k^2 + 1 = 0$ gives $k = \sqrt{L}$, so $f(\sqrt{L}) = 2\sqrt{L}$."
        id="checkpoint-placement-thm"
      />

      <ExampleBlock
        title="Memory Savings with Gradient Checkpointing"
        problem="Estimate memory savings for a 7B model with 32 layers, batch size 4, sequence length 4096."
        steps={[
          {
            formula: '\\text{Activation per layer} \\approx B \\times S \\times d \\times 2 = 4 \\times 4096 \\times 4096 \\times 2 \\approx 128 \\text{ MB}',
            explanation: 'Each layer stores activations in FP16 (2 bytes). Hidden dim d=4096.'
          },
          {
            formula: '\\text{Without checkpointing: } 32 \\times 128 = 4096 \\text{ MB} \\approx 4 \\text{ GB}',
            explanation: 'All 32 layers store activations. This is just activations, not weights.'
          },
          {
            formula: '\\text{With checkpointing (every } \\sqrt{32} \\approx 6 \\text{ layers): } 2 \\times \\sqrt{32} \\times 128 \\approx 1.4 \\text{ GB}',
            explanation: 'Only checkpoint layers plus one segment of recomputed activations in memory.'
          }
        ]}
        id="checkpoint-savings-example"
      />

      <NoteBlock
        type="tip"
        title="Selective Checkpointing"
        content="Not all layers use equal memory for activations. Attention layers store QKV projections and attention weights (O(S^2) per head), while MLP layers store intermediate activations. Modern implementations checkpoint attention layers selectively, since they are the primary memory bottleneck, while keeping MLP activations that are cheaper to store."
        id="selective-checkpoint-note"
      />

      <PythonCode
        title="checkpointing.py"
        code={`import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Gradient checkpointing with PyTorch
class TransformerBlock(nn.Module):
    def __init__(self, d_model=1024):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x

class CheckpointedModel(nn.Module):
    def __init__(self, n_layers=32, d_model=1024, use_checkpoint=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model) for _ in range(n_layers)]
        )
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # Recompute activations in backward pass
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

# Compare memory usage
def measure_memory(use_checkpoint):
    if not torch.cuda.is_available():
        return 0
    torch.cuda.reset_peak_memory_stats()
    model = CheckpointedModel(
        n_layers=16, d_model=512, use_checkpoint=use_checkpoint
    ).cuda()
    x = torch.randn(2, 256, 512).cuda()
    out = model(x)
    out.sum().backward()
    peak = torch.cuda.max_memory_allocated() / 1e9
    return peak

if torch.cuda.is_available():
    mem_no_ckpt = measure_memory(False)
    mem_ckpt = measure_memory(True)
    print(f"Without checkpointing: {mem_no_ckpt:.2f} GB peak")
    print(f"With checkpointing:    {mem_ckpt:.2f} GB peak")
    print(f"Savings: {(1 - mem_ckpt/mem_no_ckpt)*100:.1f}%")
else:
    print("Memory comparison requires CUDA")

# HuggingFace gradient checkpointing
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.gradient_checkpointing_enable()
print(f"\\nGradient checkpointing enabled: {model.is_gradient_checkpointing}")

# Model saving and loading checkpoints
def save_training_checkpoint(model, optimizer, scheduler, step, path):
    """Save full training state for recovery."""
    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "rng_state": torch.random.get_rng_state(),
    }
    torch.save(checkpoint_dict, path)
    print(f"Saved checkpoint at step {step} to {path}")

# Checkpoint every N steps during training
print("\\nTypical checkpointing schedule:")
print("  Every 1000 steps: save full checkpoint (~recovery)")
print("  Every 100 steps: save lightweight metrics/logs")
print("  Keep last 3-5 checkpoints to save disk space")`}
        id="checkpointing-code"
      />

      <WarningBlock
        title="Checkpoint Storage Can Be Enormous"
        content="A 70B model checkpoint with optimizer states takes ~800GB (model 140GB + optimizer 560GB in FP32 + gradients 140GB). Saving every 1000 steps over a 100K step run produces 100 checkpoints = 80TB. Use sharded saving (save each GPU's shard separately), keep only recent checkpoints, and use asynchronous I/O to avoid stalling training."
        id="checkpoint-storage-warning"
      />
    </div>
  )
}
