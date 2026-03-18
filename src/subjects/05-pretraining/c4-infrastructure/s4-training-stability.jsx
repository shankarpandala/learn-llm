import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function TrainingStability() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Training Stability: Loss Spikes and Debugging</h1>
      <p className="text-lg text-gray-300">
        Large-scale pretraining runs are plagued by instabilities: loss spikes, divergence,
        and NaN gradients. Understanding the causes and having mitigation strategies is
        essential for successful multi-week training runs costing millions of dollars.
      </p>

      <DefinitionBlock
        title="Loss Spike"
        definition="A loss spike is a sudden, large increase in training loss that disrupts the learning trajectory. Common causes include: bad data batches (corrupted or extremely out-of-distribution), learning rate too high, numerical overflow in FP16, or gradient explosion. Loss spikes can be recoverable or lead to permanent divergence."
        id="loss-spike-def"
      />

      <ExampleBlock
        title="Common Stability Issues and Fixes"
        problem="Diagnose and fix common training instabilities."
        steps={[
          {
            formula: '\\text{Issue: Gradient explosion} \\rightarrow ||g|| > 10^3',
            explanation: 'Fix: Gradient clipping (typically max_norm=1.0). Also check for learning rate warmup.'
          },
          {
            formula: '\\text{Issue: Loss spike from bad batch} \\rightarrow \\mathcal{L}_t > 3 \\cdot \\text{EMA}(\\mathcal{L})',
            explanation: 'Fix: Skip parameter update for outlier batches. Log the batch for investigation.'
          },
          {
            formula: '\\text{Issue: NaN in attention} \\rightarrow \\text{softmax overflow}',
            explanation: 'Fix: Use BF16 instead of FP16, or apply attention score capping (e.g., cap at 30.0 before softmax).'
          },
          {
            formula: '\\text{Issue: Slow divergence} \\rightarrow \\text{loss plateau then increase}',
            explanation: 'Fix: Reduce learning rate, increase warmup, check for data leakage or repetition.'
          }
        ]}
        id="stability-issues-example"
      />

      <NoteBlock
        type="historical"
        title="Lessons from Large-Scale Training"
        content="The PaLM paper (2022) reported loss spikes during training and resolved them by rewinding to a checkpoint ~100 steps before the spike and skipping ~200-500 data batches. The OPT-175B logbook documented numerous instabilities including hardware failures, loss spikes, and divergence. Meta's LLaMA team reported that the 65B model training required manual intervention for loss spikes."
        id="stability-history"
      />

      <TheoremBlock
        title="Gradient Clipping"
        statement="Gradient clipping by global norm scales the gradient vector when its norm exceeds a threshold: $\hat{g} = g \cdot \min\left(1, \frac{c}{||g||_2}\right)$ where $c$ is the clipping threshold. This bounds the step size without changing the gradient direction."
        proof="When $||g||_2 \leq c$: $\hat{g} = g$ (no change). When $||g||_2 > c$: $||\hat{g}||_2 = ||g|| \cdot c/||g|| = c$, so the gradient is rescaled to exactly norm $c$. The direction $\hat{g}/||\hat{g}|| = g/||g||$ is preserved."
        id="grad-clip-thm"
      />

      <PythonCode
        title="training_stability.py"
        code={`import torch
import torch.nn as nn
import numpy as np

class TrainingMonitor:
    """Monitor and detect training instabilities."""

    def __init__(self, window_size=100, spike_threshold=3.0):
        self.losses = []
        self.grad_norms = []
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self.spikes = []

    def log(self, step, loss, grad_norm):
        self.losses.append(loss)
        self.grad_norms.append(grad_norm)

        # Detect loss spike
        if len(self.losses) > self.window_size:
            recent = self.losses[-self.window_size:-1]
            ema = np.mean(recent)
            if loss > self.spike_threshold * ema:
                self.spikes.append(step)
                return "SPIKE_DETECTED"

        # Detect NaN
        if np.isnan(loss) or np.isinf(loss):
            return "NAN_DETECTED"

        # Detect gradient explosion
        if grad_norm > 100.0:
            return "GRAD_EXPLOSION"

        return "OK"

    def summary(self):
        print(f"Total steps: {len(self.losses)}")
        print(f"Loss spikes: {len(self.spikes)} at steps {self.spikes}")
        print(f"Max grad norm: {max(self.grad_norms):.2f}")
        print(f"Final loss: {self.losses[-1]:.4f}")

# Simulate training with instabilities
monitor = TrainingMonitor()
np.random.seed(42)

for step in range(500):
    # Normal loss decrease with noise
    base_loss = 4.0 * np.exp(-step / 200) + 2.0
    noise = np.random.normal(0, 0.05)
    loss = base_loss + noise

    # Inject a loss spike at step 200
    if step == 200:
        loss *= 5.0
    # Inject gradient explosion at step 350
    grad_norm = np.random.exponential(1.0)
    if step == 350:
        grad_norm = 500.0

    status = monitor.log(step, loss, grad_norm)
    if status != "OK":
        print(f"Step {step}: {status} (loss={loss:.3f}, grad_norm={grad_norm:.1f})")

monitor.summary()

# Practical stability techniques
class StableTrainer:
    """Training loop with stability measures."""

    def __init__(self, model, optimizer, max_grad_norm=1.0, skip_threshold=5.0):
        self.model = model
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.skip_threshold = skip_threshold
        self.loss_ema = None
        self.ema_decay = 0.99

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(**batch)
        loss = outputs.loss

        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN/Inf loss detected! Skipping batch.")
            return None

        # Check for loss spike
        if self.loss_ema is not None:
            if loss.item() > self.skip_threshold * self.loss_ema:
                print(f"Loss spike: {loss.item():.2f} vs EMA {self.loss_ema:.2f}")
                return None

        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )

        # Check gradient norm
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print("NaN gradient norm! Skipping update.")
            return None

        self.optimizer.step()

        # Update EMA
        l = loss.item()
        if self.loss_ema is None:
            self.loss_ema = l
        else:
            self.loss_ema = self.ema_decay * self.loss_ema + (1 - self.ema_decay) * l

        return {"loss": l, "grad_norm": grad_norm.item()}

print("\\nKey stability settings:")
print("  grad_clip=1.0, warmup=2000, wd=0.1, beta2=0.95, bf16=True")`}
        id="stability-code"
      />

      <WarningBlock
        title="Recovery from Divergence"
        content="When training diverges (loss explodes or goes to NaN), the standard recovery procedure is: (1) rewind to the last good checkpoint, (2) skip the problematic data batches, (3) optionally reduce the learning rate by 10-50%, (4) resume training. For persistent instability, consider reducing model size, increasing batch size, or switching to BF16. Each recovery attempt costs hours of compute."
        id="divergence-recovery-warning"
      />

      <NoteBlock
        type="tip"
        title="Pre-LN vs Post-LN"
        content="Pre-Layer Normalization (applying LayerNorm before attention/MLP) is significantly more stable than Post-LN. Most modern LLMs use Pre-LN with RMSNorm. QK-norm further stabilizes attention at large scale."
        id="preln-note"
      />
    </div>
  )
}
