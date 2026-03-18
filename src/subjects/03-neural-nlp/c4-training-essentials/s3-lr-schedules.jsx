import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LRSchedules() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Learning Rate Schedules</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Learning rate schedules adjust the learning rate during training to improve convergence
        and final performance. A warmup phase prevents early instability when gradients are
        noisy, while decay phases allow the model to settle into sharper minima. The right
        schedule can mean the difference between a good model and a great one.
      </p>

      <DefinitionBlock
        title="Learning Rate Warmup"
        definition="Warmup linearly increases the learning rate from 0 (or a small value) to the peak rate over the first $W$ training steps: $\eta_t = \eta_{\max} \cdot \min(1, t / W)$. This prevents large, unstable updates early in training when the model weights are random and gradients are unreliable."
        notation="$W$ = warmup steps, $\eta_{\max}$ = peak learning rate, $t$ = current step."
        id="def-warmup"
      />

      <h2 className="text-2xl font-semibold">Common Schedules</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The transformer paper (Vaswani et al., 2017) introduced a specific schedule combining
        warmup with inverse square root decay:
      </p>
      <BlockMath math="\eta_t = d_{\text{model}}^{-0.5} \cdot \min(t^{-0.5}, t \cdot W^{-1.5})" />
      <p className="text-gray-700 dark:text-gray-300">
        Modern practice favors cosine decay after warmup:
      </p>
      <BlockMath math="\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - W}{T - W}\pi\right)\right)" />

      <ExampleBlock
        title="Warmup + Cosine Schedule"
        problem="For a 100K step training run with 2K warmup steps, lr_max=3e-4, lr_min=1e-5, trace the learning rate."
        steps={[
          { formula: 't=0: \\eta = 0', explanation: 'Training starts with zero learning rate.' },
          { formula: 't=1000: \\eta = 1.5 \\times 10^{-4}', explanation: 'Halfway through warmup, LR is half of peak.' },
          { formula: 't=2000: \\eta = 3 \\times 10^{-4}', explanation: 'End of warmup, LR reaches peak value.' },
          { formula: 't=51000: \\eta \\approx 1.55 \\times 10^{-4}', explanation: 'Halfway through cosine decay, LR is approximately halfway between max and min.' },
          { formula: 't=100000: \\eta = 1 \\times 10^{-5}', explanation: 'End of training, LR reaches minimum value.' },
        ]}
        id="example-cosine"
      />

      <PythonCode
        title="lr_schedules.py"
        code={`import torch
import torch.optim as optim
import math

# 1. Linear warmup + cosine decay (most common for transformers)
def get_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr=1e-5):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max(min_lr / optimizer.defaults['lr'],
                   0.5 * (1 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 2. Transformer (Vaswani) schedule
def get_transformer_schedule(optimizer, d_model, warmup_steps):
    def lr_lambda(step):
        step = max(step, 1)
        return d_model ** (-0.5) * min(step ** (-0.5),
                                        step * warmup_steps ** (-1.5))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 3. Linear warmup + linear decay
def get_linear_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Demo: print LR at key points
model = torch.nn.Linear(512, 512)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
scheduler = get_cosine_schedule(optimizer, warmup_steps=2000,
                                 total_steps=100000)

steps_to_check = [0, 500, 1000, 2000, 10000, 50000, 90000, 100000]
for step in steps_to_check:
    # Reset and step to target
    for pg in optimizer.param_groups:
        pg['lr'] = 3e-4
    scheduler = get_cosine_schedule(optimizer, 2000, 100000)
    for _ in range(step):
        scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print(f"Step {step:>6d}: lr = {lr:.6f}")`}
        id="code-lr-schedules"
      />

      <PythonCode
        title="lr_schedule_training_loop.py"
        code={`import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

# Full training loop with LR scheduling
model = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=256, nhead=4),
    num_layers=4
)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# OneCycleLR: warm up then anneal in one cycle (popular for fine-tuning)
total_steps = 10000
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=total_steps,
    pct_start=0.1,     # 10% warmup
    anneal_strategy='cos',
    div_factor=25,      # initial_lr = max_lr / 25
    final_div_factor=1000,  # final_lr = max_lr / 25 / 1000
)

# Training loop skeleton
criterion = nn.CrossEntropyLoss()
for step in range(total_steps):
    optimizer.zero_grad()
    # Simulate forward pass
    x = torch.randn(8, 32, 256)
    out = model(x)
    loss = out.sum()  # dummy loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()  # update LR AFTER optimizer step

    if step % 2000 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Step {step}: lr={lr:.6f}, loss={loss.item():.4f}")`}
        id="code-training-loop"
      />

      <NoteBlock
        type="intuition"
        title="Why Warmup Prevents Instability"
        content="At initialization, the model's weights are random and produce near-uniform attention distributions. Gradients in this regime are noisy and can have very large magnitude. A large learning rate would cause wild parameter updates that the model never recovers from. Warmup lets the model first find a reasonable region of parameter space before taking large steps."
        id="note-warmup-intuition"
      />

      <WarningBlock
        title="Warmup Steps Must Scale with Model Size"
        content="Larger models need more warmup steps. GPT-3 used 375M tokens of warmup (~375 steps at batch size 1M tokens). BERT used 10K warmup steps. Too few warmup steps cause training divergence, especially with large learning rates. A common heuristic: warmup for 1-5% of total training steps."
        id="warning-warmup-scaling"
      />

      <NoteBlock
        type="tip"
        title="Cosine vs. Linear Decay"
        content="Cosine decay keeps the learning rate higher for longer before a smooth decline at the end, while linear decay drops steadily. Empirically, cosine decay tends to produce slightly better results for transformer training. For fine-tuning on small datasets, linear decay is often sufficient."
        id="note-cosine-vs-linear"
      />
    </div>
  )
}
