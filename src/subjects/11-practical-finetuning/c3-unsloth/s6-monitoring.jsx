import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Monitoring() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Monitoring Training: WandB and Loss Curves</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Monitoring training metrics is essential for diagnosing issues and knowing when to stop.
        Weights and Biases (WandB) provides real-time dashboards, while understanding loss curves
        helps you detect overfitting, underfitting, and convergence.
      </p>

      <DefinitionBlock
        title="Training Loss"
        definition="The cross-entropy loss on training batches, measuring how well the model predicts the next token. For a sequence of $T$ tokens, the loss is $L = -\\frac{1}{T}\\sum_{t=1}^{T} \\log P(x_t | x_{<t})$. Lower is better, but the absolute value depends on the dataset and tokenizer."
        id="def-training-loss"
      />

      <h2 className="text-2xl font-semibold">Setting Up WandB</h2>

      <PythonCode
        title="setup_wandb.py"
        code={`import wandb
import os

# Option 1: Login interactively
wandb.login()

# Option 2: Set API key as environment variable
os.environ["WANDB_API_KEY"] = "your-key-here"
os.environ["WANDB_PROJECT"] = "llama3-finetune"

# Initialize a run
run = wandb.init(
    project="llama3-finetune",
    name="qlora-r16-alpaca",
    config={
        "model": "Meta-Llama-3.1-8B-Instruct",
        "method": "QLoRA",
        "rank": 16,
        "alpha": 16,
        "lr": 2e-4,
        "batch_size": 8,
        "epochs": 1,
        "dataset": "alpaca-cleaned",
        "dataset_size": 51760,
    },
)

# WandB integrates with TrainingArguments automatically
# Just set report_to="wandb" in TrainingArguments
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    report_to="wandb",          # Enable WandB logging
    logging_steps=10,            # Log every 10 steps
    # ... other args
)`}
        id="code-wandb-setup"
      />

      <h2 className="text-2xl font-semibold">Interpreting Loss Curves</h2>

      <PythonCode
        title="analyze_loss_curves.py"
        code={`import matplotlib.pyplot as plt
import numpy as np

# Simulated training scenarios
steps = np.arange(0, 1000)

# Scenario 1: Good training (smooth decrease, slight plateau)
good_loss = 2.5 * np.exp(-steps/200) + 0.8 + np.random.normal(0, 0.02, len(steps))

# Scenario 2: Overfitting (train decreases, val increases)
overfit_train = 2.5 * np.exp(-steps/100) + 0.3
overfit_val = 2.0 * np.exp(-steps/300) + 0.7 + 0.001 * steps

# Scenario 3: Learning rate too high (unstable)
unstable = 2.5 * np.exp(-steps/300) + 0.8 + 0.5 * np.sin(steps/20)

# Key metrics to monitor:
metrics = {
    "train/loss": "Should decrease smoothly. Spikes = bad batch or LR too high",
    "train/learning_rate": "Should follow warmup -> decay schedule",
    "train/grad_norm": "Should be stable. Spikes > 10 = gradient explosion",
    "eval/loss": "Compare to train loss. If diverges = overfitting",
    "train/epoch": "Track progress through dataset",
}

for metric, desc in metrics.items():
    print(f"{metric}: {desc}")

# When to stop training:
# 1. Train loss plateaus for >50 steps
# 2. Eval loss starts increasing (overfitting)
# 3. Grad norm explodes (reduce LR or check data)
# 4. Reached target loss (typically 0.5-1.0 for instruction tuning)`}
        id="code-loss-curves"
      />

      <ExampleBlock
        title="Diagnosing Common Training Issues"
        problem="What do different loss curve patterns indicate?"
        steps={[
          { formula: '\\text{Loss stuck at } \\sim 2.5', explanation: 'Model not learning. Check: wrong chat template, LR too low, data not formatted correctly.' },
          { formula: '\\text{Loss oscillating wildly}', explanation: 'Learning rate too high. Reduce by 2-5x. Also check batch size and gradient accumulation.' },
          { formula: '\\text{Loss drops then rises}', explanation: 'Overfitting. Reduce epochs, increase dropout, or add more diverse data.' },
          { formula: '\\text{Loss = NaN or Inf}', explanation: 'Numerical instability. Switch to bf16, reduce LR, enable gradient clipping (max_grad_norm=1.0).' },
        ]}
        id="example-diagnosis"
      />

      <PythonCode
        title="custom_logging_callback.py"
        code={`from transformers import TrainerCallback
import torch

class MemoryMonitorCallback(TrainerCallback):
    """Log GPU memory usage during training."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.max_memory_reserved() / 1e9
            if logs is not None:
                logs["gpu_allocated_gb"] = round(allocated, 2)
                logs["gpu_reserved_gb"] = round(reserved, 2)

    def on_step_end(self, args, state, control, **kwargs):
        # Check for gradient issues
        if state.log_history and len(state.log_history) > 0:
            last_log = state.log_history[-1]
            if "loss" in last_log and last_log["loss"] > 10:
                print(f"WARNING: Loss spike at step {state.global_step}: "
                      f"{last_log['loss']:.4f}")

# Add to trainer
# trainer = SFTTrainer(..., callbacks=[MemoryMonitorCallback()])`}
        id="code-callback"
      />

      <NoteBlock
        type="tip"
        title="Expected Loss Values"
        content="For instruction tuning on clean data: initial loss ~2.0-3.0, final loss ~0.5-1.0. If final loss drops below 0.3, you may be overfitting. For continued pretraining on domain text: initial ~3.0-4.0, final ~1.5-2.5. These are rough guidelines -- actual values depend on the dataset and tokenizer."
        id="note-expected-loss"
      />

      <WarningBlock
        title="Do Not Over-optimize Training Loss"
        content="A very low training loss does not mean a good model. Overfitting to training data causes the model to memorize examples rather than learn patterns. Always evaluate on held-out data and with qualitative tests. Use early stopping or train for 1-3 epochs maximum."
        id="warning-overfit"
      />
    </div>
  )
}
