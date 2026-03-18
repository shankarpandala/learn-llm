import React from 'react';
import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx';
import ExampleBlock from '../../../components/content/ExampleBlock.jsx';
import NoteBlock from '../../../components/content/NoteBlock.jsx';
import WarningBlock from '../../../components/content/WarningBlock.jsx';
import PythonCode from '../../../components/content/PythonCode.jsx';

const GRID_SEARCH_CODE = `# Hyperparameter sweep for image/video fine-tuning
import itertools
import subprocess
import json

# Define search space
search_space = {
    "learning_rate": [1e-5, 5e-5, 1e-4],
    "lora_rank": [8, 32, 64],
    "train_steps": [500, 1000, 2000],
    "text_encoder_lr": [0, 1e-5, 5e-5],
}

# Generate all combinations
keys = list(search_space.keys())
combos = list(itertools.product(*search_space.values()))

results = []
for i, values in enumerate(combos):
    config = dict(zip(keys, values))
    print(f"\\nRun {i+1}/{len(combos)}: {config}")

    # Launch training with these hyperparameters
    cmd = [
        "accelerate", "launch", "train_lora.py",
        f"--learning_rate={config['learning_rate']}",
        f"--lora_rank={config['lora_rank']}",
        f"--max_train_steps={config['train_steps']}",
        f"--output_dir=./sweep/run_{i}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Log results
    results.append({
        "run": i, "config": config,
        "status": "success" if result.returncode == 0 else "failed",
    })

# Save sweep results
with open("sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)`;

const MONITORING_CODE = `# Training monitoring and early stopping
import wandb
from pathlib import Path

# Initialize W&B tracking
wandb.init(project="diffusion-finetune", config={
    "model": "stable-diffusion-xl",
    "method": "lora",
    "rank": 32,
    "learning_rate": 1e-4,
})

# Common failure indicators to watch:
failure_checks = {
    "loss_spike": lambda loss, prev: loss > prev * 3,
    "loss_nan": lambda loss, _: loss != loss,  # NaN check
    "loss_plateau": lambda losses: (
        len(losses) > 100 and
        abs(losses[-1] - losses[-100]) < 1e-6
    ),
}

# Typical healthy training curves:
# Step 0-100:   loss ~0.15 (rapid decrease)
# Step 100-500: loss ~0.08 (gradual decrease)
# Step 500+:    loss ~0.05 (plateau = convergence)

# Key hyperparameters and their effects:
# learning_rate:    Too high → artifacts, too low → no learning
# lora_rank:        Higher → more capacity but overfitting risk
# train_steps:      Too many → overfitting, too few → underfitting
# text_encoder_lr:  Helps prompt adherence, risk of catastrophic forgetting
# prior_loss_weight: DreamBooth only, prevents subject drift`;

export default function HyperparameterTuning() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <div>
        <h1 className="text-3xl font-extrabold tracking-tight text-gray-900 dark:text-white">
          Hyperparameter Tuning & Common Failures
        </h1>
        <p className="mt-3 text-lg text-gray-600 dark:text-gray-400">
          Image and video fine-tuning is highly sensitive to hyperparameters. A wrong learning
          rate or too many training steps can produce artifacts, mode collapse, or catastrophic
          forgetting. This section covers systematic tuning and debugging strategies.
        </p>
      </div>

      <DefinitionBlock
        title="Key Hyperparameters for Diffusion Fine-tuning"
        definition="The critical hyperparameters are: learning rate ($\eta$), LoRA rank ($r$), training steps ($T$), text encoder learning rate ($\eta_{te}$), and prior preservation weight ($\lambda$). The effective model update is $\Delta W = \eta \cdot \sum_{t=1}^{T} \nabla_W \mathcal{L}_t$."
      />

      <ExampleBlock
        title="Recommended Starting Points by Method"
        problem="What hyperparameters should I start with for different fine-tuning methods?"
        steps={[
          { formula: 'DreamBooth: $\\eta = 5 \\times 10^{-6}$, steps = 800-1200, $\\lambda = 1.0$', explanation: 'Conservative LR prevents identity loss' },
          { formula: 'LoRA SD/SDXL: $\\eta = 1 \\times 10^{-4}$, rank = 32, steps = 1000-3000', explanation: 'Higher LR is safe with low-rank adaptation' },
          { formula: 'LoRA FLUX: $\\eta = 5 \\times 10^{-5}$, rank = 16-64, steps = 500-2000', explanation: 'FLUX is more sensitive to learning rate' },
          { formula: 'AnimateDiff: $\\eta = 1 \\times 10^{-4}$, rank = 32, frames = 16', explanation: 'Temporal LoRA needs consistent motion data' },
        ]}
      />

      <PythonCode code={GRID_SEARCH_CODE} title="hyperparameter_sweep.py" />

      <WarningBlock title="Common Failure Modes">
        <div className="space-y-2">
          <p><strong>Color shift / saturation:</strong> Learning rate too high. Reduce by 2-5×.</p>
          <p><strong>Loss of diversity:</strong> Overfitting. Reduce steps, increase dataset size, add regularization.</p>
          <p><strong>Ignoring prompt:</strong> Text encoder weights corrupted. Freeze text encoder or lower its LR.</p>
          <p><strong>Artifacts / noise:</strong> Gradient explosion. Enable gradient clipping (max_norm=1.0).</p>
          <p><strong>NaN loss:</strong> Learning rate too high or data issue. Check for corrupt images/videos.</p>
        </div>
      </WarningBlock>

      <PythonCode code={MONITORING_CODE} title="monitoring.py" />

      <NoteBlock type="tip" title="The 3-Stage Tuning Strategy">
        <div className="space-y-1">
          <p><strong>Stage 1:</strong> Quick test with 200 steps at high LR (1e-4) to verify the pipeline works.</p>
          <p><strong>Stage 2:</strong> Full training at moderate LR (5e-5) for 1000 steps, generate validation images every 200 steps.</p>
          <p><strong>Stage 3:</strong> Pick the best checkpoint, optionally continue with lower LR (1e-5) for refinement.</p>
        </div>
      </NoteBlock>
    </div>
  );
}
