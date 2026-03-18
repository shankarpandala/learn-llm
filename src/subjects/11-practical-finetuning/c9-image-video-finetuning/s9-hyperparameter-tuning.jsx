import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function HyperparameterTuning() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Hyperparameter Tuning & Failures</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Diffusion model fine-tuning is sensitive to hyperparameters. Small changes in
        learning rate, rank, or training steps can mean the difference between a great
        result and complete failure. This section provides systematic tuning strategies
        and a catalog of common failures with their fixes.
      </p>

      <DefinitionBlock
        title="Key Hyperparameters"
        definition="The primary hyperparameters for diffusion LoRA training are: learning rate $\eta$ (typically $10^{-5}$ to $5 \times 10^{-4}$), LoRA rank $r$ (4 to 128), training steps $T$ (200 to 5000), and the LoRA alpha scaling $\alpha$ (usually set equal to $r$). The effective learning rate for the LoRA update is $\eta \cdot \frac{\alpha}{r}$."
        id="def-hyperparams"
      />

      <ExampleBlock
        title="Failure Diagnosis Guide"
        problem="How do you identify and fix common diffusion fine-tuning failures?"
        steps={[
          { formula: '\\text{Blurry outputs} \\Rightarrow \\text{LR too high or too many steps}', explanation: 'The model has overfit and lost detail. Reduce learning rate by 2-5x or train fewer steps.' },
          { formula: '\\text{No effect (identical to base)} \\Rightarrow \\text{LR too low or rank too small}', explanation: 'The adapter has not learned enough. Increase LR, rank, or training steps.' },
          { formula: '\\text{Color artifacts / distortion} \\Rightarrow \\text{VAE or dtype issue}', explanation: 'Ensure VAE is in fp32 for SD 1.5, or use bf16 (not fp16) for SDXL/Flux.' },
          { formula: '\\text{Concept bleeding (wrong subjects)} \\Rightarrow \\text{Bad captions}', explanation: 'Captions do not properly isolate the target concept. Fix trigger words and descriptions.' },
          { formula: '\\text{Training images reproduced exactly} \\Rightarrow \\text{Overfit}', explanation: 'Too many steps on too few images. Reduce steps or add more training images.' },
        ]}
        id="example-failures"
      />

      <PythonCode
        title="hyperparameter_sweep.py"
        code={`import subprocess
import itertools
import json
import os
from datetime import datetime

def run_lora_sweep(base_config, param_grid):
    """Run a hyperparameter sweep for diffusion LoRA training."""
    results = []

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"Total configurations: {len(combinations)}")

    for i, combo in enumerate(combinations):
        config = {**base_config}
        for key, val in zip(keys, combo):
            config[key] = val

        run_name = (
            f"sweep_{i:03d}_lr{config['lr']}_r{config['rank']}_s{config['steps']}"
        )
        config["output_dir"] = f"./sweep_results/{run_name}"
        os.makedirs(config["output_dir"], exist_ok=True)

        print(f"\\nRun {i+1}/{len(combinations)}: {run_name}")

        cmd = [
            "accelerate", "launch",
            "diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py",
            f"--pretrained_model_name_or_path={config['model']}",
            f"--dataset_name={config['dataset']}",
            f"--output_dir={config['output_dir']}",
            f"--resolution={config['resolution']}",
            f"--learning_rate={config['lr']}",
            f"--rank={config['rank']}",
            f"--max_train_steps={config['steps']}",
            "--train_batch_size=1",
            "--gradient_checkpointing",
            "--mixed_precision=bf16",
            "--seed=42",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        config["success"] = result.returncode == 0
        config["timestamp"] = datetime.now().isoformat()
        results.append(config)

        with open("./sweep_results/sweep_log.json", "w") as f:
            json.dump(results, f, indent=2)

    return results

base_config = {
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "dataset": "./my_dataset",
    "resolution": 1024,
}

param_grid = {
    "lr": [5e-5, 1e-4, 3e-4],
    "rank": [8, 16, 32],
    "steps": [500, 1000, 2000],
}

# This runs 27 configurations
# run_lora_sweep(base_config, param_grid)`}
        id="code-sweep"
      />

      <PythonCode
        title="evaluate_lora_quality.py"
        code={`import torch
from diffusers import DiffusionPipeline
from pathlib import Path

def evaluate_lora_checkpoints(base_model, lora_dir, eval_prompts):
    """Generate images from each checkpoint for visual comparison."""
    pipe = DiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16
    ).to("cuda")

    checkpoints = sorted(Path(lora_dir).glob("checkpoint-*"))
    print(f"Found {len(checkpoints)} checkpoints")

    for ckpt in checkpoints:
        step = ckpt.name.split("-")[1]
        print(f"\\nGenerating from checkpoint step {step}...")

        pipe.load_lora_weights(str(ckpt))

        for j, prompt in enumerate(eval_prompts):
            image = pipe(
                prompt, num_inference_steps=28, guidance_scale=7.5,
                generator=torch.Generator("cuda").manual_seed(42),
            ).images[0]
            image.save(f"eval_step{step}_prompt{j}.png")

        pipe.unload_lora_weights()

    print("\\nCompare images across steps to find optimal checkpoint.")

# CLIP score for quantitative evaluation
def clip_score(images, prompts):
    """Compute CLIP similarity between images and prompts."""
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    scores = []
    for img, prompt in zip(images, prompts):
        inputs = processor(text=[prompt], images=img, return_tensors="pt")
        outputs = model(**inputs)
        score = outputs.logits_per_image.item()
        scores.append(score)

    avg = sum(scores) / len(scores)
    print(f"Average CLIP score: {avg:.3f}")
    return scores

eval_prompts = [
    "a photo of sks in a forest",
    "sks standing on a beach at sunset",
    "a painting of sks in impressionist style",
]`}
        id="code-evaluate"
      />

      <NoteBlock
        type="intuition"
        title="The Learning Rate Sweet Spot"
        content="For diffusion LoRA, the learning rate window is narrow. Too low (below 1e-5) and nothing is learned. Too high (above 5e-4) and the model quickly produces artifacts. Start with 1e-4 and adjust in 2x increments. If you see artifacts at 1e-4, try 5e-5. If you see no effect, try 2e-4."
        id="note-lr-sweet-spot"
      />

      <WarningBlock
        title="Do Not Tune Everything at Once"
        content="Change one hyperparameter at a time. If you change learning rate, rank, and training steps simultaneously and get a bad result, you will not know which change caused the problem. Start with the defaults, then adjust learning rate first, then rank, then steps."
        id="warning-one-at-a-time"
      />

      <NoteBlock
        type="tip"
        title="Save Checkpoints Frequently"
        content="Save a checkpoint every 200-500 steps. Overfitting in diffusion models happens rapidly, and the best result is often an intermediate checkpoint rather than the final one. Compare outputs across checkpoints to find the sweet spot before quality degrades."
        id="note-save-checkpoints"
      />
    </div>
  )
}
