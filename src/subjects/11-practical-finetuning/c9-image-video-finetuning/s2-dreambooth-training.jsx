import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function DreamBoothTraining() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">DreamBooth Training Step-by-Step</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        DreamBooth fine-tunes the entire diffusion model to learn a specific subject from
        just 3-10 images. It binds a unique identifier token to the concept, allowing you
        to generate the subject in novel contexts. This section walks through the full
        DreamBooth training pipeline using the diffusers library.
      </p>

      <DefinitionBlock
        title="DreamBooth"
        definition="DreamBooth fine-tunes all parameters of a diffusion model to associate a rare token identifier (e.g., 'sks') with a specific subject. The training loss combines a reconstruction term on subject images with an optional prior preservation term: $\mathcal{L} = \mathbb{E}[\|{\epsilon - \epsilon_\theta(x_t, t, c_{\text{sks}})}\|^2] + \lambda \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t, c_{\text{class}})\|^2]$ where $\lambda$ controls prior preservation strength."
        id="def-dreambooth"
      />

      <ExampleBlock
        title="DreamBooth Training Checklist"
        problem="What do you need for a successful DreamBooth training run?"
        steps={[
          { formula: '\\text{3-10 high-quality images of the subject}', explanation: 'Diverse angles, lighting, and backgrounds. Crop to subject, resize to model resolution.' },
          { formula: '\\text{Unique token: } \\texttt{sks}, \\texttt{ohwx}', explanation: 'Choose a rare token unlikely to have strong existing associations in the model.' },
          { formula: '\\text{Class prompt: "a photo of a [class]"}', explanation: 'The general category (person, dog, car) for prior preservation regularization.' },
          { formula: '\\text{200-1000 prior preservation images}', explanation: 'Generated from the base model using the class prompt to prevent language drift.' },
        ]}
        id="example-checklist"
      />

      <PythonCode
        title="dreambooth_training.py"
        code={`# DreamBooth training with diffusers
# Requires: pip install diffusers[training] accelerate transformers

import os

INSTANCE_DIR = "./data/dreambooth/sks_dog"
CLASS_DIR = "./data/dreambooth/class_dog"
OUTPUT_DIR = "./models/dreambooth-dog"
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"

# Step 1: Launch DreamBooth LoRA training
TRAIN_CMD = f"""
accelerate launch diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py \\
    --pretrained_model_name_or_path="{MODEL_NAME}" \\
    --instance_data_dir="{INSTANCE_DIR}" \\
    --class_data_dir="{CLASS_DIR}" \\
    --output_dir="{OUTPUT_DIR}" \\
    --instance_prompt="a photo of sks dog" \\
    --class_prompt="a photo of a dog" \\
    --with_prior_preservation \\
    --prior_loss_weight=1.0 \\
    --num_class_images=200 \\
    --resolution=1024 \\
    --train_batch_size=1 \\
    --gradient_accumulation_steps=4 \\
    --gradient_checkpointing \\
    --learning_rate=1e-4 \\
    --lr_scheduler="constant" \\
    --lr_warmup_steps=0 \\
    --max_train_steps=500 \\
    --seed=42 \\
    --mixed_precision="bf16" \\
    --enable_xformers_memory_efficient_attention
"""

print("Place images in", INSTANCE_DIR)
print("Training command:")
print(TRAIN_CMD)`}
        id="code-dreambooth-train"
      />

      <PythonCode
        title="dreambooth_inference.py"
        code={`import torch
from diffusers import DiffusionPipeline

def generate_dreambooth(model_dir, prompts, num_images_per_prompt=4):
    """Generate images using a DreamBooth-trained model."""
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    ).to("cuda")

    # Load the DreamBooth LoRA weights
    pipe.load_lora_weights(model_dir)

    results = []
    for prompt in prompts:
        images = pipe(
            prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images
        results.extend(images)

        for i, img in enumerate(images):
            safe_name = prompt[:30].replace(" ", "_")
            img.save(f"output_{safe_name}_{i}.png")

    return results

prompts = [
    "a photo of sks dog wearing a top hat",
    "a photo of sks dog on the moon",
    "a painting of sks dog in the style of Van Gogh",
    "a photo of sks dog sitting in a cafe in Paris",
]

generate_dreambooth("./models/dreambooth-dog", prompts)`}
        id="code-dreambooth-inference"
      />

      <NoteBlock
        type="tip"
        title="DreamBooth + LoRA Is Usually Better"
        content="Full DreamBooth fine-tuning modifies all model weights and is prone to overfitting. DreamBooth with LoRA (train_dreambooth_lora) produces comparable results with much less VRAM and risk. Start with LoRA DreamBooth unless you have a specific reason to train all weights."
        id="note-dreambooth-lora"
      />

      <WarningBlock
        title="Overfitting Is the Biggest Risk"
        content="DreamBooth overfits very quickly -- often within 200-400 steps for LoRA. Signs of overfitting: the model generates exact copies of training images, or non-subject generations degrade. Train for fewer steps than you think, and generate test images every 100 steps."
        id="warning-overfit"
      />

      <NoteBlock
        type="note"
        title="Token Selection Matters"
        content="The identifier token (sks, ohwx, etc.) should be rare enough that it does not already have a strong meaning in the model. Avoid common words. Some practitioners use random 3-letter strings. Test the token in the base model first to confirm it does not generate a specific concept."
        id="note-token-selection"
      />
    </div>
  )
}
