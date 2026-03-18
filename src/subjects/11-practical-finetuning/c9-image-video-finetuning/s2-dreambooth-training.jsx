import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function DreamboothTraining() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">DreamBooth: Personalized Image Generation</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        DreamBooth finetunes a diffusion model to learn a specific subject (person, object, style)
        from just 3-10 images. By associating the subject with a unique token, the model can then
        generate new images of that subject in any context or style.
      </p>

      <DefinitionBlock
        title="DreamBooth"
        definition="DreamBooth finetunes all UNet parameters (and optionally the text encoder) to bind a unique identifier token (e.g., 'sks') to a specific subject. It uses a prior preservation loss to maintain diversity: $L = L_{\\text{reconstruction}} + \\lambda L_{\\text{prior}}$ where the prior loss uses class-specific generated images to prevent language drift."
        id="def-dreambooth"
      />

      <h2 className="text-2xl font-semibold">Step-by-Step DreamBooth Training</h2>

      <PythonCode
        title="dreambooth_training.py"
        code={`# DreamBooth with diffusers library
# pip install diffusers accelerate transformers

import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.training_utils import EMAModel
from accelerate import Accelerator
from torchvision import transforms
from PIL import Image
import os

# Configuration
model_name = "stabilityai/stable-diffusion-xl-base-1.0"
instance_prompt = "a photo of sks dog"        # Unique token + class
class_prompt = "a photo of a dog"              # Class for prior preservation
instance_data_dir = "./my-dog-images"          # 5-10 images of your subject
class_data_dir = "./class-dog-images"          # Generated class images
output_dir = "./dreambooth-output"

# Training hyperparameters
train_config = {
    "resolution": 1024,               # SDXL native resolution
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-6,            # Low LR for DreamBooth
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,
    "max_train_steps": 800,           # 400-1200 steps typical
    "prior_loss_weight": 1.0,         # Prior preservation strength
    "train_text_encoder": True,       # Also finetune text encoder
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
}

print("DreamBooth training configuration:")
for k, v in train_config.items():
    print(f"  {k}: {v}")

# Actual training uses the diffusers training script:
# accelerate launch train_dreambooth_sdxl.py \\
#   --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \\
#   --instance_data_dir=./my-dog-images \\
#   --instance_prompt="a photo of sks dog" \\
#   --class_data_dir=./class-dog-images \\
#   --class_prompt="a photo of a dog" \\
#   --output_dir=./dreambooth-output \\
#   --resolution=1024 \\
#   --train_batch_size=1 \\
#   --learning_rate=5e-6 \\
#   --max_train_steps=800 \\
#   --with_prior_preservation --prior_loss_weight=1.0`}
        id="code-dreambooth"
      />

      <h2 className="text-2xl font-semibold">Preparing Training Images</h2>

      <PythonCode
        title="prepare_dreambooth_images.py"
        code={`from PIL import Image
import os

def prepare_training_images(input_dir, output_dir, target_size=1024):
    """Prepare images for DreamBooth training."""
    os.makedirs(output_dir, exist_ok=True)

    guidelines = {
        "quantity": "5-10 images (more is not always better)",
        "variety": "Different angles, lighting, backgrounds",
        "quality": "Sharp, well-lit, subject clearly visible",
        "format": "Square crop or close to target resolution",
        "avoid": "Blurry images, heavy filters, watermarks",
    }

    print("Image preparation guidelines:")
    for k, v in guidelines.items():
        print(f"  {k}: {v}")

    count = 0
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue

        img = Image.open(os.path.join(input_dir, filename))

        # Center crop to square
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))

        # Resize to target
        img = img.resize((target_size, target_size), Image.LANCZOS)

        # Save as PNG
        output_path = os.path.join(output_dir, f"image_{count:03d}.png")
        img.save(output_path)
        count += 1

    print(f"\\nPrepared {count} images in {output_dir}")
    return count

# prepare_training_images("./raw-photos", "./training-images")`}
        id="code-prepare-images"
      />

      <ExampleBlock
        title="DreamBooth Hyperparameters"
        problem="How to tune DreamBooth for best results?"
        steps={[
          { formula: '\\text{Steps: 400-800 for SDXL}', explanation: 'Too few = underfitting, too many = overfitting. Check output at 400, 600, 800 steps.' },
          { formula: '\\text{LR: 1e-6 to 5e-6}', explanation: 'Very low learning rate prevents destroying the base model knowledge.' },
          { formula: '\\text{Prior preservation: on}', explanation: 'Prevents the model from forgetting what the class looks like in general.' },
          { formula: '\\text{Text encoder: train for faces, skip for objects}', explanation: 'Training the text encoder helps for people/faces but can hurt for objects/styles.' },
        ]}
        id="example-dreambooth-params"
      />

      <NoteBlock
        type="tip"
        title="Unique Token Choice"
        content="Use a rare token like 'sks', 'ohwx', or 'p3rs0n' as your identifier. Avoid common words that the model already associates with specific concepts. The token should be in the tokenizer's vocabulary but rarely used in training data."
        id="note-token-choice"
      />

      <WarningBlock
        title="Overfitting in DreamBooth"
        content="DreamBooth is very prone to overfitting due to the small dataset (5-10 images). Signs: outputs look exactly like training images with no variation. Mitigation: use prior preservation, train fewer steps, save checkpoints every 100 steps and compare outputs."
        id="warning-dreambooth-overfit"
      />
    </div>
  )
}
