import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LoraSdTraining() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LoRA for Stable Diffusion Training</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        LoRA for image generation applies the same low-rank adaptation technique to the UNet and
        text encoder of Stable Diffusion models. It produces small adapter files (10-200 MB) that
        can teach the model new concepts, styles, or characters with far less VRAM than DreamBooth.
      </p>

      <DefinitionBlock
        title="SD LoRA Training"
        definition="LoRA for Stable Diffusion adds low-rank matrices to the attention layers of the UNet denoiser and optionally the text encoder. The base model is frozen, and only the LoRA weights are trained. The resulting adapter file is small (10-200 MB) and can be loaded/unloaded at inference time."
        id="def-sd-lora"
      />

      <h2 className="text-2xl font-semibold">Training with diffusers</h2>

      <PythonCode
        title="train_sd_lora.sh"
        code={`# Using the diffusers training script for SDXL LoRA
# Download the script from the diffusers repository

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATASET_DIR="./training-images"
export OUTPUT_DIR="./sdxl-lora-output"

accelerate launch train_text_to_image_lora_sdxl.py \\
  --pretrained_model_name_or_path=$MODEL_NAME \\
  --train_data_dir=$DATASET_DIR \\
  --output_dir=$OUTPUT_DIR \\
  --resolution=1024 \\
  --train_batch_size=1 \\
  --gradient_accumulation_steps=4 \\
  --learning_rate=1e-4 \\
  --lr_scheduler="cosine" \\
  --lr_warmup_steps=0 \\
  --max_train_steps=1000 \\
  --rank=32 \\
  --mixed_precision="bf16" \\
  --gradient_checkpointing \\
  --enable_xformers_memory_efficient_attention \\
  --validation_prompt="a photo of sks style landscape" \\
  --validation_epochs=5 \\
  --seed=42`}
        id="code-train-sdxl-lora"
      />

      <PythonCode
        title="kohya_ss_training.py"
        code={`# Kohya-ss is the most popular tool for SD LoRA training
# pip install kohya-ss or use the GUI

# Kohya configuration (TOML format)
kohya_config = {
    # Model settings
    "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
    "output_dir": "./kohya-lora-output",
    "output_name": "my_style_lora",

    # Training settings
    "resolution": "1024,1024",
    "train_batch_size": 1,
    "max_train_epochs": 10,
    "learning_rate": 1e-4,
    "unet_lr": 1e-4,
    "text_encoder_lr": 5e-5,        # Lower LR for text encoder
    "lr_scheduler": "cosine_with_restarts",
    "optimizer_type": "AdamW8bit",

    # LoRA settings
    "network_module": "networks.lora",
    "network_dim": 32,               # LoRA rank
    "network_alpha": 16,             # LoRA alpha (alpha/rank = 0.5)
    "network_train_unet_only": False, # Also train text encoder

    # Memory optimization
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "cache_latents": True,           # Pre-encode images with VAE
    "cache_latents_to_disk": True,

    # Regularization
    "noise_offset": 0.0357,          # Improves contrast
    "min_snr_gamma": 5.0,            # Min-SNR weighting
}

import json
print(json.dumps(kohya_config, indent=2))

# Launch Kohya training:
# python sdxl_train_network.py --config_file config.toml`}
        id="code-kohya"
      />

      <PythonCode
        title="load_sd_lora.py"
        code={`from diffusers import StableDiffusionXLPipeline
import torch

# Load base model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# Load your trained LoRA
pipe.load_lora_weights("./sdxl-lora-output")

# Generate with LoRA
image = pipe(
    prompt="a photo of sks style mountain landscape at sunset",
    num_inference_steps=30,
    guidance_scale=7.5,
    cross_attention_kwargs={"scale": 0.8},  # LoRA strength (0-1)
).images[0]

image.save("lora_output.png")

# Unload LoRA (back to base model)
pipe.unload_lora_weights()

# Load multiple LoRAs
pipe.load_lora_weights("./style-lora", adapter_name="style")
pipe.load_lora_weights("./character-lora", adapter_name="character")
pipe.set_adapters(["style", "character"], adapter_weights=[0.7, 0.9])`}
        id="code-load-lora"
      />

      <ExampleBlock
        title="LoRA vs DreamBooth for Image Models"
        problem="When should you use LoRA vs DreamBooth for image finetuning?"
        steps={[
          { formula: '\\text{LoRA: styles, concepts, broad subjects}', explanation: 'Better for learning art styles, visual concepts, general categories (10+ images).' },
          { formula: '\\text{DreamBooth: specific subjects, faces}', explanation: 'Better for learning one specific person, pet, or object (3-10 images).' },
          { formula: '\\text{LoRA: 6-16 GB VRAM, 10-200 MB output}', explanation: 'Much more memory efficient. Adapter files are small and composable.' },
          { formula: '\\text{DreamBooth: 16-24 GB VRAM, full model output}', explanation: 'Produces a full model. Higher quality for specific subjects but larger.' },
        ]}
        id="example-lora-vs-dreambooth"
      />

      <NoteBlock
        type="tip"
        title="Caption Your Training Images"
        content="Good captions dramatically improve LoRA quality. Use BLIP-2 or Florence-2 to auto-caption, then manually edit to include your trigger word and describe relevant details. Each caption should describe what makes the image unique."
        id="note-captions"
      />

      <WarningBlock
        title="LoRA Rank for Image Models"
        content="Image LoRA ranks are typically higher than LLM LoRA ranks. Use rank 16-64 for styles, 32-128 for characters/subjects. Higher ranks capture more detail but risk overfitting. Always save checkpoints at multiple training stages to find the sweet spot."
        id="warning-rank"
      />
    </div>
  )
}
