import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LoraFlux() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LoRA for FLUX Image Generation</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        FLUX.1 by Black Forest Labs represents a new generation of image models using a DiT
        (Diffusion Transformer) architecture with 12B parameters. LoRA training on FLUX produces
        exceptional quality but requires more VRAM and careful hyperparameter tuning compared to
        Stable Diffusion models.
      </p>

      <DefinitionBlock
        title="FLUX.1 Architecture"
        definition="FLUX.1 uses a flow-matching objective with a Diffusion Transformer (DiT) backbone instead of a UNet. It incorporates dual text encoders (CLIP and T5-XXL) and processes images as sequences of patches. The 12B parameter model produces state-of-the-art image quality."
        id="def-flux"
      />

      <h2 className="text-2xl font-semibold">Training FLUX LoRA with AI-Toolkit</h2>

      <PythonCode
        title="flux_lora_training.py"
        code={`# AI-Toolkit is the recommended tool for FLUX LoRA training
# pip install ai-toolkit (or clone from github.com/ostris/ai-toolkit)

# YAML configuration for FLUX LoRA training
flux_config = """
job: extension
config:
  name: my_flux_lora
  process:
    - type: sd_trainer
      training_folder: ./output
      device: cuda:0
      trigger_word: ohwx
      network:
        type: lora
        linear: 16          # LoRA rank
        linear_alpha: 16    # LoRA alpha

      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4

      datasets:
        - folder_path: ./training-images
          caption_ext: txt
          caption_dropout_rate: 0.05
          resolution: [1024]
          batch_size: 1
          shuffle: true

      train:
        batch_size: 1
        steps: 2000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false    # T5 encoder frozen
        content_or_style: balanced
        gradient_checkpointing: true
        noise_scheduler: flowmatch

      model:
        name_or_path: black-forest-labs/FLUX.1-dev
        quantize: true               # Quantize base model to save VRAM

      optimizer:
        optimizer_type: adamw8bit
        lr: 4e-4                     # Higher LR than SD
"""

print(flux_config)
# Save as config.yaml and run:
# python run.py config.yaml`}
        id="code-flux-lora"
      />

      <PythonCode
        title="flux_lora_diffusers.py"
        code={`# Using diffusers for FLUX LoRA training
from diffusers import FluxPipeline
import torch

# Load FLUX model
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Load trained LoRA
pipe.load_lora_weights("./flux-lora-output/my_flux_lora.safetensors")

# Generate images with LoRA
image = pipe(
    prompt="a portrait of ohwx person in a cyberpunk city",
    height=1024,
    width=1024,
    num_inference_steps=28,
    guidance_scale=3.5,          # FLUX uses lower guidance
    joint_attention_kwargs={"scale": 1.0},  # LoRA strength
).images[0]

image.save("flux_lora_output.png")

# Adjust LoRA strength
for strength in [0.5, 0.8, 1.0, 1.2]:
    image = pipe(
        prompt="a portrait of ohwx person smiling",
        num_inference_steps=28,
        guidance_scale=3.5,
        joint_attention_kwargs={"scale": strength},
    ).images[0]
    image.save(f"flux_strength_{strength}.png")
    print(f"Generated with LoRA strength={strength}")`}
        id="code-flux-inference"
      />

      <ExampleBlock
        title="FLUX LoRA vs SDXL LoRA"
        problem="What are the differences when training LoRA for FLUX vs SDXL?"
        steps={[
          { formula: '\\text{Architecture: DiT (12B) vs UNet (2.6B)}', explanation: 'FLUX is much larger, requiring more VRAM but producing better quality.' },
          { formula: '\\text{VRAM: 16-24 GB vs 12-16 GB}', explanation: 'FLUX needs more VRAM. Quantizing the base model helps fit on 24 GB GPUs.' },
          { formula: '\\text{LR: 4e-4 vs 1e-4}', explanation: 'FLUX LoRA typically uses higher learning rates than SD LoRA.' },
          { formula: '\\text{Steps: 1500-3000 vs 500-1500}', explanation: 'FLUX needs more training steps for convergence.' },
          { formula: '\\text{Guidance: 3-4 vs 7-8}', explanation: 'FLUX uses much lower guidance scale during inference.' },
        ]}
        id="example-flux-vs-sdxl"
      />

      <NoteBlock
        type="tip"
        title="Quantized Training"
        content="FLUX LoRA training with a quantized (NF4) base model fits on 24 GB GPUs with minimal quality loss. Set quantize: true in AI-Toolkit or use bitsandbytes quantization in diffusers. The LoRA weights themselves are trained in full precision."
        id="note-quantized-training"
      />

      <WarningBlock
        title="FLUX License Restrictions"
        content="FLUX.1-dev is released under a non-commercial license. Check the license terms before using FLUX LoRAs in commercial applications. FLUX.1-schnell is Apache 2.0 licensed but has lower quality. Consider SDXL (CreativeML Open RAIL) for commercial projects."
        id="warning-flux-license"
      />
    </div>
  )
}
