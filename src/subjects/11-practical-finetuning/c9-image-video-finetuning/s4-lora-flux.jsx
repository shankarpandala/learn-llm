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
      <h1 className="text-3xl font-bold">LoRA Training for Flux</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Flux is a next-generation diffusion transformer model from Black Forest Labs
        that produces highly detailed images with excellent prompt adherence. Training
        LoRA adapters for Flux follows a similar pattern to SDXL but requires more VRAM
        due to the larger architecture. This section covers the practical workflow.
      </p>

      <DefinitionBlock
        title="Flux Architecture"
        definition="Flux uses a diffusion transformer (DiT) architecture rather than the UNet used in Stable Diffusion. The model processes image patches as tokens through transformer blocks with cross-attention to text embeddings. Flux.1-dev has approximately 12B parameters, making LoRA essential for consumer hardware fine-tuning."
        id="def-flux"
      />

      <ExampleBlock
        title="Flux LoRA Training Parameters"
        problem="What settings work best for Flux LoRA training?"
        steps={[
          { formula: '\\text{Rank: } r = 16 \\text{ (style)} \\text{ or } 32 \\text{ (character)}', explanation: 'Flux benefits from slightly higher ranks than SDXL due to its transformer architecture.' },
          { formula: '\\text{Learning rate: } 1 \\times 10^{-4}', explanation: 'Standard starting point. Reduce to 5e-5 if you see artifacts early.' },
          { formula: '\\text{Resolution: 512-1024 (train), 1024+ (inference)}', explanation: 'Training at 512 is faster and still transfers well to higher resolutions.' },
          { formula: '\\text{Steps: 500-1500 for 20-50 images}', explanation: 'Flux LoRA converges faster than SDXL. Check outputs every 250 steps.' },
        ]}
        id="example-flux-params"
      />

      <PythonCode
        title="train_flux_lora.py"
        code={`# Flux LoRA training using diffusers
# Requires: pip install diffusers[training] accelerate peft bitsandbytes

# Option 1: Using the diffusers training script
TRAIN_CMD = """
accelerate launch diffusers/examples/dreambooth/train_dreambooth_lora_flux.py \\
    --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \\
    --dataset_name="./my_flux_dataset" \\
    --instance_prompt="a photo of sks person" \\
    --output_dir="./flux-lora-output" \\
    --resolution=512 \\
    --train_batch_size=1 \\
    --gradient_accumulation_steps=4 \\
    --learning_rate=1e-4 \\
    --lr_scheduler="constant" \\
    --lr_warmup_steps=0 \\
    --max_train_steps=1000 \\
    --rank=16 \\
    --seed=42 \\
    --mixed_precision="bf16" \\
    --gradient_checkpointing \\
    --validation_prompt="a photo of sks person wearing sunglasses" \\
    --validation_epochs=100
"""

# Option 2: ai-toolkit config
AI_TOOLKIT_CONFIG = """
# config.yaml for ai-toolkit Flux LoRA training
job: extension
config:
  name: flux_lora_my_subject
  process:
    - type: sd_trainer
      training_folder: ./output
      device: cuda:0
      trigger_word: sks
      network:
        type: lora
        linear: 16
        linear_alpha: 16
      datasets:
        - folder_path: ./training_images
          caption_ext: .txt
          resolution: 512
          batch_size: 1
      train:
        steps: 1000
        lr: 1e-4
        optimizer: adamw8bit
"""

print("Flux LoRA Training Options:")
print("1. diffusers script (recommended)")
print(TRAIN_CMD)
print("\\n2. ai-toolkit config")
print(AI_TOOLKIT_CONFIG)`}
        id="code-train-flux"
      />

      <PythonCode
        title="flux_lora_inference.py"
        code={`import torch
from diffusers import FluxPipeline

def generate_with_flux_lora(lora_path, prompt, num_images=4):
    """Generate images using a Flux model with LoRA adapter."""
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    pipe.load_lora_weights(lora_path)

    images = pipe(
        prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=28,
        guidance_scale=3.5,
        height=1024,
        width=1024,
    ).images

    for i, img in enumerate(images):
        img.save(f"flux_lora_{i}.png")

    return images

# For lower VRAM: use quantized Flux
def generate_flux_quantized(lora_path, prompt):
    """Run Flux with 4-bit quantization for lower VRAM."""
    from diffusers import BitsAndBytesConfig

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
    )
    pipe.load_lora_weights(lora_path)

    image = pipe(prompt, num_inference_steps=28, guidance_scale=3.5).images[0]
    return image

images = generate_with_flux_lora(
    "./flux-lora-output",
    "a photo of sks person riding a motorcycle through Tokyo at night"
)`}
        id="code-flux-inference"
      />

      <NoteBlock
        type="tip"
        title="Flux Guidance Scale"
        content="Flux uses much lower guidance scales than SDXL. Start with guidance_scale=3.5 for Flux.1-dev. Going above 5.0 often produces oversaturated, artifact-heavy images. Flux.1-schnell uses no guidance (guidance_scale=0) by design."
        id="note-flux-guidance"
      />

      <WarningBlock
        title="Flux VRAM Requirements"
        content="Flux.1-dev at bf16 needs ~24GB VRAM just for inference. For training with LoRA and gradient checkpointing, expect 20-32GB depending on resolution and batch size. Use fp8 or NF4 quantization if your GPU has less than 24GB."
        id="warning-flux-vram"
      />

      <NoteBlock
        type="note"
        title="Flux vs SDXL LoRA"
        content="Flux LoRAs tend to generalize better than SDXL LoRAs because the transformer architecture has more uniform attention patterns. However, Flux LoRA training is 2-3x slower per step and the adapter files are larger (100-400MB vs 50-200MB for SDXL)."
        id="note-flux-vs-sdxl"
      />
    </div>
  )
}
