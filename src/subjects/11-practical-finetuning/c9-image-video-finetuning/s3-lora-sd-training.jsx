import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LoraSDTraining() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LoRA Training for Stable Diffusion</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        LoRA is the most popular method for fine-tuning Stable Diffusion models because
        it is fast, memory-efficient, and produces small adapter files that can be easily
        shared and combined. This section covers training LoRA adapters for SD 1.5 and
        SDXL to learn new styles, characters, and concepts.
      </p>

      <DefinitionBlock
        title="LoRA for Diffusion Models"
        definition="LoRA (Low-Rank Adaptation) for diffusion models applies low-rank updates to the UNet cross-attention layers and optionally the text encoder. The weight update is $W' = W + BA$ where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$. Typical LoRA files for SDXL are 50-200 MB versus 6.5 GB for full weights."
        id="def-lora-diffusion"
      />

      <ExampleBlock
        title="LoRA Hyperparameters for Stable Diffusion"
        problem="What are the recommended LoRA settings for style and character training?"
        steps={[
          { formula: '\\text{Rank } r: 4\\text{-}8 \\text{ for styles, } 16\\text{-}32 \\text{ for characters}', explanation: 'Higher rank captures more detail but increases file size and overfitting risk.' },
          { formula: '\\text{Alpha } \\alpha = r \\text{ or } 2r', explanation: 'The scaling factor. Setting alpha=rank gives effective scaling of 1.0.' },
          { formula: '\\text{Learning rate: } 1 \\times 10^{-4} \\text{ to } 5 \\times 10^{-4}', explanation: 'Lower than LLM LoRA rates. Start with 1e-4 and adjust based on results.' },
          { formula: '\\text{Steps: 500-2000 for style, 1000-4000 for character}', explanation: 'Fewer images need fewer steps. Monitor for overfitting every 200-500 steps.' },
        ]}
        id="example-hyperparams"
      />

      <PythonCode
        title="train_lora_sdxl.py"
        code={`# LoRA training for SDXL using diffusers
# Install: pip install diffusers[training] accelerate peft

TRAIN_CMD = """
accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py \\
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \\
    --dataset_name="./my_dataset" \\
    --caption_column="text" \\
    --image_column="image" \\
    --output_dir="./sdxl-lora-output" \\
    --resolution=1024 \\
    --train_batch_size=1 \\
    --gradient_accumulation_steps=4 \\
    --num_train_epochs=100 \\
    --learning_rate=1e-4 \\
    --lr_scheduler="cosine" \\
    --lr_warmup_steps=100 \\
    --rank=16 \\
    --seed=42 \\
    --mixed_precision="bf16" \\
    --gradient_checkpointing \\
    --enable_xformers_memory_efficient_attention \\
    --validation_prompt="a painting in the style of sks" \\
    --validation_epochs=25 \\
    --checkpointing_steps=500
"""

# Python API for LoRA configuration
from peft import LoraConfig

def setup_lora_training():
    """Configure LoRA for SDXL training."""
    unet_lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ],
    )

    text_encoder_lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )

    return unet_lora_config, text_encoder_lora_config

unet_config, te_config = setup_lora_training()
print(f"UNet LoRA rank: {unet_config.r}")
print(f"Text encoder LoRA rank: {te_config.r}")
print(TRAIN_CMD)`}
        id="code-train-lora"
      />

      <PythonCode
        title="use_lora_sdxl.py"
        code={`import torch
from diffusers import DiffusionPipeline

def load_and_generate(base_model, lora_path, prompt, lora_scale=0.8):
    """Load a LoRA adapter and generate images."""
    pipe = DiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")

    pipe.load_lora_weights(lora_path)

    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        cross_attention_kwargs={"scale": lora_scale},
    ).images[0]

    return image

# Combine multiple LoRAs
def combine_loras(base_model, lora_configs):
    """Load and combine multiple LoRA adapters."""
    pipe = DiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16
    ).to("cuda")

    for name, path, scale in lora_configs:
        pipe.load_lora_weights(path, adapter_name=name)

    adapter_names = [c[0] for c in lora_configs]
    adapter_weights = [c[2] for c in lora_configs]
    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    return pipe

# Example: combine style + character LoRAs
pipe = combine_loras(
    "stabilityai/stable-diffusion-xl-base-1.0",
    [
        ("style", "./lora-watercolor", 0.7),
        ("character", "./lora-my-character", 0.9),
    ]
)
image = pipe("sks character in watercolor style").images[0]
image.save("combined_lora_output.png")`}
        id="code-use-lora"
      />

      <NoteBlock
        type="tip"
        title="Train Text Encoder Too for Characters"
        content="For character or object concepts, training the text encoder LoRA alongside the UNet LoRA significantly improves identity preservation. For pure style transfer, UNet-only LoRA is usually sufficient."
        id="note-text-encoder"
      />

      <WarningBlock
        title="Caption Quality Drives LoRA Quality"
        content="The most common cause of bad LoRA results is poor captions. Every training image must have an accurate, detailed caption. Use BLIP-2 or CogVLM to auto-caption, then manually review and edit. Include the trigger word (e.g., 'sks style') in every caption."
        id="warning-captions"
      />

      <NoteBlock
        type="note"
        title="LoRA File Compatibility"
        content="LoRA files trained with diffusers can be loaded in ComfyUI, Automatic1111, and other UIs. However, the naming conventions may differ. Use safetensors format for maximum compatibility across tools."
        id="note-compatibility"
      />
    </div>
  )
}
