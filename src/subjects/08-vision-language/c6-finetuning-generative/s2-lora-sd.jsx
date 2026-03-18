import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LoRAStableDiffusion() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LoRA for Stable Diffusion</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Low-Rank Adaptation (LoRA) is the most popular method for fine-tuning Stable Diffusion
        models. Instead of updating all model weights (860M+ parameters), LoRA trains small
        rank-decomposed matrices that are added to specific layers. This reduces training memory
        by 3x and produces small checkpoint files (2-200MB vs 2-7GB for full models) that can
        be easily shared, stacked, and swapped at inference time.
      </p>

      <DefinitionBlock
        title="LoRA for Diffusion"
        definition="For a pretrained weight matrix $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$, LoRA learns a low-rank update $\Delta\mathbf{W} = \mathbf{B}\mathbf{A}$ where $\mathbf{B} \in \mathbb{R}^{d \times r}$ and $\mathbf{A} \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$. The forward pass becomes: $\mathbf{h} = \mathbf{W}_0\mathbf{x} + \alpha \cdot \mathbf{B}\mathbf{A}\mathbf{x}$ where $\alpha$ is a scaling factor."
        id="def-lora-sd"
      />

      <h2 className="text-2xl font-semibold">Where to Apply LoRA</h2>
      <p className="text-gray-700 dark:text-gray-300">
        In Stable Diffusion, LoRA is typically applied to the attention layers (Q, K, V, O
        projections) and optionally to the cross-attention layers. Some trainers also apply
        LoRA to the text encoder for better prompt understanding.
      </p>
      <BlockMath math="\text{Trainable params} = r \times (d + k) \times N_{\text{layers}} \ll d \times k \times N_{\text{layers}}" />

      <ExampleBlock
        title="LoRA Parameter Count"
        problem="SD 1.5 U-Net has attention layers with d=k=320 to 1280. With rank 32, how many LoRA parameters?"
        steps={[
          { formula: '\\text{Per layer (rank 32, d=k=768): } 32 \\times 768 \\times 2 = 49{,}152', explanation: 'Each LoRA pair (A, B) for one projection layer.' },
          { formula: '\\text{4 projections (Q,K,V,O) per attention block} \\times 49{,}152 \\approx 200K', explanation: 'All attention projections in one block.' },
          { formula: '\\text{Total across U-Net: } \\sim 3\\text{-}50M \\text{ params}', explanation: 'Depends on which layers and rank. ~1-5% of full model.' },
        ]}
        id="example-lora-params"
      />

      <PythonCode
        title="lora_sd_training.py"
        code={`# LoRA training for Stable Diffusion with diffusers + PEFT
# Command-line training:
# accelerate launch train_text_to_image_lora.py \\
#   --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\
#   --dataset_name="my_dataset" \\
#   --output_dir="./lora_output" \\
#   --resolution=512 \\
#   --train_batch_size=4 \\
#   --gradient_accumulation_steps=4 \\
#   --num_train_epochs=100 \\
#   --learning_rate=1e-4 \\
#   --lr_scheduler="cosine" \\
#   --rank=32 \\
#   --mixed_precision="fp16"

from diffusers import StableDiffusionPipeline
import torch

# Load base model and apply LoRA
# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
# ).to("cuda")

# # Load a single LoRA
# pipe.load_lora_weights("path/to/lora", adapter_name="style_lora")
# pipe.set_adapters(["style_lora"], [0.8])  # weight=0.8

# # Stack multiple LoRAs
# pipe.load_lora_weights("path/to/lora2", adapter_name="character_lora")
# pipe.set_adapters(
#     ["style_lora", "character_lora"],
#     [0.7, 0.5]  # Different weights for each
# )

# # Generate with combined LoRAs
# image = pipe("a portrait of sks person in anime style",
#              num_inference_steps=30).images[0]

# LoRA weight merging for inference speed
# pipe.fuse_lora(lora_scale=0.8)  # Merge LoRA into base weights
# pipe.unload_lora_weights()       # Free LoRA adapter memory

# LoRA rank comparison
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scale

# Compare ranks
for rank in [4, 8, 16, 32, 64, 128]:
    lora = LoRALayer(768, 768, rank=rank)
    params = sum(p.numel() for p in lora.parameters())
    full_params = 768 * 768
    print(f"Rank {rank:3d}: {params:>8,} params ({params/full_params*100:.1f}% of full)")`}
        id="code-lora-sd"
      />

      <NoteBlock
        type="tip"
        title="Rank Selection Guide"
        content="Rank 4-8: Style transfer, simple concepts. Rank 16-32: Character consistency, specific art styles. Rank 64-128: Complex subjects, photorealistic faces. Higher ranks capture more detail but risk overfitting and produce larger files. Start with rank 32 as a default."
        id="note-rank-guide"
      />

      <WarningBlock
        title="LoRA Compatibility"
        content="LoRAs are model-specific: a LoRA trained on SD 1.5 will not work with SDXL or vice versa. The architecture, attention dimensions, and layer names must match exactly. SDXL LoRAs are typically 2-4x larger than SD 1.5 LoRAs due to the larger U-Net."
        id="warning-lora-compat"
      />
    </div>
  )
}
