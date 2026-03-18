import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function SDXLFlux() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">SDXL and FLUX: Next-Generation Image Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        SDXL and FLUX represent major architectural advances beyond Stable Diffusion 1.5. SDXL
        scales the U-Net and adds a refiner model for high-resolution generation. FLUX replaces
        the U-Net entirely with a Diffusion Transformer (DiT), bringing the scaling properties
        of transformers to image generation with superior prompt adherence and image quality.
      </p>

      <DefinitionBlock
        title="Diffusion Transformer (DiT)"
        definition="A Diffusion Transformer replaces the U-Net with a transformer architecture for noise prediction. Instead of convolutional downsampling/upsampling, DiT processes latent patches as a flat token sequence with self-attention. The timestep and class/text conditioning are injected via adaptive layer normalization (adaLN)."
        id="def-dit"
      />

      <h2 className="text-2xl font-semibold">SDXL Architecture</h2>
      <p className="text-gray-700 dark:text-gray-300">
        SDXL uses a 2.6B parameter U-Net (3x larger than SD 1.5), dual text encoders
        (CLIP ViT-L + OpenCLIP ViT-bigG), and generates at 1024x1024 base resolution.
        An optional refiner model improves fine details in a second pass.
      </p>

      <h2 className="text-2xl font-semibold">FLUX Architecture</h2>
      <p className="text-gray-700 dark:text-gray-300">
        FLUX from Black Forest Labs uses a 12B parameter DiT with rectified flow matching
        instead of DDPM noise scheduling. It processes both text and image tokens in a unified
        transformer with joint attention, achieving remarkable text rendering and prompt following.
      </p>
      <BlockMath math="\text{FLUX loss: } \mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \left[ \| \mathbf{v}_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0) \|^2 \right]" />

      <ExampleBlock
        title="Architecture Comparison"
        problem="Compare SD 1.5, SDXL, and FLUX architectures."
        steps={[
          { formula: '\\text{SD 1.5: 860M U-Net, 1 CLIP, 512px, DDPM}', explanation: 'Original Stable Diffusion with single text encoder.' },
          { formula: '\\text{SDXL: 2.6B U-Net, 2 CLIPs, 1024px, DDPM + refiner}', explanation: 'Larger model with dual text encoders and optional refinement stage.' },
          { formula: '\\text{FLUX: 12B DiT, T5-XXL + CLIP, 1024px, Flow Matching}', explanation: 'Transformer-based with flow matching for straighter sampling trajectories.' },
        ]}
        id="example-arch-comparison"
      />

      <PythonCode
        title="sdxl_flux_generation.py"
        code={`from diffusers import (
    StableDiffusionXLPipeline,
    # FluxPipeline,
    EulerDiscreteScheduler,
)
import torch

# === SDXL ===
# pipe_xl = StableDiffusionXLPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     torch_dtype=torch.float16,
#     variant="fp16",
# ).to("cuda")

# image = pipe_xl(
#     prompt="An astronaut riding a horse on Mars, cinematic lighting",
#     negative_prompt="blurry, low quality",
#     num_inference_steps=30,
#     guidance_scale=7.0,
#     height=1024, width=1024,
# ).images[0]

# === FLUX ===
# pipe_flux = FluxPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-dev",
#     torch_dtype=torch.bfloat16,
# ).to("cuda")

# # FLUX uses guidance-distilled model (no negative prompt needed)
# image = pipe_flux(
#     prompt="A cat wearing a tiny hat, sitting in a teacup, watercolor style",
#     num_inference_steps=28,
#     guidance_scale=3.5,
#     height=1024, width=1024,
# ).images[0]

# Key differences
print("=== Architecture Comparison ===")
comparison = {
    "Component":     ["Backbone",    "Text Encoder",          "Resolution", "Scheduler",    "Params"],
    "SD 1.5":        ["U-Net",       "CLIP ViT-L",            "512px",      "DDPM/DDIM",    "~1B"],
    "SDXL":          ["U-Net",       "CLIP-L + OpenCLIP-G",   "1024px",     "DDPM/Euler",   "~3.5B"],
    "FLUX.1-dev":    ["DiT (MMDiT)", "T5-XXL + CLIP-L",       "1024px",     "Flow Matching", "~12B"],
}
header = comparison["Component"]
for key in ["SD 1.5", "SDXL", "FLUX.1-dev"]:
    row = comparison[key]
    print(f"\\n{key}:")
    for h, v in zip(header, row):
        print(f"  {h}: {v}")

# Flow matching vs DDPM
print("\\n=== Flow Matching ===")
print("DDPM: curved noise trajectories, many steps needed")
print("Flow: straight paths from noise to data, fewer steps")
print("FLUX achieves good quality in 4 steps (with guidance distillation)")`}
        id="code-sdxl-flux"
      />

      <NoteBlock
        type="intuition"
        title="Why DiT Beats U-Net"
        content="U-Nets rely on hand-designed skip connections and resolution hierarchies. DiTs treat the problem as sequence modeling, leveraging transformers' proven scaling laws. As compute increases, DiTs improve more predictably. FLUX's joint text-image attention also eliminates the cross-attention bottleneck, giving the model better access to text conditioning."
        id="note-dit-advantage"
      />

      <WarningBlock
        title="VRAM Requirements"
        content="FLUX.1-dev requires ~24GB VRAM in float16, making it impractical for consumer GPUs without quantization. SDXL needs ~7GB in float16. Consider using model offloading (pipe.enable_model_cpu_offload()) or quantized versions (GGUF, NF4) for limited hardware."
        id="warning-vram"
      />
    </div>
  )
}
