import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ImageVideoHardware() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Hardware Requirements for Image and Video Finetuning</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Finetuning image and video generation models has different hardware requirements than LLM
        finetuning. Image models like Stable Diffusion require less VRAM than LLMs, while video
        models can be extremely demanding. This section covers GPU requirements for each approach.
      </p>

      <DefinitionBlock
        title="Diffusion Model VRAM"
        definition="Diffusion models store the UNet/Transformer, text encoder(s), and VAE in VRAM during training. For Stable Diffusion XL, the UNet alone has ~2.6B parameters. VRAM usage scales with image resolution: training at 1024x1024 uses roughly 4x the activation memory of 512x512."
        id="def-diffusion-vram"
      />

      <ExampleBlock
        title="VRAM Requirements by Task"
        problem="How much VRAM is needed for different image/video finetuning methods?"
        steps={[
          { formula: '\\text{SD 1.5 LoRA (512x512): } 6\\text{-}8 \\text{ GB}', explanation: 'Stable Diffusion 1.5 LoRA finetuning fits on most consumer GPUs.' },
          { formula: '\\text{SDXL LoRA (1024x1024): } 12\\text{-}16 \\text{ GB}', explanation: 'SDXL requires more VRAM due to larger UNet and higher resolution.' },
          { formula: '\\text{FLUX LoRA (1024x1024): } 16\\text{-}24 \\text{ GB}', explanation: 'FLUX.1 has a 12B parameter transformer, requiring significant VRAM.' },
          { formula: '\\text{DreamBooth (full UNet): } 16\\text{-}24 \\text{ GB}', explanation: 'DreamBooth finetuning updates all UNet weights, using more memory.' },
          { formula: '\\text{Video (CogVideoX LoRA): } 24\\text{-}48 \\text{ GB}', explanation: 'Video models process 3D tensors, massively increasing activation memory.' },
        ]}
        id="example-vram-requirements"
      />

      <PythonCode
        title="check_gpu_for_image_training.py"
        code={`import torch

def check_image_training_compatibility():
    """Check GPU compatibility for image/video finetuning."""
    if not torch.cuda.is_available():
        print("No CUDA GPU available!")
        return

    gpu = torch.cuda.get_device_properties(0)
    vram_gb = gpu.total_mem / 1e9
    compute = f"{gpu.major}.{gpu.minor}"

    print(f"GPU: {gpu.name}")
    print(f"VRAM: {vram_gb:.1f} GB")
    print(f"Compute capability: {compute}")

    recommendations = []
    if vram_gb >= 8:
        recommendations.append("SD 1.5 LoRA (512x512)")
        recommendations.append("Textual Inversion")
    if vram_gb >= 12:
        recommendations.append("SDXL LoRA (1024x1024)")
        recommendations.append("DreamBooth (SD 1.5)")
    if vram_gb >= 16:
        recommendations.append("FLUX.1 LoRA (with quantization)")
        recommendations.append("DreamBooth (SDXL)")
    if vram_gb >= 24:
        recommendations.append("FLUX.1 LoRA (full precision)")
        recommendations.append("AnimateDiff LoRA")
    if vram_gb >= 48:
        recommendations.append("CogVideoX LoRA")
        recommendations.append("Video generation finetuning")

    print(f"\\nSupported training methods:")
    for r in recommendations:
        print(f"  - {r}")

    # Check for xformers (memory-efficient attention)
    try:
        import xformers
        print(f"\\nxformers: {xformers.__version__} (memory-efficient attention available)")
    except ImportError:
        print("\\nxformers: NOT INSTALLED (recommended for image training)")

check_image_training_compatibility()`}
        id="code-check-gpu"
      />

      <PythonCode
        title="memory_optimization_tips.py"
        code={`# Memory optimization techniques for image/video training

optimizations = {
    "gradient_checkpointing": {
        "savings": "30-50% VRAM",
        "cost": "25-35% slower training",
        "code": "unet.enable_gradient_checkpointing()",
    },
    "mixed_precision_bf16": {
        "savings": "~50% VRAM for activations",
        "cost": "Negligible (Ampere+ GPUs)",
        "code": 'accelerator = Accelerator(mixed_precision="bf16")',
    },
    "xformers_attention": {
        "savings": "20-40% VRAM",
        "cost": "Slightly different outputs (numerically)",
        "code": "unet.enable_xformers_memory_efficient_attention()",
    },
    "8bit_adam": {
        "savings": "~30% optimizer VRAM",
        "cost": "Negligible quality impact",
        "code": 'optimizer = bnb.optim.AdamW8bit(params, lr=1e-4)',
    },
    "train_text_encoder_off": {
        "savings": "~2-4 GB VRAM",
        "cost": "May slightly reduce quality",
        "code": "text_encoder.requires_grad_(False)",
    },
}

for name, info in optimizations.items():
    print(f"\\n{name}:")
    print(f"  Savings: {info['savings']}")
    print(f"  Cost: {info['cost']}")`}
        id="code-optimizations"
      />

      <NoteBlock
        type="tip"
        title="Cloud GPU Recommendations"
        content="For image LoRA training: A10G (24 GB, ~$0.50/hr on Lambda). For DreamBooth or FLUX: A100 40GB (~$1.10/hr). For video finetuning: A100 80GB or H100 (~$2-3/hr). RunPod and vast.ai offer the best prices for spot instances."
        id="note-cloud-gpu"
      />

      <WarningBlock
        title="Video Training Is Resource-Intensive"
        content="Video model finetuning requires 2-10x more VRAM than image models due to temporal dimensions. A single training batch of 16-frame 512x512 video clips can use 40+ GB. Start with short clips (8-16 frames) at lower resolution and scale up."
        id="warning-video-resources"
      />
    </div>
  )
}
