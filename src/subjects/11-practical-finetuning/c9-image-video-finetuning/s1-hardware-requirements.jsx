import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function HardwareRequirements() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Hardware for Diffusion Fine-tuning</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Fine-tuning diffusion models for image and video generation has different hardware
        requirements than LLM fine-tuning. VRAM is the primary bottleneck, and the
        requirements vary dramatically between methods like DreamBooth, LoRA, and
        Textual Inversion.
      </p>

      <DefinitionBlock
        title="VRAM Requirements for Diffusion Training"
        definition="Diffusion model training stores the model weights, optimizer states, gradients, and intermediate activations in GPU memory. For a model with $P$ parameters at mixed precision, the minimum VRAM is approximately $\text{VRAM} \approx 2P + 8P_{\text{trainable}} + A$ bytes, where $A$ covers activations and batch data."
        id="def-vram-diffusion"
      />

      <ExampleBlock
        title="VRAM by Method and Model"
        problem="How much VRAM do different diffusion fine-tuning methods require?"
        steps={[
          { formula: '\\text{SD 1.5 LoRA: } \\sim 6\\text{ GB}', explanation: 'Stable Diffusion 1.5 with LoRA rank 4-8 fits on consumer GPUs easily.' },
          { formula: '\\text{SDXL LoRA: } \\sim 12\\text{ GB}', explanation: 'SDXL is larger; fits on RTX 3060 12GB or better with gradient checkpointing.' },
          { formula: '\\text{SDXL DreamBooth: } \\sim 24\\text{ GB}', explanation: 'Full DreamBooth training requires RTX 3090/4090 or A5000.' },
          { formula: '\\text{Flux LoRA: } \\sim 16\\text{-}24\\text{ GB}', explanation: 'Flux.1 models are large; LoRA training needs RTX 4090 or A100.' },
          { formula: '\\text{Video (AnimateDiff/CogVideoX): } \\sim 24\\text{-}80\\text{ GB}', explanation: 'Video models need A100 80GB or multi-GPU setups.' },
        ]}
        id="example-vram-table"
      />

      <PythonCode
        title="check_gpu_setup.py"
        code={`import torch

def check_gpu_capabilities():
    """Check GPU setup for diffusion model training."""
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected!")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}\\n")

    total_vram = 0
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_mem / 1024**3
        total_vram += vram_gb
        print(f"GPU {i}: {props.name}")
        print(f"  VRAM:         {vram_gb:.1f} GB")
        print(f"  Compute:      {props.major}.{props.minor}")
        print(f"  SMs:          {props.multi_processor_count}")

    gpu_vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"\\n--- Recommendations for {gpu_vram:.0f} GB VRAM ---")

    methods = []
    if gpu_vram >= 6:
        methods.append("SD 1.5 LoRA / Textual Inversion")
    if gpu_vram >= 12:
        methods.append("SDXL LoRA (with gradient checkpointing)")
    if gpu_vram >= 16:
        methods.append("Flux LoRA (fp8 or quantized)")
    if gpu_vram >= 24:
        methods.append("SDXL DreamBooth / Flux LoRA (bf16)")
        methods.append("AnimateDiff LoRA (short clips)")
    if gpu_vram >= 48:
        methods.append("CogVideoX LoRA")
    if gpu_vram >= 80:
        methods.append("Full diffusion model fine-tuning")
        methods.append("Video model DreamBooth")

    for m in methods:
        print(f"  OK: {m}")

    if gpu_vram < 6:
        print("  Insufficient VRAM for local training.")
        print("    Consider: RunPod, Vast.ai, or Google Colab Pro")

check_gpu_capabilities()

# Check xformers (memory-efficient attention)
try:
    import xformers
    print(f"\\nxformers {xformers.__version__} installed")
except ImportError:
    print("\\nxformers not installed - install for 20-30% VRAM savings:")
    print("  pip install xformers")`}
        id="code-check-gpu"
      />

      <PythonCode
        title="cloud_gpu_pricing.py"
        code={`# Quick reference: cloud GPU pricing for diffusion training

CLOUD_OPTIONS = {
    "RunPod": {
        "RTX 4090 (24 GB)":  {"hourly": 0.44, "good_for": "SDXL LoRA/DreamBooth"},
        "A100 80GB":         {"hourly": 1.64, "good_for": "Video models, large batches"},
        "H100 80GB":         {"hourly": 3.29, "good_for": "Fast training"},
    },
    "Vast.ai": {
        "RTX 4090 (24 GB)":  {"hourly": 0.30, "good_for": "Budget SDXL training"},
        "A100 40GB":         {"hourly": 0.90, "good_for": "Flux LoRA, AnimateDiff"},
    },
    "Lambda Labs": {
        "A100 80GB":         {"hourly": 1.10, "good_for": "Reliable longer runs"},
        "H100 80GB":         {"hourly": 2.49, "good_for": "Production pipelines"},
    },
}

def estimate_training_cost(method, target_steps=1000):
    """Estimate cloud training cost."""
    time_per_1k = {
        "sd15_lora":        {"gpu": "RTX 4090", "minutes": 15},
        "sdxl_lora":        {"gpu": "RTX 4090", "minutes": 30},
        "sdxl_dreambooth":  {"gpu": "RTX 4090", "minutes": 45},
        "flux_lora":        {"gpu": "A100 80GB", "minutes": 60},
        "animatediff_lora": {"gpu": "A100 80GB", "minutes": 120},
        "cogvideox_lora":   {"gpu": "A100 80GB", "minutes": 180},
    }

    info = time_per_1k.get(method, {"gpu": "A100 80GB", "minutes": 60})
    hours = (info["minutes"] * target_steps / 1000) / 60
    cost = hours * 1.64
    print(f"{method}: ~{hours:.1f}h on {info['gpu']} = ~USD{cost:.2f}")

for method in ["sd15_lora", "sdxl_lora", "flux_lora", "cogvideox_lora"]:
    estimate_training_cost(method)`}
        id="code-pricing"
      />

      <NoteBlock
        type="tip"
        title="Gradient Checkpointing Saves VRAM"
        content="Enable gradient checkpointing to trade compute time for memory. This recomputes intermediate activations during the backward pass instead of storing them, typically saving 30-50% VRAM at the cost of ~20% slower training. Almost always worth it for consumer GPUs."
        id="note-grad-checkpoint"
      />

      <WarningBlock
        title="VRAM Estimates Are Approximate"
        content="Actual VRAM usage depends on image resolution, batch size, gradient accumulation steps, and whether xformers or flash attention is enabled. Always test with a single training step before committing to a long run. Use torch.cuda.max_memory_allocated() to measure actual peak usage."
        id="warning-vram-estimates"
      />
    </div>
  )
}
