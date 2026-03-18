import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Performance() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">ComfyUI Performance Optimization</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Optimizing ComfyUI performance involves reducing VRAM usage, accelerating inference,
        and efficiently managing model loading. Key techniques include model quantization,
        attention optimization, VAE tiling, and strategic use of CPU offloading. These
        optimizations can make the difference between running models at all on consumer
        hardware versus needing expensive cloud GPUs.
      </p>

      <DefinitionBlock
        title="Performance Dimensions"
        definition="ComfyUI performance has three key dimensions: (1) VRAM usage -- how much GPU memory models and intermediates consume, (2) inference speed -- time per generation step, measured in iterations/second, and (3) model loading time -- how quickly checkpoints are swapped. Optimizing one often trades off against another."
        id="def-performance"
      />

      <h2 className="text-2xl font-semibold">VRAM Optimization</h2>
      <p className="text-gray-700 dark:text-gray-300">
        VRAM is the primary bottleneck for most users. ComfyUI provides several mechanisms
        to reduce memory usage at the cost of some speed.
      </p>

      <ExampleBlock
        title="VRAM Usage by Model"
        problem="How much VRAM do different models need in various precision modes?"
        steps={[
          { formula: '\\text{SD 1.5 (fp16): } \\sim 3.5\\text{GB model} + 1\\text{GB working} = 4.5\\text{GB}', explanation: 'Fits on most modern GPUs.' },
          { formula: '\\text{SDXL (fp16): } \\sim 6.5\\text{GB model} + 2\\text{GB working} = 8.5\\text{GB}', explanation: 'Needs 10GB+ GPU for comfortable use.' },
          { formula: '\\text{FLUX (fp8): } \\sim 12\\text{GB model} + 3\\text{GB working} = 15\\text{GB}', explanation: 'Requires fp8 quantization for consumer GPUs.' },
          { formula: '\\text{FLUX (fp16): } \\sim 24\\text{GB} \\to \\text{needs A5000/3090/4090}', explanation: 'Full precision requires high-end GPU or offloading.' },
        ]}
        id="example-vram"
      />

      <PythonCode
        title="performance_optimization.py"
        code={`# ComfyUI performance optimization techniques

# 1. Command-line flags for memory management
cli_flags = {
    "--lowvram":   "Aggressively offload to CPU, ~3x slower but uses minimal VRAM",
    "--novram":    "Everything on CPU except active computation, very slow",
    "--gpu-only":  "Keep everything in VRAM (fast, needs lots of VRAM)",
    "--highvram":  "Disable smart memory management (fastest if you have 24GB+)",
    "--fp8_e4m3fn-unet": "Load UNet in fp8 (halves UNet VRAM)",
    "--fp16-vae":  "Run VAE in fp16 instead of fp32",
    "--bf16-vae":  "Run VAE in bf16 (Ampere+ GPUs)",
}

print("ComfyUI CLI Flags:")
for flag, desc in cli_flags.items():
    print(f"  {flag:25s} {desc}")

# 2. Attention optimization
attention_modes = {
    "pytorch":     "Default PyTorch attention",
    "xformers":    "xformers memory-efficient attention (install separately)",
    "sdp":         "PyTorch 2.0+ scaled dot product attention (recommended)",
}

print("\\nAttention Backends:")
for mode, desc in attention_modes.items():
    print(f"  {mode}: {desc}")

# 3. VRAM estimation helper
def estimate_vram(model_params_B, precision="fp16", working_memory_GB=2.0):
    """Estimate VRAM needed for a model."""
    bytes_per_param = {
        "fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5
    }
    model_gb = model_params_B * bytes_per_param[precision] / 1e9
    total = model_gb + working_memory_GB
    return model_gb, total

models = [
    ("SD 1.5 UNet", 0.86),
    ("SDXL UNet", 2.6),
    ("FLUX DiT", 12.0),
    ("T5-XXL", 4.7),
    ("CLIP-L", 0.12),
]

print("\\nVRAM Estimates:")
for name, params in models:
    for prec in ["fp16", "fp8"]:
        model_gb, total = estimate_vram(params, prec)
        print(f"  {name} ({prec}): model={model_gb:.1f}GB, total~{total:.1f}GB")

# 4. Generation speed benchmarks (approximate)
print("\\nApprox. Generation Speed (RTX 4090, 512x512, 20 steps):")
benchmarks = {
    "SD 1.5 (fp16)":    "~2.5 seconds",
    "SD 1.5 + LoRA":    "~2.7 seconds",
    "SDXL (fp16)":      "~8 seconds at 1024x1024",
    "FLUX (fp8)":       "~15 seconds at 1024x1024",
    "AnimateDiff 16f":  "~20 seconds",
}
for model, speed in benchmarks.items():
    print(f"  {model}: {speed}")

# 5. Model caching strategy
print("\\nModel Caching Tips:")
tips = [
    "ComfyUI caches the last-used model in VRAM automatically",
    "Switching models triggers unload + reload (slow for large models)",
    "Use 'Keep Models in Memory' for workflows using multiple models",
    "For batch jobs with same model: queue all at once to avoid reloading",
    "FLUX: preload T5 and keep it; swap only the UNET if comparing versions",
]
for i, tip in enumerate(tips, 1):
    print(f"  {i}. {tip}")`}
        id="code-performance"
      />

      <NoteBlock
        type="tip"
        title="Quick Performance Wins"
        content="(1) Use --fp16-vae flag to halve VAE memory. (2) Enable PyTorch 2.0 SDP attention (default in recent ComfyUI). (3) Use fp8 UNet loading for FLUX. (4) Enable VAE tiling for images above 1024px. (5) Reduce preview frequency in settings. These five changes alone can reduce VRAM usage by 30-50%."
        id="note-quick-wins"
      />

      <NoteBlock
        type="note"
        title="torch.compile Support"
        content="Recent ComfyUI versions support torch.compile() for the UNet/DiT, which can speed up inference by 10-30% after an initial compilation delay. Add --use-pytorch-cross-attention --force-channels-last for best results. This requires PyTorch 2.0+ and works best on Ampere/Ada GPUs."
        id="note-torch-compile"
      />

      <WarningBlock
        title="Quantization Artifacts"
        content="FP8 and INT8 quantization can introduce subtle quality degradation, particularly in skin tones, fine textures, and color gradients. Always compare quantized vs full-precision outputs for quality-sensitive applications. NF4 quantization (4-bit) has more visible artifacts and is best for previews only."
        id="warning-quantization"
      />
    </div>
  )
}
