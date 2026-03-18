import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function SDXLFluxWorkflows() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">SDXL and FLUX Workflows in ComfyUI</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        SDXL and FLUX require different ComfyUI workflows than SD 1.5 due to their distinct
        architectures. SDXL uses dual text encoders and an optional refiner model. FLUX uses
        separate CLIP and T5 text encoders with a completely different model loading approach.
        Understanding these differences is essential for building effective workflows.
      </p>

      <DefinitionBlock
        title="SDXL Dual Encoding"
        definition="SDXL uses two CLIP text encoders: CLIP-L (from SD 1.5) and OpenCLIP-G (larger, more capable). Both encoders' outputs are concatenated to form the conditioning tensor. In ComfyUI, the CLIPTextEncode node automatically handles both encoders when an SDXL checkpoint is loaded."
        id="def-sdxl-dual"
      />

      <h2 className="text-2xl font-semibold">SDXL Base + Refiner</h2>
      <p className="text-gray-700 dark:text-gray-300">
        SDXL optionally uses a two-stage pipeline: the base model generates for the first 80%
        of steps, then the refiner model takes over for the final 20% to add fine details.
        In ComfyUI, this requires two KSampler nodes with coordinated denoise ranges.
      </p>

      <ExampleBlock
        title="FLUX Model Loading"
        problem="How does FLUX model loading differ from SD/SDXL in ComfyUI?"
        steps={[
          { formula: '\\text{SD/SDXL: Single checkpoint contains MODEL + CLIP + VAE}', explanation: 'CheckpointLoaderSimple outputs all three from one file.' },
          { formula: '\\text{FLUX: Separate UNET, CLIP, T5, and VAE files}', explanation: 'Each component loaded by its own node.' },
          { formula: '\\text{FLUX uses DualCLIPLoader for CLIP-L + T5-XXL}', explanation: 'Two text encoders loaded together into a combined CLIP output.' },
        ]}
        id="example-flux-loading"
      />

      <PythonCode
        title="sdxl_flux_workflows.py"
        code={`# SDXL Base + Refiner workflow
def build_sdxl_refiner_workflow(
    base_ckpt="sd_xl_base_1.0.safetensors",
    refiner_ckpt="sd_xl_refiner_1.0.safetensors",
    prompt="A majestic eagle soaring over snow-capped mountains",
    negative="blurry, low quality",
    total_steps=30, switch_at=0.8,
    seed=42, cfg=7.0,
):
    """SDXL with base + refiner handoff."""
    base_steps = int(total_steps * switch_at)
    return {
        # Load base and refiner models
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": base_ckpt}},
        "2": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": refiner_ckpt}},
        # Text encoding (using base CLIP)
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["1", 1]}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["1", 1]}},
        # Refiner text encoding
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["2", 1]}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["2", 1]}},
        # Empty latent at SDXL resolution
        "7": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        # Base sampler (first 80% of steps)
        "8": {"class_type": "KSampler",
              "inputs": {
                  "model": ["1", 0], "positive": ["3", 0],
                  "negative": ["4", 0], "latent_image": ["7", 0],
                  "seed": seed, "steps": total_steps, "cfg": cfg,
                  "sampler_name": "dpmpp_2m", "scheduler": "karras",
                  "denoise": 1.0,
                  "start_at_step": 0, "end_at_step": base_steps,
              }},
        # Refiner sampler (final 20%)
        "9": {"class_type": "KSampler",
              "inputs": {
                  "model": ["2", 0], "positive": ["5", 0],
                  "negative": ["6", 0], "latent_image": ["8", 0],
                  "seed": seed, "steps": total_steps, "cfg": cfg,
                  "sampler_name": "dpmpp_2m", "scheduler": "karras",
                  "denoise": 1.0,
                  "start_at_step": base_steps, "end_at_step": total_steps,
              }},
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0], "filename_prefix": "sdxl_refiner"}},
    }

# FLUX workflow (separate model components)
def build_flux_workflow(
    unet="flux1-dev.safetensors",
    clip_l="clip_l.safetensors",
    t5xxl="t5xxl_fp16.safetensors",
    vae="ae.safetensors",
    prompt="A cat sitting on a windowsill watching rain",
    steps=28, guidance=3.5, seed=42,
):
    """FLUX.1-dev workflow with separate model loading."""
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet, "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": clip_l, "clip_name2": t5xxl,
                         "type": "flux"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": vae}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "6": {"class_type": "KSampler",
              "inputs": {
                  "model": ["1", 0], "positive": ["4", 0],
                  "negative": ["4", 0],  # FLUX: same or empty
                  "latent_image": ["5", 0],
                  "seed": seed, "steps": steps, "cfg": guidance,
                  "sampler_name": "euler", "scheduler": "simple",
                  "denoise": 1.0,
              }},
        "7": {"class_type": "VAEDecode",
              "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0], "filename_prefix": "flux"}},
    }

print("SDXL workflow:", len(build_sdxl_refiner_workflow()), "nodes")
print("FLUX workflow:", len(build_flux_workflow()), "nodes")`}
        id="code-sdxl-flux"
      />

      <NoteBlock
        type="note"
        title="FLUX Negative Prompts"
        content="FLUX.1-dev was trained with guidance distillation, which means it does not use traditional classifier-free guidance with negative prompts. The guidance_scale parameter works differently -- it controls the distilled guidance. Setting it too high (>5) can cause artifacts. Typical range is 2.5-4.0."
        id="note-flux-cfg"
      />

      <WarningBlock
        title="FLUX Memory Requirements"
        content="FLUX requires loading UNET (~12B params), T5-XXL (~4.7B), CLIP-L, and VAE separately. Total VRAM needed: ~24GB in fp16, ~12GB in fp8. Use fp8_e4m3fn weight dtype for the UNET in ComfyUI to fit on 16GB GPUs. T5 can be loaded in fp16 or fp8."
        id="warning-flux-memory"
      />
    </div>
  )
}
