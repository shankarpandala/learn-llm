import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Upscaling() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Upscaling Workflows in ComfyUI</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Upscaling takes a generated or input image and increases its resolution while adding
        detail. ComfyUI supports two main approaches: model-based upscaling (ESRGAN, RealESRGAN)
        for direct super-resolution, and latent upscaling with img2img re-diffusion for
        adding AI-generated detail. The best results combine both in a multi-pass workflow.
      </p>

      <DefinitionBlock
        title="Upscaling Approaches"
        definition="Model-based upscaling uses dedicated super-resolution neural networks (e.g., RealESRGAN 4x) that directly map low-res to high-res pixels. Latent upscaling encodes the upscaled image to latent space and runs a partial diffusion pass (denoise 0.3-0.5) to add AI-generated fine details that pure SR models cannot hallucinate."
        id="def-upscaling"
      />

      <h2 className="text-2xl font-semibold">Two-Pass Upscaling</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The recommended approach: (1) upscale with an ESRGAN model for clean resolution
        increase, (2) run img2img with low denoise to refine details. This avoids the
        blurriness of pure SR while preventing the hallucination artifacts of pure re-diffusion.
      </p>

      <ExampleBlock
        title="Upscaling Pipeline"
        problem="Upscale a 512x512 image to 2048x2048 with detail enhancement."
        steps={[
          { formula: '\\text{Step 1: ESRGAN 4x: } 512 \\to 2048 \\text{ pixels}', explanation: 'Clean upscale preserving structure.' },
          { formula: '\\text{Step 2: VAEEncode(2048x2048)} \\to \\text{latent 256x256}', explanation: 'Encode upscaled image to latent space.' },
          { formula: '\\text{Step 3: KSampler(denoise=0.35)} \\to \\text{refined latent}', explanation: 'Add fine detail via partial re-diffusion.' },
          { formula: '\\text{Step 4: VAEDecode} \\to \\text{2048x2048 detailed output}', explanation: 'Final high-resolution image with enhanced detail.' },
        ]}
        id="example-upscale-pipeline"
      />

      <PythonCode
        title="upscaling_workflow.py"
        code={`# Two-pass upscaling workflow for ComfyUI
def build_upscale_workflow(
    checkpoint="sd_v1-5.safetensors",
    upscale_model="RealESRGAN_x4plus.pth",
    input_image="input/generated.png",
    prompt="highly detailed, sharp focus, 8k uhd",
    negative="blurry, low quality, pixelated",
    denoise=0.35, steps=20, cfg=7.0, seed=42,
):
    """Two-pass upscaling: ESRGAN + img2img refinement."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": checkpoint}},
        # Load upscale model
        "2": {"class_type": "UpscaleModelLoader",
              "inputs": {"model_name": upscale_model}},
        # Load input image
        "3": {"class_type": "LoadImage",
              "inputs": {"image": input_image}},
        # Pass 1: ESRGAN upscale (pixel-space)
        "4": {"class_type": "ImageUpscaleWithModel",
              "inputs": {
                  "upscale_model": ["2", 0],
                  "image": ["3", 0],
              }},
        # Optionally resize to exact target (ESRGAN is fixed 4x)
        "5": {"class_type": "ImageScale",
              "inputs": {
                  "image": ["4", 0],
                  "upscale_method": "lanczos",
                  "width": 2048, "height": 2048,
                  "crop": "center",
              }},
        # Pass 2: Encode to latent for img2img refinement
        "6": {"class_type": "VAEEncode",
              "inputs": {"pixels": ["5", 0], "vae": ["1", 2]}},
        # Text encoding
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["1", 1]}},
        "8": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["1", 1]}},
        # Refinement sampling (low denoise)
        "9": {"class_type": "KSampler",
              "inputs": {
                  "model": ["1", 0],
                  "positive": ["7", 0], "negative": ["8", 0],
                  "latent_image": ["6", 0],
                  "seed": seed, "steps": steps, "cfg": cfg,
                  "sampler_name": "dpmpp_2m", "scheduler": "karras",
                  "denoise": denoise,
              }},
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0], "filename_prefix": "upscaled"}},
    }

workflow = build_upscale_workflow()
print(f"Upscaling workflow: {len(workflow)} nodes")

# Upscale model comparison
models = {
    "RealESRGAN_x4plus":     {"scale": "4x", "quality": "General purpose, good default"},
    "RealESRGAN_x4plus_anime": {"scale": "4x", "quality": "Optimized for anime/illustration"},
    "4x-UltraSharp":        {"scale": "4x", "quality": "Sharp, good for photos"},
    "4x_NMKD-Siax_200k":    {"scale": "4x", "quality": "Balanced, less artifacts"},
    "ESRGAN_4x":             {"scale": "4x", "quality": "Original ESRGAN, classic"},
}

print("\\nUpscale Models:")
for name, info in models.items():
    print(f"  {name}: {info['scale']} - {info['quality']}")

# Denoise guide for upscaling
print("\\nDenoise for Upscaling:")
for d in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    effect = ("minimal change" if d < 0.2 else
              "subtle detail" if d < 0.35 else
              "balanced" if d < 0.45 else
              "significant repainting")
    print(f"  {d:.1f}: {effect}")`}
        id="code-upscaling"
      />

      <NoteBlock
        type="tip"
        title="Tiled Upscaling"
        content="For very large outputs (4K+), use tiled VAE encoding/decoding to avoid VRAM overflow. ComfyUI's VAETile node processes the image in overlapping tiles. Also consider the 'Ultimate SD Upscale' custom node which automates the tile-based img2img refinement process."
        id="note-tiled-upscale"
      />

      <WarningBlock
        title="Over-Sharpening"
        content="Using too high a denoise value (>0.5) during upscale refinement can hallucinate new content that was not in the original image, changing faces or adding unwanted elements. Keep denoise at 0.25-0.4 for detail enhancement without content alteration."
        id="warning-oversharpening"
      />
    </div>
  )
}
