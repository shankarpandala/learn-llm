import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Img2Img() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Image-to-Image Workflows</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Image-to-image (img2img) generation starts from an existing image rather than pure
        noise. The image is encoded to latent space, noise is added to a specified level
        (controlled by the denoise parameter), and then the model denoises it guided by a text
        prompt. This enables style transfer, image editing, sketch-to-photo conversion, and
        iterative refinement of generated images.
      </p>

      <DefinitionBlock
        title="Denoise Strength"
        definition="The denoise parameter $d \in [0, 1]$ controls how much of the original image is preserved. At $d = 0$, the output is the original image. At $d = 1.0$, the image is fully replaced with noise (equivalent to txt2img). In latent space: $\mathbf{z}_{\text{start}} = \sqrt{\bar{\alpha}_{t_d}} \cdot \mathbf{z}_0 + \sqrt{1 - \bar{\alpha}_{t_d}} \cdot \boldsymbol{\epsilon}$ where $t_d = \lfloor d \cdot T \rfloor$."
        id="def-denoise"
      />

      <h2 className="text-2xl font-semibold">ComfyUI Img2Img Pipeline</h2>
      <p className="text-gray-700 dark:text-gray-300">
        In ComfyUI, img2img replaces the EmptyLatentImage node with a LoadImage + VAEEncode
        chain. The KSampler's denoise parameter is set below 1.0 to preserve the input image
        structure while applying the style/content from the prompt.
      </p>

      <ExampleBlock
        title="Denoise Strength Effects"
        problem="How does denoise strength affect the output?"
        steps={[
          { formula: 'd = 0.2: \\text{Subtle changes, color correction, minor style}', explanation: 'Mostly preserves the original, good for refinement.' },
          { formula: 'd = 0.5: \\text{Balanced, keeps composition but changes style}', explanation: 'Good starting point for style transfer.' },
          { formula: 'd = 0.7: \\text{Major changes, only broad structure preserved}', explanation: 'Good for converting sketches to detailed images.' },
          { formula: 'd = 1.0: \\text{Complete regeneration (same as txt2img)}', explanation: 'Original image has no influence; only dimensions matter.' },
        ]}
        id="example-denoise"
      />

      <PythonCode
        title="img2img_workflow.py"
        code={`# ComfyUI img2img workflow
def build_img2img_workflow(
    checkpoint="sd_v1-5.safetensors",
    image_path="input/photo.png",
    prompt="oil painting style, masterpiece",
    negative="ugly, blurry",
    denoise=0.6,
    steps=30, cfg=7.0,
    sampler="dpmpp_2m", scheduler="karras",
    seed=42
):
    """Build img2img workflow for ComfyUI."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint}
        },
        # Load input image instead of empty latent
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": image_path}
        },
        # Encode image to latent space
        "3": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["2", 0],  # IMAGE from LoadImage
                "vae": ["1", 2],     # VAE from checkpoint
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["1", 1]}
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 1]}
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["3", 0],  # Encoded input image
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": denoise,  # Key difference from txt2img
            }
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["6", 0], "vae": ["1", 2]}
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {"images": ["7", 0], "filename_prefix": "img2img"}
        },
    }

# Build workflow
workflow = build_img2img_workflow(denoise=0.6)
print(f"Img2Img workflow: {len(workflow)} nodes")

# Common img2img use cases
use_cases = {
    "Style Transfer":     {"denoise": 0.5, "desc": "Apply art style to photo"},
    "Sketch to Photo":    {"denoise": 0.75, "desc": "Convert rough sketch to detailed image"},
    "Color Correction":   {"denoise": 0.2, "desc": "Adjust colors/lighting"},
    "Inpainting Prep":    {"denoise": 0.4, "desc": "Refine specific areas"},
    "Upscale Refinement": {"denoise": 0.3, "desc": "Add detail to upscaled image"},
}

print("\\nImg2Img Use Cases:")
for name, info in use_cases.items():
    print(f"  {name}: denoise={info['denoise']} - {info['desc']}")`}
        id="code-img2img"
      />

      <NoteBlock
        type="tip"
        title="Iterative Refinement"
        content="Chain multiple img2img passes with decreasing denoise strength (0.7 -> 0.5 -> 0.3) for progressive refinement. Each pass adds more detail while preserving the structure established in previous passes. This 'multi-pass' approach often produces better results than a single high-denoise pass."
        id="note-iterative"
      />

      <WarningBlock
        title="Resolution Matching"
        content="The input image should match the model's native resolution or be a multiple of 64. If the input image has a different resolution, resize it before VAE encoding. Mismatched resolutions can cause the VAE to produce artifacts, especially at the image borders."
        id="warning-resolution-match"
      />
    </div>
  )
}
