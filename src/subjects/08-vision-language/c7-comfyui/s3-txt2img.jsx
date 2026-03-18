import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Txt2Img() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Text-to-Image Workflows in ComfyUI</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The text-to-image (txt2img) workflow is the foundational ComfyUI pipeline. Starting
        from a text prompt and random noise, the model iteratively denoises a latent tensor to
        produce an image. Understanding this basic workflow is essential before building more
        complex pipelines with ControlNet, LoRA, or multi-pass refinement.
      </p>

      <DefinitionBlock
        title="Classifier-Free Guidance (CFG)"
        definition="CFG steers generation toward the text prompt by combining conditioned and unconditioned noise predictions: $\hat{\epsilon} = \epsilon_{\text{uncond}} + s \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$ where $s$ is the guidance scale. Higher $s$ increases prompt adherence but can reduce diversity and cause artifacts."
        id="def-cfg"
      />

      <h2 className="text-2xl font-semibold">Sampler and Scheduler Selection</h2>
      <p className="text-gray-700 dark:text-gray-300">
        ComfyUI separates the sampling algorithm (how noise is removed per step) from the
        scheduler (how noise levels are distributed across steps). Common samplers include
        Euler, DPM++ 2M, and UniPC. Common schedulers include normal, karras, and exponential.
      </p>

      <ExampleBlock
        title="Sampler Comparison"
        problem="Which sampler to use for different scenarios?"
        steps={[
          { formula: '\\text{euler: Fast, good at 20-30 steps, general purpose}', explanation: 'Default choice for quick generation.' },
          { formula: '\\text{dpmpp\\_2m + karras: Best quality, 20-30 steps}', explanation: 'DPM++ 2M with Karras scheduler produces sharp, detailed results.' },
          { formula: '\\text{euler\\_ancestral: More creative/varied, slightly noisy}', explanation: 'Adds noise during sampling for more diverse outputs.' },
          { formula: '\\text{uni\\_pc: Fast convergence, good at 10-15 steps}', explanation: 'Best for quick previews or when speed matters most.' },
        ]}
        id="example-samplers"
      />

      <PythonCode
        title="txt2img_workflow_api.py"
        code={`# Complete txt2img workflow with parameter exploration
import json

def build_txt2img_workflow(
    checkpoint="sd_v1-5.safetensors",
    positive_prompt="a serene lake surrounded by mountains at golden hour, photorealistic",
    negative_prompt="blurry, low quality, distorted, watermark",
    width=512, height=512,
    steps=25, cfg=7.0,
    sampler="dpmpp_2m", scheduler="karras",
    seed=42, batch_size=1
):
    """Build a complete txt2img ComfyUI workflow."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint}
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": ["1", 1]}
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["1", 1]}
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": batch_size}
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["2", 0],
                "negative": ["3", 0], "latent_image": ["4", 0],
                "seed": seed, "steps": steps, "cfg": cfg,
                "sampler_name": sampler, "scheduler": scheduler,
                "denoise": 1.0,
            }
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": "txt2img"}
        },
    }

# Generate workflow
workflow = build_txt2img_workflow()
print(f"Workflow nodes: {len(workflow)}")

# Parameter sweep for CFG scale
print("\\nCFG Scale Guide:")
cfg_guide = {
    1.0: "No guidance (random, ignores prompt)",
    3.0: "Subtle guidance, creative and loose",
    5.0: "Moderate, good balance for artistic styles",
    7.0: "Standard, good prompt adherence",
    10.0: "Strong, very literal interpretation",
    15.0: "Very strong, may cause saturation/artifacts",
    20.0: "Extreme, usually produces artifacts",
}
for cfg, desc in cfg_guide.items():
    print(f"  CFG {cfg:5.1f}: {desc}")

# Resolution guide for different models
print("\\nResolution Guide:")
resolutions = {
    "SD 1.5":  [(512, 512), (512, 768), (768, 512)],
    "SDXL":    [(1024, 1024), (896, 1152), (1152, 896)],
    "FLUX":    [(1024, 1024), (768, 1344), (1344, 768)],
}
for model, res_list in resolutions.items():
    dims = ", ".join(f"{w}x{h}" for w, h in res_list)
    print(f"  {model}: {dims}")`}
        id="code-txt2img"
      />

      <NoteBlock
        type="tip"
        title="Prompt Engineering Tips"
        content="Place the most important elements at the beginning of the prompt. Use commas to separate concepts. Quality modifiers like 'masterpiece, best quality, detailed' help with many models. SDXL and FLUX respond better to natural language descriptions than keyword-style prompts."
        id="note-prompt-tips"
      />

      <WarningBlock
        title="Resolution Restrictions"
        content="Each model has a native training resolution. Generating far outside this range produces poor results. SD 1.5 works best at 512px, SDXL at 1024px. For other resolutions, generate at native and then upscale. Dimensions should be multiples of 8 (or 64 for some models)."
        id="warning-resolution"
      />
    </div>
  )
}
