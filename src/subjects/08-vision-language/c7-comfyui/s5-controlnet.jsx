import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ControlNet() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">ControlNet in ComfyUI</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        ControlNet adds spatial conditioning to diffusion models through edge maps, depth
        maps, pose skeletons, segmentation maps, and other structural guides. It enables
        precise control over the generated image's composition while the text prompt controls
        style and content. In ComfyUI, ControlNet is applied via dedicated nodes that inject
        conditioning into the U-Net's encoder.
      </p>

      <DefinitionBlock
        title="ControlNet"
        definition="ControlNet creates a trainable copy of the U-Net encoder blocks that processes a conditioning image $\mathbf{c}_{\text{spatial}}$ (e.g., Canny edges). The copy's outputs are added to the main U-Net's skip connections with a learned zero-initialized scaling: $\mathbf{y}_i = \mathbf{F}_i(\mathbf{x}) + \alpha_i \cdot \mathbf{G}_i(\mathbf{c}_{\text{spatial}})$ where $\alpha_i$ starts at zero."
        id="def-controlnet"
      />

      <h2 className="text-2xl font-semibold">ControlNet Types</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Different ControlNet models accept different types of conditioning images, each
        controlling different aspects of the output.
      </p>

      <ExampleBlock
        title="ControlNet Conditioning Types"
        problem="What spatial conditions can ControlNet use?"
        steps={[
          { formula: '\\text{Canny: Edge detection} \\to \\text{controls object boundaries}', explanation: 'Sharp edges from input image guide structural outlines.' },
          { formula: '\\text{Depth: Monocular depth map} \\to \\text{controls spatial layout}', explanation: 'MiDaS or Depth Anything depth estimation for 3D structure.' },
          { formula: '\\text{OpenPose: Skeleton keypoints} \\to \\text{controls human pose}', explanation: 'Stick figure skeletons guide body positions.' },
          { formula: '\\text{Lineart/Scribble: Line drawing} \\to \\text{sketch to image}', explanation: 'Convert rough drawings to detailed rendered images.' },
        ]}
        id="example-controlnet-types"
      />

      <PythonCode
        title="controlnet_workflow.py"
        code={`# ControlNet workflow in ComfyUI
def build_controlnet_workflow(
    checkpoint="sd_v1-5.safetensors",
    controlnet_model="control_v11p_sd15_canny.safetensors",
    control_image="input/canny_edges.png",
    prompt="beautiful landscape painting, oil on canvas",
    negative="ugly, blurry",
    strength=1.0,
    steps=30, cfg=7.0, seed=42,
):
    """ControlNet txt2img workflow."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint}
        },
        # Load ControlNet model
        "2": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": controlnet_model}
        },
        # Load conditioning image
        "3": {
            "class_type": "LoadImage",
            "inputs": {"image": control_image}
        },
        # Text encoding
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["1", 1]}
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 1]}
        },
        # Apply ControlNet to positive conditioning
        "6": {
            "class_type": "ControlNetApply",
            "inputs": {
                "conditioning": ["4", 0],       # Positive CONDITIONING
                "control_net": ["2", 0],         # CONTROL_NET model
                "image": ["3", 0],               # Control IMAGE
                "strength": strength,            # How strongly to apply
            }
        },
        # Standard sampling
        "7": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1}
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["6", 0],   # ControlNet-enhanced conditioning
                "negative": ["5", 0],
                "latent_image": ["7", 0],
                "seed": seed, "steps": steps, "cfg": cfg,
                "sampler_name": "dpmpp_2m", "scheduler": "karras",
                "denoise": 1.0,
            }
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["1", 2]}
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {"images": ["9", 0], "filename_prefix": "controlnet"}
        },
    }

workflow = build_controlnet_workflow()
print(f"ControlNet workflow: {len(workflow)} nodes")

# ControlNet preprocessor nodes (from ControlNet Aux)
preprocessors = {
    "CannyEdgePreprocessor":    {"low": 100, "high": 200},
    "DepthAnythingPreprocessor": {"model": "depth_anything_vitl14"},
    "OpenposePreprocessor":     {"detect_body": True, "detect_hand": True},
    "LineartPreprocessor":      {"coarse": False},
    "ScribblePreprocessor":     {},
    "NormalMapPreprocessor":    {"bg_threshold": 0.4},
    "SegmentAnythingPreprocessor": {"model": "sam_vit_h"},
}

print("\\nAvailable preprocessors:")
for name, params in preprocessors.items():
    print(f"  {name}: {params}")`}
        id="code-controlnet"
      />

      <NoteBlock
        type="tip"
        title="Multi-ControlNet"
        content="You can chain multiple ControlNet nodes: apply Canny for edges AND depth for 3D layout AND pose for characters. Each ControlNetApply takes the previous conditioning as input. Reduce individual strengths (0.5-0.7 each) when stacking to avoid over-constraining the generation."
        id="note-multi-controlnet"
      />

      <WarningBlock
        title="ControlNet Model Matching"
        content="ControlNet models must match the base model version: SD 1.5 ControlNets only work with SD 1.5 checkpoints. SDXL has its own ControlNet models. Using mismatched versions produces garbage output without any error message. Always verify model compatibility."
        id="warning-model-match"
      />
    </div>
  )
}
