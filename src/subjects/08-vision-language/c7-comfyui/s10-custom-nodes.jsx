import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function CustomNodes() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Custom Nodes in ComfyUI</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        ComfyUI's power comes from its extensible custom node ecosystem. Community-built nodes
        add capabilities like IP-Adapter for image-guided generation, face restoration,
        advanced schedulers, and integration with external tools. Understanding how to install,
        use, and even create custom nodes unlocks the full potential of ComfyUI.
      </p>

      <DefinitionBlock
        title="Custom Node"
        definition="A ComfyUI custom node is a Python class that defines inputs, outputs, and a processing function. It is packaged in a directory under custom_nodes/ with an __init__.py that registers the node class. Each node exposes typed input/output slots and a category for the node menu."
        id="def-custom-node"
      />

      <h2 className="text-2xl font-semibold">Essential Custom Node Packs</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The ComfyUI ecosystem has hundreds of custom node packs. A few are considered essential
        for most advanced workflows: ComfyUI-Manager for easy installation, ControlNet
        auxiliary preprocessors, AnimateDiff nodes, and IP-Adapter for image conditioning.
      </p>

      <ExampleBlock
        title="Essential Custom Nodes"
        problem="Which custom node packs should you install first?"
        steps={[
          { formula: '\\text{ComfyUI-Manager: GUI for installing other custom nodes}', explanation: 'Install this first -- it provides a UI for managing all other nodes.' },
          { formula: '\\text{comfyui\\_controlnet\\_aux: ControlNet preprocessors}', explanation: 'Canny, depth, pose, lineart detectors for ControlNet conditioning.' },
          { formula: '\\text{ComfyUI-AnimateDiff-Evolved: Video generation}', explanation: 'AnimateDiff integration with motion modules and video output.' },
          { formula: '\\text{ComfyUI\\_IPAdapter\\_plus: Image-guided generation}', explanation: 'Use reference images to guide style and content.' },
        ]}
        id="example-essential-nodes"
      />

      <PythonCode
        title="custom_node_creation.py"
        code={`# Creating a custom ComfyUI node
# File: custom_nodes/my_nodes/__init__.py

# Minimal custom node example
class ImageBrightnessAdjust:
    """Custom node to adjust image brightness."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),           # Typed input: IMAGE tensor
                "brightness": ("FLOAT", {      # Float parameter with constraints
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)          # Output type
    RETURN_NAMES = ("adjusted_image",) # Output name
    FUNCTION = "adjust"                # Method to call
    CATEGORY = "image/adjustments"     # Menu category

    def adjust(self, image, brightness):
        # image is a torch tensor: (B, H, W, C) in [0, 1]
        import torch
        adjusted = torch.clamp(image * brightness, 0.0, 1.0)
        return (adjusted,)  # Must return a tuple

# Register nodes
NODE_CLASS_MAPPINGS = {
    "ImageBrightnessAdjust": ImageBrightnessAdjust,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBrightnessAdjust": "Brightness Adjust",
}

# Installation via ComfyUI-Manager or git
install_methods = """
# Method 1: ComfyUI-Manager (GUI)
# Click Manager -> Install Custom Nodes -> Search

# Method 2: Git clone
cd ComfyUI/custom_nodes
git clone https://github.com/user/custom-node-pack.git
pip install -r custom-node-pack/requirements.txt

# Method 3: Manual
# Copy Python files to custom_nodes/my_nodes/
# Restart ComfyUI
"""
print(install_methods)

# Popular custom node packs
packs = {
    "ComfyUI-Manager":              "Node management GUI",
    "comfyui_controlnet_aux":       "ControlNet preprocessors",
    "ComfyUI-AnimateDiff-Evolved":  "AnimateDiff video generation",
    "ComfyUI_IPAdapter_plus":       "Image prompt adapter",
    "ComfyUI-Impact-Pack":          "Face detection, SAM, utilities",
    "ComfyUI_UltimateSDUpscale":    "Tiled upscaling workflow",
    "ComfyUI-KJNodes":              "Utility nodes (math, logic, etc)",
    "rgthree-comfy":                "Quality-of-life improvements",
}
for name, desc in packs.items():
    print(f"  {name}: {desc}")`}
        id="code-custom-nodes"
      />

      <NoteBlock
        type="tip"
        title="IP-Adapter for Style Transfer"
        content="IP-Adapter (Image Prompt Adapter) is one of the most powerful custom nodes. It takes a reference image and extracts style/content features that condition the generation. Unlike img2img, IP-Adapter preserves the reference style without constraining the composition. Use strength 0.5-0.8 for style influence."
        id="note-ip-adapter"
      />

      <WarningBlock
        title="Custom Node Stability"
        content="Custom nodes are community-maintained and may break with ComfyUI updates. Pin your ComfyUI version for production use. Some custom nodes may conflict with each other or have incompatible dependencies. Always test in isolation before adding to complex workflows."
        id="warning-stability"
      />
    </div>
  )
}
