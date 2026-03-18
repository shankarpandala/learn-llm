import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ComfyUIIntro() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Introduction to ComfyUI</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        ComfyUI is a powerful node-based visual interface for building and executing Stable
        Diffusion workflows. Unlike button-based UIs like Automatic1111, ComfyUI exposes the
        full diffusion pipeline as a graph of connected nodes, giving users complete control
        over every step of the generation process. This transparency makes it ideal for
        advanced workflows and experimentation.
      </p>

      <DefinitionBlock
        title="ComfyUI"
        definition="ComfyUI is a modular, graph-based UI and backend for diffusion model inference. Each processing step (model loading, text encoding, sampling, VAE decoding) is represented as a node with typed inputs and outputs. Nodes are connected via edges to form a directed acyclic graph (DAG) that defines the generation pipeline."
        id="def-comfyui"
      />

      <h2 className="text-2xl font-semibold">Why ComfyUI?</h2>
      <p className="text-gray-700 dark:text-gray-300">
        ComfyUI has become the de facto standard for advanced Stable Diffusion workflows because
        it offers full pipeline transparency, efficient memory management, workflow sharing via
        JSON, and a growing ecosystem of community custom nodes for every use case.
      </p>

      <ExampleBlock
        title="ComfyUI vs Other UIs"
        problem="Compare ComfyUI with Automatic1111 and Forge."
        steps={[
          { formula: '\\text{A1111: Button-based, beginner-friendly, extension system}', explanation: 'Easy to use but limited in complex pipeline customization.' },
          { formula: '\\text{Forge: Optimized A1111 fork, better memory management}', explanation: 'Faster than A1111 but still limited by the button-based paradigm.' },
          { formula: '\\text{ComfyUI: Node graph, full control, API-ready, reproducible}', explanation: 'Most flexible; workflows are exportable JSON DAGs.' },
        ]}
        id="example-ui-comparison"
      />

      <PythonCode
        title="comfyui_setup.py"
        code={`# ComfyUI installation and setup
import os

# Installation steps
setup_commands = """
# 1. Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\activate   # Windows

# 3. Install PyTorch (CUDA 12.1 example)
pip install torch torchvision torchaudio --index-url \\
    https://download.pytorch.org/whl/cu121

# 4. Install ComfyUI dependencies
pip install -r requirements.txt

# 5. Download models to correct directories
# models/checkpoints/  - SD, SDXL checkpoints
# models/loras/        - LoRA adapters
# models/vae/          - VAE models
# models/embeddings/   - Textual inversions
# models/controlnet/   - ControlNet models
# models/clip/         - CLIP models (for FLUX)

# 6. Run ComfyUI
python main.py --listen 0.0.0.0 --port 8188
"""
print(setup_commands)

# Directory structure
dirs = {
    "models/checkpoints":   "Base model files (.safetensors)",
    "models/loras":         "LoRA adapter files",
    "models/vae":           "VAE model files",
    "models/embeddings":    "Textual inversion embeddings",
    "models/controlnet":    "ControlNet model files",
    "models/clip":          "CLIP text encoder models",
    "models/unet":          "U-Net / DiT model files (FLUX)",
    "custom_nodes":         "Community node packages",
    "input":                "Input images for img2img, etc.",
    "output":               "Generated images output",
}

print("\\nComfyUI Directory Structure:")
for path, desc in dirs.items():
    print(f"  {path:30s} - {desc}")`}
        id="code-comfyui-setup"
      />

      <NoteBlock
        type="tip"
        title="Getting Started"
        content="Start with the default workflow (drag an image onto the canvas to load its workflow metadata). The basic txt2img workflow has just 6 nodes: CheckpointLoader, CLIPTextEncode (positive/negative), EmptyLatentImage, KSampler, and VAEDecode. Understanding these 6 nodes gives you the foundation for everything else."
        id="note-getting-started"
      />

      <WarningBlock
        title="GPU Memory Management"
        content="ComfyUI keeps models in VRAM between runs for speed. If you run out of memory, use the 'Free model memory' option in the queue menu, or add --lowvram or --novram flags when starting ComfyUI. For FLUX models, --lowvram is usually required on consumer GPUs."
        id="warning-memory"
      />
    </div>
  )
}
