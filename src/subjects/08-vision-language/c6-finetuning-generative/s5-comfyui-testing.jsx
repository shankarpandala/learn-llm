import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ComfyUITesting() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Testing Fine-Tuned Models in ComfyUI</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        After fine-tuning models with DreamBooth, LoRA, or textual inversion, ComfyUI provides
        a visual node-based interface for testing and iterating on results. Its workflow system
        makes it easy to compare different checkpoints, LoRA combinations, and generation
        parameters without writing code.
      </p>

      <DefinitionBlock
        title="ComfyUI Workflow Testing"
        definition="A ComfyUI testing workflow connects model loading nodes (checkpoints, LoRAs, embeddings) with sampling and preview nodes to evaluate fine-tuned models. Workflows can be saved as JSON and shared, ensuring reproducible evaluation across different setups."
        id="def-comfyui-testing"
      />

      <h2 className="text-2xl font-semibold">Loading Custom Models</h2>
      <p className="text-gray-700 dark:text-gray-300">
        ComfyUI supports loading DreamBooth checkpoints, LoRA files, and textual inversion
        embeddings through dedicated nodes. Files are placed in specific directories within
        the ComfyUI installation.
      </p>

      <ExampleBlock
        title="Model File Placement"
        problem="Where do fine-tuned model files go in ComfyUI?"
        steps={[
          { formula: '\\text{DreamBooth: models/checkpoints/my\\_model.safetensors}', explanation: 'Full model checkpoints in the checkpoints folder.' },
          { formula: '\\text{LoRA: models/loras/my\\_style.safetensors}', explanation: 'LoRA adapters in the loras folder.' },
          { formula: '\\text{Textual Inv: models/embeddings/my\\_concept.pt}', explanation: 'Embedding files in the embeddings folder.' },
          { formula: '\\text{VAE: models/vae/custom\\_vae.safetensors}', explanation: 'Custom VAE files for improved decoding.' },
        ]}
        id="example-file-placement"
      />

      <PythonCode
        title="comfyui_testing_workflow.py"
        code={`# ComfyUI workflow as Python API (JSON equivalent)
# This demonstrates the workflow structure for testing fine-tuned models

def create_lora_test_workflow():
    """Build a ComfyUI-style workflow for LoRA A/B testing."""
    workflow = {
        "nodes": {
            # Load base checkpoint
            "checkpoint_loader": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_v1-5.safetensors"}
            },
            # Load LoRA adapter
            "lora_loader": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["checkpoint_loader", "MODEL"],
                    "clip": ["checkpoint_loader", "CLIP"],
                    "lora_name": "my_style_lora.safetensors",
                    "strength_model": 0.8,
                    "strength_clip": 0.8,
                }
            },
            # Text prompts
            "positive": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["lora_loader", "CLIP"],
                    "text": "a beautiful landscape, <my-concept> style"
                }
            },
            "negative": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["lora_loader", "CLIP"],
                    "text": "blurry, low quality, distorted"
                }
            },
            # Sampler
            "sampler": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["lora_loader", "MODEL"],
                    "positive": ["positive", "CONDITIONING"],
                    "negative": ["negative", "CONDITIONING"],
                    "latent_image": ["empty_latent", "LATENT"],
                    "seed": 42,
                    "steps": 30,
                    "cfg": 7.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                }
            },
            "empty_latent": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 512, "height": 512, "batch_size": 1}
            },
            "decode": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["sampler", "LATENT"],
                    "vae": ["checkpoint_loader", "VAE"],
                }
            },
            "preview": {
                "class_type": "PreviewImage",
                "inputs": {"images": ["decode", "IMAGE"]}
            },
        }
    }
    return workflow

# A/B testing helper
def create_ab_test(lora_a, lora_b, prompt, seed=42):
    """Create side-by-side comparison workflow."""
    tests = []
    for lora_name, weight in [(lora_a, 0.8), (lora_b, 0.8)]:
        tests.append({
            "lora": lora_name,
            "weight": weight,
            "prompt": prompt,
            "seed": seed,  # Same seed for fair comparison
            "steps": 30,
            "cfg": 7.5,
        })
    return tests

# Example A/B test
tests = create_ab_test(
    "style_v1.safetensors",
    "style_v2.safetensors",
    "portrait of a woman in a garden, masterpiece"
)
for i, t in enumerate(tests):
    print(f"Test {i+1}: {t['lora']} @ weight {t['weight']}")

# LoRA weight sweep
print("\\nLoRA weight sweep:")
for w in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]:
    print(f"  weight={w}: {'undershoot' if w < 0.5 else 'good' if w <= 1.0 else 'overshoot'}")`}
        id="code-comfyui-workflow"
      />

      <NoteBlock
        type="tip"
        title="Systematic Testing"
        content="Use fixed seeds and identical prompts when comparing models. Vary one parameter at a time: LoRA weight, CFG scale, or sampler. ComfyUI's batch processing lets you generate grids of images across parameter sweeps for visual comparison."
        id="note-systematic-testing"
      />

      <WarningBlock
        title="ComfyUI Version Compatibility"
        content="Custom nodes and workflows may break between ComfyUI updates. Always pin your ComfyUI version during testing campaigns, and document the exact node versions used. Export workflows as JSON for reproducibility."
        id="warning-version-compat"
      />
    </div>
  )
}
