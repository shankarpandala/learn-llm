import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function CoreConcepts() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">ComfyUI Core Concepts</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Understanding ComfyUI's core concepts -- nodes, slots, links, and data types -- is
        essential for building effective workflows. Every generation pipeline is a directed
        acyclic graph (DAG) where data flows from loader nodes through processing nodes to
        output nodes. Each connection carries a specific data type that must match between
        source and destination.
      </p>

      <DefinitionBlock
        title="ComfyUI Data Types"
        definition="ComfyUI uses typed connections between nodes. The primary types are: MODEL (diffusion model weights), CLIP (text encoder), VAE (autoencoder), CONDITIONING (encoded text embeddings), LATENT (latent space tensors), and IMAGE (decoded pixel tensors). Connections are color-coded: purple for MODEL, yellow for CLIP, red for VAE, orange for CONDITIONING, pink for LATENT, green for IMAGE."
        id="def-data-types"
      />

      <h2 className="text-2xl font-semibold">Node Anatomy</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Each node has input slots (left side), output slots (right side), and widget
        parameters (internal controls like dropdowns, sliders, text fields). Inputs can be
        either connected from another node or set via widgets.
      </p>

      <ExampleBlock
        title="KSampler Node Inputs"
        problem="What inputs does the KSampler node require?"
        steps={[
          { formula: '\\text{model: MODEL} \\leftarrow \\text{CheckpointLoader or LoRA output}', explanation: 'The diffusion model to use for denoising.' },
          { formula: '\\text{positive/negative: CONDITIONING} \\leftarrow \\text{CLIPTextEncode}', explanation: 'Encoded text prompts for guidance.' },
          { formula: '\\text{latent\\_image: LATENT} \\leftarrow \\text{EmptyLatent or VAEEncode}', explanation: 'The starting noise or encoded image to denoise.' },
          { formula: '\\text{Widgets: seed, steps, cfg, sampler, scheduler, denoise}', explanation: 'Parameters controlling the sampling process.' },
        ]}
        id="example-ksampler"
      />

      <PythonCode
        title="comfyui_workflow_structure.py"
        code={`# ComfyUI workflow as JSON (the format used for saving/sharing)
import json

# Minimal txt2img workflow
workflow = {
    "3": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_v1-5.safetensors"},
        "_meta": {"title": "Load Checkpoint"}
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "beautiful sunset over mountains, masterpiece",
            "clip": ["3", 1]  # [node_id, output_index]
        },
        "_meta": {"title": "Positive Prompt"}
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "ugly, blurry, low quality",
            "clip": ["3", 1]
        },
        "_meta": {"title": "Negative Prompt"}
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
        "_meta": {"title": "Empty Latent"}
    },
    "8": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["3", 0],          # MODEL output
            "positive": ["6", 0],        # CONDITIONING
            "negative": ["7", 0],        # CONDITIONING
            "latent_image": ["5", 0],    # LATENT
            "seed": 42,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
        },
        "_meta": {"title": "KSampler"}
    },
    "9": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["8", 0],   # LATENT from KSampler
            "vae": ["3", 2],       # VAE from checkpoint
        },
        "_meta": {"title": "VAE Decode"}
    },
    "10": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["9", 0],    # IMAGE from decode
            "filename_prefix": "ComfyUI_output"
        },
        "_meta": {"title": "Save Image"}
    }
}

# Print the workflow DAG
print("Workflow DAG:")
for node_id, node in sorted(workflow.items()):
    title = node.get("_meta", {}).get("title", node["class_type"])
    connections = []
    for key, val in node["inputs"].items():
        if isinstance(val, list):
            src_id, src_slot = val
            src_title = workflow[str(src_id)].get("_meta", {}).get("title", "?")
            connections.append(f"{key} <- {src_title}[{src_slot}]")
    conn_str = ", ".join(connections) if connections else "(no connections)"
    print(f"  [{node_id}] {title}: {conn_str}")

# Save workflow
print(f"\\nWorkflow JSON size: {len(json.dumps(workflow))} bytes")`}
        id="code-workflow-structure"
      />

      <NoteBlock
        type="note"
        title="Checkpoint Outputs"
        content="The CheckpointLoaderSimple node outputs three things: MODEL (index 0), CLIP (index 1), and VAE (index 2). These three outputs feed the rest of the workflow. Understanding this triple output is key to understanding all ComfyUI workflows."
        id="note-checkpoint-outputs"
      />

      <WarningBlock
        title="Type Mismatches"
        content="Connecting incompatible types (e.g., IMAGE to a LATENT input) will cause errors. ComfyUI color-codes connections to help prevent this, but the error messages can be cryptic. If a workflow fails silently, check that all connections have matching types."
        id="warning-type-mismatch"
      />
    </div>
  )
}
