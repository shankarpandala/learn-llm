import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LoRAStacking() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LoRA Stacking in ComfyUI</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        ComfyUI allows chaining multiple LoRA nodes to combine different fine-tuned effects.
        A style LoRA for art direction, a character LoRA for consistent faces, and a detail
        LoRA for texture quality can all be applied simultaneously. Understanding how to
        balance LoRA weights and resolve conflicts is key to effective multi-LoRA workflows.
      </p>

      <DefinitionBlock
        title="LoRA Stacking"
        definition="LoRA stacking applies multiple low-rank updates sequentially to the model weights. Given base weights $\mathbf{W}_0$ and $K$ LoRAs with weights $\alpha_k$, the effective weight is: $\mathbf{W}_{\text{eff}} = \mathbf{W}_0 + \sum_{k=1}^{K} \alpha_k \cdot \mathbf{B}_k \mathbf{A}_k$. In ComfyUI, each LoraLoader node takes MODEL and CLIP inputs and outputs modified MODEL and CLIP."
        id="def-lora-stacking"
      />

      <h2 className="text-2xl font-semibold">Chaining LoRA Nodes</h2>
      <p className="text-gray-700 dark:text-gray-300">
        In ComfyUI, LoRAs are stacked by chaining LoraLoader nodes: the output MODEL and CLIP
        of one LoraLoader feed into the input of the next. Each node applies its LoRA on top
        of the previous result.
      </p>

      <ExampleBlock
        title="Multi-LoRA Setup"
        problem="Stack a style LoRA, character LoRA, and detail LoRA."
        steps={[
          { formula: '\\text{Checkpoint} \\to \\text{LoRA\\_style (0.7)} \\to \\text{LoRA\\_char (0.8)} \\to \\text{LoRA\\_detail (0.5)}', explanation: 'Each LoRA node passes its modified MODEL/CLIP to the next.' },
          { formula: '\\mathbf{W} = \\mathbf{W}_0 + 0.7\\mathbf{B}_1\\mathbf{A}_1 + 0.8\\mathbf{B}_2\\mathbf{A}_2 + 0.5\\mathbf{B}_3\\mathbf{A}_3', explanation: 'Effective weight is base plus all weighted LoRA contributions.' },
          { formula: '\\text{Total LoRA effect} \\leq 2.0 \\text{ recommended}', explanation: 'Sum of weights above 2.0 often causes artifacts or instability.' },
        ]}
        id="example-multi-lora"
      />

      <PythonCode
        title="lora_stacking_workflow.py"
        code={`# Multi-LoRA workflow for ComfyUI
def build_lora_stack_workflow(
    checkpoint="sd_v1-5.safetensors",
    loras=[
        ("anime_style.safetensors", 0.7, 0.7),    # (file, model_strength, clip_strength)
        ("character_face.safetensors", 0.8, 0.6),
        ("detail_enhancer.safetensors", 0.5, 0.3),
    ],
    prompt="1girl, walking in garden, cherry blossoms, masterpiece",
    negative="ugly, blurry, deformed",
    steps=30, cfg=7.0, seed=42,
):
    """Build LoRA stacking workflow."""
    nodes = {}
    node_id = 1

    # Checkpoint loader
    nodes[str(node_id)] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": checkpoint},
        "_meta": {"title": "Base Checkpoint"}
    }
    model_src = [str(node_id), 0]  # MODEL output
    clip_src = [str(node_id), 1]   # CLIP output
    vae_src = [str(node_id), 2]    # VAE output
    node_id += 1

    # Chain LoRA loaders
    for lora_file, model_str, clip_str in loras:
        nodes[str(node_id)] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_src,
                "clip": clip_src,
                "lora_name": lora_file,
                "strength_model": model_str,
                "strength_clip": clip_str,
            },
            "_meta": {"title": f"LoRA: {lora_file}"}
        }
        model_src = [str(node_id), 0]  # Updated MODEL
        clip_src = [str(node_id), 1]   # Updated CLIP
        node_id += 1

    # Rest of the pipeline uses final model_src and clip_src
    nodes[str(node_id)] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": prompt, "clip": clip_src}
    }
    pos_id = str(node_id)
    node_id += 1

    nodes[str(node_id)] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": negative, "clip": clip_src}
    }
    neg_id = str(node_id)
    node_id += 1

    nodes[str(node_id)] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 768, "batch_size": 1}
    }
    latent_id = str(node_id)
    node_id += 1

    nodes[str(node_id)] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_src, "positive": [pos_id, 0],
            "negative": [neg_id, 0], "latent_image": [latent_id, 0],
            "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "dpmpp_2m", "scheduler": "karras",
            "denoise": 1.0,
        }
    }
    sample_id = str(node_id)
    node_id += 1

    nodes[str(node_id)] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [sample_id, 0], "vae": vae_src}
    }
    decode_id = str(node_id)
    node_id += 1

    nodes[str(node_id)] = {
        "class_type": "SaveImage",
        "inputs": {"images": [decode_id, 0], "filename_prefix": "lora_stack"}
    }

    return nodes

workflow = build_lora_stack_workflow()
print(f"LoRA stack workflow: {len(workflow)} nodes")

# Print chain
print("\\nLoRA chain:")
for nid, node in sorted(workflow.items(), key=lambda x: int(x[0])):
    title = node.get("_meta", {}).get("title", node["class_type"])
    print(f"  [{nid}] {title}")`}
        id="code-lora-stacking"
      />

      <NoteBlock
        type="tip"
        title="Weight Balancing"
        content="When LoRAs conflict (e.g., two style LoRAs), reduce both weights proportionally. A good starting approach: dominant LoRA at 0.7-0.9, supplementary LoRAs at 0.3-0.5. Model strength and CLIP strength can be set independently -- higher CLIP strength means stronger prompt influence from that LoRA's training."
        id="note-weight-balance"
      />

      <WarningBlock
        title="LoRA Conflicts"
        content="Stacking too many LoRAs or using total weights above 2.0 often produces muddy, oversaturated, or artifact-prone images. If results degrade, reduce weights or remove conflicting LoRAs. Two LoRAs that modify the same concept (e.g., two face LoRAs) will typically interfere."
        id="warning-conflicts"
      />
    </div>
  )
}
