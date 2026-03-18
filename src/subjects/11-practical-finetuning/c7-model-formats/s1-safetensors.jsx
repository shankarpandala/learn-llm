import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Safetensors() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Safetensors Format</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Safetensors is the standard format for storing and sharing model weights in the Hugging Face
        ecosystem. It replaces the older pickle-based .bin format with a safe, fast, memory-mapped
        file format that prevents arbitrary code execution.
      </p>

      <DefinitionBlock
        title="Safetensors"
        definition="Safetensors is a binary format for storing tensors that is safe (no arbitrary code execution), fast (supports memory mapping for instant loading), and simple (header + raw tensor data). Files use the .safetensors extension and contain a JSON header followed by contiguous tensor data."
        id="def-safetensors"
      />

      <h2 className="text-2xl font-semibold">Why Safetensors Over .bin</h2>

      <ExampleBlock
        title="Safetensors vs PyTorch .bin"
        problem="Why did the ecosystem move from .bin to .safetensors?"
        steps={[
          { formula: '\\text{Security: no pickle} \\Rightarrow \\text{no code execution}', explanation: 'PyTorch .bin files use pickle, which can execute arbitrary Python code on load. Safetensors is just data.' },
          { formula: '\\text{Speed: memory-mapped loading}', explanation: 'Safetensors uses mmap, loading tensors lazily. 10-100x faster for large models.' },
          { formula: '\\text{Sharding: built-in multi-file support}', explanation: 'Large models split across multiple .safetensors files with an index.json.' },
          { formula: '\\text{Zero-copy: no deserialization}', explanation: 'Tensors are stored in their native format and can be used directly from disk.' },
        ]}
        id="example-safetensors-vs-bin"
      />

      <PythonCode
        title="working_with_safetensors.py"
        code={`from safetensors import safe_open
from safetensors.torch import save_file, load_file
import torch

# Save tensors to safetensors format
tensors = {
    "weight": torch.randn(1024, 768),
    "bias": torch.zeros(1024),
    "embedding": torch.randn(32000, 4096),
}

save_file(tensors, "model.safetensors")
print("Saved model.safetensors")

# Load tensors (fast, memory-mapped)
loaded = load_file("model.safetensors")
print(f"Loaded keys: {list(loaded.keys())}")
print(f"Weight shape: {loaded['weight'].shape}")

# Inspect without loading all tensors into memory
with safe_open("model.safetensors", framework="pt") as f:
    # Check available tensors
    print(f"Tensor names: {f.keys()}")

    # Load individual tensors on demand
    weight = f.get_tensor("weight")
    print(f"Weight dtype: {weight.dtype}, shape: {weight.shape}")

    # Get metadata
    metadata = f.metadata()
    print(f"Metadata: {metadata}")

# Convert from .bin to .safetensors
# from transformers import AutoModel
# model = AutoModel.from_pretrained("model-path")
# model.save_pretrained("model-path", safe_serialization=True)`}
        id="code-safetensors"
      />

      <PythonCode
        title="inspect_model_files.py"
        code={`import json
import os

def inspect_model_dir(model_path):
    """Inspect a Hugging Face model directory."""
    files = os.listdir(model_path)

    safetensors = [f for f in files if f.endswith('.safetensors')]
    bin_files = [f for f in files if f.endswith('.bin')]
    json_files = [f for f in files if f.endswith('.json')]

    print(f"Model directory: {model_path}")
    print(f"Safetensors files: {len(safetensors)}")
    print(f"PyTorch bin files: {len(bin_files)}")
    print(f"JSON files: {json_files}")

    # Check model index for sharded models
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        total_size = index.get("metadata", {}).get("total_size", 0)
        print(f"Total model size: {total_size / 1e9:.1f} GB")
        print(f"Number of shards: {len(set(index['weight_map'].values()))}")

    # Check config for model architecture
    config_file = os.path.join(model_path, "config.json")
    if os.path.exists(config_file):
        with open(config_file) as f:
            config = json.load(f)
        print(f"Architecture: {config.get('architectures', ['unknown'])}")
        print(f"Hidden size: {config.get('hidden_size', 'N/A')}")
        print(f"Num layers: {config.get('num_hidden_layers', 'N/A')}")

# inspect_model_dir("/path/to/your/model")`}
        id="code-inspect-model"
      />

      <NoteBlock
        type="note"
        title="Safetensors Is the Default"
        content="As of transformers 4.39+, save_pretrained() uses safetensors by default. All new models on Hugging Face Hub use safetensors. The library automatically handles both formats: if safetensors files exist, they are preferred over .bin files."
        id="note-default-format"
      />

      <WarningBlock
        title="Do Not Trust .bin Files from Unknown Sources"
        content="PyTorch .bin files use Python's pickle module, which can execute arbitrary code when loaded. Never load .bin files from untrusted sources. Always prefer safetensors format. If you must use .bin files, scan them with tools like picklescan before loading."
        id="warning-pickle-security"
      />
    </div>
  )
}
