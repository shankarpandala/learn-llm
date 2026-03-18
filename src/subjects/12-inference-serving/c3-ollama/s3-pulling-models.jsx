import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function PullingModels() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Pulling & Running Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Ollama uses a Docker-like pull mechanism to download models. Each model has a name and
        optional tag specifying the size and quantization variant. Once pulled, models can be
        run interactively or served via API.
      </p>

      <PythonCode
        title="Terminal"
        code={`# Pull a model (downloads if not already present)
ollama pull llama3.2
# pulling manifest
# pulling dde5aa3fc5ff... 100% 2.0 GB
# pulling 966de95ca8a6... 100% 1.4 KB
# verifying sha256 digest
# writing manifest
# success

# Pull a specific size/quantization variant
ollama pull llama3.2:1b        # 1B parameter version
ollama pull llama3.2:3b        # 3B parameter version
ollama pull llama3.1:8b-q4_0   # 8B with Q4_0 quantization

# Pull other popular models
ollama pull mistral             # Mistral 7B
ollama pull phi3:mini           # Microsoft Phi-3 Mini
ollama pull gemma2:9b           # Google Gemma 2 9B
ollama pull qwen2.5:7b          # Alibaba Qwen 2.5 7B
ollama pull deepseek-r1:8b      # DeepSeek R1 distilled 8B`}
        id="code-pull"
      />

      <PythonCode
        title="Terminal"
        code={`# Run interactively (pulls automatically if not present)
ollama run llama3.2
# >>> Send a message (/? for help)
# Type your prompt and press Enter
# Use /bye to exit

# Run with a one-shot prompt (non-interactive)
ollama run llama3.2 "Explain Docker in one sentence"
# Docker is a platform for building and running applications
# in isolated containers.

# Pipe input from stdin
echo "Translate to French: Hello world" | ollama run llama3.2
# Bonjour le monde

# Use with files
cat code.py | ollama run llama3.2 "Review this Python code:"

# List downloaded models
ollama list
# NAME              ID           SIZE    MODIFIED
# llama3.2:latest   a80c4f17acd5 2.0 GB  5 min ago
# mistral:latest    f974a74358d6 4.1 GB  1 hour ago
# phi3:mini         4f2222927938 2.2 GB  2 hours ago`}
        id="code-run"
      />

      <ExampleBlock
        title="Understanding Model Tags"
        problem="What does 'llama3.1:8b-instruct-q4_K_M' mean?"
        steps={[
          { formula: 'llama3.1 = model family', explanation: 'Meta LLaMA 3.1 model family.' },
          { formula: '8b = 8 billion parameters', explanation: 'The parameter count variant.' },
          { formula: 'instruct = instruction-tuned', explanation: 'Fine-tuned to follow instructions (vs base/raw model).' },
          { formula: 'q4_K_M = 4-bit quantization (K-quant, medium)', explanation: 'Specific quantization format balancing quality and size.' },
        ]}
        id="example-tags"
      />

      <PythonCode
        title="manage_models.py"
        code={`import ollama

# List all local models
models = ollama.list()
for model in models["models"]:
    size_gb = model["size"] / (1024**3)
    print(f"{model['name']:<30} {size_gb:.1f} GB")

# Get model details
info = ollama.show("llama3.2")
print(f"Format: {info.get('details', {}).get('format', 'N/A')}")
print(f"Family: {info.get('details', {}).get('family', 'N/A')}")
print(f"Parameters: {info.get('details', {}).get('parameter_size', 'N/A')}")
print(f"Quantization: {info.get('details', {}).get('quantization_level', 'N/A')}")

# Delete a model to free space
# ollama.delete("mistral:latest")`}
        id="code-manage"
      />

      <NoteBlock
        type="tip"
        title="Resumable Downloads"
        content="If a download is interrupted, running 'ollama pull' again resumes from where it left off. Ollama uses content-addressable storage, so layers shared between models are only downloaded once."
        id="note-resume"
      />

      <WarningBlock
        title="Disk Space Requirements"
        content="Models can be large. A 7B model at Q4_K_M is about 4GB, while a 70B model is 40GB+. Check available disk space before pulling large models. Use 'ollama rm <model>' to delete models you no longer need."
        id="warning-disk"
      />
    </div>
  )
}
