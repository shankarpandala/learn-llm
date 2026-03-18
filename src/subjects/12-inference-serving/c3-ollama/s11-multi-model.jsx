import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function MultiModel() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Multi-Model Serving</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Ollama can load and serve multiple models concurrently. Understanding how Ollama manages
        model loading, unloading, and memory sharing is key to running multiple models efficiently
        on limited hardware.
      </p>

      <DefinitionBlock
        title="Concurrent Model Serving"
        definition="Ollama keeps recently used models in memory and can serve requests to different models simultaneously. Models are loaded on first request and unloaded after an idle timeout (default 5 minutes). The OLLAMA_MAX_LOADED_MODELS environment variable controls how many models stay loaded."
        id="def-multi-model"
      />

      <PythonCode
        title="Terminal"
        code={`# See currently loaded models
ollama ps
# NAME              ID           SIZE    PROCESSOR    UNTIL
# llama3.2:latest   a80c4f17acd5 3.5 GB  100% GPU     4 min
# mistral:latest    f974a74358d6 5.1 GB  100% GPU     2 min

# Configure concurrent models
export OLLAMA_MAX_LOADED_MODELS=3      # Keep up to 3 models in memory
export OLLAMA_KEEP_ALIVE="10m"         # Models stay loaded for 10 minutes
export OLLAMA_NUM_PARALLEL=4           # Handle 4 concurrent requests per model

# Start Ollama with these settings
ollama serve

# Or set keep_alive per request via API
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Hello",
  "keep_alive": "30m"
}'

# Unload a model immediately
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "keep_alive": 0
}'`}
        id="code-multi-model"
      />

      <PythonCode
        title="model_router.py"
        code={`import ollama

# Route requests to different models based on task type
ROUTING_TABLE = {
    "code": "qwen2.5-coder:7b",
    "general": "llama3.2",
    "creative": "llama3.1:8b",
    "math": "deepseek-r1:8b",
    "fast": "llama3.2:1b",
}

def classify_task(prompt):
    """Simple keyword-based task classification."""
    prompt_lower = prompt.lower()
    if any(w in prompt_lower for w in ["code", "function", "program", "debug"]):
        return "code"
    if any(w in prompt_lower for w in ["poem", "story", "creative", "imagine"]):
        return "creative"
    if any(w in prompt_lower for w in ["calculate", "math", "equation", "proof"]):
        return "math"
    return "general"

def smart_generate(prompt, task_type=None):
    """Route to the best model for the task."""
    if task_type is None:
        task_type = classify_task(prompt)
    model = ROUTING_TABLE.get(task_type, ROUTING_TABLE["general"])
    print(f"Routing to {model} (task: {task_type})")

    response = ollama.generate(model=model, prompt=prompt)
    return response["response"]

# Test with different prompts
print(smart_generate("Write a Python function to parse CSV files"))
print(smart_generate("Write a haiku about autumn"))
print(smart_generate("What is the integral of sin(x)?"))`}
        id="code-router"
      />

      <ExampleBlock
        title="Memory Planning for Multi-Model"
        problem="You have 24GB GPU memory. How many models can you serve?"
        steps={[
          { formula: 'CUDA context: ~500MB overhead', explanation: 'Base GPU memory used by the CUDA runtime.' },
          { formula: '3B model (Q4_K_M): ~2GB each', explanation: 'Small models for fast responses.' },
          { formula: '8B model (Q4_K_M): ~5GB each', explanation: 'Medium models for quality responses.' },
          { formula: 'Example: 1x 8B + 2x 3B + KV-cache = 5+4+2 ≈ 11GB', explanation: 'Leaves 13GB for KV-cache and concurrent requests.' },
        ]}
        id="example-memory-planning"
      />

      <NoteBlock
        type="tip"
        title="Preload Models"
        content="Send a request with an empty prompt to preload a model without generating output. This avoids cold-start latency when the first real request arrives. Useful for models you know will be needed soon."
        id="note-preload"
      />

      <WarningBlock
        title="OOM with Multiple Models"
        content="If total model memory exceeds available GPU RAM, Ollama will fall back to CPU for some models or refuse to load. Monitor GPU memory with 'nvidia-smi' and set OLLAMA_MAX_LOADED_MODELS conservatively. It is better to have models swap in/out than to crash from OOM."
        id="warning-oom"
      />
    </div>
  )
}
