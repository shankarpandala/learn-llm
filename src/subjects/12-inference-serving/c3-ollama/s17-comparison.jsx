import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Comparison() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Ollama vs llama.cpp vs LM Studio</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Ollama is not the only way to run LLMs locally. llama.cpp provides the raw inference
        engine, while LM Studio offers a full desktop GUI. Understanding the tradeoffs helps you
        choose the right tool for your workflow.
      </p>

      <DefinitionBlock
        title="The Local LLM Stack"
        definition="llama.cpp is the C++ inference engine that powers both Ollama and LM Studio. Ollama wraps it with model management and an API server. LM Studio wraps it with a graphical interface and model discovery. All three use the same GGUF format and produce identical outputs for the same model and parameters."
        id="def-stack"
      />

      <ExampleBlock
        title="Feature Comparison"
        problem="When should you use each tool?"
        steps={[
          { formula: 'Ollama: API-first, CLI, Docker, automation', explanation: 'Best for developers building applications. Simple CLI, REST API, scriptable.' },
          { formula: 'llama.cpp: maximum control, custom builds, research', explanation: 'Best when you need specific compile flags, custom kernels, or cutting-edge features.' },
          { formula: 'LM Studio: GUI, visual model browser, no coding', explanation: 'Best for non-developers or quick model exploration. Download and chat in clicks.' },
        ]}
        id="example-when-to-use"
      />

      <PythonCode
        title="Terminal"
        code={`# Ollama: simplest workflow
ollama run llama3.2 "Hello world"

# llama.cpp: most control
# Build from source:
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build

# Run with explicit parameters:
./build/bin/llama-cli \\
    -m ./models/llama-3.2-3b-q4_K_M.gguf \\
    -p "Hello world" \\
    --n-gpu-layers 99 \\
    --ctx-size 4096 \\
    --temp 0.7 \\
    --top-p 0.9 \\
    --repeat-penalty 1.1

# llama.cpp server (OpenAI-compatible)
./build/bin/llama-server \\
    -m ./models/llama-3.2-3b-q4_K_M.gguf \\
    --host 0.0.0.0 --port 8080 \\
    --n-gpu-layers 99 \\
    --ctx-size 4096 \\
    --parallel 4

# LM Studio: download from https://lmstudio.ai
# No CLI - it is a desktop application with GUI`}
        id="code-comparison"
      />

      <PythonCode
        title="benchmark_comparison.py"
        code={`import requests
import time

# Benchmark Ollama vs llama.cpp server on the same model
ENDPOINTS = {
    "Ollama": "http://localhost:11434/v1/chat/completions",
    "llama.cpp": "http://localhost:8080/v1/chat/completions",
    # LM Studio also exposes an OpenAI-compatible endpoint
    "LM Studio": "http://localhost:1234/v1/chat/completions",
}

prompt = "Explain the difference between TCP and UDP in 3 sentences."

for name, url in ENDPOINTS.items():
    try:
        start = time.time()
        resp = requests.post(url, json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.7,
        }, timeout=30)
        elapsed = time.time() - start

        if resp.status_code == 200:
            data = resp.json()
            tokens = data.get("usage", {}).get("completion_tokens", "?")
            print(f"{name:<12} {elapsed:.2f}s  {tokens} tokens")
        else:
            print(f"{name:<12} Error: {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"{name:<12} Not running")

# Note: Performance should be very similar since they all use llama.cpp
# Differences come from default settings, batching, and overhead`}
        id="code-benchmark"
      />

      <NoteBlock
        type="note"
        title="Detailed Comparison"
        content="Ollama: auto GPU detection, model management, Docker support, no GUI. llama.cpp: compile-time optimizations, grammar constraints, LoRA hot-loading, batch API. LM Studio: model discovery UI, chat interface, parameter sliders, export conversations. All support the same GGUF models and OpenAI-compatible APIs."
        id="note-detailed"
      />

      <NoteBlock
        type="tip"
        title="They Are Not Mutually Exclusive"
        content="Many developers use LM Studio for interactive exploration, Ollama for local development APIs, and llama.cpp server for production edge deployments. You can even run them side-by-side on different ports since they all use the same model files."
        id="note-coexist"
      />

      <WarningBlock
        title="Ollama Abstracts Away Important Details"
        content="Ollama's simplicity means less control. You cannot easily set specific compile flags, use grammar-constrained generation (natively), or fine-tune batch processing parameters. If you hit Ollama's limits, dropping down to llama.cpp directly gives you full control."
        id="warning-abstraction"
      />
    </div>
  )
}
