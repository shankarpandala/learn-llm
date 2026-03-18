import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LlamaCppServer() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">llama.cpp Server & koboldcpp</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        llama.cpp provides a lightweight, dependency-free HTTP server for serving GGUF models
        with an OpenAI-compatible API. It runs on CPUs, Apple Silicon, and NVIDIA/AMD GPUs
        without requiring heavy frameworks. koboldcpp extends this with a richer UI and API
        for creative writing and roleplay workloads.
      </p>

      <DefinitionBlock
        title="llama-server"
        definition="llama-server (formerly llama.cpp server) is an HTTP server bundled with llama.cpp that serves GGUF-format models via an OpenAI-compatible REST API. It supports continuous batching, parallel requests, prompt caching, GPU offloading, grammar-constrained generation, and embedding extraction."
        id="def-llama-server"
      />

      <PythonCode
        title="Terminal"
        code={`# Build llama.cpp with CUDA support
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j$(nproc)

# Start the server with a GGUF model
./bin/llama-server \\
    --model ~/models/Llama-3.1-8B-Instruct-Q4_K_M.gguf \\
    --host 0.0.0.0 \\
    --port 8080 \\
    --ctx-size 8192 \\
    --n-gpu-layers 99 \\
    --parallel 4 \\
    --cont-batching

# CPU-only (no GPU flags needed)
./bin/llama-server \\
    --model ~/models/Phi-3-mini-4k-instruct-Q4_K_M.gguf \\
    --host 0.0.0.0 \\
    --port 8080 \\
    --ctx-size 4096 \\
    --threads 8

# Test with curl (OpenAI-compatible)
curl http://localhost:8080/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 128,
        "temperature": 0.7
    }'

# Docker alternative
docker run -p 8080:8080 \\
    -v ~/models:/models \\
    ghcr.io/ggerganov/llama.cpp:server-cuda \\
    --model /models/Llama-3.1-8B-Instruct-Q4_K_M.gguf \\
    --n-gpu-layers 99 --ctx-size 8192 --parallel 4`}
        id="code-llama-server"
      />

      <PythonCode
        title="llamacpp_client.py"
        code={`from openai import OpenAI
import requests

# OpenAI SDK works directly with llama-server
client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")

response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain GGUF format briefly."},
    ],
    temperature=0.7,
    max_tokens=200,
)
print(response.choices[0].message.content)

# Grammar-constrained generation (JSON output)
resp = requests.post("http://localhost:8080/completion", json={
    "prompt": "List three colors as JSON:\\n",
    "n_predict": 128,
    "grammar": '''root ::= "{" ws "\\"colors\\"" ws ":" ws arr "}"
arr ::= "[" ws string ("," ws string)* ws "]"
string ::= "\\"" [a-zA-Z]+ "\\""
ws ::= " "?''',
})
print("Grammar-constrained:", resp.json()["content"])

# Generate embeddings
resp = requests.post("http://localhost:8080/v1/embeddings", json={
    "input": "Hello world",
    "model": "local-model",
})
embedding = resp.json()["data"][0]["embedding"]
print(f"Embedding dims: {len(embedding)}")

# Server health and slots info
health = requests.get("http://localhost:8080/health").json()
print(f"Server status: {health['status']}")`}
        id="code-llamacpp-client"
      />

      <ExampleBlock
        title="koboldcpp"
        problem="How does koboldcpp extend llama.cpp?"
        steps={[
          { formula: 'Built-in web UI for chat and story writing', explanation: 'No separate frontend needed; includes KoboldAI Lite interface.' },
          { formula: 'KoboldAI API + OpenAI API compatibility', explanation: 'Supports both API standards for broad client compatibility.' },
          { formula: 'One-click Windows/Linux/macOS launchers', explanation: 'Pre-built binaries with GUI for model selection and GPU config.' },
          { formula: 'Multimodal support (LLaVA models)', explanation: 'Serve vision-language models with image input support.' },
        ]}
        id="example-koboldcpp"
      />

      <NoteBlock
        type="tip"
        title="Prompt Caching"
        content="llama-server automatically caches prompt evaluations. When multiple requests share the same prefix (e.g., a system prompt), subsequent requests skip re-evaluating that prefix. This significantly reduces time-to-first-token for chat applications."
        id="note-prompt-cache"
      />

      <WarningBlock
        title="Parallel Slots"
        content="The --parallel flag sets the number of concurrent requests. Each slot reserves context memory, so --parallel 4 --ctx-size 8192 allocates 4 x 8192 tokens of KV cache. This can quickly exceed GPU VRAM with large contexts. Monitor memory with --verbose and reduce parallel slots if you see OOM errors."
        id="warning-parallel"
      />
    </div>
  )
}
