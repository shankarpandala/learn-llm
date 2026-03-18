import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function WhatIsOllama() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">What is Ollama & Why It Matters</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Ollama is an open-source tool that makes running large language models locally as simple as
        running a Docker container. It wraps llama.cpp with a user-friendly CLI, REST API, and model
        management system, making local LLM inference accessible to everyone.
      </p>

      <DefinitionBlock
        title="Ollama"
        definition="Ollama is a local LLM runtime that downloads, manages, and serves quantized language models. It provides a Docker-like experience for LLMs: pull a model by name, run it with one command, and interact via CLI or REST API."
        id="def-ollama"
      />

      <NoteBlock
        type="intuition"
        title="Docker for LLMs"
        content="Just as Docker made it trivial to run complex server software with 'docker run nginx', Ollama makes it trivial to run LLMs with 'ollama run llama3.2'. It handles model downloads, quantization format selection, GPU detection, memory management, and API serving -- all automatically."
        id="note-docker-analogy"
      />

      <ExampleBlock
        title="Ollama in 30 Seconds"
        problem="Go from zero to chatting with an LLM locally."
        steps={[
          { formula: 'curl -fsSL https://ollama.com/install.sh | sh', explanation: 'One-line install on macOS/Linux.' },
          { formula: 'ollama run llama3.2', explanation: 'Downloads the model (~2GB for 3B) and starts an interactive chat.' },
          { formula: 'curl http://localhost:11434/api/generate -d \'{"model":"llama3.2","prompt":"Hello"}\'', explanation: 'Or use the REST API from any programming language.' },
        ]}
        id="example-quick-start"
      />

      <PythonCode
        title="Terminal"
        code={`# Check if Ollama is running
ollama --version
# ollama version is 0.5.x

# List available local models
ollama list
# NAME              ID           SIZE    MODIFIED
# llama3.2:latest   a80c4f17acd5 2.0 GB  2 hours ago

# Quick test
ollama run llama3.2 "What is the capital of France?"
# The capital of France is Paris.

# Check system info
ollama ps
# NAME          ID           SIZE    PROCESSOR  UNTIL
# llama3.2      a80c4f17acd5 3.5 GB 100% GPU   4 minutes from now`}
        id="code-quick-start"
      />

      <PythonCode
        title="ollama_python.py"
        code={`# Using the official Ollama Python library
# pip install ollama
import ollama

# Simple generation
response = ollama.generate(
    model="llama3.2",
    prompt="Explain quantum computing in one paragraph."
)
print(response["response"])

# Chat with message history
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to reverse a string."},
]
response = ollama.chat(model="llama3.2", messages=messages)
print(response["message"]["content"])

# Streaming responses
for chunk in ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Tell me a short joke."}],
    stream=True,
):
    print(chunk["message"]["content"], end="", flush=True)
print()`}
        id="code-python-lib"
      />

      <NoteBlock
        type="note"
        title="Key Features"
        content="Ollama supports: automatic GPU detection and offloading, GGUF model format, Modelfiles for customization, concurrent model serving, OpenAI-compatible API, vision models (LLaVA), embedding models, and cross-platform support (macOS, Linux, Windows)."
        id="note-features"
      />

      <WarningBlock
        title="Not for Production at Scale"
        content="Ollama is designed for local development and small-scale serving. For production workloads with many concurrent users, consider vLLM, TGI, or TensorRT-LLM which offer continuous batching, tensor parallelism, and higher throughput. Ollama excels at simplicity, not raw performance."
        id="warning-scale"
      />
    </div>
  )
}
