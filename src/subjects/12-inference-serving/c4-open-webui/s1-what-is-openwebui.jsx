import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function WhatIsOpenWebUI() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">What is Open WebUI</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Open WebUI is a self-hosted, feature-rich chat interface for LLMs. It provides a
        ChatGPT-like experience that works with local models (via Ollama) and cloud APIs,
        giving you full control over your data and infrastructure.
      </p>

      <DefinitionBlock
        title="Open WebUI"
        definition="Open WebUI (formerly Ollama WebUI) is an open-source web application that provides a polished chat interface for interacting with LLMs. It supports multiple backends (Ollama, OpenAI, vLLM), multi-user access, RAG, web search, tool use, and extensive customization."
        id="def-open-webui"
      />

      <ExampleBlock
        title="Key Features"
        problem="What makes Open WebUI stand out among LLM interfaces?"
        steps={[
          { formula: 'Multi-backend: Ollama + OpenAI + any compatible API', explanation: 'Switch between local and cloud models in the same conversation.' },
          { formula: 'RAG: upload documents and chat with them', explanation: 'Built-in document processing, embedding, and retrieval.' },
          { formula: 'Multi-user: authentication, roles, sharing', explanation: 'Deploy for a team with access control and conversation sharing.' },
          { formula: 'Extensible: tools, pipelines, plugins', explanation: 'Add web search, code execution, image generation, and custom functions.' },
        ]}
        id="example-features"
      />

      <PythonCode
        title="Terminal"
        code={`# Quickest way to try Open WebUI
# (Assumes Ollama is already running on localhost:11434)

# Option 1: Docker (recommended)
docker run -d -p 3000:8080 \\
    --add-host=host.docker.internal:host-gateway \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    --restart always \\
    ghcr.io/open-webui/open-webui:main

# Open http://localhost:3000 in your browser
# Create an admin account on first visit

# Option 2: Bundled with Ollama (all-in-one)
docker run -d -p 3000:8080 \\
    --gpus all \\
    -v ollama:/root/.ollama \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:ollama

# Option 3: pip install (no Docker)
pip install open-webui
open-webui serve --port 3000`}
        id="code-quickstart"
      />

      <PythonCode
        title="verify_setup.py"
        code={`import requests

# Check if Open WebUI is running
try:
    resp = requests.get("http://localhost:3000/api/version")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Open WebUI version: {data.get('version', 'unknown')}")
        print("Status: Running!")
    else:
        print(f"Unexpected status: {resp.status_code}")
except requests.ConnectionError:
    print("Open WebUI is not running on port 3000")

# Check connected backends
try:
    resp = requests.get("http://localhost:3000/api/models")
    if resp.status_code == 200:
        models = resp.json()
        print(f"Available models: {len(models.get('data', []))}")
        for m in models.get("data", [])[:5]:
            print(f"  - {m.get('id', 'unknown')}")
except Exception as e:
    print(f"Could not fetch models: {e}")`}
        id="code-verify"
      />

      <NoteBlock
        type="note"
        title="Data Privacy"
        content="All data stays on your server. Conversations, uploaded documents, and user information are stored locally in the open-webui Docker volume. Nothing is sent to external services unless you explicitly configure a cloud API backend."
        id="note-privacy"
      />

      <WarningBlock
        title="Resource Requirements"
        content="Open WebUI itself is lightweight (uses ~200MB RAM), but it runs alongside your LLM backend. A system running both Ollama with a 7B model and Open WebUI needs at least 8GB RAM total. The web interface works best with a modern browser."
        id="warning-resources"
      />
    </div>
  )
}
