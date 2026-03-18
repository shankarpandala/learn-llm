import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function OpenAIAPIs() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Connecting to OpenAI-Compatible APIs</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Open WebUI is not limited to Ollama. It can connect to any OpenAI-compatible API, including
        OpenAI itself, Anthropic (via proxy), vLLM, TGI, LiteLLM, and other serving frameworks.
        This lets you use the same interface for local and cloud models.
      </p>

      <DefinitionBlock
        title="OpenAI-Compatible API"
        definition="An API that implements the same endpoint structure as OpenAI's Chat Completions API (/v1/chat/completions, /v1/models). Most LLM serving frameworks implement this standard, enabling tool interoperability."
        id="def-openai-api"
      />

      <PythonCode
        title="Terminal"
        code={`# Configure OpenAI API in Open WebUI via environment variables
docker run -d -p 3000:8080 \\
    --add-host=host.docker.internal:host-gateway \\
    -e OPENAI_API_BASE_URLS="https://api.openai.com/v1;http://localhost:8000/v1" \\
    -e OPENAI_API_KEYS="sk-your-openai-key;none" \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Or configure via the UI:
# 1. Go to Admin Panel > Settings > Connections
# 2. Add a new OpenAI API connection
# 3. Enter the base URL and API key

# Common OpenAI-compatible endpoints:
# OpenAI:     https://api.openai.com/v1
# vLLM:       http://localhost:8000/v1
# TGI:        http://localhost:8080/v1
# LiteLLM:    http://localhost:4000/v1
# Together:   https://api.together.xyz/v1
# Groq:       https://api.groq.com/openai/v1
# Ollama:     http://localhost:11434/v1`}
        id="code-config"
      />

      <PythonCode
        title="test_connections.py"
        code={`import requests

# Test various OpenAI-compatible endpoints
ENDPOINTS = {
    "OpenAI": {
        "url": "https://api.openai.com/v1/models",
        "headers": {"Authorization": "Bearer sk-your-key"},
    },
    "vLLM (local)": {
        "url": "http://localhost:8000/v1/models",
        "headers": {},
    },
    "Ollama (OpenAI compat)": {
        "url": "http://localhost:11434/v1/models",
        "headers": {},
    },
    "Groq": {
        "url": "https://api.groq.com/openai/v1/models",
        "headers": {"Authorization": "Bearer gsk_your-key"},
    },
}

for name, config in ENDPOINTS.items():
    try:
        resp = requests.get(
            config["url"],
            headers=config["headers"],
            timeout=5,
        )
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            model_names = [m["id"] for m in models[:3]]
            print(f"{name}: {len(models)} models - {model_names}")
        else:
            print(f"{name}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"{name}: {type(e).__name__}")`}
        id="code-test"
      />

      <ExampleBlock
        title="Setting Up Cloud Providers"
        problem="How to connect Open WebUI to various cloud providers?"
        steps={[
          { formula: 'OpenAI: URL=https://api.openai.com/v1, key=sk-...', explanation: 'Direct access to GPT-4, GPT-4o, etc.' },
          { formula: 'Groq: URL=https://api.groq.com/openai/v1, key=gsk_...', explanation: 'Ultra-fast inference for LLaMA, Mixtral on Groq hardware.' },
          { formula: 'Together AI: URL=https://api.together.xyz/v1, key=...', explanation: 'Access to 100+ open models with pay-per-token pricing.' },
          { formula: 'LiteLLM proxy: URL=http://localhost:4000/v1', explanation: 'Unified proxy that routes to any provider, including Anthropic.' },
        ]}
        id="example-providers"
      />

      <NoteBlock
        type="tip"
        title="Using Anthropic Models"
        content="Anthropic's API is not directly OpenAI-compatible, but LiteLLM can translate between them. Run LiteLLM as a proxy and point Open WebUI to it. This gives you access to Claude models through the same interface."
        id="note-anthropic"
      />

      <WarningBlock
        title="API Key Security"
        content="When adding cloud API keys to Open WebUI, they are stored in the application database. In multi-user setups, admin-configured keys are shared across all users. Be careful about who has admin access and consider using per-user API keys instead."
        id="warning-keys"
      />
    </div>
  )
}
