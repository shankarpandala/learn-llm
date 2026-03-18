import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function MultiBackend() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Multi-Backend Configuration</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Open WebUI can connect to multiple LLM backends simultaneously, letting users choose between
        local Ollama models, cloud APIs, and specialized serving frameworks from a single unified
        interface. This is ideal for teams that need flexibility.
      </p>

      <DefinitionBlock
        title="Multi-Backend Setup"
        definition="Open WebUI supports connecting to one Ollama instance and multiple OpenAI-compatible API endpoints at the same time. All models from all backends appear in a single model selector, and users can switch between them mid-conversation."
        id="def-multi-backend"
      />

      <PythonCode
        title="Terminal"
        code={`# Docker Compose with multiple backends
cat > docker-compose.yml << 'EOF'
services:
  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    volumes:
      - ./litellm-config.yaml:/app/config.yaml
    command: ["--config", "/app/config.yaml", "--port", "4000"]
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=\${ANTHROPIC_API_KEY}

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    volumes:
      - open_webui_data:/app/backend/data
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - OPENAI_API_BASE_URLS=http://litellm:4000/v1
      - OPENAI_API_KEYS=sk-litellm
    depends_on:
      - ollama
      - litellm

volumes:
  ollama_data:
  open_webui_data:
EOF

# LiteLLM config for multiple providers
cat > litellm-config.yaml << 'EOF'
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: llama-groq
    litellm_params:
      model: groq/llama-3.1-70b-versatile
      api_key: os.environ/GROQ_API_KEY
EOF

docker compose up -d`}
        id="code-compose"
      />

      <PythonCode
        title="test_multi_backend.py"
        code={`from openai import OpenAI

# Test that Open WebUI can proxy to all backends
owui = OpenAI(
    base_url="http://localhost:3000/api",  # Open WebUI API
    api_key="your-open-webui-api-key",     # Generated in settings
)

# List all available models across backends
models = owui.models.list()
print("Available models:")
for m in models.data:
    print(f"  {m.id}")

# Test different backends through Open WebUI
backends = {
    "llama3.2": "Ollama (local)",
    "gpt-4o": "OpenAI (cloud)",
    "claude-sonnet": "Anthropic via LiteLLM",
}

for model, backend in backends.items():
    try:
        resp = owui.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10,
        )
        print(f"{backend}: {resp.choices[0].message.content}")
    except Exception as e:
        print(f"{backend}: {e}")`}
        id="code-test"
      />

      <ExampleBlock
        title="Use Cases for Multi-Backend"
        problem="Why run multiple backends?"
        steps={[
          { formula: 'Cost optimization: local for drafts, cloud for final', explanation: 'Use cheap local models for iteration, expensive cloud models for quality.' },
          { formula: 'Capability routing: code model + general model', explanation: 'Specialized models for specific tasks, general models for everything else.' },
          { formula: 'Fallback: if local GPU is busy, fall back to cloud', explanation: 'Ensures availability when local resources are constrained.' },
        ]}
        id="example-use-cases"
      />

      <NoteBlock
        type="tip"
        title="Model Naming"
        content="When models from different backends have the same name, Open WebUI disambiguates them. You can also rename models in the admin settings to make them more user-friendly, e.g., renaming 'meta-llama/Meta-Llama-3.1-8B-Instruct' to 'LLaMA 3.1 8B'."
        id="note-naming"
      />

      <WarningBlock
        title="Latency Varies by Backend"
        content="Users may not realize that switching from a local Ollama model to a cloud API changes latency characteristics significantly. Consider labeling models with their backend type so users can make informed choices about speed vs quality."
        id="warning-latency"
      />
    </div>
  )
}
