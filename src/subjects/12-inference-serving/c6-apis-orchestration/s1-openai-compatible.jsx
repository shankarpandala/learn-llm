import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function OpenAICompatible() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">OpenAI-Compatible APIs</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The OpenAI API format has become the de facto standard for LLM APIs. Nearly every
        serving framework (vLLM, TGI, llama.cpp, Ollama, LiteLLM) exposes endpoints that
        accept the same JSON schema as OpenAI. This means code written for the OpenAI SDK
        works with local models by changing just the base URL.
      </p>

      <DefinitionBlock
        title="OpenAI-Compatible API"
        definition="An OpenAI-compatible API implements the same HTTP endpoints and JSON schema as OpenAI's Chat Completions API. The key endpoints are POST /v1/chat/completions for chat, POST /v1/completions for text completion, POST /v1/embeddings for embeddings, and GET /v1/models for listing available models."
        id="def-openai-compat"
      />

      <PythonCode
        title="openai_compatible.py"
        code={`from openai import OpenAI

# The same code works with ANY OpenAI-compatible backend
# Just change base_url and api_key

backends = {
    "openai":    {"base_url": "https://api.openai.com/v1",   "api_key": "sk-..."},
    "vllm":      {"base_url": "http://localhost:8000/v1",     "api_key": "EMPTY"},
    "tgi":       {"base_url": "http://localhost:8080/v1",     "api_key": "EMPTY"},
    "ollama":    {"base_url": "http://localhost:11434/v1",    "api_key": "ollama"},
    "llamacpp":  {"base_url": "http://localhost:8081/v1",     "api_key": "EMPTY"},
    "litellm":   {"base_url": "http://localhost:4000",        "api_key": "sk-master"},
}

def query_backend(name, config, model="default"):
    client = OpenAI(**config)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What are you?"},
            ],
            max_tokens=50,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Query each available backend
models = {
    "openai": "gpt-4o-mini", "vllm": "llama-3.1-8b",
    "tgi": "tgi", "ollama": "llama3.2",
    "llamacpp": "local", "litellm": "gpt-4",
}
for name, config in backends.items():
    result = query_backend(name, config, models.get(name, "default"))
    print(f"{name:>10}: {result[:80]}")`}
        id="code-backends"
      />

      <PythonCode
        title="Terminal"
        code={`# The Chat Completions request format (works everywhere)
curl http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -H "Authorization: Bearer EMPTY" \\
    -d '{
        "model": "llama-3.1-8b",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ],
        "temperature": 0.7,
        "max_tokens": 128,
        "top_p": 0.95,
        "stream": false
    }'

# List available models
curl http://localhost:8000/v1/models \\
    -H "Authorization: Bearer EMPTY"

# Embeddings endpoint
curl http://localhost:8000/v1/embeddings \\
    -H "Content-Type: application/json" \\
    -d '{"model": "llama-3.1-8b", "input": "Hello world"}'`}
        id="code-curl-api"
      />

      <ExampleBlock
        title="Standard API Endpoints"
        problem="What endpoints does an OpenAI-compatible server expose?"
        steps={[
          { formula: 'POST /v1/chat/completions', explanation: 'Chat-style completion with messages array. The most commonly used endpoint.' },
          { formula: 'POST /v1/completions', explanation: 'Legacy text completion with a single prompt string.' },
          { formula: 'POST /v1/embeddings', explanation: 'Generate vector embeddings for text. Not all servers support this.' },
          { formula: 'GET /v1/models', explanation: 'List available models. Returns model IDs you can use in requests.' },
        ]}
        id="example-endpoints"
      />

      <NoteBlock
        type="note"
        title="Compatibility Gaps"
        content="While the core chat completions format is well-standardized, advanced features like function/tool calling, JSON mode, logprobs, and vision inputs vary across implementations. vLLM and LiteLLM have the broadest feature coverage. Always test specific features you rely on against your chosen backend."
        id="note-gaps"
      />

      <WarningBlock
        title="Model Names Are Not Standardized"
        content="Each backend uses different model name conventions. OpenAI uses 'gpt-4o', vLLM uses the HuggingFace path, Ollama uses short names like 'llama3.2', and TGI uses 'tgi'. When switching backends, you must also update model names. LiteLLM solves this by letting you define custom model aliases."
        id="warning-model-names"
      />
    </div>
  )
}
