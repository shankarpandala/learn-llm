import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function APIAccess() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">API Access to Open WebUI</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Open WebUI exposes an OpenAI-compatible API that routes through all configured backends.
        This means you can use Open WebUI as a unified API gateway, getting the benefits of its
        model management, RAG, and pipeline features from any client application.
      </p>

      <DefinitionBlock
        title="Open WebUI API"
        definition="Open WebUI provides an API at /api/chat/completions that is compatible with the OpenAI SDK. Requests go through Open WebUI's pipeline system (filters, RAG, tools) before reaching the backend model. Authentication uses bearer tokens generated in the UI settings."
        id="def-api"
      />

      <PythonCode
        title="Terminal"
        code={`# Generate an API key:
# 1. Open WebUI > Settings > Account
# 2. Click "Create API Key"
# 3. Copy the key (shown only once)

# Test the API with curl
curl http://localhost:3000/api/chat/completions \\
    -H "Authorization: Bearer sk-your-open-webui-key" \\
    -H "Content-Type: application/json" \\
    -d '{
        "model": "llama3.2",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "stream": false
    }'

# List available models
curl http://localhost:3000/api/models \\
    -H "Authorization: Bearer sk-your-open-webui-key"

# Streaming response
curl http://localhost:3000/api/chat/completions \\
    -H "Authorization: Bearer sk-your-open-webui-key" \\
    -H "Content-Type: application/json" \\
    -d '{
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "Count to 5"}],
        "stream": true
    }'`}
        id="code-curl"
      />

      <PythonCode
        title="openai_sdk_client.py"
        code={`from openai import OpenAI

# Point the standard OpenAI SDK at Open WebUI
client = OpenAI(
    base_url="http://localhost:3000/api",
    api_key="sk-your-open-webui-key",
)

# Chat completion (works with any configured backend)
response = client.chat.completions.create(
    model="llama3.2",  # Any model visible in Open WebUI
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Docker in one sentence."},
    ],
    temperature=0.7,
    max_tokens=200,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Write a haiku about AI"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()

# List models from all backends
models = client.models.list()
for model in models.data:
    print(f"  {model.id}")

# This means any tool that supports OpenAI API works with Open WebUI:
# - LangChain
# - LlamaIndex
# - AutoGen
# - CrewAI
# - Custom applications`}
        id="code-sdk"
      />

      <PythonCode
        title="langchain_integration.py"
        code={`# Use Open WebUI as backend for LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(
    base_url="http://localhost:3000/api",
    api_key="sk-your-open-webui-key",
    model="llama3.2",
    temperature=0.7,
)

# Simple chat
response = llm.invoke([
    SystemMessage(content="You are a Python expert."),
    HumanMessage(content="What is a decorator?"),
])
print(response.content)

# Batch processing
prompts = [
    [HumanMessage(content=f"Explain {concept} briefly")]
    for concept in ["REST APIs", "GraphQL", "gRPC"]
]
results = llm.batch(prompts)
for prompt, result in zip(["REST", "GraphQL", "gRPC"], results):
    print(f"\\n{prompt}: {result.content[:100]}...")`}
        id="code-langchain"
      />

      <ExampleBlock
        title="API Gateway Benefits"
        problem="Why use Open WebUI as an API gateway?"
        steps={[
          { formula: 'Unified endpoint for all backends', explanation: 'One API URL for Ollama, OpenAI, Anthropic, and custom models.' },
          { formula: 'Pipeline processing for all API requests', explanation: 'Content filters, logging, and augmentation apply to API calls too.' },
          { formula: 'User-level API keys and access control', explanation: 'Each user gets their own key with appropriate model access.' },
          { formula: 'Conversation logging and analytics', explanation: 'All API interactions are tracked alongside web UI usage.' },
        ]}
        id="example-gateway"
      />

      <NoteBlock
        type="tip"
        title="Replace OpenAI in Existing Apps"
        content="Any application using the OpenAI SDK can switch to Open WebUI by changing two lines: base_url and api_key. No other code changes needed. This makes it trivial to move from cloud to local models."
        id="note-migration"
      />

      <WarningBlock
        title="API Rate Limits"
        content="Open WebUI does not have built-in API rate limiting. High-volume API usage can overwhelm the backend, especially with local models. Implement rate limiting in a reverse proxy (nginx) if exposing the API to multiple clients or applications."
        id="warning-rate-limits"
      />
    </div>
  )
}
