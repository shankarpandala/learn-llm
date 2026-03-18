import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function RestAPI() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Ollama REST API Deep Dive</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Ollama exposes a REST API on port 11434 that lets you integrate LLM inference into any
        application. The API supports generation, chat, embeddings, and model management, plus
        an OpenAI-compatible endpoint for drop-in compatibility.
      </p>

      <DefinitionBlock
        title="Ollama API Endpoints"
        definition="The Ollama REST API provides: /api/generate (text completion), /api/chat (multi-turn conversation), /api/embeddings (vector embeddings), /api/tags (list models), /api/show (model info), /api/pull and /api/push (model management), and /v1/* (OpenAI-compatible)."
        id="def-api"
      />

      <PythonCode
        title="Terminal"
        code={`# Generate endpoint -- single-turn completion
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the sky blue?",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "top_p": 0.9,
    "num_predict": 200
  }
}'
# Returns: {"model":"llama3.2","response":"The sky appears blue...","done":true,...}

# Chat endpoint -- multi-turn conversation
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user", "content": "What are its main features?"}
  ],
  "stream": false
}'

# Embeddings endpoint
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "Ollama makes running LLMs easy"
}'

# List local models
curl http://localhost:11434/api/tags`}
        id="code-curl"
      />

      <PythonCode
        title="ollama_api_client.py"
        code={`import requests
import json

BASE_URL = "http://localhost:11434"

def generate(prompt, model="llama3.2", **options):
    """Call the generate endpoint."""
    resp = requests.post(f"{BASE_URL}/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    })
    return resp.json()

def chat(messages, model="llama3.2", **options):
    """Call the chat endpoint."""
    resp = requests.post(f"{BASE_URL}/api/chat", json={
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    })
    return resp.json()

def get_embeddings(text, model="nomic-embed-text"):
    """Get text embeddings."""
    resp = requests.post(f"{BASE_URL}/api/embeddings", json={
        "model": model,
        "prompt": text,
    })
    return resp.json()["embedding"]

# Usage examples
result = generate("Explain REST APIs in one sentence", temperature=0.3)
print(f"Generate: {result['response']}")

result = chat([
    {"role": "user", "content": "What is 2+2?"},
], temperature=0)
print(f"Chat: {result['message']['content']}")

embedding = get_embeddings("Hello world")
print(f"Embedding dim: {len(embedding)}")  # 768 for nomic-embed

# Using the OpenAI-compatible endpoint
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
resp = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(f"OpenAI compat: {resp.choices[0].message.content}")`}
        id="code-python-api"
      />

      <ExampleBlock
        title="API Response Fields"
        problem="What information does the generate endpoint return?"
        steps={[
          { formula: 'response: the generated text', explanation: 'The main output string.' },
          { formula: 'eval_count: number of tokens generated', explanation: 'Useful for tracking token usage.' },
          { formula: 'eval_duration: time spent generating (nanoseconds)', explanation: 'Compute tokens/second as eval_count / (eval_duration / 1e9).' },
          { formula: 'prompt_eval_count: tokens in the prompt', explanation: 'Number of tokens the prompt was tokenized into.' },
          { formula: 'total_duration: total request time', explanation: 'Includes model loading, prompt processing, and generation.' },
        ]}
        id="example-response"
      />

      <NoteBlock
        type="tip"
        title="OpenAI Compatibility"
        content="The /v1/chat/completions endpoint is compatible with the OpenAI Python SDK. Set base_url='http://localhost:11434/v1' and api_key='ollama'. This means you can swap between Ollama and OpenAI by changing just two lines of code."
        id="note-openai-compat"
      />

      <WarningBlock
        title="No Built-in Authentication"
        content="The Ollama API has no authentication mechanism. Anyone who can reach port 11434 can use your models. If exposing to a network, put a reverse proxy (nginx, caddy) with authentication in front of Ollama."
        id="warning-auth"
      />
    </div>
  )
}
