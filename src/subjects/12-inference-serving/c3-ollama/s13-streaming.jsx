import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Streaming() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Streaming Responses & Context Windows</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Streaming lets users see tokens as they are generated rather than waiting for the full
        response. This dramatically improves perceived latency. Understanding context windows
        is equally important for managing conversation length.
      </p>

      <DefinitionBlock
        title="Streaming"
        definition="In streaming mode, the server sends each generated token as a separate JSON object as soon as it is produced. The client receives a series of newline-delimited JSON (NDJSON) chunks, each containing a partial response. The final chunk has done: true."
        id="def-streaming"
      />

      <PythonCode
        title="Terminal"
        code={`# Streaming with curl (default behavior)
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Count from 1 to 10"
}'
# {"model":"llama3.2","response":"1","done":false}
# {"model":"llama3.2","response":",","done":false}
# {"model":"llama3.2","response":" 2","done":false}
# ... one chunk per token ...
# {"model":"llama3.2","response":"","done":true,"eval_count":25,...}

# Non-streaming (wait for full response)
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Count from 1 to 10",
  "stream": false
}'
# Returns one JSON object with the complete response`}
        id="code-streaming-curl"
      />

      <PythonCode
        title="streaming_client.py"
        code={`import requests
import json
import sys

def stream_generate(prompt, model="llama3.2"):
    """Stream tokens from Ollama and print them in real-time."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt},
        stream=True,  # Enable HTTP streaming
    )

    full_response = ""
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get("response", "")
            full_response += token
            print(token, end="", flush=True)

            if chunk.get("done"):
                stats = {
                    "tokens": chunk.get("eval_count", 0),
                    "speed": chunk.get("eval_count", 0) /
                             (chunk.get("eval_duration", 1) / 1e9),
                }
                print(f"\\n\\n--- {stats['tokens']} tokens at "
                      f"{stats['speed']:.1f} tok/s ---")
    return full_response

# Using the official Python library (streaming)
import ollama

def stream_chat(messages, model="llama3.2"):
    """Stream a chat response."""
    for chunk in ollama.chat(model=model, messages=messages, stream=True):
        print(chunk["message"]["content"], end="", flush=True)
    print()

stream_generate("Write a short poem about coding")
print("\\n" + "="*50)
stream_chat([{"role": "user", "content": "Tell me a joke"}])`}
        id="code-streaming-python"
      />

      <DefinitionBlock
        title="Context Window"
        definition="The context window is the maximum number of tokens a model can process in a single request (prompt + response combined). Common sizes: 4096 (GPT-2), 8192 (LLaMA 3), 128K (LLaMA 3.1). Exceeding the window causes the model to lose early context."
        id="def-context-window"
      />

      <PythonCode
        title="Terminal"
        code={`# Set context window size in Ollama
# Via API options:
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Hello",
  "options": {
    "num_ctx": 8192
  }
}'

# Via Modelfile:
# PARAMETER num_ctx 16384

# Default is typically 2048 to save memory
# Increase for long conversations or documents
# Maximum depends on the model (check model card)

# Check a model's default context size
ollama show llama3.2 --modelfile | grep num_ctx`}
        id="code-context"
      />

      <NoteBlock
        type="tip"
        title="Context Window vs Memory"
        content="Doubling the context window roughly doubles the KV-cache memory usage. A 3B model with num_ctx=2048 might use 3.5GB, but with num_ctx=32768 it could use 6GB+. Only increase context size when you actually need it."
        id="note-context-memory"
      />

      <WarningBlock
        title="Conversation History Accumulates"
        content="In chat mode, every previous message is re-sent with each request. A long conversation can silently exceed the context window, causing the model to drop early messages. Monitor token counts and implement conversation summarization or sliding window truncation for long chats."
        id="warning-history"
      />
    </div>
  )
}
