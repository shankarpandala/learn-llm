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
      <h1 className="text-3xl font-bold">Streaming & Server-Sent Events</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        LLMs generate tokens one at a time, and streaming delivers each token to the client
        as soon as it is produced rather than waiting for the full response. This dramatically
        reduces perceived latency: users see the first token in milliseconds instead of waiting
        seconds for the complete answer. Streaming uses the Server-Sent Events (SSE) protocol
        over HTTP.
      </p>

      <DefinitionBlock
        title="Server-Sent Events (SSE)"
        definition="Server-Sent Events is an HTTP protocol where the server sends a stream of text events over a single long-lived connection. Each event is prefixed with 'data: ' and separated by double newlines. The stream ends with a 'data: [DONE]' sentinel. LLM APIs use SSE to stream individual token chunks as they are generated."
        id="def-sse"
      />

      <PythonCode
        title="streaming_client.py"
        code={`from openai import OpenAI
import time

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# Streaming with the OpenAI SDK
print("=== Streaming Response ===")
start = time.time()
first_token_time = None

stream = client.chat.completions.create(
    model="llama-3.1-8b",
    messages=[{"role": "user", "content": "Explain streaming in 3 sentences."}],
    max_tokens=200,
    stream=True,
)

full_response = ""
for chunk in stream:
    if chunk.choices[0].delta.content:
        token = chunk.choices[0].delta.content
        if first_token_time is None:
            first_token_time = time.time() - start
        full_response += token
        print(token, end="", flush=True)

total_time = time.time() - start
print(f"\\n\\nTime to first token: {first_token_time*1000:.0f}ms")
print(f"Total time: {total_time*1000:.0f}ms")
print(f"Tokens: ~{len(full_response.split())}")

# Non-streaming comparison
start = time.time()
response = client.chat.completions.create(
    model="llama-3.1-8b",
    messages=[{"role": "user", "content": "Explain streaming in 3 sentences."}],
    max_tokens=200,
    stream=False,
)
print(f"\\nNon-streaming total: {(time.time()-start)*1000:.0f}ms")
print("(User sees nothing until this moment)")`}
        id="code-streaming-client"
      />

      <PythonCode
        title="raw_sse_client.py"
        code={`import requests
import json

# Raw SSE parsing (useful for understanding the protocol)
url = "http://localhost:8000/v1/chat/completions"
payload = {
    "model": "llama-3.1-8b",
    "messages": [{"role": "user", "content": "Count to 5."}],
    "max_tokens": 100,
    "stream": True,
}

response = requests.post(url, json=payload, stream=True,
                          headers={"Content-Type": "application/json"})

print("=== Raw SSE Events ===")
for line in response.iter_lines(decode_unicode=True):
    if not line:
        continue
    if line.startswith("data: "):
        data = line[6:]  # Strip "data: " prefix
        if data == "[DONE]":
            print("\\n[Stream complete]")
            break
        chunk = json.loads(data)
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            print(delta["content"], end="", flush=True)

# FastAPI server with SSE (building your own streaming proxy)
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import httpx, asyncio

app = FastAPI()

@app.post("/proxy/chat")
async def proxy_chat(request: dict):
    async def stream_response():
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", "http://localhost:8000/v1/chat/completions",
                                      json={**request, "stream": True}) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        yield f"{line}\\n\\n"
    return StreamingResponse(stream_response(), media_type="text/event-stream")
"""`}
        id="code-raw-sse"
      />

      <ExampleBlock
        title="Streaming vs Non-Streaming"
        problem="When should you use streaming?"
        steps={[
          { formula: 'Interactive chat: always stream', explanation: 'Users perceive the response as faster when tokens appear immediately.' },
          { formula: 'Batch processing: do not stream', explanation: 'Non-streaming is simpler to parse and has slightly less overhead.' },
          { formula: 'Tool/function calling: stream with care', explanation: 'Function call arguments arrive incrementally; accumulate before parsing JSON.' },
          { formula: 'Proxy/middleware: forward the stream', explanation: 'Pass SSE through without buffering to preserve low latency.' },
        ]}
        id="example-when-stream"
      />

      <NoteBlock
        type="intuition"
        title="Why Streaming Feels Faster"
        content="A 200-token response at 50 tok/s takes 4 seconds. Without streaming, the user stares at a blank screen for 4 seconds. With streaming, they see the first token after ~100ms (prefill time) and can start reading immediately. The total wall-clock time is the same, but perceived latency drops from 4 seconds to 100ms."
        id="note-perception"
      />

      <WarningBlock
        title="Buffering Breaks Streaming"
        content="Reverse proxies (nginx, Cloudflare) and load balancers may buffer SSE responses, defeating the purpose of streaming. Configure proxy_buffering off in nginx, and disable response buffering in your CDN. Also set appropriate headers: Cache-Control: no-cache, X-Accel-Buffering: no."
        id="warning-buffering"
      />
    </div>
  )
}
