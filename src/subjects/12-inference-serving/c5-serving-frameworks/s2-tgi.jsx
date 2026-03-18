import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function TGI() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Text Generation Inference (TGI)</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        TGI is HuggingFace's production-grade serving solution for large language models. Written
        in Rust for performance-critical paths, TGI provides continuous batching, tensor parallelism,
        quantization, and tight integration with the HuggingFace Hub. It powers HuggingFace's
        Inference Endpoints and is a battle-tested choice for deploying models at scale.
      </p>

      <DefinitionBlock
        title="Text Generation Inference (TGI)"
        definition="TGI is an open-source toolkit from HuggingFace for deploying LLMs. It features a Rust-based token streaming server, Flash Attention, continuous batching, and native support for GPTQ/AWQ/EETQ quantization. TGI exposes both an OpenAI-compatible API and its own Messages API."
        id="def-tgi"
      />

      <PythonCode
        title="Terminal"
        code={`# Launch TGI with Docker (recommended)
docker run --gpus all --shm-size 1g -p 8080:80 \\
    -v $HOME/.cache/huggingface:/data \\
    ghcr.io/huggingface/text-generation-inference:latest \\
    --model-id meta-llama/Llama-3.1-8B-Instruct \\
    --max-input-tokens 4096 \\
    --max-total-tokens 8192 \\
    --max-batch-prefill-tokens 4096

# Multi-GPU with tensor parallelism
docker run --gpus all --shm-size 1g -p 8080:80 \\
    -v $HOME/.cache/huggingface:/data \\
    ghcr.io/huggingface/text-generation-inference:latest \\
    --model-id meta-llama/Llama-3.1-70B-Instruct \\
    --num-shard 4 \\
    --quantize bitsandbytes-nf4

# Serve a GPTQ-quantized model
docker run --gpus all --shm-size 1g -p 8080:80 \\
    -v $HOME/.cache/huggingface:/data \\
    ghcr.io/huggingface/text-generation-inference:latest \\
    --model-id TheBloke/Llama-2-13B-chat-GPTQ \\
    --quantize gptq

# Test with curl (TGI native endpoint)
curl http://localhost:8080/generate \\
    -H "Content-Type: application/json" \\
    -d '{"inputs": "What is deep learning?", "parameters": {"max_new_tokens": 128}}'

# OpenAI-compatible endpoint
curl http://localhost:8080/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{
        "model": "tgi",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 128
    }'`}
        id="code-tgi-launch"
      />

      <PythonCode
        title="tgi_client.py"
        code={`import requests
from huggingface_hub import InferenceClient

# Option 1: HuggingFace InferenceClient (recommended)
client = InferenceClient("http://localhost:8080")

response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain TGI in two sentences."},
    ],
    max_tokens=200,
    temperature=0.7,
)
print(response.choices[0].message.content)

# Streaming with InferenceClient
for token in client.chat_completion(
    messages=[{"role": "user", "content": "Count from 1 to 10"}],
    max_tokens=100,
    stream=True,
):
    delta = token.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()

# Option 2: OpenAI SDK (works with TGI's compatible endpoint)
from openai import OpenAI

openai_client = OpenAI(base_url="http://localhost:8080/v1", api_key="-")
resp = openai_client.chat.completions.create(
    model="tgi",
    messages=[{"role": "user", "content": "What is Flash Attention?"}],
    max_tokens=150,
)
print(resp.choices[0].message.content)

# Option 3: Direct HTTP for TGI-specific features
resp = requests.post("http://localhost:8080/generate", json={
    "inputs": "Translate to French: Hello, how are you?",
    "parameters": {
        "max_new_tokens": 64,
        "temperature": 0.3,
        "repetition_penalty": 1.1,
        "return_full_text": False,
    }
})
print(resp.json()["generated_text"])`}
        id="code-tgi-client"
      />

      <ExampleBlock
        title="TGI Configuration Options"
        problem="What are the key TGI server parameters?"
        steps={[
          { formula: '--max-input-tokens 4096', explanation: 'Maximum number of tokens in the input prompt.' },
          { formula: '--max-total-tokens 8192', explanation: 'Maximum total tokens (input + output) per request.' },
          { formula: '--max-batch-prefill-tokens 4096', explanation: 'Controls prefill batch size. Lower values reduce latency spikes.' },
          { formula: '--num-shard 4', explanation: 'Number of GPU shards for tensor parallelism.' },
          { formula: '--quantize gptq | awq | bitsandbytes-nf4', explanation: 'Quantization method to reduce memory usage.' },
        ]}
        id="example-config"
      />

      <NoteBlock
        type="tip"
        title="TGI Health & Metrics"
        content="TGI exposes a /health endpoint for load balancer health checks and a /metrics endpoint with Prometheus-compatible metrics including queue size, batch size, and inference latency. Use these to monitor and autoscale your deployment."
        id="note-metrics"
      />

      <WarningBlock
        title="Shared Memory Requirement"
        content="TGI uses shared memory for inter-process communication. Always set --shm-size 1g (or higher) in Docker, otherwise the container will crash with 'Bus error' on multi-GPU setups. In Kubernetes, mount an emptyDir volume at /dev/shm with a sizeLimit."
        id="warning-shm"
      />
    </div>
  )
}
