import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function VLLM() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">vLLM</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        vLLM is a high-throughput, memory-efficient inference engine for LLMs. Its key innovation
        is PagedAttention, which manages KV-cache memory like operating system virtual memory pages,
        enabling 2-4x higher throughput than naive serving implementations.
      </p>

      <DefinitionBlock
        title="vLLM"
        definition="vLLM (Virtual LLM) is an open-source LLM serving library featuring PagedAttention for efficient KV-cache management, continuous batching, tensor parallelism, speculative decoding, and an OpenAI-compatible API server. It is the most popular production LLM serving framework."
        id="def-vllm"
      />

      <DefinitionBlock
        title="PagedAttention"
        definition="PagedAttention stores KV-cache in non-contiguous memory blocks (pages) mapped via a block table. This eliminates memory fragmentation and waste from pre-allocation, achieving near-optimal memory utilization. Memory waste drops from 60-80% to under 4%."
        id="def-paged-attention"
      />

      <PythonCode
        title="Terminal"
        code={`# Install vLLM
pip install vllm

# Start the OpenAI-compatible API server
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --gpu-memory-utilization 0.9 \\
    --max-model-len 8192 \\
    --dtype auto

# With tensor parallelism for large models
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-70B-Instruct \\
    --tensor-parallel-size 4 \\
    --gpu-memory-utilization 0.9

# With speculative decoding
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-70B-Instruct \\
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \\
    --num-speculative-tokens 5

# Docker deployment
docker run --gpus all -p 8000:8000 \\
    -v ~/.cache/huggingface:/root/.cache/huggingface \\
    vllm/vllm-openai:latest \\
    --model meta-llama/Llama-3.1-8B-Instruct`}
        id="code-server"
      />

      <PythonCode
        title="vllm_client.py"
        code={`from openai import OpenAI
import time

# vLLM serves an OpenAI-compatible API
client = OpenAI(base_url="http://localhost:8000/v1", api_key="vllm")

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain PagedAttention in 3 sentences."},
    ],
    temperature=0.7,
    max_tokens=200,
)
print(response.choices[0].message.content)

# Batch processing for maximum throughput
import concurrent.futures

def send_request(prompt):
    start = time.time()
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    return time.time() - start, resp.usage.completion_tokens

prompts = [f"Explain concept #{i} in machine learning" for i in range(32)]
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
    results = list(pool.map(send_request, prompts))
total = time.time() - start

total_tokens = sum(t for _, t in results)
print(f"Throughput: {total_tokens/total:.0f} tok/s ({len(prompts)} requests in {total:.1f}s)")`}
        id="code-client"
      />

      <ExampleBlock
        title="vLLM Key Features"
        problem="What makes vLLM the go-to production serving framework?"
        steps={[
          { formula: 'PagedAttention: 2-4x throughput improvement', explanation: 'Efficient KV-cache memory management eliminates waste.' },
          { formula: 'Continuous batching: near-100% GPU utilization', explanation: 'New requests join the batch as old ones complete.' },
          { formula: 'Tensor parallelism: serve 70B+ models across GPUs', explanation: 'Split model weights across multiple GPUs for large models.' },
          { formula: 'OpenAI-compatible API: drop-in replacement', explanation: 'Works with any tool that supports the OpenAI API.' },
        ]}
        id="example-features"
      />

      <NoteBlock
        type="tip"
        title="Prefix Caching"
        content="Enable prefix caching with --enable-prefix-caching for workloads where many requests share the same system prompt. The KV-cache for the shared prefix is computed once and reused, reducing time-to-first-token by 50-90% for chat applications."
        id="note-prefix-caching"
      />

      <WarningBlock
        title="Memory Planning"
        content="vLLM pre-allocates GPU memory based on --gpu-memory-utilization (default 0.9 = 90%). Set this lower if other processes share the GPU. The --max-model-len flag limits context length and thus KV-cache size. Start conservative and increase based on actual usage."
        id="warning-memory"
      />
    </div>
  )
}
