import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Comparison() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Comparing Serving Frameworks</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Choosing the right serving framework depends on your hardware, model, latency
        requirements, and operational constraints. This section compares the key frameworks
        across the dimensions that matter most for production deployments.
      </p>

      <DefinitionBlock
        title="Serving Framework Selection Criteria"
        definition="The key dimensions for comparing LLM serving frameworks are: throughput (tokens/second), latency (time-to-first-token and inter-token latency), hardware support (NVIDIA, AMD, CPU, Apple Silicon), model format support (HuggingFace, GGUF, TensorRT engines), ease of deployment, and API compatibility."
        id="def-criteria"
      />

      <ExampleBlock
        title="Framework Comparison Matrix"
        problem="How do the major frameworks compare?"
        steps={[
          { formula: 'vLLM: Best general-purpose GPU serving', explanation: 'Highest throughput for most models on NVIDIA GPUs. Easy setup. PagedAttention + continuous batching. OpenAI API.' },
          { formula: 'TGI: Best HuggingFace integration', explanation: 'Tight Hub integration. Rust performance. Production-proven at HuggingFace scale. Good quantization support.' },
          { formula: 'TensorRT-LLM: Maximum NVIDIA performance', explanation: '1.5-3x faster than vLLM on same hardware. Requires compilation step. Best for stable, high-volume deployments.' },
          { formula: 'llama.cpp: Best for CPU / edge / Mac', explanation: 'Runs anywhere. GGUF quantized models. Minimal dependencies. Great for local and low-resource deployments.' },
          { formula: 'LiteLLM: Best proxy / multi-provider gateway', explanation: 'Not a serving engine itself. Unified API across 100+ providers. Load balancing and cost tracking.' },
        ]}
        id="example-matrix"
      />

      <PythonCode
        title="benchmark_frameworks.py"
        code={`import requests
import time
import concurrent.futures
import json

def benchmark_endpoint(url, model, num_requests=50, max_tokens=128):
    """Benchmark an OpenAI-compatible endpoint."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Write a short paragraph about AI."}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    headers = {"Content-Type": "application/json"}

    latencies = []
    total_tokens = 0

    def send_one(_):
        start = time.time()
        resp = requests.post(url, json=payload, headers=headers)
        elapsed = time.time() - start
        data = resp.json()
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return elapsed, tokens

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(send_one, range(num_requests)))
    wall_time = time.time() - start

    for latency, tokens in results:
        latencies.append(latency)
        total_tokens += tokens

    latencies.sort()
    return {
        "total_time": round(wall_time, 2),
        "throughput_tok_s": round(total_tokens / wall_time, 1),
        "avg_latency": round(sum(latencies) / len(latencies), 3),
        "p50_latency": round(latencies[len(latencies)//2], 3),
        "p99_latency": round(latencies[int(len(latencies)*0.99)], 3),
        "total_tokens": total_tokens,
    }

# Compare frameworks (adjust URLs and models)
frameworks = {
    "vLLM":      ("http://localhost:8000/v1/chat/completions", "llama-3.1-8b"),
    "TGI":       ("http://localhost:8080/v1/chat/completions", "tgi"),
    "llama.cpp": ("http://localhost:8081/v1/chat/completions", "local"),
}

for name, (url, model) in frameworks.items():
    try:
        result = benchmark_endpoint(url, model, num_requests=50)
        print(f"\\n{name}:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"\\n{name}: unavailable ({e})")`}
        id="code-benchmark"
      />

      <PythonCode
        title="Terminal"
        code={`# Quick benchmarks with curl timing
# Time-to-first-token (TTFT) measurement
echo "--- vLLM TTFT ---"
curl -w "TTFT: %{time_starttransfer}s Total: %{time_total}s\\n" -s -o /dev/null \\
    http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"model":"llama","messages":[{"role":"user","content":"Hi"}],"max_tokens":1}'

echo "--- TGI TTFT ---"
curl -w "TTFT: %{time_starttransfer}s Total: %{time_total}s\\n" -s -o /dev/null \\
    http://localhost:8080/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"model":"tgi","messages":[{"role":"user","content":"Hi"}],"max_tokens":1}'

# Use wrk or hey for load testing
# hey -n 200 -c 32 -m POST \\
#   -H "Content-Type: application/json" \\
#   -d '{"model":"llama","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}' \\
#   http://localhost:8000/v1/chat/completions`}
        id="code-curl-bench"
      />

      <NoteBlock
        type="intuition"
        title="Decision Flowchart"
        content="Start with this: (1) Need to run on CPU or Mac? Use llama.cpp. (2) Need multi-provider routing? Use LiteLLM as a proxy in front of any engine. (3) Deploying on NVIDIA GPUs with stable config? Try TensorRT-LLM for max performance. (4) Want the easiest GPU deployment? Use vLLM. (5) Already in the HuggingFace ecosystem? TGI integrates seamlessly."
        id="note-flowchart"
      />

      <WarningBlock
        title="Benchmarks Depend on Context"
        content="Published benchmarks often use specific batch sizes, sequence lengths, and hardware that may not match your workload. Always benchmark with your actual model, expected request patterns, and hardware. A framework that wins on throughput may lose on latency, and vice versa."
        id="warning-benchmarks"
      />
    </div>
  )
}
