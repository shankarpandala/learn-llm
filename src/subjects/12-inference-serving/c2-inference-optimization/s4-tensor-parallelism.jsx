import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function TensorParallelism() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Tensor Parallelism for Inference</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        When a model is too large for a single GPU, tensor parallelism splits individual weight
        matrices across multiple GPUs so they compute portions of each layer simultaneously.
        Pipeline parallelism is the alternative, splitting by layers instead of within layers.
      </p>

      <DefinitionBlock
        title="Tensor Parallelism (TP)"
        definition="Tensor parallelism partitions weight matrices across $N$ GPUs. For a linear layer $Y = XW$, the weight $W$ is split column-wise: $W = [W_1 | W_2 | \ldots | W_N]$. Each GPU computes $Y_i = X W_i$, then results are combined via all-reduce or all-gather."
        notation="TP degree $N$ = number of GPUs sharing each layer. Communication cost: one all-reduce per layer."
        id="def-tensor-parallelism"
      />

      <DefinitionBlock
        title="Pipeline Parallelism (PP)"
        definition="Pipeline parallelism assigns different layers to different GPUs. GPU 0 runs layers 0-15, GPU 1 runs layers 16-31. Data flows sequentially through GPUs. This requires less inter-GPU communication but creates pipeline bubbles."
        id="def-pipeline-parallelism"
      />

      <ExampleBlock
        title="LLaMA-3.1 70B Across GPUs"
        problem="How to serve a 70B parameter model that requires ~140GB in fp16?"
        steps={[
          { formula: 'Single GPU (80GB A100): not enough for fp16', explanation: 'Need at least 140GB for weights alone, plus KV-cache and activations.' },
          { formula: 'TP=2 on 2x A100: 70GB weights per GPU', explanation: 'Each GPU holds half of every weight matrix. Fast all-reduce via NVLink.' },
          { formula: 'TP=4 on 4x A100: 35GB per GPU, room for large batches', explanation: 'More headroom for KV-cache means higher throughput.' },
          { formula: 'PP=2 + TP=2 on 4x A100: hybrid approach', explanation: 'Layers 0-39 on GPUs 0-1 (TP=2), layers 40-79 on GPUs 2-3 (TP=2).' },
        ]}
        id="example-70b-serving"
      />

      <PythonCode
        title="Terminal"
        code={`# vLLM with tensor parallelism
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-70B-Instruct \\
    --tensor-parallel-size 4 \\
    --gpu-memory-utilization 0.9 \\
    --max-model-len 8192

# TGI with tensor parallelism
docker run --gpus all -p 8080:80 \\
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \\
    ghcr.io/huggingface/text-generation-inference:latest \\
    --model-id meta-llama/Llama-3.1-70B-Instruct \\
    --num-shard 4

# Check GPU memory distribution
nvidia-smi --query-gpu=index,memory.used,memory.total \\
    --format=csv,noheader,nounits`}
        id="code-tp-serving"
      />

      <PythonCode
        title="tp_benchmark.py"
        code={`import requests
import time
import concurrent.futures

API_URL = "http://localhost:8000/v1/completions"

def send_request(prompt):
    start = time.time()
    resp = requests.post(API_URL, json={
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
    })
    latency = time.time() - start
    tokens = resp.json()["usage"]["completion_tokens"]
    return latency, tokens

# Benchmark: measure throughput with concurrent requests
prompts = [f"Explain concept {i} in machine learning:" for i in range(32)]

start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    results = list(pool.map(send_request, prompts))
total_time = time.time() - start

total_tokens = sum(t for _, t in results)
avg_latency = sum(l for l, _ in results) / len(results)

print(f"Total time: {total_time:.1f}s")
print(f"Throughput: {total_tokens/total_time:.0f} tokens/sec")
print(f"Avg latency: {avg_latency:.2f}s")
print(f"Requests completed: {len(results)}")`}
        id="code-benchmark"
      />

      <NoteBlock
        type="tip"
        title="Choosing TP vs PP"
        content="Use tensor parallelism when GPUs are connected via fast interconnects (NVLink, 600 GB/s). Use pipeline parallelism when GPUs are on different nodes connected by slower networks (InfiniBand, 200 Gb/s). For inference, TP is almost always preferred because it reduces latency -- every GPU participates in every token."
        id="note-tp-vs-pp"
      />

      <WarningBlock
        title="Diminishing Returns with High TP"
        content="Communication overhead grows with TP degree. TP=8 on 8 GPUs means 8 all-reduce operations per layer. If the all-reduce time exceeds the compute time, adding more GPUs actually hurts latency. For small models, TP=2 is often the sweet spot."
        id="warning-diminishing-returns"
      />

      <NoteBlock
        type="note"
        title="Quantization as an Alternative"
        content="Before reaching for multi-GPU serving, consider quantization. A 70B model in 4-bit quantization fits on a single 80GB GPU (~35GB). This avoids all communication overhead. Quality loss is minimal with modern quantization methods (GPTQ, AWQ)."
        id="note-quantization-alt"
      />
    </div>
  )
}
