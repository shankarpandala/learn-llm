import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LatencyOptimization() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Latency Optimization: Batching, Caching, and Scheduling</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Serving LLMs efficiently at scale requires optimizing beyond the model itself.
        Key-value caching avoids redundant computation, continuous batching maximizes GPU
        utilization, and memory management strategies like PagedAttention eliminate
        fragmentation. Together, these techniques can improve serving throughput by 10-20x.
      </p>

      <DefinitionBlock
        title="KV Cache"
        definition="During autoregressive generation, the KV cache stores the key and value tensors from all previous tokens, avoiding recomputation. For each new token, only one new K and V vector per layer is computed and appended. Cache size: $2 \times L \times n_{kv} \times d_h \times T$ values, where $L$ is layers, $n_{kv}$ is KV heads, $d_h$ is head dimension, and $T$ is sequence length."
        notation="Without KV cache: generating $T$ tokens costs $O(T^2 \cdot L \cdot d)$ FLOPs. With KV cache: $O(T \cdot L \cdot d)$ — a $T\times$ speedup."
        id="def-kv-cache"
      />

      <DefinitionBlock
        title="Continuous Batching"
        definition="Continuous (or inflight) batching dynamically adds and removes requests from a running batch as individual sequences finish. Unlike static batching (which waits for the longest sequence), continuous batching immediately fills freed slots with new requests, achieving near-100% GPU utilization."
        id="def-continuous-batching"
      />

      <ExampleBlock
        title="KV Cache Memory Calculation"
        problem="Calculate KV cache memory for a 7B model serving 32 concurrent users with 4K context."
        steps={[
          {
            formula: '\\text{LLaMA-7B: } L=32, n_{kv}=32, d_h=128',
            explanation: 'Model configuration: 32 layers, 32 KV heads, head dim 128.'
          },
          {
            formula: '\\text{Per-token KV} = 2 \\times 32 \\times 32 \\times 128 \\times 2 = 524\\text{ KB}',
            explanation: '2 (K+V) * layers * heads * dim * 2 bytes (FP16).'
          },
          {
            formula: '\\text{Per-user (4K ctx)} = 524\\text{ KB} \\times 4096 = 2.1\\text{ GB}',
            explanation: 'Each user with full 4K context needs 2.1 GB of KV cache.'
          },
          {
            formula: '\\text{32 users} = 2.1 \\times 32 = 67.1\\text{ GB}',
            explanation: 'KV cache alone exceeds one 80GB GPU! This is why GQA and PagedAttention matter.'
          }
        ]}
        id="example-kv-cache-memory"
      />

      <PythonCode
        title="serving_optimizations.py"
        code={`import torch
import time
from collections import deque

class KVCache:
    """Simple KV cache implementation."""
    def __init__(self, n_layers, n_heads, head_dim, max_seq_len, dtype=torch.float16):
        self.n_layers = n_layers
        self.cache_k = torch.zeros(n_layers, max_seq_len, n_heads, head_dim, dtype=dtype)
        self.cache_v = torch.zeros(n_layers, max_seq_len, n_heads, head_dim, dtype=dtype)
        self.seq_len = 0

    def update(self, layer_idx, new_k, new_v):
        """Append new K, V vectors for one token."""
        self.cache_k[layer_idx, self.seq_len] = new_k
        self.cache_v[layer_idx, self.seq_len] = new_v

    def get(self, layer_idx):
        """Get all cached K, V up to current sequence length."""
        return (self.cache_k[layer_idx, :self.seq_len + 1],
                self.cache_v[layer_idx, :self.seq_len + 1])

    def advance(self):
        self.seq_len += 1

    def memory_bytes(self):
        return self.cache_k.numel() * 2 + self.cache_v.numel() * 2  # FP16

# Memory analysis for different configurations
configs = {
    "LLaMA-7B": {"layers": 32, "kv_heads": 32, "head_dim": 128},
    "LLaMA-7B GQA-4": {"layers": 32, "kv_heads": 4, "head_dim": 128},
    "Mistral-7B": {"layers": 32, "kv_heads": 8, "head_dim": 128},
}

print("KV Cache Memory (per user, FP16):")
print(f"{'Model':<20} {'1K ctx':>8} {'4K ctx':>8} {'32K ctx':>8} {'128K ctx':>9}")
for name, cfg in configs.items():
    for ctx in [1024, 4096, 32768, 131072]:
        cache = KVCache(cfg["layers"], cfg["kv_heads"], cfg["head_dim"], ctx)
        mb = cache.memory_bytes() / 1e6
        if ctx == 1024:
            print(f"{name:<20}", end="")
        print(f" {mb:>7.0f}M", end="")
    print()

# Continuous batching simulator
class ContinuousBatcher:
    """Simulates continuous batching for LLM serving."""
    def __init__(self, max_batch_size=32):
        self.max_batch = max_batch_size
        self.active = {}       # slot_id -> remaining_tokens
        self.queue = deque()   # Waiting requests
        self.completed = 0
        self.total_tokens = 0
        self.idle_slots = 0

    def add_request(self, request_id, output_length):
        self.queue.append((request_id, output_length))

    def step(self):
        """Process one generation step for all active requests."""
        # Fill empty slots from queue
        while len(self.active) < self.max_batch and self.queue:
            req_id, length = self.queue.popleft()
            slot = len(self.active)
            self.active[slot] = (req_id, length)

        # Generate one token for each active request
        finished = []
        for slot, (req_id, remaining) in self.active.items():
            remaining -= 1
            if remaining <= 0:
                finished.append(slot)
                self.completed += 1
            else:
                self.active[slot] = (req_id, remaining)

        # Free finished slots immediately (continuous batching!)
        for slot in finished:
            del self.active[slot]

        active_count = len(self.active)
        self.idle_slots += self.max_batch - active_count
        self.total_tokens += active_count
        return active_count

# Simulate serving workload
batcher = ContinuousBatcher(max_batch_size=16)
import random
random.seed(42)
for i in range(100):
    batcher.add_request(f"req_{i}", random.randint(10, 200))

steps = 0
while batcher.active or batcher.queue:
    batcher.step()
    steps += 1

utilization = batcher.total_tokens / (steps * 16)
print(f"\\nContinuous Batching Simulation:")
print(f"  Requests served: {batcher.completed}")
print(f"  Total steps: {steps}")
print(f"  GPU utilization: {utilization:.1%}")
print(f"  Throughput: {batcher.total_tokens / steps:.1f} tokens/step")`}
        id="code-serving"
      />

      <NoteBlock
        type="note"
        title="PagedAttention (vLLM)"
        content="PagedAttention (Kwon et al., 2023) manages KV cache like virtual memory pages. Instead of pre-allocating contiguous memory per sequence, it allocates fixed-size blocks on demand. This eliminates 60-80% of memory waste from internal fragmentation and enables serving 2-4x more concurrent users on the same hardware."
        id="note-paged-attention"
      />

      <NoteBlock
        type="tip"
        title="Optimization Priority"
        content="For single-user latency: (1) quantize weights to INT4, (2) enable KV cache, (3) use Flash Attention, (4) consider speculative decoding. For multi-user throughput: (1) continuous batching, (2) PagedAttention, (3) tensor parallelism across GPUs, (4) quantization for memory savings. Profile your specific workload to find the bottleneck."
        id="note-optimization-priority"
      />

      <WarningBlock
        title="Prefill vs. Decode Bottlenecks"
        content="LLM serving has two distinct phases: prefill (processing the prompt, compute-bound) and decode (generating tokens one at a time, memory-bound). Optimizing for one can hurt the other. Chunked prefill and separate prefill/decode batching help manage this tension in production systems."
        id="warning-prefill-decode"
      />
    </div>
  )
}
