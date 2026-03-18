import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function AttentionComplexity() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">The O(n²) Attention Problem</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Standard self-attention computes pairwise interactions between all token pairs, resulting
        in quadratic time and memory complexity with respect to sequence length. This bottleneck
        limits the context length that transformers can practically handle.
      </p>

      <DefinitionBlock
        title="Attention Complexity"
        definition="For a sequence of length $n$ with model dimension $d$, scaled dot-product attention requires: time $O(n^2 d)$ for the matrix multiplication $QK^T$, and memory $O(n^2)$ to store the attention matrix (per head)."
        notation="n = sequence length, d = model/head dimension, h = number of heads"
        id="def-complexity"
      />

      <h2 className="text-2xl font-semibold">Where the Quadratic Cost Comes From</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The attention matrix <InlineMath math="A = \text{softmax}(QK^T / \sqrt{d_k})" /> has
        shape <InlineMath math="n \times n" />. Computing it requires <InlineMath math="n^2" />{' '}
        dot products, and storing it requires <InlineMath math="n^2" /> floating-point numbers
        per head. For long sequences, memory is often the binding constraint.
      </p>
      <BlockMath math="\text{Memory: } n^2 \cdot h \cdot \text{sizeof(float)} \quad \text{FLOPs: } 2n^2 d \text{ per QK}^T + 2n^2 d_v \text{ per AV}" />

      <ExampleBlock
        title="Memory Requirements at Scale"
        problem="Calculate attention memory for different sequence lengths with h=32 heads in fp16."
        steps={[
          { formula: 'n=2048: \\; 2048^2 \\times 32 \\times 2\\text{B} = 256\\text{MB}', explanation: 'Manageable — fits easily on a modern GPU.' },
          { formula: 'n=8192: \\; 8192^2 \\times 32 \\times 2\\text{B} = 4\\text{GB}', explanation: 'Significant — consumes a large fraction of GPU memory.' },
          { formula: 'n=32768: \\; 32768^2 \\times 32 \\times 2\\text{B} = 64\\text{GB}', explanation: 'Exceeds most single GPU memory — requires special techniques.' },
          { formula: 'n=131072: \\; 131072^2 \\times 32 \\times 2\\text{B} = 1\\text{TB}', explanation: 'Impossible with naive attention — FlashAttention or sparse methods required.' },
        ]}
        id="example-memory"
      />

      <PythonCode
        title="attention_complexity_benchmark.py"
        code={`import torch
import torch.nn.functional as F
import math
import time

def benchmark_attention(seq_lengths, d_model=64, num_heads=1, device='cpu'):
    """Benchmark attention time and memory for different sequence lengths."""
    results = []
    for n in seq_lengths:
        Q = torch.randn(1, num_heads, n, d_model, device=device)
        K = torch.randn(1, num_heads, n, d_model, device=device)
        V = torch.randn(1, num_heads, n, d_model, device=device)

        # Time the attention computation
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Memory of attention matrix
        attn_mem_mb = n * n * num_heads * 4 / (1024 ** 2)  # fp32

        results.append({
            'seq_len': n,
            'time_ms': elapsed * 1000,
            'attn_matrix_mb': attn_mem_mb,
        })
        del Q, K, V, scores, attn, out

    return results

# Run benchmark
lengths = [128, 256, 512, 1024, 2048, 4096]
results = benchmark_attention(lengths)

print(f"{'Seq Len':>8} | {'Time (ms)':>10} | {'Attn Mem (MB)':>14} | {'Ratio':>6}")
print("-" * 50)
for i, r in enumerate(results):
    ratio = r['time_ms'] / results[0]['time_ms'] if i > 0 else 1.0
    print(f"{r['seq_len']:>8} | {r['time_ms']:>10.2f} | "
          f"{r['attn_matrix_mb']:>14.2f} | {ratio:>6.1f}x")

# Theoretical scaling
print("\\nTheoretical O(n²) scaling:")
for n in lengths:
    relative = (n / lengths[0]) ** 2
    print(f"  n={n}: {relative:.1f}x vs n={lengths[0]}")`}
        id="code-benchmark"
      />

      <NoteBlock
        type="note"
        title="FLOPs vs. Memory Bottleneck"
        content="Modern GPUs have enormous compute throughput but limited memory bandwidth. The attention matrix is often memory-bound rather than compute-bound. For n=8192 with 32 heads, the attention matrices alone consume 4GB — before accounting for gradients, activations, and the model itself. This is why memory-efficient methods like FlashAttention focus on IO rather than FLOPs."
        id="note-memory-bound"
      />

      <WarningBlock
        title="Quadratic Scaling Limits Context Length"
        content="Doubling the sequence length quadruples both compute and memory for attention. This is why early transformers used max_len=512 (BERT) or 1024 (GPT-2). Reaching 100K+ context windows required architectural innovations (FlashAttention, sparse patterns, linear attention) and significant engineering."
        id="warning-scaling-limit"
      />

      <NoteBlock
        type="intuition"
        title="Not All Attention Weights Matter"
        content="Empirically, attention matrices are often sparse — most weights are near zero. This observation motivates sparse attention methods: if most entries are negligible, we can approximate the full attention by computing only the important entries, reducing O(n²) toward O(n log n) or O(n)."
        id="note-sparsity"
      />
    </div>
  )
}
