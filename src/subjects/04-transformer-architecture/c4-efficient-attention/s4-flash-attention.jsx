import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function FlashAttention() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">FlashAttention: IO-Aware Exact Attention</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        FlashAttention computes exact softmax attention without materializing the full{' '}
        <InlineMath math="n \times n" /> attention matrix in GPU HBM. By using tiling and
        recomputation, it reduces memory usage from <InlineMath math="O(n^2)" /> to{' '}
        <InlineMath math="O(n)" /> while also running 2-4x faster than standard attention
        by minimizing HBM reads/writes.
      </p>

      <DefinitionBlock
        title="FlashAttention"
        definition="An IO-aware attention algorithm that tiles the Q, K, V matrices into blocks that fit in GPU SRAM (shared memory), computes partial attention within tiles using the online softmax trick, and never writes the full $n \times n$ attention matrix to HBM."
        notation="HBM = High Bandwidth Memory (GPU main memory, ~2TB/s), SRAM = on-chip memory (~19TB/s but only ~20MB)"
        id="def-flash-attention"
      />

      <h2 className="text-2xl font-semibold">The GPU Memory Hierarchy</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The key insight is that GPU compute (TFLOPs) has grown much faster than memory bandwidth.
        Standard attention is bottlenecked by reading/writing the <InlineMath math="n^2" />{' '}
        attention matrix to HBM. FlashAttention keeps intermediate results in fast SRAM.
      </p>

      <ExampleBlock
        title="IO Cost: Standard vs. FlashAttention"
        problem="Compare HBM reads/writes for n=4096, d=64 with block size B=256."
        steps={[
          { formula: '\\text{Standard: read } Q,K,V \\in O(nd) + \\text{write } A \\in O(n^2)', explanation: 'Must write the full 4096² attention matrix to HBM.' },
          { formula: '\\text{Standard IO: } O(n^2) = 4096^2 \\approx 16.7\\text{M elements}', explanation: 'Memory bandwidth becomes the bottleneck, not compute.' },
          { formula: '\\text{Flash: read } Q,K,V \\text{ in blocks of } B=256', explanation: 'Process 256-token tiles that fit in SRAM (~100KB each).' },
          { formula: '\\text{Flash IO: } O(n^2 d / M) \\text{ where } M = \\text{SRAM size}', explanation: 'With sufficient SRAM, IO is subquadratic. Never writes full attention matrix.' },
        ]}
        id="example-io-cost"
      />

      <NoteBlock
        type="intuition"
        title="The Online Softmax Trick"
        content="The main challenge with tiling is that softmax requires knowing the maximum value across the entire row (for numerical stability) before computing exp(). The online softmax algorithm (Milakov & Gimelshein, 2018) maintains running max and sum statistics, allowing softmax to be computed incrementally across tiles without seeing the full row at once."
        id="note-online-softmax"
      />

      <PythonCode
        title="flash_attention_concept.py"
        code={`import torch
import torch.nn.functional as F
import math

def flash_attention_simplified(Q, K, V, block_size=256):
    """
    Simplified FlashAttention in pure PyTorch (for understanding).
    Real FlashAttention uses custom CUDA kernels for SRAM tiling.
    """
    B, n, d = Q.shape
    output = torch.zeros_like(V)
    row_max = torch.full((B, n, 1), float('-inf'), device=Q.device)
    row_sum = torch.zeros(B, n, 1, device=Q.device)

    num_blocks = math.ceil(n / block_size)

    for j in range(num_blocks):
        # Load one block of K, V
        j_start = j * block_size
        j_end = min(j_start + block_size, n)
        K_block = K[:, j_start:j_end, :]
        V_block = V[:, j_start:j_end, :]

        # Compute scores for this block
        scores = torch.matmul(Q, K_block.transpose(-2, -1)) / math.sqrt(d)

        # Online softmax update
        block_max = scores.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(row_max, block_max)

        # Rescale previous accumulator
        exp_old = torch.exp(row_max - new_max)
        exp_new = torch.exp(scores - new_max)

        # Update output and statistics
        output = output * exp_old
        output = output + torch.matmul(exp_new, V_block)
        row_sum = row_sum * exp_old + exp_new.sum(dim=-1, keepdim=True)
        row_max = new_max

    # Final normalization
    output = output / row_sum
    return output

# Verify correctness against standard attention
B, n, d = 2, 1024, 64
Q = torch.randn(B, n, d)
K = torch.randn(B, n, d)
V = torch.randn(B, n, d)

standard_out = F.scaled_dot_product_attention(Q, K, V)
flash_out = flash_attention_simplified(Q, K, V, block_size=128)
max_diff = (standard_out - flash_out).abs().max().item()
print(f"Max difference: {max_diff:.8f}")  # Should be ~1e-6 (numerical)
print(f"Results match: {max_diff < 1e-4}")

# PyTorch 2.0+ uses FlashAttention automatically
print("\\nPyTorch SDPA backends:")
print(f"  FlashAttention available: {torch.backends.cuda.flash_sdp_enabled()}"
      if hasattr(torch.backends, 'cuda') else "  (CPU — no FlashAttention)")

# Memory comparison (conceptual)
n_vals = [1024, 4096, 16384]
for n in n_vals:
    standard_mem = n * n * 4 / 1e6   # fp32 attention matrix in MB
    flash_mem = 2 * n * d * 4 / 1e6  # Only Q, K, V blocks + O(n) stats
    print(f"n={n:>6}: standard={standard_mem:>8.1f}MB, flash={flash_mem:>6.1f}MB, "
          f"saving={standard_mem/flash_mem:.0f}x")`}
        id="code-flash"
      />

      <WarningBlock
        title="FlashAttention Is Not an Approximation"
        content="Unlike sparse or linear attention, FlashAttention computes the exact same result as standard softmax attention (up to floating-point precision). It is purely an implementation optimization — same math, different computation order. This is why it has become universally adopted: no quality tradeoff."
        id="warning-exact"
      />

      <NoteBlock
        type="note"
        title="FlashAttention Versions"
        content="FlashAttention v1 (Dao et al., 2022) introduced tiled attention with online softmax. FlashAttention v2 (2023) improved parallelism and work partitioning for 2x further speedup. FlashAttention v3 (2024) leveraged H100-specific features (TMA, FP8). The algorithm is now the default backend in PyTorch's scaled_dot_product_attention."
        id="note-versions"
      />

      <NoteBlock
        type="tip"
        title="Using FlashAttention in Practice"
        content="In PyTorch 2.0+, simply use torch.nn.functional.scaled_dot_product_attention(Q, K, V) — it automatically selects FlashAttention when available. For Hugging Face models, pass attn_implementation='flash_attention_2' to from_pretrained(). No code changes needed for the standard case."
        id="note-practical"
      />
    </div>
  )
}
