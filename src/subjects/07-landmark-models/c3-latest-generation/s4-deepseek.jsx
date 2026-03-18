import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function DeepSeek() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">DeepSeek V2 and V3: Efficient Scale</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        DeepSeek, a Chinese AI lab backed by the quantitative trading firm High-Flyer, produced
        some of the most innovative open models. DeepSeek-V2 introduced Multi-head Latent Attention
        (MLA) and DeepSeekMoE, while DeepSeek-V3 scaled to 671B total parameters with only 37B
        active, matching frontier closed models at a fraction of the training cost.
      </p>

      <DefinitionBlock
        title="Multi-head Latent Attention (MLA)"
        definition="An attention mechanism that compresses the KV cache by projecting keys and values into a low-rank latent space before storage. Instead of caching full-dimensional KV pairs, MLA stores a compressed latent vector $c_t = W_{DKV} [k_t; v_t]$ of dimension $d_c \ll 2 \times n_h \times d_h$. At attention time, keys and values are reconstructed from the latent. This reduces KV cache by 93.3% compared to standard MHA."
        id="def-mla"
      />

      <h2 className="text-2xl font-semibold">DeepSeek-V2 Architecture</h2>
      <p className="text-gray-700 dark:text-gray-300">
        DeepSeek-V2 (236B total, 21B active) combined MLA with a fine-grained MoE architecture
        using 160 routed experts and 2 shared experts per layer, with top-6 routing. This achieved
        GPT-4-level performance while requiring only 42.5% of the training compute of DeepSeek-V1 67B.
      </p>

      <ExampleBlock
        title="DeepSeek Architecture Comparison"
        problem="Compare DeepSeek V2 and V3 architectures."
        steps={[
          { formula: '\\text{V2}: 236\\text{B total}, 21\\text{B active}, L{=}60', explanation: 'MLA + DeepSeekMoE (160 routed + 2 shared experts per layer, top-6). Context: 128K.' },
          { formula: '\\text{V3}: 671\\text{B total}, 37\\text{B active}, L{=}61', explanation: 'MLA + DeepSeekMoE (256 routed + 1 shared expert, top-8). Auxiliary-loss-free load balancing.' },
          { formula: '\\text{V3 KV cache}: d_c = 512 \\text{ vs MHA } d_{kv} = 7680', explanation: 'MLA compresses KV cache from 7680 dimensions to 512, a 93.3% reduction per layer.' },
          { formula: '\\text{V3 training cost}: \\sim\\$5.5\\text{M} (2.788\\text{M H800 GPU-hours})', explanation: 'Remarkably low cost for a frontier model. 14.8T training tokens with FP8 mixed precision.' },
        ]}
        id="example-deepseek-arch"
      />

      <h2 className="text-2xl font-semibold">FP8 Mixed-Precision Training</h2>
      <p className="text-gray-700 dark:text-gray-300">
        DeepSeek-V3 pioneered FP8 training for large MoE models, using 8-bit floating point for
        most matrix multiplications while maintaining FP32 master weights. This nearly doubled
        training throughput compared to BF16 on H800 GPUs, contributing significantly to the
        low training cost.
      </p>

      <PythonCode
        title="deepseek_usage.py"
        code={`from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load DeepSeek-V2-Lite (a smaller variant for experimentation)
model_name = "deepseek-ai/DeepSeek-V2-Lite"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Inspect MoE configuration
config = model.config
print(f"Hidden size: {config.hidden_size}")
print(f"Num layers: {config.num_hidden_layers}")
print(f"Num experts: {getattr(config, 'n_routed_experts', 'N/A')}")
print(f"Top-K experts: {getattr(config, 'num_experts_per_tok', 'N/A')}")

# Count total vs active parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e9:.1f}B")

# MLA KV cache comparison
d_model = 5120   # DeepSeek-V2 hidden dim
n_heads = 128    # number of attention heads
d_head = 128     # head dimension
d_c = 512        # MLA compressed dimension

mha_kv_size = 2 * n_heads * d_head  # standard MHA KV per token
mla_kv_size = d_c                    # MLA compressed KV per token

print(f"\\nKV cache per token per layer:")
print(f"  Standard MHA: {mha_kv_size} dimensions ({mha_kv_size * 2} bytes in FP16)")
print(f"  MLA: {mla_kv_size} dimensions ({mla_kv_size * 2} bytes in FP16)")
print(f"  Compression ratio: {mla_kv_size / mha_kv_size:.1%}")

# For a 128K context, 60-layer model
seq_len = 131072
for name, kv_dim in [("MHA", mha_kv_size), ("MLA", mla_kv_size)]:
    cache_gb = 2 * 60 * seq_len * kv_dim * 2 / 1e9
    print(f"  {name} total KV cache (128K ctx): {cache_gb:.1f} GB")`}
        id="code-deepseek"
      />

      <NoteBlock
        type="intuition"
        title="Why MLA Is Transformative"
        content="Traditional KV cache grows as O(n * L * h * d_h) and is the primary bottleneck for long-context inference. MLA compresses this to O(n * L * d_c) where d_c is much smaller than h * d_h. Think of it as storing a compressed summary of each token's key-value information rather than the full representation, with the ability to reconstruct the full KV when needed."
        id="note-mla-intuition"
      />

      <WarningBlock
        title="Custom CUDA Kernels Required"
        content="DeepSeek-V2/V3's MLA and fine-grained MoE require custom CUDA kernels for efficient inference. Standard HuggingFace transformers inference may be significantly slower than the optimized implementation. For production deployment, use the official DeepSeek inference framework or vLLM with DeepSeek support."
        id="warning-deepseek-kernels"
      />
    </div>
  )
}
