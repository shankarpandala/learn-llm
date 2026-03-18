import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function KVCache() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">KV-Cache & Paged Attention</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Autoregressive generation recomputes attention over all previous tokens at each step. The
        KV-cache stores previously computed key and value matrices, converting generation from
        quadratic to linear complexity per step. Paged Attention further optimizes memory allocation.
      </p>

      <DefinitionBlock
        title="KV-Cache"
        definition="During autoregressive generation, the key ($K$) and value ($V$) projections for all previous tokens are cached in GPU memory. At step $t$, only the new token's $Q$, $K$, $V$ are computed, and attention is: $\text{Attn}(q_t, K_{1:t}, V_{1:t}) = \text{softmax}(q_t K_{1:t}^T / \sqrt{d_k}) V_{1:t}$."
        notation="Without cache: $O(t^2 d)$ per step. With cache: $O(t \cdot d)$ per step."
        id="def-kv-cache"
      />

      <ExampleBlock
        title="KV-Cache Memory Calculation"
        problem="Calculate KV-cache memory for LLaMA-3 8B with sequence length 4096."
        steps={[
          { formula: 'Layers = 32, heads = 32, d_{head} = 128', explanation: 'LLaMA-3 8B architecture parameters.' },
          { formula: 'KV per token = 2 \\times 32 \\times 32 \\times 128 \\times 2 = 524\\text{KB}', explanation: '2 for K and V, 32 layers, 32 heads, 128 dims, 2 bytes (fp16).' },
          { formula: 'Total for 4096 tokens: 524\\text{KB} \\times 4096 \\approx 2\\text{GB}', explanation: 'Each sequence requires ~2GB of KV-cache in GPU memory.' },
          { formula: 'With GQA (8 KV heads): 2GB / 4 = 512\\text{MB}', explanation: 'Grouped-Query Attention reduces KV-cache by the group factor.' },
        ]}
        id="example-kv-memory"
      />

      <PythonCode
        title="kv_cache_demo.py"
        code={`import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()

prompt = "The key-value cache is important because"
input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

# WITHOUT KV-cache: recompute everything each step
start = time.time()
generated = input_ids.clone()
for _ in range(50):
    with torch.no_grad():
        outputs = model(generated)
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        generated = torch.cat([generated, next_token], dim=-1)
no_cache_time = time.time() - start

# WITH KV-cache: only compute the new token
start = time.time()
generated = input_ids.clone()
past_key_values = None
for _ in range(50):
    with torch.no_grad():
        outputs = model(
            generated[:, -1:] if past_key_values else generated,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        generated = torch.cat([generated, next_token], dim=-1)
cache_time = time.time() - start

print(f"Without KV-cache: {no_cache_time:.3f}s")
print(f"With KV-cache:    {cache_time:.3f}s")
print(f"Speedup:          {no_cache_time/cache_time:.1f}x")`}
        id="code-kv-cache"
      />

      <DefinitionBlock
        title="Paged Attention"
        definition="Paged Attention (vLLM) manages KV-cache like virtual memory pages. Instead of pre-allocating contiguous memory for the maximum sequence length, it allocates fixed-size blocks on demand and uses a block table to map logical positions to physical memory locations."
        id="def-paged-attention"
      />

      <NoteBlock
        type="intuition"
        title="Why Paging Matters"
        content="Without paging, serving a batch of 8 requests with max_length=4096 requires pre-allocating 8 x 2GB = 16GB even if most sequences are short. Paged Attention only allocates memory as tokens are generated, reducing waste by 60-80% and enabling 2-4x higher throughput."
        id="note-paging-intuition"
      />

      <PythonCode
        title="Terminal"
        code={`# vLLM uses Paged Attention automatically
# Install and run vLLM to see the benefits
pip install vllm

# Start vLLM server with automatic KV-cache management
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --gpu-memory-utilization 0.9 \\
    --max-model-len 8192

# vLLM reports KV-cache usage in logs:
# INFO: GPU blocks: 2450, CPU blocks: 512
# Each block holds KV for 16 tokens
# Total KV-cache capacity: 2450 * 16 = 39,200 tokens`}
        id="code-vllm-paged"
      />

      <WarningBlock
        title="KV-Cache Is the Memory Bottleneck"
        content="For long-context models (128K+ tokens), KV-cache can exceed the model weights in memory usage. A 7B model with 128K context needs ~32GB just for KV-cache in fp16. Techniques like GQA, KV-cache quantization (fp8), and sliding window attention are essential for long contexts."
        id="warning-memory"
      />

      <NoteBlock
        type="note"
        title="Prefix Caching"
        content="When multiple requests share a common prefix (system prompt), the KV-cache for that prefix can be computed once and shared. vLLM supports automatic prefix caching, which is especially valuable for chat applications where every request includes the same system prompt."
        id="note-prefix-caching"
      />
    </div>
  )
}
