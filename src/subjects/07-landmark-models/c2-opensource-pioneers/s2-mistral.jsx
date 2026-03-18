import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Mistral() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Mistral 7B: Sliding Window Attention</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Mistral 7B (September 2023) from the Paris-based startup Mistral AI outperformed LLaMA 2 13B
        on all benchmarks and matched LLaMA 1 34B on many tasks, despite having only 7.3B parameters.
        It introduced sliding window attention (SWA) and a rolling KV cache to handle long sequences
        efficiently while maintaining strong performance.
      </p>

      <DefinitionBlock
        title="Sliding Window Attention (SWA)"
        definition="An attention pattern where each token attends only to the previous $W$ tokens (the window size) rather than all preceding tokens. With $W{=}4096$ and $L$ layers, information can propagate across $W \times L$ tokens through the network. For Mistral 7B with $W{=}4096$ and $L{=}32$, this gives a theoretical receptive field of $4096 \times 32 = 131{,}072$ tokens."
        id="def-swa"
      />

      <h2 className="text-2xl font-semibold">Architecture Details</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Mistral 7B uses 32 layers, a hidden dimension of 4096, 32 query heads with 8 KV heads (GQA),
        and an intermediate size of 14336. It combines SWA with GQA and a rolling KV cache that only
        stores the last W positions, reducing memory from O(n) to O(W) for the cache.
      </p>

      <ExampleBlock
        title="KV Cache Memory Savings"
        problem="Compare KV cache memory for a 16K token sequence between full attention and SWA with W=4096."
        steps={[
          { formula: '\\text{Full KV cache} = 2 \\times L \\times n \\times h_{kv} \\times d_h \\times 2\\text{B}', explanation: 'For Mistral 7B: 2 * 32 * 16384 * 8 * 128 * 2 = 2.1 GB in float16.' },
          { formula: '\\text{SWA KV cache} = 2 \\times L \\times W \\times h_{kv} \\times d_h \\times 2\\text{B}', explanation: 'With W=4096: 2 * 32 * 4096 * 8 * 128 * 2 = 0.5 GB in float16.' },
          { formula: '\\text{Savings} = 1 - \\frac{W}{n} = 1 - \\frac{4096}{16384} = 75\\%', explanation: 'The rolling cache gives 4x memory reduction for this sequence length.' },
        ]}
        id="example-kv-savings"
      />

      <h2 className="text-2xl font-semibold">Rolling Buffer Cache</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Instead of growing the KV cache linearly with sequence length, Mistral uses a fixed-size
        circular buffer of size W. Position i is stored at index (i mod W). When the buffer is
        full, the oldest entries are overwritten. This ensures constant memory usage regardless
        of sequence length.
      </p>

      <PythonCode
        title="mistral_sliding_window.py"
        code={`from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Mistral 7B
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Inspect sliding window config
config = model.config
print(f"Hidden size: {config.hidden_size}")             # 4096
print(f"Num layers: {config.num_hidden_layers}")        # 32
print(f"Num attention heads: {config.num_attention_heads}")  # 32
print(f"Num KV heads: {config.num_key_value_heads}")    # 8 (GQA)
print(f"Sliding window: {config.sliding_window}")       # 4096
print(f"Intermediate size: {config.intermediate_size}") # 14336
print(f"Vocab size: {config.vocab_size}")               # 32000

# Conceptual rolling buffer implementation
class RollingKVCache:
    def __init__(self, window_size, num_heads, head_dim, dtype=torch.float16):
        self.window_size = window_size
        self.keys = torch.zeros(window_size, num_heads, head_dim, dtype=dtype)
        self.values = torch.zeros(window_size, num_heads, head_dim, dtype=dtype)
        self.position = 0

    def update(self, key, value):
        idx = self.position % self.window_size
        self.keys[idx] = key
        self.values[idx] = value
        self.position += 1

    def get_cache(self):
        if self.position < self.window_size:
            return self.keys[:self.position], self.values[:self.position]
        return self.keys, self.values

# The cache never exceeds window_size entries
cache = RollingKVCache(window_size=4096, num_heads=8, head_dim=128)
print(f"Cache memory: {cache.keys.numel() * 2 * 2 / 1024**2:.1f} MB (fixed)")`}
        id="code-mistral-swa"
      />

      <NoteBlock
        type="note"
        title="Mistral vs LLaMA 2 Performance"
        content="Mistral 7B outperforms LLaMA 2 13B on MMLU (60.1 vs 54.8), HellaSwag (81.3 vs 80.7), and all reasoning benchmarks despite having nearly half the parameters. This efficiency comes from better training data curation, GQA, and the SWA mechanism that enables longer effective context."
        id="note-mistral-perf"
      />

      <WarningBlock
        title="SWA Information Loss"
        content="Sliding window attention means tokens beyond the window cannot be directly attended to. While information propagates through layers (giving a larger effective context), this is lossy -- the model cannot perfectly recall details from early in a very long document. For tasks requiring precise long-range retrieval, full attention or retrieval augmentation may be needed."
        id="warning-swa-loss"
      />
    </div>
  )
}
