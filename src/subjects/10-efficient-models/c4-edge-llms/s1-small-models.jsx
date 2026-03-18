import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function SmallLanguageModels() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Small Language Models: Phi, Gemma, and Beyond</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Small language models (SLMs) with 1-3 billion parameters challenge the assumption
        that scale is everything. Through careful data curation, architectural innovations,
        and training recipes, models like Phi-2, Gemma-2B, and StableLM demonstrate that
        compact models can match or exceed much larger models on many benchmarks.
      </p>

      <DefinitionBlock
        title="Data Quality Over Quantity"
        definition="The Phi series demonstrates that training data quality dominates model size for smaller models. Phi-1 (1.3B) was trained on 'textbook quality' data — a curated mix of filtered web data and synthetically generated textbook-style content. The key insight: $\text{Performance} \propto f(\text{data quality}) \cdot g(\text{model size})$ where data quality has a larger effect at small scales."
        id="def-data-quality"
      />

      <ExampleBlock
        title="Model Size and Memory Calculations"
        problem="Compare memory requirements for running Phi-2 (2.7B), Gemma-2B, and LLaMA-7B."
        steps={[
          {
            formula: '\\text{Phi-2: } 2.7B \\times 2 \\text{ bytes} = 5.4\\text{ GB (FP16)}',
            explanation: 'At FP16, each parameter uses 2 bytes.'
          },
          {
            formula: '\\text{Phi-2 INT4: } 2.7B \\times 0.5 = 1.35\\text{ GB}',
            explanation: 'Quantized to 4-bit, Phi-2 fits comfortably in phone RAM.'
          },
          {
            formula: '\\text{Gemma-2B INT4: } 2.0B \\times 0.5 = 1.0\\text{ GB}',
            explanation: 'Even smaller, with architecture optimized for the 2B scale.'
          },
          {
            formula: '\\text{LLaMA-7B INT4: } 7.0B \\times 0.5 = 3.5\\text{ GB}',
            explanation: 'For comparison, LLaMA-7B needs 2.5x more memory even at INT4.'
          }
        ]}
        id="example-model-sizes"
      />

      <PythonCode
        title="small_model_comparison.py"
        code={`import torch

# Model configurations for popular small LLMs
models = {
    "Phi-2 (2.7B)": {
        "layers": 32, "d_model": 2560, "heads": 32,
        "vocab": 51200, "context": 2048,
    },
    "Gemma-2B": {
        "layers": 18, "d_model": 2048, "heads": 8,
        "vocab": 256000, "context": 8192,
    },
    "StableLM-2-1.6B": {
        "layers": 24, "d_model": 2048, "heads": 32,
        "vocab": 100289, "context": 4096,
    },
    "TinyLlama-1.1B": {
        "layers": 22, "d_model": 2048, "heads": 32,
        "vocab": 32000, "context": 2048,
    },
    "LLaMA-7B (baseline)": {
        "layers": 32, "d_model": 4096, "heads": 32,
        "vocab": 32000, "context": 4096,
    },
}

def count_params(cfg):
    L, d, V = cfg["layers"], cfg["d_model"], cfg["vocab"]
    # Attention: Q, K, V, O
    attn = 4 * d * d * L
    # FFN: gate + up + down (SwiGLU style: 3 matrices of d x 4d/3*2)
    ffn = 3 * d * int(d * 8/3) * L  # Approximate SwiGLU
    # Embeddings (tied)
    embed = V * d
    # Layer norms
    norms = 2 * d * L
    return attn + ffn + embed + norms

def kv_cache_size(cfg, seq_len, n_kv_heads=None, dtype_bytes=2):
    d_head = cfg["d_model"] // cfg["heads"]
    kv_heads = n_kv_heads or cfg["heads"]
    # 2 (K+V) * layers * seq_len * kv_heads * d_head
    return 2 * cfg["layers"] * seq_len * kv_heads * d_head * dtype_bytes

print(f"{'Model':<22} {'Params':>10} {'FP16 GB':>8} {'INT4 GB':>8} {'KV@2K':>8}")
print("-" * 60)
for name, cfg in models.items():
    params = count_params(cfg)
    fp16_gb = params * 2 / 1e9
    int4_gb = params * 0.5 / 1e9
    kv_gb = kv_cache_size(cfg, 2048) / 1e9
    print(f"{name:<22} {params/1e9:>9.1f}B {fp16_gb:>7.1f} {int4_gb:>7.2f} {kv_gb:>7.3f}")

# Training data comparison
training_data = {
    "Phi-1 (1.3B)": "6B tokens (textbook quality + CodeExercises)",
    "Phi-2 (2.7B)": "1.4T tokens (filtered web + synthetic textbooks)",
    "Gemma-2B": "2T tokens (web, code, math)",
    "TinyLlama": "3T tokens (SlimPajama + StarCoder)",
    "LLaMA-7B": "1T tokens (CommonCrawl, C4, etc.)",
}
print("\\nTraining data:")
for name, data in training_data.items():
    print(f"  {name}: {data}")`}
        id="code-small-models"
      />

      <NoteBlock
        type="intuition"
        title="Why Small Models Can Punch Above Their Weight"
        content="Large models trained on internet-scale data spend capacity learning low-quality patterns, duplicated knowledge, and noise. Small models trained on curated data focus their limited capacity on high-value knowledge. This is why Phi-2 (2.7B) can match LLaMA-7B on reasoning benchmarks — it learned more efficiently from better data."
        id="note-small-model-intuition"
      />

      <NoteBlock
        type="note"
        title="Key Architectural Choices for Small Models"
        content="Successful SLMs use: (1) Grouped-query attention (GQA) to reduce KV cache, (2) SwiGLU or GeGLU activations for better parameter efficiency, (3) RoPE positional embeddings for context extension, (4) deeper-and-narrower architectures over shallow-and-wide ones. Gemma-2B uses only 8 KV heads (vs 8 query heads) to minimize serving cost."
        id="note-arch-choices"
      />

      <WarningBlock
        title="Benchmark Caveats"
        content="Small models often excel on academic benchmarks but struggle with complex multi-step reasoning, following nuanced instructions, and generating long coherent text. Their reduced capacity means they cannot store as much factual knowledge. Always evaluate on your specific use case, not just headline benchmark numbers."
        id="warning-benchmarks"
      />
    </div>
  )
}
