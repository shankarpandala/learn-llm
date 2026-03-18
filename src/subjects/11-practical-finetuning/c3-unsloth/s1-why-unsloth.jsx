import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function WhyUnsloth() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Why Unsloth: 2x Faster, 60% Less Memory</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Unsloth is an open-source library that optimizes LLM finetuning through custom Triton
        kernels and intelligent memory management. It provides dramatic speedups and memory
        reductions while maintaining full compatibility with the Hugging Face ecosystem.
      </p>

      <DefinitionBlock
        title="Unsloth"
        definition="Unsloth is a finetuning acceleration library that rewrites key operations (cross-entropy loss, RoPE embeddings, attention) in custom Triton kernels. It achieves 2x training speedup and 60% memory reduction compared to standard Hugging Face training, with zero accuracy loss."
        id="def-unsloth"
      />

      <h2 className="text-2xl font-semibold">How Unsloth Achieves Its Speedups</h2>

      <ExampleBlock
        title="Unsloth Optimizations"
        problem="What specific optimizations does Unsloth apply compared to standard HF training?"
        steps={[
          { formula: '\\text{Fused cross-entropy: } O(V) \\rightarrow O(1) \\text{ memory}', explanation: 'Standard cross-entropy materializes logits for full vocabulary (~128K). Unsloth computes loss in chunks.' },
          { formula: '\\text{Manual autograd: no intermediate storage}', explanation: 'Custom backward passes skip PyTorch autograd overhead, saving memory.' },
          { formula: '\\text{Fused RoPE + RMSNorm kernels}', explanation: 'Operations are fused into single Triton kernels, reducing memory transfers.' },
          { formula: '\\text{Intelligent gradient checkpointing}', explanation: 'Selective checkpointing of only the most memory-intensive operations.' },
        ]}
        id="example-optimizations"
      />

      <PythonCode
        title="unsloth_vs_standard.py"
        code={`# Memory comparison: Standard HF vs Unsloth
# For LLaMA 3.1 8B with QLoRA, batch_size=2, seq_len=2048

standard_hf = {
    "base_model_4bit": 5.0,      # GB
    "lora_adapters": 0.3,        # GB
    "optimizer_states": 0.6,     # GB (8-bit paged AdamW)
    "gradients": 0.3,            # GB
    "activations": 4.8,          # GB (gradient checkpointing)
    "logits_buffer": 2.0,        # GB (vocab_size * batch * seq)
    "total": 13.0,               # GB
}

unsloth = {
    "base_model_4bit": 5.0,      # GB (same)
    "lora_adapters": 0.3,        # GB (same)
    "optimizer_states": 0.6,     # GB (same)
    "gradients": 0.3,            # GB (same)
    "activations": 1.5,          # GB (optimized checkpointing)
    "logits_buffer": 0.01,       # GB (chunked cross-entropy)
    "total": 7.7,                # GB
}

print("Memory Comparison (LLaMA 3.1 8B QLoRA):")
print(f"{'Component':<25} {'Standard':>10} {'Unsloth':>10} {'Savings':>10}")
print("-" * 60)
for key in standard_hf:
    std = standard_hf[key]
    uns = unsloth[key]
    savings = (1 - uns/std) * 100 if std > 0 else 0
    print(f"{key:<25} {std:>9.1f}G {uns:>9.1f}G {savings:>9.0f}%")

# Output:
# Total savings: ~40% memory, allowing larger batch sizes
# Speed: ~2x faster due to fused kernels and reduced memory ops`}
        id="code-comparison"
      />

      <PythonCode
        title="unsloth_quick_start.py"
        code={`from unsloth import FastLanguageModel
import torch

# Unsloth provides a simplified API
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    dtype=None,          # Auto-detect (bf16 on Ampere+)
    load_in_4bit=True,   # QLoRA
)

# Add LoRA with Unsloth optimizations
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,       # Unsloth recommends 0 for speed
    bias="none",
    use_gradient_checkpointing="unsloth",  # Custom checkpointing
    use_rslora=False,
)

# Check optimizations applied
print(f"Model type: {type(model)}")
print(f"Unsloth patches applied: True")
model.print_trainable_parameters()`}
        id="code-quick-start"
      />

      <NoteBlock
        type="note"
        title="Supported Models"
        content="Unsloth supports LLaMA 1/2/3/3.1/3.2/3.3, Mistral, Mixtral, Phi-3/4, Gemma 1/2, Qwen 2/2.5, DeepSeek, and many more architectures. New models are typically supported within days of release. Check the Unsloth GitHub for the current list."
        id="note-supported-models"
      />

      <NoteBlock
        type="tip"
        title="Free Colab Notebooks"
        content="Unsloth maintains ready-to-run Google Colab notebooks for every supported model. These are the fastest way to start finetuning: search 'unsloth [model-name] colab' to find the appropriate notebook. The free T4 GPU tier works for 7-8B models with QLoRA."
        id="note-colab"
      />

      <WarningBlock
        title="Single-GPU Only"
        content="Unsloth currently only supports single-GPU training. For multi-GPU setups, use standard Hugging Face with FSDP or DeepSpeed, or use Axolotl/LLaMA-Factory which handle distributed training. Unsloth's efficiency often makes single-GPU sufficient for models up to 70B with QLoRA."
        id="warning-single-gpu"
      />
    </div>
  )
}
