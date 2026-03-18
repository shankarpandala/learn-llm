import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function MixtralArchitecture() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Mixtral: Sparse MoE in Practice</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Mixtral 8x7B (December 2023) from Mistral AI was the first widely-deployed open-source
        MoE language model. With 46.7B total parameters but only 12.9B active per token, it
        matched or exceeded LLaMA 2 70B on most benchmarks while being 6x faster at inference.
        It demonstrated that MoE was practical for real-world deployment, not just research.
      </p>

      <DefinitionBlock
        title="Mixtral Architecture"
        definition="Mixtral replaces every feed-forward layer in a Mistral 7B-style transformer with a Sparse MoE layer containing 8 experts and top-2 routing. The attention layers remain shared (not replicated). Each expert is a standard SwiGLU FFN identical to Mistral 7B's FFN. Total params: 8 experts * 5.6B FFN params + 7.1B shared = 46.7B. Active per token: 2 experts * 5.6B + 7.1B = 12.9B."
        id="def-mixtral"
      />

      <h2 className="text-2xl font-semibold">Architecture Breakdown</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Mixtral uses 32 transformer layers, each with a shared GQA attention block (32 query heads,
        8 KV heads) and an MoE feed-forward block. The hidden dimension is 4096, each expert
        has an intermediate size of 14336 (SwiGLU), and top-2 routing is used with a softmax
        gate. Sliding window attention from Mistral 7B is retained.
      </p>

      <ExampleBlock
        title="Mixtral Parameter Budget"
        problem="Break down where the 46.7B parameters in Mixtral 8x7B are allocated."
        steps={[
          { formula: '\\text{Attention (shared)}: 32 \\times (4 \\times d^2 / \\text{GQA}) \\approx 2.1\\text{B}', explanation: 'Q, K, V, O projections with GQA (32 query, 8 KV heads). Shared across all tokens.' },
          { formula: '\\text{Per expert FFN}: 3 \\times d \\times d_{ff} = 3 \\times 4096 \\times 14336 \\approx 176\\text{M}', explanation: 'Three weight matrices for SwiGLU (w1, w2, w3). Per layer, per expert.' },
          { formula: '\\text{Total FFN}: 32 \\times 8 \\times 176\\text{M} \\approx 45.1\\text{B}', explanation: '32 layers * 8 experts * 176M params. This dominates the parameter count.' },
          { formula: '\\text{Active FFN}: 32 \\times 2 \\times 176\\text{M} \\approx 11.3\\text{B}', explanation: 'Only 2 experts active per token, so effective compute is ~12.9B total.' },
        ]}
        id="example-mixtral-params"
      />

      <h2 className="text-2xl font-semibold">Expert Specialization</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Analysis of Mixtral's routing patterns reveals that experts do not specialize by topic
        (e.g., "science expert" or "code expert"). Instead, specialization is more syntactic:
        experts tend to handle specific token types, positions in sentences, or linguistic patterns.
        The routing is not deterministic and varies with context.
      </p>

      <PythonCode
        title="mixtral_usage.py"
        code={`from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Mixtral 8x7B (needs ~90GB+ for float16, use quantized for less)
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,  # Quantize to fit in ~24GB
)

# Inspect MoE configuration
config = model.config
print(f"Hidden size: {config.hidden_size}")              # 4096
print(f"Num layers: {config.num_hidden_layers}")         # 32
print(f"Num experts: {config.num_local_experts}")        # 8
print(f"Top-K: {config.num_experts_per_tok}")            # 2
print(f"Intermediate size: {config.intermediate_size}")  # 14336
print(f"Sliding window: {config.sliding_window}")        # 4096

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"\\nTotal parameters: {total / 1e9:.1f}B")

# Estimate active parameters per token
attn_params = sum(
    p.numel() for name, p in model.named_parameters()
    if "self_attn" in name
)
expert_params_each = sum(
    p.numel() for name, p in model.named_parameters()
    if "experts.0" in name
)
active_params = attn_params + expert_params_each * 2  # top-2 routing
print(f"Active params per token: ~{active_params / 1e9:.1f}B")

# Generate
messages = [{"role": "user", "content": "Compare MoE and dense transformers in 3 points."}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
print(tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))`}
        id="code-mixtral"
      />

      <NoteBlock
        type="note"
        title="Mixtral 8x22B"
        content="Mistral followed up with Mixtral 8x22B (April 2024), scaling to 176B total parameters with 39B active. It uses 8 experts with top-2 routing, a 64K vocabulary, and 64K context length. It significantly outperforms Mixtral 8x7B and competes with LLaMA 3 70B while being faster at inference."
        id="note-mixtral-8x22b"
      />

      <WarningBlock
        title="Memory Requirements for MoE"
        content="Despite using only 12.9B active parameters, Mixtral 8x7B requires loading all 46.7B parameters into memory (~93GB in float16). This means it needs more GPUs than a dense 13B model despite similar inference speed. Quantization (4-bit GPTQ/AWQ) reduces this to ~24GB, making single-GPU inference feasible."
        id="warning-mixtral-memory"
      />

      <NoteBlock
        type="intuition"
        title="MoE as Model Compression"
        content="Think of MoE as a form of conditional computation: the model has the knowledge capacity of a 47B model but the inference cost of a 13B model. It's like having a 47B model that was 'compressed' at inference time by only activating the relevant parts. This is fundamentally different from weight pruning or quantization."
        id="note-moe-compression"
      />
    </div>
  )
}
