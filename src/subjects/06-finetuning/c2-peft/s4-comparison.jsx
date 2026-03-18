import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function PEFTComparison() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Comparing PEFT Methods</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Parameter-Efficient Fine-Tuning (PEFT) encompasses a family of methods that adapt
        large models by training only a small subset of parameters. Each method makes different
        tradeoffs between parameter count, inference overhead, implementation complexity, and
        task performance. Understanding these tradeoffs is essential for choosing the right
        approach.
      </p>

      <DefinitionBlock
        title="Parameter-Efficient Fine-Tuning (PEFT)"
        definition="PEFT methods adapt a pretrained model by modifying or adding a small number of parameters $\Delta\theta$ while keeping the vast majority of original parameters $\theta_0$ frozen. The effective model is $\theta = \theta_0 + \Delta\theta$ where $|\Delta\theta| \ll |\theta_0|$. Major families include additive (adapters, prefix tuning), reparameterization (LoRA), and selective (BitFit, sparse finetuning)."
        id="def-peft"
      />

      <h2 className="text-2xl font-semibold">Method Comparison</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The table below summarizes key properties of popular PEFT methods. The right choice
        depends on your constraints: if inference latency matters, choose LoRA (mergeable).
        If you need to switch tasks rapidly, adapters or LoRA adapters can be hot-swapped.
        If you want minimal parameters, prompt tuning uses the fewest.
      </p>

      <ExampleBlock
        title="PEFT Methods at a Glance"
        problem="Compare LoRA, Adapters, Prefix Tuning, and Prompt Tuning on a 7B parameter model."
        steps={[
          { formula: '\\text{LoRA (r=16): } \\sim 0.2\\% \\text{ params, 0 inference overhead (merged)}', explanation: 'LoRA matrices can be absorbed into the base weights, making inference identical to the original model.' },
          { formula: '\\text{Adapters (m=64): } \\sim 0.5\\% \\text{ params, 5-10\\% latency overhead}', explanation: 'Adapter layers add sequential computation that cannot be removed at inference time.' },
          { formula: '\\text{Prefix Tuning (l=20): } \\sim 0.1\\% \\text{ params, minor KV cache overhead}', explanation: 'Prefix vectors consume context window positions but add minimal compute.' },
          { formula: '\\text{Prompt Tuning (l=20): } \\sim 0.002\\% \\text{ params, minimal overhead}', explanation: 'Fewest parameters but requires very large models (10B+) to match other methods.' },
        ]}
        id="example-comparison"
      />

      <PythonCode
        title="peft_methods_benchmark.py"
        code={`from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig, PrefixTuningConfig, PromptTuningConfig,
    get_peft_model, TaskType
)

model_name = "meta-llama/Llama-2-7b-hf"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

# Count base model params
total_params = sum(p.numel() for p in base_model.parameters())
print(f"Base model: {total_params:,} parameters")

# Method 1: LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# Method 2: Prefix Tuning
prefix_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20,
)

# Method 3: Prompt Tuning
prompt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20,
)

# Compare trainable parameters
configs = {
    "LoRA (r=16)": lora_config,
    "Prefix Tuning (l=20)": prefix_config,
    "Prompt Tuning (l=20)": prompt_config,
}

for name, config in configs.items():
    model = get_peft_model(base_model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / total_params
    print(f"{name:30s}: {trainable:>12,} trainable ({pct:.4f}%)")
    del model  # Free memory

# Typical results for Llama-2-7B:
# LoRA (r=16)                  :   13,631,488 trainable (0.2019%)
# Prefix Tuning (l=20)         :    3,276,800 trainable (0.0485%)
# Prompt Tuning (l=20)         :       81,920 trainable (0.0012%)`}
        id="code-peft-benchmark"
      />

      <NoteBlock
        type="tip"
        title="QLoRA: Quantized LoRA"
        content="QLoRA (Dettmers et al., 2023) combines 4-bit quantization of the base model with LoRA adapters trained in full precision. This enables finetuning a 65B parameter model on a single 48GB GPU. The base model is loaded in 4-bit NormalFloat format while LoRA gradients flow in BFloat16, giving near-lossless quality at a fraction of the memory cost."
        id="note-qlora"
      />

      <WarningBlock
        title="No Free Lunch"
        content="PEFT methods trade capacity for efficiency. On complex tasks requiring significant behavioral changes (e.g., teaching a model a new language), full finetuning may still outperform PEFT methods. Always benchmark on your specific task. Additionally, PEFT methods can be combined: LoRA on attention layers plus adapters on FFN layers sometimes outperforms either alone."
        id="warning-no-free-lunch"
      />

      <NoteBlock
        type="note"
        title="Practical Recommendation"
        content="For most practitioners in 2024-2025, LoRA (or QLoRA) is the default recommendation. It offers the best balance of simplicity, performance, and flexibility. Use rank r = 16-64, apply to all linear layers, and set alpha = 2r. If memory is extremely constrained, QLoRA with 4-bit quantization lets you finetune models much larger than your GPU would normally support."
        id="note-recommendation"
      />
    </div>
  )
}
