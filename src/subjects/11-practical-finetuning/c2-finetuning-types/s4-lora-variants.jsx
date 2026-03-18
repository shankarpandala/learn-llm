import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LoraVariants() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LoRA Variants: DoRA, rsLoRA, and LoRA+</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Since the original LoRA paper, several improvements have been proposed to address its
        limitations. DoRA decomposes weight updates into magnitude and direction, rsLoRA fixes
        scaling for high ranks, and LoRA+ uses different learning rates for A and B matrices.
      </p>

      <DefinitionBlock
        title="DoRA (Weight-Decomposed Low-Rank Adaptation)"
        definition="DoRA decomposes the pretrained weight into magnitude $m$ and direction $V$: $W = m \\frac{V + BA}{\\|V + BA\\|_c}$. By separating magnitude and direction updates, DoRA more closely matches the learning dynamics of full finetuning while maintaining LoRA's parameter efficiency."
        id="def-dora"
      />

      <DefinitionBlock
        title="rsLoRA (Rank-Stabilized LoRA)"
        definition="Standard LoRA uses scaling factor $\\frac{\\alpha}{r}$, which decreases with rank. rsLoRA replaces this with $\\frac{\\alpha}{\\sqrt{r}}$, stabilizing the per-parameter update magnitude as rank increases. This allows effective use of higher ranks without retuning the learning rate."
        id="def-rslora"
      />

      <DefinitionBlock
        title="LoRA+"
        definition="LoRA+ assigns different learning rates to matrices $A$ and $B$. Specifically, it sets $\\eta_B = \\lambda \\cdot \\eta_A$ where $\\lambda \\approx 16$. This corrects for the asymmetric initialization (A is random, B is zero) and improves convergence speed by ~2x."
        id="def-lora-plus"
      />

      <ExampleBlock
        title="Comparing LoRA Variants"
        problem="When should you use each LoRA variant?"
        steps={[
          { formula: '\\text{Standard LoRA: } r \\leq 32', explanation: 'Works well at low ranks. Use for quick experiments and when memory is tight.' },
          { formula: '\\text{DoRA: } \\text{quality-critical tasks}', explanation: 'Slightly better accuracy than LoRA, especially on reasoning and math tasks. ~10% more trainable params.' },
          { formula: '\\text{rsLoRA: } r \\geq 64', explanation: 'Use when you need high rank without retuning learning rate. Drop-in replacement for standard LoRA.' },
          { formula: '\\text{LoRA+: } \\text{faster convergence}', explanation: 'Reaches same quality as LoRA in fewer steps. Good when compute time matters.' },
        ]}
        id="example-variant-comparison"
      />

      <PythonCode
        title="lora_variants_peft.py"
        code={`from peft import LoraConfig, get_peft_model

# --- Standard LoRA ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# --- DoRA (available in PEFT >= 0.10) ---
dora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    use_dora=True,  # Enable weight decomposition
)

# --- rsLoRA (available in PEFT >= 0.9) ---
rslora_config = LoraConfig(
    r=64,           # Works better at higher ranks
    lora_alpha=128,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    use_rslora=True,  # Use rank-stabilized scaling
)

# --- LoRA+ (use with custom optimizer) ---
# LoRA+ requires setting different learning rates for A and B matrices
# This is done through the optimizer, not the LoRA config

# With Unsloth (easiest way to use LoRA+):
# model, tokenizer = FastLanguageModel.from_pretrained(...)
# model = FastLanguageModel.get_peft_model(
#     model,
#     r=16,
#     lora_alpha=32,
#     use_rslora=True,
#     loraplus_lr_ratio=16.0,  # eta_B = 16 * eta_A
# )

# Compare parameter counts
for name, config in [("LoRA", lora_config), ("DoRA", dora_config),
                      ("rsLoRA", rslora_config)]:
    print(f"{name}: r={config.r}, alpha={config.lora_alpha}, "
          f"dora={config.use_dora}, rslora={config.use_rslora}")`}
        id="code-lora-variants"
      />

      <PythonCode
        title="loraplus_manual.py"
        code={`# Manual LoRA+ implementation with different learning rates
import torch
from torch.optim import AdamW

def create_loraplus_optimizer(model, lr=1e-4, loraplus_ratio=16.0):
    """Create optimizer with different LRs for LoRA A and B matrices."""
    param_groups = []
    lora_A_params = []
    lora_B_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            lora_A_params.append(param)
        elif "lora_B" in name:
            lora_B_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": lora_A_params, "lr": lr},
        {"params": lora_B_params, "lr": lr * loraplus_ratio},
        {"params": other_params, "lr": lr},
    ]

    print(f"LoRA A params: {len(lora_A_params)} groups, lr={lr}")
    print(f"LoRA B params: {len(lora_B_params)} groups, lr={lr*loraplus_ratio}")

    return AdamW(param_groups, weight_decay=0.01)

# optimizer = create_loraplus_optimizer(model, lr=1e-4, loraplus_ratio=16)`}
        id="code-loraplus"
      />

      <NoteBlock
        type="note"
        title="DoRA Memory Overhead"
        content="DoRA adds a learnable magnitude vector per adapted layer, increasing trainable parameters by roughly 10%. Memory overhead is minimal. In benchmarks, DoRA consistently outperforms LoRA by 1-3% on commonsense reasoning tasks."
        id="note-dora-overhead"
      />

      <WarningBlock
        title="Variant Compatibility"
        content="Not all variants are supported by all frameworks. Check your PEFT version: DoRA requires >= 0.10, rsLoRA requires >= 0.9. Unsloth supports all variants natively. Some variants may interact: you can combine rsLoRA + DoRA but should test carefully."
        id="warning-compatibility"
      />
    </div>
  )
}
