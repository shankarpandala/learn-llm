import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function AdapterComparison() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Adapter Methods Compared</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Beyond LoRA, several other parameter-efficient methods exist for adapting pretrained models.
        Each makes different tradeoffs between parameter count, inference overhead, and adaptation
        quality. This section compares the most practical approaches.
      </p>

      <DefinitionBlock
        title="Adapter Layers"
        definition="Adapter layers are small bottleneck modules inserted between existing transformer layers. They consist of a down-projection $W_{\\text{down}} \\in \\mathbb{R}^{d \\times r}$, a nonlinearity, and an up-projection $W_{\\text{up}} \\in \\mathbb{R}^{r \\times d}$, with a residual connection: $h = h + f(h W_{\\text{down}}) W_{\\text{up}}$."
        id="def-adapter-layers"
      />

      <DefinitionBlock
        title="Prefix Tuning"
        definition="Prefix tuning prepends learnable virtual tokens to the key and value matrices at each attention layer. For prefix length $l$ and model dimension $d$, it adds $2ld$ trainable parameters per layer (for keys and values)."
        id="def-prefix-tuning"
      />

      <ExampleBlock
        title="Method Comparison for 7B Model"
        problem="Compare parameter counts and inference overhead for different PEFT methods on a 7B model."
        steps={[
          { formula: '\\text{LoRA (r=16)}: \\approx 42\\text{M params, 0\\% inference overhead}', explanation: 'LoRA weights can be merged into base weights for zero-cost inference.' },
          { formula: '\\text{Adapters (r=64)}: \\approx 50\\text{M params, 5-10\\% overhead}', explanation: 'Adapter layers add serial computation that cannot be removed at inference.' },
          { formula: '\\text{Prefix (l=20)}: \\approx 10\\text{M params, 2-5\\% overhead}', explanation: 'Virtual tokens increase effective sequence length slightly.' },
          { formula: '\\text{IA3}: \\approx 0.5\\text{M params, 0\\% overhead}', explanation: 'Learned rescaling vectors; extremely parameter-efficient but limited capacity.' },
        ]}
        id="example-method-comparison"
      />

      <PythonCode
        title="peft_methods_comparison.py"
        code={`from peft import (
    LoraConfig,
    PrefixTuningConfig,
    IA3Config,
    AdaLoraConfig,
    get_peft_model,
    TaskType,
)

# --- LoRA: Most popular, zero inference overhead ---
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules="all-linear",
    task_type=TaskType.CAUSAL_LM,
)

# --- Prefix Tuning: Prepends virtual tokens ---
prefix_config = PrefixTuningConfig(
    num_virtual_tokens=20,
    task_type=TaskType.CAUSAL_LM,
)

# --- IA3: Minimal parameters ---
ia3_config = IA3Config(
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],
    task_type=TaskType.CAUSAL_LM,
)

# --- AdaLoRA: Adaptive rank allocation ---
adalora_config = AdaLoraConfig(
    init_r=12,           # Initial rank
    target_r=4,          # Target average rank after pruning
    beta1=0.85,
    beta2=0.85,
    tinit=200,           # Steps before pruning starts
    tfinal=1000,         # Steps when pruning ends
    deltaT=10,           # Pruning interval
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)

# Print configs
configs = {
    "LoRA": lora_config,
    "Prefix": prefix_config,
    "IA3": ia3_config,
    "AdaLoRA": adalora_config,
}
for name, cfg in configs.items():
    print(f"\\n{name}:")
    print(f"  Type: {type(cfg).__name__}")
    if hasattr(cfg, 'r'):
        print(f"  Rank: {cfg.r}")`}
        id="code-peft-methods"
      />

      <h2 className="text-2xl font-semibold">Practical Recommendation Flow</h2>
      <PythonCode
        title="choose_method.py"
        code={`def recommend_peft_method(
    gpu_vram_gb: float,
    model_params_b: float,
    dataset_size: int,
    need_multi_adapter: bool = False,
    inference_latency_critical: bool = True,
):
    """Recommend a PEFT method based on constraints."""

    # Estimate if QLoRA fits
    qlora_mem = 0.5 * model_params_b + 1.5  # base 4-bit + overhead
    lora_mem = 2 * model_params_b + 3       # base fp16 + overhead

    recommendations = []

    if gpu_vram_gb >= lora_mem:
        method = "LoRA (fp16/bf16)"
        reason = "Enough VRAM for full-precision LoRA - faster training"
    elif gpu_vram_gb >= qlora_mem:
        method = "QLoRA (4-bit)"
        reason = "4-bit quantization fits within VRAM"
    else:
        method = "QLoRA with gradient checkpointing"
        reason = "Tight on memory - enable all memory optimizations"

    rank = 16 if dataset_size < 10000 else 32 if dataset_size < 50000 else 64

    print(f"Recommended: {method}")
    print(f"Reason: {reason}")
    print(f"Suggested rank: {rank}")
    print(f"Estimated VRAM: {qlora_mem:.0f}-{lora_mem:.0f} GB")

    if need_multi_adapter:
        print("Tip: LoRA adapters can be hot-swapped at inference time")
    if inference_latency_critical:
        print("Tip: Merge LoRA weights after training for zero overhead")

# Example usage
recommend_peft_method(
    gpu_vram_gb=24, model_params_b=8, dataset_size=5000
)`}
        id="code-recommend"
      />

      <NoteBlock
        type="tip"
        title="LoRA Wins in Practice"
        content="Despite the variety of PEFT methods, LoRA (and its variants QLoRA, DoRA) dominates in practice due to: (1) zero inference overhead after merging, (2) simple hyperparameter tuning, (3) compatibility with all model architectures, and (4) excellent framework support. Start with LoRA unless you have a specific reason to use another method."
        id="note-lora-wins"
      />

      <WarningBlock
        title="Adapter Serving Complexity"
        content="Methods that cannot be merged (prefix tuning, adapter layers) add inference overhead and complicate serving. If you need to serve multiple adapters, LoRA is ideal: adapters are small files that can be loaded/swapped dynamically, and libraries like LoRAX and S-LoRA enable efficient multi-adapter serving."
        id="warning-serving"
      />
    </div>
  )
}
