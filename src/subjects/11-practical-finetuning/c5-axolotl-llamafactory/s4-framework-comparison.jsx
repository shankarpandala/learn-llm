import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function FrameworkComparison() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Finetuning Framework Comparison</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Multiple frameworks exist for finetuning LLMs, each with different strengths. This section
        compares Unsloth, TRL, Axolotl, and LLaMA-Factory across key dimensions to help you
        choose the right tool for your needs.
      </p>

      <h2 className="text-2xl font-semibold">Feature Matrix</h2>

      <PythonCode
        title="framework_comparison.py"
        code={`# Framework comparison matrix
frameworks = {
    "Unsloth": {
        "ease_of_use": "High (Python API)",
        "multi_gpu": "No (single GPU only)",
        "speed": "2x faster (custom Triton kernels)",
        "memory": "60% less VRAM",
        "methods": "SFT, DPO (via TRL)",
        "model_support": "50+ architectures",
        "gui": "No",
        "best_for": "Single-GPU speed, memory-constrained setups",
    },
    "TRL (HuggingFace)": {
        "ease_of_use": "Medium (Python API)",
        "multi_gpu": "Yes (via accelerate/FSDP/DeepSpeed)",
        "speed": "Baseline",
        "memory": "Baseline",
        "methods": "SFT, DPO, PPO, ORPO, KTO, reward modeling",
        "model_support": "All HF models",
        "gui": "No",
        "best_for": "Custom training loops, RLHF, research",
    },
    "Axolotl": {
        "ease_of_use": "Medium (YAML config)",
        "multi_gpu": "Yes (DeepSpeed, FSDP)",
        "speed": "Near-baseline (Flash Attention)",
        "memory": "Good (packing, checkpointing)",
        "methods": "SFT, DPO, RLHF",
        "model_support": "Most popular models",
        "gui": "No",
        "best_for": "Production pipelines, multi-dataset mixing",
    },
    "LLaMA-Factory": {
        "ease_of_use": "Very High (Web GUI + CLI)",
        "multi_gpu": "Yes (auto-detected)",
        "speed": "Near-baseline",
        "memory": "Good",
        "methods": "SFT, DPO, PPO, ORPO, KTO",
        "model_support": "100+ architectures",
        "gui": "Yes (LLaMA Board)",
        "best_for": "Beginners, quick experimentation, no-code",
    },
}

for name, info in frameworks.items():
    print(f"\\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for key, value in info.items():
        print(f"  {key:>15}: {value}")`}
        id="code-comparison-matrix"
      />

      <ExampleBlock
        title="Decision Tree: Which Framework?"
        problem="How to choose the right finetuning framework for your situation?"
        steps={[
          { formula: '\\text{Single GPU + max speed?} \\Rightarrow \\text{Unsloth}', explanation: 'Unsloth gives the best single-GPU performance with custom kernels.' },
          { formula: '\\text{Multi-GPU needed?} \\Rightarrow \\text{Axolotl or TRL}', explanation: 'Unsloth is single-GPU only. Use Axolotl for YAML-driven multi-GPU or TRL for code-first.' },
          { formula: '\\text{No coding preferred?} \\Rightarrow \\text{LLaMA-Factory}', explanation: 'The web GUI allows complete finetuning without writing any code.' },
          { formula: '\\text{Custom RLHF/DPO/PPO?} \\Rightarrow \\text{TRL}', explanation: 'TRL provides the most flexibility for alignment research and custom reward functions.' },
          { formula: '\\text{Production pipeline?} \\Rightarrow \\text{Axolotl}', explanation: 'YAML configs are easy to version control and integrate into CI/CD.' },
        ]}
        id="example-decision-tree"
      />

      <PythonCode
        title="combine_frameworks.py"
        code={`# You can combine frameworks! Common patterns:

# Pattern 1: Unsloth for SFT + TRL for DPO
# - Fast SFT training with Unsloth's optimizations
# - Then load the SFT model in TRL for DPO alignment

# Pattern 2: LLaMA-Factory for prototyping + Axolotl for production
# - Quick experiments in the GUI
# - Convert winning config to Axolotl YAML for reproducible runs

# Pattern 3: Unsloth model loading + custom training loop
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16,
    target_modules="all-linear",
    use_gradient_checkpointing="unsloth",
)

# Use model with any trainer/loop - you still get Unsloth's kernel optimizations
# trainer = YourCustomTrainer(model=model, ...)

# Pattern 4: Start simple, scale up
# Start: Unsloth on 1 GPU (fastest iteration)
# Scale: Axolotl + DeepSpeed on 4 GPUs (same config, more compute)
# Deploy: Export merged model or GGUF from either`}
        id="code-combine"
      />

      <NoteBlock
        type="tip"
        title="Start Simple"
        content="For most finetuning projects, start with Unsloth on a single GPU. It is the fastest to set up, the cheapest to run, and produces results identical to other frameworks. Only move to multi-GPU frameworks when your model/data requires it."
        id="note-start-simple"
      />

      <WarningBlock
        title="Framework Lock-In"
        content="All these frameworks produce standard Hugging Face models and LoRA adapters. You are NOT locked into any framework. A model trained with Unsloth can be loaded by TRL, Axolotl, or any HF-compatible tool. Choose based on convenience, not fear of lock-in."
        id="warning-lockin"
      />
    </div>
  )
}
