import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function FullVsPeft() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Full Finetuning vs Parameter-Efficient Finetuning</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        When adapting a pretrained model to a new task, you can update all parameters (full
        finetuning) or only a small subset (PEFT). Each approach has distinct tradeoffs in memory,
        compute, data requirements, and model quality.
      </p>

      <DefinitionBlock
        title="Full Finetuning"
        definition="Full finetuning updates all $P$ parameters of a pretrained model using gradient descent on the target dataset. This provides maximum expressiveness but requires storing the full model weights, optimizer states ($8P$ bytes for AdamW in fp32), and gradients ($2P$ bytes in fp16)."
        id="def-full-ft"
      />

      <DefinitionBlock
        title="Parameter-Efficient Finetuning (PEFT)"
        definition="PEFT methods freeze most pretrained weights and only train a small number of additional or selected parameters. If the trainable parameter count is $P_{\\text{train}} \\ll P$, memory for optimizer states drops from $8P$ to $8P_{\\text{train}}$ bytes."
        id="def-peft"
      />

      <h2 className="text-2xl font-semibold">Comparison Table</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The key tradeoffs between full finetuning and PEFT methods like LoRA:
      </p>

      <ExampleBlock
        title="Memory Comparison: 7B Model"
        problem="Compare VRAM requirements for full finetuning vs LoRA vs QLoRA on a 7B parameter model."
        steps={[
          { formula: '\\text{Full FP16: } 2(7B) + 8(7B) + 2(7B) \\approx 84\\text{ GB}', explanation: 'Weights (fp16) + AdamW states (fp32) + gradients (fp16).' },
          { formula: '\\text{LoRA FP16: } 2(7B) + 8(0.07B) + 2(0.07B) \\approx 14.7\\text{ GB}', explanation: 'Full weights frozen in fp16, only ~1% params trained.' },
          { formula: '\\text{QLoRA: } 0.5(7B) + 8(0.07B) + 2(0.07B) \\approx 4.2\\text{ GB}', explanation: 'Base weights in 4-bit (~0.5 bytes/param), LoRA in fp16.' },
        ]}
        id="example-memory-comparison"
      />

      <PythonCode
        title="compare_finetuning_methods.py"
        code={`from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# --- Full Finetuning Setup ---
# model_full = AutoModelForCausalLM.from_pretrained(
#     model_name, torch_dtype=torch.float16, device_map="auto"
# )
# All params trainable
# total = sum(p.numel() for p in model_full.parameters())
# trainable = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
# print(f"Full FT: {trainable/1e6:.0f}M / {total/1e6:.0f}M params trainable")

# --- LoRA Setup ---
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"
)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model_lora = get_peft_model(model, lora_config)
model_lora.print_trainable_parameters()
# trainable params: 13,631,488 || all params: 8,030,261,248
# trainable%: 0.1698

# Key insight: QLoRA trains the same LoRA params but base is 4-bit
mem = model.get_memory_footprint() / 1e9
print(f"Base model memory (4-bit): {mem:.1f} GB")`}
        id="code-compare-methods"
      />

      <NoteBlock
        type="intuition"
        title="Why Does PEFT Work So Well?"
        content="Pretrained LLMs already encode vast knowledge in their weights. Finetuning is really about steering the model toward specific behaviors, not teaching it new knowledge. A low-rank update (LoRA rank 8-64) is sufficient to capture this behavioral shift, because the 'direction' of adaptation lies in a low-dimensional subspace."
        id="note-peft-intuition"
      />

      <WarningBlock
        title="When Full Finetuning Is Better"
        content="PEFT may underperform full finetuning when: (1) the target domain is very different from pretraining data, (2) you have a large high-quality dataset (100K+ examples), (3) you need maximum performance and have the compute budget, or (4) you are doing continued pretraining on a new language or domain."
        id="warning-when-full"
      />

      <NoteBlock
        type="tip"
        title="Practical Recommendation"
        content="Start with QLoRA (rank 16-64) for rapid experimentation. If quality is insufficient after hyperparameter tuning, try LoRA in fp16/bf16. Only resort to full finetuning if PEFT clearly underperforms and you have sufficient compute."
        id="note-recommendation"
      />
    </div>
  )
}
