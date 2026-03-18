import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function QloraDeepDive() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">QLoRA: 4-Bit Quantization + LoRA</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        QLoRA combines 4-bit quantization of the base model with LoRA adapters trained in higher
        precision. This breakthrough technique enables finetuning 65B+ parameter models on a single
        48 GB GPU, democratizing LLM adaptation for researchers with limited hardware.
      </p>

      <DefinitionBlock
        title="QLoRA"
        definition="QLoRA (Quantized Low-Rank Adaptation) stores pretrained weights in 4-bit NormalFloat (NF4) format while training LoRA adapters in BFloat16/Float16. During the forward pass, 4-bit weights are dequantized to compute dtype on-the-fly. Gradients only flow through the LoRA parameters, never updating the quantized base."
        id="def-qlora"
      />

      <h2 className="text-2xl font-semibold">Key Innovations in QLoRA</h2>

      <DefinitionBlock
        title="NF4 (NormalFloat4)"
        definition="NF4 is a 4-bit data type optimized for normally distributed weights. It uses an information-theoretically optimal quantization grid for $\\mathcal{N}(0, \\sigma^2)$ data, providing better accuracy than standard INT4 or FP4 at the same bit-width."
        id="def-nf4"
      />

      <ExampleBlock
        title="Double Quantization"
        problem="How does double quantization further reduce memory in QLoRA?"
        steps={[
          { formula: '\\text{Block-wise quant: 64 weights share 1 FP32 scale}', explanation: 'Standard quantization uses one 32-bit scale factor per block of 64 weights.' },
          { formula: '\\text{Scale overhead: } \\frac{32}{64} = 0.5 \\text{ bits/param}', explanation: 'The scale factors add 0.5 bits per parameter overhead.' },
          { formula: '\\text{Double quant: quantize the scales to FP8}', explanation: 'QLoRA quantizes the scale factors themselves, reducing overhead.' },
          { formula: '\\text{New overhead: } \\frac{8}{64} + \\frac{32}{64^2} \\approx 0.13 \\text{ bits/param}', explanation: 'Double quantization reduces scale overhead from 0.5 to 0.13 bits per parameter.' },
        ]}
        id="example-double-quant"
      />

      <PythonCode
        title="qlora_complete_setup.py"
        code={`import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# Step 1: Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Enable 4-bit loading
    bnb_4bit_quant_type="nf4",      # NormalFloat4 data type
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16
    bnb_4bit_use_double_quant=True, # Double quantization
)

# Step 2: Load model in 4-bit
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Step 3: Prepare model for QLoRA training
model = prepare_model_for_kbit_training(model)

# Step 4: Add LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",  # Apply to all linear layers
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Memory footprint
mem_gb = model.get_memory_footprint() / 1e9
print(f"Total memory: {mem_gb:.1f} GB (base 4-bit + LoRA fp16)")`}
        id="code-qlora-setup"
      />

      <h2 className="text-2xl font-semibold">Paged Optimizers</h2>
      <p className="text-gray-700 dark:text-gray-300">
        QLoRA introduces paged optimizers that use NVIDIA unified memory to automatically page
        optimizer states between CPU and GPU memory. This prevents out-of-memory crashes during
        gradient spikes without manual intervention.
      </p>

      <PythonCode
        title="paged_optimizer.py"
        code={`from transformers import TrainingArguments

# Training arguments optimized for QLoRA
training_args = TrainingArguments(
    output_dir="./qlora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # Effective batch size = 16
    learning_rate=2e-4,              # Higher LR works well with QLoRA
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,                       # Use bf16 compute
    # Paged optimizer to prevent OOM
    optim="paged_adamw_8bit",        # 8-bit paged AdamW
    gradient_checkpointing=True,     # Trade compute for memory
    max_grad_norm=0.3,               # Gradient clipping
    group_by_length=True,            # Group similar-length sequences
)

print(f"Optimizer: {training_args.optim}")
print(f"Gradient checkpointing: {training_args.gradient_checkpointing}")`}
        id="code-paged-optimizer"
      />

      <NoteBlock
        type="intuition"
        title="Why QLoRA Barely Hurts Quality"
        content="The 4-bit quantized weights are frozen -- they only participate in the forward pass. All gradient computation and weight updates happen in the LoRA adapters at full precision. The quantization error in the base weights is a fixed 'noise floor' that the LoRA adapters learn to compensate for."
        id="note-qlora-quality"
      />

      <WarningBlock
        title="QLoRA Training Speed"
        content="QLoRA is ~30-40% slower than full-precision LoRA due to dequantization overhead during the forward pass. If you have enough VRAM for LoRA in fp16/bf16, it will train faster and produce slightly better results. QLoRA shines when GPU memory is the bottleneck."
        id="warning-qlora-speed"
      />
    </div>
  )
}
