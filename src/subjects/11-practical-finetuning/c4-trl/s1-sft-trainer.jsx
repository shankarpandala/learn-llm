import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function SftTrainer() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">SFTTrainer: Supervised Finetuning with TRL</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        TRL (Transformer Reinforcement Learning) provides SFTTrainer, the standard tool for
        supervised finetuning of language models. It extends the Hugging Face Trainer with
        features like sequence packing, chat template handling, and completion-only training.
      </p>

      <DefinitionBlock
        title="Supervised Finetuning (SFT)"
        definition="SFT trains a language model on input-output pairs using the standard next-token prediction objective. Given a prompt $x$ and desired response $y$, the loss is $L = -\\sum_{t=1}^{|y|} \\log P_\\theta(y_t | x, y_{<t})$. This is the most common finetuning approach."
        id="def-sft"
      />

      <h2 className="text-2xl font-semibold">Basic SFTTrainer Usage</h2>

      <PythonCode
        title="sft_basic.py"
        code={`from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset in messages format
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:5000]")

# SFTConfig replaces TrainingArguments in newer TRL
sft_config = SFTConfig(
    output_dir="./sft-output",
    max_seq_length=2048,
    packing=True,                    # Pack multiple examples per sequence
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    optim="adamw_torch",
)

# Create trainer - handles chat template automatically
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()`}
        id="code-sft-basic"
      />

      <h2 className="text-2xl font-semibold">SFTTrainer with QLoRA</h2>

      <PythonCode
        title="sft_qlora.py"
        code={`from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# SFTTrainer handles PEFT integration automatically
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./sft-qlora",
        max_seq_length=2048,
        packing=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,       # Higher LR for LoRA
        bf16=True,
        logging_steps=10,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
    ),
    peft_config=peft_config,      # Pass PEFT config directly!
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.model.print_trainable_parameters()
trainer.train()`}
        id="code-sft-qlora"
      />

      <ExampleBlock
        title="SFTTrainer Key Parameters"
        problem="What are the most important SFTTrainer configuration options?"
        steps={[
          { formula: '\\texttt{max\\_seq\\_length}', explanation: 'Maximum token length. Longer = more VRAM. Match to your data distribution.' },
          { formula: '\\texttt{packing=True}', explanation: 'Concatenates short examples into full sequences. 2-5x throughput improvement.' },
          { formula: '\\texttt{dataset\\_text\\_field}', explanation: 'Column name for pre-formatted text. Or use "messages" column for auto-formatting.' },
          { formula: '\\texttt{peft\\_config}', explanation: 'Pass LoRA config directly -- SFTTrainer applies it and handles prepare_model_for_kbit_training.' },
        ]}
        id="example-sft-params"
      />

      <NoteBlock
        type="tip"
        title="Using the Messages Format"
        content="When your dataset has a 'messages' column (list of role/content dicts), SFTTrainer automatically applies the tokenizer's chat template. This is the cleanest approach -- no manual template formatting needed. Just ensure your tokenizer has a chat_template attribute."
        id="note-messages-format"
      />

      <WarningBlock
        title="Packing and Completion-Only Are Incompatible"
        content="You cannot use packing=True with DataCollatorForCompletionOnlyLM. Packing concatenates examples, destroying the boundary between prompt and response. If you need completion-only training (masking prompt tokens), set packing=False."
        id="warning-packing-completion"
      />
    </div>
  )
}
