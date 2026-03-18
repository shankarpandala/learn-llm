import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function FinetuneMistral() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Finetuning Mistral with Unsloth</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Mistral 7B and its variants (Mistral NeMo, Mistral Small) are popular choices for
        finetuning due to their strong performance-to-size ratio. This walkthrough shows how
        to finetune Mistral models using Unsloth, highlighting differences from LLaMA finetuning.
      </p>

      <DefinitionBlock
        title="Mistral Architecture Differences"
        definition="Mistral uses Grouped Query Attention (GQA) with a sliding window attention mechanism (window size 4096 in Mistral 7B). It uses a different chat template with [INST] and [/INST] tokens. The tokenizer vocabulary is 32,000 tokens compared to LLaMA 3's 128,000."
        id="def-mistral-arch"
      />

      <PythonCode
        title="finetune_mistral_complete.py"
        code={`from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Step 1: Load Mistral model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.3",
    max_seq_length=4096,       # Mistral supports 32K but train at 4K
    dtype=None,
    load_in_4bit=True,
)

# Step 2: Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=32,                      # Slightly higher rank for Mistral
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
model.print_trainable_parameters()

# Step 3: Format dataset with Mistral chat template
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:5000]")

def format_mistral(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_mistral)
print(f"Sample: {dataset[0]['text'][:200]}...")

# Step 4: Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    packing=True,
    args=TrainingArguments(
        output_dir="./mistral-finetune",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        optim="adamw_8bit",
        save_strategy="steps",
        save_steps=500,
        seed=42,
    ),
)
trainer.train()`}
        id="code-mistral-finetune"
      />

      <h2 className="text-2xl font-semibold">Mistral Chat Template</h2>

      <PythonCode
        title="mistral_chat_template.py"
        code={`# Mistral Instruct v0.3 chat format
# The tokenizer handles this automatically with apply_chat_template

# Manual format (for reference):
mistral_template = """<s>[INST] {system_message}

{user_message} [/INST] {assistant_message}</s>"""

# With the tokenizer:
messages = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "Write a Python function to sort a list."},
    {"role": "assistant", "content": "Here is a simple sort function:\\n\\ndef sort_list(lst):\\n    return sorted(lst)"},
]

# Let the tokenizer format correctly
formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
print(formatted)

# For multi-turn conversations:
multi_turn = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user", "content": "Give me an example."},
    {"role": "assistant", "content": "print('Hello, World!')"},
]
formatted_multi = tokenizer.apply_chat_template(
    multi_turn, tokenize=False, add_generation_prompt=False
)
print(f"\\nMulti-turn:\\n{formatted_multi}")`}
        id="code-mistral-template"
      />

      <ExampleBlock
        title="Mistral vs LLaMA Finetuning Differences"
        problem="What changes are needed when switching from LLaMA to Mistral finetuning?"
        steps={[
          { formula: '\\text{Chat template: [INST]...[/INST] vs <|...|>}', explanation: 'Different special tokens. Always use tokenizer.apply_chat_template().' },
          { formula: '\\text{Vocab size: 32K vs 128K}', explanation: 'Smaller vocab means less memory for logits computation.' },
          { formula: '\\text{GQA: 8 KV heads vs 8 KV heads}', explanation: 'Both use GQA, but Mistral has sliding window attention.' },
          { formula: '\\text{LoRA rank: 32 recommended for Mistral}', explanation: 'Slightly higher rank compensates for the smaller model architecture.' },
        ]}
        id="example-mistral-vs-llama"
      />

      <NoteBlock
        type="tip"
        title="Mistral NeMo 12B"
        content="Mistral NeMo 12B (released with NVIDIA) offers better quality than Mistral 7B with the same QLoRA memory footprint on 24 GB GPUs. It uses a larger 128K vocabulary with tiktoken-style BPE. Use model_name='unsloth/Mistral-Nemo-Instruct-2407' with Unsloth."
        id="note-nemo"
      />

      <WarningBlock
        title="Sliding Window Attention in Training"
        content="Mistral's sliding window attention (4096 tokens) means the model cannot attend to earlier tokens beyond the window. When finetuning on long documents, be aware that the model may lose context. For long-context tasks, consider LLaMA 3.1 (128K context) instead."
        id="warning-sliding-window"
      />
    </div>
  )
}
