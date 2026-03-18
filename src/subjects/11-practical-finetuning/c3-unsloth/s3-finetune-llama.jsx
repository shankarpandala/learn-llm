import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function FinetuneLlama() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Finetuning LLaMA with Unsloth</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        This section provides a complete, end-to-end walkthrough of finetuning LLaMA 3.1 8B Instruct
        using Unsloth and QLoRA. We cover model loading, dataset formatting, training configuration,
        and saving the final model.
      </p>

      <h2 className="text-2xl font-semibold">Step 1: Load Model and Tokenizer</h2>

      <PythonCode
        title="step1_load_model.py"
        code={`from unsloth import FastLanguageModel
import torch

# Load LLaMA 3.1 8B Instruct in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=2048,      # Context length for training
    dtype=None,                # Auto: bf16 on Ampere+, fp16 otherwise
    load_in_4bit=True,         # QLoRA 4-bit quantization
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                      # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,            # 0 is optimized in Unsloth
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% longer context
    random_state=42,
)

model.print_trainable_parameters()
# trainable: ~42M / 8B total (~0.52%)`}
        id="code-load-model"
      />

      <h2 className="text-2xl font-semibold">Step 2: Prepare Dataset</h2>

      <PythonCode
        title="step2_prepare_dataset.py"
        code={`from datasets import load_dataset

# Load an instruction-following dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
print(f"Dataset size: {len(dataset)}")
print(f"Columns: {dataset.column_names}")
# Columns: ['instruction', 'input', 'output']

# Define chat template for LLaMA 3.1 Instruct
llama3_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""

def format_example(example):
    input_text = f"\\n\\n{example['input']}" if example['input'] else ""
    text = llama3_template.format(
        instruction=example['instruction'],
        input_text=input_text,
        output=example['output'],
    )
    return {"text": text}

dataset = dataset.map(format_example)
print(f"\\nSample formatted text:\\n{dataset[0]['text'][:300]}...")`}
        id="code-prepare-dataset"
      />

      <h2 className="text-2xl font-semibold">Step 3: Configure and Train</h2>

      <PythonCode
        title="step3_train.py"
        code={`from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=True,              # Pack short sequences together
    args=TrainingArguments(
        output_dir="./llama3-finetune",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch = 8
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        save_strategy="steps",
        save_steps=200,
        report_to="wandb",        # Optional: WandB logging
    ),
)

# Show GPU memory before training
gpu_stats = torch.cuda.get_device_properties(0)
reserved = torch.cuda.max_memory_reserved() / 1e9
print(f"GPU: {gpu_stats.name}, Reserved: {reserved:.1f} GB")

# Train!
trainer_stats = trainer.train()
print(f"Training time: {trainer_stats.metrics['train_runtime']:.0f}s")
print(f"Training loss: {trainer_stats.metrics['train_loss']:.4f}")`}
        id="code-train"
      />

      <h2 className="text-2xl font-semibold">Step 4: Save and Test</h2>

      <PythonCode
        title="step4_save_test.py"
        code={`# Save LoRA adapter
model.save_pretrained("./llama3-lora-adapter")
tokenizer.save_pretrained("./llama3-lora-adapter")

# Test the finetuned model
FastLanguageModel.for_inference(model)  # Enable fast inference

messages = [
    {"role": "user", "content": "Explain quantum entanglement simply."},
]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs, max_new_tokens=256,
    temperature=0.7, top_p=0.9,
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# Save merged model (optional - for deployment)
# model.save_pretrained_merged("./llama3-merged", tokenizer)
# model.save_pretrained_gguf("./llama3-gguf", tokenizer, "q4_k_m")`}
        id="code-save-test"
      />

      <NoteBlock
        type="tip"
        title="Packing for Efficiency"
        content="Setting packing=True in SFTTrainer concatenates multiple short examples into single sequences, avoiding wasted padding tokens. This can speed up training by 2-5x on datasets with variable-length examples. Unsloth's packing implementation is particularly efficient."
        id="note-packing"
      />

      <WarningBlock
        title="Chat Template Matters"
        content="Using the wrong chat template will severely degrade model quality. Always use the template matching your base model: LLaMA 3 uses <|begin_of_text|> tags, Mistral uses [INST] tags, ChatML uses <|im_start|> tags. The tokenizer.apply_chat_template() method handles this automatically."
        id="warning-chat-template"
      />
    </div>
  )
}
