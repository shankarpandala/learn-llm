import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function FullWalkthrough() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">End-to-End Finetuning Walkthrough</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        This section ties together everything covered in TRL by walking through a complete
        finetuning pipeline: from data preparation through SFT, DPO alignment, evaluation,
        and model export. Follow this as a template for your own projects.
      </p>

      <h2 className="text-2xl font-semibold">Phase 1: Data Preparation</h2>

      <PythonCode
        title="phase1_data_prep.py"
        code={`from datasets import load_dataset, DatasetDict
import json

# Load and split dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Train/validation split
split = dataset.train_test_split(test_size=0.05, seed=42)
train_data = split["train"]
eval_data = split["test"]
print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

# Inspect data quality
lengths = [len(ex["output"]) for ex in train_data]
print(f"Response lengths: min={min(lengths)}, max={max(lengths)}, "
      f"median={sorted(lengths)[len(lengths)//2]}")

# Filter out very short or very long examples
train_data = train_data.filter(
    lambda x: 10 < len(x["output"]) < 2000
)
print(f"After filtering: {len(train_data)}")

# Convert to messages format
def to_messages(example):
    user_msg = example["instruction"]
    if example.get("input"):
        user_msg += f"\\n\\n{example['input']}"
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": example["output"]},
        ]
    }

train_data = train_data.map(to_messages)
eval_data = eval_data.map(to_messages)`}
        id="code-phase1"
      />

      <h2 className="text-2xl font-semibold">Phase 2: SFT Training</h2>

      <PythonCode
        title="phase2_sft.py"
        code={`from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import torch

# Load model with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=eval_data,
    args=SFTConfig(
        output_dir="./phase2-sft",
        max_seq_length=2048,
        packing=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        optim="adamw_8bit",
    ),
)

trainer.train()
trainer.save_model("./sft-checkpoint")`}
        id="code-phase2"
      />

      <h2 className="text-2xl font-semibold">Phase 3: Evaluation and Export</h2>

      <PythonCode
        title="phase3_eval_export.py"
        code={`# Quick qualitative evaluation
FastLanguageModel.for_inference(model)

test_prompts = [
    "Explain the difference between a list and a tuple in Python.",
    "Write a haiku about machine learning.",
    "What are three tips for effective public speaking?",
]

for prompt in test_prompts:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs, max_new_tokens=256,
        temperature=0.7, top_p=0.9, do_sample=True,
    )
    response = tokenizer.decode(outputs[0][inputs.shape[1]:],
                                 skip_special_tokens=True)
    print(f"Q: {prompt}")
    print(f"A: {response}\\n{'='*60}")

# Export options
# 1. Save LoRA adapter only (small, ~100 MB)
model.save_pretrained("./final-lora-adapter")
tokenizer.save_pretrained("./final-lora-adapter")

# 2. Save merged model (full size, ~16 GB for 8B model)
# model.save_pretrained_merged("./final-merged", tokenizer)

# 3. Export to GGUF for llama.cpp
# model.save_pretrained_gguf("./final-gguf", tokenizer, "q4_k_m")

# 4. Push to Hugging Face Hub
# model.push_to_hub("your-username/your-model-name")
# tokenizer.push_to_hub("your-username/your-model-name")`}
        id="code-phase3"
      />

      <ExampleBlock
        title="Complete Pipeline Timeline"
        problem="How long does each phase take for a typical 8B model finetune?"
        steps={[
          { formula: '\\text{Data prep: } 10\\text{-}30 \\text{ minutes}', explanation: 'Loading, filtering, formatting. One-time cost.' },
          { formula: '\\text{SFT (50K examples, 1 epoch): } 2\\text{-}4 \\text{ hours}', explanation: 'On RTX 4090 with QLoRA + Unsloth. 1 hour on A100.' },
          { formula: '\\text{DPO (optional, 10K pairs): } 1\\text{-}2 \\text{ hours}', explanation: 'After SFT. Improves helpfulness and safety alignment.' },
          { formula: '\\text{Evaluation: } 30 \\text{ minutes}', explanation: 'Run benchmark suite and qualitative tests.' },
          { formula: '\\text{Export (GGUF): } 10\\text{-}20 \\text{ minutes}', explanation: 'Merging and quantizing for deployment.' },
        ]}
        id="example-timeline"
      />

      <NoteBlock
        type="tip"
        title="Iteration Strategy"
        content="Do not aim for perfection on the first run. Start with a small subset (1K-5K examples), train for 1 epoch, evaluate qualitatively, then iterate on the data. Common iterations: fixing formatting issues, removing low-quality examples, adding more examples for weak areas."
        id="note-iteration"
      />

      <WarningBlock
        title="Save Checkpoints Frequently"
        content="Always save checkpoints during training. GPU crashes, OOM errors, and power outages happen. Use save_steps=200 and keep at least the last 3 checkpoints. Training can be resumed from any checkpoint with trainer.train(resume_from_checkpoint='./checkpoint-400')."
        id="warning-checkpoints"
      />
    </div>
  )
}
