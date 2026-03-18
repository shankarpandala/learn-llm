import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function CustomDatasets() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Custom Dataset Preparation for Unsloth</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Real-world finetuning requires working with your own data. This section covers how to
        prepare custom datasets in the formats that Unsloth and SFTTrainer expect, including
        single-turn instructions, multi-turn conversations, and completion-only training.
      </p>

      <DefinitionBlock
        title="Dataset Formats"
        definition="SFTTrainer accepts datasets in two main formats: (1) a 'text' field containing the fully formatted conversation string, or (2) a 'messages' field containing a list of role/content dictionaries that will be formatted using the tokenizer's chat template."
        id="def-formats"
      />

      <h2 className="text-2xl font-semibold">Format 1: Pre-formatted Text</h2>

      <PythonCode
        title="custom_dataset_text.py"
        code={`from datasets import Dataset
import json

# Load your custom data (JSONL, CSV, etc.)
raw_data = [
    {
        "instruction": "Summarize this article about climate change.",
        "input": "Global temperatures have risen 1.1C since pre-industrial times...",
        "output": "Global temperatures increased 1.1C, with significant impacts..."
    },
    {
        "instruction": "Translate to French.",
        "input": "The weather is beautiful today.",
        "output": "Le temps est magnifique aujourd'hui."
    },
]

# Format as chat messages for LLaMA 3
def format_to_llama3(example, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example["instruction"] +
         (f"\\n\\n{example['input']}" if example.get("input") else "")},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

# Create HuggingFace dataset
dataset = Dataset.from_list(raw_data)
# dataset = dataset.map(lambda x: format_to_llama3(x, tokenizer))
print(f"Dataset: {len(dataset)} examples")
print(f"Columns: {dataset.column_names}")`}
        id="code-text-format"
      />

      <h2 className="text-2xl font-semibold">Format 2: Messages (Recommended)</h2>

      <PythonCode
        title="custom_dataset_messages.py"
        code={`from datasets import Dataset
import json

# Multi-turn conversation format
conversations = [
    {
        "messages": [
            {"role": "system", "content": "You are a Python tutor."},
            {"role": "user", "content": "How do I read a file in Python?"},
            {"role": "assistant", "content": "Use the open() function with a context manager:\\n\\nwith open('file.txt', 'r') as f:\\n    content = f.read()"},
            {"role": "user", "content": "What about reading line by line?"},
            {"role": "assistant", "content": "Use readlines() or iterate:\\n\\nwith open('file.txt', 'r') as f:\\n    for line in f:\\n        print(line.strip())"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Explain list comprehensions."},
            {"role": "assistant", "content": "List comprehensions create lists concisely:\\n\\nsquares = [x**2 for x in range(10)]\\n\\nThis is equivalent to a for loop with append."},
        ]
    },
]

dataset = Dataset.from_list(conversations)

# Use with SFTTrainer (messages format handled automatically)
# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     # No dataset_text_field needed - uses 'messages' automatically
# )

# Loading from JSONL file
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)

# Loading from CSV/Parquet
# dataset = Dataset.from_csv("data.csv")
# dataset = Dataset.from_parquet("data.parquet")

print(f"Dataset: {len(dataset)} conversations")
print(f"First conversation turns: {len(dataset[0]['messages'])}")`}
        id="code-messages-format"
      />

      <h2 className="text-2xl font-semibold">Completion-Only Training</h2>

      <PythonCode
        title="completion_only_training.py"
        code={`from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# Train only on assistant responses, not user prompts
# This prevents the model from learning to generate user messages

# Define the response template (text that marks start of assistant response)
response_template = "<|start_header_id|>assistant<|end_header_id|>\\n\\n"

# Create data collator that masks non-response tokens
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

# Use in SFTTrainer
# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     dataset_text_field="text",
#     data_collator=collator,    # Masks loss on user/system tokens
#     max_seq_length=2048,
# )

# For Mistral format:
# response_template_mistral = "[/INST]"

# Verify masking works correctly
sample_text = dataset[0]["text"]
tokens = tokenizer(sample_text, return_tensors="pt")
# Labels should be -100 (masked) for non-response tokens
print(f"Total tokens: {tokens.input_ids.shape[1]}")
print("Completion-only training: loss computed only on assistant responses")`}
        id="code-completion-only"
      />

      <ExampleBlock
        title="Dataset Size Recommendations"
        problem="How much data do you need for different finetuning goals?"
        steps={[
          { formula: '\\text{Style/persona: } 100\\text{-}500 \\text{ examples}', explanation: 'Teaching a model a specific writing style or persona requires few examples.' },
          { formula: '\\text{Task-specific: } 1\\text{K-}10\\text{K examples}', explanation: 'Domain-specific tasks like summarization or classification need moderate data.' },
          { formula: '\\text{General instruction: } 10\\text{K-}100\\text{K examples}', explanation: 'Broad instruction-following improvement needs substantial diverse data.' },
        ]}
        id="example-dataset-size"
      />

      <NoteBlock
        type="tip"
        title="Data Quality Over Quantity"
        content="500 high-quality, diverse examples often outperform 50,000 noisy ones. Focus on: (1) accurate and helpful responses, (2) diverse instruction types, (3) consistent formatting, and (4) no contradictions. Manually review at least 100 random examples before training."
        id="note-quality"
      />

      <WarningBlock
        title="Tokenization Length Limits"
        content="Examples longer than max_seq_length are silently truncated. Check your dataset: if many examples exceed the limit, increase max_seq_length or split long examples. Use dataset.map(lambda x: {'length': len(tokenizer(x['text']).input_ids)}) to check lengths."
        id="warning-truncation"
      />
    </div>
  )
}
