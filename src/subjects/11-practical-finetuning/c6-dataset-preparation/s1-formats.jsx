import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function DatasetFormats() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Dataset Formats: Alpaca, ShareGPT, and Conversation</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Different finetuning frameworks expect data in specific formats. The three most common are
        Alpaca (instruction-input-output), ShareGPT (multi-turn conversations), and the Hugging
        Face messages format. Understanding and converting between these is essential.
      </p>

      <DefinitionBlock
        title="Alpaca Format"
        definition="The Alpaca format uses three fields per example: instruction (the task description), input (optional additional context), and output (the desired response). It was introduced by the Stanford Alpaca project and remains widely used for single-turn instruction datasets."
        id="def-alpaca"
      />

      <h2 className="text-2xl font-semibold">Format Examples</h2>

      <PythonCode
        title="dataset_formats.py"
        code={`import json

# --- Alpaca Format ---
alpaca_example = {
    "instruction": "Classify the sentiment of this review.",
    "input": "The food was amazing and the service was excellent!",
    "output": "Positive. The review expresses satisfaction with both the food quality and service."
}

# --- ShareGPT Format ---
sharegpt_example = {
    "conversations": [
        {"from": "human", "value": "What is photosynthesis?"},
        {"from": "gpt", "value": "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen."},
        {"from": "human", "value": "What is the chemical equation?"},
        {"from": "gpt", "value": "6CO2 + 6H2O + light energy -> C6H12O6 + 6O2"},
    ]
}

# --- HuggingFace Messages Format ---
hf_messages_example = {
    "messages": [
        {"role": "system", "content": "You are a science tutor."},
        {"role": "user", "content": "What is photosynthesis?"},
        {"role": "assistant", "content": "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen."},
        {"role": "user", "content": "What is the chemical equation?"},
        {"role": "assistant", "content": "6CO2 + 6H2O + light energy -> C6H12O6 + 6O2"},
    ]
}

for name, ex in [("Alpaca", alpaca_example), ("ShareGPT", sharegpt_example),
                  ("HF Messages", hf_messages_example)]:
    print(f"\\n{name}:")
    print(json.dumps(ex, indent=2)[:300])`}
        id="code-format-examples"
      />

      <h2 className="text-2xl font-semibold">Converting Between Formats</h2>

      <PythonCode
        title="format_conversion.py"
        code={`def alpaca_to_messages(example):
    """Convert Alpaca format to HF messages format."""
    messages = []
    user_content = example["instruction"]
    if example.get("input"):
        user_content += f"\\n\\n{example['input']}"
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": example["output"]})
    return {"messages": messages}

def sharegpt_to_messages(example):
    """Convert ShareGPT format to HF messages format."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = []
    for turn in example["conversations"]:
        role = role_map.get(turn["from"], turn["from"])
        messages.append({"role": role, "content": turn["value"]})
    return {"messages": messages}

def messages_to_alpaca(example):
    """Convert single-turn HF messages to Alpaca format."""
    msgs = example["messages"]
    user_msgs = [m for m in msgs if m["role"] == "user"]
    asst_msgs = [m for m in msgs if m["role"] == "assistant"]
    return {
        "instruction": user_msgs[0]["content"] if user_msgs else "",
        "input": "",
        "output": asst_msgs[0]["content"] if asst_msgs else "",
    }

# Apply to datasets
from datasets import load_dataset

# Convert Alpaca dataset to messages format
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:100]")
dataset = dataset.map(alpaca_to_messages)
print(f"Converted: {dataset[0]['messages']}")`}
        id="code-conversion"
      />

      <ExampleBlock
        title="Chat Template Application"
        problem="How does the messages format get converted to the model-specific token format?"
        steps={[
          { formula: '\\text{messages} \\rightarrow \\texttt{apply\\_chat\\_template()}', explanation: 'The tokenizer converts the structured messages into model-specific tokens.' },
          { formula: '\\text{LLaMA 3: } \\texttt{<|start\\_header\\_id|>user<|end\\_header\\_id|>}', explanation: 'LLaMA 3 uses header tags to delimit role boundaries.' },
          { formula: '\\text{Mistral: } \\texttt{[INST] ... [/INST]}', explanation: 'Mistral wraps user messages in instruction tags.' },
          { formula: '\\text{ChatML: } \\texttt{<|im\\_start|>role ... <|im\\_end|>}', explanation: 'Qwen, Phi, and others use the ChatML template standard.' },
        ]}
        id="example-chat-templates"
      />

      <NoteBlock
        type="tip"
        title="Use Messages Format"
        content="The HuggingFace messages format is the most universal. SFTTrainer handles it natively, and tokenizer.apply_chat_template() converts it to any model-specific format. Always store your data in messages format for maximum compatibility."
        id="note-use-messages"
      />

      <WarningBlock
        title="Inconsistent Formatting Degrades Quality"
        content="Mixing different formats or having inconsistent formatting within a dataset confuses the model. Common issues: missing system prompts in some examples, inconsistent newlines, HTML artifacts in text. Standardize format before training."
        id="warning-consistency"
      />
    </div>
  )
}
