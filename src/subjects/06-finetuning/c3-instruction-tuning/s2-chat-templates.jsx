import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ChatTemplates() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Chat Templates: ChatML, System/User/Assistant</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Chat templates define the structured format for multi-turn conversations with language
        models. They separate messages by role (system, user, assistant) using special tokens
        or delimiters. Consistent formatting during training and inference is crucial: a mismatch
        between the template used during finetuning and during deployment leads to degraded
        performance.
      </p>

      <DefinitionBlock
        title="Chat Template"
        definition="A chat template is a formatting specification that converts a list of role-tagged messages into a single string for model input. It defines how to encode message boundaries, role indicators, and special tokens. Common formats include ChatML (OpenAI), Llama-style, and Alpaca-style templates. The tokenizer's apply_chat_template method handles this conversion."
        id="def-chat-template"
      />

      <h2 className="text-2xl font-semibold">Common Chat Formats</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Different model families use different chat formats. Using the wrong format at inference
        time is one of the most common causes of poor model performance in production.
      </p>

      <ExampleBlock
        title="Chat Format Comparison"
        problem="Show how the same conversation is formatted in ChatML and Llama-2 templates."
        steps={[
          { formula: '\\text{ChatML: } \\texttt{<|im\\_start|>system} \\ldots \\texttt{<|im\\_end|>}', explanation: 'ChatML uses <|im_start|>role and <|im_end|> delimiters. Used by many open models.' },
          { formula: '\\text{Llama-2: } \\texttt{[INST] <<SYS>>} \\ldots \\texttt{<</SYS>>} \\ldots \\texttt{[/INST]}', explanation: 'Llama-2 nests the system prompt inside the first [INST] block with <<SYS>> tags.' },
          { formula: '\\text{Llama-3: } \\texttt{<|begin\\_of\\_text|><|start\\_header\\_id|>system<|end\\_header\\_id|>}', explanation: 'Llama-3 switched to a cleaner header-based format with explicit role markers.' },
        ]}
        id="example-format-comparison"
      />

      <PythonCode
        title="chat_templates.py"
        code={`from transformers import AutoTokenizer

# Define a multi-turn conversation
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I reverse a list in Python?"},
    {"role": "assistant", "content": "You can use list[::-1] or list.reverse()."},
    {"role": "user", "content": "What's the difference between the two?"},
]

# ChatML format (used by many models including Qwen, Mistral variants)
chatml_template = """{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}<|im_start|>assistant
"""

# Apply templates with different tokenizers
model_names = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"=== {model_name.split('/')[-1]} ===")
    print(formatted[:300])
    print()

# Custom template for training
# Mask system and user tokens; only compute loss on assistant responses
def create_training_labels(tokenizer, messages):
    """Create input_ids and labels with masked non-assistant tokens."""
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    input_ids = tokenizer.encode(full_text)
    labels = [-100] * len(input_ids)  # Mask everything initially

    # Unmask only assistant response tokens
    # (Implementation depends on the specific template format)
    return input_ids, labels`}
        id="code-chat-templates"
      />

      <NoteBlock
        type="tip"
        title="The Jinja2 Template System"
        content="HuggingFace tokenizers use Jinja2 templates stored in tokenizer_config.json. You can inspect any model's template with tokenizer.chat_template. When finetuning, always use the same template the model was pretrained/instruction-tuned with. If training from a base model, pick a standard template (ChatML is a good default) and use it consistently."
        id="note-jinja"
      />

      <WarningBlock
        title="Template Mismatch"
        content="Using the wrong chat template at inference is equivalent to feeding the model garbled input. A model trained with ChatML will not respond properly if prompted with Llama-2 format. Always verify the template matches the model. Common symptoms of mismatch: the model repeats the prompt, generates the wrong role's response, or produces incoherent output."
        id="warning-mismatch"
      />

      <NoteBlock
        type="note"
        title="Loss Masking for Chat Training"
        content="During supervised finetuning on chat data, the loss should only be computed on assistant response tokens. System prompts and user messages are part of the input context but should not contribute to the training loss. This is achieved by setting labels to -100 (the PyTorch ignore index) for all non-assistant tokens."
        id="note-loss-masking"
      />
    </div>
  )
}
