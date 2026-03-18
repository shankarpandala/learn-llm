import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Qwen() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Qwen: Alibaba's Multilingual Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Qwen (Tongyi Qianwen) from Alibaba Cloud represents China's strongest contribution to
        open-weight language models. The Qwen 2 and Qwen 2.5 series span from 0.5B to 72B
        parameters, with particularly strong performance on multilingual tasks, mathematics,
        and code generation.
      </p>

      <DefinitionBlock
        title="Qwen Architecture"
        definition="Qwen 2 uses a standard decoder-only transformer with GQA, SwiGLU activations, RMSNorm, RoPE positional embeddings, and a vocabulary of 151,646 tokens. It supports a native context length of 32,768 tokens extendable to 131,072 with YaRN (Yet another RoPE extensioN) scaling."
        id="def-qwen-arch"
      />

      <h2 className="text-2xl font-semibold">Qwen 2.5 Model Family</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Qwen 2.5 (September 2024) expanded the family to include specialized variants for code
        (Qwen2.5-Coder) and mathematics (Qwen2.5-Math). The base models range from 0.5B to 72B,
        all trained on 18 trillion tokens, making them among the most overtrained open models.
      </p>

      <ExampleBlock
        title="Qwen 2.5 Model Sizes"
        problem="Compare the Qwen 2.5 model family configurations."
        steps={[
          { formula: '\\text{Qwen2.5-0.5B}: L{=}24, d{=}896, h_q{=}14, h_{kv}{=}2', explanation: 'Tiny model suitable for edge devices. GQA with 7:1 ratio.' },
          { formula: '\\text{Qwen2.5-7B}: L{=}28, d{=}3584, h_q{=}28, h_{kv}{=}4', explanation: 'Competitive with LLaMA 3 8B and Mistral 7B. 7:1 GQA ratio.' },
          { formula: '\\text{Qwen2.5-32B}: L{=}64, d{=}5120, h_q{=}40, h_{kv}{=}8', explanation: 'Strong mid-range model. 5:1 GQA ratio.' },
          { formula: '\\text{Qwen2.5-72B}: L{=}80, d{=}8192, h_q{=}64, h_{kv}{=}8', explanation: 'Flagship model. Competitive with LLaMA 3.1 70B. 8:1 GQA ratio.' },
        ]}
        id="example-qwen-sizes"
      />

      <h2 className="text-2xl font-semibold">Training Data and Multilingual Focus</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Qwen models are trained on 18T tokens covering 29+ languages with strong emphasis on
        Chinese and English. The training mix includes web text, books, code, and curated
        multilingual data. Qwen's tokenizer uses a byte-level BPE with 151,646 tokens, designed
        for efficient encoding of CJK characters.
      </p>

      <PythonCode
        title="qwen_usage.py"
        code={`from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load Qwen2.5-7B-Instruct
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Inspect architecture
config = model.config
print(f"Hidden size: {config.hidden_size}")             # 3584
print(f"Layers: {config.num_hidden_layers}")            # 28
print(f"Query heads: {config.num_attention_heads}")     # 28
print(f"KV heads: {config.num_key_value_heads}")        # 4
print(f"Intermediate: {config.intermediate_size}")      # 18944
print(f"Vocab size: {config.vocab_size}")               # 152064
print(f"RoPE theta: {config.rope_theta}")               # 1000000.0

# Multilingual tokenizer efficiency
texts = {
    "English": "The quick brown fox jumps over the lazy dog.",
    "Chinese": "Transformer 是一种强大的神经网络架构。",
    "Code": "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
}
for lang, text in texts.items():
    tokens = tokenizer.encode(text)
    print(f"{lang}: {len(tokens)} tokens for {len(text)} chars (ratio: {len(tokens)/len(text):.2f})")

# Generate with Qwen chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain GQA in 3 bullet points."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
print(tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))`}
        id="code-qwen"
      />

      <NoteBlock
        type="note"
        title="Qwen2.5-Coder"
        content="The Qwen2.5-Coder series is specifically trained on 5.5T tokens of code data and achieves state-of-the-art performance among open models on code benchmarks. The 32B-Instruct variant matches GPT-4o on HumanEval and MBPP, making it one of the strongest open code models available."
        id="note-qwen-coder"
      />

      <WarningBlock
        title="License Considerations"
        content="Qwen models use a custom Apache-2.0-compatible license, but some variants have usage restrictions for models above certain sizes or for specific commercial applications. Always check the specific model card on HuggingFace for the exact license terms before deploying in production."
        id="warning-qwen-license"
      />
    </div>
  )
}
