import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LLaMA3() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LLaMA 3: Scaling Open Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        LLaMA 3 (April 2024) marked a significant leap for open-weight models. The 8B and 70B
        models matched or exceeded GPT-3.5 on most benchmarks, while the 405B variant approached
        GPT-4 and Claude 3.5 Sonnet. Key changes included a 128K vocabulary, 8K default context
        (128K extended), and training on 15T+ tokens.
      </p>

      <DefinitionBlock
        title="Over-Training"
        definition="Training a model on significantly more tokens than Chinchilla-optimal. LLaMA 3 8B was trained on 15T tokens, approximately $100\times$ the Chinchilla-optimal amount of ~150B tokens. This sacrifices training compute efficiency for improved inference efficiency: a smaller, over-trained model is cheaper to deploy than a larger, compute-optimally trained one."
        id="def-overtraining"
      />

      <h2 className="text-2xl font-semibold">Architecture Improvements</h2>
      <p className="text-gray-700 dark:text-gray-300">
        LLaMA 3 maintains the same core recipe as LLaMA 2 (RMSNorm, RoPE, SwiGLU, GQA) but
        with important refinements: a 4x larger vocabulary (128,256 vs 32,000), GQA across all
        model sizes (not just 70B+), and training on 15T tokens of multilingual data.
      </p>

      <ExampleBlock
        title="LLaMA 3 Model Family"
        problem="Compare LLaMA 3 model configurations against LLaMA 2 equivalents."
        steps={[
          { formula: '\\text{LLaMA 3 8B}: L{=}32, d{=}4096, h_q{=}32, h_{kv}{=}8', explanation: 'GQA with 8 KV heads (vs MHA in LLaMA 2 7B). Vocab 128K. Trained on 15T tokens.' },
          { formula: '\\text{LLaMA 3 70B}: L{=}80, d{=}8192, h_q{=}64, h_{kv}{=}8', explanation: 'Same GQA ratio as LLaMA 2 70B. 15T training tokens (7.5x more than LLaMA 2).' },
          { formula: '\\text{LLaMA 3.1 405B}: L{=}126, d{=}16384, h_q{=}128, h_{kv}{=}8', explanation: 'Largest open model at release. 128K context with RoPE scaling. Approaches GPT-4.' },
          { formula: '\\text{Vocab}: 32{,}000 \\to 128{,}256', explanation: '4x larger vocabulary dramatically improves multilingual and code tokenization efficiency.' },
        ]}
        id="example-llama3-configs"
      />

      <h2 className="text-2xl font-semibold">Training Data and Process</h2>
      <p className="text-gray-700 dark:text-gray-300">
        LLaMA 3 was trained on over 15 trillion tokens from a curated mix of web data, with
        improved filtering using Llama 2 itself as a quality classifier. The data mix was ~5%
        code, with significantly more multilingual content than LLaMA 2. Post-training involved
        both SFT and DPO (Direct Preference Optimization) rather than PPO-based RLHF.
      </p>

      <PythonCode
        title="llama3_usage.py"
        code={`from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load LLaMA 3 8B Instruct
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

config = model.config
print(f"Vocab size: {config.vocab_size}")                 # 128256
print(f"Hidden size: {config.hidden_size}")               # 4096
print(f"Layers: {config.num_hidden_layers}")              # 32
print(f"Query heads: {config.num_attention_heads}")       # 32
print(f"KV heads: {config.num_key_value_heads}")          # 8
print(f"Intermediate: {config.intermediate_size}")        # 14336
print(f"RoPE theta: {config.rope_theta}")                 # 500000.0

# Compare tokenizer efficiency
llama2_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
llama3_tok = tokenizer

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "Bonjour le monde, comment allez-vous aujourd'hui?",
]
print("\\nTokenizer efficiency comparison:")
for text in texts:
    l2 = len(llama2_tok.encode(text))
    l3 = len(llama3_tok.encode(text))
    saving = (1 - l3 / l2) * 100
    print(f"  '{text[:40]}...' L2={l2} L3={l3} ({saving:.0f}% fewer tokens)")

# Chat generation with LLaMA 3 format
messages = [
    {"role": "system", "content": "You are a concise technical assistant."},
    {"role": "user", "content": "What are the key improvements in LLaMA 3?"},
]
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = pipe(messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])`}
        id="code-llama3"
      />

      <NoteBlock
        type="note"
        title="LLaMA 3.1 and Extended Context"
        content="LLaMA 3.1 extended context to 128K tokens using a progressive RoPE frequency scaling approach. The base RoPE theta was increased from 500,000 to 500,000 with additional NTK-aware scaling. Training on long documents was done in stages: first 8K, then 32K, then 128K context lengths."
        id="note-llama31-context"
      />

      <WarningBlock
        title="Memory Requirements"
        content="LLaMA 3 405B requires ~810GB in bfloat16, needing at least 8x A100 80GB GPUs. Even the 70B model needs ~140GB in bfloat16. For practical deployment, quantization (GPTQ, AWQ, or GGUF 4-bit) reduces the 70B model to ~35GB, fitting on a single A100 or two consumer GPUs."
        id="warning-llama3-memory"
      />
    </div>
  )
}
