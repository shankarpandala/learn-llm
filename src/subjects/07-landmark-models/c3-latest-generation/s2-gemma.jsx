import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Gemma() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Google Gemma: Lightweight Open Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Gemma (February 2024) is Google DeepMind's family of open-weight models built on the same
        research and technology as the Gemini models. Available in 2B and 7B sizes, Gemma demonstrated
        that Google's internal training infrastructure and data pipeline could produce highly
        competitive small models for the open-source community.
      </p>

      <DefinitionBlock
        title="Gemma Architecture"
        definition="Gemma uses a decoder-only transformer with Multi-Query Attention (2B) or Multi-Head Attention (7B), RoPE positional embeddings, GeGLU activation (a GELU-gated variant of GLU), and RMSNorm. A distinctive feature is embedding tying: the input and output embedding matrices are shared, and the embedding dimension is normalized by $\sqrt{d}$ before the final projection."
        id="def-gemma-arch"
      />

      <h2 className="text-2xl font-semibold">Architecture Details</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Gemma follows the modern transformer recipe but with a few Google-specific choices. It uses
        GeGLU (GELU-gated linear unit) instead of SwiGLU, and applies a learnable scaling factor
        to embeddings. Both models were trained on 6T tokens of primarily English web data, code,
        and mathematics.
      </p>

      <ExampleBlock
        title="Gemma Model Configurations"
        problem="Compare Gemma 1 and Gemma 2 model variants."
        steps={[
          { formula: '\\text{Gemma 2B}: L{=}18, d{=}2048, h{=}8, d_{ff}{=}16384', explanation: 'Uses MQA (1 KV head). 2T tokens. Surprisingly capable for its size.' },
          { formula: '\\text{Gemma 7B}: L{=}28, d{=}3072, h{=}16, d_{ff}{=}24576', explanation: 'Uses MHA (16 KV heads). 6T tokens. Outperforms LLaMA 2 7B and Mistral 7B.' },
          { formula: '\\text{Gemma 2 9B}: L{=}42, d{=}3584, h{=}16, h_{kv}{=}8', explanation: 'Uses GQA. Alternates local (4096) and global attention layers. Knowledge distillation from larger model.' },
          { formula: '\\text{Gemma 2 27B}: L{=}46, d{=}4608, h{=}32, h_{kv}{=}16', explanation: 'GQA with soft-capping on attention logits. Matches LLaMA 3 70B on several benchmarks.' },
        ]}
        id="example-gemma-configs"
      />

      <h2 className="text-2xl font-semibold">Gemma 2 Innovations</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Gemma 2 introduced alternating local and global attention layers (reducing the quadratic
        cost of full attention), logit soft-capping to prevent attention weights from becoming too
        sharp, and knowledge distillation from a larger teacher model during pre-training.
      </p>

      <DefinitionBlock
        title="Logit Soft-Capping"
        definition="A technique that prevents attention logits from growing unboundedly by applying $\text{logits} = \text{cap} \cdot \tanh\left(\frac{\text{logits}}{\text{cap}}\right)$ where the cap value (e.g., 50.0) limits the maximum attention logit magnitude. This stabilizes training and prevents entropy collapse in attention distributions."
        id="def-soft-capping"
      />

      <PythonCode
        title="gemma_usage.py"
        code={`from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load Gemma 2 9B
model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Inspect architecture
config = model.config
print(f"Model type: {config.model_type}")
print(f"Hidden size: {config.hidden_size}")           # 3584
print(f"Num layers: {config.num_hidden_layers}")      # 42
print(f"Query heads: {config.num_attention_heads}")   # 16
print(f"KV heads: {config.num_key_value_heads}")      # 8
print(f"Intermediate: {config.intermediate_size}")    # 14336
print(f"Vocab size: {config.vocab_size}")             # 256000

# Gemma uses a very large vocabulary (256K)
texts = ["Hello world", "Bonjour le monde", "import torch"]
for text in texts:
    tokens = tokenizer.encode(text)
    print(f"'{text}' -> {len(tokens)} tokens: {tokens}")

# Count parameters by component
embedding_params = model.model.embed_tokens.weight.numel()
total_params = sum(p.numel() for p in model.parameters())
print(f"\\nEmbedding params: {embedding_params / 1e9:.2f}B "
      f"({embedding_params / total_params * 100:.1f}% of total)")
print(f"Total params: {total_params / 1e9:.2f}B")

# Generate with Gemma chat format
messages = [{"role": "user", "content": "Explain attention soft-capping in 2 sentences."}]
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = pipe(messages, max_new_tokens=100)
print(output[0]["generated_text"][-1]["content"])`}
        id="code-gemma"
      />

      <NoteBlock
        type="note"
        title="Gemma's 256K Vocabulary"
        content="Gemma uses a 256,000-token SentencePiece vocabulary, the largest among open models. This dramatically improves tokenization efficiency for non-English languages and code. However, the large embedding matrix (256K * d) means embedding parameters account for a larger fraction of the total model size, especially for the 2B variant."
        id="note-gemma-vocab"
      />

      <WarningBlock
        title="Attention Compatibility"
        content="Gemma 2's alternating local/global attention and logit soft-capping require custom attention implementations. Standard FlashAttention does not support soft-capping natively, which initially limited inference optimization. Updated versions of frameworks like vLLM and TGI added support, but check compatibility before deploying."
        id="warning-gemma-compat"
      />
    </div>
  )
}
