import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LLaMA() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LLaMA 1 and 2: Meta's Open Approach</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Meta's LLaMA (Large Language Model Meta AI) series catalyzed the open-source LLM movement.
        LLaMA 1 (February 2023) showed that smaller, well-trained models could match or exceed
        much larger ones. LLaMA 2 (July 2023) added commercial licensing and RLHF-tuned chat variants.
      </p>

      <DefinitionBlock
        title="Chinchilla-Optimal Training"
        definition="The Chinchilla scaling law (Hoffmann et al., 2022) states that for a given compute budget $C$, the optimal model size $N$ and dataset size $D$ scale equally: $N \propto C^{0.5}$ and $D \propto C^{0.5}$. Concretely, the optimal token count is approximately $20 \times N$, meaning a 7B model should be trained on ~140B tokens."
        id="def-chinchilla"
      />

      <h2 className="text-2xl font-semibold">LLaMA 1 Architecture</h2>
      <p className="text-gray-700 dark:text-gray-300">
        LLaMA 1 combined proven techniques: pre-RMSNorm, RoPE positional embeddings, SwiGLU
        activations, and no bias terms. Crucially, it trained for much longer than Chinchilla
        prescribed -- the 7B model saw 1T tokens (7x the Chinchilla-optimal 140B), which improved
        inference-time performance at the cost of more training compute.
      </p>

      <ExampleBlock
        title="LLaMA 1 Model Family"
        problem="Compare the four LLaMA 1 model sizes and their training configurations."
        steps={[
          { formula: '\\text{LLaMA-7B}: L{=}32, d{=}4096, h{=}32, \\text{FFN}{=}11008', explanation: 'Trained on 1T tokens. Outperformed GPT-3 (175B) on most benchmarks.' },
          { formula: '\\text{LLaMA-13B}: L{=}40, d{=}5120, h{=}40, \\text{FFN}{=}13824', explanation: 'Trained on 1T tokens. Competitive with Chinchilla (70B) and PaLM (540B) on many tasks.' },
          { formula: '\\text{LLaMA-33B}: L{=}60, d{=}6656, h{=}52, \\text{FFN}{=}17920', explanation: 'Trained on 1.4T tokens. Strong performance across reasoning benchmarks.' },
          { formula: '\\text{LLaMA-65B}: L{=}80, d{=}8192, h{=}64, \\text{FFN}{=}22016', explanation: 'Trained on 1.4T tokens. Matched or exceeded PaLM-540B despite being 8x smaller.' },
        ]}
        id="example-llama1-sizes"
      />

      <h2 className="text-2xl font-semibold">LLaMA 2 Improvements</h2>
      <p className="text-gray-700 dark:text-gray-300">
        LLaMA 2 trained on 2T tokens (2x LLaMA 1), extended context from 2048 to 4096 tokens,
        and introduced Grouped-Query Attention (GQA) in the 34B and 70B variants. The Chat versions
        used extensive RLHF with over 1 million human annotations.
      </p>

      <DefinitionBlock
        title="Grouped-Query Attention (GQA)"
        definition="A compromise between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). In GQA, $h$ query heads share $g$ key-value heads where $1 < g < h$. Each group of $h/g$ query heads shares one KV head. This reduces KV cache memory by a factor of $h/g$ while maintaining most of MHA's quality. LLaMA 2 70B uses $h{=}64$ query heads with $g{=}8$ KV heads."
        id="def-gqa"
      />

      <PythonCode
        title="load_llama_model.py"
        code={`from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load LLaMA 2 7B (requires access approval on HuggingFace)
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Inspect architecture
config = model.config
print(f"Hidden size: {config.hidden_size}")          # 4096
print(f"Num layers: {config.num_hidden_layers}")     # 32
print(f"Num attention heads: {config.num_attention_heads}")      # 32
print(f"Num KV heads: {config.num_key_value_heads}")             # 32 (MHA for 7B)
print(f"Intermediate size: {config.intermediate_size}")          # 11008
print(f"Vocab size: {config.vocab_size}")             # 32000
print(f"Max position: {config.max_position_embeddings}")  # 4096
print(f"RoPE theta: {config.rope_theta}")             # 10000.0

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")  # ~6.7B

# Generate text
inputs = tokenizer("The key innovation of LLaMA is", return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100, temperature=0.7, top_p=0.9)
print(tokenizer.decode(output[0], skip_special_tokens=True))`}
        id="code-llama-load"
      />

      <NoteBlock
        type="historical"
        title="The LLaMA Leak"
        content="LLaMA 1 weights were initially released under a research-only license, but within a week they were leaked publicly. This accidental open-sourcing sparked an explosion of community fine-tunes (Alpaca, Vicuna, WizardLM) and fundamentally shifted the LLM landscape toward openness. Meta embraced this with LLaMA 2's commercial license."
        id="note-llama-leak"
      />

      <WarningBlock
        title="Tokenizer Limitations"
        content="LLaMA 1/2 use a SentencePiece BPE tokenizer with only 32,000 tokens. This is much smaller than GPT-4's ~100K vocabulary, which means LLaMA tokenizes non-English text and code less efficiently, requiring more tokens for the same content. LLaMA 3 addressed this with a 128K vocabulary."
        id="warning-tokenizer"
      />
    </div>
  )
}
