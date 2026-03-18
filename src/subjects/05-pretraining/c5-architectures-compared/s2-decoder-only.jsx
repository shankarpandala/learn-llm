import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function DecoderOnly() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Decoder-Only Models: The GPT Family</h1>
      <p className="text-lg text-gray-300">
        Decoder-only architectures use causal (left-to-right) self-attention where each token
        can only attend to preceding tokens. This enables autoregressive generation and has
        become the dominant architecture for large language models.
      </p>

      <DefinitionBlock
        title="Decoder-Only Architecture"
        definition="A decoder-only Transformer applies causal self-attention: $M_{ij} = \mathbb{1}[j \leq i]$. Position $i$ attends only to positions $1, \ldots, i$. This enables autoregressive generation where $P(x_t \mid x_{<t})$ is computed from the hidden state at position $t$. The same architecture serves both training (teacher forcing on full sequences) and inference (token-by-token generation)."
        notation="$h_t = \text{CausalTransformer}(x_1, \ldots, x_t)_t$, $P(x_{t+1}) = \text{softmax}(W_{\text{LM}} h_t)$."
        id="decoder-only-def"
      />

      <ExampleBlock
        title="GPT Family Evolution"
        problem="Trace the evolution of decoder-only models from GPT-1 to modern LLMs."
        steps={[
          {
            formula: '\\text{GPT-1 (2018): 117M, 12 layers, BookCorpus}',
            explanation: 'Showed unsupervised pretraining + supervised fine-tuning works across NLP tasks.'
          },
          {
            formula: '\\text{GPT-2 (2019): 1.5B, 48 layers, WebText}',
            explanation: 'Demonstrated zero-shot transfer. Generated coherent paragraphs. Not initially released.'
          },
          {
            formula: '\\text{GPT-3 (2020): 175B, 96 layers, 300B tokens}',
            explanation: 'In-context learning via prompting. Few-shot performance rivaled fine-tuned models.'
          },
          {
            formula: '\\text{LLaMA (2023): 7-65B, RMSNorm, SwiGLU, RoPE}',
            explanation: 'Open weights. Showed smaller models trained on more data can match larger ones.'
          },
          {
            formula: '\\text{LLaMA-3 (2024): 8-405B, 128K vocab, 15T tokens, GQA}',
            explanation: 'Trained far beyond Chinchilla-optimal. 405B rivaled GPT-4 on benchmarks.'
          }
        ]}
        id="gpt-family-example"
      />

      <NoteBlock
        type="note"
        title="Modern Decoder-Only Architectural Innovations"
        content="Modern decoder-only models incorporate several innovations beyond the original GPT: RMSNorm instead of LayerNorm, SwiGLU or GeGLU activation in MLP, Rotary Position Embeddings (RoPE) instead of learned positions, Grouped-Query Attention (GQA) for efficient inference, and Pre-LN (normalization before sublayers). These changes improve training stability, inference speed, and long-context performance."
        id="modern-decoder-note"
      />

      <TheoremBlock
        title="Training Efficiency of Causal LM"
        statement="A causal LM training step on a sequence of length $T$ produces $T-1$ loss terms (one per position except the first), giving $T-1$ gradient signals per sequence. This is more data-efficient per token than MLM which only computes loss on ~15% of tokens ($0.15T$ signals per sequence)."
        proof="Causal LM: $\mathcal{L} = -\frac{1}{T-1}\sum_{t=2}^T \log P(x_t|x_{<t})$, yielding $T-1$ cross-entropy terms. MLM: $\mathcal{L} = -\frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \log P(x_i|x_{\backslash\mathcal{M}})$ with $|\mathcal{M}| \approx 0.15T$. Ratio: $(T-1)/(0.15T) \approx 6.7\times$ more training signal per sequence for causal LM."
        id="causal-efficiency-thm"
      />

      <PythonCode
        title="decoder_only_models.py"
        code={`from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

# Compare decoder-only architectures
configs = {
    "GPT-2": "gpt2",
    "GPT-2 Medium": "gpt2-medium",
}

text = "The future of artificial intelligence"
for name, model_id in configs.items():
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    n_params = sum(p.numel() for p in model.parameters())

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    print(f"{name} ({model_id}):")
    print(f"  Layers: {config.n_layer}, Heads: {config.n_head}, "
          f"Hidden: {config.n_embd}")
    print(f"  Parameters: {n_params/1e6:.0f}M")
    print(f"  Loss: {outputs.loss.item():.3f}, PPL: {torch.exp(outputs.loss).item():.1f}")

# Modern architecture features (LLaMA-style)
print("\\n--- Modern Decoder Architecture (LLaMA-style) ---")
llama_config = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,       # GQA: 4 query groups per KV head
    "intermediate_size": 11008,      # SwiGLU MLP
    "rms_norm_eps": 1e-5,           # RMSNorm
    "rope_theta": 10000.0,          # RoPE base frequency
    "vocab_size": 32000,
    "max_position_embeddings": 4096,
}
for k, v in llama_config.items():
    print(f"  {k}: {v}")

# Generation with GPT-2
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

input_ids = tokenizer.encode("In a world where AI", return_tensors="pt")
with torch.no_grad():
    generated = model.generate(
        input_ids,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
    )
print(f"\\nGenerated: {tokenizer.decode(generated[0], skip_special_tokens=True)}")

# KV-Cache demonstration
with torch.no_grad():
    # First pass: compute all KV pairs
    out1 = model(input_ids, use_cache=True)
    past_kv = out1.past_key_values

    # Next token: only process the new token, reuse cached KV
    next_token = out1.logits[:, -1:, :].argmax(dim=-1)
    out2 = model(next_token, past_key_values=past_kv, use_cache=True)

    print(f"\\nKV-Cache:")
    print(f"  Layers cached: {len(past_kv)}")
    print(f"  KV shape per layer: {past_kv[0][0].shape}")  # [batch, heads, seq, d_head]
    print(f"  Cache memory: {sum(k.nelement() + v.nelement() for k, v in past_kv) * 4 / 1e6:.1f} MB")`}
        id="decoder-only-code"
      />

      <WarningBlock
        title="Decoder-Only Models Are Inefficient Encoders"
        content="Because causal attention prevents tokens from seeing future context, decoder-only models produce weaker representations for understanding tasks compared to bidirectional models of equal size. Workarounds include instruction tuning (asking the model to classify via generation) or using the last token's representation, but a dedicated encoder model will be more compute-efficient for embedding and classification tasks."
        id="decoder-encoding-warning"
      />
    </div>
  )
}
