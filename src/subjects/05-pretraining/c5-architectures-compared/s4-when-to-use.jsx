import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function WhenToUse() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Choosing the Right Architecture</h1>
      <p className="text-lg text-gray-300">
        Selecting between encoder-only, decoder-only, and encoder-decoder architectures depends
        on your task, latency requirements, model scale, and deployment constraints. Each
        architecture has distinct strengths that make it optimal for specific use cases.
      </p>

      <DefinitionBlock
        title="Architecture Selection Criteria"
        definition="The three key dimensions for architecture selection are: (1) Task type: understanding (classification, retrieval) vs. generation (text, code), (2) Input-output structure: fixed-length classification vs. variable-length generation vs. sequence-to-sequence transformation, (3) Scale and efficiency: parameter count, inference latency, throughput requirements."
        id="selection-criteria-def"
      />

      <ExampleBlock
        title="Task-Architecture Mapping"
        problem="Match common NLP tasks to their optimal architecture."
        steps={[
          {
            formula: '\\text{Text Classification, NER, Retrieval} \\rightarrow \\text{Encoder-Only (BERT, DeBERTa)}',
            explanation: 'Bidirectional context gives best representations for understanding. Small models (110-340M) suffice. Fast inference.'
          },
          {
            formula: '\\text{Open-ended Generation, Chat, Code} \\rightarrow \\text{Decoder-Only (GPT, LLaMA)}',
            explanation: 'Autoregressive generation is natural. Scales well with parameters. In-context learning eliminates task-specific fine-tuning.'
          },
          {
            formula: '\\text{Translation, Summarization} \\rightarrow \\text{Encoder-Decoder (T5, mBART)}',
            explanation: 'Natural fit: encode source, decode target. Cross-attention aligns input-output. Best for structured transformations.'
          },
          {
            formula: '\\text{General-purpose / Unknown tasks} \\rightarrow \\text{Decoder-Only (large LLM)}',
            explanation: 'Large decoder-only models handle nearly all tasks via prompting. Default choice when task diversity is high.'
          }
        ]}
        id="task-mapping-example"
      />

      <NoteBlock
        type="tip"
        title="Decision Framework"
        content="Ask these questions in order: (1) Do you need to generate text? If no, use encoder-only for efficiency. (2) Is the task a structured input-to-output transformation (translation, summarization)? If yes, consider encoder-decoder. (3) Do you need flexibility across many tasks or conversational interaction? Use decoder-only. (4) Are you constrained on model size (<500M params)? Encoder-only models offer the best quality-per-parameter for understanding tasks."
        id="decision-framework-note"
      />

      <TheoremBlock
        title="Compute Efficiency Comparison"
        statement="For a fixed parameter budget $N$ and sequence length $S$: Encoder-only processes the full sequence in one pass: $C_{\text{enc}} = 2NS$ FLOPs. Decoder-only with KV-cache generates $T$ tokens: $C_{\text{dec}} = 2NS + 2NT$ FLOPs (prefill + generation). Encoder-decoder with input $S$ and output $T$: $C_{\text{enc-dec}} = 2N_{\text{enc}}S + 2N_{\text{dec}}T + C_{\text{cross}}$."
        proof="Each Transformer forward pass costs approximately $2N$ FLOPs per token (each parameter participates in one multiply-add). For encoding $S$ tokens: $2NS$. For generating $T$ tokens with KV-cache: each new token costs $2N$ FLOPs, so total generation is $2NT$. Encoder-decoder has separate parameter budgets for encoder and decoder, plus cross-attention overhead."
        id="compute-comparison-thm"
      />

      <PythonCode
        title="architecture_comparison.py"
        code={`from transformers import (
    AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoTokenizer, pipeline
)
import torch
import time

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def benchmark_inference(model_name, model_class, task_input, generate=False):
    """Benchmark model inference."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(task_input, return_tensors="pt", truncation=True, max_length=512)
    params = count_params(model)

    start = time.perf_counter()
    with torch.no_grad():
        if generate:
            out = model.generate(inputs["input_ids"], max_new_tokens=20)
            result = tokenizer.decode(out[0], skip_special_tokens=True)
        else:
            out = model(**inputs)
            result = f"hidden shape: {out.last_hidden_state.shape}"
    elapsed = time.perf_counter() - start

    return {
        "model": model_name,
        "params": f"{params/1e6:.0f}M",
        "time_ms": f"{elapsed*1000:.0f}",
        "result": result[:80],
    }

# Compare architectures
text = "Natural language processing has been transformed by deep learning."
comparisons = [
    ("bert-base-uncased", AutoModel, text, False),
    ("gpt2", AutoModelForCausalLM, text, True),
    ("t5-small", AutoModelForSeq2SeqLM, "summarize: " + text, True),
]

print("=== Architecture Comparison ===")
print(f"{'Model':>25s} {'Params':>8s} {'Time':>8s}  Result")
print("-" * 90)
for model_name, model_cls, inp, gen in comparisons:
    try:
        result = benchmark_inference(model_name, model_cls, inp, gen)
        print(f"{result['model']:>25s} {result['params']:>8s} "
              f"{result['time_ms']:>6s}ms  {result['result']}")
    except Exception as e:
        print(f"{model_name:>25s}: {str(e)[:60]}")

# Architecture recommendation engine
def recommend_architecture(task_type, scale, latency_ms, generates_text):
    """Recommend architecture based on requirements."""
    recommendations = []

    if not generates_text:
        if scale == "small":
            recommendations.append(("Encoder-Only (BERT/DeBERTa)", 0.95))
        else:
            recommendations.append(("Encoder-Only (large)", 0.80))
            recommendations.append(("Decoder-Only (prompted)", 0.70))
    elif task_type in ["translation", "summarization"]:
        recommendations.append(("Encoder-Decoder (T5/BART)", 0.90))
        recommendations.append(("Decoder-Only (prompted)", 0.75))
    elif task_type in ["chat", "code", "general"]:
        recommendations.append(("Decoder-Only (LLaMA/GPT)", 0.95))
    else:
        recommendations.append(("Decoder-Only (general)", 0.80))
        recommendations.append(("Encoder-Decoder", 0.60))

    return sorted(recommendations, key=lambda x: -x[1])

# Example recommendations
scenarios = [
    {"task_type": "classification", "scale": "small", "latency_ms": 10, "generates_text": False},
    {"task_type": "chat", "scale": "large", "latency_ms": 500, "generates_text": True},
    {"task_type": "translation", "scale": "medium", "latency_ms": 200, "generates_text": True},
    {"task_type": "retrieval", "scale": "small", "latency_ms": 5, "generates_text": False},
]

print("\\n=== Architecture Recommendations ===")
for s in scenarios:
    recs = recommend_architecture(**s)
    print(f"\\nTask: {s['task_type']}, Scale: {s['scale']}, Generates: {s['generates_text']}")
    for arch, score in recs:
        print(f"  {score:.0%} -> {arch}")

print("\\n=== Summary ===")
print("Encoder-only:    Best for embeddings, classification, retrieval")
print("Decoder-only:    Best for generation, chat, general-purpose LLMs")
print("Encoder-decoder: Best for structured seq2seq (translation, summarization)")`}
        id="comparison-code"
      />

      <WarningBlock
        title="The Convergence Trend"
        content="The distinction between architectures is blurring. Large decoder-only models can perform classification via generation ('Is this positive or negative?'). Encoder-decoder models can do open-ended generation. The practical choice increasingly comes down to: use a large decoder-only model for versatility, or a small encoder-only model for efficient specialization. Pure encoder-decoder models are becoming rarer in new development."
        id="convergence-warning"
      />

      <NoteBlock
        type="note"
        title="Hybrid Approaches"
        content="Some architectures blur the lines: PrefixLM (like UL2) uses bidirectional attention on a prefix and causal attention on the rest. Mixture-of-Denoisers (MoD) trains with multiple objectives simultaneously. These approaches attempt to combine the strengths of different architectures in a single model."
        id="hybrid-note"
      />
    </div>
  )
}
