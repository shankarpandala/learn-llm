import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function PhiModels() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Phi Models: Small Models, High-Quality Data</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Microsoft's Phi series challenged the assumption that larger models are always better.
        Phi-1 (1.3B), Phi-2 (2.7B), and Phi-3 (3.8B) demonstrated that carefully curated,
        high-quality training data can produce small models that punch far above their weight class,
        often matching models 10-25x their size.
      </p>

      <DefinitionBlock
        title="Data Quality Scaling"
        definition="The principle that model performance depends not just on dataset size but critically on data quality. For small models, the quality-to-quantity ratio becomes paramount: $\text{Performance} \propto f(\text{quality}) \cdot g(N, D)$ where the quality function $f$ dominates at small model sizes $N$."
        id="def-data-quality"
      />

      <h2 className="text-2xl font-semibold">Phi-1: Textbooks Are All You Need</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Phi-1 (June 2023) was a 1.3B parameter model trained specifically for code generation.
        It achieved 50.6% on HumanEval, outperforming models 10x larger. The secret was "textbook
        quality" data: a combination of filtered code from The Stack, GPT-3.5-generated textbook
        explanations, and GPT-3.5-generated exercises totaling only 7B tokens.
      </p>

      <ExampleBlock
        title="Phi Series Evolution"
        problem="Trace the Phi family's progression in size, data strategy, and benchmark performance."
        steps={[
          { formula: '\\text{Phi-1 (1.3B)}: \\text{HumanEval} = 50.6\\%', explanation: 'Code-only model. Trained on 7B tokens of "textbook quality" code data. Matched StarCoder 15B.' },
          { formula: '\\text{Phi-1.5 (1.3B)}: \\text{Common sense reasoning competitive with 5x larger}', explanation: 'Extended to natural language. Used 30B tokens of synthetic textbook + web data.' },
          { formula: '\\text{Phi-2 (2.7B)}: \\text{MMLU} = 56.7\\%', explanation: 'Matched Mistral 7B and LLaMA 2 70B on some benchmarks. Trained on 1.4T tokens of curated web + synthetic data.' },
          { formula: '\\text{Phi-3-mini (3.8B)}: \\text{MMLU} = 69.0\\%', explanation: 'Matched LLaMA 3 8B. Used heavily filtered web data + synthetic data. 3.3T tokens.' },
        ]}
        id="example-phi-evolution"
      />

      <h2 className="text-2xl font-semibold">Synthetic Data Pipeline</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The Phi models pioneered the use of LLM-generated synthetic data for training. GPT-3.5/4
        was prompted to generate textbook-style explanations, exercises with solutions, and
        step-by-step reasoning chains. This "data distillation" transfers knowledge from a larger
        model into training data for a smaller one.
      </p>

      <PythonCode
        title="phi_model_usage.py"
        code={`from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load Phi-3-mini (3.8B parameters)
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Check model size
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e9:.2f}B")
print(f"Memory footprint: {total_params * 2 / 1e9:.2f} GB (float16)")

# Phi-3 uses ChatML format
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function for binary search."},
]

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = pipe(messages, max_new_tokens=200, temperature=0.1, do_sample=True)
print(output[0]["generated_text"][-1]["content"])

# Compare model sizes for perspective
models = {
    "Phi-3-mini": 3.8e9,
    "Mistral-7B": 7.3e9,
    "LLaMA-2-13B": 13e9,
    "LLaMA-2-70B": 70e9,
}
print("\\nModel size comparison:")
for name, params in models.items():
    ratio = params / 3.8e9
    print(f"  {name}: {params/1e9:.1f}B ({ratio:.1f}x Phi-3-mini)")`}
        id="code-phi-usage"
      />

      <NoteBlock
        type="intuition"
        title="Why Synthetic Data Works"
        content="Web text contains noise, irrelevant content, and poor explanations. Synthetic textbook data provides clear, structured explanations with consistent quality. Think of it as the difference between learning from random blog posts versus a well-written textbook -- the same amount of reading yields very different learning outcomes."
        id="note-synthetic-intuition"
      />

      <WarningBlock
        title="Benchmark Contamination Concerns"
        content="A recurring criticism of Phi models is potential benchmark contamination: if GPT-4-generated training data inadvertently includes content similar to benchmark test sets, performance may be inflated. Microsoft has addressed this with contamination analysis, but the concern highlights the difficulty of evaluating models trained on synthetic data derived from models that may have seen the benchmarks."
        id="warning-contamination"
      />

      <NoteBlock
        type="tip"
        title="Running Phi on Consumer Hardware"
        content="Phi-3-mini at 3.8B parameters requires only ~7.6GB in float16 or ~2GB when quantized to 4-bit. This makes it runnable on most modern laptops and even some phones. Use the GGUF format with llama.cpp or MLX for best performance on consumer devices."
        id="note-phi-hardware"
      />
    </div>
  )
}
