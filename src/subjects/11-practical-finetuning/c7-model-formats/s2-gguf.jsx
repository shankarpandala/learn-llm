import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function GgufFormat() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">GGUF Format for llama.cpp</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        GGUF (GPT-Generated Unified Format) is the file format used by llama.cpp for efficient
        CPU and GPU inference. It is the standard format for running quantized LLMs locally on
        consumer hardware, including laptops and desktops without dedicated GPUs.
      </p>

      <DefinitionBlock
        title="GGUF"
        definition="GGUF is a single-file binary format that contains model weights, tokenizer, and metadata. It supports various quantization levels (Q2_K through Q8_0) and is optimized for the llama.cpp inference engine. A single .gguf file is all you need to run a model."
        id="def-gguf"
      />

      <h2 className="text-2xl font-semibold">Quantization Levels</h2>

      <ExampleBlock
        title="GGUF Quantization Types"
        problem="What are the common GGUF quantization levels and their tradeoffs?"
        steps={[
          { formula: '\\text{Q2\\_K: } \\sim 2.5 \\text{ bits/weight}', explanation: 'Smallest size, significant quality loss. Only for very large models where nothing else fits.' },
          { formula: '\\text{Q4\\_K\\_M: } \\sim 4.8 \\text{ bits/weight}', explanation: 'Best balance of size and quality. Most popular choice for local inference.' },
          { formula: '\\text{Q5\\_K\\_M: } \\sim 5.7 \\text{ bits/weight}', explanation: 'Slightly better quality than Q4 with modest size increase.' },
          { formula: '\\text{Q8\\_0: } \\sim 8.5 \\text{ bits/weight}', explanation: 'Near-lossless quantization. Quality very close to fp16 at half the size.' },
        ]}
        id="example-quant-levels"
      />

      <PythonCode
        title="convert_to_gguf.py"
        code={`# Method 1: Using Unsloth (easiest)
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "your-finetuned-model-path",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Save as GGUF with different quantization levels
quant_methods = ["q4_k_m", "q5_k_m", "q8_0"]
for quant in quant_methods:
    model.save_pretrained_gguf(
        f"./model-{quant}",
        tokenizer,
        quantization_method=quant,
    )
    print(f"Saved {quant} GGUF")

# Method 2: Using llama.cpp directly
# git clone https://github.com/ggerganov/llama.cpp
# cd llama.cpp && make

# Convert safetensors to GGUF (fp16 first)
# python convert_hf_to_gguf.py /path/to/model --outfile model-f16.gguf

# Quantize to desired level
# ./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M`}
        id="code-convert-gguf"
      />

      <PythonCode
        title="gguf_size_estimation.py"
        code={`def estimate_gguf_size(params_billions, quant_type="q4_k_m"):
    """Estimate GGUF file size for different quantization types."""
    bits_per_weight = {
        "q2_k": 2.5,
        "q3_k_m": 3.4,
        "q4_0": 4.5,
        "q4_k_m": 4.8,
        "q5_0": 5.5,
        "q5_k_m": 5.7,
        "q6_k": 6.6,
        "q8_0": 8.5,
        "f16": 16.0,
    }

    bpw = bits_per_weight.get(quant_type, 4.8)
    size_gb = (params_billions * 1e9 * bpw) / (8 * 1e9)
    ram_needed = size_gb * 1.1  # 10% overhead for context

    return {"file_size_gb": size_gb, "ram_needed_gb": ram_needed}

# Compare sizes for common models
print(f"{'Model':>10} {'Quant':>8} {'Size':>8} {'RAM':>8}")
print("-" * 40)
for params in [7, 8, 13, 34, 70]:
    for quant in ["q4_k_m", "q5_k_m", "q8_0"]:
        result = estimate_gguf_size(params, quant)
        print(f"{params}B {quant:>8} {result['file_size_gb']:>7.1f}G "
              f"{result['ram_needed_gb']:>7.1f}G")

# Output:
#   7B   q4_k_m     4.2G     4.6G
#   7B   q5_k_m     5.0G     5.5G
#   7B   q8_0       7.4G     8.2G
#   70B  q4_k_m    42.0G    46.2G`}
        id="code-gguf-sizes"
      />

      <NoteBlock
        type="tip"
        title="Recommended Quantization"
        content="For most use cases, Q4_K_M offers the best quality-to-size ratio. Use Q5_K_M if you have extra RAM and want slightly better quality. Q8_0 is useful for evaluation and when quality is paramount. Avoid Q2_K unless the model absolutely does not fit otherwise."
        id="note-recommended-quant"
      />

      <WarningBlock
        title="Quantization Quality Loss"
        content="Quantization always introduces some quality degradation. Q4_K_M typically loses 1-3% on benchmarks compared to fp16. For critical applications, always evaluate the quantized model against the fp16 version on your specific tasks before deploying."
        id="warning-quant-loss"
      />

      <NoteBlock
        type="note"
        title="Running GGUF Models"
        content="GGUF files work with llama.cpp, Ollama, LM Studio, GPT4All, and many other local inference tools. To run: ollama create mymodel -f Modelfile (where Modelfile points to your .gguf). Or use llama.cpp: ./llama-cli -m model.gguf -p 'Your prompt here'."
        id="note-running-gguf"
      />
    </div>
  )
}
