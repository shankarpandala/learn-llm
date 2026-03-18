import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Quantization() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Quantization Levels in Ollama</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Quantization reduces model size by representing weights with fewer bits. Ollama models come
        in various quantization levels, each offering a different tradeoff between quality, speed,
        and memory usage. Understanding these helps you choose the right variant.
      </p>

      <DefinitionBlock
        title="GGUF Quantization Types"
        definition="GGUF supports multiple quantization schemes. The naming convention is Q{bits}_{type}: Q4_0 (basic 4-bit), Q4_K_M (4-bit with k-quant medium), Q5_K_M (5-bit k-quant medium), Q8_0 (8-bit), and F16 (16-bit float). K-quants use importance-based mixed precision."
        id="def-quantization"
      />

      <ExampleBlock
        title="Quantization Comparison for LLaMA-3.1-8B"
        problem="Compare size, speed, and quality across quantization levels."
        steps={[
          { formula: 'Q4_0: 4.3 GB — basic 4-bit, fastest, lowest quality', explanation: 'Simple round-to-nearest quantization. Noticeable quality loss.' },
          { formula: 'Q4_K_M: 4.9 GB — recommended default', explanation: 'K-quant keeps important weights at higher precision. Best quality/size ratio.' },
          { formula: 'Q5_K_M: 5.7 GB — higher quality 5-bit', explanation: 'Barely noticeable quality loss. Good if you have the extra RAM.' },
          { formula: 'Q8_0: 8.5 GB — near lossless 8-bit', explanation: 'Almost indistinguishable from fp16 for most tasks.' },
          { formula: 'F16: 16 GB — full fp16 precision', explanation: 'Maximum quality, but requires 2-4x more memory.' },
        ]}
        id="example-comparison"
      />

      <PythonCode
        title="Terminal"
        code={`# Pull different quantization levels of the same model
ollama pull llama3.1:8b-instruct-q4_0
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull llama3.1:8b-instruct-q5_K_M
ollama pull llama3.1:8b-instruct-q8_0

# Compare sizes
ollama list | grep llama3.1
# llama3.1:8b-instruct-q4_0    4.3 GB
# llama3.1:8b-instruct-q4_K_M  4.9 GB
# llama3.1:8b-instruct-q5_K_M  5.7 GB
# llama3.1:8b-instruct-q8_0    8.5 GB

# Quantize your own GGUF with llama.cpp
# Available types: Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_1,
#   Q4_K_S, Q4_K_M, Q5_0, Q5_1, Q5_K_S, Q5_K_M, Q6_K, Q8_0, F16, F32
./llama-quantize model-f16.gguf model-q4_K_M.gguf Q4_K_M`}
        id="code-quant-levels"
      />

      <PythonCode
        title="benchmark_quants.py"
        code={`import ollama
import time

MODELS = [
    "llama3.1:8b-instruct-q4_0",
    "llama3.1:8b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q8_0",
]

PROMPTS = [
    "What is the derivative of x^3 + 2x?",
    "Write a Python quicksort implementation.",
    "Explain the difference between TCP and UDP.",
]

for model in MODELS:
    total_tokens = 0
    total_time = 0

    for prompt in PROMPTS:
        start = time.time()
        resp = ollama.generate(model=model, prompt=prompt)
        elapsed = time.time() - start

        total_tokens += resp.get("eval_count", 0)
        total_time += elapsed

    speed = total_tokens / total_time if total_time > 0 else 0
    print(f"{model:<40} {speed:>6.1f} tok/s  ({total_tokens} tokens)")

# Typical results (Apple M2 Pro):
# q4_0:   ~45 tok/s  (fastest but lower quality)
# q4_K_M: ~40 tok/s  (best balance)
# q8_0:   ~25 tok/s  (highest quality, slower)`}
        id="code-benchmark"
      />

      <NoteBlock
        type="tip"
        title="The Sweet Spot: Q4_K_M"
        content="For most users, Q4_K_M is the best default. It uses importance-based mixed precision (keeping attention and output layers at higher precision) and produces nearly the same quality as Q5_K_M at smaller size. This is why Ollama uses Q4_K_M as the default tag for most models."
        id="note-sweet-spot"
      />

      <WarningBlock
        title="Quality Degrades Below Q4"
        content="Q3_K and Q2_K quantizations show significant quality degradation, especially for reasoning and math tasks. Avoid going below Q4 unless your use case only involves simple text generation or you are severely memory-constrained."
        id="warning-quality"
      />

      <NoteBlock
        type="note"
        title="Bigger Model, Lower Quant vs Smaller Model, Higher Quant"
        content="A 13B model at Q4_K_M often outperforms a 7B model at Q8_0 despite similar memory usage. Larger models are more robust to quantization. When memory is fixed, prefer a bigger model with more aggressive quantization over a smaller model at higher precision."
        id="note-bigger-better"
      />
    </div>
  )
}
