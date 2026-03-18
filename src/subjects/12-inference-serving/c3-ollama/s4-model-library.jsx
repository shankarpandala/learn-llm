import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ModelLibrary() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Ollama Model Library & Tags</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The Ollama model library at ollama.com/library hosts hundreds of pre-quantized models
        ready to run. Understanding the library structure helps you pick the right model for your
        hardware and use case.
      </p>

      <DefinitionBlock
        title="Ollama Model Library"
        definition="A curated registry of GGUF-quantized models hosted by Ollama. Each model entry includes multiple tags for different sizes (1B, 3B, 7B, 70B) and quantization levels (q4_0, q4_K_M, q8_0, fp16). The default tag is typically the best quality-to-size ratio."
        id="def-library"
      />

      <PythonCode
        title="Terminal"
        code={`# Browse the library from CLI
ollama list   # shows local models

# Popular model families and their sizes:
#
# General Purpose:
#   llama3.2:1b     (1.3GB)  - Small, fast, good for simple tasks
#   llama3.2:3b     (2.0GB)  - Good balance for edge devices
#   llama3.1:8b     (4.7GB)  - Strong general-purpose model
#   llama3.1:70b    (40GB)   - Near GPT-4 quality
#
# Code:
#   codellama:7b    (3.8GB)  - Code generation
#   deepseek-coder-v2:16b (8.9GB) - Strong code model
#   qwen2.5-coder:7b (4.7GB) - Alibaba code model
#
# Small & Fast:
#   phi3:mini       (2.2GB)  - Microsoft, punches above weight
#   gemma2:2b       (1.6GB)  - Google, very capable for size
#   qwen2.5:0.5b    (0.4GB)  - Tiny but functional
#
# Reasoning:
#   deepseek-r1:8b  (4.9GB)  - Chain-of-thought reasoning
#   qwq:32b         (20GB)   - Strong reasoning
#
# Vision:
#   llava:7b        (4.7GB)  - Image understanding
#   llama3.2-vision:11b (7.9GB) - LLaMA vision

# See all available tags for a model
ollama show llama3.2 --list`}
        id="code-library"
      />

      <ExampleBlock
        title="Choosing the Right Model"
        problem="You have a laptop with 16GB RAM and integrated GPU. Which model should you run?"
        steps={[
          { formula: 'Available RAM for model: ~10GB (OS uses 4-6GB)', explanation: 'Leave headroom for the operating system and other applications.' },
          { formula: 'Best options: llama3.2:3b (2GB) or llama3.1:8b-q4_0 (4.7GB)', explanation: 'Both fit comfortably. The 8B model is significantly smarter.' },
          { formula: 'For coding: qwen2.5-coder:7b (4.7GB)', explanation: 'Specialized code models outperform general models on code tasks.' },
          { formula: 'Avoid: any 70B model (40GB+)', explanation: 'Will not fit in memory and will be extremely slow with CPU-only inference.' },
        ]}
        id="example-choosing"
      />

      <PythonCode
        title="compare_models.py"
        code={`import ollama
import time

models_to_test = ["llama3.2:1b", "llama3.2:3b", "phi3:mini"]
prompt = "Write a Python function to find the nth Fibonacci number."

for model_name in models_to_test:
    try:
        start = time.time()
        response = ollama.generate(model=model_name, prompt=prompt)
        elapsed = time.time() - start

        tokens = response.get("eval_count", 0)
        speed = tokens / elapsed if elapsed > 0 else 0

        print(f"\\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Time: {elapsed:.1f}s | Tokens: {tokens} | Speed: {speed:.0f} tok/s")
        print(f"Response preview: {response['response'][:200]}...")
    except Exception as e:
        print(f"{model_name}: {e} (not downloaded?)")`}
        id="code-compare"
      />

      <NoteBlock
        type="tip"
        title="Model Naming Convention"
        content="The pattern is: family:size-variant-quantization. If you just specify 'ollama pull llama3.2', you get the default tag which is usually the instruction-tuned version with q4_K_M quantization -- the best balance of quality and size."
        id="note-naming"
      />

      <WarningBlock
        title="Model Licenses Vary"
        content="Not all models on Ollama's library are fully open. LLaMA models have Meta's community license, Gemma has Google's terms, and some models restrict commercial use. Always check the model's license before deploying in production."
        id="warning-licenses"
      />
    </div>
  )
}
