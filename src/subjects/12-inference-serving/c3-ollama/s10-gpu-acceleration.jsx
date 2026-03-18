import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function GPUAcceleration() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">GPU Acceleration & Layer Offloading</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Ollama automatically detects GPUs and offloads model layers for acceleration. Understanding
        how GPU offloading works helps you optimize performance, especially when the model does not
        fully fit in GPU memory.
      </p>

      <DefinitionBlock
        title="Layer Offloading"
        definition="A transformer model consists of many layers (e.g., 32 for a 7B model). Each layer can be placed on GPU or CPU independently. Offloading $N$ of $L$ total layers to GPU means those layers run at GPU speed while the rest run on CPU. More GPU layers = faster inference."
        id="def-offloading"
      />

      <PythonCode
        title="Terminal"
        code={`# Check GPU detection
ollama ps
# NAME        ID           SIZE    PROCESSOR     UNTIL
# llama3.2    a80c4f17acd5 3.5 GB  100% GPU      4 min from now

# Environment variables for GPU control:

# Force number of GPU layers (0 = CPU only)
OLLAMA_GPU_LAYERS=20 ollama serve

# Specify which GPU to use (for multi-GPU systems)
CUDA_VISIBLE_DEVICES=0 ollama serve    # Use GPU 0 only
CUDA_VISIBLE_DEVICES=0,1 ollama serve  # Use GPUs 0 and 1

# Set GPU memory limit (leave room for KV cache)
OLLAMA_GPU_MEMORY=6g ollama serve      # Limit to 6GB GPU RAM

# Check NVIDIA GPU status
nvidia-smi
# Shows GPU utilization, memory usage, temperature

# Monitor GPU usage during generation
watch -n 0.5 nvidia-smi`}
        id="code-gpu-config"
      />

      <ExampleBlock
        title="Partial GPU Offloading"
        problem="You have a 6GB GPU and want to run a model that needs 8GB."
        steps={[
          { formula: 'Model: 32 layers, ~250MB per layer = 8GB total', explanation: 'The full model does not fit in 6GB of VRAM.' },
          { formula: 'Offload 22 layers to GPU: 22 \\times 250MB = 5.5GB', explanation: 'Leave ~500MB headroom for KV-cache and CUDA context.' },
          { formula: 'Remaining 10 layers run on CPU', explanation: 'CPU layers are slower but the GPU layers dominate throughput.' },
          { formula: 'Result: ~60% of GPU-only speed', explanation: 'Partial offloading is much faster than pure CPU inference.' },
        ]}
        id="example-partial"
      />

      <PythonCode
        title="gpu_benchmark.py"
        code={`import ollama
import time
import subprocess

def get_gpu_memory():
    """Get current GPU memory usage via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        used, total = result.stdout.strip().split(", ")
        return int(used), int(total)
    except Exception:
        return None, None

def benchmark_model(model_name, prompt, n_tokens=100):
    """Benchmark a model and report GPU usage."""
    used_before, total = get_gpu_memory()

    start = time.time()
    resp = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={"num_predict": n_tokens}
    )
    elapsed = time.time() - start

    used_after, _ = get_gpu_memory()
    tokens = resp.get("eval_count", 0)
    speed = tokens / elapsed if elapsed > 0 else 0

    print(f"Model: {model_name}")
    print(f"  Speed: {speed:.1f} tok/s")
    print(f"  GPU memory: {used_before}MB -> {used_after}MB (total: {total}MB)")
    print(f"  Processor: {resp.get('model', 'unknown')}")

prompt = "Write a detailed explanation of how neural networks learn."
benchmark_model("llama3.2", prompt)
benchmark_model("llama3.1:8b", prompt)`}
        id="code-benchmark"
      />

      <NoteBlock
        type="tip"
        title="Apple Silicon"
        content="On Apple Silicon Macs (M1/M2/M3/M4), Ollama uses Metal for GPU acceleration with unified memory. The GPU and CPU share the same RAM, so there is no data transfer overhead. A Mac with 32GB unified memory can run models that would need a 24GB+ discrete GPU."
        id="note-apple-silicon"
      />

      <WarningBlock
        title="VRAM Is Not Just for Weights"
        content="The model weights are only part of GPU memory usage. KV-cache grows with context length (hundreds of MB for long conversations), and CUDA/Metal context takes 300-500MB. Leave at least 1-2GB of headroom beyond the model size."
        id="warning-vram"
      />

      <NoteBlock
        type="note"
        title="Flash Attention in Ollama"
        content="Ollama benefits from flash attention implementations in llama.cpp, which reduce GPU memory usage and speed up attention computation. This is enabled automatically when your GPU supports it (CUDA compute capability 7.0+, all Apple Silicon)."
        id="note-flash-attention"
      />
    </div>
  )
}
