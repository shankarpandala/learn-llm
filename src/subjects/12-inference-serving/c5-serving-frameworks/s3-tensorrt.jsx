import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function TensorRT() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">TensorRT-LLM</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        TensorRT-LLM is NVIDIA's high-performance inference library that compiles LLM
        architectures into optimized TensorRT engines. It squeezes maximum throughput from
        NVIDIA GPUs through kernel fusion, quantization, in-flight batching, and custom CUDA
        kernels. When absolute performance on NVIDIA hardware matters, TensorRT-LLM is the
        gold standard.
      </p>

      <DefinitionBlock
        title="TensorRT-LLM"
        definition="TensorRT-LLM is NVIDIA's open-source library for compiling and serving LLMs. It converts model weights into optimized TensorRT engines with fused kernels, FP8/INT4 quantization, paged KV-cache, in-flight batching, and multi-GPU tensor/pipeline parallelism. Models must be compiled before serving."
        id="def-tensorrt"
      />

      <PythonCode
        title="Terminal"
        code={`# Pull the TensorRT-LLM Docker image
docker pull nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3

# Clone TensorRT-LLM for build scripts
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM/examples/llama

# Step 1: Convert HuggingFace checkpoint
python convert_checkpoint.py \\
    --model_dir meta-llama/Llama-3.1-8B-Instruct \\
    --output_dir ./checkpoint \\
    --dtype float16

# Step 2: Build the TensorRT engine
trtllm-build \\
    --checkpoint_dir ./checkpoint \\
    --output_dir ./engine \\
    --gemm_plugin float16 \\
    --max_input_len 4096 \\
    --max_seq_len 8192 \\
    --max_batch_size 64

# Step 3: Run with Triton Inference Server
docker run --gpus all -p 8000:8000 -p 8001:8001 \\
    -v $(pwd)/engine:/engines \\
    nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3 \\
    tritonserver --model-repository=/engines

# Quantize to FP8 for Hopper GPUs (H100)
python convert_checkpoint.py \\
    --model_dir meta-llama/Llama-3.1-8B-Instruct \\
    --output_dir ./checkpoint_fp8 \\
    --dtype float16 \\
    --qformat fp8 \\
    --calib_size 512`}
        id="code-tensorrt-build"
      />

      <PythonCode
        title="tensorrt_client.py"
        code={`import requests
import json

# Triton HTTP endpoint
TRITON_URL = "http://localhost:8000/v2/models/llama/generate"

def generate(prompt, max_tokens=128):
    """Send request to TensorRT-LLM via Triton."""
    payload = {
        "text_input": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": False,
    }
    resp = requests.post(TRITON_URL, json=payload)
    resp.raise_for_status()
    return resp.json()["text_output"]

result = generate("Explain TensorRT-LLM in two sentences.")
print(result)

# Streaming via Server-Sent Events
def stream_generate(prompt, max_tokens=128):
    payload = {
        "text_input": prompt,
        "max_tokens": max_tokens,
        "stream": True,
    }
    with requests.post(TRITON_URL, json=payload, stream=True) as resp:
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                print(data.get("text_output", ""), end="", flush=True)
    print()

stream_generate("List 5 benefits of model compilation.")`}
        id="code-tensorrt-client"
      />

      <ExampleBlock
        title="TensorRT-LLM Optimization Pipeline"
        problem="What happens during TensorRT-LLM engine compilation?"
        steps={[
          { formula: 'Weight conversion: HF -> TRT checkpoint', explanation: 'Rearranges weights into the format TensorRT expects.' },
          { formula: 'Kernel fusion: combine multiple ops into one', explanation: 'Fuses attention, LayerNorm, and activation into single GPU kernels.' },
          { formula: 'Quantization: FP16 -> FP8 / INT4-AWQ', explanation: 'Reduces memory and increases compute throughput on supported hardware.' },
          { formula: 'Engine build: compile optimized execution plan', explanation: 'Generates a GPU-specific binary optimized for the target hardware.' },
        ]}
        id="example-pipeline"
      />

      <NoteBlock
        type="note"
        title="Performance Advantage"
        content="TensorRT-LLM typically achieves 1.5-3x higher throughput than general-purpose frameworks on the same NVIDIA hardware. The tradeoff is a longer setup process: you must compile engines for each specific GPU architecture (e.g., A100 vs H100) and model configuration. Engines are not portable across GPU types."
        id="note-performance"
      />

      <WarningBlock
        title="Build Time & Complexity"
        content="Engine compilation can take 10-60 minutes depending on model size. The build is tied to specific GPU architecture, max batch size, and sequence length. Changing any of these requires a rebuild. For rapid experimentation, start with vLLM or TGI and switch to TensorRT-LLM once your deployment configuration is stable."
        id="warning-complexity"
      />
    </div>
  )
}
