import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function GptqAwqExport() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">GPTQ and AWQ: GPU-Optimized Quantization</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        While GGUF targets CPU inference, GPTQ and AWQ are quantization methods optimized for
        GPU inference. They produce 4-bit models that run efficiently with libraries like
        vLLM, TGI, and the transformers library itself.
      </p>

      <DefinitionBlock
        title="GPTQ (GPT Quantization)"
        definition="GPTQ quantizes weights to 4-bit integers using a calibration dataset. It processes weights layer by layer, minimizing the output error using approximate second-order information (Hessian). The quantized model uses custom CUDA kernels for fast 4-bit matrix multiplication on GPUs."
        id="def-gptq"
      />

      <DefinitionBlock
        title="AWQ (Activation-Aware Weight Quantization)"
        definition="AWQ identifies the most important weights by analyzing activation magnitudes and protects them during quantization. It applies per-channel scaling to reduce quantization error for salient weights. AWQ is typically faster than GPTQ for quantization and produces slightly better quality at 4-bit."
        id="def-awq"
      />

      <PythonCode
        title="quantize_gptq.py"
        code={`from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch

model_name = "./your-finetuned-model"  # Merged fp16 model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure GPTQ quantization
gptq_config = GPTQConfig(
    bits=4,                    # 4-bit quantization
    dataset="c4",              # Calibration dataset
    tokenizer=tokenizer,
    group_size=128,            # Quantization group size
    desc_act=False,            # Disable desc_act for vLLM compat
    sym=True,                  # Symmetric quantization
)

# Load and quantize
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=gptq_config,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Save quantized model
model.save_pretrained("./model-gptq-4bit")
tokenizer.save_pretrained("./model-gptq-4bit")
print("GPTQ model saved!")`}
        id="code-gptq"
      />

      <PythonCode
        title="quantize_awq.py"
        code={`from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "./your-finetuned-model"

# Load model for AWQ quantization
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Configure AWQ
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",        # GEMM for general use, GEMV for batch=1
}

# Quantize (requires calibration data)
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data="pileval",      # Calibration dataset
)

# Save
model.save_quantized("./model-awq-4bit")
tokenizer.save_pretrained("./model-awq-4bit")
print("AWQ model saved!")

# Load and use the quantized model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "./model-awq-4bit",
    device_map="auto",
)
# Works with standard generate() API`}
        id="code-awq"
      />

      <ExampleBlock
        title="GPTQ vs AWQ vs GGUF"
        problem="When should you use each quantization method?"
        steps={[
          { formula: '\\text{GGUF: CPU inference, local deployment}', explanation: 'Use with llama.cpp, Ollama. Best for laptops and CPU-only servers.' },
          { formula: '\\text{GPTQ: GPU inference with vLLM/TGI}', explanation: 'Widely supported in GPU serving frameworks. Slower to quantize but well-tested.' },
          { formula: '\\text{AWQ: GPU inference, best quality}', explanation: 'Slightly better quality than GPTQ, faster quantization. Good vLLM support.' },
          { formula: '\\text{bitsandbytes NF4: training only}', explanation: 'Used during QLoRA training. Not ideal for production inference.' },
        ]}
        id="example-comparison"
      />

      <NoteBlock
        type="tip"
        title="vLLM Compatibility"
        content="For production GPU serving with vLLM, AWQ with GEMM kernel is recommended. It provides the best throughput and is well-supported. Ensure desc_act=False for GPTQ if targeting vLLM, as vLLM does not support activation reordering."
        id="note-vllm"
      />

      <WarningBlock
        title="Quantization Requires a Calibration Dataset"
        content="Both GPTQ and AWQ need a calibration dataset to determine optimal quantization parameters. The calibration data should be representative of your model's actual use case. Using generic calibration data (like C4) works but domain-specific calibration data yields better results."
        id="warning-calibration"
      />
    </div>
  )
}
