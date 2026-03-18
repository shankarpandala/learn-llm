import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ConvertingFormats() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Format Conversion Between Model Types</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Different inference engines require different model formats. Converting between safetensors,
        GGUF, GPTQ, and AWQ is a common post-training task. This section provides practical
        conversion workflows for every major format combination.
      </p>

      <h2 className="text-2xl font-semibold">Conversion Paths</h2>

      <ExampleBlock
        title="Format Conversion Map"
        problem="What are the conversion paths between model formats?"
        steps={[
          { formula: '\\text{Safetensors (fp16)} \\rightarrow \\text{GGUF}', explanation: 'Use llama.cpp convert_hf_to_gguf.py, then quantize with llama-quantize.' },
          { formula: '\\text{Safetensors (fp16)} \\rightarrow \\text{GPTQ}', explanation: 'Use auto-gptq or transformers GPTQConfig with calibration data.' },
          { formula: '\\text{Safetensors (fp16)} \\rightarrow \\text{AWQ}', explanation: 'Use autoawq library with calibration data.' },
          { formula: '\\text{GGUF} \\rightarrow \\text{Safetensors}', explanation: 'Use llama.cpp convert tools or llama-cpp-python for dequantization (lossy).' },
        ]}
        id="example-conversion-paths"
      />

      <PythonCode
        title="convert_safetensors_to_gguf.sh"
        code={`# Step 1: Clone llama.cpp (if not already)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt

# Step 2: Convert HF model to GGUF (fp16)
python convert_hf_to_gguf.py \\
    /path/to/your/merged-model \\
    --outfile model-f16.gguf \\
    --outtype f16

# Step 3: Quantize to desired level
# Build llama.cpp first
make -j$(nproc)

# Quantize
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
./llama-quantize model-f16.gguf model-q5_k_m.gguf Q5_K_M
./llama-quantize model-f16.gguf model-q8_0.gguf Q8_0

# Step 4: Test the quantized model
./llama-cli -m model-q4_k_m.gguf \\
    -p "Write a Python function to calculate fibonacci numbers." \\
    -n 256 --temp 0.7

# Available quantization types:
# Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M,
# Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0, F16, F32`}
        id="code-to-gguf"
      />

      <PythonCode
        title="format_conversion_utils.py"
        code={`import os
import shutil

def convert_model(input_path, output_path, target_format, quant_level="q4_k_m"):
    """Convert a model between formats."""

    print(f"Converting: {input_path} -> {target_format}")

    if target_format == "gguf":
        # Requires llama.cpp
        import subprocess
        llama_cpp = os.environ.get("LLAMA_CPP_PATH", "~/llama.cpp")

        # Step 1: Convert to fp16 GGUF
        fp16_path = output_path.replace(".gguf", "-f16.gguf")
        subprocess.run([
            "python", f"{llama_cpp}/convert_hf_to_gguf.py",
            input_path, "--outfile", fp16_path, "--outtype", "f16"
        ], check=True)

        # Step 2: Quantize
        subprocess.run([
            f"{llama_cpp}/llama-quantize",
            fp16_path, output_path, quant_level.upper()
        ], check=True)

        # Clean up fp16 intermediate
        if os.path.exists(fp16_path):
            os.remove(fp16_path)

    elif target_format == "gptq":
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
        import torch

        tokenizer = AutoTokenizer.from_pretrained(input_path)
        gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            input_path, quantization_config=gptq_config,
            torch_dtype=torch.float16, device_map="auto"
        )
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

    elif target_format == "awq":
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        model = AutoAWQForCausalLM.from_pretrained(input_path)
        tokenizer = AutoTokenizer.from_pretrained(input_path)
        model.quantize(tokenizer, quant_config={"w_bit": 4, "q_group_size": 128})
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)

    print(f"Conversion complete: {output_path}")

# Usage:
# convert_model("./merged-model", "./model.gguf", "gguf", "q4_k_m")
# convert_model("./merged-model", "./model-gptq", "gptq")
# convert_model("./merged-model", "./model-awq", "awq")`}
        id="code-conversion-utils"
      />

      <NoteBlock
        type="tip"
        title="Always Start from fp16"
        content="The golden rule: always convert from the fp16/bf16 merged model. Converting from one quantized format to another (e.g., GPTQ to GGUF) compounds quantization errors. Keep your fp16 merged model as the source of truth for all format conversions."
        id="note-start-fp16"
      />

      <WarningBlock
        title="Lossy Conversions"
        content="Converting from a quantized format back to fp16 (e.g., GGUF Q4 to safetensors) is lossy -- you cannot recover the original precision. The result will have the same quality as the quantized version, just in a different container. Only the original fp16 weights are truly full precision."
        id="warning-lossy"
      />

      <NoteBlock
        type="note"
        title="Verifying Conversions"
        content="After conversion, always verify the model works by running a few test prompts. Compare outputs between the original and converted model. Check for: garbled text (tokenizer issues), repetitive output (quantization too aggressive), or wrong language (metadata mismatch)."
        id="note-verify"
      />
    </div>
  )
}
