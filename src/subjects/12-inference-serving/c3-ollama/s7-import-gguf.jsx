import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ImportGGUF() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Importing Custom GGUF Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        While Ollama's library covers popular models, you may want to run models from Hugging Face
        or your own quantized models. Ollama supports importing any GGUF file, the standard format
        for quantized models used by llama.cpp.
      </p>

      <DefinitionBlock
        title="GGUF Format"
        definition="GGUF (GPT-Generated Unified Format) is a binary file format for storing quantized LLM weights and metadata. It is the successor to GGML and is the native format used by llama.cpp and Ollama. A single .gguf file contains the full model: weights, tokenizer, and configuration."
        id="def-gguf"
      />

      <PythonCode
        title="Terminal"
        code={`# Step 1: Download a GGUF from Hugging Face
# Many users upload GGUF quantizations (e.g., TheBloke, bartowski)
pip install huggingface-hub

# Download a specific GGUF file
huggingface-cli download \\
    bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \\
    Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \\
    --local-dir ./models

# Step 2: Create a Modelfile pointing to the GGUF
cat > Modelfile << 'EOF'
FROM ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Set appropriate chat template for LLaMA 3
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
EOF

# Step 3: Import into Ollama
ollama create my-llama3 -f Modelfile
# transferring model data
# using existing layer sha256:abc123...
# creating model layer
# writing manifest
# success

# Step 4: Run it
ollama run my-llama3 "Hello, who are you?"`}
        id="code-import-gguf"
      />

      <PythonCode
        title="convert_to_gguf.py"
        code={`# Convert a Hugging Face model to GGUF using llama.cpp
# First, clone llama.cpp and install requirements
# git clone https://github.com/ggerganov/llama.cpp
# pip install -r llama.cpp/requirements.txt

import subprocess
import os

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "./converted"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Download model from Hugging Face
from huggingface_hub import snapshot_download
model_path = snapshot_download(MODEL_ID, local_dir=f"{OUTPUT_DIR}/hf-model")

# Step 2: Convert to GGUF (fp16)
subprocess.run([
    "python", "llama.cpp/convert_hf_to_gguf.py",
    f"{OUTPUT_DIR}/hf-model",
    "--outfile", f"{OUTPUT_DIR}/model-f16.gguf",
    "--outtype", "f16",
], check=True)

# Step 3: Quantize to Q4_K_M
subprocess.run([
    "./llama.cpp/build/bin/llama-quantize",
    f"{OUTPUT_DIR}/model-f16.gguf",
    f"{OUTPUT_DIR}/model-q4_K_M.gguf",
    "Q4_K_M",
], check=True)

print(f"Quantized model: {OUTPUT_DIR}/model-q4_K_M.gguf")
size_mb = os.path.getsize(f"{OUTPUT_DIR}/model-q4_K_M.gguf") / (1024**2)
print(f"Size: {size_mb:.0f} MB")`}
        id="code-convert"
      />

      <ExampleBlock
        title="Common GGUF Sources"
        problem="Where to find pre-quantized GGUF files?"
        steps={[
          { formula: 'Hugging Face: search for \"GGUF\" in model names', explanation: 'Users like bartowski, TheBloke, and others upload quantized versions of popular models.' },
          { formula: 'Ollama library: all models are GGUF internally', explanation: 'Ollama downloads are GGUF files stored in ~/.ollama/models/blobs/.' },
          { formula: 'Self-quantize: use llama.cpp convert scripts', explanation: 'Convert any Hugging Face safetensors model to GGUF with custom quantization.' },
        ]}
        id="example-sources"
      />

      <NoteBlock
        type="tip"
        title="Check Metadata Before Importing"
        content="Use 'llama.cpp/build/bin/llama-gguf-info model.gguf' to inspect a GGUF file's metadata: architecture, quantization type, context length, tokenizer info. This helps you set the correct template and parameters in your Modelfile."
        id="note-metadata"
      />

      <WarningBlock
        title="Chat Template Must Match"
        content="When importing a GGUF, Ollama cannot always auto-detect the correct chat template. If your imported model gives garbled output, the most likely cause is a wrong or missing TEMPLATE in the Modelfile. Check the original model's documentation for the expected format."
        id="warning-template"
      />
    </div>
  )
}
