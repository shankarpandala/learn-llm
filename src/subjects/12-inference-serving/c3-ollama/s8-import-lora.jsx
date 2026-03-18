import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ImportLoRA() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Importing Fine-tuned LoRA Weights</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        If you have fine-tuned a model using LoRA, you can import those adapter weights into Ollama.
        This lets you serve your custom fine-tuned models with the same ease as any Ollama model,
        complete with API access and model management.
      </p>

      <DefinitionBlock
        title="LoRA Adapter in Ollama"
        definition="Ollama supports loading LoRA (Low-Rank Adaptation) weights on top of a base model. The adapter must be in GGUF format. At inference time, the adapter weights are merged with the base model, producing the fine-tuned behavior."
        id="def-lora-ollama"
      />

      <PythonCode
        title="Terminal"
        code={`# Step 1: Convert LoRA adapter to GGUF format
# If your adapter is from Hugging Face / PEFT format:
python llama.cpp/convert_lora_to_gguf.py \\
    --base ./base-model-hf \\
    --lora ./my-lora-adapter \\
    --outfile ./my-adapter.gguf

# Step 2: Create a Modelfile with the ADAPTER instruction
cat > Modelfile-lora << 'EOF'
# Base model (must match what the LoRA was trained on)
FROM llama3.1:8b

# Apply the LoRA adapter
ADAPTER ./my-adapter.gguf

# Optional: customize system prompt for your fine-tune
SYSTEM """You are a customer support agent for Acme Corp.
Answer questions about our products and policies helpfully."""

PARAMETER temperature 0.3
EOF

# Step 3: Build the model
ollama create acme-support -f Modelfile-lora

# Step 4: Test it
ollama run acme-support "What is your return policy?"

# Verify the model shows the adapter info
ollama show acme-support`}
        id="code-import-lora"
      />

      <ExampleBlock
        title="LoRA Import Workflow"
        problem="End-to-end: train a LoRA with Unsloth, deploy with Ollama."
        steps={[
          { formula: 'Train LoRA with Unsloth/PEFT on your dataset', explanation: 'Fine-tune produces adapter weights (adapter_model.safetensors).' },
          { formula: 'Convert adapter to GGUF: convert_lora_to_gguf.py', explanation: 'Transforms the safetensors LoRA into Ollama-compatible format.' },
          { formula: 'Create Modelfile with FROM base + ADAPTER path', explanation: 'The base model must match the model the LoRA was trained on.' },
          { formula: 'ollama create my-model -f Modelfile', explanation: 'Ollama merges the adapter and creates a servable model.' },
        ]}
        id="example-workflow"
      />

      <PythonCode
        title="full_lora_pipeline.py"
        code={`# Complete pipeline: export LoRA from PEFT and import to Ollama
import subprocess
import ollama

# Assume we have a PEFT LoRA adapter at ./lora-output/
LORA_PATH = "./lora-output"
BASE_MODEL_HF = "./base-model-hf"  # HF format base model
ADAPTER_GGUF = "./adapter.gguf"

# Step 1: Convert LoRA to GGUF
print("Converting LoRA adapter to GGUF...")
subprocess.run([
    "python", "llama.cpp/convert_lora_to_gguf.py",
    "--base", BASE_MODEL_HF,
    "--lora", LORA_PATH,
    "--outfile", ADAPTER_GGUF,
], check=True)

# Step 2: Create Ollama model with adapter
modelfile = f"""FROM llama3.1:8b
ADAPTER {ADAPTER_GGUF}

SYSTEM \"\"\"You are an AI assistant fine-tuned for medical Q&A.
Provide accurate, evidence-based answers. Always recommend
consulting a healthcare professional for medical decisions.\"\"\"

PARAMETER temperature 0.2
PARAMETER top_p 0.9
"""

print("Creating Ollama model...")
ollama.create(model="medical-qa", modelfile=modelfile)

# Step 3: Test the fine-tuned model
response = ollama.chat(
    model="medical-qa",
    messages=[{
        "role": "user",
        "content": "What are the common symptoms of type 2 diabetes?"
    }]
)
print(response["message"]["content"])`}
        id="code-pipeline"
      />

      <NoteBlock
        type="tip"
        title="Unsloth Direct Export"
        content="Unsloth can export directly to Ollama-compatible GGUF format using model.save_pretrained_gguf() and model.push_to_hub_gguf(). This skips the manual conversion step entirely -- the recommended approach if you use Unsloth for training."
        id="note-unsloth"
      />

      <WarningBlock
        title="Base Model Must Match"
        content="The LoRA adapter is only compatible with the exact base model it was trained on. A LoRA trained on LLaMA-3.1-8B cannot be applied to LLaMA-3.2-3B or Mistral-7B. If the base model does not match, you will get errors or nonsensical outputs."
        id="warning-base-match"
      />

      <NoteBlock
        type="note"
        title="Merged vs Adapter Serving"
        content="Ollama merges the LoRA weights into the base model at load time. This means inference speed is identical to the base model -- there is no overhead from having a separate adapter. The tradeoff is that you cannot hot-swap adapters at runtime."
        id="note-merged"
      />
    </div>
  )
}
