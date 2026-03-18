import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function MergingLora() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Merging LoRA Weights into Base Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        After finetuning with LoRA, you have a small adapter file (~100 MB) that must be combined
        with the base model for inference. Merging the LoRA weights into the base model eliminates
        inference overhead and produces a standalone model that can be deployed like any other.
      </p>

      <DefinitionBlock
        title="LoRA Merging"
        definition="LoRA merging computes the combined weight $W_{\\text{merged}} = W_0 + \\frac{\\alpha}{r} B A$ and saves it as a regular model. After merging, the model no longer needs the PEFT library and has zero inference overhead from the adaptation."
        notation="W_{merged} = W_0 + \frac{\alpha}{r} BA"
        id="def-lora-merge"
      />

      <h2 className="text-2xl font-semibold">Merging Methods</h2>

      <PythonCode
        title="merge_lora_weights.py"
        code={`from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Method 1: Standard PEFT merge
base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter_path = "./my-lora-adapter"

# Load base model in fp16
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="cpu",  # Merge on CPU to avoid VRAM issues
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load and merge LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()  # Merge weights and remove PEFT layers

# Save merged model
model.save_pretrained("./merged-model", safe_serialization=True)
tokenizer.save_pretrained("./merged-model")
print("Merged model saved!")

# Verify: model should have no PEFT modules
print(f"Model type: {type(model).__name__}")
# Should be LlamaForCausalLM, not PeftModelForCausalLM`}
        id="code-merge-standard"
      />

      <PythonCode
        title="merge_with_unsloth.py"
        code={`from unsloth import FastLanguageModel

# Method 2: Unsloth merge (handles everything automatically)
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Load your trained adapter
model.load_adapter("./my-lora-adapter")

# Save merged model in different formats
# Option A: Merged safetensors (for HF ecosystem)
model.save_pretrained_merged(
    "./merged-16bit",
    tokenizer,
    save_method="merged_16bit",  # Full fp16
)

# Option B: Merged + quantized to 4-bit (for HF + bitsandbytes)
model.save_pretrained_merged(
    "./merged-4bit",
    tokenizer,
    save_method="merged_4bit",
)

# Option C: Direct to GGUF (for llama.cpp)
model.save_pretrained_gguf(
    "./merged-gguf",
    tokenizer,
    quantization_method="q4_k_m",
)

# Option D: Push merged model to HuggingFace Hub
model.push_to_hub_merged(
    "your-username/model-name",
    tokenizer,
    save_method="merged_16bit",
)`}
        id="code-merge-unsloth"
      />

      <ExampleBlock
        title="Merging QLoRA Adapters"
        problem="How to merge a QLoRA adapter (trained on 4-bit base) into a full-precision model?"
        steps={[
          { formula: '\\text{Load base model in fp16 (not 4-bit)}', explanation: 'The merged model should be full precision for quality. Load base on CPU if GPU lacks VRAM.' },
          { formula: '\\text{Load LoRA adapter onto fp16 base}', explanation: 'PeftModel.from_pretrained handles the dtype mismatch automatically.' },
          { formula: '\\text{merge\\_and\\_unload()}', explanation: 'Compute W_merged = W_base + (alpha/r) * B @ A for each adapted layer.' },
          { formula: '\\text{Save as safetensors}', explanation: 'The result is a standard model with no PEFT dependency.' },
        ]}
        id="example-qlora-merge"
      />

      <NoteBlock
        type="tip"
        title="Merge on CPU"
        content="Merging requires loading both the base model and adapter in fp16, which may exceed GPU VRAM. Use device_map='cpu' to merge on CPU RAM (needs ~2x model size in RAM: ~16 GB for a 7B model). This is a one-time operation so speed is not critical."
        id="note-merge-cpu"
      />

      <WarningBlock
        title="Do Not Merge Quantized Base Weights"
        content="Never merge LoRA adapters into a 4-bit quantized base model. The quantization noise in the base weights will permanently degrade quality. Always merge into the fp16/bf16 base model, then quantize the merged model separately if needed."
        id="warning-quant-merge"
      />

      <NoteBlock
        type="note"
        title="Multiple LoRA Adapters"
        content="You can merge multiple LoRA adapters sequentially (e.g., SFT adapter then DPO adapter). Load and merge one at a time. The order matters: merge SFT first, then DPO on top. Alternatively, some tools support adapter arithmetic for combining adapters with weighted sums."
        id="note-multi-adapter"
      />
    </div>
  )
}
