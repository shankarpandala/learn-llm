import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function HuggingfaceHub() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Pushing Models to Hugging Face Hub</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The Hugging Face Hub is the standard platform for sharing and distributing models. Uploading
        your finetuned model makes it accessible to others and enables easy deployment. This section
        covers uploading models, creating model cards, and managing repositories.
      </p>

      <DefinitionBlock
        title="Hugging Face Hub"
        definition="The Hugging Face Hub is a platform hosting over 500,000 models, 100,000 datasets, and 100,000 demo applications. Model repositories use Git LFS for large file storage, support versioning, and include model cards for documentation."
        id="def-hf-hub"
      />

      <h2 className="text-2xl font-semibold">Authentication and Setup</h2>

      <PythonCode
        title="hf_hub_setup.sh"
        code={`# Install the Hub CLI
pip install huggingface_hub

# Login (interactive - opens browser)
huggingface-cli login

# Or set token directly
huggingface-cli login --token hf_xxxxxxxxxxxxx

# Or via environment variable
export HF_TOKEN=hf_xxxxxxxxxxxxx

# Verify authentication
huggingface-cli whoami`}
        id="code-hf-setup"
      />

      <h2 className="text-2xl font-semibold">Pushing Models</h2>

      <PythonCode
        title="push_model_to_hub.py"
        code={`from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, create_repo
import torch

# Method 1: Using model.push_to_hub()
model_path = "./my-finetuned-model"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Push to Hub (creates repo if it doesn't exist)
repo_name = "your-username/llama3-my-task"
model.push_to_hub(repo_name, safe_serialization=True)
tokenizer.push_to_hub(repo_name)

# Method 2: Push LoRA adapter only (much smaller)
# from peft import PeftModel
# model = PeftModel.from_pretrained(base_model, "./lora-adapter")
# model.push_to_hub("your-username/llama3-lora-adapter")

# Method 3: Upload arbitrary files (GGUF, configs, etc.)
api = HfApi()

# Create repository
create_repo(repo_name, exist_ok=True, repo_type="model")

# Upload GGUF file
api.upload_file(
    path_or_fileobj="./model-q4_k_m.gguf",
    path_in_repo="model-q4_k_m.gguf",
    repo_id=repo_name,
)

# Upload entire directory
api.upload_folder(
    folder_path="./merged-model",
    repo_id=repo_name,
    commit_message="Upload merged fp16 model",
)

print(f"Model uploaded to: https://huggingface.co/{repo_name}")`}
        id="code-push-model"
      />

      <PythonCode
        title="create_model_card.py"
        code={`# Create a model card (README.md) for your model
model_card = """---
language:
- en
license: apache-2.0
tags:
- llama
- finetuned
- qlora
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
datasets:
- yahma/alpaca-cleaned
pipeline_tag: text-generation
---

# My Finetuned LLaMA 3.1 8B

## Description
This model is a QLoRA finetune of Meta-Llama-3.1-8B-Instruct
on the Alpaca-cleaned dataset for improved instruction following.

## Training Details
- **Method**: QLoRA (4-bit NF4 + LoRA rank 16)
- **Dataset**: alpaca-cleaned (51K examples)
- **Epochs**: 2
- **Learning rate**: 2e-4
- **Hardware**: 1x RTX 4090 (24 GB)
- **Training time**: 3 hours
- **Framework**: Unsloth + TRL

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/model-name")
tokenizer = AutoTokenizer.from_pretrained("your-username/model-name")
```

## Evaluation
| Benchmark | Base Model | Finetuned |
|-----------|-----------|-----------|
| MMLU      | 68.2      | 69.1      |
| HellaSwag | 82.1      | 82.5      |
"""

# Save model card
with open("./merged-model/README.md", "w") as f:
    f.write(model_card)

# Upload model card
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="./merged-model/README.md",
    path_in_repo="README.md",
    repo_id="your-username/model-name",
)`}
        id="code-model-card"
      />

      <ExampleBlock
        title="Repository Best Practices"
        problem="What should a well-organized model repository contain?"
        steps={[
          { formula: '\\text{README.md: model card with training details}', explanation: 'Description, training config, usage examples, evaluation results.' },
          { formula: '\\text{Model files: safetensors + config.json}', explanation: 'The model weights and architecture configuration.' },
          { formula: '\\text{Tokenizer files: tokenizer.json + special tokens}', explanation: 'Everything needed to reconstruct the tokenizer.' },
          { formula: '\\text{Optional: GGUF files for local inference}', explanation: 'Upload Q4_K_M and Q8_0 GGUF variants for llama.cpp users.' },
        ]}
        id="example-repo-structure"
      />

      <NoteBlock
        type="tip"
        title="Private Repositories"
        content="For proprietary models, create private repos: create_repo(repo_name, private=True). Team members can be granted access through organization settings. Private repos work the same as public ones but require authentication to download."
        id="note-private-repos"
      />

      <WarningBlock
        title="Check for Sensitive Data"
        content="Before pushing to Hub, ensure your model does not contain sensitive information embedded in the weights (e.g., from training on private data). Also check that no credentials, API keys, or personal data are included in config files or the model card."
        id="warning-sensitive-data"
      />
    </div>
  )
}
