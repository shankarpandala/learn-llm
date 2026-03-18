import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LlamaFactory() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LLaMA-Factory: GUI-Based Finetuning</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        LLaMA-Factory provides a web-based GUI (LLaMA Board) for configuring and running finetuning
        jobs without writing code. It supports 100+ LLM architectures, all major training methods,
        and integrates dataset management, training, evaluation, and deployment.
      </p>

      <DefinitionBlock
        title="LLaMA-Factory"
        definition="LLaMA-Factory is an open-source framework for efficient finetuning of 100+ LLMs. It features a Gradio-based web UI (LLaMA Board) for no-code finetuning, CLI for scripted workflows, and supports SFT, RLHF, DPO, and PPO with LoRA/QLoRA/full finetuning."
        id="def-llama-factory"
      />

      <h2 className="text-2xl font-semibold">Installation and Setup</h2>

      <PythonCode
        title="install_llama_factory.sh"
        code={`# Install LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# For QLoRA support
pip install bitsandbytes

# For Flash Attention
pip install flash-attn --no-build-isolation

# Launch the web UI (LLaMA Board)
llamafactory-cli webui

# This opens a Gradio interface at http://localhost:7860
# Features:
# - Model selection dropdown
# - Dataset browser and preview
# - Training method selection (SFT/DPO/PPO/ORPO)
# - LoRA/QLoRA configuration
# - Real-time training metrics
# - Chat interface for evaluation`}
        id="code-install-lf"
      />

      <h2 className="text-2xl font-semibold">CLI-Based Training</h2>

      <PythonCode
        title="llama_factory_cli.sh"
        code={`# Train via CLI (equivalent to GUI configuration)
llamafactory-cli train \\
    --stage sft \\
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \\
    --dataset alpaca_en \\
    --template llama3 \\
    --finetuning_type lora \\
    --lora_rank 16 \\
    --lora_alpha 32 \\
    --lora_target all \\
    --output_dir ./llama3-sft \\
    --per_device_train_batch_size 2 \\
    --gradient_accumulation_steps 4 \\
    --learning_rate 2e-4 \\
    --num_train_epochs 2 \\
    --quantization_bit 4 \\
    --bf16 true \\
    --flash_attn fa2 \\
    --logging_steps 10

# Or use a JSON config file
llamafactory-cli train examples/train_lora/llama3_lora_sft.json

# Chat with the finetuned model
llamafactory-cli chat \\
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \\
    --adapter_name_or_path ./llama3-sft \\
    --template llama3 \\
    --finetuning_type lora \\
    --quantization_bit 4

# Export merged model
llamafactory-cli export \\
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \\
    --adapter_name_or_path ./llama3-sft \\
    --template llama3 \\
    --finetuning_type lora \\
    --export_dir ./llama3-merged \\
    --export_size 2  # Shard size in GB`}
        id="code-lf-cli"
      />

      <h2 className="text-2xl font-semibold">Custom Datasets in LLaMA-Factory</h2>

      <PythonCode
        title="custom_dataset_config.py"
        code={`# LLaMA-Factory uses a dataset_info.json to register datasets
# Add your custom dataset to data/dataset_info.json

dataset_config = {
    "my_custom_data": {
        "file_name": "my_data.json",    # In the data/ directory
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    },
    "my_sharegpt_data": {
        "file_name": "conversations.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    },
}

import json
print(json.dumps(dataset_config, indent=2))

# Data format for alpaca-style:
# [{"instruction": "...", "input": "...", "output": "..."}, ...]

# Data format for sharegpt-style:
# [{"conversations": [
#     {"from": "human", "value": "..."},
#     {"from": "gpt", "value": "..."}
# ]}, ...]`}
        id="code-custom-dataset"
      />

      <ExampleBlock
        title="LLaMA-Factory vs Axolotl"
        problem="When should you choose LLaMA-Factory over Axolotl?"
        steps={[
          { formula: '\\text{LLaMA-Factory: Web UI, beginners}', explanation: 'Better for newcomers who prefer a GUI and visual configuration.' },
          { formula: '\\text{Axolotl: YAML, advanced users}', explanation: 'Better for reproducible pipelines, CI/CD, and complex multi-dataset configs.' },
          { formula: '\\text{LLaMA-Factory: built-in chat eval}', explanation: 'Integrated chat interface for quick qualitative evaluation.' },
          { formula: '\\text{Axolotl: DeepSpeed integration}', explanation: 'Better multi-GPU scaling with native DeepSpeed/FSDP support.' },
        ]}
        id="example-lf-vs-axolotl"
      />

      <NoteBlock
        type="tip"
        title="Template System"
        content="LLaMA-Factory uses a template system to handle chat formatting for different models. Use --template llama3 for LLaMA 3, --template mistral for Mistral, --template chatml for Qwen. The template ensures correct special tokens are applied."
        id="note-templates"
      />

      <WarningBlock
        title="GUI vs CLI Consistency"
        content="Settings configured in the GUI may not persist between sessions. For reproducibility, always export your GUI configuration as a JSON file and use the CLI for production training runs. The GUI is best for experimentation and quick iterations."
        id="warning-gui-persistence"
      />
    </div>
  )
}
