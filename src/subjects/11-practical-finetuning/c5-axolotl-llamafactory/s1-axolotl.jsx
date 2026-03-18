import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Axolotl() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Axolotl: YAML-Driven Finetuning</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Axolotl is a flexible finetuning framework that configures entire training runs through
        YAML files. It supports multi-GPU training, dozens of model architectures, all PEFT
        methods, and advanced features like multi-dataset mixing and sample packing.
      </p>

      <DefinitionBlock
        title="Axolotl"
        definition="Axolotl is an open-source finetuning tool that wraps Hugging Face Transformers, PEFT, and DeepSpeed/FSDP behind a declarative YAML configuration. A single YAML file specifies the model, dataset, training method, and hyperparameters -- no Python coding required."
        id="def-axolotl"
      />

      <h2 className="text-2xl font-semibold">Installation</h2>

      <PythonCode
        title="install_axolotl.sh"
        code={`# Install Axolotl from source
git clone https://github.com/OpenAccess-AI-Collective/axolotl.git
cd axolotl
pip install packaging ninja
pip install -e '.[flash-attn,deepspeed]'

# Or use Docker (recommended for reproducibility)
docker pull winglian/axolotl:main-latest
docker run --gpus all -v $(pwd):/workspace \\
    winglian/axolotl:main-latest \\
    accelerate launch -m axolotl.cli.train /workspace/config.yml

# Verify installation
python -m axolotl.cli.train --help`}
        id="code-install-axolotl"
      />

      <h2 className="text-2xl font-semibold">YAML Configuration</h2>

      <PythonCode
        title="qlora_llama3.yml"
        code={`# Axolotl YAML config for QLoRA finetuning LLaMA 3.1 8B
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer

# Load in 4-bit for QLoRA
load_in_4bit: true
adapter: qlora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_linear: true    # Apply to all linear layers

# Dataset configuration
datasets:
  - path: yahma/alpaca-cleaned
    type: alpaca            # Built-in format handler
  - path: Open-Orca/OpenOrca
    type: sharegpt           # Multi-turn conversation format
    conversation: chatml
dataset_prepared_path: ./prepared-data
val_set_size: 0.05

# Training parameters
sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

num_epochs: 2
micro_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2e-4
lr_scheduler: cosine
warmup_steps: 10
optimizer: paged_adamw_8bit

# Precision
bf16: auto
tf32: true

# Memory optimization
gradient_checkpointing: true
flash_attention: true

# Output
output_dir: ./qlora-llama3-output
logging_steps: 10
save_strategy: steps
save_steps: 500

# Weights & Biases
wandb_project: axolotl-finetune
wandb_run_id: qlora-llama3-r16`}
        id="code-yaml-config"
      />

      <h2 className="text-2xl font-semibold">Running Training</h2>

      <PythonCode
        title="run_axolotl.sh"
        code={`# Preprocess the dataset (creates tokenized cache)
python -m axolotl.cli.preprocess config.yml

# Train on single GPU
python -m axolotl.cli.train config.yml

# Train on multiple GPUs with accelerate
accelerate launch --multi_gpu --num_processes 4 \\
    -m axolotl.cli.train config.yml

# Train with DeepSpeed ZeRO Stage 2
accelerate launch --config_file deepspeed_config.yml \\
    -m axolotl.cli.train config.yml

# Inference after training
python -m axolotl.cli.inference config.yml \\
    --lora_model_dir ./qlora-llama3-output

# Merge LoRA weights into base model
python -m axolotl.cli.merge_lora config.yml \\
    --lora_model_dir ./qlora-llama3-output`}
        id="code-run-axolotl"
      />

      <ExampleBlock
        title="Axolotl Dataset Types"
        problem="What dataset formats does Axolotl support natively?"
        steps={[
          { formula: '\\texttt{type: alpaca}', explanation: 'Alpaca format: instruction, input, output fields.' },
          { formula: '\\texttt{type: sharegpt}', explanation: 'ShareGPT format: multi-turn conversations with from/value pairs.' },
          { formula: '\\texttt{type: completion}', explanation: 'Raw text completion: just a text field for continued pretraining.' },
          { formula: '\\texttt{type: chat\\_template}', explanation: 'Generic chat format using the tokenizer chat template.' },
        ]}
        id="example-dataset-types"
      />

      <NoteBlock
        type="tip"
        title="Multi-Dataset Training"
        content="Axolotl excels at mixing multiple datasets in a single training run. Simply list them in the datasets array. Each can have a different format type. The framework handles interleaving and proper formatting automatically."
        id="note-multi-dataset"
      />

      <WarningBlock
        title="YAML Syntax Pitfalls"
        content="YAML is whitespace-sensitive. Common errors: using tabs instead of spaces, missing colons, incorrect indentation for nested values. Use a YAML validator before running. Also ensure boolean values are lowercase (true/false, not True/False)."
        id="warning-yaml"
      />
    </div>
  )
}
