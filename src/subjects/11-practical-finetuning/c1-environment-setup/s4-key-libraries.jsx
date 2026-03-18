import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function KeyLibraries() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Key Libraries: transformers, peft, trl, bitsandbytes</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The Hugging Face ecosystem provides the core building blocks for finetuning. Understanding
        what each library does and how they work together is essential for building effective
        finetuning pipelines.
      </p>

      <DefinitionBlock
        title="Hugging Face Transformers"
        definition="The transformers library provides thousands of pretrained models and a unified API for loading, running inference, and training them. It abstracts away model-specific details behind a consistent AutoModel / AutoTokenizer interface."
        id="def-transformers"
      />

      <h2 className="text-2xl font-semibold">Library Overview</h2>
      <PythonCode
        title="install_core_libraries.sh"
        code={`# Core finetuning stack
pip install transformers>=4.44.0    # Model loading, tokenizers, Trainer
pip install peft>=0.12.0            # LoRA, QLoRA, adapter methods
pip install trl>=0.9.0              # SFTTrainer, DPO, RLHF training
pip install bitsandbytes>=0.43.0    # 4-bit/8-bit quantization
pip install accelerate>=0.33.0      # Multi-GPU, mixed precision
pip install datasets>=2.20.0        # Dataset loading and processing

# Optional but recommended
pip install flash-attn --no-build-isolation  # Flash Attention 2
pip install wandb                            # Experiment tracking
pip install scipy                            # For some trainer features`}
        id="code-install-libs"
      />

      <h2 className="text-2xl font-semibold">transformers: Model Loading</h2>
      <PythonCode
        title="transformers_basics.py"
        code={`from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch

# Load a model and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load in 4-bit for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

print(f"Model loaded. Memory: {model.get_memory_footprint()/1e9:.1f} GB")`}
        id="code-transformers"
      />

      <h2 className="text-2xl font-semibold">PEFT: Parameter-Efficient Finetuning</h2>
      <PythonCode
        title="peft_lora_setup.py"
        code={`from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # Rank of the update matrices
    lora_alpha=32,                 # Scaling factor (alpha/r)
    lora_dropout=0.05,             # Dropout on LoRA layers
    target_modules=[               # Which layers to apply LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
)

# Apply LoRA to the quantized model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 8,030,261,248
# trainable%: 0.5223`}
        id="code-peft"
      />

      <ExampleBlock
        title="Library Interaction Flow"
        problem="How do transformers, PEFT, bitsandbytes, and TRL work together for QLoRA finetuning?"
        steps={[
          { formula: '\\texttt{bitsandbytes} \\rightarrow \\text{4-bit quantized model}', explanation: 'bitsandbytes quantizes model weights to 4-bit NF4 format during loading.' },
          { formula: '\\texttt{PEFT} \\rightarrow \\text{adds LoRA adapters}', explanation: 'PEFT adds small trainable LoRA matrices to frozen quantized layers.' },
          { formula: '\\texttt{TRL SFTTrainer} \\rightarrow \\text{training loop}', explanation: 'TRL manages the supervised finetuning loop with proper chat formatting.' },
          { formula: '\\texttt{transformers Trainer} \\rightarrow \\text{optimization}', explanation: 'Under the hood, TRL uses the transformers Trainer for gradient computation and optimization.' },
        ]}
        id="example-lib-flow"
      />

      <NoteBlock
        type="note"
        title="bitsandbytes for Quantization"
        content="bitsandbytes provides 4-bit and 8-bit quantization kernels optimized for NVIDIA GPUs. The NF4 (NormalFloat4) data type is specifically designed for normally distributed neural network weights, providing better accuracy than standard int4 quantization."
        id="note-bnb"
      />

      <WarningBlock
        title="Version Compatibility"
        content="These libraries evolve rapidly. Always check compatibility: transformers 4.44+ requires peft 0.12+, and trl 0.9+ requires transformers 4.42+. Pin versions in your requirements.txt after confirming a working setup."
        id="warning-versions"
      />

      <NoteBlock
        type="tip"
        title="accelerate for Multi-GPU"
        content="The accelerate library handles distributed training transparently. Run accelerate config once to set up your hardware profile, then use accelerate launch train.py instead of python train.py. It handles data parallelism, model parallelism, and mixed precision automatically."
        id="note-accelerate"
      />
    </div>
  )
}
