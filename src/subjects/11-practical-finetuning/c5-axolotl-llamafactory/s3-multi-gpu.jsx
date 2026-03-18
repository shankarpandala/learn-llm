import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function MultiGpu() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Multi-GPU Finetuning Setup</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        When a model does not fit on a single GPU or you want to speed up training, multi-GPU
        strategies become necessary. This section covers data parallelism, model parallelism,
        FSDP, and DeepSpeed for distributed finetuning.
      </p>

      <DefinitionBlock
        title="Data Parallelism"
        definition="Each GPU holds a complete copy of the model and processes different batches simultaneously. Gradients are synchronized (all-reduce) after each step. Effective batch size is $\\text{per\\_device\\_batch} \\times \\text{num\\_gpus} \\times \\text{gradient\\_accumulation}$."
        id="def-data-parallel"
      />

      <DefinitionBlock
        title="FSDP (Fully Sharded Data Parallelism)"
        definition="FSDP shards model parameters, gradients, and optimizer states across GPUs. Each GPU holds only $1/N$ of the model state, enabling training of models $N\\times$ larger than single-GPU capacity. Parameters are gathered on-demand for computation."
        id="def-fsdp"
      />

      <h2 className="text-2xl font-semibold">accelerate Configuration</h2>

      <PythonCode
        title="setup_multi_gpu.sh"
        code={`# Configure accelerate for multi-GPU training
accelerate config
# Interactive prompts:
# - Distributed type: multi-GPU
# - Number of machines: 1
# - Number of GPUs: 4
# - Mixed precision: bf16

# This creates ~/.cache/huggingface/accelerate/default_config.yaml
# Or specify config directly:

cat > accelerate_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
num_processes: 4
mixed_precision: bf16
gpu_ids: 0,1,2,3
EOF

# Launch training with accelerate
accelerate launch --config_file accelerate_config.yaml \\
    train.py --args...

# For Axolotl:
accelerate launch --multi_gpu --num_processes 4 \\
    -m axolotl.cli.train config.yml

# For LLaMA-Factory:
llamafactory-cli train config.json  # Auto-detects GPUs`}
        id="code-accelerate"
      />

      <h2 className="text-2xl font-semibold">DeepSpeed ZeRO</h2>

      <PythonCode
        title="deepspeed_config.json"
        code={`{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}

// DeepSpeed ZeRO Stages:
// Stage 1: Shard optimizer states only (~4x memory reduction)
// Stage 2: Shard optimizer + gradients (~8x memory reduction)
// Stage 3: Shard optimizer + gradients + parameters (~Nx reduction)
// Stage 3 + offload: Offload params/optimizer to CPU/NVMe`}
        id="code-deepspeed"
      />

      <PythonCode
        title="fsdp_training.py"
        code={`# FSDP configuration for Hugging Face Trainer
from transformers import TrainingArguments

# FSDP via TrainingArguments
training_args = TrainingArguments(
    output_dir="./fsdp-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    # FSDP configuration
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        "fsdp_backward_prefetch": "backward_pre",
        "fsdp_forward_prefetch": True,
        "fsdp_use_orig_params": True,
        "fsdp_state_dict_type": "FULL_STATE_DICT",
    },
)

# Launch with:
# torchrun --nproc_per_node=4 train.py`}
        id="code-fsdp"
      />

      <ExampleBlock
        title="Choosing a Multi-GPU Strategy"
        problem="Which multi-GPU strategy should you use for different scenarios?"
        steps={[
          { formula: '\\text{Model fits on 1 GPU: Data Parallel}', explanation: 'Simplest approach. Each GPU gets full model copy. Linear speedup.' },
          { formula: '\\text{QLoRA + multi-GPU: DeepSpeed ZeRO-2}', explanation: 'Shards optimizer states. Works well with QLoRA on 2-8 GPUs.' },
          { formula: '\\text{Full FT, 13-70B: FSDP or ZeRO-3}', explanation: 'Shards everything. Required when model does not fit on one GPU.' },
          { formula: '\\text{70B+ with limited GPUs: ZeRO-3 + CPU offload}', explanation: 'Offloads optimizer to CPU RAM. Slower but enables very large models.' },
        ]}
        id="example-strategy"
      />

      <NoteBlock
        type="tip"
        title="QLoRA + Multi-GPU"
        content="For QLoRA multi-GPU training, use DeepSpeed ZeRO-2 (not ZeRO-3). ZeRO-3 does not work well with 4-bit quantized models because it tries to shard the quantized weights. ZeRO-2 shards only optimizer states and gradients, which are the LoRA parameters in fp16."
        id="note-qlora-multigpu"
      />

      <WarningBlock
        title="Communication Overhead"
        content="Multi-GPU training adds communication overhead for gradient synchronization. With 2 GPUs you get ~1.8x speedup, with 4 GPUs ~3.2x, and with 8 GPUs ~5-6x (not 8x). NVLink between GPUs dramatically reduces this overhead compared to PCIe."
        id="warning-comm-overhead"
      />
    </div>
  )
}
