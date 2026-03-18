import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function DistributedTraining() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Distributed Training: Data, Tensor, and Pipeline Parallelism</h1>
      <p className="text-lg text-gray-300">
        Training large language models requires distributing computation across many GPUs.
        Three fundamental parallelism strategies -- data, tensor, and pipeline parallelism --
        can be combined to train models that exceed the memory and compute capacity of any single device.
      </p>

      <DefinitionBlock
        title="Data Parallelism (DP)"
        definition="Each GPU holds a complete copy of the model. The training batch is split across $P$ GPUs, each processing $B/P$ samples. Gradients are synchronized via all-reduce before the optimizer step. Communication cost per step: $O(|\\theta|)$ where $|\\theta|$ is the number of parameters."
        notation="Effective batch size = $B_{\text{per\\_gpu}} \times P$. With ZeRO optimization, model states (optimizer, gradients, parameters) can be sharded across GPUs."
        id="data-parallel-def"
      />

      <DefinitionBlock
        title="Tensor Parallelism (TP)"
        definition="Individual layers are split across GPUs. For a linear layer $Y = XW$, the weight matrix $W \in \mathbb{R}^{d \times d}$ is partitioned column-wise: $W = [W_1, W_2, \ldots, W_P]$ across $P$ GPUs. Each GPU computes $Y_i = XW_i$ and results are combined via all-reduce."
        notation="Communication per layer: two all-reduce ops of size $O(B \cdot d)$. Typically used within a single node (high-bandwidth NVLink)."
        id="tensor-parallel-def"
      />

      <DefinitionBlock
        title="Pipeline Parallelism (PP)"
        definition="Model layers are partitioned into $S$ stages across GPUs. Stage $s$ holds layers $[l_s, l_{s+1})$. Micro-batches flow through stages sequentially. GPipe and PipeDream schedule micro-batches to minimize bubble overhead (idle time): bubble ratio $\approx (S-1)/(S-1+M)$ where $M$ is the number of micro-batches."
        notation="With $M$ micro-batches and $S$ stages, pipeline bubble fraction $\approx (S-1)/M$."
        id="pipeline-parallel-def"
      />

      <ExampleBlock
        title="3D Parallelism for a 70B Model"
        problem="Design a parallelism strategy for training a 70B parameter model on 128 A100 GPUs (80GB each)."
        steps={[
          {
            formula: '\\text{Memory per GPU (FP16): } 70\\text{B} \\times 2 = 140\\text{GB (model only)}',
            explanation: 'Model weights alone exceed single GPU memory. Need model parallelism.'
          },
          {
            formula: '\\text{Tensor Parallel} = 8 \\text{ (within each node of 8 GPUs)}',
            explanation: 'TP=8 splits each layer across 8 GPUs. Each holds 70B/8 = 8.75B params/GPU for weights.'
          },
          {
            formula: '\\text{Pipeline Parallel} = 4 \\text{ (across 4 nodes)}',
            explanation: 'PP=4 splits 80 layers into 4 stages of 20 layers each. Reduces per-GPU memory.'
          },
          {
            formula: '\\text{Data Parallel} = 128 / (8 \\times 4) = 4',
            explanation: 'Remaining GPUs do data parallelism. Total: TP=8 x PP=4 x DP=4 = 128 GPUs.'
          }
        ]}
        id="3d-parallel-example"
      />

      <NoteBlock
        type="tip"
        title="ZeRO: Memory-Efficient Data Parallelism"
        content="DeepSpeed ZeRO (Zero Redundancy Optimizer) eliminates memory redundancy in data parallelism. ZeRO-1 shards optimizer states (4x savings). ZeRO-2 also shards gradients (8x savings). ZeRO-3 shards parameters too, enabling training models that don't fit on any single GPU using only data parallelism. FSDP (Fully Sharded Data Parallel) in PyTorch implements similar concepts."
        id="zero-note"
      />

      <PythonCode
        title="distributed_training.py"
        code={`import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Data Parallel training setup
def setup_ddp(rank, world_size):
    """Initialize distributed training."""
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

# Example: DDP wrapper
# model = MyModel().to(rank)
# model = DDP(model, device_ids=[rank])

# Using HuggingFace Accelerate for easy multi-GPU
from accelerate import Accelerator

def train_with_accelerate():
    accelerator = Accelerator()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Accelerate handles device placement and parallelism
    model, optimizer = accelerator.prepare(model, optimizer)

    print(f"Device: {accelerator.device}")
    print(f"Num processes: {accelerator.num_processes}")
    print(f"Mixed precision: {accelerator.mixed_precision}")

# DeepSpeed ZeRO configuration
deepspeed_config = {
    "train_batch_size": 256,
    "gradient_accumulation_steps": 8,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,             # ZeRO Stage 3: shard everything
        "offload_optimizer": {
            "device": "cpu"     # Offload optimizer states to CPU
        },
        "offload_param": {
            "device": "cpu"     # Offload params to CPU when not in use
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
    },
}
print("DeepSpeed ZeRO-3 config ready")

# Memory estimation
def estimate_memory(N_params, precision="fp16", zero_stage=0, num_gpus=1):
    """Estimate per-GPU memory for training."""
    bytes_per_param = 2 if precision == "fp16" else 4
    # Model weights
    model_mem = N_params * bytes_per_param
    # Gradients
    grad_mem = N_params * bytes_per_param
    # Optimizer states (AdamW: 2 states in FP32)
    opt_mem = N_params * 4 * 2  # momentum + variance

    total = model_mem + grad_mem + opt_mem
    if zero_stage >= 1:
        opt_mem /= num_gpus
    if zero_stage >= 2:
        grad_mem /= num_gpus
    if zero_stage >= 3:
        model_mem /= num_gpus

    per_gpu = (model_mem + grad_mem + opt_mem) / 1e9
    print(f"N={N_params/1e9:.0f}B, {precision}, ZeRO-{zero_stage}, "
          f"{num_gpus} GPUs -> {per_gpu:.1f} GB/GPU")
    return per_gpu

estimate_memory(7e9, "fp16", zero_stage=0, num_gpus=1)
estimate_memory(7e9, "fp16", zero_stage=3, num_gpus=8)
estimate_memory(70e9, "fp16", zero_stage=3, num_gpus=64)`}
        id="distributed-code"
      />

      <WarningBlock
        title="Communication Overhead"
        content="Distributed training adds communication overhead between GPUs. Tensor parallelism requires low-latency, high-bandwidth connections (NVLink within a node). Pipeline parallelism introduces bubble overhead. Data parallelism requires gradient all-reduce. Poor network topology or bandwidth can make scaling efficiency drop well below the ideal linear speedup."
        id="comm-overhead-warning"
      />
    </div>
  )
}
