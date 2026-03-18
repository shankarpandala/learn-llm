import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function GpuRequirements() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">GPU Requirements and VRAM Calculations</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Finetuning large language models demands significant GPU resources. Understanding VRAM
        requirements before you begin saves hours of frustration from out-of-memory errors. This
        section covers how to estimate memory needs and choose the right hardware.
      </p>

      <DefinitionBlock
        title="VRAM (Video RAM)"
        definition="VRAM is the dedicated memory on a GPU used to store model weights, optimizer states, gradients, and activations during training. For a model with $P$ parameters in fp16, the base weight memory is $2P$ bytes."
        notation="VRAM_{total} = W_{model} + W_{optimizer} + W_{gradients} + W_{activations}"
        id="def-vram"
      />

      <h2 className="text-2xl font-semibold">Memory Breakdown for Finetuning</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Full finetuning of a model with <InlineMath math="P" /> parameters in fp16 requires roughly:
      </p>
      <BlockMath math="\text{VRAM}_{\text{full}} \approx 2P + 8P + 2P + \text{activations} \approx 12P \text{ bytes}" />
      <p className="text-gray-700 dark:text-gray-300">
        The 8P comes from the AdamW optimizer storing two fp32 states per parameter. With LoRA
        (rank 16, targeting q_proj and v_proj), trainable parameters drop to roughly 0.1-1% of P.
      </p>

      <ExampleBlock
        title="VRAM Estimation for LLaMA 3 8B"
        problem="Estimate the VRAM needed to finetune LLaMA 3 8B with full finetuning vs QLoRA."
        steps={[
          { formula: 'P = 8 \\times 10^9', explanation: 'LLaMA 3 8B has 8 billion parameters.' },
          { formula: '\\text{Full FP16} = 12 \\times 8 \\times 10^9 \\approx 96\\text{ GB}', explanation: 'Full finetuning needs ~96 GB VRAM (2x A100 80GB or 1x H100).' },
          { formula: '\\text{QLoRA 4-bit} \\approx 0.5P + 12 \\times 0.01P \\approx 5\\text{ GB}', explanation: 'QLoRA loads base model in 4-bit (~4 GB) plus trains ~1% params in fp16.' },
          { formula: '\\text{QLoRA total} \\approx 6\\text{-}10\\text{ GB}', explanation: 'With activations and overhead, QLoRA fits on a single 24 GB consumer GPU.' },
        ]}
        id="example-vram-calc"
      />

      <PythonCode
        title="check_gpu_memory.py"
        code={`import torch

# Check available GPU memory
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(i)
        total_gb = gpu.total_mem / 1e9
        free_mem = torch.cuda.mem_get_info(i)
        free_gb = free_mem[0] / 1e9
        print(f"GPU {i}: {gpu.name}")
        print(f"  Total VRAM: {total_gb:.1f} GB")
        print(f"  Free VRAM:  {free_gb:.1f} GB")
        print(f"  Compute capability: {gpu.major}.{gpu.minor}")
else:
    print("No CUDA GPU available")

# Estimate VRAM for a given model size
def estimate_vram(params_billions, method="qlora"):
    P = params_billions * 1e9
    if method == "full_fp16":
        return 12 * P / 1e9  # ~12 bytes per param
    elif method == "lora_fp16":
        return (2 * P + 12 * 0.01 * P) / 1e9
    elif method == "qlora":
        return (0.5 * P + 12 * 0.01 * P) / 1e9
    elif method == "qlora_dora":
        return (0.5 * P + 12 * 0.02 * P) / 1e9

models = [7, 13, 34, 70]
for size in models:
    full = estimate_vram(size, "full_fp16")
    qlora = estimate_vram(size, "qlora")
    print(f"{size}B model: Full={full:.0f}GB, QLoRA={qlora:.0f}GB")`}
        id="code-check-gpu"
      />

      <NoteBlock
        type="tip"
        title="GPU Recommendations by Model Size"
        content="For 7-8B models: RTX 3090/4090 (24 GB) works with QLoRA. For 13B: RTX 4090 with QLoRA or A100 40GB with LoRA. For 34-70B: A100 80GB or multiple GPUs with QLoRA. Cloud options (RunPod, Lambda, vast.ai) are cost-effective for occasional finetuning."
        id="note-gpu-recs"
      />

      <WarningBlock
        title="Batch Size and Activation Memory"
        content="The estimates above assume batch size 1. Larger batch sizes increase activation memory linearly. Gradient checkpointing can trade compute for memory, reducing activation memory by ~60% at the cost of ~30% slower training. Always enable it for large models."
        id="warning-batch-size"
      />

      <NoteBlock
        type="note"
        title="Flash Attention Reduces Memory"
        content="Flash Attention 2 reduces activation memory from O(n^2) to O(n) for sequence length n. For context lengths of 2048+, this can save several GB of VRAM. Most modern finetuning frameworks enable it automatically when available."
        id="note-flash-attn"
      />
    </div>
  )
}
