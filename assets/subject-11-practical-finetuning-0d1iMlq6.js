import{j as e}from"./vendor-DWbzdFaj.js";import{r}from"./vendor-katex-BYl39Yo6.js";import{D as o,E as n,P as t,N as a,W as i,T as s}from"./subject-01-text-fundamentals-DG6tAvii.js";function l(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"GPU Requirements and VRAM Calculations"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Finetuning large language models demands significant GPU resources. Understanding VRAM requirements before you begin saves hours of frustration from out-of-memory errors. This section covers how to estimate memory needs and choose the right hardware."}),e.jsx(o,{title:"VRAM (Video RAM)",definition:"VRAM is the dedicated memory on a GPU used to store model weights, optimizer states, gradients, and activations during training. For a model with $P$ parameters in fp16, the base weight memory is $2P$ bytes.",notation:"VRAM_{total} = W_{model} + W_{optimizer} + W_{gradients} + W_{activations}",id:"def-vram"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Memory Breakdown for Finetuning"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["Full finetuning of a model with ",e.jsx(r.InlineMath,{math:"P"})," parameters in fp16 requires roughly:"]}),e.jsx(r.BlockMath,{math:"\\text{VRAM}_{\\text{full}} \\approx 2P + 8P + 2P + \\text{activations} \\approx 12P \\text{ bytes}"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The 8P comes from the AdamW optimizer storing two fp32 states per parameter. With LoRA (rank 16, targeting q_proj and v_proj), trainable parameters drop to roughly 0.1-1% of P."}),e.jsx(n,{title:"VRAM Estimation for LLaMA 3 8B",problem:"Estimate the VRAM needed to finetune LLaMA 3 8B with full finetuning vs QLoRA.",steps:[{formula:"P = 8 \\times 10^9",explanation:"LLaMA 3 8B has 8 billion parameters."},{formula:"\\text{Full FP16} = 12 \\times 8 \\times 10^9 \\approx 96\\text{ GB}",explanation:"Full finetuning needs ~96 GB VRAM (2x A100 80GB or 1x H100)."},{formula:"\\text{QLoRA 4-bit} \\approx 0.5P + 12 \\times 0.01P \\approx 5\\text{ GB}",explanation:"QLoRA loads base model in 4-bit (~4 GB) plus trains ~1% params in fp16."},{formula:"\\text{QLoRA total} \\approx 6\\text{-}10\\text{ GB}",explanation:"With activations and overhead, QLoRA fits on a single 24 GB consumer GPU."}],id:"example-vram-calc"}),e.jsx(t,{title:"check_gpu_memory.py",code:`import torch

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
    print(f"{size}B model: Full={full:.0f}GB, QLoRA={qlora:.0f}GB")`,id:"code-check-gpu"}),e.jsx(a,{type:"tip",title:"GPU Recommendations by Model Size",content:"For 7-8B models: RTX 3090/4090 (24 GB) works with QLoRA. For 13B: RTX 4090 with QLoRA or A100 40GB with LoRA. For 34-70B: A100 80GB or multiple GPUs with QLoRA. Cloud options (RunPod, Lambda, vast.ai) are cost-effective for occasional finetuning.",id:"note-gpu-recs"}),e.jsx(i,{title:"Batch Size and Activation Memory",content:"The estimates above assume batch size 1. Larger batch sizes increase activation memory linearly. Gradient checkpointing can trade compute for memory, reducing activation memory by ~60% at the cost of ~30% slower training. Always enable it for large models.",id:"warning-batch-size"}),e.jsx(a,{type:"note",title:"Flash Attention Reduces Memory",content:"Flash Attention 2 reduces activation memory from O(n^2) to O(n) for sequence length n. For context lengths of 2048+, this can save several GB of VRAM. Most modern finetuning frameworks enable it automatically when available.",id:"note-flash-attn"})]})}const pe=Object.freeze(Object.defineProperty({__proto__:null,default:l},Symbol.toStringTag,{value:"Module"}));function d(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"CUDA, cuDNN, and Driver Setup"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"A working CUDA stack is the foundation of GPU-accelerated finetuning. Mismatched versions between the NVIDIA driver, CUDA toolkit, and cuDNN are the most common source of environment issues. This section walks through getting everything configured correctly."}),e.jsx(o,{title:"CUDA Toolkit",definition:"CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform. The CUDA toolkit includes the compiler (nvcc), runtime libraries, and tools needed to run GPU-accelerated code. PyTorch ships with its own CUDA runtime, so you typically only need matching NVIDIA drivers.",id:"def-cuda"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Checking Your Current Setup"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Before installing anything, check what you already have. The NVIDIA driver version determines the maximum CUDA version your system supports."}),e.jsx(t,{title:"check_cuda_environment.py",code:`import subprocess, torch

# Check NVIDIA driver version
result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(result.stdout[:500])

# Check PyTorch CUDA support
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (PyTorch): {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Quick sanity test
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x.T
    print(f"Matrix multiply test: {y.shape} - PASSED")

# Check if bfloat16 is supported (Ampere+ GPUs)
if torch.cuda.is_bf16_supported():
    print("BFloat16: supported (Ampere or newer GPU)")
else:
    print("BFloat16: NOT supported (use fp16 instead)")`,id:"code-check-cuda"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Installing NVIDIA Drivers"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"On Ubuntu/Debian systems, the recommended approach is to use the official NVIDIA package repository. Driver version 535+ is recommended for CUDA 12.x support."}),e.jsx(t,{title:"install_drivers.sh",code:`# Ubuntu 22.04/24.04 - Install NVIDIA drivers
# Option 1: Ubuntu's built-in driver manager
sudo ubuntu-drivers autoinstall

# Option 2: Specific driver version from NVIDIA repo
sudo apt-get install -y nvidia-driver-550

# After installation, reboot and verify
sudo reboot
# Then run:
nvidia-smi

# For cloud instances (Lambda, RunPod, etc.)
# Drivers are typically pre-installed - just verify:
nvidia-smi | head -5
# Should show driver version >= 535 for CUDA 12.x`,id:"code-install-drivers"}),e.jsx(n,{title:"CUDA Version Compatibility",problem:"Determine which PyTorch CUDA version to use given driver version 550.54.",steps:[{formula:"\\text{Driver } 550.x \\Rightarrow \\text{Max CUDA } 12.4",explanation:"Check the NVIDIA CUDA compatibility table for your driver version."},{formula:"\\text{PyTorch 2.4+} \\Rightarrow \\text{CUDA 12.1 or 12.4}",explanation:"PyTorch ships its own CUDA runtime; pick the matching build."},{formula:"\\texttt{pip install torch --index-url .../cu124}",explanation:"Install PyTorch built against CUDA 12.4 for best compatibility."}],id:"example-cuda-compat"}),e.jsx(a,{type:"tip",title:"PyTorch Bundles Its Own CUDA",content:"You do NOT need to install the CUDA toolkit system-wide for PyTorch. PyTorch pip packages include the necessary CUDA libraries. You only need the NVIDIA driver installed. This simplifies setup enormously - just match the driver version to the PyTorch CUDA build.",id:"note-pytorch-cuda"}),e.jsx(i,{title:"Avoid Multiple CUDA Installations",content:"Having multiple CUDA toolkit versions installed system-wide causes PATH and library conflicts. If you must have system CUDA (e.g., for compiling custom kernels), use environment variables to manage versions: export CUDA_HOME=/usr/local/cuda-12.4 and update PATH accordingly.",id:"warning-multi-cuda"}),e.jsx(a,{type:"note",title:"Flash Attention 2 Compilation",content:"Flash Attention 2 requires CUDA toolkit headers for compilation. Install with: pip install flash-attn --no-build-isolation. If compilation fails, ensure you have the CUDA toolkit matching your PyTorch version, or use a pre-built wheel from the flash-attn releases page.",id:"note-flash-attn-compile"})]})}const me=Object.freeze(Object.defineProperty({__proto__:null,default:d},Symbol.toStringTag,{value:"Module"}));function p(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Python Environment Setup"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"A clean, isolated Python environment prevents dependency conflicts that plague ML projects. This section covers setting up virtual environments with venv, conda, and managing packages with pip for finetuning workflows."}),e.jsx(o,{title:"Virtual Environment",definition:"A virtual environment is an isolated Python installation with its own packages, independent of the system Python. This prevents version conflicts between projects and ensures reproducibility.",id:"def-venv"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Option 1: venv (Recommended for Simplicity)"}),e.jsx(t,{title:"setup_venv.sh",code:`# Create a virtual environment with Python 3.10+
python3.11 -m venv ~/finetune-env

# Activate the environment
source ~/finetune-env/bin/activate

# Verify Python version
python --version  # Should be 3.10 or 3.11

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio \\
    --index-url https://download.pytorch.org/whl/cu124

# Verify GPU access
python -c "import torch; print(torch.cuda.is_available())"

# Save environment for reproducibility
pip freeze > requirements.txt`,id:"code-venv-setup"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Option 2: Conda (Better for Complex Dependencies)"}),e.jsx(t,{title:"setup_conda.sh",code:`# Install miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
eval "$(~/miniconda3/bin/conda shell.bash hook)"

# Create environment with specific Python version
conda create -n finetune python=3.11 -y
conda activate finetune

# Install PyTorch via conda (handles CUDA automatically)
conda install pytorch torchvision torchaudio \\
    pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Export environment
conda env export > environment.yml

# Recreate environment on another machine
conda env create -f environment.yml`,id:"code-conda-setup"}),e.jsx(n,{title:"Complete Finetuning Environment",problem:"Set up a complete environment for QLoRA finetuning with unsloth.",steps:[{formula:"\\texttt{python3.11 -m venv ~/qlora-env}",explanation:"Create isolated environment with Python 3.11."},{formula:"\\texttt{pip install torch --index-url .../cu124}",explanation:"Install PyTorch with CUDA 12.4 support."},{formula:"\\texttt{pip install unsloth[cu124]}",explanation:"Install unsloth with all finetuning dependencies."},{formula:"\\texttt{pip install wandb}",explanation:"Install Weights & Biases for experiment tracking."}],id:"example-complete-env"}),e.jsx(t,{title:"verify_environment.py",code:`# Run this script to verify your finetuning environment
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError:
    print("Transformers: NOT INSTALLED")

try:
    import peft
    print(f"PEFT: {peft.__version__}")
except ImportError:
    print("PEFT: NOT INSTALLED")

try:
    import trl
    print(f"TRL: {trl.__version__}")
except ImportError:
    print("TRL: NOT INSTALLED")

try:
    import bitsandbytes
    print(f"bitsandbytes: {bitsandbytes.__version__}")
except ImportError:
    print("bitsandbytes: NOT INSTALLED")

try:
    from unsloth import FastLanguageModel
    print("Unsloth: INSTALLED")
except ImportError:
    print("Unsloth: NOT INSTALLED")

print("\\nEnvironment check complete!")`,id:"code-verify-env"}),e.jsx(i,{title:"Python Version Matters",content:"Use Python 3.10 or 3.11 for finetuning. Python 3.12+ may have compatibility issues with some CUDA libraries and finetuning packages. Always check the documentation of your chosen finetuning framework for supported Python versions.",id:"warning-python-version"}),e.jsx(a,{type:"tip",title:"Using requirements.txt for Reproducibility",content:"Always pin exact versions in requirements.txt for production environments. Use pip freeze > requirements.txt after confirming everything works. Share this file with collaborators to ensure identical environments.",id:"note-requirements"})]})}const ce=Object.freeze(Object.defineProperty({__proto__:null,default:p},Symbol.toStringTag,{value:"Module"}));function m(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Key Libraries: transformers, peft, trl, bitsandbytes"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The Hugging Face ecosystem provides the core building blocks for finetuning. Understanding what each library does and how they work together is essential for building effective finetuning pipelines."}),e.jsx(o,{title:"Hugging Face Transformers",definition:"The transformers library provides thousands of pretrained models and a unified API for loading, running inference, and training them. It abstracts away model-specific details behind a consistent AutoModel / AutoTokenizer interface.",id:"def-transformers"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Library Overview"}),e.jsx(t,{title:"install_core_libraries.sh",code:`# Core finetuning stack
pip install transformers>=4.44.0    # Model loading, tokenizers, Trainer
pip install peft>=0.12.0            # LoRA, QLoRA, adapter methods
pip install trl>=0.9.0              # SFTTrainer, DPO, RLHF training
pip install bitsandbytes>=0.43.0    # 4-bit/8-bit quantization
pip install accelerate>=0.33.0      # Multi-GPU, mixed precision
pip install datasets>=2.20.0        # Dataset loading and processing

# Optional but recommended
pip install flash-attn --no-build-isolation  # Flash Attention 2
pip install wandb                            # Experiment tracking
pip install scipy                            # For some trainer features`,id:"code-install-libs"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"transformers: Model Loading"}),e.jsx(t,{title:"transformers_basics.py",code:`from transformers import (
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

print(f"Model loaded. Memory: {model.get_memory_footprint()/1e9:.1f} GB")`,id:"code-transformers"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"PEFT: Parameter-Efficient Finetuning"}),e.jsx(t,{title:"peft_lora_setup.py",code:`from peft import LoraConfig, get_peft_model, TaskType

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
# trainable%: 0.5223`,id:"code-peft"}),e.jsx(n,{title:"Library Interaction Flow",problem:"How do transformers, PEFT, bitsandbytes, and TRL work together for QLoRA finetuning?",steps:[{formula:"\\texttt{bitsandbytes} \\rightarrow \\text{4-bit quantized model}",explanation:"bitsandbytes quantizes model weights to 4-bit NF4 format during loading."},{formula:"\\texttt{PEFT} \\rightarrow \\text{adds LoRA adapters}",explanation:"PEFT adds small trainable LoRA matrices to frozen quantized layers."},{formula:"\\texttt{TRL SFTTrainer} \\rightarrow \\text{training loop}",explanation:"TRL manages the supervised finetuning loop with proper chat formatting."},{formula:"\\texttt{transformers Trainer} \\rightarrow \\text{optimization}",explanation:"Under the hood, TRL uses the transformers Trainer for gradient computation and optimization."}],id:"example-lib-flow"}),e.jsx(a,{type:"note",title:"bitsandbytes for Quantization",content:"bitsandbytes provides 4-bit and 8-bit quantization kernels optimized for NVIDIA GPUs. The NF4 (NormalFloat4) data type is specifically designed for normally distributed neural network weights, providing better accuracy than standard int4 quantization.",id:"note-bnb"}),e.jsx(i,{title:"Version Compatibility",content:"These libraries evolve rapidly. Always check compatibility: transformers 4.44+ requires peft 0.12+, and trl 0.9+ requires transformers 4.42+. Pin versions in your requirements.txt after confirming a working setup.",id:"warning-versions"}),e.jsx(a,{type:"tip",title:"accelerate for Multi-GPU",content:"The accelerate library handles distributed training transparently. Run accelerate config once to set up your hardware profile, then use accelerate launch train.py instead of python train.py. It handles data parallelism, model parallelism, and mixed precision automatically.",id:"note-accelerate"})]})}const ue=Object.freeze(Object.defineProperty({__proto__:null,default:m},Symbol.toStringTag,{value:"Module"}));function c(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Full Finetuning vs Parameter-Efficient Finetuning"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"When adapting a pretrained model to a new task, you can update all parameters (full finetuning) or only a small subset (PEFT). Each approach has distinct tradeoffs in memory, compute, data requirements, and model quality."}),e.jsx(o,{title:"Full Finetuning",definition:"Full finetuning updates all $P$ parameters of a pretrained model using gradient descent on the target dataset. This provides maximum expressiveness but requires storing the full model weights, optimizer states ($8P$ bytes for AdamW in fp32), and gradients ($2P$ bytes in fp16).",id:"def-full-ft"}),e.jsx(o,{title:"Parameter-Efficient Finetuning (PEFT)",definition:"PEFT methods freeze most pretrained weights and only train a small number of additional or selected parameters. If the trainable parameter count is $P_{\\\\text{train}} \\\\ll P$, memory for optimizer states drops from $8P$ to $8P_{\\\\text{train}}$ bytes.",id:"def-peft"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Comparison Table"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The key tradeoffs between full finetuning and PEFT methods like LoRA:"}),e.jsx(n,{title:"Memory Comparison: 7B Model",problem:"Compare VRAM requirements for full finetuning vs LoRA vs QLoRA on a 7B parameter model.",steps:[{formula:"\\text{Full FP16: } 2(7B) + 8(7B) + 2(7B) \\approx 84\\text{ GB}",explanation:"Weights (fp16) + AdamW states (fp32) + gradients (fp16)."},{formula:"\\text{LoRA FP16: } 2(7B) + 8(0.07B) + 2(0.07B) \\approx 14.7\\text{ GB}",explanation:"Full weights frozen in fp16, only ~1% params trained."},{formula:"\\text{QLoRA: } 0.5(7B) + 8(0.07B) + 2(0.07B) \\approx 4.2\\text{ GB}",explanation:"Base weights in 4-bit (~0.5 bytes/param), LoRA in fp16."}],id:"example-memory-comparison"}),e.jsx(t,{title:"compare_finetuning_methods.py",code:`from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# --- Full Finetuning Setup ---
# model_full = AutoModelForCausalLM.from_pretrained(
#     model_name, torch_dtype=torch.float16, device_map="auto"
# )
# All params trainable
# total = sum(p.numel() for p in model_full.parameters())
# trainable = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
# print(f"Full FT: {trainable/1e6:.0f}M / {total/1e6:.0f}M params trainable")

# --- LoRA Setup ---
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"
)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model_lora = get_peft_model(model, lora_config)
model_lora.print_trainable_parameters()
# trainable params: 13,631,488 || all params: 8,030,261,248
# trainable%: 0.1698

# Key insight: QLoRA trains the same LoRA params but base is 4-bit
mem = model.get_memory_footprint() / 1e9
print(f"Base model memory (4-bit): {mem:.1f} GB")`,id:"code-compare-methods"}),e.jsx(a,{type:"intuition",title:"Why Does PEFT Work So Well?",content:"Pretrained LLMs already encode vast knowledge in their weights. Finetuning is really about steering the model toward specific behaviors, not teaching it new knowledge. A low-rank update (LoRA rank 8-64) is sufficient to capture this behavioral shift, because the 'direction' of adaptation lies in a low-dimensional subspace.",id:"note-peft-intuition"}),e.jsx(i,{title:"When Full Finetuning Is Better",content:"PEFT may underperform full finetuning when: (1) the target domain is very different from pretraining data, (2) you have a large high-quality dataset (100K+ examples), (3) you need maximum performance and have the compute budget, or (4) you are doing continued pretraining on a new language or domain.",id:"warning-when-full"}),e.jsx(a,{type:"tip",title:"Practical Recommendation",content:"Start with QLoRA (rank 16-64) for rapid experimentation. If quality is insufficient after hyperparameter tuning, try LoRA in fp16/bf16. Only resort to full finetuning if PEFT clearly underperforms and you have sufficient compute.",id:"note-recommendation"})]})}const fe=Object.freeze(Object.defineProperty({__proto__:null,default:c},Symbol.toStringTag,{value:"Module"}));function u(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"LoRA Deep Dive: Hyperparameters and Rank Selection"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"LoRA (Low-Rank Adaptation) decomposes weight updates into low-rank matrices, drastically reducing trainable parameters. Understanding its hyperparameters -- rank, alpha, target modules, and dropout -- is critical for getting good results."}),e.jsx(o,{title:"LoRA Decomposition",definition:"For a pretrained weight matrix $W_0 \\\\in \\\\mathbb{R}^{d \\\\times k}$, LoRA adds a low-rank update: $W = W_0 + \\\\frac{\\\\alpha}{r} B A$ where $B \\\\in \\\\mathbb{R}^{d \\\\times r}$, $A \\\\in \\\\mathbb{R}^{r \\\\times k}$, and $r \\\\ll \\\\min(d, k)$. Matrix $A$ is initialized with random Gaussian values and $B$ is initialized to zero, so the update starts at zero.",notation:"W = W_0 + \\frac{\\alpha}{r} BA",id:"def-lora"}),e.jsx(s,{title:"LoRA Parameter Count",statement:"For a single linear layer of shape $d \\times k$ with LoRA rank $r$, the number of trainable parameters is $r(d + k)$, compared to $dk$ for full finetuning.",proof:"Matrix A has shape $r \\times k$ (rk parameters) and matrix B has shape $d \\times r$ (dr parameters). Total: $dr + rk = r(d+k)$.",id:"thm-lora-params"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Key Hyperparameters"}),e.jsx(n,{title:"Rank Selection Guidelines",problem:"How to choose the LoRA rank r for different scenarios?",steps:[{formula:"r = 8",explanation:"Good starting point for simple tasks (classification, style transfer). Very memory efficient."},{formula:"r = 16\\text{-}32",explanation:"Recommended default for instruction tuning and chat finetuning. Good quality-efficiency balance."},{formula:"r = 64\\text{-}128",explanation:"For complex domain adaptation or when you have large datasets (50K+ examples)."},{formula:"r = 256",explanation:"Approaches full finetuning expressiveness. Rarely needed; consider full FT at this point."}],id:"example-rank-selection"}),e.jsx(t,{title:"lora_hyperparameters.py",code:`from peft import LoraConfig, get_peft_model

# Recommended configuration for instruction tuning
lora_config = LoraConfig(
    r=16,                    # Rank: controls capacity of adaptation
    lora_alpha=32,           # Scaling: effective lr multiplier = alpha/r = 2
    lora_dropout=0.05,       # Regularization: 0.05-0.1 for small datasets
    target_modules=[         # Which modules to apply LoRA to
        "q_proj",            # Query projection (attention)
        "k_proj",            # Key projection (attention)
        "v_proj",            # Value projection (attention)
        "o_proj",            # Output projection (attention)
        "gate_proj",         # MLP gate projection
        "up_proj",           # MLP up projection
        "down_proj",         # MLP down projection
    ],
    task_type="CAUSAL_LM",
    bias="none",             # Don't train biases (saves memory)
)

# Apply and inspect
# model_lora = get_peft_model(model, lora_config)
# model_lora.print_trainable_parameters()

# --- Experiment with different ranks ---
import torch

d, k = 4096, 4096  # Typical hidden dim for 7B model
for r in [8, 16, 32, 64, 128, 256]:
    lora_params = r * (d + k)
    full_params = d * k
    ratio = lora_params / full_params * 100
    print(f"r={r:>3d}: {lora_params:>10,} params ({ratio:.2f}% of full)")

# r=  8:      65,536 params (0.39% of full)
# r= 16:     131,072 params (0.78% of full)
# r= 32:     262,144 params (1.56% of full)
# r= 64:     524,288 params (3.12% of full)
# r=128:   1,048,576 params (6.25% of full)
# r=256:   2,097,152 params (12.50% of full)`,id:"code-lora-hyperparams"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The Alpha/Rank Ratio"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The effective scaling factor is ",e.jsx(r.InlineMath,{math:"\\frac{\\alpha}{r}"}),". When you change the rank, keep ",e.jsx(r.InlineMath,{math:"\\frac{\\alpha}{r}"})," constant (typically 1 or 2) to maintain similar learning dynamics. For example, if ",e.jsx(r.InlineMath,{math:"r=16, \\alpha=32"}),", then for ",e.jsx(r.InlineMath,{math:"r=64"}),", use ",e.jsx(r.InlineMath,{math:"\\alpha=128"}),"."]}),e.jsx(t,{title:"alpha_rank_experiment.py",code:`# Impact of alpha/r ratio on training dynamics
import numpy as np

def simulate_lora_update(d, r, alpha, lr=1e-4):
    """Simulate the magnitude of a LoRA weight update."""
    # Random initialization (simplified)
    A = np.random.randn(r, d) * 0.01  # Small init
    B = np.zeros((d, r))               # Zero init

    # After one gradient step (simplified)
    grad_A = np.random.randn(r, d) * 0.1
    grad_B = np.random.randn(d, r) * 0.1
    A -= lr * grad_A
    B -= lr * grad_B

    # Effective weight update
    delta_W = (alpha / r) * (B @ A)
    return np.linalg.norm(delta_W)

d = 4096
for r in [8, 16, 32, 64]:
    # Fixed alpha
    norm_fixed = simulate_lora_update(d, r, alpha=16)
    # Scaled alpha (alpha/r = 2)
    norm_scaled = simulate_lora_update(d, r, alpha=2*r)
    print(f"r={r:>2d}: fixed alpha=16 -> |dW|={norm_fixed:.4f}, "
          f"scaled alpha={2*r} -> |dW|={norm_scaled:.4f}")`,id:"code-alpha-experiment"}),e.jsx(a,{type:"tip",title:"Target Module Selection",content:"Targeting all linear layers (q, k, v, o, gate, up, down) gives better results than only q_proj and v_proj, at the cost of more trainable parameters. For 7B models with r=16, targeting all 7 modules adds ~42M trainable params (~0.5%). This is the recommended default.",id:"note-target-modules"}),e.jsx(i,{title:"Overfitting with High Rank",content:"Higher rank does not always mean better results. With small datasets (<1000 examples), rank 8-16 with dropout 0.1 often outperforms rank 64+ which may overfit. Monitor validation loss and use early stopping.",id:"warning-overfit"})]})}const ge=Object.freeze(Object.defineProperty({__proto__:null,default:u},Symbol.toStringTag,{value:"Module"}));function f(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"QLoRA: 4-Bit Quantization + LoRA"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"QLoRA combines 4-bit quantization of the base model with LoRA adapters trained in higher precision. This breakthrough technique enables finetuning 65B+ parameter models on a single 48 GB GPU, democratizing LLM adaptation for researchers with limited hardware."}),e.jsx(o,{title:"QLoRA",definition:"QLoRA (Quantized Low-Rank Adaptation) stores pretrained weights in 4-bit NormalFloat (NF4) format while training LoRA adapters in BFloat16/Float16. During the forward pass, 4-bit weights are dequantized to compute dtype on-the-fly. Gradients only flow through the LoRA parameters, never updating the quantized base.",id:"def-qlora"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Key Innovations in QLoRA"}),e.jsx(o,{title:"NF4 (NormalFloat4)",definition:"NF4 is a 4-bit data type optimized for normally distributed weights. It uses an information-theoretically optimal quantization grid for $\\\\mathcal{N}(0, \\\\sigma^2)$ data, providing better accuracy than standard INT4 or FP4 at the same bit-width.",id:"def-nf4"}),e.jsx(n,{title:"Double Quantization",problem:"How does double quantization further reduce memory in QLoRA?",steps:[{formula:"\\text{Block-wise quant: 64 weights share 1 FP32 scale}",explanation:"Standard quantization uses one 32-bit scale factor per block of 64 weights."},{formula:"\\text{Scale overhead: } \\frac{32}{64} = 0.5 \\text{ bits/param}",explanation:"The scale factors add 0.5 bits per parameter overhead."},{formula:"\\text{Double quant: quantize the scales to FP8}",explanation:"QLoRA quantizes the scale factors themselves, reducing overhead."},{formula:"\\text{New overhead: } \\frac{8}{64} + \\frac{32}{64^2} \\approx 0.13 \\text{ bits/param}",explanation:"Double quantization reduces scale overhead from 0.5 to 0.13 bits per parameter."}],id:"example-double-quant"}),e.jsx(t,{title:"qlora_complete_setup.py",code:`import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# Step 1: Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Enable 4-bit loading
    bnb_4bit_quant_type="nf4",      # NormalFloat4 data type
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16
    bnb_4bit_use_double_quant=True, # Double quantization
)

# Step 2: Load model in 4-bit
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Step 3: Prepare model for QLoRA training
model = prepare_model_for_kbit_training(model)

# Step 4: Add LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",  # Apply to all linear layers
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Memory footprint
mem_gb = model.get_memory_footprint() / 1e9
print(f"Total memory: {mem_gb:.1f} GB (base 4-bit + LoRA fp16)")`,id:"code-qlora-setup"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Paged Optimizers"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"QLoRA introduces paged optimizers that use NVIDIA unified memory to automatically page optimizer states between CPU and GPU memory. This prevents out-of-memory crashes during gradient spikes without manual intervention."}),e.jsx(t,{title:"paged_optimizer.py",code:`from transformers import TrainingArguments

# Training arguments optimized for QLoRA
training_args = TrainingArguments(
    output_dir="./qlora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # Effective batch size = 16
    learning_rate=2e-4,              # Higher LR works well with QLoRA
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,                       # Use bf16 compute
    # Paged optimizer to prevent OOM
    optim="paged_adamw_8bit",        # 8-bit paged AdamW
    gradient_checkpointing=True,     # Trade compute for memory
    max_grad_norm=0.3,               # Gradient clipping
    group_by_length=True,            # Group similar-length sequences
)

print(f"Optimizer: {training_args.optim}")
print(f"Gradient checkpointing: {training_args.gradient_checkpointing}")`,id:"code-paged-optimizer"}),e.jsx(a,{type:"intuition",title:"Why QLoRA Barely Hurts Quality",content:"The 4-bit quantized weights are frozen -- they only participate in the forward pass. All gradient computation and weight updates happen in the LoRA adapters at full precision. The quantization error in the base weights is a fixed 'noise floor' that the LoRA adapters learn to compensate for.",id:"note-qlora-quality"}),e.jsx(i,{title:"QLoRA Training Speed",content:"QLoRA is ~30-40% slower than full-precision LoRA due to dequantization overhead during the forward pass. If you have enough VRAM for LoRA in fp16/bf16, it will train faster and produce slightly better results. QLoRA shines when GPU memory is the bottleneck.",id:"warning-qlora-speed"})]})}const he=Object.freeze(Object.defineProperty({__proto__:null,default:f},Symbol.toStringTag,{value:"Module"}));function g(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"LoRA Variants: DoRA, rsLoRA, and LoRA+"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Since the original LoRA paper, several improvements have been proposed to address its limitations. DoRA decomposes weight updates into magnitude and direction, rsLoRA fixes scaling for high ranks, and LoRA+ uses different learning rates for A and B matrices."}),e.jsx(o,{title:"DoRA (Weight-Decomposed Low-Rank Adaptation)",definition:"DoRA decomposes the pretrained weight into magnitude $m$ and direction $V$: $W = m \\\\frac{V + BA}{\\\\|V + BA\\\\|_c}$. By separating magnitude and direction updates, DoRA more closely matches the learning dynamics of full finetuning while maintaining LoRA's parameter efficiency.",id:"def-dora"}),e.jsx(o,{title:"rsLoRA (Rank-Stabilized LoRA)",definition:"Standard LoRA uses scaling factor $\\\\frac{\\\\alpha}{r}$, which decreases with rank. rsLoRA replaces this with $\\\\frac{\\\\alpha}{\\\\sqrt{r}}$, stabilizing the per-parameter update magnitude as rank increases. This allows effective use of higher ranks without retuning the learning rate.",id:"def-rslora"}),e.jsx(o,{title:"LoRA+",definition:"LoRA+ assigns different learning rates to matrices $A$ and $B$. Specifically, it sets $\\\\eta_B = \\\\lambda \\\\cdot \\\\eta_A$ where $\\\\lambda \\\\approx 16$. This corrects for the asymmetric initialization (A is random, B is zero) and improves convergence speed by ~2x.",id:"def-lora-plus"}),e.jsx(n,{title:"Comparing LoRA Variants",problem:"When should you use each LoRA variant?",steps:[{formula:"\\text{Standard LoRA: } r \\leq 32",explanation:"Works well at low ranks. Use for quick experiments and when memory is tight."},{formula:"\\text{DoRA: } \\text{quality-critical tasks}",explanation:"Slightly better accuracy than LoRA, especially on reasoning and math tasks. ~10% more trainable params."},{formula:"\\text{rsLoRA: } r \\geq 64",explanation:"Use when you need high rank without retuning learning rate. Drop-in replacement for standard LoRA."},{formula:"\\text{LoRA+: } \\text{faster convergence}",explanation:"Reaches same quality as LoRA in fewer steps. Good when compute time matters."}],id:"example-variant-comparison"}),e.jsx(t,{title:"lora_variants_peft.py",code:`from peft import LoraConfig, get_peft_model

# --- Standard LoRA ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# --- DoRA (available in PEFT >= 0.10) ---
dora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    use_dora=True,  # Enable weight decomposition
)

# --- rsLoRA (available in PEFT >= 0.9) ---
rslora_config = LoraConfig(
    r=64,           # Works better at higher ranks
    lora_alpha=128,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    use_rslora=True,  # Use rank-stabilized scaling
)

# --- LoRA+ (use with custom optimizer) ---
# LoRA+ requires setting different learning rates for A and B matrices
# This is done through the optimizer, not the LoRA config

# With Unsloth (easiest way to use LoRA+):
# model, tokenizer = FastLanguageModel.from_pretrained(...)
# model = FastLanguageModel.get_peft_model(
#     model,
#     r=16,
#     lora_alpha=32,
#     use_rslora=True,
#     loraplus_lr_ratio=16.0,  # eta_B = 16 * eta_A
# )

# Compare parameter counts
for name, config in [("LoRA", lora_config), ("DoRA", dora_config),
                      ("rsLoRA", rslora_config)]:
    print(f"{name}: r={config.r}, alpha={config.lora_alpha}, "
          f"dora={config.use_dora}, rslora={config.use_rslora}")`,id:"code-lora-variants"}),e.jsx(t,{title:"loraplus_manual.py",code:`# Manual LoRA+ implementation with different learning rates
import torch
from torch.optim import AdamW

def create_loraplus_optimizer(model, lr=1e-4, loraplus_ratio=16.0):
    """Create optimizer with different LRs for LoRA A and B matrices."""
    param_groups = []
    lora_A_params = []
    lora_B_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            lora_A_params.append(param)
        elif "lora_B" in name:
            lora_B_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {"params": lora_A_params, "lr": lr},
        {"params": lora_B_params, "lr": lr * loraplus_ratio},
        {"params": other_params, "lr": lr},
    ]

    print(f"LoRA A params: {len(lora_A_params)} groups, lr={lr}")
    print(f"LoRA B params: {len(lora_B_params)} groups, lr={lr*loraplus_ratio}")

    return AdamW(param_groups, weight_decay=0.01)

# optimizer = create_loraplus_optimizer(model, lr=1e-4, loraplus_ratio=16)`,id:"code-loraplus"}),e.jsx(a,{type:"note",title:"DoRA Memory Overhead",content:"DoRA adds a learnable magnitude vector per adapted layer, increasing trainable parameters by roughly 10%. Memory overhead is minimal. In benchmarks, DoRA consistently outperforms LoRA by 1-3% on commonsense reasoning tasks.",id:"note-dora-overhead"}),e.jsx(i,{title:"Variant Compatibility",content:"Not all variants are supported by all frameworks. Check your PEFT version: DoRA requires >= 0.10, rsLoRA requires >= 0.9. Unsloth supports all variants natively. Some variants may interact: you can combine rsLoRA + DoRA but should test carefully.",id:"warning-compatibility"})]})}const _e=Object.freeze(Object.defineProperty({__proto__:null,default:g},Symbol.toStringTag,{value:"Module"}));function h(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Adapter Methods Compared"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Beyond LoRA, several other parameter-efficient methods exist for adapting pretrained models. Each makes different tradeoffs between parameter count, inference overhead, and adaptation quality. This section compares the most practical approaches."}),e.jsx(o,{title:"Adapter Layers",definition:"Adapter layers are small bottleneck modules inserted between existing transformer layers. They consist of a down-projection $W_{\\\\text{down}} \\\\in \\\\mathbb{R}^{d \\\\times r}$, a nonlinearity, and an up-projection $W_{\\\\text{up}} \\\\in \\\\mathbb{R}^{r \\\\times d}$, with a residual connection: $h = h + f(h W_{\\\\text{down}}) W_{\\\\text{up}}$.",id:"def-adapter-layers"}),e.jsx(o,{title:"Prefix Tuning",definition:"Prefix tuning prepends learnable virtual tokens to the key and value matrices at each attention layer. For prefix length $l$ and model dimension $d$, it adds $2ld$ trainable parameters per layer (for keys and values).",id:"def-prefix-tuning"}),e.jsx(n,{title:"Method Comparison for 7B Model",problem:"Compare parameter counts and inference overhead for different PEFT methods on a 7B model.",steps:[{formula:"\\text{LoRA (r=16)}: \\approx 42\\text{M params, 0\\% inference overhead}",explanation:"LoRA weights can be merged into base weights for zero-cost inference."},{formula:"\\text{Adapters (r=64)}: \\approx 50\\text{M params, 5-10\\% overhead}",explanation:"Adapter layers add serial computation that cannot be removed at inference."},{formula:"\\text{Prefix (l=20)}: \\approx 10\\text{M params, 2-5\\% overhead}",explanation:"Virtual tokens increase effective sequence length slightly."},{formula:"\\text{IA3}: \\approx 0.5\\text{M params, 0\\% overhead}",explanation:"Learned rescaling vectors; extremely parameter-efficient but limited capacity."}],id:"example-method-comparison"}),e.jsx(t,{title:"peft_methods_comparison.py",code:`from peft import (
    LoraConfig,
    PrefixTuningConfig,
    IA3Config,
    AdaLoraConfig,
    get_peft_model,
    TaskType,
)

# --- LoRA: Most popular, zero inference overhead ---
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules="all-linear",
    task_type=TaskType.CAUSAL_LM,
)

# --- Prefix Tuning: Prepends virtual tokens ---
prefix_config = PrefixTuningConfig(
    num_virtual_tokens=20,
    task_type=TaskType.CAUSAL_LM,
)

# --- IA3: Minimal parameters ---
ia3_config = IA3Config(
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],
    task_type=TaskType.CAUSAL_LM,
)

# --- AdaLoRA: Adaptive rank allocation ---
adalora_config = AdaLoraConfig(
    init_r=12,           # Initial rank
    target_r=4,          # Target average rank after pruning
    beta1=0.85,
    beta2=0.85,
    tinit=200,           # Steps before pruning starts
    tfinal=1000,         # Steps when pruning ends
    deltaT=10,           # Pruning interval
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)

# Print configs
configs = {
    "LoRA": lora_config,
    "Prefix": prefix_config,
    "IA3": ia3_config,
    "AdaLoRA": adalora_config,
}
for name, cfg in configs.items():
    print(f"\\n{name}:")
    print(f"  Type: {type(cfg).__name__}")
    if hasattr(cfg, 'r'):
        print(f"  Rank: {cfg.r}")`,id:"code-peft-methods"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Practical Recommendation Flow"}),e.jsx(t,{title:"choose_method.py",code:`def recommend_peft_method(
    gpu_vram_gb: float,
    model_params_b: float,
    dataset_size: int,
    need_multi_adapter: bool = False,
    inference_latency_critical: bool = True,
):
    """Recommend a PEFT method based on constraints."""

    # Estimate if QLoRA fits
    qlora_mem = 0.5 * model_params_b + 1.5  # base 4-bit + overhead
    lora_mem = 2 * model_params_b + 3       # base fp16 + overhead

    recommendations = []

    if gpu_vram_gb >= lora_mem:
        method = "LoRA (fp16/bf16)"
        reason = "Enough VRAM for full-precision LoRA - faster training"
    elif gpu_vram_gb >= qlora_mem:
        method = "QLoRA (4-bit)"
        reason = "4-bit quantization fits within VRAM"
    else:
        method = "QLoRA with gradient checkpointing"
        reason = "Tight on memory - enable all memory optimizations"

    rank = 16 if dataset_size < 10000 else 32 if dataset_size < 50000 else 64

    print(f"Recommended: {method}")
    print(f"Reason: {reason}")
    print(f"Suggested rank: {rank}")
    print(f"Estimated VRAM: {qlora_mem:.0f}-{lora_mem:.0f} GB")

    if need_multi_adapter:
        print("Tip: LoRA adapters can be hot-swapped at inference time")
    if inference_latency_critical:
        print("Tip: Merge LoRA weights after training for zero overhead")

# Example usage
recommend_peft_method(
    gpu_vram_gb=24, model_params_b=8, dataset_size=5000
)`,id:"code-recommend"}),e.jsx(a,{type:"tip",title:"LoRA Wins in Practice",content:"Despite the variety of PEFT methods, LoRA (and its variants QLoRA, DoRA) dominates in practice due to: (1) zero inference overhead after merging, (2) simple hyperparameter tuning, (3) compatibility with all model architectures, and (4) excellent framework support. Start with LoRA unless you have a specific reason to use another method.",id:"note-lora-wins"}),e.jsx(i,{title:"Adapter Serving Complexity",content:"Methods that cannot be merged (prefix tuning, adapter layers) add inference overhead and complicate serving. If you need to serve multiple adapters, LoRA is ideal: adapters are small files that can be loaded/swapped dynamically, and libraries like LoRAX and S-LoRA enable efficient multi-adapter serving.",id:"warning-serving"})]})}const xe=Object.freeze(Object.defineProperty({__proto__:null,default:h},Symbol.toStringTag,{value:"Module"}));function _(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Why Unsloth: 2x Faster, 60% Less Memory"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Unsloth is an open-source library that optimizes LLM finetuning through custom Triton kernels and intelligent memory management. It provides dramatic speedups and memory reductions while maintaining full compatibility with the Hugging Face ecosystem."}),e.jsx(o,{title:"Unsloth",definition:"Unsloth is a finetuning acceleration library that rewrites key operations (cross-entropy loss, RoPE embeddings, attention) in custom Triton kernels. It achieves 2x training speedup and 60% memory reduction compared to standard Hugging Face training, with zero accuracy loss.",id:"def-unsloth"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"How Unsloth Achieves Its Speedups"}),e.jsx(n,{title:"Unsloth Optimizations",problem:"What specific optimizations does Unsloth apply compared to standard HF training?",steps:[{formula:"\\text{Fused cross-entropy: } O(V) \\rightarrow O(1) \\text{ memory}",explanation:"Standard cross-entropy materializes logits for full vocabulary (~128K). Unsloth computes loss in chunks."},{formula:"\\text{Manual autograd: no intermediate storage}",explanation:"Custom backward passes skip PyTorch autograd overhead, saving memory."},{formula:"\\text{Fused RoPE + RMSNorm kernels}",explanation:"Operations are fused into single Triton kernels, reducing memory transfers."},{formula:"\\text{Intelligent gradient checkpointing}",explanation:"Selective checkpointing of only the most memory-intensive operations."}],id:"example-optimizations"}),e.jsx(t,{title:"unsloth_vs_standard.py",code:`# Memory comparison: Standard HF vs Unsloth
# For LLaMA 3.1 8B with QLoRA, batch_size=2, seq_len=2048

standard_hf = {
    "base_model_4bit": 5.0,      # GB
    "lora_adapters": 0.3,        # GB
    "optimizer_states": 0.6,     # GB (8-bit paged AdamW)
    "gradients": 0.3,            # GB
    "activations": 4.8,          # GB (gradient checkpointing)
    "logits_buffer": 2.0,        # GB (vocab_size * batch * seq)
    "total": 13.0,               # GB
}

unsloth = {
    "base_model_4bit": 5.0,      # GB (same)
    "lora_adapters": 0.3,        # GB (same)
    "optimizer_states": 0.6,     # GB (same)
    "gradients": 0.3,            # GB (same)
    "activations": 1.5,          # GB (optimized checkpointing)
    "logits_buffer": 0.01,       # GB (chunked cross-entropy)
    "total": 7.7,                # GB
}

print("Memory Comparison (LLaMA 3.1 8B QLoRA):")
print(f"{'Component':<25} {'Standard':>10} {'Unsloth':>10} {'Savings':>10}")
print("-" * 60)
for key in standard_hf:
    std = standard_hf[key]
    uns = unsloth[key]
    savings = (1 - uns/std) * 100 if std > 0 else 0
    print(f"{key:<25} {std:>9.1f}G {uns:>9.1f}G {savings:>9.0f}%")

# Output:
# Total savings: ~40% memory, allowing larger batch sizes
# Speed: ~2x faster due to fused kernels and reduced memory ops`,id:"code-comparison"}),e.jsx(t,{title:"unsloth_quick_start.py",code:`from unsloth import FastLanguageModel
import torch

# Unsloth provides a simplified API
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    dtype=None,          # Auto-detect (bf16 on Ampere+)
    load_in_4bit=True,   # QLoRA
)

# Add LoRA with Unsloth optimizations
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,       # Unsloth recommends 0 for speed
    bias="none",
    use_gradient_checkpointing="unsloth",  # Custom checkpointing
    use_rslora=False,
)

# Check optimizations applied
print(f"Model type: {type(model)}")
print(f"Unsloth patches applied: True")
model.print_trainable_parameters()`,id:"code-quick-start"}),e.jsx(a,{type:"note",title:"Supported Models",content:"Unsloth supports LLaMA 1/2/3/3.1/3.2/3.3, Mistral, Mixtral, Phi-3/4, Gemma 1/2, Qwen 2/2.5, DeepSeek, and many more architectures. New models are typically supported within days of release. Check the Unsloth GitHub for the current list.",id:"note-supported-models"}),e.jsx(a,{type:"tip",title:"Free Colab Notebooks",content:"Unsloth maintains ready-to-run Google Colab notebooks for every supported model. These are the fastest way to start finetuning: search 'unsloth [model-name] colab' to find the appropriate notebook. The free T4 GPU tier works for 7-8B models with QLoRA.",id:"note-colab"}),e.jsx(i,{title:"Single-GPU Only",content:"Unsloth currently only supports single-GPU training. For multi-GPU setups, use standard Hugging Face with FSDP or DeepSpeed, or use Axolotl/LLaMA-Factory which handle distributed training. Unsloth's efficiency often makes single-GPU sufficient for models up to 70B with QLoRA.",id:"warning-single-gpu"})]})}const ye=Object.freeze(Object.defineProperty({__proto__:null,default:_},Symbol.toStringTag,{value:"Module"}));function x(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Installing Unsloth"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Unsloth installation requires matching the correct version to your CUDA toolkit and PyTorch build. This section covers installation for local machines, cloud instances, and Google Colab."}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Installation Methods"}),e.jsx(t,{title:"install_unsloth.sh",code:`# Method 1: pip install (recommended for most users)
# First, install PyTorch with your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Then install Unsloth
pip install "unsloth[cu124] @ git+https://github.com/unslothai/unsloth.git"

# Method 2: For CUDA 12.1
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"

# Method 3: Conda environment (clean install)
conda create -n unsloth python=3.11 -y
conda activate unsloth
conda install pytorch-cuda=12.4 pytorch cudatoolkit -c pytorch -c nvidia -y
pip install "unsloth[cu124] @ git+https://github.com/unslothai/unsloth.git"

# Method 4: Google Colab (run in first cell)
# %%capture
# !pip install unsloth
# Also installs xformers for additional speed on Colab GPUs`,id:"code-install"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Verifying Installation"}),e.jsx(t,{title:"verify_unsloth.py",code:`# Verify Unsloth installation
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test Unsloth import
from unsloth import FastLanguageModel
print("Unsloth imported successfully!")

# Check Triton (needed for custom kernels)
try:
    import triton
    print(f"Triton: {triton.__version__}")
except ImportError:
    print("WARNING: Triton not found - some optimizations disabled")

# Check key dependencies
import transformers, peft, trl, bitsandbytes
print(f"transformers: {transformers.__version__}")
print(f"peft: {peft.__version__}")
print(f"trl: {trl.__version__}")
print(f"bitsandbytes: {bitsandbytes.__version__}")

# Quick model load test (downloads small model)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/tinyllama",  # Small test model
    max_seq_length=512,
    load_in_4bit=True,
)
print(f"\\nTest model loaded: {model.get_memory_footprint()/1e6:.0f} MB")
print("Installation verified!")`,id:"code-verify"}),e.jsx(n,{title:"Troubleshooting Common Installation Issues",problem:"How to fix the most common Unsloth installation errors?",steps:[{formula:"\\texttt{ImportError: triton}",explanation:"Install triton: pip install triton. Required for custom kernels."},{formula:"\\texttt{bitsandbytes CUDA error}",explanation:"Ensure CUDA version matches: pip install bitsandbytes --force-reinstall."},{formula:"\\texttt{xformers compatibility}",explanation:"Unsloth bundles its own kernels; xformers is optional. Remove if conflicting."},{formula:"\\texttt{CUDA out of memory on import}",explanation:"Another process may be using the GPU. Run nvidia-smi to check and kill stale processes."}],id:"example-troubleshooting"}),e.jsx(a,{type:"tip",title:"Pre-built Docker Images",content:"For reproducible environments, use the Unsloth Docker image: docker pull unsloth/unsloth. This includes all dependencies pre-configured. Alternatively, RunPod and Lambda Labs offer Unsloth as a pre-installed template.",id:"note-docker"}),e.jsx(i,{title:"Version Pinning",content:"Unsloth updates frequently. For production workflows, pin the commit hash: pip install 'unsloth @ git+https://github.com/unslothai/unsloth.git@abc123'. This prevents breaking changes from affecting your training pipeline.",id:"warning-version-pin"}),e.jsx(a,{type:"note",title:"Hugging Face Token",content:"Many models (LLaMA, Mistral) require accepting a license on Hugging Face. Run: huggingface-cli login and enter your token. Or set the environment variable: export HF_TOKEN=hf_xxxxx. Without this, model downloads will fail with a 401 error.",id:"note-hf-token"})]})}const be=Object.freeze(Object.defineProperty({__proto__:null,default:x},Symbol.toStringTag,{value:"Module"}));function y(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Finetuning LLaMA with Unsloth"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"This section provides a complete, end-to-end walkthrough of finetuning LLaMA 3.1 8B Instruct using Unsloth and QLoRA. We cover model loading, dataset formatting, training configuration, and saving the final model."}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Step 1: Load Model and Tokenizer"}),e.jsx(t,{title:"step1_load_model.py",code:`from unsloth import FastLanguageModel
import torch

# Load LLaMA 3.1 8B Instruct in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=2048,      # Context length for training
    dtype=None,                # Auto: bf16 on Ampere+, fp16 otherwise
    load_in_4bit=True,         # QLoRA 4-bit quantization
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                      # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,            # 0 is optimized in Unsloth
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% longer context
    random_state=42,
)

model.print_trainable_parameters()
# trainable: ~42M / 8B total (~0.52%)`,id:"code-load-model"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Step 2: Prepare Dataset"}),e.jsx(t,{title:"step2_prepare_dataset.py",code:`from datasets import load_dataset

# Load an instruction-following dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
print(f"Dataset size: {len(dataset)}")
print(f"Columns: {dataset.column_names}")
# Columns: ['instruction', 'input', 'output']

# Define chat template for LLaMA 3.1 Instruct
llama3_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""

def format_example(example):
    input_text = f"\\n\\n{example['input']}" if example['input'] else ""
    text = llama3_template.format(
        instruction=example['instruction'],
        input_text=input_text,
        output=example['output'],
    )
    return {"text": text}

dataset = dataset.map(format_example)
print(f"\\nSample formatted text:\\n{dataset[0]['text'][:300]}...")`,id:"code-prepare-dataset"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Step 3: Configure and Train"}),e.jsx(t,{title:"step3_train.py",code:`from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=True,              # Pack short sequences together
    args=TrainingArguments(
        output_dir="./llama3-finetune",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch = 8
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        save_strategy="steps",
        save_steps=200,
        report_to="wandb",        # Optional: WandB logging
    ),
)

# Show GPU memory before training
gpu_stats = torch.cuda.get_device_properties(0)
reserved = torch.cuda.max_memory_reserved() / 1e9
print(f"GPU: {gpu_stats.name}, Reserved: {reserved:.1f} GB")

# Train!
trainer_stats = trainer.train()
print(f"Training time: {trainer_stats.metrics['train_runtime']:.0f}s")
print(f"Training loss: {trainer_stats.metrics['train_loss']:.4f}")`,id:"code-train"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Step 4: Save and Test"}),e.jsx(t,{title:"step4_save_test.py",code:`# Save LoRA adapter
model.save_pretrained("./llama3-lora-adapter")
tokenizer.save_pretrained("./llama3-lora-adapter")

# Test the finetuned model
FastLanguageModel.for_inference(model)  # Enable fast inference

messages = [
    {"role": "user", "content": "Explain quantum entanglement simply."},
]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs, max_new_tokens=256,
    temperature=0.7, top_p=0.9,
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# Save merged model (optional - for deployment)
# model.save_pretrained_merged("./llama3-merged", tokenizer)
# model.save_pretrained_gguf("./llama3-gguf", tokenizer, "q4_k_m")`,id:"code-save-test"}),e.jsx(a,{type:"tip",title:"Packing for Efficiency",content:"Setting packing=True in SFTTrainer concatenates multiple short examples into single sequences, avoiding wasted padding tokens. This can speed up training by 2-5x on datasets with variable-length examples. Unsloth's packing implementation is particularly efficient.",id:"note-packing"}),e.jsx(i,{title:"Chat Template Matters",content:"Using the wrong chat template will severely degrade model quality. Always use the template matching your base model: LLaMA 3 uses <|begin_of_text|> tags, Mistral uses [INST] tags, ChatML uses <|im_start|> tags. The tokenizer.apply_chat_template() method handles this automatically.",id:"warning-chat-template"})]})}const ve=Object.freeze(Object.defineProperty({__proto__:null,default:y},Symbol.toStringTag,{value:"Module"}));function b(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Finetuning Mistral with Unsloth"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Mistral 7B and its variants (Mistral NeMo, Mistral Small) are popular choices for finetuning due to their strong performance-to-size ratio. This walkthrough shows how to finetune Mistral models using Unsloth, highlighting differences from LLaMA finetuning."}),e.jsx(o,{title:"Mistral Architecture Differences",definition:"Mistral uses Grouped Query Attention (GQA) with a sliding window attention mechanism (window size 4096 in Mistral 7B). It uses a different chat template with [INST] and [/INST] tokens. The tokenizer vocabulary is 32,000 tokens compared to LLaMA 3's 128,000.",id:"def-mistral-arch"}),e.jsx(t,{title:"finetune_mistral_complete.py",code:`from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Step 1: Load Mistral model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.3",
    max_seq_length=4096,       # Mistral supports 32K but train at 4K
    dtype=None,
    load_in_4bit=True,
)

# Step 2: Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=32,                      # Slightly higher rank for Mistral
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
model.print_trainable_parameters()

# Step 3: Format dataset with Mistral chat template
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:5000]")

def format_mistral(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_mistral)
print(f"Sample: {dataset[0]['text'][:200]}...")

# Step 4: Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    packing=True,
    args=TrainingArguments(
        output_dir="./mistral-finetune",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        optim="adamw_8bit",
        save_strategy="steps",
        save_steps=500,
        seed=42,
    ),
)
trainer.train()`,id:"code-mistral-finetune"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Mistral Chat Template"}),e.jsx(t,{title:"mistral_chat_template.py",code:`# Mistral Instruct v0.3 chat format
# The tokenizer handles this automatically with apply_chat_template

# Manual format (for reference):
mistral_template = """<s>[INST] {system_message}

{user_message} [/INST] {assistant_message}</s>"""

# With the tokenizer:
messages = [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "Write a Python function to sort a list."},
    {"role": "assistant", "content": "Here is a simple sort function:\\n\\ndef sort_list(lst):\\n    return sorted(lst)"},
]

# Let the tokenizer format correctly
formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
print(formatted)

# For multi-turn conversations:
multi_turn = [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user", "content": "Give me an example."},
    {"role": "assistant", "content": "print('Hello, World!')"},
]
formatted_multi = tokenizer.apply_chat_template(
    multi_turn, tokenize=False, add_generation_prompt=False
)
print(f"\\nMulti-turn:\\n{formatted_multi}")`,id:"code-mistral-template"}),e.jsx(n,{title:"Mistral vs LLaMA Finetuning Differences",problem:"What changes are needed when switching from LLaMA to Mistral finetuning?",steps:[{formula:"\\text{Chat template: [INST]...[/INST] vs <|...|>}",explanation:"Different special tokens. Always use tokenizer.apply_chat_template()."},{formula:"\\text{Vocab size: 32K vs 128K}",explanation:"Smaller vocab means less memory for logits computation."},{formula:"\\text{GQA: 8 KV heads vs 8 KV heads}",explanation:"Both use GQA, but Mistral has sliding window attention."},{formula:"\\text{LoRA rank: 32 recommended for Mistral}",explanation:"Slightly higher rank compensates for the smaller model architecture."}],id:"example-mistral-vs-llama"}),e.jsx(a,{type:"tip",title:"Mistral NeMo 12B",content:"Mistral NeMo 12B (released with NVIDIA) offers better quality than Mistral 7B with the same QLoRA memory footprint on 24 GB GPUs. It uses a larger 128K vocabulary with tiktoken-style BPE. Use model_name='unsloth/Mistral-Nemo-Instruct-2407' with Unsloth.",id:"note-nemo"}),e.jsx(i,{title:"Sliding Window Attention in Training",content:"Mistral's sliding window attention (4096 tokens) means the model cannot attend to earlier tokens beyond the window. When finetuning on long documents, be aware that the model may lose context. For long-context tasks, consider LLaMA 3.1 (128K context) instead.",id:"warning-sliding-window"})]})}const we=Object.freeze(Object.defineProperty({__proto__:null,default:b},Symbol.toStringTag,{value:"Module"}));function v(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Custom Dataset Preparation for Unsloth"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Real-world finetuning requires working with your own data. This section covers how to prepare custom datasets in the formats that Unsloth and SFTTrainer expect, including single-turn instructions, multi-turn conversations, and completion-only training."}),e.jsx(o,{title:"Dataset Formats",definition:"SFTTrainer accepts datasets in two main formats: (1) a 'text' field containing the fully formatted conversation string, or (2) a 'messages' field containing a list of role/content dictionaries that will be formatted using the tokenizer's chat template.",id:"def-formats"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Format 1: Pre-formatted Text"}),e.jsx(t,{title:"custom_dataset_text.py",code:`from datasets import Dataset
import json

# Load your custom data (JSONL, CSV, etc.)
raw_data = [
    {
        "instruction": "Summarize this article about climate change.",
        "input": "Global temperatures have risen 1.1C since pre-industrial times...",
        "output": "Global temperatures increased 1.1C, with significant impacts..."
    },
    {
        "instruction": "Translate to French.",
        "input": "The weather is beautiful today.",
        "output": "Le temps est magnifique aujourd'hui."
    },
]

# Format as chat messages for LLaMA 3
def format_to_llama3(example, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example["instruction"] +
         (f"\\n\\n{example['input']}" if example.get("input") else "")},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

# Create HuggingFace dataset
dataset = Dataset.from_list(raw_data)
# dataset = dataset.map(lambda x: format_to_llama3(x, tokenizer))
print(f"Dataset: {len(dataset)} examples")
print(f"Columns: {dataset.column_names}")`,id:"code-text-format"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Format 2: Messages (Recommended)"}),e.jsx(t,{title:"custom_dataset_messages.py",code:`from datasets import Dataset
import json

# Multi-turn conversation format
conversations = [
    {
        "messages": [
            {"role": "system", "content": "You are a Python tutor."},
            {"role": "user", "content": "How do I read a file in Python?"},
            {"role": "assistant", "content": "Use the open() function with a context manager:\\n\\nwith open('file.txt', 'r') as f:\\n    content = f.read()"},
            {"role": "user", "content": "What about reading line by line?"},
            {"role": "assistant", "content": "Use readlines() or iterate:\\n\\nwith open('file.txt', 'r') as f:\\n    for line in f:\\n        print(line.strip())"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Explain list comprehensions."},
            {"role": "assistant", "content": "List comprehensions create lists concisely:\\n\\nsquares = [x**2 for x in range(10)]\\n\\nThis is equivalent to a for loop with append."},
        ]
    },
]

dataset = Dataset.from_list(conversations)

# Use with SFTTrainer (messages format handled automatically)
# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     # No dataset_text_field needed - uses 'messages' automatically
# )

# Loading from JSONL file
def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)

# Loading from CSV/Parquet
# dataset = Dataset.from_csv("data.csv")
# dataset = Dataset.from_parquet("data.parquet")

print(f"Dataset: {len(dataset)} conversations")
print(f"First conversation turns: {len(dataset[0]['messages'])}")`,id:"code-messages-format"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Completion-Only Training"}),e.jsx(t,{title:"completion_only_training.py",code:`from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# Train only on assistant responses, not user prompts
# This prevents the model from learning to generate user messages

# Define the response template (text that marks start of assistant response)
response_template = "<|start_header_id|>assistant<|end_header_id|>\\n\\n"

# Create data collator that masks non-response tokens
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

# Use in SFTTrainer
# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     dataset_text_field="text",
#     data_collator=collator,    # Masks loss on user/system tokens
#     max_seq_length=2048,
# )

# For Mistral format:
# response_template_mistral = "[/INST]"

# Verify masking works correctly
sample_text = dataset[0]["text"]
tokens = tokenizer(sample_text, return_tensors="pt")
# Labels should be -100 (masked) for non-response tokens
print(f"Total tokens: {tokens.input_ids.shape[1]}")
print("Completion-only training: loss computed only on assistant responses")`,id:"code-completion-only"}),e.jsx(n,{title:"Dataset Size Recommendations",problem:"How much data do you need for different finetuning goals?",steps:[{formula:"\\text{Style/persona: } 100\\text{-}500 \\text{ examples}",explanation:"Teaching a model a specific writing style or persona requires few examples."},{formula:"\\text{Task-specific: } 1\\text{K-}10\\text{K examples}",explanation:"Domain-specific tasks like summarization or classification need moderate data."},{formula:"\\text{General instruction: } 10\\text{K-}100\\text{K examples}",explanation:"Broad instruction-following improvement needs substantial diverse data."}],id:"example-dataset-size"}),e.jsx(a,{type:"tip",title:"Data Quality Over Quantity",content:"500 high-quality, diverse examples often outperform 50,000 noisy ones. Focus on: (1) accurate and helpful responses, (2) diverse instruction types, (3) consistent formatting, and (4) no contradictions. Manually review at least 100 random examples before training.",id:"note-quality"}),e.jsx(i,{title:"Tokenization Length Limits",content:"Examples longer than max_seq_length are silently truncated. Check your dataset: if many examples exceed the limit, increase max_seq_length or split long examples. Use dataset.map(lambda x: {'length': len(tokenizer(x['text']).input_ids)}) to check lengths.",id:"warning-truncation"})]})}const ke=Object.freeze(Object.defineProperty({__proto__:null,default:v},Symbol.toStringTag,{value:"Module"}));function w(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Monitoring Training: WandB and Loss Curves"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Monitoring training metrics is essential for diagnosing issues and knowing when to stop. Weights and Biases (WandB) provides real-time dashboards, while understanding loss curves helps you detect overfitting, underfitting, and convergence."}),e.jsx(o,{title:"Training Loss",definition:"The cross-entropy loss on training batches, measuring how well the model predicts the next token. For a sequence of $T$ tokens, the loss is $L = -\\\\frac{1}{T}\\\\sum_{t=1}^{T} \\\\log P(x_t | x_{<t})$. Lower is better, but the absolute value depends on the dataset and tokenizer.",id:"def-training-loss"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Setting Up WandB"}),e.jsx(t,{title:"setup_wandb.py",code:`import wandb
import os

# Option 1: Login interactively
wandb.login()

# Option 2: Set API key as environment variable
os.environ["WANDB_API_KEY"] = "your-key-here"
os.environ["WANDB_PROJECT"] = "llama3-finetune"

# Initialize a run
run = wandb.init(
    project="llama3-finetune",
    name="qlora-r16-alpaca",
    config={
        "model": "Meta-Llama-3.1-8B-Instruct",
        "method": "QLoRA",
        "rank": 16,
        "alpha": 16,
        "lr": 2e-4,
        "batch_size": 8,
        "epochs": 1,
        "dataset": "alpaca-cleaned",
        "dataset_size": 51760,
    },
)

# WandB integrates with TrainingArguments automatically
# Just set report_to="wandb" in TrainingArguments
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./output",
    report_to="wandb",          # Enable WandB logging
    logging_steps=10,            # Log every 10 steps
    # ... other args
)`,id:"code-wandb-setup"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Interpreting Loss Curves"}),e.jsx(t,{title:"analyze_loss_curves.py",code:`import matplotlib.pyplot as plt
import numpy as np

# Simulated training scenarios
steps = np.arange(0, 1000)

# Scenario 1: Good training (smooth decrease, slight plateau)
good_loss = 2.5 * np.exp(-steps/200) + 0.8 + np.random.normal(0, 0.02, len(steps))

# Scenario 2: Overfitting (train decreases, val increases)
overfit_train = 2.5 * np.exp(-steps/100) + 0.3
overfit_val = 2.0 * np.exp(-steps/300) + 0.7 + 0.001 * steps

# Scenario 3: Learning rate too high (unstable)
unstable = 2.5 * np.exp(-steps/300) + 0.8 + 0.5 * np.sin(steps/20)

# Key metrics to monitor:
metrics = {
    "train/loss": "Should decrease smoothly. Spikes = bad batch or LR too high",
    "train/learning_rate": "Should follow warmup -> decay schedule",
    "train/grad_norm": "Should be stable. Spikes > 10 = gradient explosion",
    "eval/loss": "Compare to train loss. If diverges = overfitting",
    "train/epoch": "Track progress through dataset",
}

for metric, desc in metrics.items():
    print(f"{metric}: {desc}")

# When to stop training:
# 1. Train loss plateaus for >50 steps
# 2. Eval loss starts increasing (overfitting)
# 3. Grad norm explodes (reduce LR or check data)
# 4. Reached target loss (typically 0.5-1.0 for instruction tuning)`,id:"code-loss-curves"}),e.jsx(n,{title:"Diagnosing Common Training Issues",problem:"What do different loss curve patterns indicate?",steps:[{formula:"\\text{Loss stuck at } \\sim 2.5",explanation:"Model not learning. Check: wrong chat template, LR too low, data not formatted correctly."},{formula:"\\text{Loss oscillating wildly}",explanation:"Learning rate too high. Reduce by 2-5x. Also check batch size and gradient accumulation."},{formula:"\\text{Loss drops then rises}",explanation:"Overfitting. Reduce epochs, increase dropout, or add more diverse data."},{formula:"\\text{Loss = NaN or Inf}",explanation:"Numerical instability. Switch to bf16, reduce LR, enable gradient clipping (max_grad_norm=1.0)."}],id:"example-diagnosis"}),e.jsx(t,{title:"custom_logging_callback.py",code:`from transformers import TrainerCallback
import torch

class MemoryMonitorCallback(TrainerCallback):
    """Log GPU memory usage during training."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.max_memory_reserved() / 1e9
            if logs is not None:
                logs["gpu_allocated_gb"] = round(allocated, 2)
                logs["gpu_reserved_gb"] = round(reserved, 2)

    def on_step_end(self, args, state, control, **kwargs):
        # Check for gradient issues
        if state.log_history and len(state.log_history) > 0:
            last_log = state.log_history[-1]
            if "loss" in last_log and last_log["loss"] > 10:
                print(f"WARNING: Loss spike at step {state.global_step}: "
                      f"{last_log['loss']:.4f}")

# Add to trainer
# trainer = SFTTrainer(..., callbacks=[MemoryMonitorCallback()])`,id:"code-callback"}),e.jsx(a,{type:"tip",title:"Expected Loss Values",content:"For instruction tuning on clean data: initial loss ~2.0-3.0, final loss ~0.5-1.0. If final loss drops below 0.3, you may be overfitting. For continued pretraining on domain text: initial ~3.0-4.0, final ~1.5-2.5. These are rough guidelines -- actual values depend on the dataset and tokenizer.",id:"note-expected-loss"}),e.jsx(i,{title:"Do Not Over-optimize Training Loss",content:"A very low training loss does not mean a good model. Overfitting to training data causes the model to memorize examples rather than learn patterns. Always evaluate on held-out data and with qualitative tests. Use early stopping or train for 1-3 epochs maximum.",id:"warning-overfit"})]})}const Ae=Object.freeze(Object.defineProperty({__proto__:null,default:w},Symbol.toStringTag,{value:"Module"}));function k(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"SFTTrainer: Supervised Finetuning with TRL"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"TRL (Transformer Reinforcement Learning) provides SFTTrainer, the standard tool for supervised finetuning of language models. It extends the Hugging Face Trainer with features like sequence packing, chat template handling, and completion-only training."}),e.jsx(o,{title:"Supervised Finetuning (SFT)",definition:"SFT trains a language model on input-output pairs using the standard next-token prediction objective. Given a prompt $x$ and desired response $y$, the loss is $L = -\\\\sum_{t=1}^{|y|} \\\\log P_\\\\theta(y_t | x, y_{<t})$. This is the most common finetuning approach.",id:"def-sft"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Basic SFTTrainer Usage"}),e.jsx(t,{title:"sft_basic.py",code:`from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset in messages format
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:5000]")

# SFTConfig replaces TrainingArguments in newer TRL
sft_config = SFTConfig(
    output_dir="./sft-output",
    max_seq_length=2048,
    packing=True,                    # Pack multiple examples per sequence
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    optim="adamw_torch",
)

# Create trainer - handles chat template automatically
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()`,id:"code-sft-basic"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"SFTTrainer with QLoRA"}),e.jsx(t,{title:"sft_qlora.py",code:`from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# SFTTrainer handles PEFT integration automatically
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./sft-qlora",
        max_seq_length=2048,
        packing=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,       # Higher LR for LoRA
        bf16=True,
        logging_steps=10,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
    ),
    peft_config=peft_config,      # Pass PEFT config directly!
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.model.print_trainable_parameters()
trainer.train()`,id:"code-sft-qlora"}),e.jsx(n,{title:"SFTTrainer Key Parameters",problem:"What are the most important SFTTrainer configuration options?",steps:[{formula:"\\texttt{max\\_seq\\_length}",explanation:"Maximum token length. Longer = more VRAM. Match to your data distribution."},{formula:"\\texttt{packing=True}",explanation:"Concatenates short examples into full sequences. 2-5x throughput improvement."},{formula:"\\texttt{dataset\\_text\\_field}",explanation:'Column name for pre-formatted text. Or use "messages" column for auto-formatting.'},{formula:"\\texttt{peft\\_config}",explanation:"Pass LoRA config directly -- SFTTrainer applies it and handles prepare_model_for_kbit_training."}],id:"example-sft-params"}),e.jsx(a,{type:"tip",title:"Using the Messages Format",content:"When your dataset has a 'messages' column (list of role/content dicts), SFTTrainer automatically applies the tokenizer's chat template. This is the cleanest approach -- no manual template formatting needed. Just ensure your tokenizer has a chat_template attribute.",id:"note-messages-format"}),e.jsx(i,{title:"Packing and Completion-Only Are Incompatible",content:"You cannot use packing=True with DataCollatorForCompletionOnlyLM. Packing concatenates examples, destroying the boundary between prompt and response. If you need completion-only training (masking prompt tokens), set packing=False.",id:"warning-packing-completion"})]})}const Le=Object.freeze(Object.defineProperty({__proto__:null,default:k},Symbol.toStringTag,{value:"Module"}));function A(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"DPO Training with TRL"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Direct Preference Optimization (DPO) aligns language models with human preferences without training a separate reward model. It directly optimizes the policy using pairs of preferred and rejected responses, making it simpler and more stable than traditional RLHF."}),e.jsx(o,{title:"DPO Loss",definition:"Given a prompt $x$, chosen response $y_w$, and rejected response $y_l$, DPO optimizes: $L_{\\\\text{DPO}} = -\\\\log \\\\sigma\\\\left(\\\\beta \\\\left[\\\\log \\\\frac{\\\\pi_\\\\theta(y_w|x)}{\\\\pi_{\\\\text{ref}}(y_w|x)} - \\\\log \\\\frac{\\\\pi_\\\\theta(y_l|x)}{\\\\pi_{\\\\text{ref}}(y_l|x)}\\\\right]\\\\right)$ where $\\\\pi_{\\\\text{ref}}$ is the reference (SFT) model and $\\\\beta$ controls the deviation penalty.",notation:"L_{\\text{DPO}} = -\\log \\sigma(\\beta[\\log r_w - \\log r_l])",id:"def-dpo"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"DPO Dataset Format"}),e.jsx(t,{title:"dpo_dataset.py",code:`from datasets import Dataset

# DPO requires: prompt, chosen response, rejected response
dpo_data = [
    {
        "prompt": "Explain recursion in programming.",
        "chosen": "Recursion is when a function calls itself to solve smaller subproblems. For example, calculating factorial: fact(n) = n * fact(n-1), with base case fact(0) = 1. Each call reduces the problem until hitting the base case, then results build back up.",
        "rejected": "Recursion is a programming concept. It means something calls itself. It's used in many algorithms.",
    },
    {
        "prompt": "What causes seasons on Earth?",
        "chosen": "Seasons are caused by Earth's 23.5-degree axial tilt. As Earth orbits the Sun, different hemispheres receive more direct sunlight at different times. When the Northern Hemisphere tilts toward the Sun, it experiences summer while the Southern Hemisphere has winter.",
        "rejected": "Seasons happen because Earth gets closer to or farther from the Sun during its orbit.",
    },
]

# Or load an existing preference dataset
from datasets import load_dataset
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned",
                       split="train[:5000]")
print(f"Columns: {dataset.column_names}")
# Typically: prompt, chosen, rejected (may need reformatting)`,id:"code-dpo-dataset"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Training with DPOTrainer"}),e.jsx(t,{title:"dpo_training.py",code:`from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
import torch

# Load the SFT model (already finetuned with SFTTrainer)
model_name = "your-sft-model-path"  # or HF model id

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# DPO uses the same model as both policy and reference
# The reference model is created internally by DPOTrainer
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

dpo_config = DPOConfig(
    output_dir="./dpo-output",
    beta=0.1,                        # KL penalty strength
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-5,              # Lower LR than SFT
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    # DPO-specific
    loss_type="sigmoid",             # Standard DPO loss
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()`,id:"code-dpo-training"}),e.jsx(n,{title:"DPO Hyperparameter Selection",problem:"How to choose the key DPO hyperparameters?",steps:[{formula:"\\beta = 0.1\\text{-}0.5",explanation:"Controls how much the model can deviate from the reference. Lower = more deviation. Start with 0.1."},{formula:"\\text{LR} = 5 \\times 10^{-6}\\text{ to }5 \\times 10^{-5}",explanation:"DPO needs lower learning rates than SFT to avoid forgetting."},{formula:"\\text{Epochs} = 1\\text{-}3",explanation:"DPO is sensitive to overfitting. Usually 1 epoch suffices."},{formula:'\\text{loss\\_type} = \\text{"sigmoid"}',explanation:'Standard DPO. Alternatives: "hinge" (more robust), "ipo" (identity preference optimization).'}],id:"example-dpo-hyperparams"}),e.jsx(a,{type:"intuition",title:"Why DPO Over RLHF?",content:"Traditional RLHF requires training a separate reward model and then optimizing with PPO -- a complex, unstable process. DPO shows that the optimal policy has a closed-form solution given preferences, eliminating the need for RL entirely. The result is simpler code, faster training, and comparable quality.",id:"note-dpo-vs-rlhf"}),e.jsx(i,{title:"DPO Data Quality",content:"DPO is very sensitive to data quality. The chosen/rejected pairs must have meaningful quality differences. If the rejected responses are nearly as good as the chosen ones, DPO learns little. Similarly, if rejected responses are absurdly bad, the signal is too easy and the model does not improve on realistic edge cases.",id:"warning-dpo-data"})]})}const Te=Object.freeze(Object.defineProperty({__proto__:null,default:A},Symbol.toStringTag,{value:"Module"}));function L(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Reward Model Training"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"A reward model learns to score responses based on human preferences. It is the foundation of RLHF (Reinforcement Learning from Human Feedback) and can also be used for data filtering, best-of-N sampling, and response ranking."}),e.jsx(o,{title:"Reward Model",definition:"A reward model $R_\\\\phi(x, y) \\\\rightarrow \\\\mathbb{R}$ maps a prompt $x$ and response $y$ to a scalar score. It is trained on preference pairs using the Bradley-Terry loss: $L = -\\\\log \\\\sigma(R_\\\\phi(x, y_w) - R_\\\\phi(x, y_l))$ where $y_w$ is preferred over $y_l$.",notation:"L = -\\log \\sigma(R(x, y_w) - R(x, y_l))",id:"def-reward-model"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Training a Reward Model"}),e.jsx(t,{title:"train_reward_model.py",code:`from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch

# Load base model for reward modeling
# Use a smaller model than the policy for efficiency
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,              # Single scalar output
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Load preference dataset
dataset = load_dataset(
    "argilla/ultrafeedback-binarized-preferences-cleaned",
    split="train[:5000]"
)

# Format: each example needs 'chosen' and 'rejected' fields
# These should be full conversation strings
print(f"Dataset size: {len(dataset)}")
print(f"Columns: {dataset.column_names}")

# Configure reward training
reward_config = RewardConfig(
    output_dir="./reward-model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-5,
    bf16=True,
    logging_steps=10,
    max_length=1024,
    gradient_checkpointing=True,
)

trainer = RewardTrainer(
    model=model,
    args=reward_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()`,id:"code-train-reward"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Using the Reward Model"}),e.jsx(t,{title:"use_reward_model.py",code:`import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load trained reward model
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "./reward-model", torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./reward-model")

def score_response(prompt, response):
    """Score a single response given a prompt."""
    text = f"User: {prompt}\\nAssistant: {response}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=1024).to(reward_model.device)
    with torch.no_grad():
        score = reward_model(**inputs).logits[0].item()
    return score

# Best-of-N sampling: generate N responses, pick highest scored
def best_of_n(prompt, responses):
    scored = [(r, score_response(prompt, r)) for r in responses]
    scored.sort(key=lambda x: x[1], reverse=True)
    for resp, score in scored:
        print(f"Score: {score:.3f} | {resp[:80]}...")
    return scored[0][0]

# Example usage
prompt = "What are the benefits of exercise?"
responses = [
    "Exercise is good for you. You should do it.",
    "Regular exercise improves cardiovascular health, boosts mood through endorphin release, strengthens muscles and bones, and helps maintain a healthy weight. Even 30 minutes of moderate activity daily can significantly reduce the risk of chronic diseases.",
    "idk maybe google it lol",
]
best = best_of_n(prompt, responses)
print(f"\\nBest response: {best[:100]}...")`,id:"code-use-reward"}),e.jsx(n,{title:"Reward Model Architecture",problem:"How is a language model adapted into a reward model?",steps:[{formula:"\\text{Base LLM} \\rightarrow \\text{Remove LM head}",explanation:"The next-token prediction head is removed."},{formula:"\\text{Add linear head: } \\mathbb{R}^d \\rightarrow \\mathbb{R}^1",explanation:"A single linear layer maps the last hidden state to a scalar score."},{formula:"\\text{Pool over last token}",explanation:"The reward is the score at the last token position (like [CLS] in BERT)."},{formula:"\\text{Train with Bradley-Terry loss}",explanation:"The model learns that chosen responses should score higher than rejected ones."}],id:"example-rm-architecture"}),e.jsx(a,{type:"note",title:"Reward Model Size",content:"The reward model does not need to be as large as the policy model. A 3B reward model can effectively guide an 8B policy model. This saves compute during RLHF training where the reward model is called every step.",id:"note-rm-size"}),e.jsx(i,{title:"Reward Hacking",content:"When using a reward model with RL optimization, the policy may learn to exploit weaknesses in the reward model rather than genuinely improving. This is called reward hacking. Mitigations include: KL penalties, ensembles of reward models, and periodic human evaluation of high-scoring outputs.",id:"warning-reward-hacking"})]})}const je=Object.freeze(Object.defineProperty({__proto__:null,default:L},Symbol.toStringTag,{value:"Module"}));function T(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"ORPO: Odds Ratio Preference Optimization"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"ORPO combines supervised finetuning and preference alignment into a single training stage. Unlike DPO, which requires a separate SFT step first, ORPO modifies the SFT loss to incorporate preference information directly, simplifying the training pipeline."}),e.jsx(o,{title:"ORPO Loss",definition:"ORPO adds an odds ratio penalty to the standard SFT loss: $L_{\\\\text{ORPO}} = L_{\\\\text{SFT}}(y_w) + \\\\lambda \\\\cdot L_{\\\\text{OR}}$ where $L_{\\\\text{OR}} = -\\\\log \\\\sigma\\\\left(\\\\log \\\\frac{\\\\text{odds}_\\\\theta(y_w|x)}{\\\\text{odds}_\\\\theta(y_l|x)}\\\\right)$ and $\\\\text{odds}(y|x) = \\\\frac{P(y|x)}{1 - P(y|x)}$.",notation:"L_{ORPO} = L_{SFT}(y_w) + \\lambda \\cdot L_{OR}",id:"def-orpo"}),e.jsx(n,{title:"ORPO vs DPO Pipeline",problem:"Compare the training pipeline for DPO vs ORPO.",steps:[{formula:"\\text{DPO: SFT} \\rightarrow \\text{DPO (2 stages)}",explanation:"DPO requires a separate SFT training run first, then DPO alignment."},{formula:"\\text{ORPO: Single stage}",explanation:"ORPO handles both SFT and alignment simultaneously, halving training time."},{formula:"\\text{DPO needs reference model}",explanation:"DPO computes KL divergence against a frozen reference model (doubles memory)."},{formula:"\\text{ORPO: no reference model}",explanation:"ORPO uses odds ratios, eliminating the need for a reference model."}],id:"example-orpo-vs-dpo"}),e.jsx(t,{title:"orpo_training.py",code:`from trl import ORPOTrainer, ORPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from datasets import load_dataset
import torch

# Load base model (NOT an SFT model - ORPO does SFT internally)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",  # Base model, not Instruct
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token

# LoRA config
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# Load preference dataset (same format as DPO)
dataset = load_dataset(
    "argilla/ultrafeedback-binarized-preferences-cleaned",
    split="train[:10000]"
)

# ORPO configuration
orpo_config = ORPOConfig(
    output_dir="./orpo-output",
    beta=0.1,                        # Odds ratio weight (lambda)
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=8e-6,              # ORPO uses lower LR
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()

# Save the model
trainer.save_model("./orpo-final")`,id:"code-orpo-training"}),e.jsx(t,{title:"orpo_metrics.py",code:`# Key metrics to monitor during ORPO training
# These are logged automatically by ORPOTrainer

metrics_guide = {
    "train/loss": "Total ORPO loss (SFT + lambda * OR loss)",
    "train/sft_loss": "Standard next-token prediction loss on chosen",
    "train/odds_ratio_loss": "Preference alignment loss component",
    "train/log_odds_ratio": "Log odds ratio between chosen and rejected",
    "train/log_odds_chosen": "Log odds of chosen responses",
    "train/log_odds_rejected": "Log odds of rejected responses",
    "train/reward_margin": "Difference in implicit reward (chosen - rejected)",
}

for metric, description in metrics_guide.items():
    print(f"{metric}:")
    print(f"  {description}")

# Healthy training indicators:
# - sft_loss decreasing smoothly
# - log_odds_ratio increasing (chosen becoming more likely vs rejected)
# - reward_margin positive and increasing
# - odds_ratio_loss decreasing`,id:"code-orpo-metrics"}),e.jsx(a,{type:"tip",title:"When to Use ORPO",content:"ORPO is ideal when: (1) you are starting from a base model (not instruction-tuned), (2) you want to minimize training stages, (3) GPU memory is limited (no reference model needed). ORPO is particularly effective when combined with high-quality preference data like UltraFeedback.",id:"note-when-orpo"}),e.jsx(i,{title:"ORPO Learning Rate Sensitivity",content:"ORPO is more sensitive to the learning rate than DPO. Too high and the model collapses; too low and it does not align. Start with 5e-6 to 1e-5 and adjust based on the reward margin metric. If reward_margin is not increasing after 100 steps, try a higher LR.",id:"warning-orpo-lr"}),e.jsx(a,{type:"note",title:"SimPO and KTO Alternatives",content:"SimPO (Simple Preference Optimization) and KTO (Kahneman-Tversky Optimization) are newer alternatives. SimPO uses length-normalized log probabilities. KTO works with binary feedback (good/bad) without needing paired preferences. Both are available in TRL.",id:"note-alternatives"})]})}const Me=Object.freeze(Object.defineProperty({__proto__:null,default:T},Symbol.toStringTag,{value:"Module"}));function j(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"End-to-End Finetuning Walkthrough"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"This section ties together everything covered in TRL by walking through a complete finetuning pipeline: from data preparation through SFT, DPO alignment, evaluation, and model export. Follow this as a template for your own projects."}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Phase 1: Data Preparation"}),e.jsx(t,{title:"phase1_data_prep.py",code:`from datasets import load_dataset, DatasetDict
import json

# Load and split dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Train/validation split
split = dataset.train_test_split(test_size=0.05, seed=42)
train_data = split["train"]
eval_data = split["test"]
print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

# Inspect data quality
lengths = [len(ex["output"]) for ex in train_data]
print(f"Response lengths: min={min(lengths)}, max={max(lengths)}, "
      f"median={sorted(lengths)[len(lengths)//2]}")

# Filter out very short or very long examples
train_data = train_data.filter(
    lambda x: 10 < len(x["output"]) < 2000
)
print(f"After filtering: {len(train_data)}")

# Convert to messages format
def to_messages(example):
    user_msg = example["instruction"]
    if example.get("input"):
        user_msg += f"\\n\\n{example['input']}"
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": example["output"]},
        ]
    }

train_data = train_data.map(to_messages)
eval_data = eval_data.map(to_messages)`,id:"code-phase1"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Phase 2: SFT Training"}),e.jsx(t,{title:"phase2_sft.py",code:`from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import torch

# Load model with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=eval_data,
    args=SFTConfig(
        output_dir="./phase2-sft",
        max_seq_length=2048,
        packing=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        optim="adamw_8bit",
    ),
)

trainer.train()
trainer.save_model("./sft-checkpoint")`,id:"code-phase2"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Phase 3: Evaluation and Export"}),e.jsx(t,{title:"phase3_eval_export.py",code:`# Quick qualitative evaluation
FastLanguageModel.for_inference(model)

test_prompts = [
    "Explain the difference between a list and a tuple in Python.",
    "Write a haiku about machine learning.",
    "What are three tips for effective public speaking?",
]

for prompt in test_prompts:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs, max_new_tokens=256,
        temperature=0.7, top_p=0.9, do_sample=True,
    )
    response = tokenizer.decode(outputs[0][inputs.shape[1]:],
                                 skip_special_tokens=True)
    print(f"Q: {prompt}")
    print(f"A: {response}\\n{'='*60}")

# Export options
# 1. Save LoRA adapter only (small, ~100 MB)
model.save_pretrained("./final-lora-adapter")
tokenizer.save_pretrained("./final-lora-adapter")

# 2. Save merged model (full size, ~16 GB for 8B model)
# model.save_pretrained_merged("./final-merged", tokenizer)

# 3. Export to GGUF for llama.cpp
# model.save_pretrained_gguf("./final-gguf", tokenizer, "q4_k_m")

# 4. Push to Hugging Face Hub
# model.push_to_hub("your-username/your-model-name")
# tokenizer.push_to_hub("your-username/your-model-name")`,id:"code-phase3"}),e.jsx(n,{title:"Complete Pipeline Timeline",problem:"How long does each phase take for a typical 8B model finetune?",steps:[{formula:"\\text{Data prep: } 10\\text{-}30 \\text{ minutes}",explanation:"Loading, filtering, formatting. One-time cost."},{formula:"\\text{SFT (50K examples, 1 epoch): } 2\\text{-}4 \\text{ hours}",explanation:"On RTX 4090 with QLoRA + Unsloth. 1 hour on A100."},{formula:"\\text{DPO (optional, 10K pairs): } 1\\text{-}2 \\text{ hours}",explanation:"After SFT. Improves helpfulness and safety alignment."},{formula:"\\text{Evaluation: } 30 \\text{ minutes}",explanation:"Run benchmark suite and qualitative tests."},{formula:"\\text{Export (GGUF): } 10\\text{-}20 \\text{ minutes}",explanation:"Merging and quantizing for deployment."}],id:"example-timeline"}),e.jsx(a,{type:"tip",title:"Iteration Strategy",content:"Do not aim for perfection on the first run. Start with a small subset (1K-5K examples), train for 1 epoch, evaluate qualitatively, then iterate on the data. Common iterations: fixing formatting issues, removing low-quality examples, adding more examples for weak areas.",id:"note-iteration"}),e.jsx(i,{title:"Save Checkpoints Frequently",content:"Always save checkpoints during training. GPU crashes, OOM errors, and power outages happen. Use save_steps=200 and keep at least the last 3 checkpoints. Training can be resumed from any checkpoint with trainer.train(resume_from_checkpoint='./checkpoint-400').",id:"warning-checkpoints"})]})}const Re=Object.freeze(Object.defineProperty({__proto__:null,default:j},Symbol.toStringTag,{value:"Module"}));function M(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Axolotl: YAML-Driven Finetuning"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Axolotl is a flexible finetuning framework that configures entire training runs through YAML files. It supports multi-GPU training, dozens of model architectures, all PEFT methods, and advanced features like multi-dataset mixing and sample packing."}),e.jsx(o,{title:"Axolotl",definition:"Axolotl is an open-source finetuning tool that wraps Hugging Face Transformers, PEFT, and DeepSpeed/FSDP behind a declarative YAML configuration. A single YAML file specifies the model, dataset, training method, and hyperparameters -- no Python coding required.",id:"def-axolotl"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Installation"}),e.jsx(t,{title:"install_axolotl.sh",code:`# Install Axolotl from source
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
python -m axolotl.cli.train --help`,id:"code-install-axolotl"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"YAML Configuration"}),e.jsx(t,{title:"qlora_llama3.yml",code:`# Axolotl YAML config for QLoRA finetuning LLaMA 3.1 8B
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
wandb_run_id: qlora-llama3-r16`,id:"code-yaml-config"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Running Training"}),e.jsx(t,{title:"run_axolotl.sh",code:`# Preprocess the dataset (creates tokenized cache)
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
    --lora_model_dir ./qlora-llama3-output`,id:"code-run-axolotl"}),e.jsx(n,{title:"Axolotl Dataset Types",problem:"What dataset formats does Axolotl support natively?",steps:[{formula:"\\texttt{type: alpaca}",explanation:"Alpaca format: instruction, input, output fields."},{formula:"\\texttt{type: sharegpt}",explanation:"ShareGPT format: multi-turn conversations with from/value pairs."},{formula:"\\texttt{type: completion}",explanation:"Raw text completion: just a text field for continued pretraining."},{formula:"\\texttt{type: chat\\_template}",explanation:"Generic chat format using the tokenizer chat template."}],id:"example-dataset-types"}),e.jsx(a,{type:"tip",title:"Multi-Dataset Training",content:"Axolotl excels at mixing multiple datasets in a single training run. Simply list them in the datasets array. Each can have a different format type. The framework handles interleaving and proper formatting automatically.",id:"note-multi-dataset"}),e.jsx(i,{title:"YAML Syntax Pitfalls",content:"YAML is whitespace-sensitive. Common errors: using tabs instead of spaces, missing colons, incorrect indentation for nested values. Use a YAML validator before running. Also ensure boolean values are lowercase (true/false, not True/False).",id:"warning-yaml"})]})}const Pe=Object.freeze(Object.defineProperty({__proto__:null,default:M},Symbol.toStringTag,{value:"Module"}));function R(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"LLaMA-Factory: GUI-Based Finetuning"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"LLaMA-Factory provides a web-based GUI (LLaMA Board) for configuring and running finetuning jobs without writing code. It supports 100+ LLM architectures, all major training methods, and integrates dataset management, training, evaluation, and deployment."}),e.jsx(o,{title:"LLaMA-Factory",definition:"LLaMA-Factory is an open-source framework for efficient finetuning of 100+ LLMs. It features a Gradio-based web UI (LLaMA Board) for no-code finetuning, CLI for scripted workflows, and supports SFT, RLHF, DPO, and PPO with LoRA/QLoRA/full finetuning.",id:"def-llama-factory"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Installation and Setup"}),e.jsx(t,{title:"install_llama_factory.sh",code:`# Install LLaMA-Factory
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
# - Chat interface for evaluation`,id:"code-install-lf"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"CLI-Based Training"}),e.jsx(t,{title:"llama_factory_cli.sh",code:`# Train via CLI (equivalent to GUI configuration)
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
    --export_size 2  # Shard size in GB`,id:"code-lf-cli"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Custom Datasets in LLaMA-Factory"}),e.jsx(t,{title:"custom_dataset_config.py",code:`# LLaMA-Factory uses a dataset_info.json to register datasets
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
# ]}, ...]`,id:"code-custom-dataset"}),e.jsx(n,{title:"LLaMA-Factory vs Axolotl",problem:"When should you choose LLaMA-Factory over Axolotl?",steps:[{formula:"\\text{LLaMA-Factory: Web UI, beginners}",explanation:"Better for newcomers who prefer a GUI and visual configuration."},{formula:"\\text{Axolotl: YAML, advanced users}",explanation:"Better for reproducible pipelines, CI/CD, and complex multi-dataset configs."},{formula:"\\text{LLaMA-Factory: built-in chat eval}",explanation:"Integrated chat interface for quick qualitative evaluation."},{formula:"\\text{Axolotl: DeepSpeed integration}",explanation:"Better multi-GPU scaling with native DeepSpeed/FSDP support."}],id:"example-lf-vs-axolotl"}),e.jsx(a,{type:"tip",title:"Template System",content:"LLaMA-Factory uses a template system to handle chat formatting for different models. Use --template llama3 for LLaMA 3, --template mistral for Mistral, --template chatml for Qwen. The template ensures correct special tokens are applied.",id:"note-templates"}),e.jsx(i,{title:"GUI vs CLI Consistency",content:"Settings configured in the GUI may not persist between sessions. For reproducibility, always export your GUI configuration as a JSON file and use the CLI for production training runs. The GUI is best for experimentation and quick iterations.",id:"warning-gui-persistence"})]})}const ze=Object.freeze(Object.defineProperty({__proto__:null,default:R},Symbol.toStringTag,{value:"Module"}));function P(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Multi-GPU Finetuning Setup"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"When a model does not fit on a single GPU or you want to speed up training, multi-GPU strategies become necessary. This section covers data parallelism, model parallelism, FSDP, and DeepSpeed for distributed finetuning."}),e.jsx(o,{title:"Data Parallelism",definition:"Each GPU holds a complete copy of the model and processes different batches simultaneously. Gradients are synchronized (all-reduce) after each step. Effective batch size is $\\\\text{per\\\\_device\\\\_batch} \\\\times \\\\text{num\\\\_gpus} \\\\times \\\\text{gradient\\\\_accumulation}$.",id:"def-data-parallel"}),e.jsx(o,{title:"FSDP (Fully Sharded Data Parallelism)",definition:"FSDP shards model parameters, gradients, and optimizer states across GPUs. Each GPU holds only $1/N$ of the model state, enabling training of models $N\\\\times$ larger than single-GPU capacity. Parameters are gathered on-demand for computation.",id:"def-fsdp"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"accelerate Configuration"}),e.jsx(t,{title:"setup_multi_gpu.sh",code:`# Configure accelerate for multi-GPU training
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
llamafactory-cli train config.json  # Auto-detects GPUs`,id:"code-accelerate"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"DeepSpeed ZeRO"}),e.jsx(t,{title:"deepspeed_config.json",code:`{
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
// Stage 3 + offload: Offload params/optimizer to CPU/NVMe`,id:"code-deepspeed"}),e.jsx(t,{title:"fsdp_training.py",code:`# FSDP configuration for Hugging Face Trainer
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
# torchrun --nproc_per_node=4 train.py`,id:"code-fsdp"}),e.jsx(n,{title:"Choosing a Multi-GPU Strategy",problem:"Which multi-GPU strategy should you use for different scenarios?",steps:[{formula:"\\text{Model fits on 1 GPU: Data Parallel}",explanation:"Simplest approach. Each GPU gets full model copy. Linear speedup."},{formula:"\\text{QLoRA + multi-GPU: DeepSpeed ZeRO-2}",explanation:"Shards optimizer states. Works well with QLoRA on 2-8 GPUs."},{formula:"\\text{Full FT, 13-70B: FSDP or ZeRO-3}",explanation:"Shards everything. Required when model does not fit on one GPU."},{formula:"\\text{70B+ with limited GPUs: ZeRO-3 + CPU offload}",explanation:"Offloads optimizer to CPU RAM. Slower but enables very large models."}],id:"example-strategy"}),e.jsx(a,{type:"tip",title:"QLoRA + Multi-GPU",content:"For QLoRA multi-GPU training, use DeepSpeed ZeRO-2 (not ZeRO-3). ZeRO-3 does not work well with 4-bit quantized models because it tries to shard the quantized weights. ZeRO-2 shards only optimizer states and gradients, which are the LoRA parameters in fp16.",id:"note-qlora-multigpu"}),e.jsx(i,{title:"Communication Overhead",content:"Multi-GPU training adds communication overhead for gradient synchronization. With 2 GPUs you get ~1.8x speedup, with 4 GPUs ~3.2x, and with 8 GPUs ~5-6x (not 8x). NVLink between GPUs dramatically reduces this overhead compared to PCIe.",id:"warning-comm-overhead"})]})}const Se=Object.freeze(Object.defineProperty({__proto__:null,default:P},Symbol.toStringTag,{value:"Module"}));function z(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Finetuning Framework Comparison"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Multiple frameworks exist for finetuning LLMs, each with different strengths. This section compares Unsloth, TRL, Axolotl, and LLaMA-Factory across key dimensions to help you choose the right tool for your needs."}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Feature Matrix"}),e.jsx(t,{title:"framework_comparison.py",code:`# Framework comparison matrix
frameworks = {
    "Unsloth": {
        "ease_of_use": "High (Python API)",
        "multi_gpu": "No (single GPU only)",
        "speed": "2x faster (custom Triton kernels)",
        "memory": "60% less VRAM",
        "methods": "SFT, DPO (via TRL)",
        "model_support": "50+ architectures",
        "gui": "No",
        "best_for": "Single-GPU speed, memory-constrained setups",
    },
    "TRL (HuggingFace)": {
        "ease_of_use": "Medium (Python API)",
        "multi_gpu": "Yes (via accelerate/FSDP/DeepSpeed)",
        "speed": "Baseline",
        "memory": "Baseline",
        "methods": "SFT, DPO, PPO, ORPO, KTO, reward modeling",
        "model_support": "All HF models",
        "gui": "No",
        "best_for": "Custom training loops, RLHF, research",
    },
    "Axolotl": {
        "ease_of_use": "Medium (YAML config)",
        "multi_gpu": "Yes (DeepSpeed, FSDP)",
        "speed": "Near-baseline (Flash Attention)",
        "memory": "Good (packing, checkpointing)",
        "methods": "SFT, DPO, RLHF",
        "model_support": "Most popular models",
        "gui": "No",
        "best_for": "Production pipelines, multi-dataset mixing",
    },
    "LLaMA-Factory": {
        "ease_of_use": "Very High (Web GUI + CLI)",
        "multi_gpu": "Yes (auto-detected)",
        "speed": "Near-baseline",
        "memory": "Good",
        "methods": "SFT, DPO, PPO, ORPO, KTO",
        "model_support": "100+ architectures",
        "gui": "Yes (LLaMA Board)",
        "best_for": "Beginners, quick experimentation, no-code",
    },
}

for name, info in frameworks.items():
    print(f"\\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for key, value in info.items():
        print(f"  {key:>15}: {value}")`,id:"code-comparison-matrix"}),e.jsx(n,{title:"Decision Tree: Which Framework?",problem:"How to choose the right finetuning framework for your situation?",steps:[{formula:"\\text{Single GPU + max speed?} \\Rightarrow \\text{Unsloth}",explanation:"Unsloth gives the best single-GPU performance with custom kernels."},{formula:"\\text{Multi-GPU needed?} \\Rightarrow \\text{Axolotl or TRL}",explanation:"Unsloth is single-GPU only. Use Axolotl for YAML-driven multi-GPU or TRL for code-first."},{formula:"\\text{No coding preferred?} \\Rightarrow \\text{LLaMA-Factory}",explanation:"The web GUI allows complete finetuning without writing any code."},{formula:"\\text{Custom RLHF/DPO/PPO?} \\Rightarrow \\text{TRL}",explanation:"TRL provides the most flexibility for alignment research and custom reward functions."},{formula:"\\text{Production pipeline?} \\Rightarrow \\text{Axolotl}",explanation:"YAML configs are easy to version control and integrate into CI/CD."}],id:"example-decision-tree"}),e.jsx(t,{title:"combine_frameworks.py",code:`# You can combine frameworks! Common patterns:

# Pattern 1: Unsloth for SFT + TRL for DPO
# - Fast SFT training with Unsloth's optimizations
# - Then load the SFT model in TRL for DPO alignment

# Pattern 2: LLaMA-Factory for prototyping + Axolotl for production
# - Quick experiments in the GUI
# - Convert winning config to Axolotl YAML for reproducible runs

# Pattern 3: Unsloth model loading + custom training loop
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16,
    target_modules="all-linear",
    use_gradient_checkpointing="unsloth",
)

# Use model with any trainer/loop - you still get Unsloth's kernel optimizations
# trainer = YourCustomTrainer(model=model, ...)

# Pattern 4: Start simple, scale up
# Start: Unsloth on 1 GPU (fastest iteration)
# Scale: Axolotl + DeepSpeed on 4 GPUs (same config, more compute)
# Deploy: Export merged model or GGUF from either`,id:"code-combine"}),e.jsx(a,{type:"tip",title:"Start Simple",content:"For most finetuning projects, start with Unsloth on a single GPU. It is the fastest to set up, the cheapest to run, and produces results identical to other frameworks. Only move to multi-GPU frameworks when your model/data requires it.",id:"note-start-simple"}),e.jsx(i,{title:"Framework Lock-In",content:"All these frameworks produce standard Hugging Face models and LoRA adapters. You are NOT locked into any framework. A model trained with Unsloth can be loaded by TRL, Axolotl, or any HF-compatible tool. Choose based on convenience, not fear of lock-in.",id:"warning-lockin"})]})}const Fe=Object.freeze(Object.defineProperty({__proto__:null,default:z},Symbol.toStringTag,{value:"Module"}));function S(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Dataset Formats: Alpaca, ShareGPT, and Conversation"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Different finetuning frameworks expect data in specific formats. The three most common are Alpaca (instruction-input-output), ShareGPT (multi-turn conversations), and the Hugging Face messages format. Understanding and converting between these is essential."}),e.jsx(o,{title:"Alpaca Format",definition:"The Alpaca format uses three fields per example: instruction (the task description), input (optional additional context), and output (the desired response). It was introduced by the Stanford Alpaca project and remains widely used for single-turn instruction datasets.",id:"def-alpaca"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Format Examples"}),e.jsx(t,{title:"dataset_formats.py",code:`import json

# --- Alpaca Format ---
alpaca_example = {
    "instruction": "Classify the sentiment of this review.",
    "input": "The food was amazing and the service was excellent!",
    "output": "Positive. The review expresses satisfaction with both the food quality and service."
}

# --- ShareGPT Format ---
sharegpt_example = {
    "conversations": [
        {"from": "human", "value": "What is photosynthesis?"},
        {"from": "gpt", "value": "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen."},
        {"from": "human", "value": "What is the chemical equation?"},
        {"from": "gpt", "value": "6CO2 + 6H2O + light energy -> C6H12O6 + 6O2"},
    ]
}

# --- HuggingFace Messages Format ---
hf_messages_example = {
    "messages": [
        {"role": "system", "content": "You are a science tutor."},
        {"role": "user", "content": "What is photosynthesis?"},
        {"role": "assistant", "content": "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen."},
        {"role": "user", "content": "What is the chemical equation?"},
        {"role": "assistant", "content": "6CO2 + 6H2O + light energy -> C6H12O6 + 6O2"},
    ]
}

for name, ex in [("Alpaca", alpaca_example), ("ShareGPT", sharegpt_example),
                  ("HF Messages", hf_messages_example)]:
    print(f"\\n{name}:")
    print(json.dumps(ex, indent=2)[:300])`,id:"code-format-examples"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Converting Between Formats"}),e.jsx(t,{title:"format_conversion.py",code:`def alpaca_to_messages(example):
    """Convert Alpaca format to HF messages format."""
    messages = []
    user_content = example["instruction"]
    if example.get("input"):
        user_content += f"\\n\\n{example['input']}"
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": example["output"]})
    return {"messages": messages}

def sharegpt_to_messages(example):
    """Convert ShareGPT format to HF messages format."""
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    messages = []
    for turn in example["conversations"]:
        role = role_map.get(turn["from"], turn["from"])
        messages.append({"role": role, "content": turn["value"]})
    return {"messages": messages}

def messages_to_alpaca(example):
    """Convert single-turn HF messages to Alpaca format."""
    msgs = example["messages"]
    user_msgs = [m for m in msgs if m["role"] == "user"]
    asst_msgs = [m for m in msgs if m["role"] == "assistant"]
    return {
        "instruction": user_msgs[0]["content"] if user_msgs else "",
        "input": "",
        "output": asst_msgs[0]["content"] if asst_msgs else "",
    }

# Apply to datasets
from datasets import load_dataset

# Convert Alpaca dataset to messages format
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:100]")
dataset = dataset.map(alpaca_to_messages)
print(f"Converted: {dataset[0]['messages']}")`,id:"code-conversion"}),e.jsx(n,{title:"Chat Template Application",problem:"How does the messages format get converted to the model-specific token format?",steps:[{formula:"\\text{messages} \\rightarrow \\texttt{apply\\_chat\\_template()}",explanation:"The tokenizer converts the structured messages into model-specific tokens."},{formula:"\\text{LLaMA 3: } \\texttt{<|start\\_header\\_id|>user<|end\\_header\\_id|>}",explanation:"LLaMA 3 uses header tags to delimit role boundaries."},{formula:"\\text{Mistral: } \\texttt{[INST] ... [/INST]}",explanation:"Mistral wraps user messages in instruction tags."},{formula:"\\text{ChatML: } \\texttt{<|im\\_start|>role ... <|im\\_end|>}",explanation:"Qwen, Phi, and others use the ChatML template standard."}],id:"example-chat-templates"}),e.jsx(a,{type:"tip",title:"Use Messages Format",content:"The HuggingFace messages format is the most universal. SFTTrainer handles it natively, and tokenizer.apply_chat_template() converts it to any model-specific format. Always store your data in messages format for maximum compatibility.",id:"note-use-messages"}),e.jsx(i,{title:"Inconsistent Formatting Degrades Quality",content:"Mixing different formats or having inconsistent formatting within a dataset confuses the model. Common issues: missing system prompts in some examples, inconsistent newlines, HTML artifacts in text. Standardize format before training.",id:"warning-consistency"})]})}const Ce=Object.freeze(Object.defineProperty({__proto__:null,default:S},Symbol.toStringTag,{value:"Module"}));function F(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Building Finetuning Datasets"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The quality of your finetuning dataset determines the quality of your model. This section covers strategies for collecting, curating, and building instruction-following datasets from various sources."}),e.jsx(o,{title:"Instruction Dataset",definition:"An instruction dataset consists of (instruction, response) pairs that teach a model to follow instructions. High-quality datasets have diverse instructions, accurate responses, and consistent formatting. Quality matters more than quantity: 1000 excellent examples often beat 100,000 mediocre ones.",id:"def-instruction-dataset"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Source 1: Existing Open Datasets"}),e.jsx(t,{title:"load_open_datasets.py",code:`from datasets import load_dataset

# Popular open instruction datasets
datasets_catalog = {
    # General instruction following
    "yahma/alpaca-cleaned": "52K cleaned Alpaca instructions",
    "Open-Orca/OpenOrca": "1M+ diverse instructions from GPT-4",
    "HuggingFaceH4/ultrachat_200k": "200K multi-turn conversations",

    # Domain-specific
    "sahil2801/CodeAlpaca-20k": "20K code instruction pairs",
    "lavita/medical-qa-shared-task-v1-half": "Medical Q&A",
    "garage-bAInd/Open-Platypus": "25K STEM reasoning",

    # Preference/alignment data
    "argilla/ultrafeedback-binarized-preferences-cleaned": "DPO pairs",
    "Intel/orca_dpo_pairs": "DPO preference pairs",
}

for name, desc in datasets_catalog.items():
    print(f"{name}: {desc}")

# Load and inspect a dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
print(f"\\nAlpaca-cleaned: {len(dataset)} examples")
print(f"Sample: {dataset[0]}")`,id:"code-open-datasets"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Source 2: Converting Internal Data"}),e.jsx(t,{title:"convert_internal_data.py",code:`import json
import csv
from datasets import Dataset

# Convert customer support logs to instruction format
def convert_support_logs(log_file):
    examples = []
    with open(log_file) as f:
        for line in f:
            log = json.loads(line)
            examples.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful customer support agent."},
                    {"role": "user", "content": log["customer_query"]},
                    {"role": "assistant", "content": log["agent_response"]},
                ]
            })
    return Dataset.from_list(examples)

# Convert FAQ documents to instruction pairs
def convert_faq(faq_data):
    examples = []
    for item in faq_data:
        examples.append({
            "messages": [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]},
            ]
        })
    return Dataset.from_list(examples)

# Convert documentation to instruction format
def convert_docs_to_qa(doc_chunks):
    """Use an LLM to generate Q&A pairs from documentation."""
    examples = []
    for chunk in doc_chunks:
        # Generate questions about this chunk using an LLM
        # Then pair questions with chunk-based answers
        examples.append({
            "messages": [
                {"role": "system", "content": "Answer based on the documentation."},
                {"role": "user", "content": f"Based on this context: {chunk[:200]}...\\nQ: [generated question]"},
                {"role": "assistant", "content": "[generated answer]"},
            ]
        })
    return examples

print("Conversion functions ready. Adapt to your data format.")`,id:"code-convert-data"}),e.jsx(n,{title:"Dataset Curation Strategy",problem:"How to build a high-quality dataset from scratch?",steps:[{formula:"\\text{Step 1: Define scope and tasks}",explanation:"List 10-20 specific tasks your model should handle. Be precise about input/output formats."},{formula:"\\text{Step 2: Seed examples (50-100)}",explanation:"Write high-quality examples by hand for each task category."},{formula:"\\text{Step 3: Expand with LLM (1K-10K)}",explanation:"Use GPT-4 or Claude to generate more examples following your seed patterns."},{formula:"\\text{Step 4: Human review (filter 20-30\\%)}",explanation:"Review and filter generated examples. Remove incorrect, low-quality, or duplicate ones."},{formula:"\\text{Step 5: Iterate}",explanation:"Train a model, find weaknesses, add more examples targeting those areas."}],id:"example-curation-strategy"}),e.jsx(a,{type:"tip",title:"Diversity Is Key",content:"A diverse dataset produces a more robust model. Vary: instruction phrasing, response length, complexity level, topic coverage, and multi-turn vs single-turn examples. Avoid having too many similar examples, which causes the model to become formulaic.",id:"note-diversity"}),e.jsx(i,{title:"Legal and Licensing Considerations",content:"Be careful with data sources. Some datasets are licensed for research only (not commercial use). Data generated by GPT-4 may have OpenAI Terms of Service restrictions on training competing models. Always check the license of each dataset you use.",id:"warning-licensing"})]})}const qe=Object.freeze(Object.defineProperty({__proto__:null,default:F},Symbol.toStringTag,{value:"Module"}));function C(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Data Cleaning for Finetuning"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Noisy, inconsistent, or low-quality data is the most common cause of poor finetuning results. Systematic data cleaning removes duplicates, fixes formatting issues, filters out toxic or incorrect content, and ensures consistent quality throughout the dataset."}),e.jsx(o,{title:"Data Quality Dimensions",definition:"Finetuning data quality encompasses: accuracy (factually correct responses), consistency (uniform formatting and style), diversity (varied instruction types), completeness (responses fully address the instruction), and safety (no harmful or biased content).",id:"def-data-quality"}),e.jsx(t,{title:"data_cleaning_pipeline.py",code:`from datasets import Dataset
import re
import hashlib

def clean_dataset(dataset):
    """Comprehensive data cleaning pipeline."""

    # Step 1: Remove duplicates
    seen_hashes = set()
    def deduplicate(example):
        text = example.get("output", "") or str(example.get("messages", ""))
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        return True

    before = len(dataset)
    dataset = dataset.filter(deduplicate)
    print(f"Dedup: {before} -> {len(dataset)} ({before - len(dataset)} removed)")

    # Step 2: Remove empty or very short responses
    def filter_length(example):
        response = example.get("output", "")
        if not response:
            msgs = example.get("messages", [])
            response = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
        return len(response.strip()) > 20

    dataset = dataset.filter(filter_length)
    print(f"After length filter: {len(dataset)}")

    # Step 3: Clean text artifacts
    def clean_text(example):
        for key in ["instruction", "input", "output"]:
            if key in example and example[key]:
                text = example[key]
                text = re.sub(r'<[^>]+>', '', text)        # Remove HTML tags
                text = re.sub(r'\\s+', ' ', text)            # Normalize whitespace
                text = text.strip()
                example[key] = text
        return example

    dataset = dataset.map(clean_text)

    # Step 4: Filter out known bad patterns
    bad_patterns = [
        r"as an ai language model",
        r"i cannot .* because i am",
        r"i don't have personal",
        r"my training data",
    ]

    def filter_bad_patterns(example):
        response = example.get("output", "").lower()
        return not any(re.search(p, response) for p in bad_patterns)

    dataset = dataset.filter(filter_bad_patterns)
    print(f"After pattern filter: {len(dataset)}")

    return dataset

# Usage:
# cleaned = clean_dataset(raw_dataset)`,id:"code-cleaning-pipeline"}),e.jsx(t,{title:"quality_scoring.py",code:`import numpy as np

def compute_quality_metrics(dataset):
    """Compute quality metrics for a dataset."""
    metrics = {
        "total_examples": len(dataset),
        "response_lengths": [],
        "instruction_lengths": [],
        "empty_inputs": 0,
    }

    for ex in dataset:
        resp = ex.get("output", "")
        inst = ex.get("instruction", "")
        metrics["response_lengths"].append(len(resp))
        metrics["instruction_lengths"].append(len(inst))
        if not ex.get("input", "").strip():
            metrics["empty_inputs"] += 1

    rl = np.array(metrics["response_lengths"])
    il = np.array(metrics["instruction_lengths"])

    print(f"Dataset size: {metrics['total_examples']}")
    print(f"Response length: mean={rl.mean():.0f}, median={np.median(rl):.0f}, "
          f"min={rl.min()}, max={rl.max()}")
    print(f"Instruction length: mean={il.mean():.0f}, median={np.median(il):.0f}")
    print(f"Examples with empty input: {metrics['empty_inputs']} "
          f"({metrics['empty_inputs']/len(dataset)*100:.1f}%)")

    # Check for suspicious patterns
    very_short = (rl < 20).sum()
    very_long = (rl > 5000).sum()
    print(f"Very short responses (<20 chars): {very_short}")
    print(f"Very long responses (>5000 chars): {very_long}")

    return metrics

# metrics = compute_quality_metrics(dataset)`,id:"code-quality-metrics"}),e.jsx(n,{title:"Common Data Issues and Fixes",problem:"What are the most frequent data quality problems in finetuning datasets?",steps:[{formula:"\\text{Exact duplicates: 5-15\\%}",explanation:"Remove using hash-based deduplication. Also check near-duplicates with fuzzy matching."},{formula:"\\text{Wrong language: 1-5\\%}",explanation:"Use language detection (langdetect) to filter examples in unexpected languages."},{formula:"\\text{Truncated responses}",explanation:"Responses cut off mid-sentence. Filter by checking for proper sentence endings."},{formula:"\\text{Prompt leakage}",explanation:"Responses that repeat the instruction verbatim. Filter with overlap detection."}],id:"example-common-issues"}),e.jsx(a,{type:"tip",title:"Iterative Cleaning",content:"Clean your data, train a small model, evaluate outputs, and identify remaining data issues. Repeat this cycle 2-3 times. Each iteration reveals new quality problems that are hard to catch with automated rules alone.",id:"note-iterative"}),e.jsx(i,{title:"Do Not Over-Filter",content:"Aggressive filtering can remove too much data or introduce bias. If you filter out all short responses, the model may never learn to give concise answers. Keep a balance: remove clearly bad examples but preserve diversity in response style and length.",id:"warning-overfilter"})]})}const De=Object.freeze(Object.defineProperty({__proto__:null,default:C},Symbol.toStringTag,{value:"Module"}));function q(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Generating Synthetic Data for Finetuning"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"When you lack sufficient real training data, synthetic data generated by stronger models (like GPT-4 or Claude) can bootstrap your finetuning dataset. This technique, often called distillation, transfers knowledge from a teacher model to a smaller student model."}),e.jsx(o,{title:"Synthetic Data Generation",definition:"Synthetic data generation uses a capable teacher LLM to create training examples that a smaller student model learns from. The teacher generates diverse instruction-response pairs, which are then filtered and used for finetuning. This is distinct from knowledge distillation, which uses soft logits.",id:"def-synthetic-data"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Generating with Seed Instructions"}),e.jsx(t,{title:"generate_synthetic_data.py",code:`import openai
import json
import random

client = openai.OpenAI()  # Uses OPENAI_API_KEY env var

# Seed instructions define the types of data you want
seed_instructions = [
    "Explain a scientific concept in simple terms",
    "Write a Python function to solve a specific problem",
    "Summarize a technical document",
    "Compare and contrast two related topics",
    "Debug a code snippet and explain the fix",
]

def generate_examples(seed, num_examples=5, domain="general"):
    """Generate synthetic instruction-response pairs."""
    prompt = f"""Generate {num_examples} unique instruction-response pairs for training an AI assistant.

Domain: {domain}
Style: Similar to this seed instruction: "{seed}"

Requirements:
- Instructions should be specific and varied
- Responses should be detailed, accurate, and helpful
- Each response should be 100-300 words
- Vary the difficulty level

Output as JSON array with "instruction" and "output" fields."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,  # Higher temp for diversity
        max_tokens=4000,
    )

    try:
        examples = json.loads(response.choices[0].message.content)
        return examples
    except json.JSONDecodeError:
        return []

# Generate dataset
all_examples = []
for seed in seed_instructions:
    examples = generate_examples(seed, num_examples=10)
    all_examples.extend(examples)
    print(f"Generated {len(examples)} from seed: {seed[:40]}...")

print(f"Total: {len(all_examples)} synthetic examples")`,id:"code-generate"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Self-Instruct Method"}),e.jsx(t,{title:"self_instruct.py",code:`import random

def self_instruct_generate(client, existing_instructions, batch_size=5):
    """Generate new instructions bootstrapped from existing ones."""

    # Sample diverse seed examples
    seeds = random.sample(existing_instructions, min(3, len(existing_instructions)))
    seed_text = "\\n".join(f"- {s}" for s in seeds)

    prompt = f"""Here are some example instructions:
{seed_text}

Generate {batch_size} NEW and DIFFERENT instructions.
They should be diverse in topic, format, and difficulty.
Do NOT repeat or closely paraphrase the examples above.

Output each instruction on a new line, prefixed with a number."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )

    new_instructions = []
    for line in response.choices[0].message.content.strip().split("\\n"):
        # Remove numbering
        cleaned = line.strip().lstrip("0123456789.-) ")
        if len(cleaned) > 10:
            new_instructions.append(cleaned)

    return new_instructions

def generate_response(client, instruction):
    """Generate a high-quality response for an instruction."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful, accurate, and detailed assistant."},
            {"role": "user", "content": instruction},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content

# Bootstrap loop
# instructions = seed_instructions.copy()
# for round_num in range(10):
#     new = self_instruct_generate(client, instructions)
#     for inst in new:
#         resp = generate_response(client, inst)
#         dataset.append({"instruction": inst, "output": resp})
#     instructions.extend(new)
#     print(f"Round {round_num}: {len(instructions)} instructions")`,id:"code-self-instruct"}),e.jsx(n,{title:"Synthetic Data Quality Pipeline",problem:"How to ensure synthetic data is high quality?",steps:[{formula:"\\text{Generate} \\rightarrow \\text{Filter} \\rightarrow \\text{Verify}",explanation:"Three-stage pipeline: generate broadly, filter automatically, verify manually."},{formula:"\\text{Filter: LLM-as-judge scoring}",explanation:"Use another LLM to rate each example on accuracy, helpfulness, and relevance (1-5)."},{formula:"\\text{Filter: deduplication + diversity}",explanation:"Remove near-duplicates and ensure topic/format diversity."},{formula:"\\text{Verify: human review 5-10\\%}",explanation:"Manually review a random sample to catch systematic issues."}],id:"example-quality-pipeline"}),e.jsx(a,{type:"tip",title:"Cost-Effective Generation",content:"Use GPT-4o-mini or Claude Haiku for bulk generation (10x cheaper), then use GPT-4o/Claude Opus only for quality filtering and verification. A typical 10K example dataset costs $10-50 to generate with this approach.",id:"note-cost"}),e.jsx(i,{title:"Model Collapse Risk",content:"Training exclusively on synthetic data from one model can lead to model collapse -- the student learns the teacher's biases and limitations. Mix synthetic data with real human-written data (at least 20% real) to maintain diversity and avoid degeneration.",id:"warning-model-collapse"})]})}const Oe=Object.freeze(Object.defineProperty({__proto__:null,default:q},Symbol.toStringTag,{value:"Module"}));function D(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"How Much Data Do You Need?"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"One of the most common questions in finetuning is about dataset size. The answer depends on your goal, data quality, model size, and the complexity of the target behavior. This section provides practical guidelines based on empirical results from the community."}),e.jsx(o,{title:"Scaling Laws for Finetuning",definition:"Unlike pretraining where more data almost always helps, finetuning exhibits diminishing returns and risks overfitting. For instruction tuning, the LIMA paper showed that 1,000 high-quality examples can match models trained on 52K lower-quality examples. The relationship is roughly $\\\\text{quality} \\\\propto \\\\sqrt{n \\\\cdot q}$ where $n$ is dataset size and $q$ is quality.",id:"def-scaling"}),e.jsx(n,{title:"Dataset Size by Use Case",problem:"How many examples do you need for different finetuning goals?",steps:[{formula:"\\text{Style/tone adaptation: } 50\\text{-}200",explanation:"Teaching a model to write in a specific voice or persona. Very few examples needed."},{formula:"\\text{Single task (classification, extraction): } 200\\text{-}2\\text{K}",explanation:"Narrow task with clear input/output patterns."},{formula:"\\text{Domain instruction following: } 1\\text{K-}10\\text{K}",explanation:"Teaching domain expertise (legal, medical, coding in specific frameworks)."},{formula:"\\text{General instruction improvement: } 10\\text{K-}100\\text{K}",explanation:"Broadly improving helpfulness, reasoning, and instruction following."},{formula:"\\text{Continued pretraining: } 1\\text{M-}10\\text{B tokens}",explanation:"Adapting to a new language or domain requires much more data."}],id:"example-size-by-use"}),e.jsx(t,{title:"dataset_size_experiment.py",code:`import numpy as np

def estimate_training_time(
    dataset_size,
    avg_tokens_per_example=512,
    batch_size=8,
    epochs=2,
    tokens_per_second=5000,  # Typical for QLoRA on RTX 4090
):
    """Estimate training time for a given dataset size."""
    total_tokens = dataset_size * avg_tokens_per_example * epochs
    total_steps = (dataset_size * epochs) / batch_size
    time_seconds = total_tokens / tokens_per_second
    time_hours = time_seconds / 3600
    return {
        "total_tokens": total_tokens,
        "total_steps": int(total_steps),
        "time_hours": time_hours,
    }

# Compare different dataset sizes
print(f"{'Size':>10} {'Tokens':>12} {'Steps':>8} {'Time':>10}")
print("-" * 45)
for size in [100, 500, 1000, 5000, 10000, 50000, 100000]:
    result = estimate_training_time(size)
    print(f"{size:>10,} {result['total_tokens']:>12,} "
          f"{result['total_steps']:>8,} {result['time_hours']:>9.1f}h")

# Output:
#       100      102,400      25      0.0h
#       500      512,000     125      0.0h
#     1,000    1,024,000     250      0.1h
#     5,000    5,120,000   1,250      0.3h
#    10,000   10,240,000   2,500      0.6h
#    50,000   51,200,000  12,500      2.8h
#   100,000  102,400,000  25,000      5.7h`,id:"code-estimate-time"}),e.jsx(t,{title:"optimal_epochs.py",code:`# How many epochs to train for different dataset sizes

def recommend_epochs(dataset_size, task_type="instruction"):
    """Recommend number of training epochs."""
    if task_type == "instruction":
        if dataset_size < 500:
            return 3, "Small dataset: more epochs to learn patterns"
        elif dataset_size < 5000:
            return 2, "Medium dataset: 2 epochs balances learning and overfitting"
        elif dataset_size < 50000:
            return 1, "Large dataset: 1 epoch usually sufficient"
        else:
            return 1, "Very large: may not even need full epoch"
    elif task_type == "continued_pretrain":
        return 1, "Continued pretraining: always 1 epoch to avoid repetition"
    elif task_type == "classification":
        if dataset_size < 1000:
            return 5, "Small classification: more epochs with early stopping"
        else:
            return 3, "Classification: 3 epochs with validation monitoring"

for size in [100, 500, 2000, 10000, 50000]:
    epochs, reason = recommend_epochs(size)
    print(f"Dataset {size:>6,}: {epochs} epoch(s) - {reason}")

# Key insight: number of gradient updates matters more than epochs
# target_steps = dataset_size * epochs / batch_size
# Aim for 500-5000 total steps for most instruction tuning tasks`,id:"code-epochs"}),e.jsx(a,{type:"intuition",title:"The LIMA Insight",content:"The LIMA paper (2023) demonstrated that a model finetuned on just 1,000 carefully curated examples could match models trained on 52,000 examples. The key was extreme quality: each example was handwritten by researchers to be clear, complete, and diverse. This suggests that data curation effort yields higher returns than data collection volume.",id:"note-lima"}),e.jsx(a,{type:"tip",title:"Practical Starting Point",content:"Start with 1,000-2,000 high-quality examples and train for 2 epochs. Evaluate results. If the model performs well on most tasks but struggles with specific ones, add 200-500 targeted examples for those areas. This iterative approach is more effective than training on a massive dataset from the start.",id:"note-starting-point"}),e.jsx(i,{title:"Overfitting Signs",content:"With small datasets, overfitting is the primary risk. Signs: training loss drops below 0.3, model outputs memorized responses verbatim, model performs well on training-like prompts but poorly on novel ones. Mitigations: fewer epochs, higher dropout, lower rank, data augmentation.",id:"warning-overfitting"})]})}const Ne=Object.freeze(Object.defineProperty({__proto__:null,default:D},Symbol.toStringTag,{value:"Module"}));function O(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Safetensors Format"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Safetensors is the standard format for storing and sharing model weights in the Hugging Face ecosystem. It replaces the older pickle-based .bin format with a safe, fast, memory-mapped file format that prevents arbitrary code execution."}),e.jsx(o,{title:"Safetensors",definition:"Safetensors is a binary format for storing tensors that is safe (no arbitrary code execution), fast (supports memory mapping for instant loading), and simple (header + raw tensor data). Files use the .safetensors extension and contain a JSON header followed by contiguous tensor data.",id:"def-safetensors"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Why Safetensors Over .bin"}),e.jsx(n,{title:"Safetensors vs PyTorch .bin",problem:"Why did the ecosystem move from .bin to .safetensors?",steps:[{formula:"\\text{Security: no pickle} \\Rightarrow \\text{no code execution}",explanation:"PyTorch .bin files use pickle, which can execute arbitrary Python code on load. Safetensors is just data."},{formula:"\\text{Speed: memory-mapped loading}",explanation:"Safetensors uses mmap, loading tensors lazily. 10-100x faster for large models."},{formula:"\\text{Sharding: built-in multi-file support}",explanation:"Large models split across multiple .safetensors files with an index.json."},{formula:"\\text{Zero-copy: no deserialization}",explanation:"Tensors are stored in their native format and can be used directly from disk."}],id:"example-safetensors-vs-bin"}),e.jsx(t,{title:"working_with_safetensors.py",code:`from safetensors import safe_open
from safetensors.torch import save_file, load_file
import torch

# Save tensors to safetensors format
tensors = {
    "weight": torch.randn(1024, 768),
    "bias": torch.zeros(1024),
    "embedding": torch.randn(32000, 4096),
}

save_file(tensors, "model.safetensors")
print("Saved model.safetensors")

# Load tensors (fast, memory-mapped)
loaded = load_file("model.safetensors")
print(f"Loaded keys: {list(loaded.keys())}")
print(f"Weight shape: {loaded['weight'].shape}")

# Inspect without loading all tensors into memory
with safe_open("model.safetensors", framework="pt") as f:
    # Check available tensors
    print(f"Tensor names: {f.keys()}")

    # Load individual tensors on demand
    weight = f.get_tensor("weight")
    print(f"Weight dtype: {weight.dtype}, shape: {weight.shape}")

    # Get metadata
    metadata = f.metadata()
    print(f"Metadata: {metadata}")

# Convert from .bin to .safetensors
# from transformers import AutoModel
# model = AutoModel.from_pretrained("model-path")
# model.save_pretrained("model-path", safe_serialization=True)`,id:"code-safetensors"}),e.jsx(t,{title:"inspect_model_files.py",code:`import json
import os

def inspect_model_dir(model_path):
    """Inspect a Hugging Face model directory."""
    files = os.listdir(model_path)

    safetensors = [f for f in files if f.endswith('.safetensors')]
    bin_files = [f for f in files if f.endswith('.bin')]
    json_files = [f for f in files if f.endswith('.json')]

    print(f"Model directory: {model_path}")
    print(f"Safetensors files: {len(safetensors)}")
    print(f"PyTorch bin files: {len(bin_files)}")
    print(f"JSON files: {json_files}")

    # Check model index for sharded models
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        total_size = index.get("metadata", {}).get("total_size", 0)
        print(f"Total model size: {total_size / 1e9:.1f} GB")
        print(f"Number of shards: {len(set(index['weight_map'].values()))}")

    # Check config for model architecture
    config_file = os.path.join(model_path, "config.json")
    if os.path.exists(config_file):
        with open(config_file) as f:
            config = json.load(f)
        print(f"Architecture: {config.get('architectures', ['unknown'])}")
        print(f"Hidden size: {config.get('hidden_size', 'N/A')}")
        print(f"Num layers: {config.get('num_hidden_layers', 'N/A')}")

# inspect_model_dir("/path/to/your/model")`,id:"code-inspect-model"}),e.jsx(a,{type:"note",title:"Safetensors Is the Default",content:"As of transformers 4.39+, save_pretrained() uses safetensors by default. All new models on Hugging Face Hub use safetensors. The library automatically handles both formats: if safetensors files exist, they are preferred over .bin files.",id:"note-default-format"}),e.jsx(i,{title:"Do Not Trust .bin Files from Unknown Sources",content:"PyTorch .bin files use Python's pickle module, which can execute arbitrary code when loaded. Never load .bin files from untrusted sources. Always prefer safetensors format. If you must use .bin files, scan them with tools like picklescan before loading.",id:"warning-pickle-security"})]})}const Ue=Object.freeze(Object.defineProperty({__proto__:null,default:O},Symbol.toStringTag,{value:"Module"}));function N(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"GGUF Format for llama.cpp"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"GGUF (GPT-Generated Unified Format) is the file format used by llama.cpp for efficient CPU and GPU inference. It is the standard format for running quantized LLMs locally on consumer hardware, including laptops and desktops without dedicated GPUs."}),e.jsx(o,{title:"GGUF",definition:"GGUF is a single-file binary format that contains model weights, tokenizer, and metadata. It supports various quantization levels (Q2_K through Q8_0) and is optimized for the llama.cpp inference engine. A single .gguf file is all you need to run a model.",id:"def-gguf"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Quantization Levels"}),e.jsx(n,{title:"GGUF Quantization Types",problem:"What are the common GGUF quantization levels and their tradeoffs?",steps:[{formula:"\\text{Q2\\_K: } \\sim 2.5 \\text{ bits/weight}",explanation:"Smallest size, significant quality loss. Only for very large models where nothing else fits."},{formula:"\\text{Q4\\_K\\_M: } \\sim 4.8 \\text{ bits/weight}",explanation:"Best balance of size and quality. Most popular choice for local inference."},{formula:"\\text{Q5\\_K\\_M: } \\sim 5.7 \\text{ bits/weight}",explanation:"Slightly better quality than Q4 with modest size increase."},{formula:"\\text{Q8\\_0: } \\sim 8.5 \\text{ bits/weight}",explanation:"Near-lossless quantization. Quality very close to fp16 at half the size."}],id:"example-quant-levels"}),e.jsx(t,{title:"convert_to_gguf.py",code:`# Method 1: Using Unsloth (easiest)
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "your-finetuned-model-path",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Save as GGUF with different quantization levels
quant_methods = ["q4_k_m", "q5_k_m", "q8_0"]
for quant in quant_methods:
    model.save_pretrained_gguf(
        f"./model-{quant}",
        tokenizer,
        quantization_method=quant,
    )
    print(f"Saved {quant} GGUF")

# Method 2: Using llama.cpp directly
# git clone https://github.com/ggerganov/llama.cpp
# cd llama.cpp && make

# Convert safetensors to GGUF (fp16 first)
# python convert_hf_to_gguf.py /path/to/model --outfile model-f16.gguf

# Quantize to desired level
# ./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M`,id:"code-convert-gguf"}),e.jsx(t,{title:"gguf_size_estimation.py",code:`def estimate_gguf_size(params_billions, quant_type="q4_k_m"):
    """Estimate GGUF file size for different quantization types."""
    bits_per_weight = {
        "q2_k": 2.5,
        "q3_k_m": 3.4,
        "q4_0": 4.5,
        "q4_k_m": 4.8,
        "q5_0": 5.5,
        "q5_k_m": 5.7,
        "q6_k": 6.6,
        "q8_0": 8.5,
        "f16": 16.0,
    }

    bpw = bits_per_weight.get(quant_type, 4.8)
    size_gb = (params_billions * 1e9 * bpw) / (8 * 1e9)
    ram_needed = size_gb * 1.1  # 10% overhead for context

    return {"file_size_gb": size_gb, "ram_needed_gb": ram_needed}

# Compare sizes for common models
print(f"{'Model':>10} {'Quant':>8} {'Size':>8} {'RAM':>8}")
print("-" * 40)
for params in [7, 8, 13, 34, 70]:
    for quant in ["q4_k_m", "q5_k_m", "q8_0"]:
        result = estimate_gguf_size(params, quant)
        print(f"{params}B {quant:>8} {result['file_size_gb']:>7.1f}G "
              f"{result['ram_needed_gb']:>7.1f}G")

# Output:
#   7B   q4_k_m     4.2G     4.6G
#   7B   q5_k_m     5.0G     5.5G
#   7B   q8_0       7.4G     8.2G
#   70B  q4_k_m    42.0G    46.2G`,id:"code-gguf-sizes"}),e.jsx(a,{type:"tip",title:"Recommended Quantization",content:"For most use cases, Q4_K_M offers the best quality-to-size ratio. Use Q5_K_M if you have extra RAM and want slightly better quality. Q8_0 is useful for evaluation and when quality is paramount. Avoid Q2_K unless the model absolutely does not fit otherwise.",id:"note-recommended-quant"}),e.jsx(i,{title:"Quantization Quality Loss",content:"Quantization always introduces some quality degradation. Q4_K_M typically loses 1-3% on benchmarks compared to fp16. For critical applications, always evaluate the quantized model against the fp16 version on your specific tasks before deploying.",id:"warning-quant-loss"}),e.jsx(a,{type:"note",title:"Running GGUF Models",content:"GGUF files work with llama.cpp, Ollama, LM Studio, GPT4All, and many other local inference tools. To run: ollama create mymodel -f Modelfile (where Modelfile points to your .gguf). Or use llama.cpp: ./llama-cli -m model.gguf -p 'Your prompt here'.",id:"note-running-gguf"})]})}const Ge=Object.freeze(Object.defineProperty({__proto__:null,default:N},Symbol.toStringTag,{value:"Module"}));function U(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"GPTQ and AWQ: GPU-Optimized Quantization"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"While GGUF targets CPU inference, GPTQ and AWQ are quantization methods optimized for GPU inference. They produce 4-bit models that run efficiently with libraries like vLLM, TGI, and the transformers library itself."}),e.jsx(o,{title:"GPTQ (GPT Quantization)",definition:"GPTQ quantizes weights to 4-bit integers using a calibration dataset. It processes weights layer by layer, minimizing the output error using approximate second-order information (Hessian). The quantized model uses custom CUDA kernels for fast 4-bit matrix multiplication on GPUs.",id:"def-gptq"}),e.jsx(o,{title:"AWQ (Activation-Aware Weight Quantization)",definition:"AWQ identifies the most important weights by analyzing activation magnitudes and protects them during quantization. It applies per-channel scaling to reduce quantization error for salient weights. AWQ is typically faster than GPTQ for quantization and produces slightly better quality at 4-bit.",id:"def-awq"}),e.jsx(t,{title:"quantize_gptq.py",code:`from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch

model_name = "./your-finetuned-model"  # Merged fp16 model

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure GPTQ quantization
gptq_config = GPTQConfig(
    bits=4,                    # 4-bit quantization
    dataset="c4",              # Calibration dataset
    tokenizer=tokenizer,
    group_size=128,            # Quantization group size
    desc_act=False,            # Disable desc_act for vLLM compat
    sym=True,                  # Symmetric quantization
)

# Load and quantize
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=gptq_config,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Save quantized model
model.save_pretrained("./model-gptq-4bit")
tokenizer.save_pretrained("./model-gptq-4bit")
print("GPTQ model saved!")`,id:"code-gptq"}),e.jsx(t,{title:"quantize_awq.py",code:`from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "./your-finetuned-model"

# Load model for AWQ quantization
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Configure AWQ
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",        # GEMM for general use, GEMV for batch=1
}

# Quantize (requires calibration data)
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data="pileval",      # Calibration dataset
)

# Save
model.save_quantized("./model-awq-4bit")
tokenizer.save_pretrained("./model-awq-4bit")
print("AWQ model saved!")

# Load and use the quantized model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "./model-awq-4bit",
    device_map="auto",
)
# Works with standard generate() API`,id:"code-awq"}),e.jsx(n,{title:"GPTQ vs AWQ vs GGUF",problem:"When should you use each quantization method?",steps:[{formula:"\\text{GGUF: CPU inference, local deployment}",explanation:"Use with llama.cpp, Ollama. Best for laptops and CPU-only servers."},{formula:"\\text{GPTQ: GPU inference with vLLM/TGI}",explanation:"Widely supported in GPU serving frameworks. Slower to quantize but well-tested."},{formula:"\\text{AWQ: GPU inference, best quality}",explanation:"Slightly better quality than GPTQ, faster quantization. Good vLLM support."},{formula:"\\text{bitsandbytes NF4: training only}",explanation:"Used during QLoRA training. Not ideal for production inference."}],id:"example-comparison"}),e.jsx(a,{type:"tip",title:"vLLM Compatibility",content:"For production GPU serving with vLLM, AWQ with GEMM kernel is recommended. It provides the best throughput and is well-supported. Ensure desc_act=False for GPTQ if targeting vLLM, as vLLM does not support activation reordering.",id:"note-vllm"}),e.jsx(i,{title:"Quantization Requires a Calibration Dataset",content:"Both GPTQ and AWQ need a calibration dataset to determine optimal quantization parameters. The calibration data should be representative of your model's actual use case. Using generic calibration data (like C4) works but domain-specific calibration data yields better results.",id:"warning-calibration"})]})}const Ie=Object.freeze(Object.defineProperty({__proto__:null,default:U},Symbol.toStringTag,{value:"Module"}));function G(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Merging LoRA Weights into Base Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"After finetuning with LoRA, you have a small adapter file (~100 MB) that must be combined with the base model for inference. Merging the LoRA weights into the base model eliminates inference overhead and produces a standalone model that can be deployed like any other."}),e.jsx(o,{title:"LoRA Merging",definition:"LoRA merging computes the combined weight $W_{\\\\text{merged}} = W_0 + \\\\frac{\\\\alpha}{r} B A$ and saves it as a regular model. After merging, the model no longer needs the PEFT library and has zero inference overhead from the adaptation.",notation:"W_{merged} = W_0 + \\frac{\\alpha}{r} BA",id:"def-lora-merge"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Merging Methods"}),e.jsx(t,{title:"merge_lora_weights.py",code:`from peft import PeftModel, PeftConfig
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
# Should be LlamaForCausalLM, not PeftModelForCausalLM`,id:"code-merge-standard"}),e.jsx(t,{title:"merge_with_unsloth.py",code:`from unsloth import FastLanguageModel

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
)`,id:"code-merge-unsloth"}),e.jsx(n,{title:"Merging QLoRA Adapters",problem:"How to merge a QLoRA adapter (trained on 4-bit base) into a full-precision model?",steps:[{formula:"\\text{Load base model in fp16 (not 4-bit)}",explanation:"The merged model should be full precision for quality. Load base on CPU if GPU lacks VRAM."},{formula:"\\text{Load LoRA adapter onto fp16 base}",explanation:"PeftModel.from_pretrained handles the dtype mismatch automatically."},{formula:"\\text{merge\\_and\\_unload()}",explanation:"Compute W_merged = W_base + (alpha/r) * B @ A for each adapted layer."},{formula:"\\text{Save as safetensors}",explanation:"The result is a standard model with no PEFT dependency."}],id:"example-qlora-merge"}),e.jsx(a,{type:"tip",title:"Merge on CPU",content:"Merging requires loading both the base model and adapter in fp16, which may exceed GPU VRAM. Use device_map='cpu' to merge on CPU RAM (needs ~2x model size in RAM: ~16 GB for a 7B model). This is a one-time operation so speed is not critical.",id:"note-merge-cpu"}),e.jsx(i,{title:"Do Not Merge Quantized Base Weights",content:"Never merge LoRA adapters into a 4-bit quantized base model. The quantization noise in the base weights will permanently degrade quality. Always merge into the fp16/bf16 base model, then quantize the merged model separately if needed.",id:"warning-quant-merge"}),e.jsx(a,{type:"note",title:"Multiple LoRA Adapters",content:"You can merge multiple LoRA adapters sequentially (e.g., SFT adapter then DPO adapter). Load and merge one at a time. The order matters: merge SFT first, then DPO on top. Alternatively, some tools support adapter arithmetic for combining adapters with weighted sums.",id:"note-multi-adapter"})]})}const Be=Object.freeze(Object.defineProperty({__proto__:null,default:G},Symbol.toStringTag,{value:"Module"}));function I(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Format Conversion Between Model Types"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Different inference engines require different model formats. Converting between safetensors, GGUF, GPTQ, and AWQ is a common post-training task. This section provides practical conversion workflows for every major format combination."}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Conversion Paths"}),e.jsx(n,{title:"Format Conversion Map",problem:"What are the conversion paths between model formats?",steps:[{formula:"\\text{Safetensors (fp16)} \\rightarrow \\text{GGUF}",explanation:"Use llama.cpp convert_hf_to_gguf.py, then quantize with llama-quantize."},{formula:"\\text{Safetensors (fp16)} \\rightarrow \\text{GPTQ}",explanation:"Use auto-gptq or transformers GPTQConfig with calibration data."},{formula:"\\text{Safetensors (fp16)} \\rightarrow \\text{AWQ}",explanation:"Use autoawq library with calibration data."},{formula:"\\text{GGUF} \\rightarrow \\text{Safetensors}",explanation:"Use llama.cpp convert tools or llama-cpp-python for dequantization (lossy)."}],id:"example-conversion-paths"}),e.jsx(t,{title:"convert_safetensors_to_gguf.sh",code:`# Step 1: Clone llama.cpp (if not already)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt

# Step 2: Convert HF model to GGUF (fp16)
python convert_hf_to_gguf.py \\
    /path/to/your/merged-model \\
    --outfile model-f16.gguf \\
    --outtype f16

# Step 3: Quantize to desired level
# Build llama.cpp first
make -j$(nproc)

# Quantize
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
./llama-quantize model-f16.gguf model-q5_k_m.gguf Q5_K_M
./llama-quantize model-f16.gguf model-q8_0.gguf Q8_0

# Step 4: Test the quantized model
./llama-cli -m model-q4_k_m.gguf \\
    -p "Write a Python function to calculate fibonacci numbers." \\
    -n 256 --temp 0.7

# Available quantization types:
# Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M,
# Q5_0, Q5_K_S, Q5_K_M, Q6_K, Q8_0, F16, F32`,id:"code-to-gguf"}),e.jsx(t,{title:"format_conversion_utils.py",code:`import os
import shutil

def convert_model(input_path, output_path, target_format, quant_level="q4_k_m"):
    """Convert a model between formats."""

    print(f"Converting: {input_path} -> {target_format}")

    if target_format == "gguf":
        # Requires llama.cpp
        import subprocess
        llama_cpp = os.environ.get("LLAMA_CPP_PATH", "~/llama.cpp")

        # Step 1: Convert to fp16 GGUF
        fp16_path = output_path.replace(".gguf", "-f16.gguf")
        subprocess.run([
            "python", f"{llama_cpp}/convert_hf_to_gguf.py",
            input_path, "--outfile", fp16_path, "--outtype", "f16"
        ], check=True)

        # Step 2: Quantize
        subprocess.run([
            f"{llama_cpp}/llama-quantize",
            fp16_path, output_path, quant_level.upper()
        ], check=True)

        # Clean up fp16 intermediate
        if os.path.exists(fp16_path):
            os.remove(fp16_path)

    elif target_format == "gptq":
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
        import torch

        tokenizer = AutoTokenizer.from_pretrained(input_path)
        gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            input_path, quantization_config=gptq_config,
            torch_dtype=torch.float16, device_map="auto"
        )
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

    elif target_format == "awq":
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        model = AutoAWQForCausalLM.from_pretrained(input_path)
        tokenizer = AutoTokenizer.from_pretrained(input_path)
        model.quantize(tokenizer, quant_config={"w_bit": 4, "q_group_size": 128})
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)

    print(f"Conversion complete: {output_path}")

# Usage:
# convert_model("./merged-model", "./model.gguf", "gguf", "q4_k_m")
# convert_model("./merged-model", "./model-gptq", "gptq")
# convert_model("./merged-model", "./model-awq", "awq")`,id:"code-conversion-utils"}),e.jsx(a,{type:"tip",title:"Always Start from fp16",content:"The golden rule: always convert from the fp16/bf16 merged model. Converting from one quantized format to another (e.g., GPTQ to GGUF) compounds quantization errors. Keep your fp16 merged model as the source of truth for all format conversions.",id:"note-start-fp16"}),e.jsx(i,{title:"Lossy Conversions",content:"Converting from a quantized format back to fp16 (e.g., GGUF Q4 to safetensors) is lossy -- you cannot recover the original precision. The result will have the same quality as the quantized version, just in a different container. Only the original fp16 weights are truly full precision.",id:"warning-lossy"}),e.jsx(a,{type:"note",title:"Verifying Conversions",content:"After conversion, always verify the model works by running a few test prompts. Compare outputs between the original and converted model. Check for: garbled text (tokenizer issues), repetitive output (quantization too aggressive), or wrong language (metadata mismatch).",id:"note-verify"})]})}const Ee=Object.freeze(Object.defineProperty({__proto__:null,default:I},Symbol.toStringTag,{value:"Module"}));function B(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Pushing Models to Hugging Face Hub"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The Hugging Face Hub is the standard platform for sharing and distributing models. Uploading your finetuned model makes it accessible to others and enables easy deployment. This section covers uploading models, creating model cards, and managing repositories."}),e.jsx(o,{title:"Hugging Face Hub",definition:"The Hugging Face Hub is a platform hosting over 500,000 models, 100,000 datasets, and 100,000 demo applications. Model repositories use Git LFS for large file storage, support versioning, and include model cards for documentation.",id:"def-hf-hub"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Authentication and Setup"}),e.jsx(t,{title:"hf_hub_setup.sh",code:`# Install the Hub CLI
pip install huggingface_hub

# Login (interactive - opens browser)
huggingface-cli login

# Or set token directly
huggingface-cli login --token hf_xxxxxxxxxxxxx

# Or via environment variable
export HF_TOKEN=hf_xxxxxxxxxxxxx

# Verify authentication
huggingface-cli whoami`,id:"code-hf-setup"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Pushing Models"}),e.jsx(t,{title:"push_model_to_hub.py",code:`from transformers import AutoModelForCausalLM, AutoTokenizer
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

print(f"Model uploaded to: https://huggingface.co/{repo_name}")`,id:"code-push-model"}),e.jsx(t,{title:"create_model_card.py",code:`# Create a model card (README.md) for your model
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

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("your-username/model-name")
    tokenizer = AutoTokenizer.from_pretrained("your-username/model-name")

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
)`,id:"code-model-card"}),e.jsx(n,{title:"Repository Best Practices",problem:"What should a well-organized model repository contain?",steps:[{formula:"\\text{README.md: model card with training details}",explanation:"Description, training config, usage examples, evaluation results."},{formula:"\\text{Model files: safetensors + config.json}",explanation:"The model weights and architecture configuration."},{formula:"\\text{Tokenizer files: tokenizer.json + special tokens}",explanation:"Everything needed to reconstruct the tokenizer."},{formula:"\\text{Optional: GGUF files for local inference}",explanation:"Upload Q4_K_M and Q8_0 GGUF variants for llama.cpp users."}],id:"example-repo-structure"}),e.jsx(a,{type:"tip",title:"Private Repositories",content:"For proprietary models, create private repos: create_repo(repo_name, private=True). Team members can be granted access through organization settings. Private repos work the same as public ones but require authentication to download.",id:"note-private-repos"}),e.jsx(i,{title:"Check for Sensitive Data",content:"Before pushing to Hub, ensure your model does not contain sensitive information embedded in the weights (e.g., from training on private data). Also check that no credentials, API keys, or personal data are included in config files or the model card.",id:"warning-sensitive-data"})]})}const $e=Object.freeze(Object.defineProperty({__proto__:null,default:B},Symbol.toStringTag,{value:"Module"}));function E(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Perplexity & Loss-Based Evaluation"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Perplexity is the most fundamental metric for evaluating language models. It measures how surprised the model is by a held-out test set. Lower perplexity means the model assigns higher probability to the correct continuations, indicating better language modeling quality."}),e.jsx(o,{title:"Perplexity",definition:"Perplexity is the exponentiated average negative log-likelihood of a sequence. For a token sequence of length $N$, perplexity is $\\text{PPL} = \\exp\\!\\left(-\\frac{1}{N}\\sum_{i=1}^{N}\\log p(x_i \\mid x_{<i})\\right)$. It represents the effective branching factor: a perplexity of 10 means the model is as uncertain as choosing uniformly among 10 tokens.",notation:"PPL, pp",id:"def-perplexity"}),e.jsx(n,{title:"Interpreting Perplexity Scores",problem:"How do you compare perplexity between a base model and a fine-tuned model?",steps:[{formula:"\\text{PPL}_{\\text{base}} = 8.2 \\text{ on domain data}",explanation:"Base model perplexity on your target domain evaluation set."},{formula:"\\text{PPL}_{\\text{finetuned}} = 4.1 \\text{ on domain data}",explanation:"Fine-tuned model achieves lower perplexity, meaning better fit to domain."},{formula:"\\Delta\\text{PPL} = 8.2 - 4.1 = 4.1",explanation:"A 50% reduction in perplexity indicates significant domain adaptation."},{formula:"\\text{PPL}_{\\text{finetuned}} = 12.5 \\text{ on general data}",explanation:"Check general-domain perplexity too -- if it rises sharply, the model has overfit."}],id:"example-ppl-comparison"}),e.jsx(t,{title:"compute_perplexity.py",code:`import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math

def compute_perplexity(model_path, eval_texts, max_length=2048):
    """Compute perplexity of a model on evaluation texts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for text in eval_texts:
        encodings = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
            neg_log_likelihood = outputs.loss
            num_tokens = encodings["input_ids"].size(1)

        total_loss += neg_log_likelihood.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return {"perplexity": perplexity, "avg_loss": avg_loss, "total_tokens": total_tokens}

# Usage
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
eval_texts = [t for t in dataset["text"] if len(t) > 50][:200]

base_results = compute_perplexity("meta-llama/Llama-3.1-8B", eval_texts)
ft_results = compute_perplexity("./my-finetuned-model", eval_texts)

print(f"Base model PPL:      {base_results['perplexity']:.2f}")
print(f"Finetuned model PPL: {ft_results['perplexity']:.2f}")
print(f"Improvement:         {base_results['perplexity'] - ft_results['perplexity']:.2f}")`,id:"code-compute-ppl"}),e.jsx(t,{title:"sliding_window_perplexity.py",code:`import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

def sliding_window_perplexity(model_path, text, stride=512, max_length=2048):
    """Compute perplexity using sliding window for long texts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end
        input_ids = encodings.input_ids[:, begin:end].to(device)

        target_ids = input_ids.clone()
        target_ids[:, :-target_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nll = outputs.loss * target_len

        nlls.append(nll.item())
        prev_end = end
        if end == seq_len:
            break

    ppl = math.exp(sum(nlls) / prev_end)
    return ppl

# Compare base vs finetuned on a long document
long_text = open("test_document.txt").read()
print(f"PPL: {sliding_window_perplexity('./my-model', long_text):.2f}")`,id:"code-sliding-window"}),e.jsx(a,{type:"intuition",title:"Perplexity Is Not Everything",content:"Lower perplexity means the model predicts the evaluation text better, but it does not always correlate with downstream task quality. A model fine-tuned on code may have worse perplexity on general English text but be dramatically better at writing code. Always pair perplexity with task-specific evaluations.",id:"note-ppl-limits"}),e.jsx(i,{title:"Tokenizer Mismatch Invalidates Perplexity",content:"Perplexity comparisons are only meaningful when both models use the same tokenizer. Different tokenizers produce different token counts for the same text, making perplexity values incomparable across model families (e.g., comparing LLaMA vs Mistral perplexity directly is misleading).",id:"warning-tokenizer-mismatch"}),e.jsx(a,{type:"tip",title:"Track Loss Curves During Training",content:"Monitor both training loss and validation loss during fine-tuning. If training loss decreases but validation loss increases, the model is overfitting. Save checkpoints at the lowest validation loss point, not at the end of training.",id:"note-loss-curves"})]})}const Qe=Object.freeze(Object.defineProperty({__proto__:null,default:E},Symbol.toStringTag,{value:"Module"}));function $(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Task-Specific Benchmarks (MMLU, HellaSwag)"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Standardized benchmarks provide comparable evaluation across models. MMLU tests knowledge and reasoning across 57 subjects, HellaSwag tests commonsense reasoning, and many other benchmarks target specific capabilities. Running these on your fine-tuned model helps detect regressions and measure improvement."}),e.jsx(o,{title:"MMLU (Massive Multitask Language Understanding)",definition:"MMLU is a benchmark consisting of 14,042 multiple-choice questions spanning 57 subjects from elementary to professional level. Performance is measured as accuracy: $\\text{Acc} = \\frac{\\text{correct answers}}{\\text{total questions}}$. State-of-the-art models score above 85%, while random chance yields 25%.",id:"def-mmlu"}),e.jsx(o,{title:"HellaSwag",definition:"HellaSwag is a commonsense natural language inference benchmark where the model must select the most plausible continuation of a scenario from four choices. It tests grounded commonsense reasoning with adversarially filtered wrong answers.",id:"def-hellaswag"}),e.jsx(n,{title:"Common Benchmark Suite",problem:"Which benchmarks should you run after fine-tuning?",steps:[{formula:"\\text{MMLU: knowledge breadth}",explanation:"Tests whether fine-tuning preserved general knowledge (watch for regression)."},{formula:"\\text{HellaSwag: commonsense}",explanation:"Tests commonsense reasoning -- should remain stable after domain fine-tuning."},{formula:"\\text{TruthfulQA: hallucination}",explanation:"Tests whether the model generates truthful answers vs plausible-sounding falsehoods."},{formula:"\\text{HumanEval / MBPP: coding}",explanation:"For code-focused fine-tunes, measure pass@k on code generation benchmarks."},{formula:"\\text{MT-Bench: conversation}",explanation:"Multi-turn benchmark scored by GPT-4 to evaluate chat quality (1-10 scale)."}],id:"example-benchmark-suite"}),e.jsx(t,{title:"run_lm_eval.py",code:`# lm-evaluation-harness is the standard tool for running benchmarks
# Install: pip install lm-eval

# Run MMLU (5-shot) from command line
# lm_eval --model hf \\
#     --model_args pretrained=./my-finetuned-model \\
#     --tasks mmlu \\
#     --num_fewshot 5 \\
#     --batch_size 4 \\
#     --output_path ./eval_results/mmlu

# Run multiple benchmarks at once
# lm_eval --model hf \\
#     --model_args pretrained=./my-finetuned-model,dtype=float16 \\
#     --tasks mmlu,hellaswag,truthfulqa_mc2,winogrande,arc_challenge \\
#     --batch_size auto \\
#     --output_path ./eval_results/full_suite

# Python API usage
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

model = HFLM(
    pretrained="./my-finetuned-model",
    dtype="float16",
    batch_size=4,
)

results = evaluator.simple_evaluate(
    model=model,
    tasks=["mmlu", "hellaswag", "truthfulqa_mc2"],
    num_fewshot=5,
    batch_size=4,
)

# Print results
for task, metrics in results["results"].items():
    acc = metrics.get("acc,none", metrics.get("acc_norm,none", "N/A"))
    print(f"{task}: {acc:.4f}")

# Save results to JSON
import json
with open("eval_results.json", "w") as f:
    json.dump(results["results"], f, indent=2)`,id:"code-lm-eval"}),e.jsx(t,{title:"compare_base_vs_finetuned.py",code:`import json
import os

def compare_results(base_path, ft_path):
    """Compare benchmark results between base and finetuned models."""
    with open(base_path) as f:
        base = json.load(f)
    with open(ft_path) as f:
        ft = json.load(f)

    print(f"{'Benchmark':<25} {'Base':>8} {'Finetuned':>10} {'Delta':>8}")
    print("-" * 55)

    for task in base:
        if task not in ft:
            continue
        base_acc = base[task].get("acc,none", base[task].get("acc_norm,none", 0))
        ft_acc = ft[task].get("acc,none", ft[task].get("acc_norm,none", 0))
        delta = ft_acc - base_acc
        marker = "+" if delta > 0 else ""
        print(f"{task:<25} {base_acc:>8.4f} {ft_acc:>10.4f} {marker}{delta:>7.4f}")

        if delta < -0.02:
            print(f"  ** WARNING: regression of {abs(delta)*100:.1f}% on {task}")

# Example output:
# Benchmark                     Base  Finetuned    Delta
# -------------------------------------------------------
# mmlu                        0.6820     0.6910  +0.0090
# hellaswag                   0.8210     0.8250  +0.0040
# truthfulqa_mc2              0.5100     0.5350  +0.0250

compare_results("eval_results_base.json", "eval_results_ft.json")`,id:"code-compare"}),e.jsx(a,{type:"tip",title:"Use the Open LLM Leaderboard Format",content:"If you want to submit your model to the Hugging Face Open LLM Leaderboard, run the exact tasks and settings they specify: MMLU (5-shot), HellaSwag (10-shot), TruthfulQA (0-shot), Winogrande (5-shot), GSM8K (5-shot), and ARC-Challenge (25-shot).",id:"note-leaderboard"}),e.jsx(i,{title:"Benchmark Contamination",content:"If your training data contains benchmark questions or answers, your scores will be inflated and meaningless. Always check for data contamination by searching your training set for benchmark examples. The lm-evaluation-harness has decontamination tools to help with this.",id:"warning-contamination"}),e.jsx(a,{type:"note",title:"Benchmark Limitations",content:"Multiple-choice benchmarks like MMLU test recognition rather than generation. A model might score well on MMLU but still generate poor free-form text. Use benchmarks as one signal among many, not as the sole evaluation criterion.",id:"note-limitations"})]})}const We=Object.freeze(Object.defineProperty({__proto__:null,default:$},Symbol.toStringTag,{value:"Module"}));function Q(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Human Evaluation Protocols"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Automated benchmarks cannot capture everything that matters about model quality. Human evaluation remains the gold standard for judging fluency, helpfulness, safety, and alignment with real user needs. This section covers practical protocols for running human evaluations on your fine-tuned models."}),e.jsx(o,{title:"Human Evaluation",definition:"Human evaluation involves having human raters judge model outputs on defined criteria. Common approaches include Likert-scale ratings (1-5), pairwise comparisons (A vs B), and ranking. Inter-annotator agreement is measured by Cohen's kappa: $\\kappa = \\frac{p_o - p_e}{1 - p_e}$ where $p_o$ is observed agreement and $p_e$ is expected chance agreement.",id:"def-human-eval"}),e.jsx(n,{title:"Evaluation Dimensions",problem:"What criteria should human raters evaluate?",steps:[{formula:"\\text{Helpfulness: does the answer solve the user's problem?}",explanation:"Rate 1-5 whether the response directly addresses the query and provides useful information."},{formula:"\\text{Accuracy: is the information factually correct?}",explanation:"Check claims against known facts. Flag hallucinations and errors."},{formula:"\\text{Coherence: is the response well-structured and logical?}",explanation:"Evaluate grammar, flow, and logical consistency."},{formula:"\\text{Safety: does the response avoid harmful content?}",explanation:"Check for toxicity, bias, and inappropriate suggestions."}],id:"example-eval-dimensions"}),e.jsx(t,{title:"human_eval_pipeline.py",code:`import json
import random
from datetime import datetime

def create_evaluation_batch(prompts, model_paths, num_samples=50):
    """Generate evaluation samples from multiple models."""
    from transformers import pipeline

    samples = []
    selected_prompts = random.sample(prompts, min(num_samples, len(prompts)))

    for model_path in model_paths:
        pipe = pipeline(
            "text-generation", model=model_path,
            torch_dtype="float16", device_map="auto"
        )

        for prompt in selected_prompts:
            output = pipe(
                prompt, max_new_tokens=512,
                temperature=0.7, do_sample=True
            )[0]["generated_text"]

            samples.append({
                "id": f"{hash(prompt + model_path) % 10000:04d}",
                "prompt": prompt,
                "response": output[len(prompt):].strip(),
                "model": model_path,
                "timestamp": datetime.now().isoformat(),
            })

    # Shuffle to blind raters to model identity
    random.shuffle(samples)

    # Remove model labels for blind evaluation
    blind_samples = [
        {k: v for k, v in s.items() if k != "model"} for s in samples
    ]

    with open("eval_batch_blind.json", "w") as f:
        json.dump(blind_samples, f, indent=2)
    with open("eval_batch_key.json", "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Created {len(samples)} samples for evaluation")
    return samples

prompts = [
    "Explain how photosynthesis works in simple terms.",
    "Write a Python function to find the longest palindromic substring.",
    "What are the pros and cons of remote work?",
    "How do I fix a leaking kitchen faucet?",
]

create_evaluation_batch(
    prompts,
    model_paths=["./base-model", "./finetuned-model"],
    num_samples=50
)`,id:"code-eval-pipeline"}),e.jsx(t,{title:"compute_agreement.py",code:`import numpy as np
from collections import Counter

def cohens_kappa(rater1, rater2):
    """Compute Cohen's kappa for inter-annotator agreement."""
    assert len(rater1) == len(rater2)
    n = len(rater1)

    # Observed agreement
    p_o = sum(a == b for a, b in zip(rater1, rater2)) / n

    # Expected agreement by chance
    counts1 = Counter(rater1)
    counts2 = Counter(rater2)
    categories = set(list(counts1.keys()) + list(counts2.keys()))
    p_e = sum((counts1[c] / n) * (counts2[c] / n) for c in categories)

    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0
    return kappa

def analyze_ratings(ratings_file, key_file):
    """Analyze human evaluation results."""
    import json
    with open(ratings_file) as f:
        ratings = json.load(f)
    with open(key_file) as f:
        key = json.load(f)

    id_to_model = {s["id"]: s["model"] for s in key}

    model_scores = {}
    for r in ratings:
        model = id_to_model[r["id"]]
        if model not in model_scores:
            model_scores[model] = []
        model_scores[model].append(r["score"])

    for model, scores in model_scores.items():
        print(f"{model}:")
        print(f"  Mean: {np.mean(scores):.2f} +/- {np.std(scores):.2f}")
        print(f"  Median: {np.median(scores):.1f}")
        print(f"  N: {len(scores)}")

# Example: kappa between two raters
rater1 = [4, 3, 5, 2, 4, 3, 5, 4, 3, 4]
rater2 = [4, 2, 5, 3, 4, 3, 4, 4, 3, 5]
print(f"Cohen's kappa: {cohens_kappa(rater1, rater2):.3f}")`,id:"code-agreement"}),e.jsx(a,{type:"tip",title:"Pairwise Comparison Is More Reliable",content:"Asking raters 'Which response is better, A or B?' produces more consistent results than absolute ratings on a 1-5 scale. Pairwise comparison reduces rater calibration issues and is the approach used by Chatbot Arena and LMSYS.",id:"note-pairwise"}),e.jsx(i,{title:"Minimum Rater Count",content:"Use at least 3 raters per sample and require majority agreement. A single rater introduces too much subjective bias. For high-stakes evaluations, use 5+ raters and measure inter-annotator agreement with Cohen's kappa before trusting the results.",id:"warning-rater-count"}),e.jsx(a,{type:"note",title:"LLM-as-Judge",content:"Using GPT-4 or Claude as automated evaluators (LLM-as-judge) is increasingly common as a scalable proxy for human evaluation. While faster and cheaper, it has known biases such as preferring longer responses and outputs similar to its own style. Always validate LLM-as-judge with a human evaluation subset.",id:"note-llm-judge"})]})}const He=Object.freeze(Object.defineProperty({__proto__:null,default:Q},Symbol.toStringTag,{value:"Module"}));function W(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"A/B Testing Fine-tuned Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"A/B testing compares two model variants under real-world conditions by routing traffic between them and measuring user-facing metrics. It is the definitive way to determine whether a fine-tuned model improves production outcomes compared to the baseline."}),e.jsx(o,{title:"A/B Testing for Models",definition:"A/B testing (split testing) randomly assigns users or requests to one of two model variants and compares outcome metrics. Statistical significance is determined using hypothesis testing: given $n_A$ and $n_B$ samples with success rates $p_A$ and $p_B$, we compute $z = \\frac{p_B - p_A}{\\sqrt{p(1-p)(1/n_A + 1/n_B)}}$ where $p$ is the pooled rate.",id:"def-ab-testing"}),e.jsx(n,{title:"Designing a Model A/B Test",problem:"What metrics should you track when A/B testing a fine-tuned model?",steps:[{formula:"\\text{Primary: task completion rate}",explanation:"Did the user accomplish their goal? This is the most important signal."},{formula:"\\text{Latency: } t_{\\text{p50}}, t_{\\text{p95}}, t_{\\text{p99}}",explanation:"Response time percentiles -- a better model is useless if too slow."},{formula:"\\text{User satisfaction: thumbs up/down ratio}",explanation:"Direct feedback signal from users interacting with the model."},{formula:"\\text{Engagement: follow-up rate, session length}",explanation:"Indirect signals of quality -- users engage more with better models."}],id:"example-ab-metrics"}),e.jsx(t,{title:"ab_test_router.py",code:`import random
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class ABTestConfig:
    experiment_name: str
    model_a: str  # control (base/current model)
    model_b: str  # treatment (fine-tuned model)
    traffic_split: float = 0.5  # fraction going to model_b
    min_samples: int = 1000

class ABTestRouter:
    def __init__(self, config: ABTestConfig):
        self.config = config
        self.results = {"model_a": [], "model_b": []}

    def route_request(self, request_id: str) -> str:
        """Deterministically route request to model A or B."""
        bucket = hash(request_id) % 100
        if bucket < self.config.traffic_split * 100:
            return self.config.model_b
        return self.config.model_a

    def log_result(self, request_id: str, model: str, latency: float,
                   success: bool, user_rating: Optional[int] = None):
        key = "model_b" if model == self.config.model_b else "model_a"
        self.results[key].append({
            "request_id": request_id,
            "latency": latency,
            "success": success,
            "user_rating": user_rating,
            "timestamp": time.time(),
        })

    def compute_significance(self):
        """Compute statistical significance of A/B test results."""
        import numpy as np
        from scipy import stats

        a_success = [r["success"] for r in self.results["model_a"]]
        b_success = [r["success"] for r in self.results["model_b"]]

        n_a, n_b = len(a_success), len(b_success)
        p_a = sum(a_success) / n_a
        p_b = sum(b_success) / n_b
        p_pool = (sum(a_success) + sum(b_success)) / (n_a + n_b)

        se = (p_pool * (1 - p_pool) * (1/n_a + 1/n_b)) ** 0.5
        z = (p_b - p_a) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return {
            "model_a_rate": p_a, "model_b_rate": p_b,
            "lift": (p_b - p_a) / p_a if p_a > 0 else 0,
            "z_score": z, "p_value": p_value,
            "significant": p_value < 0.05,
            "n_a": n_a, "n_b": n_b,
        }

config = ABTestConfig(
    experiment_name="llama3-finetune-v2",
    model_a="./base-model",
    model_b="./finetuned-model",
    traffic_split=0.2,
)
router = ABTestRouter(config)`,id:"code-ab-router"}),e.jsx(t,{title:"ab_test_analysis.py",code:`import numpy as np

def analyze_ab_test(results_a, results_b, metric="success"):
    """Analyze A/B test results and print summary."""
    vals_a = [r[metric] for r in results_a if metric in r]
    vals_b = [r[metric] for r in results_b if metric in r]

    print(f"{'Metric':<20} {'Model A':>10} {'Model B':>10}")
    print("-" * 42)
    print(f"{'N samples':<20} {len(vals_a):>10} {len(vals_b):>10}")
    print(f"{'Mean':<20} {np.mean(vals_a):>10.4f} {np.mean(vals_b):>10.4f}")
    print(f"{'Std':<20} {np.std(vals_a):>10.4f} {np.std(vals_b):>10.4f}")

    lat_a = [r["latency"] for r in results_a]
    lat_b = [r["latency"] for r in results_b]
    print(f"{'Latency p50':<20} {np.percentile(lat_a, 50):>10.3f}s"
          f" {np.percentile(lat_b, 50):>10.3f}s")
    print(f"{'Latency p95':<20} {np.percentile(lat_a, 95):>10.3f}s"
          f" {np.percentile(lat_b, 95):>10.3f}s")

def required_sample_size(baseline_rate, mde, alpha=0.05, power=0.8):
    """Minimum samples per group to detect minimum detectable effect."""
    from scipy.stats import norm
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    p1, p2 = baseline_rate, baseline_rate + mde
    n = ((z_a + z_b) ** 2 * (p1*(1-p1) + p2*(1-p2))) / mde**2
    return int(np.ceil(n))

print(f"Samples needed for 2% MDE: {required_sample_size(0.75, 0.02)}")`,id:"code-ab-analysis"}),e.jsx(a,{type:"tip",title:"Start with Low Traffic",content:"Begin with 5-10% traffic on the new model and ramp up gradually. This limits blast radius if the fine-tuned model has unexpected failure modes. Only go to 50/50 after initial metrics look stable.",id:"note-low-traffic"}),e.jsx(i,{title:"Do Not Peek at Results",content:"Checking results repeatedly and stopping the test early when you see a positive signal inflates false positive rates (the peeking problem). Pre-register your sample size and significance threshold before starting. Use sequential testing methods if you must monitor continuously.",id:"warning-peeking"})]})}const Ve=Object.freeze(Object.defineProperty({__proto__:null,default:W},Symbol.toStringTag,{value:"Module"}));function H(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Common Failure Modes & Debugging"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Fine-tuning often fails in subtle ways. The model trains without errors but produces bad outputs. This section catalogs the most common failure modes, their symptoms, and practical debugging strategies to identify and fix them."}),e.jsx(o,{title:"Catastrophic Forgetting",definition:"Catastrophic forgetting occurs when fine-tuning overwrites the model's pre-trained knowledge. The model performs well on the fine-tuning task but loses general capabilities. Formally, if $\\mathcal{L}_{\\text{general}}$ increases significantly while $\\mathcal{L}_{\\text{task}}$ decreases, the model has forgotten.",id:"def-catastrophic-forgetting"}),e.jsx(n,{title:"Failure Mode Checklist",problem:"What are the most common fine-tuning failures and their symptoms?",steps:[{formula:"\\text{Repetition loops: model repeats phrases endlessly}",explanation:"Caused by too high a learning rate or too many epochs. Reduce LR or add repetition penalty."},{formula:"\\text{Format collapse: wrong output format}",explanation:"Model ignores chat template or outputs raw tokens. Check tokenizer chat_template and training format."},{formula:"\\text{Hallucination increase: more confident lies}",explanation:"Fine-tuning on noisy data teaches the model to generate plausible-sounding but incorrect text."},{formula:"\\text{Catastrophic forgetting: lost general knowledge}",explanation:"Too many epochs on narrow data. Use LoRA, lower LR, or mix in general data."},{formula:"\\text{Mode collapse: same response to different inputs}",explanation:"Training data too homogeneous. Increase dataset diversity or reduce epochs."}],id:"example-failure-checklist"}),e.jsx(t,{title:"debug_model_outputs.py",code:`from transformers import pipeline
import torch

def diagnose_model(model_path, test_prompts):
    """Run diagnostic tests on a fine-tuned model."""
    pipe = pipeline(
        "text-generation", model=model_path,
        torch_dtype=torch.float16, device_map="auto"
    )

    issues = []

    for prompt in test_prompts:
        output = pipe(
            prompt, max_new_tokens=256, temperature=0.7,
            do_sample=True, return_full_text=False
        )[0]["generated_text"]

        # Check for repetition
        words = output.split()
        if len(words) > 20:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            unique_ratio = len(set(trigrams)) / len(trigrams) if trigrams else 1
            if unique_ratio < 0.5:
                issues.append(
                    f"REPETITION detected (unique trigram ratio: {unique_ratio:.2f})")
                issues.append(f"  Prompt: {prompt[:80]}...")

        # Check for empty or very short responses
        if len(output.strip()) < 10:
            issues.append(f"EMPTY/SHORT response: '{output.strip()}'")
            issues.append(f"  Prompt: {prompt[:80]}...")

        # Check for leaked special tokens
        special_tokens = ["<|", "</s>", "[INST]", "<s>"]
        for token in special_tokens:
            if token in output:
                issues.append(f"SPECIAL TOKEN leaked: '{token}' in output")
                break

        # Check for mode collapse
        outputs_set = set()
        for _ in range(3):
            o = pipe(prompt, max_new_tokens=100, temperature=0.7,
                     do_sample=True, return_full_text=False)[0]["generated_text"]
            outputs_set.add(o.strip())
        if len(outputs_set) == 1:
            issues.append(f"MODE COLLAPSE: identical outputs for: {prompt[:60]}...")

    if not issues:
        print("No obvious issues detected.")
    else:
        print(f"Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")

    return issues

test_prompts = [
    "What is the capital of France?",
    "Write a haiku about the ocean.",
    "Explain quantum entanglement simply.",
    "def fibonacci(n):",
    "Translate to Spanish: The weather is nice today.",
]

diagnose_model("./my-finetuned-model", test_prompts)`,id:"code-diagnose"}),e.jsx(t,{title:"check_training_data.py",code:`import json
from collections import Counter

def audit_training_data(data_path):
    """Check training data for common issues."""
    with open(data_path) as f:
        data = [json.loads(line) for line in f]

    print(f"Total samples: {len(data)}")

    # Check for duplicates
    texts = [json.dumps(d, sort_keys=True) for d in data]
    dupes = len(texts) - len(set(texts))
    if dupes > 0:
        print(f"WARNING: {dupes} duplicate samples ({dupes/len(data)*100:.1f}%)")

    # Check response lengths
    lengths = []
    for d in data:
        resp = d.get("output", d.get("response", d.get("completion", "")))
        lengths.append(len(resp.split()))

    print(f"Response length: min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.0f}")

    empty = sum(1 for l in lengths if l == 0)
    if empty > 0:
        print(f"WARNING: {empty} samples with empty responses")

    if "label" in data[0]:
        labels = Counter(d["label"] for d in data)
        print(f"Label distribution: {dict(labels)}")
        max_ratio = max(labels.values()) / min(labels.values())
        if max_ratio > 10:
            print(f"WARNING: Imbalanced labels (ratio {max_ratio:.1f}:1)")

    short = sum(
        1 for d in data
        if len(str(d.get("input", d.get("instruction", "")))) < 10
    )
    if short > 0:
        print(f"WARNING: {short} samples with very short inputs")

audit_training_data("training_data.jsonl")`,id:"code-audit-data"}),e.jsx(i,{title:"The Training Data Is Almost Always the Problem",content:"When a fine-tuned model produces bad outputs, the root cause is almost always the training data -- not the hyperparameters. Before tuning learning rates or LoRA rank, manually inspect 50-100 training examples for quality issues: formatting errors, incorrect labels, low-quality responses, or mismatched instruction-response pairs.",id:"warning-data-first"}),e.jsx(a,{type:"tip",title:"Quick Debugging Checklist",content:"1) Check a few training examples manually. 2) Compare base model vs fine-tuned outputs on the same prompts. 3) Look at the loss curve -- is it still decreasing or has it plateaued? 4) Test with temperature=0 to see deterministic output. 5) Check if the chat template matches between training and inference.",id:"note-debug-checklist"}),e.jsx(a,{type:"note",title:"Loss Can Be Misleading",content:"A low training loss does not guarantee good model quality. The model could be memorizing training data verbatim (overfitting) or learning superficial patterns. Always evaluate on held-out examples that were not in the training set.",id:"note-loss-misleading"})]})}const Ke=Object.freeze(Object.defineProperty({__proto__:null,default:H},Symbol.toStringTag,{value:"Module"}));function V(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"When to Retrain vs Adjust"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"After evaluating your fine-tuned model, you face a decision: retrain from scratch with different data or hyperparameters, or make adjustments to the existing model. This section provides a decision framework and practical strategies for iterating on fine-tuned models."}),e.jsx(o,{title:"Iterative Fine-tuning",definition:"Iterative fine-tuning involves making incremental improvements to a model through successive rounds of training, evaluation, and adjustment. The key trade-off is between the cost of retraining (time, compute) and the expected improvement from each iteration strategy.",id:"def-iterative"}),e.jsx(n,{title:"Decision Framework",problem:"Should you retrain from scratch or adjust the existing model?",steps:[{formula:"\\text{Data quality issues} \\Rightarrow \\text{Fix data, retrain}",explanation:"If training data has errors, duplicates, or wrong formats, fix the data and retrain from the base model."},{formula:"\\text{Underfitting (high loss)} \\Rightarrow \\text{Adjust hyperparams}",explanation:"Increase LoRA rank, learning rate, or epochs. Can continue training from checkpoint."},{formula:"\\text{Overfitting (gap between train/val loss)} \\Rightarrow \\text{Adjust}",explanation:"Reduce epochs, increase dropout, or add regularization. Resume from an earlier checkpoint."},{formula:"\\text{Wrong task behavior} \\Rightarrow \\text{Fix data, retrain}",explanation:"If the model learned the wrong behavior, the data or format needs changing."},{formula:"\\text{Needs more capability} \\Rightarrow \\text{Add data, continue}",explanation:"If the model is good but needs more coverage, add examples and continue training."}],id:"example-decision-framework"}),e.jsx(t,{title:"continue_training.py",code:`from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel, get_peft_model, LoraConfig
from trl import SFTTrainer
from datasets import load_dataset

# Strategy 1: Continue training from checkpoint
def continue_from_checkpoint(checkpoint_dir, new_data_path, extra_epochs=1):
    """Resume training from a saved checkpoint with additional data."""
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    dataset = load_dataset("json", data_files=new_data_path, split="train")

    training_args = TrainingArguments(
        output_dir="./continued-training",
        num_train_epochs=extra_epochs,
        learning_rate=5e-5,  # lower LR for continued training
        per_device_train_batch_size=4,
        save_strategy="steps",
        save_steps=100,
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=dataset, args=training_args,
    )
    trainer.train()
    return model

# Strategy 2: Merge and re-adapt
def merge_and_readapt(base_model_path, adapter_path, new_data_path):
    """Merge existing LoRA, then train a new adapter."""
    import torch

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()

    new_lora = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
    )
    model = get_peft_model(model, new_lora)
    print(f"New trainable params: {model.print_trainable_parameters()}")
    return model

# Strategy 3: Selective data refinement
def refine_dataset(original_data, model_path, threshold=0.8):
    """Remove low-quality samples based on model confidence."""
    from transformers import pipeline

    pipe = pipeline("text-generation", model=model_path, device_map="auto")
    refined = []

    for sample in original_data:
        refined.append(sample)  # Add filtering logic based on loss

    print(f"Refined: {len(refined)}/{len(original_data)} samples kept")
    return refined`,id:"code-continue-training"}),e.jsx(t,{title:"hyperparameter_iteration.py",code:`# Quick reference for hyperparameter adjustments

ITERATION_PLAYBOOK = {
    "model_repeats_itself": {
        "diagnosis": "Learning rate too high or too many epochs",
        "adjustments": [
            "Reduce learning_rate by 2-5x",
            "Reduce num_train_epochs",
            "Add repetition_penalty=1.1 at inference",
            "Check for duplicate training samples",
        ]
    },
    "model_ignores_finetuning": {
        "diagnosis": "Learning rate too low or too few epochs",
        "adjustments": [
            "Increase learning_rate by 2-5x",
            "Increase LoRA rank (r=16 -> r=32 -> r=64)",
            "Add more target modules (include MLP layers)",
            "Increase num_train_epochs",
        ]
    },
    "model_forgets_general_knowledge": {
        "diagnosis": "Catastrophic forgetting from over-training",
        "adjustments": [
            "Reduce epochs (try 1 epoch first)",
            "Lower learning_rate",
            "Mix in 10-20% general instruction data",
            "Use lower LoRA rank to limit capacity",
        ]
    },
    "high_train_loss_wont_decrease": {
        "diagnosis": "Data format mismatch or model capacity issue",
        "adjustments": [
            "Verify chat template matches training format exactly",
            "Check that labels are not masked incorrectly",
            "Increase LoRA rank or add target modules",
            "Try a larger base model",
        ]
    },
}

for issue, info in ITERATION_PLAYBOOK.items():
    print(f"\\n{'='*60}")
    print(f"Issue: {issue}")
    print(f"Diagnosis: {info['diagnosis']}")
    print("Adjustments:")
    for adj in info["adjustments"]:
        print(f"  - {adj}")`,id:"code-playbook"}),e.jsx(a,{type:"intuition",title:"The 80/20 Rule of Fine-tuning",content:"80% of improvement comes from data quality, 20% from hyperparameters. If your first attempt does not work well, spend your time improving the training data before running grid searches over learning rates and LoRA configurations.",id:"note-8020"}),e.jsx(i,{title:"Do Not Stack Too Many LoRA Adapters",content:"While you can merge one LoRA and train another on top, stacking more than 2-3 rounds of fine-tuning often degrades quality. Each round adds noise. If you find yourself iterating many times, consider retraining from the base model with a curated combined dataset.",id:"warning-stacking"}),e.jsx(a,{type:"tip",title:"Keep a Training Log",content:"Document every training run: dataset version, hyperparameters, evaluation metrics, and qualitative observations. This log becomes invaluable when deciding what to try next. Tools like Weights & Biases (wandb) automate much of this tracking.",id:"note-training-log"})]})}const Xe=Object.freeze(Object.defineProperty({__proto__:null,default:V},Symbol.toStringTag,{value:"Module"}));function K(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Hardware for Diffusion Fine-tuning"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Fine-tuning diffusion models for image and video generation has different hardware requirements than LLM fine-tuning. VRAM is the primary bottleneck, and the requirements vary dramatically between methods like DreamBooth, LoRA, and Textual Inversion."}),e.jsx(o,{title:"VRAM Requirements for Diffusion Training",definition:"Diffusion model training stores the model weights, optimizer states, gradients, and intermediate activations in GPU memory. For a model with $P$ parameters at mixed precision, the minimum VRAM is approximately $\\text{VRAM} \\approx 2P + 8P_{\\text{trainable}} + A$ bytes, where $A$ covers activations and batch data.",id:"def-vram-diffusion"}),e.jsx(n,{title:"VRAM by Method and Model",problem:"How much VRAM do different diffusion fine-tuning methods require?",steps:[{formula:"\\text{SD 1.5 LoRA: } \\sim 6\\text{ GB}",explanation:"Stable Diffusion 1.5 with LoRA rank 4-8 fits on consumer GPUs easily."},{formula:"\\text{SDXL LoRA: } \\sim 12\\text{ GB}",explanation:"SDXL is larger; fits on RTX 3060 12GB or better with gradient checkpointing."},{formula:"\\text{SDXL DreamBooth: } \\sim 24\\text{ GB}",explanation:"Full DreamBooth training requires RTX 3090/4090 or A5000."},{formula:"\\text{Flux LoRA: } \\sim 16\\text{-}24\\text{ GB}",explanation:"Flux.1 models are large; LoRA training needs RTX 4090 or A100."},{formula:"\\text{Video (AnimateDiff/CogVideoX): } \\sim 24\\text{-}80\\text{ GB}",explanation:"Video models need A100 80GB or multi-GPU setups."}],id:"example-vram-table"}),e.jsx(t,{title:"check_gpu_setup.py",code:`import torch

def check_gpu_capabilities():
    """Check GPU setup for diffusion model training."""
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected!")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}\\n")

    total_vram = 0
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_mem / 1024**3
        total_vram += vram_gb
        print(f"GPU {i}: {props.name}")
        print(f"  VRAM:         {vram_gb:.1f} GB")
        print(f"  Compute:      {props.major}.{props.minor}")
        print(f"  SMs:          {props.multi_processor_count}")

    gpu_vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"\\n--- Recommendations for {gpu_vram:.0f} GB VRAM ---")

    methods = []
    if gpu_vram >= 6:
        methods.append("SD 1.5 LoRA / Textual Inversion")
    if gpu_vram >= 12:
        methods.append("SDXL LoRA (with gradient checkpointing)")
    if gpu_vram >= 16:
        methods.append("Flux LoRA (fp8 or quantized)")
    if gpu_vram >= 24:
        methods.append("SDXL DreamBooth / Flux LoRA (bf16)")
        methods.append("AnimateDiff LoRA (short clips)")
    if gpu_vram >= 48:
        methods.append("CogVideoX LoRA")
    if gpu_vram >= 80:
        methods.append("Full diffusion model fine-tuning")
        methods.append("Video model DreamBooth")

    for m in methods:
        print(f"  OK: {m}")

    if gpu_vram < 6:
        print("  Insufficient VRAM for local training.")
        print("    Consider: RunPod, Vast.ai, or Google Colab Pro")

check_gpu_capabilities()

# Check xformers (memory-efficient attention)
try:
    import xformers
    print(f"\\nxformers {xformers.__version__} installed")
except ImportError:
    print("\\nxformers not installed - install for 20-30% VRAM savings:")
    print("  pip install xformers")`,id:"code-check-gpu"}),e.jsx(t,{title:"cloud_gpu_pricing.py",code:`# Quick reference: cloud GPU pricing for diffusion training

CLOUD_OPTIONS = {
    "RunPod": {
        "RTX 4090 (24 GB)":  {"hourly": 0.44, "good_for": "SDXL LoRA/DreamBooth"},
        "A100 80GB":         {"hourly": 1.64, "good_for": "Video models, large batches"},
        "H100 80GB":         {"hourly": 3.29, "good_for": "Fast training"},
    },
    "Vast.ai": {
        "RTX 4090 (24 GB)":  {"hourly": 0.30, "good_for": "Budget SDXL training"},
        "A100 40GB":         {"hourly": 0.90, "good_for": "Flux LoRA, AnimateDiff"},
    },
    "Lambda Labs": {
        "A100 80GB":         {"hourly": 1.10, "good_for": "Reliable longer runs"},
        "H100 80GB":         {"hourly": 2.49, "good_for": "Production pipelines"},
    },
}

def estimate_training_cost(method, target_steps=1000):
    """Estimate cloud training cost."""
    time_per_1k = {
        "sd15_lora":        {"gpu": "RTX 4090", "minutes": 15},
        "sdxl_lora":        {"gpu": "RTX 4090", "minutes": 30},
        "sdxl_dreambooth":  {"gpu": "RTX 4090", "minutes": 45},
        "flux_lora":        {"gpu": "A100 80GB", "minutes": 60},
        "animatediff_lora": {"gpu": "A100 80GB", "minutes": 120},
        "cogvideox_lora":   {"gpu": "A100 80GB", "minutes": 180},
    }

    info = time_per_1k.get(method, {"gpu": "A100 80GB", "minutes": 60})
    hours = (info["minutes"] * target_steps / 1000) / 60
    cost = hours * 1.64
    print(f"{method}: ~{hours:.1f}h on {info['gpu']} = ~USD{cost:.2f}")

for method in ["sd15_lora", "sdxl_lora", "flux_lora", "cogvideox_lora"]:
    estimate_training_cost(method)`,id:"code-pricing"}),e.jsx(a,{type:"tip",title:"Gradient Checkpointing Saves VRAM",content:"Enable gradient checkpointing to trade compute time for memory. This recomputes intermediate activations during the backward pass instead of storing them, typically saving 30-50% VRAM at the cost of ~20% slower training. Almost always worth it for consumer GPUs.",id:"note-grad-checkpoint"}),e.jsx(i,{title:"VRAM Estimates Are Approximate",content:"Actual VRAM usage depends on image resolution, batch size, gradient accumulation steps, and whether xformers or flash attention is enabled. Always test with a single training step before committing to a long run. Use torch.cuda.max_memory_allocated() to measure actual peak usage.",id:"warning-vram-estimates"})]})}const Ye=Object.freeze(Object.defineProperty({__proto__:null,default:K},Symbol.toStringTag,{value:"Module"}));function X(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"DreamBooth Training Step-by-Step"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"DreamBooth fine-tunes the entire diffusion model to learn a specific subject from just 3-10 images. It binds a unique identifier token to the concept, allowing you to generate the subject in novel contexts. This section walks through the full DreamBooth training pipeline using the diffusers library."}),e.jsx(o,{title:"DreamBooth",definition:"DreamBooth fine-tunes all parameters of a diffusion model to associate a rare token identifier (e.g., 'sks') with a specific subject. The training loss combines a reconstruction term on subject images with an optional prior preservation term: $\\mathcal{L} = \\mathbb{E}[\\|{\\epsilon - \\epsilon_\\theta(x_t, t, c_{\\text{sks}})}\\|^2] + \\lambda \\mathbb{E}[\\|\\epsilon - \\epsilon_\\theta(x_t, t, c_{\\text{class}})\\|^2]$ where $\\lambda$ controls prior preservation strength.",id:"def-dreambooth"}),e.jsx(n,{title:"DreamBooth Training Checklist",problem:"What do you need for a successful DreamBooth training run?",steps:[{formula:"\\text{3-10 high-quality images of the subject}",explanation:"Diverse angles, lighting, and backgrounds. Crop to subject, resize to model resolution."},{formula:"\\text{Unique token: } \\texttt{sks}, \\texttt{ohwx}",explanation:"Choose a rare token unlikely to have strong existing associations in the model."},{formula:'\\text{Class prompt: "a photo of a [class]"}',explanation:"The general category (person, dog, car) for prior preservation regularization."},{formula:"\\text{200-1000 prior preservation images}",explanation:"Generated from the base model using the class prompt to prevent language drift."}],id:"example-checklist"}),e.jsx(t,{title:"dreambooth_training.py",code:`# DreamBooth training with diffusers
# Requires: pip install diffusers[training] accelerate transformers

import os

INSTANCE_DIR = "./data/dreambooth/sks_dog"
CLASS_DIR = "./data/dreambooth/class_dog"
OUTPUT_DIR = "./models/dreambooth-dog"
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"

# Step 1: Launch DreamBooth LoRA training
TRAIN_CMD = f"""
accelerate launch diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py \\
    --pretrained_model_name_or_path="{MODEL_NAME}" \\
    --instance_data_dir="{INSTANCE_DIR}" \\
    --class_data_dir="{CLASS_DIR}" \\
    --output_dir="{OUTPUT_DIR}" \\
    --instance_prompt="a photo of sks dog" \\
    --class_prompt="a photo of a dog" \\
    --with_prior_preservation \\
    --prior_loss_weight=1.0 \\
    --num_class_images=200 \\
    --resolution=1024 \\
    --train_batch_size=1 \\
    --gradient_accumulation_steps=4 \\
    --gradient_checkpointing \\
    --learning_rate=1e-4 \\
    --lr_scheduler="constant" \\
    --lr_warmup_steps=0 \\
    --max_train_steps=500 \\
    --seed=42 \\
    --mixed_precision="bf16" \\
    --enable_xformers_memory_efficient_attention
"""

print("Place images in", INSTANCE_DIR)
print("Training command:")
print(TRAIN_CMD)`,id:"code-dreambooth-train"}),e.jsx(t,{title:"dreambooth_inference.py",code:`import torch
from diffusers import DiffusionPipeline

def generate_dreambooth(model_dir, prompts, num_images_per_prompt=4):
    """Generate images using a DreamBooth-trained model."""
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    ).to("cuda")

    # Load the DreamBooth LoRA weights
    pipe.load_lora_weights(model_dir)

    results = []
    for prompt in prompts:
        images = pipe(
            prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images
        results.extend(images)

        for i, img in enumerate(images):
            safe_name = prompt[:30].replace(" ", "_")
            img.save(f"output_{safe_name}_{i}.png")

    return results

prompts = [
    "a photo of sks dog wearing a top hat",
    "a photo of sks dog on the moon",
    "a painting of sks dog in the style of Van Gogh",
    "a photo of sks dog sitting in a cafe in Paris",
]

generate_dreambooth("./models/dreambooth-dog", prompts)`,id:"code-dreambooth-inference"}),e.jsx(a,{type:"tip",title:"DreamBooth + LoRA Is Usually Better",content:"Full DreamBooth fine-tuning modifies all model weights and is prone to overfitting. DreamBooth with LoRA (train_dreambooth_lora) produces comparable results with much less VRAM and risk. Start with LoRA DreamBooth unless you have a specific reason to train all weights.",id:"note-dreambooth-lora"}),e.jsx(i,{title:"Overfitting Is the Biggest Risk",content:"DreamBooth overfits very quickly -- often within 200-400 steps for LoRA. Signs of overfitting: the model generates exact copies of training images, or non-subject generations degrade. Train for fewer steps than you think, and generate test images every 100 steps.",id:"warning-overfit"}),e.jsx(a,{type:"note",title:"Token Selection Matters",content:"The identifier token (sks, ohwx, etc.) should be rare enough that it does not already have a strong meaning in the model. Avoid common words. Some practitioners use random 3-letter strings. Test the token in the base model first to confirm it does not generate a specific concept.",id:"note-token-selection"})]})}const Je=Object.freeze(Object.defineProperty({__proto__:null,default:X},Symbol.toStringTag,{value:"Module"}));function Y(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"LoRA Training for Stable Diffusion"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"LoRA is the most popular method for fine-tuning Stable Diffusion models because it is fast, memory-efficient, and produces small adapter files that can be easily shared and combined. This section covers training LoRA adapters for SD 1.5 and SDXL to learn new styles, characters, and concepts."}),e.jsx(o,{title:"LoRA for Diffusion Models",definition:"LoRA (Low-Rank Adaptation) for diffusion models applies low-rank updates to the UNet cross-attention layers and optionally the text encoder. The weight update is $W' = W + BA$ where $B \\in \\mathbb{R}^{d \\times r}$ and $A \\in \\mathbb{R}^{r \\times k}$ with rank $r \\ll \\min(d, k)$. Typical LoRA files for SDXL are 50-200 MB versus 6.5 GB for full weights.",id:"def-lora-diffusion"}),e.jsx(n,{title:"LoRA Hyperparameters for Stable Diffusion",problem:"What are the recommended LoRA settings for style and character training?",steps:[{formula:"\\text{Rank } r: 4\\text{-}8 \\text{ for styles, } 16\\text{-}32 \\text{ for characters}",explanation:"Higher rank captures more detail but increases file size and overfitting risk."},{formula:"\\text{Alpha } \\alpha = r \\text{ or } 2r",explanation:"The scaling factor. Setting alpha=rank gives effective scaling of 1.0."},{formula:"\\text{Learning rate: } 1 \\times 10^{-4} \\text{ to } 5 \\times 10^{-4}",explanation:"Lower than LLM LoRA rates. Start with 1e-4 and adjust based on results."},{formula:"\\text{Steps: 500-2000 for style, 1000-4000 for character}",explanation:"Fewer images need fewer steps. Monitor for overfitting every 200-500 steps."}],id:"example-hyperparams"}),e.jsx(t,{title:"train_lora_sdxl.py",code:`# LoRA training for SDXL using diffusers
# Install: pip install diffusers[training] accelerate peft

TRAIN_CMD = """
accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py \\
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \\
    --dataset_name="./my_dataset" \\
    --caption_column="text" \\
    --image_column="image" \\
    --output_dir="./sdxl-lora-output" \\
    --resolution=1024 \\
    --train_batch_size=1 \\
    --gradient_accumulation_steps=4 \\
    --num_train_epochs=100 \\
    --learning_rate=1e-4 \\
    --lr_scheduler="cosine" \\
    --lr_warmup_steps=100 \\
    --rank=16 \\
    --seed=42 \\
    --mixed_precision="bf16" \\
    --gradient_checkpointing \\
    --enable_xformers_memory_efficient_attention \\
    --validation_prompt="a painting in the style of sks" \\
    --validation_epochs=25 \\
    --checkpointing_steps=500
"""

# Python API for LoRA configuration
from peft import LoraConfig

def setup_lora_training():
    """Configure LoRA for SDXL training."""
    unet_lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ],
    )

    text_encoder_lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )

    return unet_lora_config, text_encoder_lora_config

unet_config, te_config = setup_lora_training()
print(f"UNet LoRA rank: {unet_config.r}")
print(f"Text encoder LoRA rank: {te_config.r}")
print(TRAIN_CMD)`,id:"code-train-lora"}),e.jsx(t,{title:"use_lora_sdxl.py",code:`import torch
from diffusers import DiffusionPipeline

def load_and_generate(base_model, lora_path, prompt, lora_scale=0.8):
    """Load a LoRA adapter and generate images."""
    pipe = DiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")

    pipe.load_lora_weights(lora_path)

    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
        cross_attention_kwargs={"scale": lora_scale},
    ).images[0]

    return image

# Combine multiple LoRAs
def combine_loras(base_model, lora_configs):
    """Load and combine multiple LoRA adapters."""
    pipe = DiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16
    ).to("cuda")

    for name, path, scale in lora_configs:
        pipe.load_lora_weights(path, adapter_name=name)

    adapter_names = [c[0] for c in lora_configs]
    adapter_weights = [c[2] for c in lora_configs]
    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    return pipe

# Example: combine style + character LoRAs
pipe = combine_loras(
    "stabilityai/stable-diffusion-xl-base-1.0",
    [
        ("style", "./lora-watercolor", 0.7),
        ("character", "./lora-my-character", 0.9),
    ]
)
image = pipe("sks character in watercolor style").images[0]
image.save("combined_lora_output.png")`,id:"code-use-lora"}),e.jsx(a,{type:"tip",title:"Train Text Encoder Too for Characters",content:"For character or object concepts, training the text encoder LoRA alongside the UNet LoRA significantly improves identity preservation. For pure style transfer, UNet-only LoRA is usually sufficient.",id:"note-text-encoder"}),e.jsx(i,{title:"Caption Quality Drives LoRA Quality",content:"The most common cause of bad LoRA results is poor captions. Every training image must have an accurate, detailed caption. Use BLIP-2 or CogVLM to auto-caption, then manually review and edit. Include the trigger word (e.g., 'sks style') in every caption.",id:"warning-captions"}),e.jsx(a,{type:"note",title:"LoRA File Compatibility",content:"LoRA files trained with diffusers can be loaded in ComfyUI, Automatic1111, and other UIs. However, the naming conventions may differ. Use safetensors format for maximum compatibility across tools.",id:"note-compatibility"})]})}const Ze=Object.freeze(Object.defineProperty({__proto__:null,default:Y},Symbol.toStringTag,{value:"Module"}));function J(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"LoRA Training for Flux"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Flux is a next-generation diffusion transformer model from Black Forest Labs that produces highly detailed images with excellent prompt adherence. Training LoRA adapters for Flux follows a similar pattern to SDXL but requires more VRAM due to the larger architecture. This section covers the practical workflow."}),e.jsx(o,{title:"Flux Architecture",definition:"Flux uses a diffusion transformer (DiT) architecture rather than the UNet used in Stable Diffusion. The model processes image patches as tokens through transformer blocks with cross-attention to text embeddings. Flux.1-dev has approximately 12B parameters, making LoRA essential for consumer hardware fine-tuning.",id:"def-flux"}),e.jsx(n,{title:"Flux LoRA Training Parameters",problem:"What settings work best for Flux LoRA training?",steps:[{formula:"\\text{Rank: } r = 16 \\text{ (style)} \\text{ or } 32 \\text{ (character)}",explanation:"Flux benefits from slightly higher ranks than SDXL due to its transformer architecture."},{formula:"\\text{Learning rate: } 1 \\times 10^{-4}",explanation:"Standard starting point. Reduce to 5e-5 if you see artifacts early."},{formula:"\\text{Resolution: 512-1024 (train), 1024+ (inference)}",explanation:"Training at 512 is faster and still transfers well to higher resolutions."},{formula:"\\text{Steps: 500-1500 for 20-50 images}",explanation:"Flux LoRA converges faster than SDXL. Check outputs every 250 steps."}],id:"example-flux-params"}),e.jsx(t,{title:"train_flux_lora.py",code:`# Flux LoRA training using diffusers
# Requires: pip install diffusers[training] accelerate peft bitsandbytes

# Option 1: Using the diffusers training script
TRAIN_CMD = """
accelerate launch diffusers/examples/dreambooth/train_dreambooth_lora_flux.py \\
    --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \\
    --dataset_name="./my_flux_dataset" \\
    --instance_prompt="a photo of sks person" \\
    --output_dir="./flux-lora-output" \\
    --resolution=512 \\
    --train_batch_size=1 \\
    --gradient_accumulation_steps=4 \\
    --learning_rate=1e-4 \\
    --lr_scheduler="constant" \\
    --lr_warmup_steps=0 \\
    --max_train_steps=1000 \\
    --rank=16 \\
    --seed=42 \\
    --mixed_precision="bf16" \\
    --gradient_checkpointing \\
    --validation_prompt="a photo of sks person wearing sunglasses" \\
    --validation_epochs=100
"""

# Option 2: ai-toolkit config
AI_TOOLKIT_CONFIG = """
# config.yaml for ai-toolkit Flux LoRA training
job: extension
config:
  name: flux_lora_my_subject
  process:
    - type: sd_trainer
      training_folder: ./output
      device: cuda:0
      trigger_word: sks
      network:
        type: lora
        linear: 16
        linear_alpha: 16
      datasets:
        - folder_path: ./training_images
          caption_ext: .txt
          resolution: 512
          batch_size: 1
      train:
        steps: 1000
        lr: 1e-4
        optimizer: adamw8bit
"""

print("Flux LoRA Training Options:")
print("1. diffusers script (recommended)")
print(TRAIN_CMD)
print("\\n2. ai-toolkit config")
print(AI_TOOLKIT_CONFIG)`,id:"code-train-flux"}),e.jsx(t,{title:"flux_lora_inference.py",code:`import torch
from diffusers import FluxPipeline

def generate_with_flux_lora(lora_path, prompt, num_images=4):
    """Generate images using a Flux model with LoRA adapter."""
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    pipe.load_lora_weights(lora_path)

    images = pipe(
        prompt,
        num_images_per_prompt=num_images,
        num_inference_steps=28,
        guidance_scale=3.5,
        height=1024,
        width=1024,
    ).images

    for i, img in enumerate(images):
        img.save(f"flux_lora_{i}.png")

    return images

# For lower VRAM: use quantized Flux
def generate_flux_quantized(lora_path, prompt):
    """Run Flux with 4-bit quantization for lower VRAM."""
    from diffusers import BitsAndBytesConfig

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
    )
    pipe.load_lora_weights(lora_path)

    image = pipe(prompt, num_inference_steps=28, guidance_scale=3.5).images[0]
    return image

images = generate_with_flux_lora(
    "./flux-lora-output",
    "a photo of sks person riding a motorcycle through Tokyo at night"
)`,id:"code-flux-inference"}),e.jsx(a,{type:"tip",title:"Flux Guidance Scale",content:"Flux uses much lower guidance scales than SDXL. Start with guidance_scale=3.5 for Flux.1-dev. Going above 5.0 often produces oversaturated, artifact-heavy images. Flux.1-schnell uses no guidance (guidance_scale=0) by design.",id:"note-flux-guidance"}),e.jsx(i,{title:"Flux VRAM Requirements",content:"Flux.1-dev at bf16 needs ~24GB VRAM just for inference. For training with LoRA and gradient checkpointing, expect 20-32GB depending on resolution and batch size. Use fp8 or NF4 quantization if your GPU has less than 24GB.",id:"warning-flux-vram"}),e.jsx(a,{type:"note",title:"Flux vs SDXL LoRA",content:"Flux LoRAs tend to generalize better than SDXL LoRAs because the transformer architecture has more uniform attention patterns. However, Flux LoRA training is 2-3x slower per step and the adapter files are larger (100-400MB vs 50-200MB for SDXL).",id:"note-flux-vs-sdxl"})]})}const et=Object.freeze(Object.defineProperty({__proto__:null,default:J},Symbol.toStringTag,{value:"Module"}));function Z(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Textual Inversion & Embeddings"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Textual Inversion learns a new text embedding to represent a concept without modifying any model weights. It is the lightest fine-tuning method, producing tiny embedding files (a few KB) that can be dropped into any compatible model. The trade-off is lower fidelity compared to LoRA or DreamBooth."}),e.jsx(o,{title:"Textual Inversion",definition:"Textual Inversion optimizes a new token embedding $v^*$ in the text encoder's embedding space to represent a target concept. Only the embedding vector is trained while all model weights remain frozen: $v^* = \\arg\\min_{v} \\mathbb{E}_{x, \\epsilon, t}[\\|\\epsilon - \\epsilon_\\theta(x_t, t, c_\\theta(v))\\|^2]$. The resulting embedding is a single vector of dimension $d = 768$ (SD 1.5) or $d = 1280$ (SDXL).",id:"def-textual-inversion"}),e.jsx(n,{title:"When to Use Textual Inversion vs LoRA",problem:"How do Textual Inversion and LoRA compare?",steps:[{formula:"\\text{TI: 4-100 KB file size}",explanation:"Extremely small. Can share hundreds of concepts without storage concerns."},{formula:"\\text{LoRA: 10-400 MB file size}",explanation:"Larger but captures much more detail about the concept."},{formula:"\\text{TI: no model weight changes}",explanation:"Works across any checkpoint using the same text encoder. Maximum compatibility."},{formula:"\\text{LoRA: modifies UNet attention}",explanation:"Tied to the specific model architecture. Better fidelity for complex concepts."},{formula:"\\text{TI: good for styles and simple concepts}",explanation:"Works well for art styles, textures, and simple objects. Struggles with faces."}],id:"example-ti-vs-lora"}),e.jsx(t,{title:"train_textual_inversion.py",code:`# Textual Inversion training with diffusers
# Install: pip install diffusers[training] accelerate

TRAIN_CMD = """
accelerate launch diffusers/examples/textual_inversion/textual_inversion.py \\
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \\
    --train_data_dir="./data/my_concept" \\
    --learnable_property="style" \\
    --placeholder_token="<my-style>" \\
    --initializer_token="painting" \\
    --resolution=512 \\
    --train_batch_size=1 \\
    --gradient_accumulation_steps=4 \\
    --max_train_steps=3000 \\
    --learning_rate=5e-4 \\
    --lr_scheduler="constant" \\
    --lr_warmup_steps=0 \\
    --output_dir="./textual-inversion-output" \\
    --save_steps=500 \\
    --mixed_precision="bf16"
"""

# Python API for initialization
from diffusers import StableDiffusionPipeline
import torch

def setup_textual_inversion(model_name, placeholder_token, init_token):
    """Initialize a new token for textual inversion."""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    num_added = tokenizer.add_tokens(placeholder_token)
    print(f"Added {num_added} token(s): '{placeholder_token}'")

    text_encoder.resize_token_embeddings(len(tokenizer))

    token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    init_id = tokenizer.convert_tokens_to_ids(init_token)

    embeds = text_encoder.get_input_embeddings().weight.data
    embeds[token_id] = embeds[init_id].clone()
    print(f"Initialized '{placeholder_token}' from '{init_token}'")

    return pipe, token_id

pipe, token_id = setup_textual_inversion(
    "stabilityai/stable-diffusion-xl-base-1.0",
    "<watercolor-sketch>",
    "watercolor"
)
print(TRAIN_CMD)`,id:"code-train-ti"}),e.jsx(t,{title:"use_textual_inversion.py",code:`import torch
from diffusers import StableDiffusionXLPipeline

def load_textual_inversion(model_name, embedding_path, token_name):
    """Load a textual inversion embedding and generate images."""
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to("cuda")

    pipe.load_textual_inversion(embedding_path, token=token_name)

    prompts = [
        f"a landscape in {token_name}",
        f"a portrait of a cat in {token_name}",
        f"a cityscape at sunset in {token_name}",
    ]

    images = []
    for prompt in prompts:
        image = pipe(
            prompt, num_inference_steps=30, guidance_scale=7.5,
        ).images[0]
        images.append(image)
        safe_name = prompt[:30].replace(" ", "_")
        image.save(f"ti_{safe_name}.png")

    return images

# Combine multiple embeddings
def combine_embeddings(model_name, embeddings):
    """Load multiple TI embeddings into one pipeline."""
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to("cuda")

    for path, token in embeddings:
        pipe.load_textual_inversion(path, token=token)

    image = pipe(
        "a <my-style> portrait with <my-lighting> effects",
        num_inference_steps=30,
    ).images[0]
    return image

load_textual_inversion(
    "stabilityai/stable-diffusion-xl-base-1.0",
    "./textual-inversion-output/learned_embeds.safetensors",
    "<my-style>"
)`,id:"code-use-ti"}),e.jsx(a,{type:"tip",title:"Initialize from a Related Token",content:"Always initialize the new embedding from a semantically related token. For a style, use 'painting' or 'art'. For a dog breed, use 'dog'. This gives the optimization a much better starting point and typically converges in fewer steps.",id:"note-init-token"}),e.jsx(i,{title:"Textual Inversion Limitations",content:"Textual Inversion can only capture what is expressible in the text embedding space. It cannot learn new visual patterns that the model has never seen. Complex subjects like specific faces or intricate patterns will not be captured well. Use LoRA or DreamBooth for high-fidelity subject preservation.",id:"warning-ti-limits"}),e.jsx(a,{type:"note",title:"Multi-Vector Embeddings",content:"Some implementations allow learning multiple embedding vectors per concept (e.g., 4-8 vectors instead of 1). This increases capacity at the cost of using more of the prompt token budget. Specify --num_vectors=4 in the training script to enable this.",id:"note-multi-vector"})]})}const tt=Object.freeze(Object.defineProperty({__proto__:null,default:Z},Symbol.toStringTag,{value:"Module"}));function ee(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Training Dataset Preparation"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Dataset quality is the single most important factor in diffusion model fine-tuning. A well-curated dataset of 10-20 images with precise captions will outperform a sloppy dataset of hundreds. This section covers the full pipeline from image collection through captioning and formatting."}),e.jsx(o,{title:"Training Dataset Structure",definition:"A diffusion training dataset consists of image-caption pairs. Each image should be high quality, properly cropped, and at or above the training resolution. Captions must accurately describe the image content and include the trigger word for the concept being learned.",id:"def-dataset-structure"}),e.jsx(n,{title:"Image Selection Guidelines",problem:"How should you select and prepare training images?",steps:[{formula:"\\text{Variety: diverse angles, lighting, backgrounds}",explanation:"More variety helps the model generalize. Avoid repetitive poses or settings."},{formula:"\\text{Quality: sharp, well-lit, high resolution}",explanation:"Minimum resolution should match training resolution (512 or 1024). No blur or artifacts."},{formula:"\\text{Quantity: 5-10 (DreamBooth), 20-100 (style LoRA)}",explanation:"More is not always better. Quality and variety matter more than raw count."},{formula:"\\text{Consistency: same subject or consistent style}",explanation:"All images should clearly represent the target concept."}],id:"example-image-guidelines"}),e.jsx(t,{title:"prepare_dataset.py",code:`import os
from pathlib import Path
from PIL import Image
import json

def prepare_image_dataset(input_dir, output_dir, resolution=1024):
    """Prepare images for diffusion model training."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [
        f for f in Path(input_dir).iterdir()
        if f.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images")

    for img_path in sorted(image_files):
        img = Image.open(img_path).convert("RGB")

        # Resize maintaining aspect ratio, then center crop
        w, h = img.size
        scale = resolution / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Center crop to square
        left = (new_w - resolution) // 2
        top = (new_h - resolution) // 2
        img = img.crop((left, top, left + resolution, top + resolution))

        out_name = f"{img_path.stem}.png"
        out_path = os.path.join(output_dir, out_name)
        img.save(out_path, "PNG")

        metadata.append({
            "file_name": out_name,
            "text": "",  # filled by captioning step
            "original": str(img_path),
        })

    with open(os.path.join(output_dir, "metadata.jsonl"), "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\\n")

    print(f"Processed {len(metadata)} images to {output_dir}")
    return metadata

prepare_image_dataset("./raw_images", "./training_dataset", resolution=1024)`,id:"code-prepare"}),e.jsx(t,{title:"auto_caption.py",code:`import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os, json

def auto_caption_dataset(image_dir, trigger_word="sks", concept_class="person"):
    """Auto-caption training images using a vision-language model."""
    model_id = "microsoft/Florence-2-large"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    captions = {}
    image_files = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg"))]

    for fname in sorted(image_files):
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")

        inputs = processor(
            text="<MORE_DETAILED_CAPTION>",
            images=img,
            return_tensors="pt"
        ).to("cuda", torch.float16)

        generated_ids = model.generate(**inputs, max_new_tokens=256, num_beams=3)
        caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Insert trigger word
        caption = caption.replace(
            f"a {concept_class}", f"a {trigger_word} {concept_class}"
        )
        if trigger_word not in caption:
            caption = f"a photo of {trigger_word} {concept_class}, {caption}"

        captions[fname] = caption
        print(f"{fname}: {caption[:100]}...")

        # Save as .txt sidecar file
        txt_path = os.path.join(image_dir, fname.rsplit(".", 1)[0] + ".txt")
        with open(txt_path, "w") as f:
            f.write(caption)

    # Also save as metadata.jsonl for diffusers
    with open(os.path.join(image_dir, "metadata.jsonl"), "w") as f:
        for fname, caption in captions.items():
            f.write(json.dumps({"file_name": fname, "text": caption}) + "\\n")

    print(f"\\nCaptioned {len(captions)} images")
    return captions

auto_caption_dataset("./training_dataset", trigger_word="sks", concept_class="dog")`,id:"code-caption"}),e.jsx(a,{type:"tip",title:"Always Review Auto-Captions",content:"Auto-generated captions are a starting point, not the final product. Manually review every caption and correct errors. Remove hallucinated details, fix trigger word placement, and ensure consistency across the dataset. 15 minutes of caption editing saves hours of bad training.",id:"note-review-captions"}),e.jsx(i,{title:"Watermarks and Artifacts",content:"Training images with watermarks, text overlays, or compression artifacts will teach the model to reproduce those artifacts. Always clean your images: remove watermarks, crop out borders, and use high-quality source files.",id:"warning-artifacts"}),e.jsx(a,{type:"note",title:"Aspect Ratio Bucketing",content:"Modern training scripts support aspect ratio bucketing, which groups images by aspect ratio instead of forcing everything to a square crop. This preserves more of the original composition. Enable it with --resolution=1024 --center_crop=False --random_flip in the training script.",id:"note-bucketing"})]})}const at=Object.freeze(Object.defineProperty({__proto__:null,default:ee},Symbol.toStringTag,{value:"Module"}));function te(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"AnimateDiff LoRA Training"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"AnimateDiff adds temporal motion modules to Stable Diffusion to generate short video clips. Training LoRA adapters for AnimateDiff lets you create custom motion styles and domain-specific animations without full model training. This section covers the practical workflow for training motion LoRAs."}),e.jsx(o,{title:"AnimateDiff",definition:"AnimateDiff inserts temporal attention layers (motion modules) into a pre-trained Stable Diffusion UNet. These modules learn to generate temporally coherent frames. A motion LoRA fine-tunes these temporal layers to learn specific motion patterns while keeping spatial generation frozen.",id:"def-animatediff"}),e.jsx(n,{title:"AnimateDiff LoRA Training Setup",problem:"What data and configuration do you need for motion LoRA training?",steps:[{formula:"\\text{Training data: 50-200 short video clips (2-4 seconds)}",explanation:"Clips should show the target motion pattern. Extract at 8 fps for 16-frame sequences."},{formula:"\\text{Resolution: 256 or 512 pixels}",explanation:"Lower resolution is common for video training due to the memory multiplier per frame."},{formula:"\\text{Frames per sample: 16}",explanation:"Standard AnimateDiff uses 16 frames. This is the temporal window size."},{formula:"\\text{VRAM: 24-40 GB depending on resolution}",explanation:"Video training multiplies memory by the number of frames. Use gradient checkpointing."}],id:"example-animatediff-setup"}),e.jsx(t,{title:"prepare_video_dataset.py",code:`import os
import subprocess
from pathlib import Path

def extract_training_clips(video_path, output_dir, fps=8, num_frames=16,
                           resolution=512):
    """Extract training clips from a video file."""
    os.makedirs(output_dir, exist_ok=True)
    clip_idx = 0

    # Get video duration
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True
    )
    duration = float(result.stdout.strip())
    clip_duration = num_frames / fps

    for start in range(0, int(duration - clip_duration), int(clip_duration // 2)):
        clip_dir = os.path.join(output_dir, f"clip_{clip_idx:04d}")
        os.makedirs(clip_dir, exist_ok=True)

        subprocess.run([
            "ffmpeg", "-y", "-ss", str(start),
            "-i", video_path,
            "-t", str(clip_duration),
            "-vf", f"fps={fps},scale={resolution}:{resolution}:"
                   f"force_original_aspect_ratio=decrease,"
                   f"pad={resolution}:{resolution}:-1:-1",
            "-frames:v", str(num_frames),
            os.path.join(clip_dir, "frame_%04d.png")
        ], capture_output=True)

        frames = list(Path(clip_dir).glob("*.png"))
        if len(frames) == num_frames:
            clip_idx += 1
        else:
            import shutil
            shutil.rmtree(clip_dir)

    print(f"Extracted {clip_idx} training clips from {video_path}")
    return clip_idx

video_dir = "./raw_videos"
output_dir = "./animatediff_training_data"
total_clips = 0

for video in Path(video_dir).glob("*.mp4"):
    clips = extract_training_clips(str(video), output_dir, fps=8, num_frames=16)
    total_clips += clips

print(f"Total training clips: {total_clips}")`,id:"code-prepare-video"}),e.jsx(t,{title:"train_animatediff_lora.py",code:`# AnimateDiff LoRA training using diffusers

TRAIN_CMD = """
accelerate launch diffusers/examples/animatediff/train_animatediff.py \\
    --pretrained_model_name_or_path="SG161222/Realistic_Vision_V5.1_noVAE" \\
    --motion_module="guoyww/animatediff-motion-adapter-v1-5-3" \\
    --train_data_dir="./animatediff_training_data" \\
    --output_dir="./animatediff-lora-output" \\
    --resolution=512 \\
    --train_batch_size=1 \\
    --gradient_accumulation_steps=4 \\
    --max_train_steps=2000 \\
    --learning_rate=1e-4 \\
    --lr_scheduler="cosine" \\
    --lr_warmup_steps=100 \\
    --rank=32 \\
    --seed=42 \\
    --mixed_precision="bf16" \\
    --gradient_checkpointing \\
    --enable_xformers_memory_efficient_attention \\
    --num_frames=16 \\
    --checkpointing_steps=500
"""

# Inference with trained motion LoRA
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

def generate_animation(motion_lora_path, prompt, num_frames=16):
    """Generate animation using trained motion LoRA."""
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3",
        torch_dtype=torch.float16,
    )

    pipe = AnimateDiffPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    ).to("cuda")

    pipe.load_lora_weights(motion_lora_path, adapter_name="motion_lora")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    output = pipe(
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=25,
        guidance_scale=7.5,
    )

    export_to_gif(output.frames[0], "animation_output.gif")
    print("Saved animation to animation_output.gif")

print(TRAIN_CMD)`,id:"code-train-animatediff"}),e.jsx(a,{type:"tip",title:"Start with Short Low-Res Clips",content:"Begin training at 256x256 resolution with 16 frames to iterate quickly. Once you have good motion patterns, scale up to 512x512. The motion patterns learned at lower resolution transfer well to higher resolution inference.",id:"note-start-small"}),e.jsx(i,{title:"Temporal Flickering",content:"The most common AnimateDiff failure mode is temporal flickering where frames are individually good but inconsistent with each other. This is usually caused by too few training clips, too high a learning rate, or insufficient training steps. Reduce LR to 5e-5 and ensure at least 50 diverse clips.",id:"warning-flickering"}),e.jsx(a,{type:"note",title:"Motion Module Compatibility",content:"AnimateDiff motion LoRAs are tied to a specific motion module version. A LoRA trained with motion-adapter-v1-5-3 may not work with v1-5-2. Always document which motion module version was used during training.",id:"note-compatibility"})]})}const ot=Object.freeze(Object.defineProperty({__proto__:null,default:te},Symbol.toStringTag,{value:"Module"})),ae=`# CogVideoX Fine-tuning Setup
# Requires: diffusers >= 0.30.0, accelerate, peft, decord
pip install diffusers accelerate peft transformers
pip install decord imageio[ffmpeg]

# CogVideoX-2b requires ~40GB VRAM for fine-tuning
# CogVideoX-5b requires ~80GB VRAM (multi-GPU recommended)

# Clone the training scripts
# git clone https://github.com/THUDM/CogVideo.git
# cd CogVideo/finetune`,oe=`# CogVideoX LoRA Fine-tuning with diffusers
from diffusers import CogVideoXPipeline
from peft import LoraConfig, get_peft_model
import torch

# 1. Load the base model
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.bfloat16,
)

# 2. Configure LoRA for the transformer
lora_config = LoraConfig(
    r=64,
    lora_alpha=64,
    target_modules=[
        "to_q", "to_k", "to_v", "to_out.0",  # attention
        "proj_in", "proj_out",  # projections
    ],
    lora_dropout=0.0,
)

# 3. Dataset preparation
# Organize videos as:
# dataset/
#   videos/
#     001.mp4  # 6 seconds, 480x720 or 720x480
#     002.mp4
#   metadata.json  # [{"file": "001.mp4", "text": "description"}]

# 4. Training command with accelerate
# accelerate launch train_cogvideox_lora.py \\
#   --pretrained_model_name_or_path="THUDM/CogVideoX-2b" \\
#   --data_root="./dataset" \\
#   --output_dir="./cogvideox-lora" \\
#   --height=480 --width=720 \\
#   --num_frames=49 --fps=8 \\
#   --train_batch_size=1 \\
#   --gradient_accumulation_steps=4 \\
#   --learning_rate=1e-4 \\
#   --lr_scheduler="cosine" \\
#   --max_train_steps=1000 \\
#   --lora_rank=64 \\
#   --mixed_precision="bf16" \\
#   --gradient_checkpointing`,ie=`# Generate video with fine-tuned CogVideoX LoRA
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import torch

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Load fine-tuned LoRA
pipe.load_lora_weights("./cogvideox-lora")

video_frames = pipe(
    prompt="A golden retriever playing in autumn leaves",
    num_frames=49,
    guidance_scale=6.0,
    num_inference_steps=50,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]

export_to_video(video_frames, "output.mp4", fps=8)`;function ne(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsxs("div",{children:[e.jsx("h1",{className:"text-3xl font-extrabold tracking-tight text-gray-900 dark:text-white",children:"Fine-tuning CogVideoX"}),e.jsx("p",{className:"mt-3 text-lg text-gray-600 dark:text-gray-400",children:"CogVideoX is an open-source video generation model using a 3D causal VAE and expert transformer blocks. Fine-tuning with LoRA lets you specialize it for specific visual styles or motion patterns."})]}),e.jsx(o,{title:"CogVideoX Architecture",definition:"CogVideoX uses a 3D causal VAE that compresses video spatiotemporally (4× temporal, 8× spatial), followed by an expert transformer with full 3D attention across all frames. The text encoder is T5-XXL.",notation:"Input video $V \\in \\mathbb{R}^{T \\times H \\times W \\times 3}$ is encoded to latent $z \\in \\mathbb{R}^{T/4 \\times H/8 \\times W/8 \\times C}$"}),e.jsx(t,{code:ae,title:"Terminal — Setup"}),e.jsx(a,{type:"note",title:"Dataset Guidelines",children:e.jsxs("p",{children:["Prepare ",e.jsx("strong",{children:"50-200 video clips"}),", each 6 seconds at 8 fps (49 frames). Resolution should be 480×720 or 720×480 (landscape/portrait). Include diverse scenes with consistent style. Each video needs a text caption describing the content and motion."]})}),e.jsx(t,{code:oe,title:"train_cogvideox_lora.py"}),e.jsx(i,{title:"Compute Requirements",children:e.jsxs("p",{children:["CogVideoX-2b LoRA fine-tuning needs ",e.jsx("strong",{children:"~40GB VRAM"})," with gradient checkpointing and bf16. For CogVideoX-5b, use multi-GPU with DeepSpeed ZeRO Stage 2. Training typically takes 4-8 hours on a single A100 for 1000 steps."]})}),e.jsx(t,{code:ie,title:"generate_video.py"}),e.jsx(a,{type:"tip",title:"Quality Tips",children:e.jsx("p",{children:"Start with a small learning rate (1e-4) and monitor validation videos every 200 steps. CogVideoX is sensitive to caption quality — use detailed, specific descriptions. If motion becomes jittery, reduce the learning rate or increase LoRA rank."})})]})}const it=Object.freeze(Object.defineProperty({__proto__:null,default:ne},Symbol.toStringTag,{value:"Module"}));function re(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Hyperparameter Tuning & Failures"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Diffusion model fine-tuning is sensitive to hyperparameters. Small changes in learning rate, rank, or training steps can mean the difference between a great result and complete failure. This section provides systematic tuning strategies and a catalog of common failures with their fixes."}),e.jsx(o,{title:"Key Hyperparameters",definition:"The primary hyperparameters for diffusion LoRA training are: learning rate $\\eta$ (typically $10^{-5}$ to $5 \\times 10^{-4}$), LoRA rank $r$ (4 to 128), training steps $T$ (200 to 5000), and the LoRA alpha scaling $\\alpha$ (usually set equal to $r$). The effective learning rate for the LoRA update is $\\eta \\cdot \\frac{\\alpha}{r}$.",id:"def-hyperparams"}),e.jsx(n,{title:"Failure Diagnosis Guide",problem:"How do you identify and fix common diffusion fine-tuning failures?",steps:[{formula:"\\text{Blurry outputs} \\Rightarrow \\text{LR too high or too many steps}",explanation:"The model has overfit and lost detail. Reduce learning rate by 2-5x or train fewer steps."},{formula:"\\text{No effect (identical to base)} \\Rightarrow \\text{LR too low or rank too small}",explanation:"The adapter has not learned enough. Increase LR, rank, or training steps."},{formula:"\\text{Color artifacts / distortion} \\Rightarrow \\text{VAE or dtype issue}",explanation:"Ensure VAE is in fp32 for SD 1.5, or use bf16 (not fp16) for SDXL/Flux."},{formula:"\\text{Concept bleeding (wrong subjects)} \\Rightarrow \\text{Bad captions}",explanation:"Captions do not properly isolate the target concept. Fix trigger words and descriptions."},{formula:"\\text{Training images reproduced exactly} \\Rightarrow \\text{Overfit}",explanation:"Too many steps on too few images. Reduce steps or add more training images."}],id:"example-failures"}),e.jsx(t,{title:"hyperparameter_sweep.py",code:`import subprocess
import itertools
import json
import os
from datetime import datetime

def run_lora_sweep(base_config, param_grid):
    """Run a hyperparameter sweep for diffusion LoRA training."""
    results = []

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"Total configurations: {len(combinations)}")

    for i, combo in enumerate(combinations):
        config = {**base_config}
        for key, val in zip(keys, combo):
            config[key] = val

        run_name = (
            f"sweep_{i:03d}_lr{config['lr']}_r{config['rank']}_s{config['steps']}"
        )
        config["output_dir"] = f"./sweep_results/{run_name}"
        os.makedirs(config["output_dir"], exist_ok=True)

        print(f"\\nRun {i+1}/{len(combinations)}: {run_name}")

        cmd = [
            "accelerate", "launch",
            "diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py",
            f"--pretrained_model_name_or_path={config['model']}",
            f"--dataset_name={config['dataset']}",
            f"--output_dir={config['output_dir']}",
            f"--resolution={config['resolution']}",
            f"--learning_rate={config['lr']}",
            f"--rank={config['rank']}",
            f"--max_train_steps={config['steps']}",
            "--train_batch_size=1",
            "--gradient_checkpointing",
            "--mixed_precision=bf16",
            "--seed=42",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        config["success"] = result.returncode == 0
        config["timestamp"] = datetime.now().isoformat()
        results.append(config)

        with open("./sweep_results/sweep_log.json", "w") as f:
            json.dump(results, f, indent=2)

    return results

base_config = {
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "dataset": "./my_dataset",
    "resolution": 1024,
}

param_grid = {
    "lr": [5e-5, 1e-4, 3e-4],
    "rank": [8, 16, 32],
    "steps": [500, 1000, 2000],
}

# This runs 27 configurations
# run_lora_sweep(base_config, param_grid)`,id:"code-sweep"}),e.jsx(t,{title:"evaluate_lora_quality.py",code:`import torch
from diffusers import DiffusionPipeline
from pathlib import Path

def evaluate_lora_checkpoints(base_model, lora_dir, eval_prompts):
    """Generate images from each checkpoint for visual comparison."""
    pipe = DiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16
    ).to("cuda")

    checkpoints = sorted(Path(lora_dir).glob("checkpoint-*"))
    print(f"Found {len(checkpoints)} checkpoints")

    for ckpt in checkpoints:
        step = ckpt.name.split("-")[1]
        print(f"\\nGenerating from checkpoint step {step}...")

        pipe.load_lora_weights(str(ckpt))

        for j, prompt in enumerate(eval_prompts):
            image = pipe(
                prompt, num_inference_steps=28, guidance_scale=7.5,
                generator=torch.Generator("cuda").manual_seed(42),
            ).images[0]
            image.save(f"eval_step{step}_prompt{j}.png")

        pipe.unload_lora_weights()

    print("\\nCompare images across steps to find optimal checkpoint.")

# CLIP score for quantitative evaluation
def clip_score(images, prompts):
    """Compute CLIP similarity between images and prompts."""
    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    scores = []
    for img, prompt in zip(images, prompts):
        inputs = processor(text=[prompt], images=img, return_tensors="pt")
        outputs = model(**inputs)
        score = outputs.logits_per_image.item()
        scores.append(score)

    avg = sum(scores) / len(scores)
    print(f"Average CLIP score: {avg:.3f}")
    return scores

eval_prompts = [
    "a photo of sks in a forest",
    "sks standing on a beach at sunset",
    "a painting of sks in impressionist style",
]`,id:"code-evaluate"}),e.jsx(a,{type:"intuition",title:"The Learning Rate Sweet Spot",content:"For diffusion LoRA, the learning rate window is narrow. Too low (below 1e-5) and nothing is learned. Too high (above 5e-4) and the model quickly produces artifacts. Start with 1e-4 and adjust in 2x increments. If you see artifacts at 1e-4, try 5e-5. If you see no effect, try 2e-4.",id:"note-lr-sweet-spot"}),e.jsx(i,{title:"Do Not Tune Everything at Once",content:"Change one hyperparameter at a time. If you change learning rate, rank, and training steps simultaneously and get a bad result, you will not know which change caused the problem. Start with the defaults, then adjust learning rate first, then rank, then steps.",id:"warning-one-at-a-time"}),e.jsx(a,{type:"tip",title:"Save Checkpoints Frequently",content:"Save a checkpoint every 200-500 steps. Overfitting in diffusion models happens rapidly, and the best result is often an intermediate checkpoint rather than the final one. Compare outputs across checkpoints to find the sweet spot before quality degrades.",id:"note-save-checkpoints"})]})}const nt=Object.freeze(Object.defineProperty({__proto__:null,default:re},Symbol.toStringTag,{value:"Module"}));export{De as A,Oe as B,Ne as C,Ue as D,Ge as E,Ie as F,Be as G,Ee as H,$e as I,Qe as J,We as K,He as L,Ve as M,Ke as N,Xe as O,Ye as P,Je as Q,Ze as R,et as S,tt as T,at as U,ot as V,it as W,nt as X,me as a,ce as b,ue as c,fe as d,ge as e,he as f,_e as g,xe as h,ye as i,be as j,ve as k,we as l,ke as m,Ae as n,Le as o,Te as p,je as q,Me as r,pe as s,Re as t,Pe as u,ze as v,Se as w,Fe as x,Ce as y,qe as z};
