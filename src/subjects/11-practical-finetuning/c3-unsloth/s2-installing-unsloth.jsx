import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function InstallingUnsloth() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Installing Unsloth</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Unsloth installation requires matching the correct version to your CUDA toolkit and PyTorch
        build. This section covers installation for local machines, cloud instances, and Google Colab.
      </p>

      <h2 className="text-2xl font-semibold">Installation Methods</h2>

      <PythonCode
        title="install_unsloth.sh"
        code={`# Method 1: pip install (recommended for most users)
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
# Also installs xformers for additional speed on Colab GPUs`}
        id="code-install"
      />

      <h2 className="text-2xl font-semibold">Verifying Installation</h2>

      <PythonCode
        title="verify_unsloth.py"
        code={`# Verify Unsloth installation
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
print("Installation verified!")`}
        id="code-verify"
      />

      <ExampleBlock
        title="Troubleshooting Common Installation Issues"
        problem="How to fix the most common Unsloth installation errors?"
        steps={[
          { formula: '\\texttt{ImportError: triton}', explanation: 'Install triton: pip install triton. Required for custom kernels.' },
          { formula: '\\texttt{bitsandbytes CUDA error}', explanation: 'Ensure CUDA version matches: pip install bitsandbytes --force-reinstall.' },
          { formula: '\\texttt{xformers compatibility}', explanation: 'Unsloth bundles its own kernels; xformers is optional. Remove if conflicting.' },
          { formula: '\\texttt{CUDA out of memory on import}', explanation: 'Another process may be using the GPU. Run nvidia-smi to check and kill stale processes.' },
        ]}
        id="example-troubleshooting"
      />

      <NoteBlock
        type="tip"
        title="Pre-built Docker Images"
        content="For reproducible environments, use the Unsloth Docker image: docker pull unsloth/unsloth. This includes all dependencies pre-configured. Alternatively, RunPod and Lambda Labs offer Unsloth as a pre-installed template."
        id="note-docker"
      />

      <WarningBlock
        title="Version Pinning"
        content="Unsloth updates frequently. For production workflows, pin the commit hash: pip install 'unsloth @ git+https://github.com/unslothai/unsloth.git@abc123'. This prevents breaking changes from affecting your training pipeline."
        id="warning-version-pin"
      />

      <NoteBlock
        type="note"
        title="Hugging Face Token"
        content="Many models (LLaMA, Mistral) require accepting a license on Hugging Face. Run: huggingface-cli login and enter your token. Or set the environment variable: export HF_TOKEN=hf_xxxxx. Without this, model downloads will fail with a 401 error."
        id="note-hf-token"
      />
    </div>
  )
}
