import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function PythonEnv() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Python Environment Setup</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        A clean, isolated Python environment prevents dependency conflicts that plague ML projects.
        This section covers setting up virtual environments with venv, conda, and managing packages
        with pip for finetuning workflows.
      </p>

      <DefinitionBlock
        title="Virtual Environment"
        definition="A virtual environment is an isolated Python installation with its own packages, independent of the system Python. This prevents version conflicts between projects and ensures reproducibility."
        id="def-venv"
      />

      <h2 className="text-2xl font-semibold">Option 1: venv (Recommended for Simplicity)</h2>
      <PythonCode
        title="setup_venv.sh"
        code={`# Create a virtual environment with Python 3.10+
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
pip freeze > requirements.txt`}
        id="code-venv-setup"
      />

      <h2 className="text-2xl font-semibold">Option 2: Conda (Better for Complex Dependencies)</h2>
      <PythonCode
        title="setup_conda.sh"
        code={`# Install miniconda (if not already installed)
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
conda env create -f environment.yml`}
        id="code-conda-setup"
      />

      <ExampleBlock
        title="Complete Finetuning Environment"
        problem="Set up a complete environment for QLoRA finetuning with unsloth."
        steps={[
          { formula: '\\texttt{python3.11 -m venv ~/qlora-env}', explanation: 'Create isolated environment with Python 3.11.' },
          { formula: '\\texttt{pip install torch --index-url .../cu124}', explanation: 'Install PyTorch with CUDA 12.4 support.' },
          { formula: '\\texttt{pip install unsloth[cu124]}', explanation: 'Install unsloth with all finetuning dependencies.' },
          { formula: '\\texttt{pip install wandb}', explanation: 'Install Weights & Biases for experiment tracking.' },
        ]}
        id="example-complete-env"
      />

      <PythonCode
        title="verify_environment.py"
        code={`# Run this script to verify your finetuning environment
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

print("\\nEnvironment check complete!")`}
        id="code-verify-env"
      />

      <WarningBlock
        title="Python Version Matters"
        content="Use Python 3.10 or 3.11 for finetuning. Python 3.12+ may have compatibility issues with some CUDA libraries and finetuning packages. Always check the documentation of your chosen finetuning framework for supported Python versions."
        id="warning-python-version"
      />

      <NoteBlock
        type="tip"
        title="Using requirements.txt for Reproducibility"
        content="Always pin exact versions in requirements.txt for production environments. Use pip freeze > requirements.txt after confirming everything works. Share this file with collaborators to ensure identical environments."
        id="note-requirements"
      />
    </div>
  )
}
