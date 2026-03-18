import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function CudaSetup() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">CUDA, cuDNN, and Driver Setup</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        A working CUDA stack is the foundation of GPU-accelerated finetuning. Mismatched versions
        between the NVIDIA driver, CUDA toolkit, and cuDNN are the most common source of environment
        issues. This section walks through getting everything configured correctly.
      </p>

      <DefinitionBlock
        title="CUDA Toolkit"
        definition="CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform. The CUDA toolkit includes the compiler (nvcc), runtime libraries, and tools needed to run GPU-accelerated code. PyTorch ships with its own CUDA runtime, so you typically only need matching NVIDIA drivers."
        id="def-cuda"
      />

      <h2 className="text-2xl font-semibold">Checking Your Current Setup</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Before installing anything, check what you already have. The NVIDIA driver version
        determines the maximum CUDA version your system supports.
      </p>

      <PythonCode
        title="check_cuda_environment.py"
        code={`import subprocess, torch

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
    print("BFloat16: NOT supported (use fp16 instead)")`}
        id="code-check-cuda"
      />

      <h2 className="text-2xl font-semibold">Installing NVIDIA Drivers</h2>
      <p className="text-gray-700 dark:text-gray-300">
        On Ubuntu/Debian systems, the recommended approach is to use the official NVIDIA package
        repository. Driver version 535+ is recommended for CUDA 12.x support.
      </p>

      <PythonCode
        title="install_drivers.sh"
        code={`# Ubuntu 22.04/24.04 - Install NVIDIA drivers
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
# Should show driver version >= 535 for CUDA 12.x`}
        id="code-install-drivers"
      />

      <ExampleBlock
        title="CUDA Version Compatibility"
        problem="Determine which PyTorch CUDA version to use given driver version 550.54."
        steps={[
          { formula: '\\text{Driver } 550.x \\Rightarrow \\text{Max CUDA } 12.4', explanation: 'Check the NVIDIA CUDA compatibility table for your driver version.' },
          { formula: '\\text{PyTorch 2.4+} \\Rightarrow \\text{CUDA 12.1 or 12.4}', explanation: 'PyTorch ships its own CUDA runtime; pick the matching build.' },
          { formula: '\\texttt{pip install torch --index-url .../cu124}', explanation: 'Install PyTorch built against CUDA 12.4 for best compatibility.' },
        ]}
        id="example-cuda-compat"
      />

      <NoteBlock
        type="tip"
        title="PyTorch Bundles Its Own CUDA"
        content="You do NOT need to install the CUDA toolkit system-wide for PyTorch. PyTorch pip packages include the necessary CUDA libraries. You only need the NVIDIA driver installed. This simplifies setup enormously - just match the driver version to the PyTorch CUDA build."
        id="note-pytorch-cuda"
      />

      <WarningBlock
        title="Avoid Multiple CUDA Installations"
        content="Having multiple CUDA toolkit versions installed system-wide causes PATH and library conflicts. If you must have system CUDA (e.g., for compiling custom kernels), use environment variables to manage versions: export CUDA_HOME=/usr/local/cuda-12.4 and update PATH accordingly."
        id="warning-multi-cuda"
      />

      <NoteBlock
        type="note"
        title="Flash Attention 2 Compilation"
        content="Flash Attention 2 requires CUDA toolkit headers for compilation. Install with: pip install flash-attn --no-build-isolation. If compilation fails, ensure you have the CUDA toolkit matching your PyTorch version, or use a pre-built wheel from the flash-attn releases page."
        id="note-flash-attn-compile"
      />
    </div>
  )
}
