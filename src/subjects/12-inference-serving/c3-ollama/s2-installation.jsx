import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Installation() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Installing Ollama</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Ollama runs on macOS, Linux, and Windows. Installation is straightforward on all platforms,
        with Docker as an additional option for containerized deployments.
      </p>

      <PythonCode
        title="Terminal — macOS"
        code={`# Option 1: Download from ollama.com (recommended)
# Visit https://ollama.com/download and download the macOS app

# Option 2: Homebrew
brew install ollama

# Start the Ollama service
ollama serve
# Or the macOS app runs it as a menu bar service automatically

# Verify installation
ollama --version`}
        id="code-macos"
      />

      <PythonCode
        title="Terminal — Linux"
        code={`# One-line install script (recommended)
curl -fsSL https://ollama.com/install.sh | sh

# This installs Ollama and sets up a systemd service
# The service starts automatically

# Check the service status
sudo systemctl status ollama

# View logs
journalctl -u ollama -f

# Manual start if needed
ollama serve`}
        id="code-linux"
      />

      <PythonCode
        title="Terminal — Docker"
        code={`# CPU only
docker run -d -v ollama:/root/.ollama -p 11434:11434 \\
    --name ollama ollama/ollama

# With NVIDIA GPU support
docker run -d --gpus all -v ollama:/root/.ollama -p 11434:11434 \\
    --name ollama ollama/ollama

# With AMD GPU support (ROCm)
docker run -d --device /dev/kfd --device /dev/dri \\
    -v ollama:/root/.ollama -p 11434:11434 \\
    --name ollama ollama/ollama:rocm

# Run a model inside the container
docker exec -it ollama ollama run llama3.2

# Docker Compose example
cat > docker-compose.yml << 'EOF'
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
volumes:
  ollama_data:
EOF

docker compose up -d`}
        id="code-docker"
      />

      <NoteBlock
        type="tip"
        title="Windows Installation"
        content="Download the installer from ollama.com/download. Ollama runs as a system tray application on Windows. It supports NVIDIA GPUs natively and AMD GPUs via ROCm. WSL2 is not required -- Ollama runs as a native Windows application."
        id="note-windows"
      />

      <PythonCode
        title="Terminal — Post-Installation Verification"
        code={`# Verify Ollama is running
curl http://localhost:11434/
# Ollama is running

# Check available endpoints
curl http://localhost:11434/api/tags
# {"models": []}  (empty until you pull a model)

# Pull a small model to test
ollama pull llama3.2:1b

# Run a quick test
ollama run llama3.2:1b "Say hello in 3 words"

# Check GPU detection
ollama ps
# Shows which processor (CPU/GPU) is being used`}
        id="code-verify"
      />

      <WarningBlock
        title="Firewall and Port Configuration"
        content="By default, Ollama only listens on localhost (127.0.0.1:11434). To allow remote access, set OLLAMA_HOST=0.0.0.0 in the environment. Be careful exposing Ollama to the network -- it has no built-in authentication."
        id="warning-firewall"
      />

      <NoteBlock
        type="note"
        title="Storage Location"
        content="Models are stored in ~/.ollama/models on macOS/Linux and C:\\Users\\<user>\\.ollama\\models on Windows. A typical 7B model requires 4-8GB of disk space depending on quantization. Set OLLAMA_MODELS to change the storage directory."
        id="note-storage"
      />
    </div>
  )
}
