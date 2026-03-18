import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function OpenWebUIInstallation() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Installation (Docker, pip, Source)</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Open WebUI can be installed via Docker (recommended), pip, or from source. Docker provides
        the most reliable setup with automatic updates, while pip is convenient for development
        environments.
      </p>

      <PythonCode
        title="Terminal — Docker Installation"
        code={`# Standard Docker install (connects to existing Ollama)
docker run -d -p 3000:8080 \\
    --add-host=host.docker.internal:host-gateway \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    --restart always \\
    ghcr.io/open-webui/open-webui:main

# With GPU passthrough (for local embedding models)
docker run -d -p 3000:8080 \\
    --gpus all \\
    --add-host=host.docker.internal:host-gateway \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    --restart always \\
    ghcr.io/open-webui/open-webui:main

# Docker Compose with Ollama included
cat > docker-compose.yml << 'EOF'
services:
  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    volumes:
      - open_webui_data:/app/backend/data
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - ollama

volumes:
  ollama_data:
  open_webui_data:
EOF

docker compose up -d`}
        id="code-docker"
      />

      <PythonCode
        title="Terminal — pip Installation"
        code={`# Install via pip (Python 3.11+ recommended)
pip install open-webui

# Start the server
open-webui serve --port 3000

# Or with custom settings
OLLAMA_BASE_URL=http://localhost:11434 \\
DATA_DIR=./open-webui-data \\
open-webui serve --port 3000 --host 0.0.0.0`}
        id="code-pip"
      />

      <PythonCode
        title="Terminal — Update and Maintenance"
        code={`# Update Docker installation
docker pull ghcr.io/open-webui/open-webui:main
docker stop open-webui
docker rm open-webui
# Re-run the docker run command (data persists in the volume)

# Update pip installation
pip install --upgrade open-webui

# Backup data (Docker)
docker cp open-webui:/app/backend/data ./open-webui-backup

# View logs
docker logs -f open-webui

# Reset admin password
docker exec open-webui open-webui reset-admin-password`}
        id="code-update"
      />

      <ExampleBlock
        title="Post-Installation Checklist"
        problem="What to do after installing Open WebUI?"
        steps={[
          { formula: 'Navigate to http://localhost:3000', explanation: 'Open the web interface in your browser.' },
          { formula: 'Create admin account', explanation: 'The first user to sign up becomes the admin.' },
          { formula: 'Verify Ollama connection', explanation: 'Go to Settings > Connections and confirm Ollama is connected.' },
          { formula: 'Pull a model from the UI or via Ollama CLI', explanation: 'You need at least one model to start chatting.' },
        ]}
        id="example-checklist"
      />

      <NoteBlock
        type="tip"
        title="Docker Volume Persistence"
        content="The -v open-webui:/app/backend/data flag ensures your data (conversations, settings, uploaded files) persists across container restarts and updates. Never use --rm or forget this volume mount, or you will lose all data on restart."
        id="note-persistence"
      />

      <WarningBlock
        title="First User Is Admin"
        content="The very first account created on a fresh Open WebUI installation automatically becomes the administrator. Set up the admin account immediately after deployment to prevent unauthorized users from claiming admin access."
        id="warning-admin"
      />
    </div>
  )
}
