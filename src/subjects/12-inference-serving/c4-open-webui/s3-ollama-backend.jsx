import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function OllamaBackend() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Connecting to Ollama Backend</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The most common setup pairs Open WebUI with a local Ollama instance. The connection is
        straightforward but requires understanding how Docker networking works to avoid common
        pitfalls.
      </p>

      <DefinitionBlock
        title="Ollama Backend Connection"
        definition="Open WebUI communicates with Ollama via its REST API. The OLLAMA_BASE_URL environment variable tells Open WebUI where to find Ollama. From inside a Docker container, 'localhost' refers to the container itself, not the host machine."
        id="def-connection"
      />

      <PythonCode
        title="Terminal"
        code={`# Scenario 1: Open WebUI in Docker, Ollama on host
# Use --add-host to make host.docker.internal resolve to the host
docker run -d -p 3000:8080 \\
    --add-host=host.docker.internal:host-gateway \\
    -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Scenario 2: Both in Docker Compose (recommended)
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

docker compose up -d

# Scenario 3: pip install (both on same host)
# No special networking needed -- localhost works
OLLAMA_BASE_URL=http://localhost:11434 open-webui serve`}
        id="code-connection"
      />

      <PythonCode
        title="troubleshoot_connection.py"
        code={`import requests

OLLAMA_URLS = [
    "http://localhost:11434",
    "http://host.docker.internal:11434",
    "http://ollama:11434",
    "http://127.0.0.1:11434",
]

print("Testing Ollama connectivity:")
for url in OLLAMA_URLS:
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            print(f"  {url} -> OK ({resp.text.strip()})")
        else:
            print(f"  {url} -> HTTP {resp.status_code}")
    except requests.ConnectionError:
        print(f"  {url} -> Connection refused")
    except requests.Timeout:
        print(f"  {url} -> Timeout")

# Also verify models are available
working_url = "http://localhost:11434"
try:
    resp = requests.get(f"{working_url}/api/tags")
    models = resp.json().get("models", [])
    print(f"\\nModels available: {len(models)}")
    for m in models:
        print(f"  - {m['name']} ({m['size']/(1024**3):.1f} GB)")
except Exception as e:
    print(f"\\nCannot list models: {e}")`}
        id="code-troubleshoot"
      />

      <ExampleBlock
        title="Common Connection Issues"
        problem="Open WebUI cannot see Ollama models. How to debug?"
        steps={[
          { formula: 'Check Ollama is running: curl http://localhost:11434/', explanation: 'Should return "Ollama is running". If not, start Ollama.' },
          { formula: 'Docker networking: use host.docker.internal, not localhost', explanation: 'Inside a container, localhost is the container itself.' },
          { formula: 'Firewall: Ollama binds to 127.0.0.1 by default', explanation: 'Set OLLAMA_HOST=0.0.0.0 if Ollama needs to accept remote connections.' },
          { formula: 'Check Open WebUI Settings > Connections', explanation: 'The URL can be changed from the admin UI after login.' },
        ]}
        id="example-troubleshoot"
      />

      <NoteBlock
        type="tip"
        title="Pull Models from the UI"
        content="Open WebUI can pull Ollama models directly from the interface. Go to the model selector dropdown and type a model name to pull it. This is convenient but can be disabled by admins who want to control which models are available."
        id="note-pull-from-ui"
      />

      <WarningBlock
        title="Ollama Must Be Running First"
        content="If Ollama is not running when Open WebUI starts, the connection will fail and no models will appear. In Docker Compose, use depends_on to ensure Ollama starts first. For manual setups, start Ollama before Open WebUI."
        id="warning-order"
      />
    </div>
  )
}
