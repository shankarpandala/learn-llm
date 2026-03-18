import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function RemoteServing() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Running Ollama on Remote Servers</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        While Ollama excels at local development, you can also run it on a remote server with a
        powerful GPU and access it from your laptop. This section covers SSH tunneling, reverse
        proxy setup, and secure remote access patterns.
      </p>

      <PythonCode
        title="Terminal — SSH Tunnel (simplest)"
        code={`# On your local machine, create an SSH tunnel to the remote Ollama
ssh -L 11434:localhost:11434 user@gpu-server.example.com

# Now Ollama on the remote server is accessible at localhost:11434
# All existing tools and scripts work without changes
curl http://localhost:11434/api/tags

# For a persistent tunnel (runs in background)
ssh -fNL 11434:localhost:11434 user@gpu-server.example.com

# Kill the tunnel when done
kill $(lsof -ti:11434 | head -1)`}
        id="code-ssh-tunnel"
      />

      <PythonCode
        title="Terminal — Remote Server Setup"
        code={`# On the remote GPU server:

# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Configure to listen on all interfaces
# Edit the systemd service
sudo systemctl edit ollama.service

# Add these lines:
# [Service]
# Environment="OLLAMA_HOST=0.0.0.0"

# Or set the environment variable directly
export OLLAMA_HOST=0.0.0.0
ollama serve

# 3. Pull models
ollama pull llama3.1:8b

# 4. Set up Nginx reverse proxy with basic auth
sudo apt install nginx apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd ollama_user

cat | sudo tee /etc/nginx/sites-available/ollama << 'NGINX'
server {
    listen 443 ssl;
    server_name ollama.example.com;

    ssl_certificate /etc/letsencrypt/live/ollama.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ollama.example.com/privkey.pem;

    auth_basic "Ollama API";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://127.0.0.1:11434;
        proxy_set_header Host $host;
        proxy_buffering off;           # Important for streaming
        proxy_read_timeout 300s;       # Long timeout for generation
    }
}
NGINX

sudo ln -s /etc/nginx/sites-available/ollama /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx`}
        id="code-server-setup"
      />

      <PythonCode
        title="remote_client.py"
        code={`import ollama
import os

# Option 1: Point Ollama client to remote server
# Set environment variable before importing ollama
os.environ["OLLAMA_HOST"] = "http://gpu-server:11434"

response = ollama.generate(model="llama3.1:8b", prompt="Hello from remote!")
print(response["response"])

# Option 2: Use requests with authentication (Nginx setup)
import requests

REMOTE_URL = "https://ollama.example.com"
AUTH = ("ollama_user", "your_password")

resp = requests.post(
    f"{REMOTE_URL}/api/generate",
    json={"model": "llama3.1:8b", "prompt": "Hello!", "stream": False},
    auth=AUTH,
)
print(resp.json()["response"])

# Option 3: Use OpenAI SDK with remote Ollama
from openai import OpenAI

client = OpenAI(
    base_url="https://ollama.example.com/v1",
    api_key="ollama",  # Not used but required by SDK
)

response = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "What GPU are you running on?"}],
)
print(response.choices[0].message.content)`}
        id="code-remote-client"
      />

      <ExampleBlock
        title="Remote Access Methods Compared"
        problem="Which remote access method should you choose?"
        steps={[
          { formula: 'SSH tunnel: simplest, most secure, no config needed', explanation: 'Best for personal use. Requires SSH access to the server.' },
          { formula: 'Reverse proxy + HTTPS: production-grade, shareable', explanation: 'Best for team access. Requires domain name and SSL certificate.' },
          { formula: 'Tailscale/WireGuard VPN: zero-config networking', explanation: 'Best for accessing from multiple devices without exposing to internet.' },
        ]}
        id="example-methods"
      />

      <WarningBlock
        title="Never Expose Ollama Directly to the Internet"
        content="Ollama has no authentication, rate limiting, or abuse prevention. Exposing port 11434 directly allows anyone to use your GPU, download models, and potentially access sensitive data in prompts. Always use a reverse proxy with authentication or a VPN."
        id="warning-security"
      />

      <NoteBlock
        type="tip"
        title="Tailscale for Easy Remote Access"
        content="Tailscale creates a private mesh VPN with zero configuration. Install it on both your laptop and GPU server, then access Ollama at the server's Tailscale IP. No port forwarding, no firewall changes, no certificates needed."
        id="note-tailscale"
      />
    </div>
  )
}
