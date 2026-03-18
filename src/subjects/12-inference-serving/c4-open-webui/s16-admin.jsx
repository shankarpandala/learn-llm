import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Admin() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Admin Panel</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The Admin Panel is the control center for managing your Open WebUI deployment. It covers
        user management, model access control, connection configuration, and system-wide settings
        that affect all users.
      </p>

      <DefinitionBlock
        title="Admin Panel"
        definition="The Admin Panel is accessible to users with the admin role. It provides centralized management of connections (Ollama, OpenAI), user accounts and roles, model visibility and access control, default settings, and system configuration."
        id="def-admin"
      />

      <ExampleBlock
        title="Admin Panel Sections"
        problem="What can you configure in the admin panel?"
        steps={[
          { formula: 'Connections: Ollama URL, OpenAI API keys, custom endpoints', explanation: 'Configure all LLM backends from one place.' },
          { formula: 'Users: approve/deny signups, assign roles, set permissions', explanation: 'Control who can access the system and what they can do.' },
          { formula: 'Models: show/hide models, set defaults, create presets', explanation: 'Control which models are visible and set default parameters.' },
          { formula: 'Settings: auth, RAG config, web search, image gen', explanation: 'System-wide configuration for all features.' },
        ]}
        id="example-sections"
      />

      <PythonCode
        title="Terminal"
        code={`# Key environment variables for admin configuration
docker run -d -p 3000:8080 \\
    -e WEBUI_AUTH=true \\
    -e ENABLE_SIGNUP=true \\
    -e DEFAULT_USER_ROLE=pending \\
    -e ENABLE_ADMIN_EXPORT=true \\
    -e ENABLE_ADMIN_CHAT_ACCESS=false \\
    -e ENABLE_COMMUNITY_SHARING=false \\
    -e DEFAULT_MODELS="llama3.2" \\
    -e MODEL_FILTER_ENABLED=true \\
    -e MODEL_FILTER_LIST="llama3.2;mistral;phi3:mini" \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Restrict model access (only listed models are visible)
# MODEL_FILTER_ENABLED=true + MODEL_FILTER_LIST filters the model dropdown

# Disable admin access to user conversations
# ENABLE_ADMIN_CHAT_ACCESS=false (privacy-respecting default)

# Reset admin password if locked out
docker exec open-webui open-webui reset-admin-password
# Outputs a temporary password`}
        id="code-admin-config"
      />

      <PythonCode
        title="admin_api.py"
        code={`import requests

BASE_URL = "http://localhost:3000/api/v1"
ADMIN_TOKEN = "your-admin-token"
HEADERS = {"Authorization": f"Bearer {ADMIN_TOKEN}"}

# Get system configuration
resp = requests.get(f"{BASE_URL}/configs", headers=HEADERS)
if resp.status_code == 200:
    config = resp.json()
    print("Current configuration:")
    for key, value in config.items():
        if not key.startswith("_"):
            print(f"  {key}: {value}")

# List all users with their roles
resp = requests.get(f"{BASE_URL}/users", headers=HEADERS)
users = resp.json()
print(f"\\nTotal users: {len(users)}")
role_counts = {}
for user in users:
    role = user.get("role", "unknown")
    role_counts[role] = role_counts.get(role, 0) + 1
    print(f"  {user['name']:<20} {user['email']:<30} {role}")
print(f"\\nRole distribution: {role_counts}")

# Export all chats (admin function for backup)
resp = requests.get(
    f"{BASE_URL}/chats/all/export",
    headers=HEADERS,
)
if resp.status_code == 200:
    import json
    with open("all_chats_backup.json", "w") as f:
        json.dump(resp.json(), f, indent=2)
    print("\\nExported all chats to all_chats_backup.json")`}
        id="code-admin-api"
      />

      <NoteBlock
        type="tip"
        title="Regular Backups"
        content="Back up the Open WebUI data volume regularly. It contains all conversations, user accounts, uploaded documents, and settings. Use 'docker cp open-webui:/app/backend/data ./backup' or mount the volume to a backed-up filesystem."
        id="note-backups"
      />

      <WarningBlock
        title="Admin Chat Access"
        content="When ENABLE_ADMIN_CHAT_ACCESS is true, admins can view all user conversations. This may be required for compliance but raises privacy concerns. Communicate the policy clearly to users and disable it unless strictly necessary."
        id="warning-chat-access"
      />
    </div>
  )
}
