import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function UserManagement() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">User Management & Authentication</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Open WebUI supports multi-user environments with role-based access control. Administrators
        can manage users, control model access, and configure authentication methods including
        OAuth and LDAP integration.
      </p>

      <DefinitionBlock
        title="User Roles"
        definition="Open WebUI has three roles: Admin (full access, system configuration, user management), User (standard chat access, can create conversations), and Pending (registered but not yet approved). Admins can also restrict which models each role can access."
        id="def-roles"
      />

      <PythonCode
        title="Terminal"
        code={`# Environment variables for authentication
docker run -d -p 3000:8080 \\
    -e WEBUI_AUTH=true \\
    -e ENABLE_SIGNUP=true \\
    -e DEFAULT_USER_ROLE=pending \\
    -e ENABLE_OAUTH_SIGNUP=true \\
    -e OAUTH_PROVIDER_NAME=google \\
    -e OAUTH_CLIENT_ID=your-client-id \\
    -e OAUTH_CLIENT_SECRET=your-client-secret \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Disable signup (admin creates all accounts)
# -e ENABLE_SIGNUP=false

# Auto-approve new users
# -e DEFAULT_USER_ROLE=user

# LDAP authentication
# -e ENABLE_LDAP=true
# -e LDAP_SERVER_HOST=ldap.example.com
# -e LDAP_SERVER_PORT=389
# -e LDAP_SEARCH_BASE=dc=example,dc=com`}
        id="code-auth-config"
      />

      <PythonCode
        title="manage_users.py"
        code={`import requests

BASE_URL = "http://localhost:3000/api/v1"
ADMIN_TOKEN = "your-admin-api-token"  # From Settings > Account > API Keys
HEADERS = {"Authorization": f"Bearer {ADMIN_TOKEN}"}

# List all users
resp = requests.get(f"{BASE_URL}/users", headers=HEADERS)
users = resp.json()
for user in users:
    print(f"  {user['name']} ({user['email']}) - role: {user['role']}")

# Approve a pending user
def approve_user(user_id):
    resp = requests.post(
        f"{BASE_URL}/users/{user_id}/role",
        headers=HEADERS,
        json={"role": "user"},
    )
    return resp.json()

# Update user role
def set_admin(user_id):
    resp = requests.post(
        f"{BASE_URL}/users/{user_id}/role",
        headers=HEADERS,
        json={"role": "admin"},
    )
    return resp.json()

# Delete a user
def delete_user(user_id):
    resp = requests.delete(
        f"{BASE_URL}/users/{user_id}",
        headers=HEADERS,
    )
    return resp.status_code == 200

# Example: approve all pending users
pending = [u for u in users if u["role"] == "pending"]
for user in pending:
    approve_user(user["id"])
    print(f"Approved: {user['name']}")`}
        id="code-manage-users"
      />

      <ExampleBlock
        title="Authentication Options"
        problem="What authentication methods does Open WebUI support?"
        steps={[
          { formula: 'Built-in: email/password with local accounts', explanation: 'Default method. No external dependencies.' },
          { formula: 'OAuth 2.0: Google, GitHub, Microsoft, custom OIDC', explanation: 'Single sign-on with existing identity providers.' },
          { formula: 'LDAP: Active Directory / OpenLDAP', explanation: 'Enterprise directory integration.' },
          { formula: 'Trusted header: reverse proxy authentication', explanation: 'Let nginx/Authelia handle auth and pass user info via headers.' },
        ]}
        id="example-auth-methods"
      />

      <NoteBlock
        type="tip"
        title="Per-User Model Access"
        content="Admins can restrict which models each user or role can access. This is useful for limiting expensive cloud model usage to specific users while giving everyone access to local models."
        id="note-model-access"
      />

      <WarningBlock
        title="Secure Your Admin Account"
        content="The admin account has full access to all conversations, user data, and system settings. Use a strong password, enable two-factor authentication if available, and limit the number of admin accounts. Regularly audit admin access."
        id="warning-admin-security"
      />
    </div>
  )
}
