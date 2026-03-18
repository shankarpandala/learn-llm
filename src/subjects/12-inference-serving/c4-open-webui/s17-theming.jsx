import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Theming() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Theming & Customization</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Open WebUI supports visual customization through themes, custom CSS, and branding options.
        You can match the interface to your organization's look and feel or simply personalize
        it for your own preferences.
      </p>

      <ExampleBlock
        title="Customization Options"
        problem="What can you visually customize in Open WebUI?"
        steps={[
          { formula: 'Dark/Light mode: toggle in settings', explanation: 'Built-in dark and light themes with automatic system detection.' },
          { formula: 'Custom CSS: inject arbitrary CSS styles', explanation: 'Full control over colors, fonts, spacing, and layout.' },
          { formula: 'Branding: custom logo, name, description', explanation: 'White-label the interface for your organization.' },
          { formula: 'Chat interface: message bubbles, code themes', explanation: 'Customize how conversations are displayed.' },
        ]}
        id="example-options"
      />

      <PythonCode
        title="Terminal"
        code={`# Set custom branding via environment variables
docker run -d -p 3000:8080 \\
    -e WEBUI_NAME="Acme AI Assistant" \\
    -e ENABLE_SIGNUP=true \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Custom CSS can be added through the Admin Panel:
# Settings > Interface > Custom CSS

# Or mount a custom CSS file:
# -v ./custom.css:/app/build/static/custom.css`}
        id="code-branding"
      />

      <PythonCode
        title="custom_themes.py"
        code={`# Example custom CSS themes for Open WebUI
# Apply via Admin Panel > Settings > Interface > Custom CSS

themes = {
    "corporate_blue": """
/* Corporate blue theme */
:root {
    --primary-color: #1e40af;
    --primary-hover: #1d4ed8;
    --background-color: #f8fafc;
    --sidebar-bg: #1e293b;
    --sidebar-text: #e2e8f0;
}

.dark {
    --background-color: #0f172a;
    --sidebar-bg: #1e293b;
}

/* Custom font */
body {
    font-family: 'Inter', -apple-system, sans-serif;
}

/* Rounded message bubbles */
.message-content {
    border-radius: 16px;
    padding: 12px 16px;
}
""",
    "minimal": """
/* Minimal, clean theme */
.sidebar {
    border-right: 1px solid #e5e7eb;
}

/* Hide the logo/branding */
.logo-container { display: none; }

/* Increase content width */
.max-w-3xl { max-width: 56rem; }

/* Subtle code blocks */
pre {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
}
""",
    "high_contrast": """
/* High contrast for accessibility */
body { font-size: 18px; }
.dark {
    --background-color: #000000;
    --text-color: #ffffff;
}
.message-content {
    line-height: 1.8;
    letter-spacing: 0.02em;
}
a { text-decoration: underline; }
""",
}

# Print theme for copy-pasting into Open WebUI
for name, css in themes.items():
    print(f"\\n{'='*40}")
    print(f"Theme: {name}")
    print(f"{'='*40}")
    print(css)`}
        id="code-themes"
      />

      <NoteBlock
        type="tip"
        title="Per-User Preferences"
        content="Each user can set their own theme preference (dark/light) independently. Custom CSS set by admins applies to all users. For per-user CSS customization, users can use browser extensions like Stylus."
        id="note-per-user"
      />

      <NoteBlock
        type="note"
        title="Custom Landing Page"
        content="You can customize the landing page shown to new visitors before they log in. This is useful for displaying your organization's AI usage policy, instructions, or branding. Configure through the admin panel under Interface settings."
        id="note-landing"
      />

      <WarningBlock
        title="CSS Injection Safety"
        content="Custom CSS is powerful but can break the interface if not carefully tested. Always test CSS changes in a development environment first. Keep a backup of working CSS before making changes. Invalid CSS can make the admin panel inaccessible."
        id="warning-css"
      />
    </div>
  )
}
