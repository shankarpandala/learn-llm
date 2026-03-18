import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function SystemPrompts() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">System Prompts & Model Presets</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        System prompts define how a model behaves, its personality, capabilities, and constraints.
        Open WebUI lets you create reusable presets combining system prompts with model parameters,
        making it easy to switch between different assistant configurations.
      </p>

      <DefinitionBlock
        title="System Prompt"
        definition="A system prompt is a special instruction given to the model before the user's message. It sets the context, personality, output format, and behavioral constraints. In the OpenAI message format, it uses role: 'system'."
        id="def-system-prompt"
      />

      <PythonCode
        title="Terminal"
        code={`# Set a global default system prompt via environment
docker run -d -p 3000:8080 \\
    -e DEFAULT_SYSTEM_PROMPT="You are a helpful, concise assistant." \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# In the UI:
# 1. Click the model selector dropdown
# 2. Click "Set System Prompt" (pencil icon)
# 3. Enter your system prompt
# 4. It persists for that conversation

# Or create a Model Preset:
# Admin Panel > Workspace > Models > Add Model
# Configure system prompt + parameters as a reusable preset`}
        id="code-system-prompt"
      />

      <PythonCode
        title="preset_examples.py"
        code={`import requests

API_URL = "http://localhost:3000/api/chat/completions"
HEADERS = {
    "Authorization": "Bearer YOUR_TOKEN",
    "Content-Type": "application/json",
}

# Define reusable presets
PRESETS = {
    "code_reviewer": {
        "system": (
            "You are an expert code reviewer. Analyze code for bugs, "
            "performance issues, security vulnerabilities, and style. "
            "Rate severity as CRITICAL/WARNING/INFO. Be concise."
        ),
        "temperature": 0.2,
        "top_p": 0.9,
    },
    "socratic_tutor": {
        "system": (
            "You are a Socratic tutor. Never give direct answers. Instead, "
            "guide the student by asking leading questions that help them "
            "discover the answer themselves. Use encouragement."
        ),
        "temperature": 0.7,
        "top_p": 0.95,
    },
    "json_api": {
        "system": (
            "You are a JSON API. Respond ONLY with valid JSON. No markdown, "
            "no explanations, no text outside JSON. Use descriptive keys."
        ),
        "temperature": 0.1,
        "top_p": 0.8,
    },
    "eli5": {
        "system": (
            "Explain everything as if to a 5-year-old. Use simple words, "
            "short sentences, fun analogies. No jargon or technical terms."
        ),
        "temperature": 0.8,
        "top_p": 0.95,
    },
}

def chat_with_preset(preset_name, user_message, model="llama3.2"):
    preset = PRESETS[preset_name]
    resp = requests.post(API_URL, headers=HEADERS, json={
        "model": model,
        "messages": [
            {"role": "system", "content": preset["system"]},
            {"role": "user", "content": user_message},
        ],
        "temperature": preset["temperature"],
        "top_p": preset["top_p"],
        "stream": False,
    })
    return resp.json()["choices"][0]["message"]["content"]

# Test presets
code = "def add(a, b): return a + b"
print("Code review:", chat_with_preset("code_reviewer", f"Review: {code}"))
print("\\nELI5:", chat_with_preset("eli5", "What is gravity?"))`}
        id="code-presets"
      />

      <ExampleBlock
        title="Effective System Prompt Patterns"
        problem="What makes a system prompt effective?"
        steps={[
          { formula: 'Role definition: \"You are a [specific role]\"', explanation: 'Give the model a clear identity and expertise area.' },
          { formula: 'Output format: \"Respond in [format]\"', explanation: 'Specify JSON, markdown, bullet points, tables, etc.' },
          { formula: 'Constraints: \"Never [behavior to avoid]\"', explanation: 'Explicitly state what the model should not do.' },
          { formula: 'Examples: show the desired output format', explanation: 'One or two examples in the system prompt dramatically improve consistency.' },
        ]}
        id="example-patterns"
      />

      <NoteBlock
        type="tip"
        title="Share Presets Across Users"
        content="Admin-created model presets are available to all users. Create presets for common team use cases (customer support, code review, documentation) so everyone benefits from optimized configurations."
        id="note-sharing"
      />

      <WarningBlock
        title="System Prompts Use Context Window"
        content="A long system prompt consumes tokens from the context window. A 500-token system prompt on a 4096-context model leaves only 3596 tokens for conversation. Keep system prompts concise -- aim for under 200 tokens."
        id="warning-context"
      />
    </div>
  )
}
