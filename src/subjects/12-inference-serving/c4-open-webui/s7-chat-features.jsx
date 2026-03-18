import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ChatFeatures() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Chat Interface Features</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Open WebUI provides a rich chat experience that goes beyond basic text exchange. Features
        like conversation branching, message editing, regeneration, and file attachments make it
        a powerful tool for working with LLMs.
      </p>

      <ExampleBlock
        title="Core Chat Features"
        problem="What can you do in an Open WebUI conversation?"
        steps={[
          { formula: 'Branching: create alternative responses at any point', explanation: 'Click the branch icon on any message to explore different response paths.' },
          { formula: 'Regenerate: retry the last response with same or different model', explanation: 'If the response is poor, regenerate without retyping the prompt.' },
          { formula: 'Edit & resubmit: modify any previous message', explanation: 'Change a prompt mid-conversation and get a new response chain.' },
          { formula: 'File attachments: upload documents, images, code', explanation: 'Attach files directly to messages for the model to process.' },
          { formula: 'Code highlighting: syntax-highlighted code blocks', explanation: 'Code in responses is automatically highlighted with copy buttons.' },
        ]}
        id="example-features"
      />

      <PythonCode
        title="Terminal"
        code={`# Open WebUI chat features are primarily used through the web UI
# But many can be accessed via the API as well

# Create a conversation via API
curl -X POST http://localhost:3000/api/v1/chats/new \\
    -H "Authorization: Bearer YOUR_TOKEN" \\
    -H "Content-Type: application/json" \\
    -d '{
        "chat": {
            "title": "API Test Chat",
            "messages": []
        }
    }'

# Send a message in a conversation
curl -X POST http://localhost:3000/api/chat/completions \\
    -H "Authorization: Bearer YOUR_TOKEN" \\
    -H "Content-Type: application/json" \\
    -d '{
        "model": "llama3.2",
        "messages": [
            {"role": "user", "content": "What is the meaning of life?"}
        ],
        "stream": true
    }'`}
        id="code-api-chat"
      />

      <PythonCode
        title="chat_export.py"
        code={`import requests
import json

BASE_URL = "http://localhost:3000/api/v1"
TOKEN = "your-api-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# List all conversations
resp = requests.get(f"{BASE_URL}/chats", headers=HEADERS)
chats = resp.json()

print(f"Total conversations: {len(chats)}")
for chat in chats[:5]:
    title = chat.get("title", "Untitled")
    msg_count = len(chat.get("chat", {}).get("messages", []))
    print(f"  [{chat['id'][:8]}] {title} ({msg_count} messages)")

# Export a specific conversation
if chats:
    chat_id = chats[0]["id"]
    resp = requests.get(f"{BASE_URL}/chats/{chat_id}", headers=HEADERS)
    chat_data = resp.json()

    # Save as JSON
    with open(f"chat_{chat_id[:8]}.json", "w") as f:
        json.dump(chat_data, f, indent=2)
    print(f"Exported chat to chat_{chat_id[:8]}.json")

    # Format as readable text
    for msg in chat_data.get("chat", {}).get("messages", []):
        role = msg["role"].upper()
        content = msg["content"][:100]
        print(f"  [{role}] {content}...")`}
        id="code-export"
      />

      <NoteBlock
        type="tip"
        title="Keyboard Shortcuts"
        content="Open WebUI supports keyboard shortcuts: Enter to send (Shift+Enter for newline), Ctrl+Shift+C to copy last response, Ctrl+/ to toggle sidebar. These shortcuts make the chat experience much faster for power users."
        id="note-shortcuts"
      />

      <NoteBlock
        type="note"
        title="Markdown Rendering"
        content="Open WebUI renders full markdown in responses including tables, code blocks with syntax highlighting, LaTeX math, lists, and headings. This makes it excellent for technical conversations where structured formatting matters."
        id="note-markdown"
      />

      <WarningBlock
        title="Conversation History Size"
        content="Each message in the conversation history is sent to the LLM with every new request. Very long conversations can exceed the model's context window, causing early messages to be truncated. Start new conversations for new topics to avoid this."
        id="warning-history"
      />
    </div>
  )
}
