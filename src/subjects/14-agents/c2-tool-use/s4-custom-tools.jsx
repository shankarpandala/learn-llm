import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function CustomTools() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Building Custom Tools</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        While many agent frameworks provide built-in tools, real-world applications
        require custom tools tailored to your domain. Building effective custom tools
        involves defining clear schemas, implementing reliable execution, handling
        authentication, and testing tool behavior with your LLM.
      </p>

      <DefinitionBlock
        title="Custom Tool"
        definition="A user-defined function exposed to an LLM agent through a tool schema. Custom tools bridge the gap between the model's reasoning capabilities and your application's specific APIs, databases, and business logic."
        id="def-custom-tool"
      />

      <h2 className="text-2xl font-semibold">Tool Implementation Pattern</h2>
      <p className="text-gray-700 dark:text-gray-300">
        A well-structured custom tool separates the schema definition (what the model sees)
        from the implementation (what actually runs). This makes tools testable,
        reusable, and easy to modify.
      </p>

      <PythonCode
        title="custom_tool_pattern.py"
        code={`import anthropic
import json
from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class Tool:
    """A custom tool with schema and implementation."""
    name: str
    description: str
    input_schema: dict
    execute: Callable[[dict], str]

def make_tool(name: str, description: str, properties: dict,
              required: list[str], handler: Callable) -> Tool:
    """Factory for creating tools with consistent structure."""
    return Tool(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required,
        },
        execute=handler,
    )

# --- Custom tool implementations ---

def handle_db_query(inputs: dict) -> str:
    """Execute a database query (simulated)."""
    sql = inputs["query"]
    # In production: validate SQL, use parameterized queries
    if "DROP" in sql.upper() or "DELETE" in sql.upper():
        return "Error: Destructive queries are not allowed"
    # Simulated result
    return json.dumps([
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"},
    ])

def handle_send_email(inputs: dict) -> str:
    """Send an email (simulated with confirmation)."""
    return (
        f"Email draft prepared:\\n"
        f"To: {inputs['to']}\\n"
        f"Subject: {inputs['subject']}\\n"
        f"Body: {inputs['body'][:100]}...\\n"
        f"Status: PENDING_CONFIRMATION (user must approve)"
    )

# --- Register tools ---

db_tool = make_tool(
    name="query_database",
    description=(
        "Execute a read-only SQL SELECT query against the application database. "
        "Tables: users (id, name, email, created_at), "
        "orders (id, user_id, amount, status, created_at). "
        "Use this to look up user or order information."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "SQL SELECT query to execute"
        }
    },
    required=["query"],
    handler=handle_db_query,
)

email_tool = make_tool(
    name="send_email",
    description=(
        "Compose and send an email. The email will be held for "
        "user confirmation before actually sending."
    ),
    properties={
        "to": {"type": "string", "description": "Recipient email address"},
        "subject": {"type": "string", "description": "Email subject line"},
        "body": {"type": "string", "description": "Email body text"},
    },
    required=["to", "subject", "body"],
    handler=handle_send_email,
)

TOOLS = [db_tool, email_tool]`}
        id="code-custom-tool-pattern"
      />

      <PythonCode
        title="tool_registry_agent.py"
        code={`import anthropic

client = anthropic.Anthropic()

class ToolRegistry:
    """Registry that maps tool names to their implementations."""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get_schemas(self) -> list[dict]:
        """Get all tool schemas for the Claude API."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self.tools.values()
        ]

    def execute(self, name: str, inputs: dict) -> str:
        """Execute a tool by name with error handling."""
        if name not in self.tools:
            return f"Error: Unknown tool '{name}'"
        try:
            return self.tools[name].execute(inputs)
        except Exception as e:
            return f"Error executing {name}: {e}"

# Build the registry
registry = ToolRegistry()
for tool in TOOLS:  # From previous example
    registry.register(tool)

# Run agent with registered tools
def run_agent(question: str, max_turns: int = 5) -> str:
    messages = [{"role": "user", "content": question}]
    schemas = registry.get_schemas()

    for _ in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=schemas,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    output = registry.execute(block.name, block.input)
                    print(f"  {block.name} -> {output[:80]}...")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    })
            messages.append({"role": "user", "content": results})
        else:
            return next(b.text for b in response.content if b.type == "text")

    return "Max turns reached"

answer = run_agent("Find all users and send a welcome email to Alice")`}
        id="code-tool-registry"
      />

      <ExampleBlock
        title="MCP: Model Context Protocol"
        problem="How do you share custom tools across different agents and applications?"
        steps={[
          { formula: 'MCP Server: Exposes tools via a standard JSON-RPC protocol', explanation: 'Any tool can be served as an MCP server, making it accessible to any MCP-compatible client.' },
          { formula: 'MCP Client: Connects to servers and discovers available tools', explanation: 'Claude Desktop, Claude Code, and other clients auto-discover tools from MCP servers.' },
          { formula: 'Benefit: Write once, use everywhere', explanation: 'A database tool written as an MCP server works in any agent framework that supports MCP.' },
        ]}
        id="example-mcp"
      />

      <NoteBlock
        type="tip"
        title="Human-in-the-Loop for Dangerous Tools"
        content="Tools that modify state (send emails, write to databases, make purchases) should always include a confirmation step. Return a 'pending confirmation' status and require explicit user approval before executing the action. This prevents costly mistakes from model errors."
        id="note-human-in-loop"
      />

      <WarningBlock
        title="Security: Tool Injection Attacks"
        content="If tool inputs include user-generated content, an attacker could craft prompts that cause the model to misuse tools (e.g., SQL injection through tool parameters). Always sanitize and validate tool inputs server-side. Never trust the model's output as safe input for sensitive operations."
        id="warning-tool-injection"
      />

      <NoteBlock
        type="note"
        title="Testing Custom Tools"
        content="Test tools at three levels: (1) Unit test the handler functions with various inputs including edge cases. (2) Integration test the tool with the LLM by verifying the model produces valid inputs for common queries. (3) End-to-end test the full agent loop to ensure tools compose correctly when used together."
        id="note-testing-tools"
      />
    </div>
  )
}
