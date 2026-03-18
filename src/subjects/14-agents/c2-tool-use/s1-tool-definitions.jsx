import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ToolDefinitions() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Function and Tool Schemas</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Tool use allows LLMs to interact with external systems by calling functions
        with structured inputs. Tools are defined using JSON Schema, which tells the
        model what tools are available, what parameters they accept, and what each
        parameter means. Well-designed tool schemas are critical for reliable agent behavior.
      </p>

      <DefinitionBlock
        title="Tool Definition"
        definition="A structured description of an external function that an LLM can invoke. It includes a name, a natural language description, and an input schema (typically JSON Schema) specifying the parameters, their types, constraints, and descriptions."
        id="def-tool-definition"
      />

      <h2 className="text-2xl font-semibold">JSON Schema for Tools</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Both Claude and OpenAI use JSON Schema to define tool parameters. The schema
        tells the model exactly what structure of input to produce when calling a tool.
      </p>

      <PythonCode
        title="tool_schema_anatomy.py"
        code={`# Anatomy of a Claude tool definition
weather_tool = {
    "name": "get_weather",               # Unique identifier
    "description": (                       # Natural language - crucial for the model
        "Get the current weather conditions for a specified city. "
        "Returns temperature, humidity, and weather condition. "
        "Use this when the user asks about weather or needs to "
        "plan outdoor activities."
    ),
    "input_schema": {                      # JSON Schema object
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'San Francisco' or 'London, UK'"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units. Default: celsius"
            }
        },
        "required": ["city"]               # Only city is required
    }
}

# OpenAI function calling format (slightly different wrapper)
openai_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {                    # Same JSON Schema, different key name
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
}`}
        id="code-tool-schema"
      />

      <ExampleBlock
        title="Designing Effective Tool Descriptions"
        problem="How should you write tool descriptions so the model reliably chooses the right tool?"
        steps={[
          { formula: 'Bad: "Database tool"', explanation: 'Too vague. The model cannot determine when to use this.' },
          { formula: 'Better: "Query the SQL database"', explanation: 'Clearer, but still lacks guidance on when to use it.' },
          { formula: 'Best: "Execute a read-only SQL query against the users database. Use when you need to look up user information like email, name, or account status. Returns up to 100 rows."', explanation: 'Describes capability, use cases, and constraints.' },
        ]}
        id="example-tool-descriptions"
      />

      <PythonCode
        title="complex_tool_schemas.py"
        code={`import anthropic

client = anthropic.Anthropic()

# Real-world tool definitions with rich schemas
tools = [
    {
        "name": "search_documents",
        "description": (
            "Search the knowledge base for relevant documents. "
            "Returns ranked results with snippets. Use for factual "
            "questions about company policies, products, or procedures."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                },
                "filters": {
                    "type": "object",
                    "description": "Optional filters to narrow results",
                    "properties": {
                        "department": {
                            "type": "string",
                            "enum": ["engineering", "sales", "hr", "legal"]
                        },
                        "date_after": {
                            "type": "string",
                            "description": "ISO date string, e.g. '2024-01-01'"
                        }
                    }
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (1-20)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "create_ticket",
        "description": (
            "Create a support ticket in the issue tracker. Use when "
            "the user reports a bug or requests a feature. Always "
            "confirm details with the user before creating."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Brief ticket title (max 100 chars)"
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of the issue"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"]
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels like 'bug', 'feature', 'docs'"
                }
            },
            "required": ["title", "description", "priority"]
        }
    }
]

# The model uses schemas to produce structured tool calls
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    messages=[{
        "role": "user",
        "content": "Find documents about our vacation policy from HR"
    }]
)

for block in response.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}")
        print(f"Input: {block.input}")
        # Tool: search_documents
        # Input: {"query": "vacation policy", "filters": {"department": "hr"}}`}
        id="code-complex-schemas"
      />

      <NoteBlock
        type="tip"
        title="Schema Design Best Practices"
        content="Use descriptive enum values instead of codes (e.g., 'high' not '3'). Add examples in descriptions for ambiguous fields. Keep required fields minimal to give the model flexibility. Use default values where sensible. Test your schemas with edge cases to ensure the model fills them correctly."
        id="note-schema-practices"
      />

      <WarningBlock
        title="Schema Complexity Limits"
        content="Extremely complex schemas with deep nesting, many optional fields, or complex conditional logic can confuse the model. If a tool needs more than 5-7 parameters, consider splitting it into multiple simpler tools. Models are also more reliable with tools when there are fewer than 20 total tools defined."
        id="warning-schema-complexity"
      />

      <NoteBlock
        type="note"
        title="Tool Schemas Across Providers"
        content="Claude uses 'input_schema' with 'tools' parameter. OpenAI uses 'parameters' wrapped in a 'function' object. Google Gemini uses a similar format to OpenAI. Despite these syntactic differences, the underlying JSON Schema definitions are interchangeable. Libraries like LiteLLM abstract these differences."
        id="note-cross-provider"
      />
    </div>
  )
}
