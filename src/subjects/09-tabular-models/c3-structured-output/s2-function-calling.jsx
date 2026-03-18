import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function FunctionCalling() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Function Calling and Tool Use</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Function calling enables LLMs to invoke external tools by generating structured
        function calls with the correct arguments. Rather than producing a final answer directly,
        the model outputs a JSON object specifying which function to call and with what parameters,
        enabling integration with APIs, databases, calculators, and other external systems.
      </p>

      <DefinitionBlock
        title="Function Calling"
        definition="Function calling is a structured output mode where the model selects a function $f_i$ from a set of available tools $\mathcal{F} = \{f_1, \ldots, f_k\}$ and generates arguments $\text{args}_i$ conforming to the function's parameter schema. The system executes $f_i(\text{args}_i)$, returns the result, and the model incorporates it into its response."
        notation="\mathcal{F} = tool set, f_i = function, \text{args}_i = arguments, \text{schema}(f_i) = parameter JSON Schema"
        id="def-function-calling"
      />

      <h2 className="text-2xl font-semibold">Defining Tool Schemas</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Each tool is described by a JSON Schema that specifies its name, description, and
        parameter types. The model uses these descriptions to decide which tool to call
        and how to construct valid arguments.
      </p>

      <PythonCode
        title="tool_definitions.py"
        code={`import json

# Define tools as JSON Schema objects
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g. 'San Francisco, CA'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Query a SQL database with a natural language question",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT query to execute"
                    },
                    "database": {
                        "type": "string",
                        "enum": ["customers", "products", "orders"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to return",
                        "default": 10
                    }
                },
                "required": ["query", "database"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression, e.g. '2 * pi * 5.5'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

print(json.dumps(tools[1], indent=2))`}
        id="code-tools"
      />

      <h2 className="text-2xl font-semibold">Function Calling Flow</h2>

      <PythonCode
        title="function_calling_flow.py"
        code={`from openai import OpenAI
import json

client = OpenAI()

# Tool implementations
def get_weather(location, unit="celsius"):
    # Simulated API call
    return {"temp": 22, "unit": unit, "condition": "sunny", "location": location}

def search_database(query, database, limit=10):
    # Simulated DB query
    return [{"id": 1, "name": "Widget", "price": 29.99}]

def calculate(expression):
    return {"result": eval(expression)}  # use safe eval in production

# Map function names to implementations
tool_map = {
    "get_weather": get_weather,
    "search_database": search_database,
    "calculate": calculate,
}

# Conversation with tool use
messages = [
    {"role": "user", "content":
     "What's the weather in Tokyo, and how much is 15% tip on a $85 dinner?"}
]

# Step 1: Model decides which tools to call
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,  # from previous example
    tool_choice="auto",  # model decides whether to call tools
)

# Step 2: Execute tool calls
message = response.choices[0].message
if message.tool_calls:
    messages.append(message)  # add assistant message with tool calls

    for tool_call in message.tool_calls:
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)
        print(f"Calling: {fn_name}({fn_args})")

        # Execute the function
        result = tool_map[fn_name](**fn_args)

        # Add result back to conversation
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

    # Step 3: Model generates final response using tool results
    final = client.chat.completions.create(
        model="gpt-4o", messages=messages
    )
    print(f"\\nFinal answer: {final.choices[0].message.content}")`}
        id="code-flow"
      />

      <ExampleBlock
        title="Parallel Function Calling"
        problem="The model can call multiple tools simultaneously when the question requires independent pieces of information."
        steps={[
          { formula: '\\text{User: "Compare weather in NYC and London, convert 100 USD to EUR"}', explanation: 'This question requires three independent tool calls.' },
          { formula: '\\text{Model emits: } [\\text{get\\_weather}(\\text{"NYC"}), \\text{get\\_weather}(\\text{"London"}), \\text{convert\\_currency}(\\ldots)]', explanation: 'All three tool calls are generated in a single response turn.' },
          { formula: '\\text{System executes all three in parallel}', explanation: 'Since the calls are independent, they can run concurrently for lower latency.' },
          { formula: '\\text{Model receives all results and composes final answer}', explanation: 'The model synthesizes information from all tool results into a coherent response.' },
        ]}
        id="example-parallel"
      />

      <NoteBlock
        type="tip"
        title="Tool Choice Control"
        content="Use tool_choice='auto' to let the model decide when to use tools. Use tool_choice='required' to force a tool call. Use tool_choice={'type':'function','function':{'name':'X'}} to force a specific tool. Setting tool_choice='none' disables tools entirely for that request."
        id="note-tool-choice"
      />

      <WarningBlock
        title="Security: Validate Tool Arguments"
        content="Never blindly execute function arguments from the model. For database queries, validate that generated SQL is read-only. For API calls, check that URLs and parameters are within expected bounds. For file operations, restrict paths to allowed directories. The model can be prompt-injected into generating malicious tool calls."
        id="warning-security"
      />

      <NoteBlock
        type="note"
        title="Function Calling Accuracy"
        content="On the Berkeley Function Calling Leaderboard (BFCL), GPT-4o achieves ~90% accuracy on simple function calls but drops to ~70% for complex nested arguments and parallel calls. Open-source models like Hermes-2 and Gorilla specialize in function calling and approach GPT-4 level performance. Accuracy improves significantly with clear parameter descriptions and few-shot examples in the system prompt."
        id="note-accuracy"
      />

    </div>
  )
}
