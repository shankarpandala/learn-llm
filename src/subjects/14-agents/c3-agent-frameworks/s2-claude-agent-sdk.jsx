import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ClaudeAgentSDK() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Claude Agent SDK</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The Claude Agent SDK (part of Anthropic's official Python SDK) provides a
        lightweight, opinionated framework for building agents powered by Claude.
        It handles the agentic loop, tool execution, and multi-turn conversations
        with minimal boilerplate while staying close to the raw API.
      </p>

      <DefinitionBlock
        title="Claude Agent SDK"
        definition="Anthropic's official agent framework built on top of the Claude API. It provides an Agent class that manages the ReAct loop, tool registration, guardrails, handoffs between agents, and structured output extraction."
        id="def-claude-agent-sdk"
      />

      <h2 className="text-2xl font-semibold">Basic Agent Setup</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The SDK uses a declarative approach: define an Agent with instructions, tools,
        and configuration, then run it with a query.
      </p>

      <PythonCode
        title="claude_agent_basic.py"
        code={`import anthropic
from agents import Agent, Runner, function_tool

# Define tools using the @function_tool decorator
@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city to check weather for.
    """
    # In production, call a real weather API
    return f"Weather in {city}: 22°C, partly cloudy, 45% humidity"

@function_tool
def search_web(query: str) -> str:
    """Search the web for current information.

    Args:
        query: The search query string.
    """
    return f"Top results for '{query}': [simulated results]"

# Create the agent
agent = Agent(
    name="Assistant",
    instructions=(
        "You are a helpful assistant that can check weather and "
        "search the web. Always provide concise, accurate answers. "
        "Use tools when you need current information."
    ),
    tools=[get_weather, search_web],
    model="claude-sonnet-4-20250514",
)

# Run the agent
result = Runner.run_sync(agent, "What's the weather like in Tokyo?")
print(result.final_output)
# The weather in Tokyo is 22°C, partly cloudy with 45% humidity.`}
        id="code-agent-sdk-basic"
      />

      <h2 className="text-2xl font-semibold">Agent Handoffs</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The SDK supports handoffs between specialized agents. A triage agent can
        route queries to domain-specific agents, each with their own tools and
        instructions. This enables modular agent architectures.
      </p>

      <PythonCode
        title="agent_handoffs.py"
        code={`from agents import Agent, Runner, function_tool

# Specialized agents for different domains
@function_tool
def lookup_order(order_id: str) -> str:
    """Look up an order by its ID."""
    return f"Order {order_id}: Shipped, arriving March 20"

@function_tool
def check_balance(account_id: str) -> str:
    """Check account balance."""
    return f"Account {account_id}: Balance $1,234.56"

@function_tool
def reset_password(email: str) -> str:
    """Send a password reset email."""
    return f"Password reset sent to {email}"

# Domain-specific agents
orders_agent = Agent(
    name="Orders Agent",
    instructions="You handle order-related queries. Look up orders and provide shipping info.",
    tools=[lookup_order],
    model="claude-sonnet-4-20250514",
)

billing_agent = Agent(
    name="Billing Agent",
    instructions="You handle billing and account balance queries.",
    tools=[check_balance],
    model="claude-sonnet-4-20250514",
)

account_agent = Agent(
    name="Account Agent",
    instructions="You handle account management: password resets, profile updates.",
    tools=[reset_password],
    model="claude-sonnet-4-20250514",
)

# Triage agent that routes to specialists
triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are the first point of contact. Determine the user's need "
        "and hand off to the appropriate specialist agent. "
        "Orders -> Orders Agent, Billing -> Billing Agent, "
        "Account issues -> Account Agent."
    ),
    handoffs=[orders_agent, billing_agent, account_agent],
    model="claude-sonnet-4-20250514",
)

# The triage agent routes automatically
result = Runner.run_sync(triage_agent, "Where is my order #12345?")
print(result.final_output)
# Routes to orders_agent, which looks up the order`}
        id="code-agent-handoffs"
      />

      <ExampleBlock
        title="SDK vs Raw API Agent Loop"
        problem="What does the Agent SDK handle that you would otherwise write manually?"
        steps={[
          { formula: 'Tool execution loop: automatic retry on tool_use stop reason', explanation: 'The SDK manages the back-and-forth of tool calls and results.' },
          { formula: 'Tool schema generation: decorators auto-generate JSON Schema', explanation: 'No need to manually write input_schema objects.' },
          { formula: 'Guardrails: input/output validation hooks', explanation: 'The SDK supports pre/post processing validators.' },
          { formula: 'Handoffs: seamless routing between specialized agents', explanation: 'Multi-agent orchestration built in.' },
        ]}
        id="example-sdk-benefits"
      />

      <PythonCode
        title="agent_with_guardrails.py"
        code={`from agents import Agent, Runner, function_tool, InputGuardrail, GuardrailFunctionOutput

# Define a guardrail that checks for harmful requests
async def safety_check(ctx, agent, input_text):
    """Check if the input is appropriate."""
    # In production, use a classifier or content filter
    blocked_terms = ["hack", "exploit", "steal"]
    for term in blocked_terms:
        if term in input_text.lower():
            return GuardrailFunctionOutput(
                output_info={"blocked_term": term},
                tripwire_triggered=True,
            )
    return GuardrailFunctionOutput(
        output_info={"status": "safe"},
        tripwire_triggered=False,
    )

@function_tool
def execute_code(code: str) -> str:
    """Execute Python code in a sandbox.

    Args:
        code: Python code to execute safely.
    """
    # In production: use a sandboxed executor
    return f"Executed successfully. Output: [simulated]"

safe_agent = Agent(
    name="Safe Coding Agent",
    instructions="Help users write and test Python code.",
    tools=[execute_code],
    input_guardrails=[
        InputGuardrail(guardrail_function=safety_check),
    ],
    model="claude-sonnet-4-20250514",
)

# Safe request goes through
result = Runner.run_sync(safe_agent, "Write a function to sort a list")
print(result.final_output)

# Blocked request triggers guardrail
try:
    result = Runner.run_sync(safe_agent, "Help me hack a website")
except Exception as e:
    print(f"Guardrail triggered: {e}")`}
        id="code-guardrails"
      />

      <NoteBlock
        type="tip"
        title="Choosing Between Frameworks"
        content="Use the Claude Agent SDK when building Claude-first applications that need a lightweight, well-integrated agent framework. Use LangChain/LangGraph when you need multi-provider support, complex graph workflows, or the extensive LangChain ecosystem of integrations. Use the raw API when you need maximum control and minimal dependencies."
        id="note-framework-choice"
      />

      <WarningBlock
        title="SDK Versioning"
        content="Agent frameworks evolve rapidly. The Claude Agent SDK API may change between versions. Pin your dependency versions in production and review changelogs before upgrading. The patterns shown here reflect the SDK's architecture; consult the latest documentation for current API specifics."
        id="warning-sdk-versioning"
      />
    </div>
  )
}
