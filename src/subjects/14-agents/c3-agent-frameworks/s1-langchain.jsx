import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LangChainAgents() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LangChain Agents</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        LangChain is one of the most widely used frameworks for building LLM applications
        and agents. It provides abstractions for models, tools, memory, and agent
        orchestration, along with LangGraph for building stateful, multi-step agent
        workflows as graphs.
      </p>

      <DefinitionBlock
        title="LangChain Agent"
        definition="An autonomous system built with the LangChain framework that uses an LLM to decide which tools to call, in what order, and with what inputs. LangChain agents follow the ReAct pattern and support customizable tool sets, memory, and execution strategies."
        id="def-langchain-agent"
      />

      <h2 className="text-2xl font-semibold">Basic Agent Setup</h2>
      <p className="text-gray-700 dark:text-gray-300">
        LangChain provides a high-level interface for creating agents with tools.
        The framework handles the ReAct loop, tool execution, and message formatting.
      </p>

      <PythonCode
        title="langchain_basic_agent.py"
        code={`from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Initialize the model
model = ChatAnthropic(model="claude-sonnet-4-20250514")

# Define tools using the @tool decorator
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Use Python math syntax.
    Examples: '2 + 2', '(10 * 5) / 3', '2 ** 10'"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_word_length(word: str) -> int:
    """Get the length of a word. Use when asked about word lengths."""
    return len(word)

@tool
def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base for information.
    Use for questions about policies, procedures, or products."""
    # Simulated search
    return f"Found 3 results for '{query}': [result summaries...]"

# Create the agent using LangGraph's prebuilt ReAct agent
tools = [calculator, get_word_length, search_knowledge_base]
agent = create_react_agent(model, tools)

# Run the agent
result = agent.invoke({
    "messages": [("user", "What is 15% of 847, and how many letters are in 'extraordinary'?")]
})

# Print the conversation
for message in result["messages"]:
    print(f"{message.type}: {message.content[:100]}...")`}
        id="code-langchain-basic"
      />

      <h2 className="text-2xl font-semibold">LangGraph: Stateful Agent Workflows</h2>
      <p className="text-gray-700 dark:text-gray-300">
        LangGraph extends LangChain with graph-based agent orchestration. It lets you
        define agents as state machines with explicit control flow, making complex
        multi-step workflows predictable and debuggable.
      </p>

      <PythonCode
        title="langgraph_workflow.py"
        code={`from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode

model = ChatAnthropic(model="claude-sonnet-4-20250514")

@tool
def search(query: str) -> str:
    """Search for information on the web."""
    return f"Search results for '{query}': ..."

@tool
def write_report(content: str, title: str) -> str:
    """Write a structured report with the given content."""
    return f"Report '{title}' written successfully ({len(content)} chars)"

tools = [search, write_report]
model_with_tools = model.bind_tools(tools)
tool_node = ToolNode(tools)

# Define the agent logic
def should_continue(state: MessagesState):
    """Decide whether to use tools or finish."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: MessagesState):
    """Invoke the model with the current messages."""
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# Build the graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

# Compile and run
app = workflow.compile()
result = app.invoke({
    "messages": [
        SystemMessage(content="You are a research assistant."),
        HumanMessage(content="Research AI agents and write a brief report"),
    ]
})`}
        id="code-langgraph"
      />

      <ExampleBlock
        title="LangChain vs Raw API"
        problem="When should you use LangChain versus the raw Claude/OpenAI API?"
        steps={[
          { formula: 'Use LangChain: Rapid prototyping, multiple LLM providers, complex chains', explanation: 'LangChain shines when you need to swap models, compose chains, or use built-in tools.' },
          { formula: 'Use raw API: Production systems, fine control, minimal dependencies', explanation: 'Direct API calls give you full control and avoid framework overhead.' },
          { formula: 'Use LangGraph: Complex multi-step workflows with branching logic', explanation: 'When your agent has explicit states, conditions, and parallel branches.' },
        ]}
        id="example-langchain-vs-raw"
      />

      <NoteBlock
        type="tip"
        title="LangSmith for Debugging"
        content="LangSmith (LangChain's observability platform) traces every step of agent execution, including model calls, tool invocations, and intermediate states. This is invaluable for debugging agents that produce unexpected results. Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable tracing."
        id="note-langsmith"
      />

      <WarningBlock
        title="Abstraction Trade-offs"
        content="LangChain's abstractions can hide important details. When debugging, you may need to understand the exact prompts being sent, token counts, and API parameters. Over-reliance on high-level abstractions can make it harder to optimize performance and costs. Always be prepared to drop down to the raw API level when needed."
        id="warning-langchain-abstractions"
      />

      <NoteBlock
        type="note"
        title="LangChain Ecosystem"
        content="The LangChain ecosystem includes: langchain-core (base abstractions), langchain-community (integrations), langchain-anthropic/openai (provider packages), LangGraph (graph orchestration), LangSmith (observability), and LangServe (deployment). Start with the specific provider package you need rather than installing the full langchain package."
        id="note-ecosystem"
      />
    </div>
  )
}
