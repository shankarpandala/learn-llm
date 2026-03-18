import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ReAct() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">ReAct: Reasoning + Acting</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        ReAct (Yao et al., 2023) interleaves reasoning traces with action execution,
        allowing an LLM to think about what to do, take an action in the environment,
        observe the result, and repeat. This Thought-Action-Observation loop is the
        foundational pattern behind most modern AI agents.
      </p>

      <DefinitionBlock
        title="ReAct Framework"
        definition="ReAct synergizes reasoning (chain-of-thought) and acting (tool use) by generating interleaved Thought, Action, and Observation steps. The model reasons about what information it needs, takes actions to obtain it, then reasons about the results to decide next steps."
        id="def-react"
      />

      <ExampleBlock
        title="ReAct Trace Example"
        problem="Answer: 'What is the population of the capital of France?'"
        steps={[
          { formula: 'Thought: I need to find the capital of France, then its population.', explanation: 'The model decomposes the question into sub-tasks.' },
          { formula: 'Action: search("capital of France")', explanation: 'The model invokes a search tool.' },
          { formula: 'Observation: Paris is the capital of France.', explanation: 'The environment returns the search result.' },
          { formula: 'Thought: Now I need the population of Paris.', explanation: 'The model reasons about what to do next.' },
          { formula: 'Action: search("population of Paris")', explanation: 'Another search action is taken.' },
          { formula: 'Observation: Paris has a population of ~2.1 million.', explanation: 'Result returned from the environment.' },
        ]}
        id="example-react-trace"
      />

      <h2 className="text-2xl font-semibold">Implementing a ReAct Loop</h2>
      <p className="text-gray-700 dark:text-gray-300">
        A ReAct agent follows a simple loop: prompt the model with the current context,
        parse out any actions, execute them, append the observations, and repeat until
        the model produces a final answer.
      </p>

      <PythonCode
        title="react_loop.py"
        code={`import anthropic
import json

client = anthropic.Anthropic()

# Define available tools
tools = [
    {
        "name": "search",
        "description": "Search the web for current information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    }
]

def execute_tool(name: str, inputs: dict) -> str:
    """Execute a tool and return the result."""
    if name == "search":
        # Simulated search results
        return f"Search results for '{inputs['query']}': [simulated result]"
    elif name == "calculator":
        try:
            result = eval(inputs["expression"])
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"
    return "Unknown tool"

def react_agent(question: str, max_steps: int = 5) -> str:
    """Run a ReAct agent loop."""
    messages = [{"role": "user", "content": question}]
    system = (
        "You are a helpful assistant. Use the provided tools to answer "
        "questions. Think step by step about what you need to find out."
    )

    for step in range(max_steps):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system,
            tools=tools,
            messages=messages,
        )

        # Check if the model wants to use a tool
        if response.stop_reason == "tool_use":
            # Collect all tool uses and results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    print(f"Step {step+1}: {block.name}({block.input})")
                    print(f"  -> {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # Model produced a final answer
            final = next(b.text for b in response.content if b.type == "text")
            print(f"Final answer: {final}")
            return final

    return "Max steps reached without answer"

answer = react_agent("What is 15% of the population of Tokyo?")`}
        id="code-react-loop"
      />

      <NoteBlock
        type="intuition"
        title="ReAct vs Pure CoT vs Pure Acting"
        content="Pure chain-of-thought reasons without access to external information, leading to hallucinations on factual questions. Pure acting (taking tool actions without reasoning) makes suboptimal decisions. ReAct combines both: reasoning grounds the actions, and observations ground the reasoning. This synergy is why ReAct agents outperform either approach alone."
        id="note-react-vs-cot"
      />

      <h2 className="text-2xl font-semibold">ReAct with Claude's Native Tool Use</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Claude's API natively supports the ReAct pattern through its tool_use capability.
        The model automatically interleaves reasoning with tool calls, and you simply
        need to handle tool execution in your application code.
      </p>

      <PythonCode
        title="react_claude_native.py"
        code={`import anthropic

client = anthropic.Anthropic()

# Claude handles ReAct natively through tool_use
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius"
            }
        },
        "required": ["city"]
    }
}]

def get_weather(city: str, units: str = "celsius") -> dict:
    """Simulated weather API."""
    return {"city": city, "temp": 22, "units": units, "condition": "sunny"}

# The conversation loop handles the ReAct pattern
messages = [{"role": "user", "content": "Should I bring an umbrella to Paris today?"}]

while True:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )

    if response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                # Claude's reasoning is implicit in its tool choice
                result = get_weather(**block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })
        messages.append({"role": "user", "content": tool_results})
    else:
        # Final answer with reasoning
        print(next(b.text for b in response.content if b.type == "text"))
        break`}
        id="code-react-native"
      />

      <WarningBlock
        title="Infinite Loops and Runaway Agents"
        content="ReAct agents can get stuck in loops, repeatedly calling the same tool or oscillating between actions. Always set a maximum step count, implement cost budgets, and add loop detection. In production, log every step for debugging and set hard timeouts on the overall agent execution."
        id="warning-react-loops"
      />

      <NoteBlock
        type="historical"
        title="From ReAct to Modern Agents"
        content="ReAct (Yao et al., 2023) unified two previously separate lines of work: reasoning (CoT) and tool-augmented LMs (like Toolformer). The Thought-Action-Observation pattern became the de facto standard for agent architectures. LangChain, Claude's tool_use API, and OpenAI's function calling all implement variants of this pattern."
        id="note-react-history"
      />
    </div>
  )
}
