import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ParallelSequentialTools() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Parallel vs Sequential Tool Calls</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Modern LLMs can request multiple tool calls in a single response. Understanding
        when tools should be called in parallel versus sequentially is essential for
        building fast and correct agents. Parallel execution reduces latency while
        sequential execution handles data dependencies.
      </p>

      <DefinitionBlock
        title="Parallel Tool Calls"
        definition="When a model requests multiple tool invocations in a single response turn, and those invocations have no data dependencies between them, they can be executed concurrently. This reduces total latency to the duration of the slowest call rather than the sum of all calls."
        id="def-parallel-tools"
      />

      <h2 className="text-2xl font-semibold">Parallel Tool Execution</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Claude can return multiple tool_use blocks in a single response. When the tools
        are independent, you should execute them concurrently for better performance.
      </p>

      <PythonCode
        title="parallel_tool_execution.py"
        code={`import anthropic
import asyncio
import time

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_time",
        "description": "Get current time in a timezone",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string"}
            },
            "required": ["timezone"]
        }
    }
]

def execute_tool(name: str, inputs: dict) -> str:
    """Simulate tool execution with latency."""
    time.sleep(1)  # Simulated API latency
    if name == "get_weather":
        return f"Weather in {inputs['city']}: 22C, sunny"
    elif name == "get_time":
        return f"Time in {inputs['timezone']}: 14:30"
    return "Unknown"

# Model may request multiple tools at once
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    messages=[{
        "role": "user",
        "content": "What's the weather and time in Tokyo and London?"
    }]
)

# Collect all tool_use blocks
tool_calls = [b for b in response.content if b.type == "tool_use"]
print(f"Model requested {len(tool_calls)} tool calls")

# SEQUENTIAL execution (slow): ~4 seconds for 4 calls
start = time.time()
sequential_results = []
for tc in tool_calls:
    result = execute_tool(tc.name, tc.input)
    sequential_results.append({"type": "tool_result", "tool_use_id": tc.id, "content": result})
print(f"Sequential: {time.time() - start:.1f}s")

# PARALLEL execution (fast): ~1 second for 4 calls
import concurrent.futures

start = time.time()
parallel_results = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(execute_tool, tc.name, tc.input): tc
        for tc in tool_calls
    }
    for future in concurrent.futures.as_completed(futures):
        tc = futures[future]
        result = future.result()
        parallel_results.append({
            "type": "tool_result",
            "tool_use_id": tc.id,
            "content": result
        })
print(f"Parallel: {time.time() - start:.1f}s")`}
        id="code-parallel-tools"
      />

      <ExampleBlock
        title="When to Use Parallel vs Sequential"
        problem="Decide execution strategy for different tool call patterns."
        steps={[
          { formula: 'Parallel: get_weather("Tokyo") + get_weather("London")', explanation: 'Independent calls with no data dependency. Safe to parallelize.' },
          { formula: 'Sequential: search("CEO of Acme") → get_profile(ceo_name)', explanation: 'Second call depends on the result of the first. Must be sequential.' },
          { formula: 'Mixed: [get_weather("Tokyo") || get_time("Tokyo")] → format_report()', explanation: 'First two are parallel, then the report depends on both results.' },
        ]}
        id="example-parallel-vs-sequential"
      />

      <h2 className="text-2xl font-semibold">Async Tool Execution</h2>
      <p className="text-gray-700 dark:text-gray-300">
        For production agents handling I/O-bound tool calls (API requests, database queries),
        async execution with asyncio provides the best performance.
      </p>

      <PythonCode
        title="async_tool_execution.py"
        code={`import anthropic
import asyncio

async_client = anthropic.AsyncAnthropic()

async def execute_tool_async(name: str, inputs: dict) -> str:
    """Async tool execution for I/O-bound operations."""
    await asyncio.sleep(1)  # Simulated async API call
    if name == "get_weather":
        return f"Weather in {inputs['city']}: 22C, sunny"
    elif name == "get_time":
        return f"Time in {inputs['timezone']}: 14:30"
    return "Unknown"

async def run_agent(question: str, tools: list, max_turns: int = 5):
    """Async agent loop with parallel tool execution."""
    messages = [{"role": "user", "content": question}]

    for turn in range(max_turns):
        response = await async_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            tool_calls = [b for b in response.content if b.type == "tool_use"]

            # Execute ALL tool calls in parallel
            tasks = [
                execute_tool_async(tc.name, tc.input)
                for tc in tool_calls
            ]
            results = await asyncio.gather(*tasks)

            # Build tool results in the correct order
            tool_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                }
                for tc, result in zip(tool_calls, results)
            ]

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            return next(b.text for b in response.content if b.type == "text")

    return "Max turns reached"

# Run the async agent
# result = asyncio.run(run_agent("Compare weather in 5 cities", tools))`}
        id="code-async-tools"
      />

      <NoteBlock
        type="tip"
        title="Encouraging Parallel Tool Calls"
        content="Models sometimes make sequential tool calls when parallel would be more efficient. You can encourage parallel calls by adding to the system prompt: 'When multiple independent pieces of information are needed, request all relevant tool calls at once rather than one at a time.' Claude tends to naturally parallelize when it recognizes independence."
        id="note-encourage-parallel"
      />

      <WarningBlock
        title="Race Conditions in Parallel Execution"
        content="If parallel tool calls modify shared state (e.g., two database writes to the same record), you may encounter race conditions. Only parallelize truly independent operations. For tools that modify state, consider whether the order of execution matters and add appropriate safeguards."
        id="warning-race-conditions"
      />

      <NoteBlock
        type="note"
        title="Batching and Rate Limits"
        content="When executing many tools in parallel, be mindful of API rate limits on the tool backends. Implement rate limiting, connection pooling, and exponential backoff. For very high parallelism (10+ concurrent calls), consider batching into groups to avoid overwhelming downstream services."
        id="note-rate-limits"
      />
    </div>
  )
}
