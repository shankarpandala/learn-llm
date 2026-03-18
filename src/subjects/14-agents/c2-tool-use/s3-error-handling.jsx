import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ErrorHandling() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Tool Error Recovery</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Tools fail. APIs timeout, databases go down, inputs are malformed, and
        permissions get denied. Robust agents must gracefully handle tool errors,
        retry when appropriate, and communicate failures clearly rather than
        crashing or hallucinating results.
      </p>

      <DefinitionBlock
        title="Tool Error Recovery"
        definition="The set of strategies an agent uses when a tool call fails, including returning error information to the model, retrying with modified inputs, falling back to alternative tools, or gracefully degrading the response to inform the user of the limitation."
        id="def-error-recovery"
      />

      <h2 className="text-2xl font-semibold">Returning Errors to the Model</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The most important principle: always return error information to the model
        rather than crashing. Claude can reason about errors and adapt its approach.
      </p>

      <PythonCode
        title="error_returning.py"
        code={`import anthropic
import json
import traceback

client = anthropic.Anthropic()

def execute_tool_safely(name: str, inputs: dict) -> dict:
    """Execute a tool with comprehensive error handling."""
    try:
        if name == "query_database":
            result = query_db(inputs["sql"])
            return {"type": "tool_result", "content": json.dumps(result)}

        elif name == "fetch_url":
            result = fetch(inputs["url"])
            return {"type": "tool_result", "content": result}

        else:
            return {
                "type": "tool_result",
                "content": f"Error: Unknown tool '{name}'",
                "is_error": True  # Claude API supports this flag
            }

    except TimeoutError:
        return {
            "type": "tool_result",
            "content": (
                f"Error: Tool '{name}' timed out after 30 seconds. "
                f"The service may be temporarily unavailable. "
                f"You can retry or try a different approach."
            ),
            "is_error": True
        }
    except PermissionError as e:
        return {
            "type": "tool_result",
            "content": f"Error: Permission denied - {e}. You do not have access to this resource.",
            "is_error": True
        }
    except Exception as e:
        return {
            "type": "tool_result",
            "content": (
                f"Error executing {name}: {type(e).__name__}: {str(e)}. "
                f"Please try a different approach or inform the user."
            ),
            "is_error": True
        }

def agent_loop_with_errors(question: str, tools: list, max_turns: int = 10):
    """Agent loop that handles tool errors gracefully."""
    messages = [{"role": "user", "content": question}]

    for turn in range(max_turns):
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
                    result = execute_tool_safely(block.name, block.input)
                    result["tool_use_id"] = block.id
                    tool_results.append(result)
            messages.append({"role": "user", "content": tool_results})
        else:
            return next(b.text for b in response.content if b.type == "text")`}
        id="code-error-returning"
      />

      <ExampleBlock
        title="Error Recovery Strategies"
        problem="What should an agent do when a tool call fails?"
        steps={[
          { formula: 'Strategy 1: Retry with same inputs', explanation: 'For transient errors like timeouts or rate limits. Add exponential backoff.' },
          { formula: 'Strategy 2: Retry with modified inputs', explanation: 'If the error suggests bad input (e.g., invalid date format), fix and retry.' },
          { formula: 'Strategy 3: Fall back to alternative tool', explanation: 'If web search fails, try a different search API or cached data.' },
          { formula: 'Strategy 4: Inform the user gracefully', explanation: 'When no recovery is possible, explain what failed and what they can do.' },
        ]}
        id="example-recovery-strategies"
      />

      <PythonCode
        title="retry_with_backoff.py"
        code={`import anthropic
import time
import random

client = anthropic.Anthropic()

def execute_with_retry(
    name: str,
    inputs: dict,
    max_retries: int = 3,
    base_delay: float = 1.0
) -> dict:
    """Execute a tool with exponential backoff retry."""
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            result = execute_tool(name, inputs)  # Your tool executor
            return {"content": result}
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"  Retry {attempt+1}/{max_retries} in {delay:.1f}s: {e}")
                time.sleep(delay)

    # All retries exhausted
    return {
        "content": (
            f"Error after {max_retries} retries: {last_error}. "
            f"The service appears to be down."
        ),
        "is_error": True
    }

def execute_with_fallback(name: str, inputs: dict, fallbacks: list) -> dict:
    """Try primary tool, then fall back to alternatives."""
    tools_to_try = [name] + fallbacks

    for tool_name in tools_to_try:
        try:
            result = execute_tool(tool_name, inputs)
            if tool_name != name:
                result = f"[via {tool_name}] {result}"
            return {"content": result}
        except Exception as e:
            print(f"  {tool_name} failed: {e}")
            continue

    return {
        "content": f"All tools failed: {name} and fallbacks {fallbacks}",
        "is_error": True
    }

# Example: search with fallback
result = execute_with_fallback(
    "web_search",
    {"query": "latest AI news"},
    fallbacks=["cached_search", "knowledge_base_search"]
)`}
        id="code-retry-fallback"
      />

      <NoteBlock
        type="tip"
        title="The is_error Flag"
        content="Claude's API supports an 'is_error' field in tool results. When set to True, it signals to Claude that the tool call failed. This helps the model distinguish between an error message that should trigger recovery and a successful result that happens to contain the word 'error'."
        id="note-is-error-flag"
      />

      <WarningBlock
        title="Retry Loops and Cost"
        content="Unbounded retries can lead to runaway API costs and infinite loops. Always set a maximum retry count and a total cost/time budget for the agent. Log all tool errors for monitoring. If a tool consistently fails, it may indicate a systemic issue rather than a transient error."
        id="warning-retry-cost"
      />

      <NoteBlock
        type="note"
        title="Input Validation Before Execution"
        content="Many tool errors can be prevented by validating inputs before execution. Validate types, check required fields, enforce constraints (string length, numeric ranges), and sanitize inputs to prevent injection attacks. This is especially important for tools that execute code or SQL queries."
        id="note-input-validation"
      />
    </div>
  )
}
