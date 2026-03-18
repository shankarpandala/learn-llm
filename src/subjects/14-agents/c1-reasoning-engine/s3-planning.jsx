import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Planning() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Task Decomposition and Plan-and-Execute</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Complex tasks often require breaking a high-level goal into smaller, manageable
        sub-tasks before execution. Plan-and-Execute agents separate the planning phase
        from the execution phase, leading to more structured and reliable agent behavior.
      </p>

      <DefinitionBlock
        title="Task Decomposition"
        definition="The process of breaking a complex task into a sequence of simpler sub-tasks that can be individually solved. In agent systems, an LLM generates a plan (ordered list of steps), then executes each step, potentially replanning as new information emerges."
        id="def-task-decomposition"
      />

      <h2 className="text-2xl font-semibold">Plan-and-Execute Architecture</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Unlike ReAct which interleaves thinking and acting at each step, Plan-and-Execute
        first creates a complete plan, then executes it step by step. This mirrors how
        humans approach complex projects: plan first, then do.
      </p>

      <ExampleBlock
        title="Plan-and-Execute for Research"
        problem="Task: Write a comparison of React vs Vue for a technical blog post."
        steps={[
          { formula: 'Plan Step 1: Research React features and recent updates', explanation: 'Identify the sub-tasks needed to complete the goal.' },
          { formula: 'Plan Step 2: Research Vue features and recent updates', explanation: 'Each step is specific and actionable.' },
          { formula: 'Plan Step 3: Identify key comparison dimensions', explanation: 'Performance, ecosystem, learning curve, etc.' },
          { formula: 'Plan Step 4: Write the comparison with evidence', explanation: 'Synthesize findings into the final output.' },
          { formula: 'Execute: Run each step sequentially, passing context forward', explanation: 'Each step builds on the results of previous steps.' },
        ]}
        id="example-plan-execute"
      />

      <PythonCode
        title="plan_and_execute.py"
        code={`import anthropic
import json

client = anthropic.Anthropic()

def create_plan(task: str) -> list[str]:
    """Use the LLM to decompose a task into steps."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Break this task into 3-6 concrete, sequential steps.\\n"
                f"Task: {task}\\n\\n"
                f"Return a JSON array of step descriptions.\\n"
                f"Example: [\\"step 1 description\\", \\"step 2 description\\"]"
            )
        }]
    )
    text = response.content[0].text
    # Extract JSON from response
    start = text.index("[")
    end = text.rindex("]") + 1
    return json.loads(text[start:end])

def execute_step(step: str, context: str) -> str:
    """Execute a single plan step with accumulated context."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                f"Previous context:\\n{context}\\n\\n"
                f"Current step: {step}\\n\\n"
                f"Complete this step thoroughly."
            )
        }]
    )
    return response.content[0].text

def plan_and_execute(task: str) -> str:
    """Full plan-and-execute agent."""
    print(f"Task: {task}\\n")

    # Phase 1: Planning
    plan = create_plan(task)
    print("Plan:")
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step}")

    # Phase 2: Execution
    context = ""
    for i, step in enumerate(plan, 1):
        print(f"\\nExecuting step {i}/{len(plan)}: {step}")
        result = execute_step(step, context)
        context += f"\\n## Step {i}: {step}\\n{result}\\n"
        print(f"  Done. ({len(result)} chars)")

    return context

result = plan_and_execute(
    "Analyze the trade-offs between SQL and NoSQL databases "
    "for a real-time analytics application"
)`}
        id="code-plan-execute"
      />

      <h2 className="text-2xl font-semibold">Adaptive Replanning</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Static plans often need adjustment as execution reveals new information or
        unexpected obstacles. Adaptive agents can replan mid-execution, combining
        the structure of planning with the flexibility of reactive approaches.
      </p>

      <PythonCode
        title="adaptive_replanning.py"
        code={`import anthropic
import json

client = anthropic.Anthropic()

def should_replan(step_result: str, remaining_plan: list[str], goal: str) -> list[str] | None:
    """Ask the LLM if the plan needs adjustment based on new information."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Goal: {goal}\\n"
                f"Latest step result: {step_result[:500]}\\n"
                f"Remaining plan: {json.dumps(remaining_plan)}\\n\\n"
                f"Should the remaining plan be adjusted? If yes, return a new "
                f"JSON array of steps. If no, return exactly: NO_CHANGE"
            )
        }]
    )
    text = response.content[0].text.strip()
    if "NO_CHANGE" in text:
        return None
    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None

def adaptive_plan_execute(task: str, max_replans: int = 3) -> str:
    """Plan-and-execute with adaptive replanning."""
    plan = create_plan(task)  # From previous example
    context = ""
    replans = 0

    i = 0
    while i < len(plan):
        step = plan[i]
        print(f"\\nStep {i+1}/{len(plan)}: {step}")
        result = execute_step(step, context)
        context += f"\\nStep: {step}\\nResult: {result}\\n"

        # Check if replanning is needed
        remaining = plan[i+1:]
        if remaining and replans < max_replans:
            new_plan = should_replan(result, remaining, task)
            if new_plan is not None:
                print(f"  Replanning! New remaining steps: {new_plan}")
                plan = plan[:i+1] + new_plan
                replans += 1
        i += 1

    return context`}
        id="code-adaptive-replan"
      />

      <NoteBlock
        type="tip"
        title="Planning Granularity"
        content="Plans that are too detailed become brittle and hard to adjust. Plans that are too vague provide no useful structure. Aim for 3-7 steps where each step represents a meaningful unit of work. Let the executor handle the details within each step."
        id="note-planning-granularity"
      />

      <WarningBlock
        title="Planning Hallucination"
        content="LLMs can generate plans that sound reasonable but contain impossible or nonsensical steps. Always validate plans against the agent's actual capabilities (available tools, permissions, API access). A plan step like 'access the production database' is useless if the agent has no database tool."
        id="warning-plan-hallucination"
      />

      <NoteBlock
        type="historical"
        title="Plan-and-Execute in AI History"
        content="Hierarchical task planning has deep roots in classical AI (STRIPS, 1971; HTN planners). The Plan-and-Execute pattern in LLM agents was popularized by frameworks like LangChain's PlanAndExecute agent and BabyAGI. Modern agents like Devin and Claude Code use sophisticated variants that continuously replan based on execution feedback."
        id="note-planning-history"
      />
    </div>
  )
}
