import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function AutonomousAgents() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">AutoGPT and BabyAGI Patterns</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Autonomous agents like AutoGPT and BabyAGI represented early attempts at
        fully self-directed LLM agents. These systems maintain persistent goals,
        create and prioritize tasks, and execute indefinitely with minimal human
        oversight. While often unreliable in practice, they introduced important
        architectural patterns still used in modern agents.
      </p>

      <DefinitionBlock
        title="Autonomous Agent"
        definition="An LLM-powered system that operates with a high-level goal, autonomously generating sub-tasks, executing them using tools, maintaining memory across steps, and iterating until the goal is achieved or a stopping condition is met, with minimal or no human intervention between steps."
        id="def-autonomous-agent"
      />

      <h2 className="text-2xl font-semibold">The BabyAGI Pattern</h2>
      <p className="text-gray-700 dark:text-gray-300">
        BabyAGI introduced a clean three-component architecture: a task creation agent,
        a task prioritization agent, and a task execution agent, all sharing a task queue.
      </p>

      <PythonCode
        title="babyagi_pattern.py"
        code={`import anthropic
import json
from collections import deque

client = anthropic.Anthropic()

class BabyAGI:
    """Simplified BabyAGI-style autonomous agent."""

    def __init__(self, objective: str):
        self.objective = objective
        self.task_queue: deque[dict] = deque()
        self.completed_tasks: list[dict] = []
        self.task_id = 0

    def create_tasks(self, last_result: str) -> list[str]:
        """Generate new tasks based on the objective and recent results."""
        completed = [t["name"] for t in self.completed_tasks[-5:]]
        pending = [t["name"] for t in self.task_queue]

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f"Objective: {self.objective}\\n"
                    f"Recently completed: {completed}\\n"
                    f"Pending tasks: {pending}\\n"
                    f"Last result: {last_result[:300]}\\n\\n"
                    f"Generate 1-3 NEW tasks that would help achieve the "
                    f"objective. Do not duplicate existing tasks.\\n"
                    f"Return as JSON array of strings."
                )
            }]
        )
        text = response.content[0].text
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return []

    def prioritize_tasks(self):
        """Reorder the task queue by priority."""
        if len(self.task_queue) <= 1:
            return
        tasks = [t["name"] for t in self.task_queue]
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    f"Objective: {self.objective}\\n"
                    f"Tasks: {json.dumps(tasks)}\\n\\n"
                    f"Reorder these tasks by priority (most important first). "
                    f"Return as JSON array of strings."
                )
            }]
        )
        text = response.content[0].text
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            ordered = json.loads(text[start:end])
            task_map = {t["name"]: t for t in self.task_queue}
            self.task_queue = deque(
                task_map[name] for name in ordered if name in task_map
            )
        except (ValueError, json.JSONDecodeError):
            pass

    def execute_task(self, task: dict) -> str:
        """Execute a single task."""
        context = "\\n".join(
            f"- {t['name']}: {t['result'][:100]}"
            for t in self.completed_tasks[-3:]
        )
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": (
                    f"Objective: {self.objective}\\n"
                    f"Previous work:\\n{context}\\n\\n"
                    f"Current task: {task['name']}\\n"
                    f"Complete this task thoroughly."
                )
            }]
        )
        return response.content[0].text

    def run(self, initial_task: str, max_iterations: int = 5):
        """Run the autonomous agent loop."""
        self.task_id += 1
        self.task_queue.append({"id": self.task_id, "name": initial_task})

        for i in range(max_iterations):
            if not self.task_queue:
                print("All tasks completed!")
                break

            task = self.task_queue.popleft()
            print(f"\\n--- Iteration {i+1}: {task['name']} ---")

            result = self.execute_task(task)
            task["result"] = result
            self.completed_tasks.append(task)
            print(f"Result: {result[:150]}...")

            new_tasks = self.create_tasks(result)
            for name in new_tasks:
                self.task_id += 1
                self.task_queue.append({"id": self.task_id, "name": name})
                print(f"  New task: {name}")

            self.prioritize_tasks()

# agent = BabyAGI("Create a comprehensive guide to Python async programming")
# agent.run("Outline the main topics for async programming")`}
        id="code-babyagi"
      />

      <ExampleBlock
        title="AutoGPT Architecture"
        problem="What are the key components of the AutoGPT pattern?"
        steps={[
          { formula: 'Goal: A persistent high-level objective', explanation: 'The agent works toward this goal across many iterations.' },
          { formula: 'Memory: Vector store for long-term recall', explanation: 'Results and observations are embedded and stored for later retrieval.' },
          { formula: 'Tools: File I/O, web search, code execution, browsing', explanation: 'A rich set of tools for interacting with the real world.' },
          { formula: 'Self-prompting: The agent generates its own next prompt', explanation: 'Each loop iteration, the agent decides its own next action.' },
        ]}
        id="example-autogpt"
      />

      <NoteBlock
        type="intuition"
        title="Why Autonomous Agents Often Fail"
        content="Autonomous agents suffer from error compounding: each step has a probability of error, and over many steps these errors multiply. A 95% accurate step becomes only 60% reliable over 10 steps (0.95^10). Without human correction, agents drift off-course, get stuck in loops, or pursue irrelevant sub-tasks. This is why modern production agents favor human-in-the-loop designs."
        id="note-why-fail"
      />

      <WarningBlock
        title="Cost and Safety Risks"
        content="Autonomous agents can consume large amounts of API tokens with no guarantee of useful output. AutoGPT-style agents have been observed spending hundreds of dollars on API calls while accomplishing nothing. Always set hard budget limits, step limits, and time limits. Never give autonomous agents access to production systems or real money without human approval gates."
        id="warning-autonomous-cost"
      />

      <NoteBlock
        type="historical"
        title="The 2023 Autonomous Agent Wave"
        content="AutoGPT (March 2023) became the fastest-growing GitHub repo at the time, capturing enormous public interest. BabyAGI followed shortly after with a cleaner architecture. While neither proved reliable for production use, they catalyzed research into agent architectures and demonstrated the potential of LLMs as autonomous reasoners. Modern agents like Claude Code and Cursor use more constrained versions of these patterns with human oversight."
        id="note-history"
      />
    </div>
  )
}
