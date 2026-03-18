import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function SelfReflection() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Reflexion and Self-Critique</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Self-reflection enables agents to evaluate their own outputs, identify mistakes,
        and iteratively improve. Reflexion (Shinn et al., 2023) formalizes this as a loop
        where an agent attempts a task, reflects on its performance, and retries with
        accumulated lessons learned stored in memory.
      </p>

      <DefinitionBlock
        title="Reflexion"
        definition="An agent architecture where the model generates an output, evaluates it against criteria (self-critique), produces a verbal reflection summarizing what went wrong, stores this reflection in memory, and uses it to improve on the next attempt."
        id="def-reflexion"
      />

      <ExampleBlock
        title="Reflexion Loop"
        problem="An agent is asked to write a Python function that passes unit tests but fails on the first attempt."
        steps={[
          { formula: 'Attempt 1: Generate function → Run tests → 2/5 pass', explanation: 'Initial attempt has bugs.' },
          { formula: 'Reflect: "I missed edge cases for empty lists and negative numbers"', explanation: 'The agent critiques its own output.' },
          { formula: 'Attempt 2: Generate improved function using reflection → 4/5 pass', explanation: 'Reflection guides the improvement.' },
          { formula: 'Reflect: "Off-by-one error in the loop boundary"', explanation: 'More specific critique on remaining failure.' },
          { formula: 'Attempt 3: Final fix → 5/5 pass', explanation: 'Iterative refinement converges to a correct solution.' },
        ]}
        id="example-reflexion"
      />

      <PythonCode
        title="reflexion_agent.py"
        code={`import anthropic

client = anthropic.Anthropic()

def generate_solution(task: str, reflections: list[str]) -> str:
    """Generate a solution, incorporating past reflections."""
    reflection_context = ""
    if reflections:
        reflection_context = "\\nPrevious reflections (learn from these):\\n"
        for i, r in enumerate(reflections, 1):
            reflection_context += f"{i}. {r}\\n"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                f"Task: {task}\\n"
                f"{reflection_context}\\n"
                f"Provide your solution:"
            )
        }]
    )
    return response.content[0].text

def evaluate_solution(task: str, solution: str) -> dict:
    """Evaluate a solution and return pass/fail with feedback."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Task: {task}\\n"
                f"Solution:\\n{solution}\\n\\n"
                f"Evaluate this solution. Is it correct and complete?\\n"
                f"Respond with JSON: {{\\"passed\\": bool, \\"feedback\\": str}}"
            )
        }]
    )
    import json
    text = response.content[0].text
    start = text.index("{")
    end = text.rindex("}") + 1
    return json.loads(text[start:end])

def reflect(task: str, solution: str, feedback: str) -> str:
    """Generate a reflection on what went wrong."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                f"Task: {task}\\n"
                f"Your solution:\\n{solution}\\n"
                f"Feedback: {feedback}\\n\\n"
                f"Reflect on what went wrong and what to do differently. "
                f"Be specific and actionable in 2-3 sentences."
            )
        }]
    )
    return response.content[0].text

def reflexion_loop(task: str, max_attempts: int = 3) -> str:
    """Run the full Reflexion loop."""
    reflections = []

    for attempt in range(1, max_attempts + 1):
        print(f"\\n--- Attempt {attempt} ---")
        solution = generate_solution(task, reflections)
        print(f"Solution generated ({len(solution)} chars)")

        evaluation = evaluate_solution(task, solution)
        print(f"Evaluation: {'PASS' if evaluation['passed'] else 'FAIL'}")
        print(f"Feedback: {evaluation['feedback']}")

        if evaluation["passed"]:
            return solution

        # Reflect and store for next attempt
        reflection = reflect(task, solution, evaluation["feedback"])
        reflections.append(reflection)
        print(f"Reflection: {reflection}")

    return solution  # Return best effort

result = reflexion_loop(
    "Write a Python function that finds the longest palindromic "
    "substring in a given string. Handle edge cases."
)`}
        id="code-reflexion"
      />

      <h2 className="text-2xl font-semibold">Self-Critique Patterns</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Self-critique can be applied without the full Reflexion loop. A simple pattern
        is to generate output, then ask the model to critique it, then revise.
      </p>

      <PythonCode
        title="self_critique.py"
        code={`import anthropic

client = anthropic.Anthropic()

def generate_and_critique(task: str) -> str:
    """Generate, critique, and revise in a single flow."""

    # Step 1: Initial generation
    draft = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": task}]
    ).content[0].text

    # Step 2: Self-critique
    critique = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Critique the following response for accuracy, completeness, "
                f"and clarity. List specific issues.\\n\\n"
                f"Task: {task}\\n\\nResponse:\\n{draft}"
            )
        }]
    ).content[0].text

    # Step 3: Revise based on critique
    revised = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                f"Original task: {task}\\n\\n"
                f"Draft response:\\n{draft}\\n\\n"
                f"Critique:\\n{critique}\\n\\n"
                f"Provide an improved response addressing the critique."
            )
        }]
    ).content[0].text

    return revised

# Constitutional AI-style self-critique
def constitutional_critique(response: str, principles: list[str]) -> str:
    """Critique against explicit principles (Constitutional AI pattern)."""
    principles_text = "\\n".join(f"- {p}" for p in principles)

    critique = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Evaluate this response against these principles:\\n"
                f"{principles_text}\\n\\nResponse:\\n{response}\\n\\n"
                f"For each principle, note if it is satisfied or violated."
            )
        }]
    ).content[0].text
    return critique

principles = [
    "The response is factually accurate",
    "The response does not make unsupported claims",
    "The response acknowledges uncertainty where appropriate",
    "The response is helpful and actionable",
]`}
        id="code-self-critique"
      />

      <NoteBlock
        type="intuition"
        title="Why Self-Reflection Works"
        content="LLMs are often better at evaluating solutions than generating them on the first try. This is analogous to how it is easier to spot a bug in code review than to write bug-free code. Self-reflection exploits this asymmetry by using the model's evaluation capability to guide its generation capability."
        id="note-reflection-intuition"
      />

      <WarningBlock
        title="Reflection Can Be Wrong"
        content="Models can generate confident but incorrect reflections, leading subsequent attempts further astray. This is especially dangerous when the model lacks knowledge needed to evaluate correctness. Use external validation (tests, type checkers, search) rather than relying solely on self-assessment when possible."
        id="warning-bad-reflection"
      />

      <NoteBlock
        type="note"
        title="Reflexion vs Fine-Tuning"
        content="Reflexion stores lessons as text in the context window (episodic memory). This is distinct from fine-tuning, which updates model weights. Reflexion is immediate and requires no training, but lessons are lost when the context resets. For persistent improvement, combine Reflexion with a long-term memory store or use the reflections as training data."
        id="note-reflexion-vs-finetuning"
      />
    </div>
  )
}
