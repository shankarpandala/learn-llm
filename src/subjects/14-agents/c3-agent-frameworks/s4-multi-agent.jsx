import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function MultiAgent() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Multi-Agent Collaboration</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Multi-agent systems use multiple specialized LLM agents that communicate and
        collaborate to solve complex tasks. Each agent has a distinct role, expertise,
        and set of tools. This mirrors how human teams work: a researcher gathers
        information, a writer produces content, a reviewer provides feedback.
      </p>

      <DefinitionBlock
        title="Multi-Agent System"
        definition="An architecture where multiple LLM-powered agents with distinct roles, instructions, and tools collaborate on a shared task. Agents communicate by passing messages, sharing artifacts, or through a central orchestrator that coordinates their work."
        id="def-multi-agent"
      />

      <h2 className="text-2xl font-semibold">Orchestrator Pattern</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The most common multi-agent pattern uses a central orchestrator agent that
        delegates sub-tasks to specialist agents and synthesizes their results.
      </p>

      <PythonCode
        title="multi_agent_orchestrator.py"
        code={`import anthropic

client = anthropic.Anthropic()

class SpecialistAgent:
    """A specialist agent with a specific role and instructions."""

    def __init__(self, name: str, role: str, instructions: str):
        self.name = name
        self.role = role
        self.instructions = instructions

    def run(self, task: str, context: str = "") -> str:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=f"You are {self.name}, a {self.role}. {self.instructions}",
            messages=[{
                "role": "user",
                "content": f"Context:\\n{context}\\n\\nTask: {task}" if context else task
            }]
        )
        return response.content[0].text

# Define specialist agents
researcher = SpecialistAgent(
    name="Research Agent",
    role="technical researcher",
    instructions=(
        "You gather and synthesize technical information. "
        "Provide detailed, factual analysis with specific examples."
    )
)

writer = SpecialistAgent(
    name="Writer Agent",
    role="technical writer",
    instructions=(
        "You write clear, engaging technical content. "
        "Transform research into well-structured prose."
    )
)

reviewer = SpecialistAgent(
    name="Reviewer Agent",
    role="critical reviewer",
    instructions=(
        "You review technical content for accuracy, clarity, and completeness. "
        "Provide specific, actionable feedback. Be constructive but thorough."
    )
)

class Orchestrator:
    """Coordinates multiple specialist agents."""

    def __init__(self, agents: dict[str, SpecialistAgent]):
        self.agents = agents

    def run_pipeline(self, task: str) -> dict:
        results = {}

        # Step 1: Research
        print("Phase 1: Research...")
        research = self.agents["researcher"].run(
            f"Research the following topic thoroughly: {task}"
        )
        results["research"] = research
        print(f"  Done ({len(research)} chars)")

        # Step 2: Writing
        print("Phase 2: Writing...")
        draft = self.agents["writer"].run(
            f"Write a comprehensive article about: {task}",
            context=f"Research findings:\\n{research}"
        )
        results["draft"] = draft
        print(f"  Done ({len(draft)} chars)")

        # Step 3: Review
        print("Phase 3: Review...")
        review = self.agents["reviewer"].run(
            "Review this article for accuracy and quality.",
            context=f"Article:\\n{draft}"
        )
        results["review"] = review
        print(f"  Done ({len(review)} chars)")

        # Step 4: Revision
        print("Phase 4: Revision...")
        final = self.agents["writer"].run(
            "Revise the article based on this feedback.",
            context=f"Original:\\n{draft}\\n\\nFeedback:\\n{review}"
        )
        results["final"] = final
        print(f"  Done ({len(final)} chars)")

        return results

orchestrator = Orchestrator({
    "researcher": researcher,
    "writer": writer,
    "reviewer": reviewer,
})
# results = orchestrator.run_pipeline("WebAssembly and its impact on web development")`}
        id="code-orchestrator"
      />

      <h2 className="text-2xl font-semibold">Debate Pattern</h2>
      <p className="text-gray-700 dark:text-gray-300">
        In the debate pattern, multiple agents with different perspectives argue
        a point, and a judge agent synthesizes the best arguments. This can
        improve reasoning quality by exploring multiple viewpoints.
      </p>

      <PythonCode
        title="multi_agent_debate.py"
        code={`import anthropic

client = anthropic.Anthropic()

def debate(question: str, rounds: int = 2) -> str:
    """Multi-agent debate for exploring complex questions."""

    agents = [
        {"name": "Optimist", "bias": "Focus on benefits, opportunities, and positive outcomes."},
        {"name": "Skeptic", "bias": "Focus on risks, limitations, and potential problems."},
        {"name": "Pragmatist", "bias": "Focus on practical implications and trade-offs."},
    ]

    debate_history = []

    for round_num in range(1, rounds + 1):
        print(f"\\n=== Round {round_num} ===")
        round_arguments = []

        for agent in agents:
            context = ""
            if debate_history:
                context = "Previous arguments:\\n" + "\\n".join(
                    f"- {a['agent']}: {a['argument'][:200]}..."
                    for a in debate_history[-len(agents):]
                )

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                system=(
                    f"You are the {agent['name']}. {agent['bias']} "
                    f"Provide a concise, well-reasoned argument (150 words max). "
                    f"If responding to others, address their points directly."
                ),
                messages=[{
                    "role": "user",
                    "content": (
                        f"Question: {question}\\n{context}\\n\\n"
                        f"Your argument (Round {round_num}):"
                    )
                }]
            )

            argument = response.content[0].text
            round_arguments.append({"agent": agent["name"], "argument": argument})
            print(f"\\n{agent['name']}: {argument[:150]}...")

        debate_history.extend(round_arguments)

    # Judge synthesizes
    all_arguments = "\\n\\n".join(
        f"{a['agent']}: {a['argument']}" for a in debate_history
    )

    synthesis = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="You are an impartial judge. Synthesize the debate into a balanced conclusion.",
        messages=[{
            "role": "user",
            "content": f"Question: {question}\\n\\nDebate:\\n{all_arguments}\\n\\nSynthesis:"
        }]
    )

    return synthesis.content[0].text

# result = debate("Should companies adopt AI coding assistants for all developers?")`}
        id="code-debate"
      />

      <ExampleBlock
        title="Multi-Agent Topologies"
        problem="What are the common patterns for organizing multiple agents?"
        steps={[
          { formula: 'Pipeline: Agent A → Agent B → Agent C', explanation: 'Sequential chain where each agent processes the output of the previous one.' },
          { formula: 'Hub-and-spoke: Orchestrator delegates to specialists', explanation: 'Central coordinator assigns tasks and collects results.' },
          { formula: 'Debate/consensus: Agents argue, judge decides', explanation: 'Multiple perspectives improve quality on subjective or complex questions.' },
          { formula: 'Hierarchical: Manager agents supervise worker agents', explanation: 'Multi-level delegation for complex projects with many sub-tasks.' },
        ]}
        id="example-topologies"
      />

      <NoteBlock
        type="tip"
        title="When Multi-Agent Is Worth the Complexity"
        content="Multi-agent systems add latency, cost, and complexity. Use them when: (1) the task genuinely requires different types of expertise, (2) quality improves measurably vs a single agent, (3) you need built-in checks and balances (reviewer agent catches writer errors). For most tasks, a well-prompted single agent with good tools outperforms a poorly designed multi-agent system."
        id="note-when-multi-agent"
      />

      <WarningBlock
        title="Coordination Overhead"
        content="Multi-agent systems multiply API costs (each agent makes its own LLM calls) and introduce coordination challenges. Agents may produce contradictory outputs, lose context across handoffs, or enter infinite feedback loops (reviewer keeps finding issues, writer keeps revising). Set clear stopping conditions and monitor total token usage across all agents."
        id="warning-coordination"
      />
    </div>
  )
}
