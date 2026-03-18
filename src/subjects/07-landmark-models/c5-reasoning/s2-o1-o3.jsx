import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function O1O3Reasoning() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">OpenAI o1 and o3: Trained Reasoning</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        OpenAI's o1 (September 2024) and o3 (December 2024) models represent a paradigm shift:
        instead of just prompting for chain-of-thought, these models are explicitly trained to
        reason through reinforcement learning. They generate internal "thinking" tokens before
        responding, achieving breakthroughs on mathematics, coding, and science benchmarks that
        seemed out of reach for standard LLMs.
      </p>

      <DefinitionBlock
        title="Test-Time Compute Scaling"
        definition="The principle that model performance can be improved by allocating more computation at inference time rather than during training. Instead of the traditional paradigm where performance scales with training compute, reasoning models scale with inference compute: $\text{Performance} \propto f(\text{thinking tokens})$. More thinking time yields better answers."
        id="def-test-time-compute"
      />

      <h2 className="text-2xl font-semibold">How o1 Works</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The o1 model generates a hidden chain-of-thought (CoT) before producing its visible
        response. This internal reasoning process is trained using large-scale reinforcement
        learning, where the model learns to break problems into steps, verify intermediate
        results, backtrack when encountering errors, and try alternative approaches.
      </p>

      <ExampleBlock
        title="o1/o3 Benchmark Results"
        problem="Compare o1 and o3 performance against GPT-4 on key reasoning benchmarks."
        steps={[
          { formula: '\\text{AIME 2024 (math)}: \\text{GPT-4} = 12\\% \\to \\text{o1} = 83\\% \\to \\text{o3} = 96\\%', explanation: 'AIME is a competitive math exam. o3 solves nearly all problems, approaching human expert level.' },
          { formula: '\\text{Codeforces (coding)}: \\text{GPT-4} = 11\\text{th pctile} \\to \\text{o1} = 89\\text{th pctile}', explanation: 'o1 reaches competitive programmer level on Codeforces rating.' },
          { formula: '\\text{GPQA Diamond (PhD science)}: \\text{GPT-4} = 56\\% \\to \\text{o1} = 78\\%', explanation: 'PhD-level science questions where domain experts achieve ~70%. o1 surpasses experts.' },
          { formula: '\\text{ARC-AGI}: \\text{GPT-4} = 5\\% \\to \\text{o3 (high)} = 88\\%', explanation: 'Abstract reasoning benchmark designed to test generalization. o3 with high compute achieves 88%.' },
        ]}
        id="example-o1-benchmarks"
      />

      <h2 className="text-2xl font-semibold">Training Methodology</h2>
      <p className="text-gray-700 dark:text-gray-300">
        While exact details are proprietary, o1/o3 likely use a process reward model (PRM) that
        evaluates each step of reasoning, combined with RL training (potentially MCTS-like search
        during training) to learn effective reasoning strategies. The model learns to generate
        productive thinking patterns: decomposition, verification, backtracking, and analogical
        reasoning.
      </p>

      <PythonCode
        title="o1_api_usage.py"
        code={`from openai import OpenAI

client = OpenAI()

# o1 models use a different API pattern
# No system message, no temperature control (always greedy)
response = client.chat.completions.create(
    model="o1",
    messages=[
        {"role": "user", "content": """
Solve this step by step:
Find all integer solutions to the equation:
x^3 + y^3 + z^3 = 42
where |x|, |y|, |z| <= 10^16
"""},
    ],
    # o1 models don't support temperature, top_p, or system messages
    # They internally generate reasoning tokens before responding
)

print(response.choices[0].message.content)
print(f"\\nCompletion tokens: {response.usage.completion_tokens}")
print(f"Reasoning tokens: {response.usage.completion_tokens_details.reasoning_tokens}")
print(f"Visible tokens: {response.usage.completion_tokens - response.usage.completion_tokens_details.reasoning_tokens}")

# Compare reasoning effort levels (o3-mini)
for effort in ["low", "medium", "high"]:
    response = client.chat.completions.create(
        model="o3-mini",
        messages=[{"role": "user", "content": "What is the 100th prime number?"}],
        reasoning_effort=effort,
    )
    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
    print(f"\\n{effort:6s} effort: {reasoning_tokens} reasoning tokens, "
          f"answer: {response.choices[0].message.content[:50]}")

# Process reward model concept
# The model internally evaluates each reasoning step
def conceptual_prm_scoring(steps):
    """Illustrates how a process reward model scores reasoning steps."""
    scores = {
        "problem decomposition": 0.95,
        "correct formula application": 0.90,
        "arithmetic check": 0.85,
        "verification of answer": 0.92,
        "final answer": 0.88,
    }
    for step, score in scores.items():
        print(f"  Step: {step:30s} -> PRM score: {score:.2f}")
    return min(scores.values())

print("\\nConceptual PRM scoring:")
min_score = conceptual_prm_scoring(None)
print(f"  Minimum step score: {min_score:.2f} (bottleneck for chain reliability)")`}
        id="code-o1-api"
      />

      <NoteBlock
        type="intuition"
        title="Why RL for Reasoning?"
        content="Standard next-token prediction (SFT) only teaches the model to imitate reasoning traces. RL lets the model discover novel reasoning strategies by rewarding correct final answers. The model can learn that backtracking (going back and trying a different approach) is valuable, even though backtracking rarely appears in human-written text. RL optimizes for outcomes, not imitation."
        id="note-rl-reasoning"
      />

      <NoteBlock
        type="note"
        title="Hidden Reasoning Tokens"
        content="o1's internal reasoning tokens are not shown to the user (only a summary is provided). This serves multiple purposes: it protects proprietary reasoning patterns, reduces the visible output length, and prevents users from becoming confused by the model's internal deliberation process. The reasoning tokens do count toward pricing."
        id="note-hidden-reasoning"
      />

      <WarningBlock
        title="Cost and Latency"
        content="o1 generates thousands of reasoning tokens before responding, making it 10-50x more expensive and slower than GPT-4o for simple tasks. Use o1/o3 only for genuinely difficult reasoning problems (math, code, logic). For simple Q&A, summarization, or creative writing, standard models are more cost-effective and often faster."
        id="warning-o1-cost"
      />
    </div>
  )
}
