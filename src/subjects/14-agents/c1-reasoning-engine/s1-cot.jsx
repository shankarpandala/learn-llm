import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ChainOfThought() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Chain-of-Thought Prompting</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Chain-of-Thought (CoT) prompting encourages LLMs to break down complex problems into
        intermediate reasoning steps before arriving at a final answer. This simple technique
        dramatically improves performance on arithmetic, commonsense, and symbolic reasoning tasks.
      </p>

      <DefinitionBlock
        title="Chain-of-Thought Prompting"
        definition="A prompting strategy where the model is instructed to produce a sequence of intermediate reasoning steps, mimicking human step-by-step problem solving, before providing a final answer."
        id="def-cot"
      />

      <h2 className="text-2xl font-semibold">Few-Shot Chain-of-Thought</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The original CoT technique by Wei et al. (2022) provides exemplars that include
        reasoning traces. The model learns to mimic this pattern for new queries.
      </p>

      <PythonCode
        title="few_shot_cot.py"
        code={`import anthropic

client = anthropic.Anthropic()

few_shot_cot_prompt = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Let's think step by step.
Roger started with 5 balls. He bought 2 cans of 3 balls each.
2 cans * 3 balls = 6 balls. 5 + 6 = 11 balls.
The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and
bought 6 more, how many apples do they have?
A: Let's think step by step.
The cafeteria started with 23 apples. They used 20, so 23 - 20 = 3.
They bought 6 more, so 3 + 6 = 9 apples.
The answer is 9.

Q: A store has 48 shirts. They sell 1/3 of them on Monday,
then receive a shipment of 20. How many shirts do they have?
A: Let's think step by step.
"""

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=[{"role": "user", "content": few_shot_cot_prompt}]
)
print(response.content[0].text)
# The store started with 48 shirts. They sold 1/3 of 48 = 16 shirts.
# 48 - 16 = 32. Then received 20 more: 32 + 20 = 52.
# The answer is 52.`}
        id="code-few-shot-cot"
      />

      <h2 className="text-2xl font-semibold">Zero-Shot CoT</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Kojima et al. (2022) discovered that simply appending "Let's think step by step" to a
        prompt activates reasoning without any exemplars. This zero-shot approach is
        surprisingly effective across diverse tasks.
      </p>

      <ExampleBlock
        title="Zero-Shot vs Standard Prompting"
        problem="Compare standard prompting to zero-shot CoT on a multi-step reasoning problem."
        steps={[
          { formula: 'Standard: "What is 17 * 24 + 13?"', explanation: 'The model may guess or make arithmetic errors without showing work.' },
          { formula: 'CoT: "What is 17 * 24 + 13? Let\'s think step by step."', explanation: 'The model decomposes: 17 * 24 = 408, then 408 + 13 = 421.' },
          { formula: 'Accuracy improvement: 17.7% → 78.7% (MultiArith)', explanation: 'Wei et al. showed CoT can improve arithmetic accuracy by 4x on some benchmarks.' },
        ]}
        id="example-zero-shot-cot"
      />

      <PythonCode
        title="zero_shot_cot.py"
        code={`import anthropic

client = anthropic.Anthropic()

def solve_with_cot(question: str) -> str:
    """Use zero-shot CoT to solve a reasoning problem."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"{question}\\n\\nLet's think step by step."
        }]
    )
    return response.content[0].text

# Multi-step word problem
answer = solve_with_cot(
    "A train travels at 60 mph for 2.5 hours, then at 80 mph "
    "for 1.5 hours. What is the total distance traveled?"
)
print(answer)
# Step 1: Distance at 60 mph = 60 * 2.5 = 150 miles
# Step 2: Distance at 80 mph = 80 * 1.5 = 120 miles
# Step 3: Total = 150 + 120 = 270 miles`}
        id="code-zero-shot-cot"
      />

      <NoteBlock
        type="intuition"
        title="Why Does CoT Work?"
        content="CoT works because it allocates more computation to harder problems. Each reasoning token generated gives the model additional forward passes to process information. In effect, CoT converts a single-step mapping into a multi-step computation, letting the model 'think aloud' through intermediate states."
        id="note-why-cot-works"
      />

      <h2 className="text-2xl font-semibold">Self-Consistency</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Self-consistency (Wang et al., 2023) samples multiple CoT reasoning paths and takes
        the majority vote on the final answer. This ensemble approach further improves accuracy
        by reducing variance from any single reasoning chain.
      </p>

      <PythonCode
        title="self_consistency.py"
        code={`import anthropic
from collections import Counter

client = anthropic.Anthropic()

def self_consistency_solve(question: str, n_samples: int = 5) -> str:
    """Sample multiple CoT paths and take majority vote."""
    answers = []
    for _ in range(n_samples):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            temperature=0.7,  # Higher temp for diverse reasoning paths
            messages=[{
                "role": "user",
                "content": (
                    f"{question}\\n\\nThink step by step, then provide "
                    f"your final answer on the last line as: ANSWER: <value>"
                )
            }]
        )
        text = response.content[0].text
        # Extract final answer
        for line in text.strip().split("\\n")[::-1]:
            if "ANSWER:" in line:
                answers.append(line.split("ANSWER:")[-1].strip())
                break

    # Majority vote
    vote = Counter(answers).most_common(1)[0]
    print(f"Votes: {Counter(answers)}")
    print(f"Consensus answer: {vote[0]} ({vote[1]}/{n_samples} votes)")
    return vote[0]

result = self_consistency_solve(
    "If a shirt costs $25 after a 20% discount, what was the original price?"
)
# Votes: Counter({'$31.25': 5})
# Consensus answer: $31.25 (5/5 votes)`}
        id="code-self-consistency"
      />

      <WarningBlock
        title="CoT Limitations"
        content="Chain-of-thought does not guarantee correct reasoning. Models can produce plausible-sounding but incorrect chains. CoT also increases token usage (and therefore cost and latency). For simple tasks, CoT may actually hurt performance by over-thinking. Always validate reasoning chains against known correct answers during development."
        id="warning-cot-limits"
      />

      <NoteBlock
        type="historical"
        title="Key CoT Papers"
        content="Wei et al. (2022) introduced few-shot CoT. Kojima et al. (2022) discovered zero-shot CoT with 'Let's think step by step.' Wang et al. (2023) proposed self-consistency decoding. These techniques form the foundation for all modern agent reasoning, including the extended thinking capabilities in Claude and the chain-of-thought reasoning in OpenAI's o1/o3 models."
        id="note-cot-history"
      />
    </div>
  )
}
