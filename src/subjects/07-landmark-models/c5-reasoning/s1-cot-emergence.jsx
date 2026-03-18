import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function CoTEmergence() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Chain-of-Thought: Emergence of Reasoning</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Chain-of-Thought (CoT) prompting (Wei et al., 2022) revealed that large language models
        can perform complex multi-step reasoning when prompted to "think step by step." This
        emergent capability appears only above a critical model size threshold (~100B parameters)
        and fundamentally changed how we elicit reasoning from language models.
      </p>

      <DefinitionBlock
        title="Chain-of-Thought Prompting"
        definition="A prompting technique that includes intermediate reasoning steps in few-shot examples, encouraging the model to generate similar step-by-step reasoning before arriving at a final answer. Formally, instead of learning $P(a \mid q)$ directly, the model generates a reasoning chain: $P(a \mid q) = \sum_r P(a \mid q, r) \cdot P(r \mid q)$ where $r$ represents the chain of reasoning steps."
        id="def-cot"
      />

      <h2 className="text-2xl font-semibold">Emergence and Scale</h2>
      <p className="text-gray-700 dark:text-gray-300">
        CoT reasoning is an emergent ability: it provides no benefit for models below ~10B
        parameters, marginal benefit at 10-60B, and dramatic improvements above 100B. With CoT,
        PaLM 540B solved 58% of GSM8K math problems (vs 18% without CoT), and GPT-3.5 improved
        from 35% to 57% on the same benchmark.
      </p>

      <ExampleBlock
        title="CoT vs Direct Prompting"
        problem="Show how CoT prompting improves math reasoning on a grade-school problem."
        steps={[
          { formula: '\\text{Direct}: Q \\to A', explanation: 'Without CoT: "Roger has 5 tennis balls. He buys 2 cans of 3. How many?" -> "11" (correct but fragile).' },
          { formula: '\\text{CoT}: Q \\to R_1 \\to R_2 \\to \\ldots \\to A', explanation: 'With CoT: "Roger starts with 5 balls. He buys 2 cans * 3 balls = 6 balls. Total = 5 + 6 = 11."' },
          { formula: '\\text{GSM8K accuracy}: 18\\% \\to 58\\% \\text{ (PaLM 540B)}', explanation: 'CoT more than triples accuracy on grade-school math, showing it enables genuine multi-step reasoning.' },
        ]}
        id="example-cot-comparison"
      />

      <h2 className="text-2xl font-semibold">Zero-Shot CoT</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Kojima et al. (2022) showed that simply appending "Let's think step by step" to a prompt
        elicits CoT reasoning without any few-shot examples. This "zero-shot CoT" works across
        diverse reasoning tasks and suggests that the reasoning capability is already present in
        large models -- it just needs to be activated.
      </p>

      <PythonCode
        title="cot_prompting.py"
        code={`from openai import OpenAI

client = OpenAI()

def solve_with_cot(question, model="gpt-4"):
    """Compare direct vs CoT prompting on a math problem."""

    # Direct prompting
    direct_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"{question}\\nAnswer with just the number."},
        ],
        temperature=0,
    )

    # Chain-of-Thought prompting
    cot_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"{question}\\nLet's think step by step."},
        ],
        temperature=0,
    )

    return {
        "direct": direct_response.choices[0].message.content,
        "cot": cot_response.choices[0].message.content,
    }

# Test problems of increasing difficulty
problems = [
    "If a store sells 3 apples for $2 and you buy 12 apples, how much do you pay?",
    "A train travels at 60 mph for 2.5 hours, then 80 mph for 1.5 hours. What's the total distance?",
    "In a room, there are 5 people. Each person shakes hands with every other person exactly once. How many handshakes occur?",
]

for problem in problems:
    result = solve_with_cot(problem)
    print(f"Q: {problem}")
    print(f"Direct: {result['direct'][:80]}")
    print(f"CoT: {result['cot'][:200]}")
    print()

# Self-Consistency: sample multiple CoT paths and take majority vote
def self_consistency(question, model="gpt-4", n_samples=5):
    answers = []
    for _ in range(n_samples):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"{question}\\nThink step by step, then give your final answer as a number."}],
            temperature=0.7,
        )
        answers.append(response.choices[0].message.content)
    return answers

# Self-consistency improves accuracy by ~10-15% over single CoT
results = self_consistency("What is 17 * 23 + 45 - 12?")
print(f"Self-consistency results ({len(results)} samples):")
for i, a in enumerate(results):
    print(f"  Sample {i}: {a[-50:]}")`}
        id="code-cot"
      />

      <NoteBlock
        type="intuition"
        title="Why CoT Works"
        content="A transformer has fixed depth, limiting the computational steps per token. Without CoT, the model must solve the entire problem in a single forward pass. CoT essentially gives the model 'scratch space': each generated token is an additional compute step. For a problem requiring k reasoning steps, CoT converts it from depth-k (impossible for a fixed-depth network) to length-k (feasible through autoregressive generation)."
        id="note-cot-intuition"
      />

      <NoteBlock
        type="note"
        title="Self-Consistency Decoding"
        content="Wang et al. (2023) improved CoT with self-consistency: sample multiple reasoning chains at temperature > 0 and take the majority vote on the final answer. This is based on the insight that correct reasoning paths are more likely to converge on the same answer than incorrect ones. Self-consistency improves GSM8K accuracy by 10-15% over single-sample CoT."
        id="note-self-consistency"
      />

      <WarningBlock
        title="CoT Limitations"
        content="CoT does not guarantee correct reasoning. Models can produce plausible-sounding but logically flawed chains, especially for problems requiring novel reasoning patterns not seen in training. CoT also increases output length (and thus cost and latency). For simple tasks, CoT can actually hurt performance by introducing unnecessary complexity."
        id="warning-cot-limitations"
      />
    </div>
  )
}
