import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function ExtendedThinking() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Extended Thinking and Test-Time Compute</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Extended thinking represents a fundamental shift in how we scale AI capabilities. Rather
        than relying solely on larger models or more training data, test-time compute scaling
        allows models to "think longer" on harder problems. This paradigm, explored by OpenAI o1,
        DeepSeek R1, and Anthropic Claude, suggests that inference-time computation may be as
        important as training-time computation.
      </p>

      <DefinitionBlock
        title="Test-Time Compute"
        definition="Additional computation performed at inference time beyond a single forward pass, used to improve output quality. This includes generating reasoning tokens, search over solution candidates, self-verification, and iterative refinement. The key insight is that performance scales log-linearly with test-time compute: $\text{accuracy} \approx a + b \cdot \log(\text{thinking tokens})$."
        id="def-test-time-compute"
      />

      <h2 className="text-2xl font-semibold">The Scaling Paradigm Shift</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Traditional scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022) focused on training
        compute. The test-time compute paradigm adds a new dimension: a smaller model with more
        inference compute can match a larger model with less. This has profound implications for
        deployment economics and capability elicitation.
      </p>

      <TheoremBlock
        title="Compute-Optimal Inference"
        statement="For a fixed total inference budget B, the optimal strategy allocates compute between model size and thinking tokens. A smaller model thinking for longer can outperform a larger model answering immediately."
        proof={<BlockMath math="B = C_{\text{model}} \cdot (N_{\text{prompt}} + N_{\text{thinking}} + N_{\text{answer}})" />}
        corollaries={[
          'C_model scales with parameter count. N_thinking can be varied per query.',
          'For easy queries: minimize thinking tokens (fast, cheap response).',
          'For hard queries: maximize thinking tokens (slow, expensive, but more accurate).',
          'Optimal policy: adapt thinking budget to problem difficulty.',
        ]}
        id="theorem-compute-optimal"
      />

      <ExampleBlock
        title="Test-Time Compute Strategies"
        problem="Compare different approaches to spending test-time compute."
        steps={[
          { formula: '\\text{Chain-of-Thought}: O(k) \\text{ tokens for } k \\text{ steps}', explanation: 'Linear chain of reasoning. Simple but no error recovery. Used by CoT prompting.' },
          { formula: '\\text{Self-Consistency}: O(n \\cdot k) \\text{ for } n \\text{ chains}', explanation: 'Sample n independent chains and majority-vote. Better accuracy but linear cost increase.' },
          { formula: '\\text{Tree Search}: O(b^d) \\text{ for branching } b, \\text{ depth } d', explanation: 'Explore a tree of reasoning paths with pruning. Used internally by o1/o3 (likely).' },
          { formula: '\\text{Iterative Refinement}: O(r \\cdot k) \\text{ for } r \\text{ rounds}', explanation: 'Generate, critique, and refine. Each round improves quality. Diminishing returns after 3-5 rounds.' },
        ]}
        id="example-ttc-strategies"
      />

      <h2 className="text-2xl font-semibold">Process Reward Models</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Process Reward Models (PRMs) evaluate individual reasoning steps rather than just the final
        answer. This enables more granular credit assignment during RL training and allows search
        algorithms to prune bad reasoning branches early. Lightman et al. (2023) showed that PRMs
        dramatically improve performance when combined with best-of-N sampling.
      </p>

      <PythonCode
        title="test_time_compute_strategies.py"
        code={`import torch
import random
from typing import List, Tuple

# Strategy 1: Best-of-N with reward model scoring
def best_of_n(generate_fn, reward_fn, prompt: str, n: int = 8) -> str:
    """Generate N responses and return the highest-scored one."""
    candidates = [generate_fn(prompt) for _ in range(n)]
    scores = [reward_fn(prompt, c) for c in candidates]
    best_idx = max(range(n), key=lambda i: scores[i])
    return candidates[best_idx], scores[best_idx]

# Strategy 2: Iterative refinement
def iterative_refine(generate_fn, critique_fn, prompt: str, rounds: int = 3) -> str:
    """Generate, critique, and refine iteratively."""
    response = generate_fn(prompt)
    for r in range(rounds):
        critique = critique_fn(prompt, response)
        if "correct" in critique.lower():
            break
        refined_prompt = f"{prompt}\\n\\nPrevious attempt: {response}\\nCritique: {critique}\\nPlease fix the issues."
        response = generate_fn(refined_prompt)
    return response

# Strategy 3: Tree search with process reward model
class ReasoningTree:
    """Simplified tree search for reasoning (conceptual)."""
    def __init__(self, step_generator, step_evaluator, beam_width=3):
        self.generate = step_generator
        self.evaluate = step_evaluator
        self.beam_width = beam_width

    def search(self, problem: str, max_depth: int = 5) -> List[str]:
        beams = [{"steps": [], "score": 0.0, "state": problem}]

        for depth in range(max_depth):
            all_candidates = []
            for beam in beams:
                # Generate candidate next steps
                next_steps = self.generate(beam["state"], n=3)
                for step in next_steps:
                    score = self.evaluate(beam["steps"] + [step])
                    all_candidates.append({
                        "steps": beam["steps"] + [step],
                        "score": beam["score"] + score,
                        "state": beam["state"] + "\\n" + step,
                    })
            # Keep top-k beams
            beams = sorted(all_candidates, key=lambda x: x["score"], reverse=True)[:self.beam_width]

        return beams[0]["steps"]

# Compute budget analysis
def compute_budget_analysis():
    """Analyze how different strategies spend compute."""
    strategies = {
        "Single pass": {"thinking_tokens": 0, "total_passes": 1},
        "CoT": {"thinking_tokens": 500, "total_passes": 1},
        "Self-consistency (8x)": {"thinking_tokens": 500, "total_passes": 8},
        "Best-of-16 + PRM": {"thinking_tokens": 500, "total_passes": 16},
        "Tree search (b=3, d=5)": {"thinking_tokens": 100, "total_passes": 3**5},
    }

    base_cost = 100  # tokens for prompt + answer
    for name, config in strategies.items():
        total_tokens = config["total_passes"] * (base_cost + config["thinking_tokens"])
        relative_cost = total_tokens / base_cost
        print(f"{name:30s}: {total_tokens:>8,} tokens ({relative_cost:>6.1f}x base cost)")

compute_budget_analysis()
# Single pass:                       100 tokens (   1.0x base cost)
# CoT:                               600 tokens (   6.0x base cost)
# Self-consistency (8x):           4,800 tokens (  48.0x base cost)
# Best-of-16 + PRM:                9,600 tokens (  96.0x base cost)
# Tree search (b=3, d=5):         48,600 tokens ( 486.0x base cost)`}
        id="code-ttc"
      />

      <NoteBlock
        type="intuition"
        title="The Inference Scaling Hypothesis"
        content="Training scaling hits diminishing returns: doubling training compute yields small, predictable improvements. But test-time compute scaling may have a different curve. For reasoning tasks, giving a model 10x more thinking time can yield qualitative breakthroughs (e.g., solving a problem it couldn't before). This suggests a future where smaller, efficient models paired with adaptive inference compute may be more practical than ever-larger trained models."
        id="note-inference-scaling"
      />

      <NoteBlock
        type="note"
        title="Adaptive Compute Budget"
        content="Not all queries need the same amount of thinking. 'What is the capital of France?' needs zero thinking tokens, while 'Prove that there are infinitely many primes' benefits from extensive reasoning. Optimal systems should estimate query difficulty and allocate thinking budget accordingly. o3-mini's low/medium/high reasoning effort settings are an early version of this."
        id="note-adaptive-compute"
      />

      <WarningBlock
        title="Diminishing Returns and Overthinking"
        content="Test-time compute scaling is not unlimited. Beyond a certain point, additional thinking tokens provide diminishing returns and can even hurt performance ('overthinking'). The model may second-guess correct answers, introduce unnecessary complexity, or hallucinate issues that don't exist. Knowing when to stop thinking is as important as knowing how to think."
        id="warning-overthinking"
      />
    </div>
  )
}
