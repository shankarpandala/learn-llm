import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function CostOptimization() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Cost Optimization Strategies</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        LLM inference costs can escalate quickly, whether you are paying per-token to a cloud
        API or operating your own GPU fleet. Effective cost optimization combines prompt
        engineering, caching, model selection, and infrastructure tuning to reduce spend while
        maintaining output quality.
      </p>

      <DefinitionBlock
        title="Cost Per Token"
        definition="LLM API costs are measured in dollars per million tokens, with input (prompt) tokens typically 3-10x cheaper than output (completion) tokens. For self-hosted models, cost is GPU-hours divided by tokens served. A well-optimized self-hosted deployment can cost $0.01-0.05 per million tokens compared to $0.50-15.00 for cloud APIs."
        id="def-cost-per-token"
      />

      <ExampleBlock
        title="Cost Optimization Strategies"
        problem="How can you reduce LLM inference costs?"
        steps={[
          { formula: 'Prompt caching: reuse common prefixes', explanation: 'Cached prompt tokens cost 50-90% less. System prompts and few-shot examples are prime candidates.' },
          { formula: 'Model routing: use smaller models for easy tasks', explanation: 'Route simple queries to a 7B model and only use 70B+ for complex reasoning.' },
          { formula: 'Output length control: set tight max_tokens', explanation: 'Every unnecessary output token costs money. Instruct the model to be concise.' },
          { formula: 'Semantic caching: cache similar query results', explanation: 'If a nearly identical question was recently answered, return the cached response.' },
          { formula: 'Batch processing: use offline batch APIs', explanation: 'OpenAI Batch API and similar offer 50% discounts for non-real-time workloads.' },
        ]}
        id="example-strategies"
      />

      <PythonCode
        title="cost_tracker.py"
        code={`import time
from dataclasses import dataclass, field
from collections import defaultdict

# Pricing per million tokens (example rates)
PRICING = {
    "gpt-4o":       {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":  {"input": 0.15, "output": 0.60},
    "claude-sonnet": {"input": 3.00, "output": 15.00},
    "llama-3.1-8b": {"input": 0.05, "output": 0.05},  # Self-hosted estimate
    "llama-3.1-70b":{"input": 0.20, "output": 0.20},  # Self-hosted estimate
}

@dataclass
class CostTracker:
    costs: list = field(default_factory=list)
    by_model: dict = field(default_factory=lambda: defaultdict(float))

    def record(self, model, input_tokens, output_tokens):
        pricing = PRICING.get(model, {"input": 1.0, "output": 3.0})
        cost = (input_tokens * pricing["input"] +
                output_tokens * pricing["output"]) / 1_000_000
        self.costs.append({"model": model, "cost": cost, "time": time.time()})
        self.by_model[model] += cost
        return cost

    def total(self):
        return sum(c["cost"] for c in self.costs)

    def report(self):
        print(f"Total cost: USD {self.total():.4f}")
        print(f"Requests: {len(self.costs)}")
        for model, cost in sorted(self.by_model.items(), key=lambda x: -x[1]):
            print(f"  {model}: USD {cost:.4f}")

tracker = CostTracker()

# Simulate a day of usage
import random
for _ in range(1000):
    model = random.choice(["gpt-4o", "gpt-4o-mini", "llama-3.1-8b"])
    tracker.record(model, random.randint(100, 2000), random.randint(50, 500))

tracker.report()`}
        id="code-cost-tracker"
      />

      <PythonCode
        title="model_router.py"
        code={`from openai import OpenAI

# Route queries to the cheapest capable model
class ModelRouter:
    def __init__(self, clients):
        self.clients = clients  # model_name -> OpenAI client

    def classify_complexity(self, messages):
        """Simple heuristic: short queries -> small model."""
        last_msg = messages[-1]["content"]
        word_count = len(last_msg.split())
        # Complex indicators: code, math, multi-step reasoning
        complex_words = ["analyze", "compare", "implement", "debug",
                         "explain why", "step by step", "code"]
        is_complex = any(w in last_msg.lower() for w in complex_words)
        if word_count > 200 or is_complex:
            return "complex"
        return "simple"

    def route(self, messages, **kwargs):
        complexity = self.classify_complexity(messages)
        if complexity == "simple":
            model, client = "llama-3.1-8b", self.clients["small"]
        else:
            model, client = "gpt-4o", self.clients["large"]

        print(f"Routing to {model} (complexity={complexity})")
        return client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

# Setup
router = ModelRouter({
    "small": OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY"),
    "large": OpenAI(api_key="sk-your-key"),
})

# Simple query -> routed to cheap local model
resp = router.route(
    [{"role": "user", "content": "What is 2+2?"}],
    max_tokens=50,
)
print(resp.choices[0].message.content)

# Complex query -> routed to powerful cloud model
resp = router.route(
    [{"role": "user", "content": "Analyze this code and explain the bug step by step: ..."}],
    max_tokens=500,
)
print(resp.choices[0].message.content)`}
        id="code-model-router"
      />

      <NoteBlock
        type="tip"
        title="Semantic Caching"
        content="Tools like GPTCache or Redis with vector search can cache LLM responses keyed by semantic similarity rather than exact match. When a new query is sufficiently similar to a cached one (cosine similarity > 0.95), return the cached response instantly at zero token cost. This works exceptionally well for FAQ-style workloads."
        id="note-semantic-cache"
      />

      <WarningBlock
        title="Quality vs Cost Tradeoff"
        content="Aggressive cost optimization can degrade output quality. Always measure quality metrics (accuracy, helpfulness ratings, task success rate) alongside cost. A model that costs 10x less but fails 30% of the time may actually cost more when you factor in retries, user frustration, and downstream errors."
        id="warning-quality"
      />
    </div>
  )
}
