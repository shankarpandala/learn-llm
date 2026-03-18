import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function AbTesting() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">A/B Testing Finetuned Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        When you have multiple finetuned model candidates, A/B testing provides a systematic way
        to determine which performs better on real-world tasks. This section covers both offline
        comparison and online A/B testing strategies.
      </p>

      <DefinitionBlock
        title="A/B Testing for LLMs"
        definition="A/B testing for LLMs compares two or more model variants on the same set of prompts. Responses are evaluated through automated metrics, LLM judges, or human reviewers. Statistical significance testing ensures observed differences are real, not due to random variation."
        id="def-ab-testing"
      />

      <h2 className="text-2xl font-semibold">Offline Model Comparison</h2>

      <PythonCode
        title="ab_testing_offline.py"
        code={`import random
import json
from collections import defaultdict

class ModelComparison:
    """Compare two models on a set of test prompts."""

    def __init__(self, model_a, model_b, tokenizer_a, tokenizer_b):
        self.models = {"A": model_a, "B": model_b}
        self.tokenizers = {"A": tokenizer_a, "B": tokenizer_b}
        self.results = []

    def generate_response(self, model_key, prompt, max_tokens=512):
        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        outputs = model.generate(
            input_ids=inputs, max_new_tokens=max_tokens,
            temperature=0.7, do_sample=True,
        )
        return tokenizer.decode(outputs[0][inputs.shape[1]:],
                                 skip_special_tokens=True)

    def run_comparison(self, prompts):
        for prompt in prompts:
            resp_a = self.generate_response("A", prompt)
            resp_b = self.generate_response("B", prompt)

            # Randomize order to prevent position bias
            if random.random() > 0.5:
                self.results.append({
                    "prompt": prompt,
                    "first": resp_a, "first_model": "A",
                    "second": resp_b, "second_model": "B",
                })
            else:
                self.results.append({
                    "prompt": prompt,
                    "first": resp_b, "first_model": "B",
                    "second": resp_a, "second_model": "A",
                })

        return self.results

    def compute_win_rate(self, judgments):
        """Compute win rates from judge results."""
        wins = defaultdict(int)
        for j in judgments:
            wins[j["winner"]] += 1
        total = len(judgments)
        print(f"Model A wins: {wins['A']/total*100:.1f}%")
        print(f"Model B wins: {wins['B']/total*100:.1f}%")
        print(f"Ties: {wins.get('tie', 0)/total*100:.1f}%")

# Usage:
# comp = ModelComparison(model_a, model_b, tok_a, tok_b)
# results = comp.run_comparison(test_prompts)
# comp.compute_win_rate(judge_results)`}
        id="code-ab-offline"
      />

      <PythonCode
        title="statistical_significance.py"
        code={`import numpy as np
from scipy import stats

def compute_significance(wins_a, wins_b, ties=0):
    """Test if the difference in win rates is statistically significant."""
    total = wins_a + wins_b + ties
    n = wins_a + wins_b  # Exclude ties

    if n == 0:
        print("No clear winners to compare")
        return

    # Binomial test: is win_rate_A significantly different from 0.5?
    p_value = stats.binomtest(wins_a, n, p=0.5).pvalue

    win_rate_a = wins_a / n
    # 95% confidence interval
    ci = stats.proportion_confint(wins_a, n, alpha=0.05, method="wilson")

    print(f"Win rate A: {win_rate_a:.1%} (95% CI: {ci[0]:.1%}-{ci[1]:.1%})")
    print(f"Win rate B: {1-win_rate_a:.1%}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant at p<0.05: {'Yes' if p_value < 0.05 else 'No'}")

    # Rule of thumb: need at least 100 comparisons for reliable results
    min_needed = int(4 / (0.1**2))  # For detecting 10% difference
    print(f"Minimum comparisons needed (10% effect): ~{min_needed}")

# Example: Model A won 62/100, Model B won 38/100
compute_significance(wins_a=62, wins_b=38)
# Win rate A: 62.0% (95% CI: 52.2%-70.9%)
# P-value: 0.0214
# Significant at p<0.05: Yes`}
        id="code-significance"
      />

      <ExampleBlock
        title="A/B Testing Checklist"
        problem="What is a reliable A/B testing procedure for finetuned models?"
        steps={[
          { formula: '\\text{1. Fix test set (100+ prompts)}', explanation: 'Use the same diverse prompts for both models. Cover all use cases.' },
          { formula: '\\text{2. Randomize presentation order}', explanation: 'Shuffle which model response appears first to avoid position bias.' },
          { formula: '\\text{3. Blind evaluation}', explanation: 'Evaluators should not know which model produced which response.' },
          { formula: '\\text{4. Multiple evaluation criteria}', explanation: 'Score helpfulness, accuracy, and style separately, not just overall preference.' },
          { formula: '\\text{5. Statistical significance test}', explanation: 'Use binomial test with p<0.05 threshold. Need ~100 comparisons minimum.' },
        ]}
        id="example-ab-checklist"
      />

      <NoteBlock
        type="tip"
        title="Use Chatbot Arena Method"
        content="The Chatbot Arena (lmsys.org) methodology is gold standard: show two anonymous responses side by side, let humans pick the better one, and compute Elo ratings. You can replicate this internally using a simple web app with Gradio."
        id="note-arena"
      />

      <WarningBlock
        title="Sample Size Matters"
        content="With only 20 comparisons, a 60/40 win rate is NOT statistically significant (p=0.12). You need at least 100 comparisons to detect a 10% difference reliably. For small differences (55/45), you may need 400+ comparisons. Plan your evaluation budget accordingly."
        id="warning-sample-size"
      />
    </div>
  )
}
