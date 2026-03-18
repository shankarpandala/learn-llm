import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function HumanEval() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Human Evaluation Protocols</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Automated benchmarks cannot capture everything that matters about model quality. Human
        evaluation remains the gold standard for judging fluency, helpfulness, safety, and
        alignment with real user needs. This section covers practical protocols for running
        human evaluations on your fine-tuned models.
      </p>

      <DefinitionBlock
        title="Human Evaluation"
        definition="Human evaluation involves having human raters judge model outputs on defined criteria. Common approaches include Likert-scale ratings (1-5), pairwise comparisons (A vs B), and ranking. Inter-annotator agreement is measured by Cohen's kappa: $\kappa = \frac{p_o - p_e}{1 - p_e}$ where $p_o$ is observed agreement and $p_e$ is expected chance agreement."
        id="def-human-eval"
      />

      <ExampleBlock
        title="Evaluation Dimensions"
        problem="What criteria should human raters evaluate?"
        steps={[
          { formula: '\\text{Helpfulness: does the answer solve the user\'s problem?}', explanation: 'Rate 1-5 whether the response directly addresses the query and provides useful information.' },
          { formula: '\\text{Accuracy: is the information factually correct?}', explanation: 'Check claims against known facts. Flag hallucinations and errors.' },
          { formula: '\\text{Coherence: is the response well-structured and logical?}', explanation: 'Evaluate grammar, flow, and logical consistency.' },
          { formula: '\\text{Safety: does the response avoid harmful content?}', explanation: 'Check for toxicity, bias, and inappropriate suggestions.' },
        ]}
        id="example-eval-dimensions"
      />

      <PythonCode
        title="human_eval_pipeline.py"
        code={`import json
import random
from datetime import datetime

def create_evaluation_batch(prompts, model_paths, num_samples=50):
    """Generate evaluation samples from multiple models."""
    from transformers import pipeline

    samples = []
    selected_prompts = random.sample(prompts, min(num_samples, len(prompts)))

    for model_path in model_paths:
        pipe = pipeline(
            "text-generation", model=model_path,
            torch_dtype="float16", device_map="auto"
        )

        for prompt in selected_prompts:
            output = pipe(
                prompt, max_new_tokens=512,
                temperature=0.7, do_sample=True
            )[0]["generated_text"]

            samples.append({
                "id": f"{hash(prompt + model_path) % 10000:04d}",
                "prompt": prompt,
                "response": output[len(prompt):].strip(),
                "model": model_path,
                "timestamp": datetime.now().isoformat(),
            })

    # Shuffle to blind raters to model identity
    random.shuffle(samples)

    # Remove model labels for blind evaluation
    blind_samples = [
        {k: v for k, v in s.items() if k != "model"} for s in samples
    ]

    with open("eval_batch_blind.json", "w") as f:
        json.dump(blind_samples, f, indent=2)
    with open("eval_batch_key.json", "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Created {len(samples)} samples for evaluation")
    return samples

prompts = [
    "Explain how photosynthesis works in simple terms.",
    "Write a Python function to find the longest palindromic substring.",
    "What are the pros and cons of remote work?",
    "How do I fix a leaking kitchen faucet?",
]

create_evaluation_batch(
    prompts,
    model_paths=["./base-model", "./finetuned-model"],
    num_samples=50
)`}
        id="code-eval-pipeline"
      />

      <PythonCode
        title="compute_agreement.py"
        code={`import numpy as np
from collections import Counter

def cohens_kappa(rater1, rater2):
    """Compute Cohen's kappa for inter-annotator agreement."""
    assert len(rater1) == len(rater2)
    n = len(rater1)

    # Observed agreement
    p_o = sum(a == b for a, b in zip(rater1, rater2)) / n

    # Expected agreement by chance
    counts1 = Counter(rater1)
    counts2 = Counter(rater2)
    categories = set(list(counts1.keys()) + list(counts2.keys()))
    p_e = sum((counts1[c] / n) * (counts2[c] / n) for c in categories)

    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0
    return kappa

def analyze_ratings(ratings_file, key_file):
    """Analyze human evaluation results."""
    import json
    with open(ratings_file) as f:
        ratings = json.load(f)
    with open(key_file) as f:
        key = json.load(f)

    id_to_model = {s["id"]: s["model"] for s in key}

    model_scores = {}
    for r in ratings:
        model = id_to_model[r["id"]]
        if model not in model_scores:
            model_scores[model] = []
        model_scores[model].append(r["score"])

    for model, scores in model_scores.items():
        print(f"{model}:")
        print(f"  Mean: {np.mean(scores):.2f} +/- {np.std(scores):.2f}")
        print(f"  Median: {np.median(scores):.1f}")
        print(f"  N: {len(scores)}")

# Example: kappa between two raters
rater1 = [4, 3, 5, 2, 4, 3, 5, 4, 3, 4]
rater2 = [4, 2, 5, 3, 4, 3, 4, 4, 3, 5]
print(f"Cohen's kappa: {cohens_kappa(rater1, rater2):.3f}")`}
        id="code-agreement"
      />

      <NoteBlock
        type="tip"
        title="Pairwise Comparison Is More Reliable"
        content="Asking raters 'Which response is better, A or B?' produces more consistent results than absolute ratings on a 1-5 scale. Pairwise comparison reduces rater calibration issues and is the approach used by Chatbot Arena and LMSYS."
        id="note-pairwise"
      />

      <WarningBlock
        title="Minimum Rater Count"
        content="Use at least 3 raters per sample and require majority agreement. A single rater introduces too much subjective bias. For high-stakes evaluations, use 5+ raters and measure inter-annotator agreement with Cohen's kappa before trusting the results."
        id="warning-rater-count"
      />

      <NoteBlock
        type="note"
        title="LLM-as-Judge"
        content="Using GPT-4 or Claude as automated evaluators (LLM-as-judge) is increasingly common as a scalable proxy for human evaluation. While faster and cheaper, it has known biases such as preferring longer responses and outputs similar to its own style. Always validate LLM-as-judge with a human evaluation subset."
        id="note-llm-judge"
      />
    </div>
  )
}
