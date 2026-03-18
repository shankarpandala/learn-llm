import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LLMJudge() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LLM-as-Judge and MT-Bench</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Traditional benchmarks use exact-match or multiple-choice formats, but open-ended generation
        quality is harder to evaluate. LLM-as-judge uses a strong model to score or compare outputs,
        providing scalable evaluation of free-form responses.
      </p>

      <DefinitionBlock
        title="LLM-as-Judge"
        definition="An evaluation paradigm where a capable language model (the judge) assesses the quality of responses from other models. The judge can provide absolute scores (single-answer grading) or relative comparisons (pairwise ranking)."
        id="def-llm-judge"
      />

      <DefinitionBlock
        title="MT-Bench"
        definition="A multi-turn benchmark consisting of 80 high-quality questions across 8 categories (writing, roleplay, extraction, reasoning, math, coding, knowledge, STEM). Each question has a follow-up turn, and responses are scored 1-10 by a GPT-4 judge."
        id="def-mt-bench"
      />

      <h2 className="text-2xl font-semibold">Judge Agreement and Reliability</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The reliability of LLM judges is measured by agreement with human annotators. Zheng et al.
        (2023) found GPT-4 judge agreement with humans exceeds 80%, comparable to inter-annotator
        agreement among humans. Agreement is measured via Cohen's kappa:
      </p>
      <BlockMath math="\kappa = \frac{p_o - p_e}{1 - p_e}" />
      <p className="text-gray-700 dark:text-gray-300">
        where <InlineMath math="p_o" /> is observed agreement and <InlineMath math="p_e" /> is
        expected agreement by chance.
      </p>

      <ExampleBlock
        title="Pairwise Comparison Prompt"
        problem="Design an LLM judge prompt that compares two model responses."
        steps={[
          { formula: '\\text{System: You are an impartial judge...}', explanation: 'Set the judge role with explicit instructions for objectivity.' },
          { formula: '\\text{Present: [Question], [Response A], [Response B]}', explanation: 'Show the original query and both candidate responses.' },
          { formula: '\\text{Criteria: helpfulness, relevance, accuracy, depth}', explanation: 'Define explicit evaluation criteria to reduce subjectivity.' },
          { formula: '\\text{Output: [[A]] or [[B]] or [[Tie]]}', explanation: 'Request structured output for easy parsing.' },
        ]}
        id="example-pairwise"
      />

      <PythonCode
        title="llm_judge_evaluation.py"
        code={`# LLM-as-Judge implementation using OpenAI API
from openai import OpenAI

client = OpenAI()

JUDGE_PROMPT = """You are an expert judge evaluating AI assistant responses.
Rate the following response on a scale of 1-10 for each criterion:
- Helpfulness: Does it address the user's question?
- Accuracy: Is the information correct?
- Clarity: Is the response well-organized and clear?
- Depth: Does it provide sufficient detail?

Question: {question}
Response: {response}

Provide your ratings as JSON:
{{"helpfulness": X, "accuracy": X, "clarity": X, "depth": X, "overall": X, "explanation": "..."}}"""

def judge_response(question, response, model="gpt-4o"):
    """Score a single response using LLM judge."""
    result = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(question=question, response=response)
        }],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    import json
    return json.loads(result.choices[0].message.content)

PAIRWISE_PROMPT = """Compare these two responses to the question below.
Question: {question}
Response A: {response_a}
Response B: {response_b}

Which response is better? Output ONLY one of: [[A]], [[B]], or [[Tie]].
Then explain your reasoning in 2-3 sentences."""

def pairwise_judge(question, response_a, response_b, model="gpt-4o"):
    """Pairwise comparison between two responses."""
    result = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PAIRWISE_PROMPT.format(
            question=question, response_a=response_a, response_b=response_b
        )}],
        temperature=0.0,
    )
    text = result.choices[0].message.content
    if "[[A]]" in text: return "A", text
    elif "[[B]]" in text: return "B", text
    return "Tie", text

# Run evaluation on MT-Bench style questions
mt_bench_sample = [
    {"question": "Explain quantum entanglement to a 10-year-old.",
     "follow_up": "Now explain it using a mathematical formulation."},
    {"question": "Write a persuasive essay about renewable energy.",
     "follow_up": "Now write a counterargument to your essay."},
]

for item in mt_bench_sample:
    scores = judge_response(item["question"], "Sample model response here...")
    print(f"Q: {item['question'][:50]}... Score: {scores['overall']}/10")`}
        id="code-llm-judge"
      />

      <WarningBlock
        title="Position Bias in LLM Judges"
        content="LLM judges exhibit position bias: they tend to favor the response presented first in pairwise comparisons. Mitigate this by running comparisons in both orders (A-B and B-A) and averaging results. Also beware of self-enhancement bias, where a model may rate its own outputs more favorably."
        id="warning-position-bias"
      />

      <NoteBlock
        type="tip"
        title="Improving Judge Reliability"
        content="Use chain-of-thought judging (ask the judge to reason before scoring), run multiple evaluations and aggregate, swap presentation order for pairwise comparisons, and use reference answers when available. Consider using multiple judge models for high-stakes evaluations."
        id="note-reliability"
      />

      <NoteBlock
        type="note"
        title="Beyond MT-Bench"
        content="AlpacaEval and WildBench extend the LLM-judge paradigm. AlpacaEval 2.0 uses length-controlled win rates to avoid favoring verbose models. WildBench uses real user queries from the wild for more representative evaluation."
        id="note-beyond-mt"
      />
    </div>
  )
}
