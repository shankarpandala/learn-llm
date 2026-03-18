import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function DataQuality() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Data Quality Over Quantity</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Research has repeatedly shown that a small, high-quality instruction dataset can outperform
        a much larger noisy one. The LIMA paper (Zhou et al., 2023) demonstrated that just 1,000
        carefully curated examples can produce a strong instruction-following model. This section
        explores what makes instruction data high-quality and how to curate effective datasets.
      </p>

      <DefinitionBlock
        title="Data Quality for Instruction Tuning"
        definition="High-quality instruction data exhibits: (1) diversity of tasks and instruction styles, (2) correctness and factual accuracy of responses, (3) appropriate response length and detail level, (4) consistent formatting, and (5) alignment with desired behavior. The LIMA hypothesis states that most of a model's knowledge comes from pretraining, and instruction tuning only needs to teach the format of interaction."
        id="def-data-quality"
      />

      <ExampleBlock
        title="Quality vs. Quantity Evidence"
        problem="Compare model performance with different dataset sizes and quality levels."
        steps={[
          { formula: '\\text{LIMA: 1K curated} \\approx \\text{Alpaca: 52K synthetic}', explanation: 'LIMA with 1,000 hand-picked examples matched or exceeded Alpaca with 52K synthetic examples.' },
          { formula: '\\text{Quality filtering 52K} \\rightarrow \\text{9K high-quality} \\uparrow \\text{performance}', explanation: 'Filtering Alpaca data with quality heuristics improved performance while reducing dataset size by 80%.' },
          { formula: '\\text{Deita (2024): score-based selection from 300K} \\rightarrow \\text{6K best}', explanation: 'Automatic quality scoring and diversity selection produced a tiny but highly effective subset.' },
        ]}
        id="example-quality-evidence"
      />

      <h2 className="text-2xl font-semibold">Quality Filtering Strategies</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Several automated and semi-automated approaches can identify high-quality examples
        from a larger pool, reducing human annotation effort while maintaining quality.
      </p>

      <PythonCode
        title="data_quality_filtering.py"
        code={`from datasets import load_dataset
import numpy as np

dataset = load_dataset("tatsu-lab/alpaca", split="train")

def quality_score(example):
    """Heuristic quality scoring for instruction data."""
    score = 0.0
    instruction = example["instruction"]
    output = example["output"]

    # Length heuristics
    if 10 < len(instruction.split()) < 100:
        score += 1.0  # Reasonable instruction length
    if 20 < len(output.split()) < 500:
        score += 1.0  # Not too short, not too long

    # Formatting quality
    if not output.startswith(instruction[:20]):
        score += 0.5  # Response doesn't just repeat the instruction
    if any(c in output for c in ["1.", "2.", "- ", "* "]):
        score += 0.5  # Structured response with lists

    # Diversity signals
    if example["input"]:
        score += 0.5  # Has additional context (more complex task)

    # Penalize low-effort responses
    if len(output.split()) < 5:
        score -= 2.0  # Very short responses are likely low quality
    if output.count("\\n") == 0 and len(output.split()) > 100:
        score -= 0.5  # Long wall of text without formatting

    return {"quality_score": score}

# Score and filter
scored = dataset.map(quality_score)
scores = np.array(scored["quality_score"])
print(f"Score distribution: mean={scores.mean():.2f}, std={scores.std():.2f}")
print(f"Total examples: {len(scored)}")

# Keep top 20% by quality
threshold = np.percentile(scores, 80)
high_quality = scored.filter(lambda x: x["quality_score"] >= threshold)
print(f"High-quality subset: {len(high_quality)} examples")

# LLM-as-judge scoring (more sophisticated)
def llm_judge_prompt(instruction, output):
    return f"""Rate this instruction-response pair on a scale of 1-5:
Instruction: {instruction}
Response: {output}

Criteria: accuracy, helpfulness, clarity, completeness.
Score (1-5):"""

# Use a strong model to score each example
# Then select top-K by LLM judge score for training`}
        id="code-quality-filtering"
      />

      <NoteBlock
        type="note"
        title="The LIMA Hypothesis"
        content="The LIMA paper (Less Is More for Alignment) argued that alignment is primarily about learning style and format, not knowledge. The base model already has vast knowledge from pretraining. SFT just teaches it when and how to deploy that knowledge in response to instructions. This explains why 1,000 well-chosen examples suffice: they cover the space of interaction patterns, not the space of all knowledge."
        id="note-lima"
      />

      <WarningBlock
        title="Deduplication is Essential"
        content="Many instruction datasets contain near-duplicate examples that waste training budget and can cause the model to overfit to specific phrasings. Always deduplicate before training using techniques like MinHash or embedding-based similarity. Even removing exact string duplicates can improve a dataset significantly."
        id="warning-dedup"
      />

      <NoteBlock
        type="tip"
        title="Diversity-Aware Selection"
        content="Beyond individual quality, the diversity of the selected subset matters. Use embedding-based clustering (e.g., k-means on sentence embeddings) to ensure coverage across task types, topics, and difficulty levels. Select examples that maximize coverage of the embedding space rather than greedily picking the highest-scored ones, which may all be similar."
        id="note-diversity"
      />
    </div>
  )
}
