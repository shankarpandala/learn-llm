import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function DataCleaning() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Data Cleaning for Finetuning</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Noisy, inconsistent, or low-quality data is the most common cause of poor finetuning
        results. Systematic data cleaning removes duplicates, fixes formatting issues, filters
        out toxic or incorrect content, and ensures consistent quality throughout the dataset.
      </p>

      <DefinitionBlock
        title="Data Quality Dimensions"
        definition="Finetuning data quality encompasses: accuracy (factually correct responses), consistency (uniform formatting and style), diversity (varied instruction types), completeness (responses fully address the instruction), and safety (no harmful or biased content)."
        id="def-data-quality"
      />

      <PythonCode
        title="data_cleaning_pipeline.py"
        code={`from datasets import Dataset
import re
import hashlib

def clean_dataset(dataset):
    """Comprehensive data cleaning pipeline."""

    # Step 1: Remove duplicates
    seen_hashes = set()
    def deduplicate(example):
        text = example.get("output", "") or str(example.get("messages", ""))
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        return True

    before = len(dataset)
    dataset = dataset.filter(deduplicate)
    print(f"Dedup: {before} -> {len(dataset)} ({before - len(dataset)} removed)")

    # Step 2: Remove empty or very short responses
    def filter_length(example):
        response = example.get("output", "")
        if not response:
            msgs = example.get("messages", [])
            response = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
        return len(response.strip()) > 20

    dataset = dataset.filter(filter_length)
    print(f"After length filter: {len(dataset)}")

    # Step 3: Clean text artifacts
    def clean_text(example):
        for key in ["instruction", "input", "output"]:
            if key in example and example[key]:
                text = example[key]
                text = re.sub(r'<[^>]+>', '', text)        # Remove HTML tags
                text = re.sub(r'\\s+', ' ', text)            # Normalize whitespace
                text = text.strip()
                example[key] = text
        return example

    dataset = dataset.map(clean_text)

    # Step 4: Filter out known bad patterns
    bad_patterns = [
        r"as an ai language model",
        r"i cannot .* because i am",
        r"i don't have personal",
        r"my training data",
    ]

    def filter_bad_patterns(example):
        response = example.get("output", "").lower()
        return not any(re.search(p, response) for p in bad_patterns)

    dataset = dataset.filter(filter_bad_patterns)
    print(f"After pattern filter: {len(dataset)}")

    return dataset

# Usage:
# cleaned = clean_dataset(raw_dataset)`}
        id="code-cleaning-pipeline"
      />

      <PythonCode
        title="quality_scoring.py"
        code={`import numpy as np

def compute_quality_metrics(dataset):
    """Compute quality metrics for a dataset."""
    metrics = {
        "total_examples": len(dataset),
        "response_lengths": [],
        "instruction_lengths": [],
        "empty_inputs": 0,
    }

    for ex in dataset:
        resp = ex.get("output", "")
        inst = ex.get("instruction", "")
        metrics["response_lengths"].append(len(resp))
        metrics["instruction_lengths"].append(len(inst))
        if not ex.get("input", "").strip():
            metrics["empty_inputs"] += 1

    rl = np.array(metrics["response_lengths"])
    il = np.array(metrics["instruction_lengths"])

    print(f"Dataset size: {metrics['total_examples']}")
    print(f"Response length: mean={rl.mean():.0f}, median={np.median(rl):.0f}, "
          f"min={rl.min()}, max={rl.max()}")
    print(f"Instruction length: mean={il.mean():.0f}, median={np.median(il):.0f}")
    print(f"Examples with empty input: {metrics['empty_inputs']} "
          f"({metrics['empty_inputs']/len(dataset)*100:.1f}%)")

    # Check for suspicious patterns
    very_short = (rl < 20).sum()
    very_long = (rl > 5000).sum()
    print(f"Very short responses (<20 chars): {very_short}")
    print(f"Very long responses (>5000 chars): {very_long}")

    return metrics

# metrics = compute_quality_metrics(dataset)`}
        id="code-quality-metrics"
      />

      <ExampleBlock
        title="Common Data Issues and Fixes"
        problem="What are the most frequent data quality problems in finetuning datasets?"
        steps={[
          { formula: '\\text{Exact duplicates: 5-15\\%}', explanation: 'Remove using hash-based deduplication. Also check near-duplicates with fuzzy matching.' },
          { formula: '\\text{Wrong language: 1-5\\%}', explanation: 'Use language detection (langdetect) to filter examples in unexpected languages.' },
          { formula: '\\text{Truncated responses}', explanation: 'Responses cut off mid-sentence. Filter by checking for proper sentence endings.' },
          { formula: '\\text{Prompt leakage}', explanation: 'Responses that repeat the instruction verbatim. Filter with overlap detection.' },
        ]}
        id="example-common-issues"
      />

      <NoteBlock
        type="tip"
        title="Iterative Cleaning"
        content="Clean your data, train a small model, evaluate outputs, and identify remaining data issues. Repeat this cycle 2-3 times. Each iteration reveals new quality problems that are hard to catch with automated rules alone."
        id="note-iterative"
      />

      <WarningBlock
        title="Do Not Over-Filter"
        content="Aggressive filtering can remove too much data or introduce bias. If you filter out all short responses, the model may never learn to give concise answers. Keep a balance: remove clearly bad examples but preserve diversity in response style and length."
        id="warning-overfilter"
      />
    </div>
  )
}
