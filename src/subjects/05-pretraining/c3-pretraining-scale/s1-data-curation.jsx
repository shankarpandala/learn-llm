import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function DataCuration() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Pretraining Data Curation</h1>
      <p className="text-lg text-gray-300">
        The quality and composition of pretraining data is one of the most critical factors
        in LLM performance. Modern LLMs are trained on trillions of tokens from diverse sources
        including web crawls, books, code, and curated datasets.
      </p>

      <DefinitionBlock
        title="Common Crawl"
        definition="Common Crawl is a nonprofit that maintains an open repository of web crawl data, producing petabytes of raw HTML monthly. It forms the foundation of most pretraining datasets but requires extensive cleaning: raw Common Crawl is estimated to be only 1-5% high-quality text."
        notation="After filtering, Common Crawl typically yields datasets of 1-5 trillion tokens."
        id="common-crawl-def"
      />

      <ExampleBlock
        title="Major Pretraining Datasets"
        problem="Compare the composition of The Pile, RedPajama, and FineWeb."
        steps={[
          {
            formula: '\\text{The Pile (2021): 825 GB, 22 diverse sources}',
            explanation: 'Created by EleutherAI. Includes Pile-CC, PubMed, ArXiv, GitHub, StackExchange, Wikipedia, Books3, and more. First large-scale curated open dataset.'
          },
          {
            formula: '\\text{RedPajama v2 (2023): 30T tokens, 5 languages}',
            explanation: 'Created by Together AI. Web data with quality signals. Includes CommonCrawl, C4, GitHub, Books, ArXiv, Wikipedia, StackExchange.'
          },
          {
            formula: '\\text{FineWeb (2024): 15T tokens, deduplicated CC}',
            explanation: 'Created by HuggingFace. Aggressive filtering and deduplication of Common Crawl with transparent methodology. FineWeb-Edu subset has educational content scoring.'
          }
        ]}
        id="datasets-comparison"
      />

      <NoteBlock
        type="historical"
        title="The Data Scaling Journey"
        content="GPT-1 used BookCorpus (~800M tokens). GPT-2 used WebText (~8B tokens). GPT-3 used a 300B token mix. LLaMA used 1.4T tokens. LLaMA-2 used 2T tokens. Modern models like LLaMA-3 train on 15T+ tokens. Each generation dramatically increased data scale and quality requirements."
        id="data-scaling-history"
      />

      <PythonCode
        title="data_curation_pipeline.py"
        code={`from datasets import load_dataset
import hashlib
import re

# Load a sample from FineWeb
# dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

# Simulated data curation pipeline
class DataCurationPipeline:
    """Pipeline for processing raw web text into pretraining data."""

    def __init__(self):
        self.stats = {"total": 0, "passed": 0, "reasons": {}}

    def language_filter(self, text, min_confidence=0.8):
        """Filter non-English text (simplified)."""
        ascii_ratio = sum(c.isascii() for c in text) / max(len(text), 1)
        return ascii_ratio > 0.8

    def quality_filter(self, text):
        """Heuristic quality filters inspired by C4/FineWeb."""
        words = text.split()

        # Minimum length
        if len(words) < 50:
            return False, "too_short"

        # Maximum fraction of lines ending with ellipsis
        lines = text.split("\\n")
        ellipsis_lines = sum(1 for l in lines if l.strip().endswith("..."))
        if lines and ellipsis_lines / len(lines) > 0.3:
            return False, "too_many_ellipsis"

        # Check for boilerplate
        boilerplate = ["cookie policy", "javascript", "subscribe now",
                       "click here", "terms of service"]
        lower = text.lower()
        if sum(1 for b in boilerplate if b in lower) >= 3:
            return False, "boilerplate"

        # Word length distribution
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 3 or avg_word_len > 12:
            return False, "unusual_word_length"

        # Repetition filter
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.1:
            return False, "too_repetitive"

        return True, "passed"

    def process(self, documents):
        """Run full pipeline on documents."""
        results = []
        for doc in documents:
            self.stats["total"] += 1
            if not self.language_filter(doc):
                self.stats["reasons"]["non_english"] = \
                    self.stats["reasons"].get("non_english", 0) + 1
                continue
            passed, reason = self.quality_filter(doc)
            if not passed:
                self.stats["reasons"][reason] = \
                    self.stats["reasons"].get(reason", 0) + 1
                continue
            self.stats["passed"] += 1
            results.append(doc)
        return results

# Run pipeline
pipeline = DataCurationPipeline()
sample_docs = [
    "The quick brown fox " * 100,  # repetitive
    "Click here to subscribe. Cookie policy. Terms of service. " * 20,
    "Short text.",
    " ".join(f"word{i}" for i in range(200)),  # normal length
    "Natural language processing has revolutionized how computers "
    "understand human language. " * 5 + "This comprehensive review "
    "covers tokenization, embeddings, and transformer architectures "
    "that power modern systems. " * 3,
]
clean = pipeline.process(sample_docs)
print(f"Input: {pipeline.stats['total']} docs")
print(f"Passed: {pipeline.stats['passed']} docs")
print(f"Filter reasons: {pipeline.stats['reasons']}")

# Data source mixing weights (typical for a 7B model)
sources = {
    "CommonCrawl": {"tokens": "4.5T", "weight": 0.67},
    "GitHub":      {"tokens": "0.5T", "weight": 0.045},
    "Wikipedia":   {"tokens": "0.1T", "weight": 0.045},
    "Books":       {"tokens": "0.3T", "weight": 0.045},
    "ArXiv":       {"tokens": "0.1T", "weight": 0.025},
    "StackExchange":{"tokens": "0.1T","weight": 0.02},
}
print("\\nTypical data mix:")
for name, info in sources.items():
    print(f"  {name:>16s}: {info['tokens']:>5s} tokens, weight={info['weight']:.1%}")`}
        id="curation-code"
      />

      <WarningBlock
        title="Data Contamination"
        content="Pretraining data may contain benchmark test sets (e.g., MMLU questions scraped from the web). This data contamination inflates evaluation scores. Careful decontamination via n-gram matching against benchmarks is essential but imperfect. Always report contamination analysis alongside benchmark results."
        id="contamination-warning"
      />
    </div>
  )
}
