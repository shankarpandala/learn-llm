import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function DataPipelines() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Data Pipelines for NLP</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Building an NLP system requires more than just a model. You need a robust data pipeline
        that ingests raw text, cleans it, transforms it into model-ready features, and handles
        edge cases gracefully. A well-designed pipeline is reproducible, efficient, and modular.
      </p>

      <DefinitionBlock
        title="NLP Data Pipeline"
        definition="An NLP data pipeline is a sequence of processing stages that transforms raw text data into a format suitable for model training or inference. Typical stages include: data collection, cleaning, normalization, tokenization, encoding, batching, and optional augmentation."
        id="def-pipeline"
      />

      <h2 className="text-2xl font-semibold">A Complete Preprocessing Pipeline</h2>

      <PythonCode
        title="nlp_pipeline.py"
        code={`import re
import unicodedata
from collections import Counter

class TextPipeline:
    """A modular NLP preprocessing pipeline."""

    def __init__(self, steps=None):
        self.steps = steps or [
            self.clean_html,
            self.normalize_unicode,
            self.normalize_whitespace,
            self.lowercase,
            self.tokenize,
        ]

    def __call__(self, text):
        result = text
        for step in self.steps:
            result = step(result)
        return result

    @staticmethod
    def clean_html(text):
        """Remove HTML tags and decode entities."""
        import html
        text = html.unescape(text)
        return re.sub(r'<[^>]+>', ' ', text)

    @staticmethod
    def normalize_unicode(text):
        """NFKC normalize and remove control characters."""
        text = unicodedata.normalize('NFKC', text)
        return ''.join(c for c in text if unicodedata.category(c) != 'Cc' or c == '\n')

    @staticmethod
    def normalize_whitespace(text):
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def lowercase(text):
        return text.lower()

    @staticmethod
    def tokenize(text):
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r"\b\w+\b", text)

# Use the pipeline
pipeline = TextPipeline()
raw = "<p>The Transformer &amp; BERT models are GREAT!! 🚀</p>"
tokens = pipeline(raw)
print(f"Raw:    {raw}")
print(f"Tokens: {tokens}")

# Custom pipeline for different tasks
search_pipeline = TextPipeline(steps=[
    TextPipeline.normalize_unicode,
    TextPipeline.normalize_whitespace,
    TextPipeline.lowercase,
    TextPipeline.tokenize,
])
print(f"\\nSearch pipeline: {search_pipeline('  Find BERT   Models  ')}")`}
        id="code-pipeline"
      />

      <h2 className="text-2xl font-semibold">Batching and Padding</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Neural networks process data in batches for efficiency. Since text sequences have
        varying lengths, they must be padded to a uniform length within each batch.
      </p>

      <PythonCode
        title="batching_padding.py"
        code={`import numpy as np

def create_batches(sequences, batch_size=3, pad_token=0, max_len=None):
    """
    Create padded batches from variable-length sequences.
    Returns batches of (padded_sequences, attention_masks, lengths).
    """
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        lengths = [len(seq) for seq in batch]

        # Pad to the max length in this batch (or global max_len)
        pad_len = max_len or max(lengths)
        padded = np.full((len(batch), pad_len), pad_token, dtype=np.int64)
        attention_mask = np.zeros((len(batch), pad_len), dtype=np.int64)

        for j, seq in enumerate(batch):
            seq_len = min(len(seq), pad_len)
            padded[j, :seq_len] = seq[:seq_len]
            attention_mask[j, :seq_len] = 1

        batches.append({
            'input_ids': padded,
            'attention_mask': attention_mask,
            'lengths': lengths,
        })
    return batches

# Simulate tokenized sequences of different lengths
sequences = [
    [101, 2023, 2003, 1037, 3231, 102],         # 6 tokens
    [101, 2312, 6251, 102],                       # 4 tokens
    [101, 1996, 4937, 4540, 2006, 1996, 13523, 102],  # 8 tokens
    [101, 7592, 102],                              # 3 tokens
    [101, 2028, 2062, 6251, 2182, 102],           # 6 tokens
]

batches = create_batches(sequences, batch_size=3)
for i, batch in enumerate(batches):
    print(f"Batch {i+1}:")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Attention mask:\\n{batch['attention_mask']}")
    print(f"  Lengths: {batch['lengths']}\\n")`}
        id="code-batching"
      />

      <ExampleBlock
        title="Pipeline Design Decisions"
        problem="You are building a pipeline to prepare data for fine-tuning a BERT model on movie review sentiment. What preprocessing steps should you include?"
        steps={[
          { formula: 'Step 1: HTML cleaning', explanation: 'Movie reviews from the web may contain HTML tags, entities, and formatting.' },
          { formula: 'Step 2: Unicode normalization (NFKC)', explanation: 'Standardize character representations. Do not remove emojis -- they carry sentiment.' },
          { formula: 'Step 3: BERT tokenizer (WordPiece)', explanation: 'Use the pre-trained tokenizer that matches your BERT model. Do NOT lowercase if using cased BERT.' },
          { formula: 'Step 4: Truncation to 512 tokens', explanation: 'BERT has a maximum sequence length of 512. Truncate or split longer reviews.' },
          { formula: 'Step 5: Add special tokens [CLS] and [SEP]', explanation: 'BERT requires these boundary markers.' },
        ]}
        id="example-pipeline-design"
      />

      <NoteBlock
        type="tip"
        title="Data Quality Over Quantity"
        content="The LLaMA paper (Touvron et al., 2023) showed that a smaller model trained on high-quality, well-curated data can outperform larger models trained on raw web scrapes. Data deduplication, quality filtering (using perplexity-based heuristics), and domain balancing are now considered as important as model architecture."
        id="note-data-quality"
      />

      <WarningBlock
        title="Reproducibility"
        content="Always version your preprocessing pipeline alongside your model. A change in tokenizer version, Unicode normalization form, or cleaning rules can silently change your data distribution and invalidate comparisons. Use deterministic processing and log every transformation step."
        id="warning-reproducibility"
      />

      <NoteBlock
        type="note"
        title="Modern Data Pipeline Tools"
        content="Production NLP pipelines use tools like Hugging Face Datasets (memory-mapped, lazy processing), Apache Beam (distributed processing), and Spark NLP. For LLM pre-training, specialized tools like The Pile's data pipeline, RedPajama, and Dolma handle terabytes of text with deduplication, quality scoring, and PII removal."
        id="note-tools"
      />
    </div>
  )
}
