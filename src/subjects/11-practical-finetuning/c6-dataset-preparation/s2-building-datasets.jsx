import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function BuildingDatasets() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Building Finetuning Datasets</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The quality of your finetuning dataset determines the quality of your model. This section
        covers strategies for collecting, curating, and building instruction-following datasets
        from various sources.
      </p>

      <DefinitionBlock
        title="Instruction Dataset"
        definition="An instruction dataset consists of (instruction, response) pairs that teach a model to follow instructions. High-quality datasets have diverse instructions, accurate responses, and consistent formatting. Quality matters more than quantity: 1000 excellent examples often beat 100,000 mediocre ones."
        id="def-instruction-dataset"
      />

      <h2 className="text-2xl font-semibold">Source 1: Existing Open Datasets</h2>

      <PythonCode
        title="load_open_datasets.py"
        code={`from datasets import load_dataset

# Popular open instruction datasets
datasets_catalog = {
    # General instruction following
    "yahma/alpaca-cleaned": "52K cleaned Alpaca instructions",
    "Open-Orca/OpenOrca": "1M+ diverse instructions from GPT-4",
    "HuggingFaceH4/ultrachat_200k": "200K multi-turn conversations",

    # Domain-specific
    "sahil2801/CodeAlpaca-20k": "20K code instruction pairs",
    "lavita/medical-qa-shared-task-v1-half": "Medical Q&A",
    "garage-bAInd/Open-Platypus": "25K STEM reasoning",

    # Preference/alignment data
    "argilla/ultrafeedback-binarized-preferences-cleaned": "DPO pairs",
    "Intel/orca_dpo_pairs": "DPO preference pairs",
}

for name, desc in datasets_catalog.items():
    print(f"{name}: {desc}")

# Load and inspect a dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
print(f"\\nAlpaca-cleaned: {len(dataset)} examples")
print(f"Sample: {dataset[0]}")`}
        id="code-open-datasets"
      />

      <h2 className="text-2xl font-semibold">Source 2: Converting Internal Data</h2>

      <PythonCode
        title="convert_internal_data.py"
        code={`import json
import csv
from datasets import Dataset

# Convert customer support logs to instruction format
def convert_support_logs(log_file):
    examples = []
    with open(log_file) as f:
        for line in f:
            log = json.loads(line)
            examples.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful customer support agent."},
                    {"role": "user", "content": log["customer_query"]},
                    {"role": "assistant", "content": log["agent_response"]},
                ]
            })
    return Dataset.from_list(examples)

# Convert FAQ documents to instruction pairs
def convert_faq(faq_data):
    examples = []
    for item in faq_data:
        examples.append({
            "messages": [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]},
            ]
        })
    return Dataset.from_list(examples)

# Convert documentation to instruction format
def convert_docs_to_qa(doc_chunks):
    """Use an LLM to generate Q&A pairs from documentation."""
    examples = []
    for chunk in doc_chunks:
        # Generate questions about this chunk using an LLM
        # Then pair questions with chunk-based answers
        examples.append({
            "messages": [
                {"role": "system", "content": "Answer based on the documentation."},
                {"role": "user", "content": f"Based on this context: {chunk[:200]}...\\nQ: [generated question]"},
                {"role": "assistant", "content": "[generated answer]"},
            ]
        })
    return examples

print("Conversion functions ready. Adapt to your data format.")`}
        id="code-convert-data"
      />

      <ExampleBlock
        title="Dataset Curation Strategy"
        problem="How to build a high-quality dataset from scratch?"
        steps={[
          { formula: '\\text{Step 1: Define scope and tasks}', explanation: 'List 10-20 specific tasks your model should handle. Be precise about input/output formats.' },
          { formula: '\\text{Step 2: Seed examples (50-100)}', explanation: 'Write high-quality examples by hand for each task category.' },
          { formula: '\\text{Step 3: Expand with LLM (1K-10K)}', explanation: 'Use GPT-4 or Claude to generate more examples following your seed patterns.' },
          { formula: '\\text{Step 4: Human review (filter 20-30\\%)}', explanation: 'Review and filter generated examples. Remove incorrect, low-quality, or duplicate ones.' },
          { formula: '\\text{Step 5: Iterate}', explanation: 'Train a model, find weaknesses, add more examples targeting those areas.' },
        ]}
        id="example-curation-strategy"
      />

      <NoteBlock
        type="tip"
        title="Diversity Is Key"
        content="A diverse dataset produces a more robust model. Vary: instruction phrasing, response length, complexity level, topic coverage, and multi-turn vs single-turn examples. Avoid having too many similar examples, which causes the model to become formulaic."
        id="note-diversity"
      />

      <WarningBlock
        title="Legal and Licensing Considerations"
        content="Be careful with data sources. Some datasets are licensed for research only (not commercial use). Data generated by GPT-4 may have OpenAI Terms of Service restrictions on training competing models. Always check the license of each dataset you use."
        id="warning-licensing"
      />
    </div>
  )
}
