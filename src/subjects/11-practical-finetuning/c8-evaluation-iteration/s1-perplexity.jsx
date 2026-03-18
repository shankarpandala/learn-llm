import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Perplexity() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Measuring Perplexity</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Perplexity is the most fundamental metric for evaluating language models. It measures how
        surprised the model is by test data -- lower perplexity means the model better predicts
        the text. While not sufficient alone, perplexity is a quick sanity check for finetuning.
      </p>

      <DefinitionBlock
        title="Perplexity"
        definition="Perplexity is the exponential of the average negative log-likelihood per token: $\\text{PPL} = \\exp\\left(-\\frac{1}{T}\\sum_{t=1}^{T} \\log P_\\theta(x_t | x_{<t})\\right)$. A perplexity of $k$ means the model is as uncertain as if choosing uniformly among $k$ options at each position."
        notation="PPL = \exp(-\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_{<t}))"
        id="def-perplexity"
      />

      <ExampleBlock
        title="Interpreting Perplexity Values"
        problem="What do different perplexity values indicate?"
        steps={[
          { formula: '\\text{PPL} \\approx 1', explanation: 'Perfect prediction. The model is certain about every token. (Only on memorized data.)' },
          { formula: '\\text{PPL} = 5\\text{-}10', explanation: 'Excellent. Typical for well-trained models on in-domain text.' },
          { formula: '\\text{PPL} = 10\\text{-}30', explanation: 'Good. Typical range for instruction-tuned models on general text.' },
          { formula: '\\text{PPL} > 100', explanation: 'Poor. Model struggles with this text. May indicate domain mismatch or training issues.' },
        ]}
        id="example-ppl-values"
      />

      <PythonCode
        title="compute_perplexity.py"
        code={`import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

def compute_perplexity(model, tokenizer, texts, max_length=2048, batch_size=4):
    """Compute perplexity on a list of texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encodings = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
            # Loss is averaged over non-padding tokens in each example
            loss = outputs.loss
            num_tokens = (encodings["attention_mask"].sum()).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

# Usage
model_name = "your-model-path"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Test texts (use domain-relevant text for meaningful evaluation)
test_texts = [
    "The transformer architecture uses self-attention to process sequences.",
    "Machine learning models learn patterns from training data.",
    "Python is a popular programming language for data science.",
]

ppl = compute_perplexity(model, tokenizer, test_texts)
print(f"Perplexity: {ppl:.2f}")`}
        id="code-perplexity"
      />

      <PythonCode
        title="compare_perplexity.py"
        code={`from datasets import load_dataset

def evaluate_on_dataset(model, tokenizer, dataset_name, split="test[:100]"):
    """Evaluate perplexity on a standard dataset."""
    dataset = load_dataset(dataset_name, split=split)

    # Extract text field
    if "text" in dataset.column_names:
        texts = dataset["text"]
    elif "content" in dataset.column_names:
        texts = dataset["content"]
    else:
        texts = [str(ex) for ex in dataset]

    # Filter empty texts
    texts = [t for t in texts if len(t.strip()) > 50]

    ppl = compute_perplexity(model, tokenizer, texts[:100])
    return ppl

# Compare base vs finetuned model
datasets_to_test = {
    "wikitext": "wikitext-2-raw-v1",
    # Add your domain-specific test set
}

print(f"{'Dataset':<20} {'Perplexity':>12}")
print("-" * 35)
for name, ds_name in datasets_to_test.items():
    try:
        ppl = evaluate_on_dataset(model, tokenizer, ds_name)
        print(f"{name:<20} {ppl:>12.2f}")
    except Exception as e:
        print(f"{name:<20} Error: {e}")

# Expected: finetuned model should have lower PPL on domain text
# but may have slightly higher PPL on general text (tradeoff)`}
        id="code-compare-ppl"
      />

      <NoteBlock
        type="intuition"
        title="Perplexity Limitations"
        content="Perplexity only measures how well the model predicts text, not how well it follows instructions or generates helpful responses. A model that memorizes its training data has very low perplexity but is useless. Always combine perplexity with task-specific evaluation."
        id="note-ppl-limitations"
      />

      <WarningBlock
        title="Perplexity Is Not Comparable Across Tokenizers"
        content="Perplexity depends on the tokenizer. A model with a 32K vocab and one with 128K vocab will have different perplexities on the same text, even if they are equally good. Only compare perplexity between models using the same tokenizer."
        id="warning-tokenizer-ppl"
      />
    </div>
  )
}
