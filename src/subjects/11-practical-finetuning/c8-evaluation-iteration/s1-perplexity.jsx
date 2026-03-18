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
      <h1 className="text-3xl font-bold">Perplexity & Loss-Based Evaluation</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Perplexity is the most fundamental metric for evaluating language models. It measures
        how surprised the model is by a held-out test set. Lower perplexity means the model
        assigns higher probability to the correct continuations, indicating better language
        modeling quality.
      </p>

      <DefinitionBlock
        title="Perplexity"
        definition="Perplexity is the exponentiated average negative log-likelihood of a sequence. For a token sequence of length $N$, perplexity is $\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\log p(x_i \mid x_{<i})\right)$. It represents the effective branching factor: a perplexity of 10 means the model is as uncertain as choosing uniformly among 10 tokens."
        notation="PPL, pp"
        id="def-perplexity"
      />

      <ExampleBlock
        title="Interpreting Perplexity Scores"
        problem="How do you compare perplexity between a base model and a fine-tuned model?"
        steps={[
          { formula: '\\text{PPL}_{\\text{base}} = 8.2 \\text{ on domain data}', explanation: 'Base model perplexity on your target domain evaluation set.' },
          { formula: '\\text{PPL}_{\\text{finetuned}} = 4.1 \\text{ on domain data}', explanation: 'Fine-tuned model achieves lower perplexity, meaning better fit to domain.' },
          { formula: '\\Delta\\text{PPL} = 8.2 - 4.1 = 4.1', explanation: 'A 50% reduction in perplexity indicates significant domain adaptation.' },
          { formula: '\\text{PPL}_{\\text{finetuned}} = 12.5 \\text{ on general data}', explanation: 'Check general-domain perplexity too -- if it rises sharply, the model has overfit.' },
        ]}
        id="example-ppl-comparison"
      />

      <PythonCode
        title="compute_perplexity.py"
        code={`import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import math

def compute_perplexity(model_path, eval_texts, max_length=2048):
    """Compute perplexity of a model on evaluation texts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for text in eval_texts:
        encodings = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
            neg_log_likelihood = outputs.loss
            num_tokens = encodings["input_ids"].size(1)

        total_loss += neg_log_likelihood.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return {"perplexity": perplexity, "avg_loss": avg_loss, "total_tokens": total_tokens}

# Usage
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
eval_texts = [t for t in dataset["text"] if len(t) > 50][:200]

base_results = compute_perplexity("meta-llama/Llama-3.1-8B", eval_texts)
ft_results = compute_perplexity("./my-finetuned-model", eval_texts)

print(f"Base model PPL:      {base_results['perplexity']:.2f}")
print(f"Finetuned model PPL: {ft_results['perplexity']:.2f}")
print(f"Improvement:         {base_results['perplexity'] - ft_results['perplexity']:.2f}")`}
        id="code-compute-ppl"
      />

      <PythonCode
        title="sliding_window_perplexity.py"
        code={`import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

def sliding_window_perplexity(model_path, text, stride=512, max_length=2048):
    """Compute perplexity using sliding window for long texts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end
        input_ids = encodings.input_ids[:, begin:end].to(device)

        target_ids = input_ids.clone()
        target_ids[:, :-target_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nll = outputs.loss * target_len

        nlls.append(nll.item())
        prev_end = end
        if end == seq_len:
            break

    ppl = math.exp(sum(nlls) / prev_end)
    return ppl

# Compare base vs finetuned on a long document
long_text = open("test_document.txt").read()
print(f"PPL: {sliding_window_perplexity('./my-model', long_text):.2f}")`}
        id="code-sliding-window"
      />

      <NoteBlock
        type="intuition"
        title="Perplexity Is Not Everything"
        content="Lower perplexity means the model predicts the evaluation text better, but it does not always correlate with downstream task quality. A model fine-tuned on code may have worse perplexity on general English text but be dramatically better at writing code. Always pair perplexity with task-specific evaluations."
        id="note-ppl-limits"
      />

      <WarningBlock
        title="Tokenizer Mismatch Invalidates Perplexity"
        content="Perplexity comparisons are only meaningful when both models use the same tokenizer. Different tokenizers produce different token counts for the same text, making perplexity values incomparable across model families (e.g., comparing LLaMA vs Mistral perplexity directly is misleading)."
        id="warning-tokenizer-mismatch"
      />

      <NoteBlock
        type="tip"
        title="Track Loss Curves During Training"
        content="Monitor both training loss and validation loss during fine-tuning. If training loss decreases but validation loss increases, the model is overfitting. Save checkpoints at the lowest validation loss point, not at the end of training."
        id="note-loss-curves"
      />
    </div>
  )
}
