import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function MaskedPrediction() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Masked Language Modeling (MLM)</h1>
      <p className="text-lg text-gray-300">
        Masked Language Modeling is BERT's primary pretraining objective. By randomly masking tokens
        and training the model to predict them from bidirectional context, MLM learns deep contextual
        representations without the constraint of unidirectional attention.
      </p>

      <DefinitionBlock
        title="MLM Objective"
        definition="Given a sequence $x = (x_1, \ldots, x_n)$, randomly select 15% of positions $\mathcal{M}$. The MLM loss is $\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i \mid x_{\backslash \mathcal{M}})$, where $x_{\backslash \mathcal{M}}$ denotes the corrupted input."
        notation="The masking follows: 80% replaced with [MASK], 10% replaced with a random token, 10% kept unchanged."
        id="mlm-def"
      />

      <TheoremBlock
        title="MLM Loss as Cross-Entropy"
        statement="For each masked position $i \in \mathcal{M}$, the model outputs logits $z_i \in \mathbb{R}^{|V|}$ over the vocabulary. The per-token MLM loss is the standard cross-entropy: $\mathcal{L}_i = -\log \frac{\exp(z_i[x_i])}{\sum_{v \in V} \exp(z_i[v])}$."
        proof="The total MLM loss averages over all masked positions: $\mathcal{L}_{\text{MLM}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \mathcal{L}_i$. This is equivalent to categorical cross-entropy between the predicted distribution and the one-hot target."
        id="mlm-loss-theorem"
      />

      <ExampleBlock
        title="Masking Procedure"
        problem="Given 'The cat sat on the mat', apply 15% masking with the 80/10/10 rule."
        steps={[
          {
            formula: '\\text{Selected position: } i = 2 \\text{ (\"cat\")}',
            explanation: '15% of 6 tokens rounds to ~1 token selected for masking.'
          },
          {
            formula: 'P(\\text{[MASK]}) = 0.8, \\; P(\\text{random}) = 0.1, \\; P(\\text{keep}) = 0.1',
            explanation: 'With 80% probability: "The [MASK] sat on the mat". With 10%: "The dog sat on the mat". With 10%: unchanged.'
          },
          {
            formula: '\\text{Target: predict } x_2 = \\text{\"cat\"}',
            explanation: 'Regardless of the corruption strategy, the model must recover the original token.'
          }
        ]}
        id="masking-example"
      />

      <NoteBlock
        type="intuition"
        title="Why the 80/10/10 Split?"
        content="If we always used [MASK], the model would never see [MASK] during fine-tuning, creating a pretrain-finetune mismatch. Keeping 10% unchanged and 10% random forces the model to maintain good representations for all positions, not just masked ones."
        id="masking-split-note"
      />

      <PythonCode
        title="mlm_training.py"
        code={`from transformers import (
    BertTokenizer, BertForMaskedLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from datasets import load_dataset
import torch

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Manual MLM example
text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"].clone()

# Mask "cat" (token index 2)
masked_ids = input_ids.clone()
masked_ids[0, 2] = tokenizer.mask_token_id  # [MASK] = 103

# Forward pass
with torch.no_grad():
    outputs = model(input_ids=masked_ids)
    logits = outputs.logits  # [1, seq_len, vocab_size]

# Prediction at masked position
predicted_id = logits[0, 2].argmax(dim=-1)
print(f"Original: {tokenizer.decode(input_ids[0, 2])}")
print(f"Predicted: {tokenizer.decode(predicted_id)}")

# Compute MLM loss manually
loss_fn = torch.nn.CrossEntropyLoss()
labels = torch.full_like(input_ids, -100)  # -100 = ignore
labels[0, 2] = input_ids[0, 2]  # Only compute loss at masked pos
loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
print(f"MLM loss at masked position: {loss.item():.4f}")

# Use DataCollator for automatic masking
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Example batch
batch = data_collator([inputs])
print(f"Masked input: {tokenizer.decode(batch['input_ids'][0])}")
print(f"Labels (non -100): {(batch['labels'][0] != -100).sum().item()} tokens masked")`}
        id="mlm-code"
      />

      <WarningBlock
        title="MLM Creates Pretrain-Finetune Discrepancy"
        content="During pretraining, 15% of tokens are corrupted. During fine-tuning, no tokens are masked. This distribution mismatch can slightly hurt performance. Models like XLNet address this with permutation language modeling, and ELECTRA uses replaced token detection instead."
        id="mlm-discrepancy-warning"
      />
    </div>
  )
}
