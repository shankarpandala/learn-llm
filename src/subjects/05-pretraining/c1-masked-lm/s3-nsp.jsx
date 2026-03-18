import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function NextSentencePrediction() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Next Sentence Prediction (NSP)</h1>
      <p className="text-lg text-gray-300">
        Next Sentence Prediction is BERT's secondary pretraining objective. It trains a binary
        classifier on the [CLS] token representation to determine whether two sentences appear
        consecutively in the original corpus, helping the model understand inter-sentence relationships.
      </p>

      <DefinitionBlock
        title="NSP Objective"
        definition="Given sentence pair $(A, B)$, NSP predicts a binary label $y \in \{\\text{IsNext}, \\text{NotNext}\}$. During training, 50% of pairs are actual consecutive sentences ($y = \\text{IsNext}$) and 50% are randomly sampled ($y = \\text{NotNext}$)."
        notation="$\mathcal{L}_{\\text{NSP}} = -[y \log \hat{y} + (1-y)\log(1-\hat{y})]$ where $\hat{y} = \sigma(W_{\\text{NSP}} \cdot h_{\\text{[CLS]}})$."
        id="nsp-def"
      />

      <ExampleBlock
        title="NSP Training Pair Construction"
        problem="Given corpus with sentences S1='The cat sat on the mat.' S2='It purred softly.' S3='Markets rose 2% today.', construct NSP pairs."
        steps={[
          {
            formula: '\\text{Positive: } (S_1, S_2) \\rightarrow \\text{IsNext}',
            explanation: 'S2 actually follows S1 in the corpus, so label is IsNext.'
          },
          {
            formula: '\\text{Negative: } (S_1, S_3) \\rightarrow \\text{NotNext}',
            explanation: 'S3 is randomly sampled from the corpus and does not follow S1.'
          },
          {
            formula: '\\text{Input format: [CLS] } S_A \\text{ [SEP] } S_B \\text{ [SEP]}',
            explanation: 'Both pairs are formatted with special tokens and segment embeddings.'
          }
        ]}
        id="nsp-pairs-example"
      />

      <NoteBlock
        type="intuition"
        title="Purpose of NSP"
        content="Many downstream tasks like question answering and natural language inference require understanding relationships between sentence pairs. NSP was designed to teach the model sentence-level coherence. The [CLS] token's representation captures this pair relationship."
        id="nsp-purpose"
      />

      <PythonCode
        title="nsp_example.py"
        code={`from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

# Positive pair (consecutive sentences)
sentence_a = "The weather is beautiful today."
sentence_b = "Let's go for a walk in the park."
inputs_pos = tokenizer(sentence_a, sentence_b, return_tensors="pt")

# Negative pair (random sentences)
sentence_c = "Quantum computing uses qubits."
inputs_neg = tokenizer(sentence_a, sentence_c, return_tensors="pt")

with torch.no_grad():
    # Positive pair
    out_pos = model(**inputs_pos)
    logits_pos = out_pos.logits  # [batch, 2]
    prob_pos = torch.softmax(logits_pos, dim=-1)
    print(f"Positive pair - IsNext prob: {prob_pos[0, 0]:.4f}")

    # Negative pair
    out_neg = model(**inputs_neg)
    logits_neg = out_neg.logits
    prob_neg = torch.softmax(logits_neg, dim=-1)
    print(f"Negative pair - IsNext prob: {prob_neg[0, 0]:.4f}")

# NSP loss computation
labels_pos = torch.tensor([0])  # 0 = IsNext
labels_neg = torch.tensor([1])  # 1 = NotNext

out_with_loss = model(**inputs_pos, labels=labels_pos)
print(f"NSP loss (positive pair): {out_with_loss.loss.item():.4f}")

out_with_loss = model(**inputs_neg, labels=labels_neg)
print(f"NSP loss (negative pair): {out_with_loss.loss.item():.4f}")

# Full BERT pretraining combines both objectives
# Total loss = MLM_loss + NSP_loss
print("\\nNote: BERT pretraining loss = L_MLM + L_NSP")`}
        id="nsp-code"
      />

      <WarningBlock
        title="NSP Was Later Found to Be Less Useful"
        content="RoBERTa (Liu et al., 2019) showed that removing NSP and training with full-length sequences actually improves performance. The topic mismatch from random sentence sampling made NSP too easy -- the model could rely on topic signals rather than coherence. ALBERT replaced NSP with Sentence Order Prediction (SOP), which uses consecutive sentences in both orders."
        id="nsp-criticism"
      />

      <NoteBlock
        type="historical"
        title="Evolution Beyond NSP"
        content="RoBERTa dropped NSP entirely and used dynamic masking. ALBERT introduced SOP where both sentences come from the same document but may be swapped. SpanBERT also dropped NSP and found improvements. The consensus is that NSP provides marginal or negative benefit compared to longer training with better objectives."
        id="nsp-evolution"
      />
    </div>
  )
}
