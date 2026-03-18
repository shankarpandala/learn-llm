import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function WholeWordMasking() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Whole Word Masking and SpanBERT</h1>
      <p className="text-lg text-gray-300">
        Standard BERT masks individual WordPiece tokens, which can leak information when partial
        words remain visible. Whole Word Masking (WWM) and SpanBERT address this by masking
        entire words or contiguous spans, forcing the model to learn deeper contextual understanding.
      </p>

      <DefinitionBlock
        title="Whole Word Masking"
        definition="When a WordPiece subtoken is selected for masking, all subtokens belonging to the same word are also masked. If 'playing' is tokenized as ['play', '##ing'] and '##ing' is selected, both 'play' and '##ing' are masked together."
        notation="For word $w$ with subtokens $(t_1, t_2, \ldots, t_k)$: if any $t_i \in \mathcal{M}$, then $\forall j: t_j \in \mathcal{M}$."
        id="wwm-def"
      />

      <ExampleBlock
        title="Standard vs Whole Word Masking"
        problem="Compare masking for 'unbelievable' tokenized as ['un', '##bel', '##iev', '##able']."
        steps={[
          {
            formula: '\\text{Standard: } [\\text{un}, \\text{[MASK]}, \\text{##iev}, \\text{##able}]',
            explanation: 'Only ##bel is masked. The model can easily guess it from remaining subtokens.'
          },
          {
            formula: '\\text{WWM: } [\\text{[MASK]}, \\text{[MASK]}, \\text{[MASK]}, \\text{[MASK]}]',
            explanation: 'All subtokens of "unbelievable" are masked, forcing real prediction from context.'
          },
          {
            formula: '\\mathcal{L}_{\\text{WWM}} = -\\sum_{j=1}^{4} \\log P(t_j \\mid x_{\\backslash w})',
            explanation: 'The loss requires predicting all subtokens from context alone.'
          }
        ]}
        id="wwm-example"
      />

      <DefinitionBlock
        title="SpanBERT"
        definition="SpanBERT (Joshi et al., 2020) masks contiguous random spans of tokens. Span lengths are sampled from a geometric distribution $\\ell \\sim \\text{Geo}(p=0.2)$, clipped to $[1, 10]$, with mean span length of 3.8 tokens."
        notation="SpanBERT adds a Span Boundary Objective (SBO): $P(x_i \\mid x_{s-1}, x_{e+1}, p_{i}) $ where $s, e$ are span boundaries and $p_i$ is the relative position within the span."
        id="spanbert-def"
      />

      <NoteBlock
        type="intuition"
        title="Why Span Masking Works Better"
        content="Masking contiguous spans forces the model to reason about multi-token concepts rather than individual subwords. This is especially beneficial for tasks like coreference resolution, relation extraction, and question answering, where understanding spans of text is critical."
        id="span-masking-intuition"
      />

      <PythonCode
        title="whole_word_masking.py"
        code={`from transformers import BertTokenizer, BertForMaskedLM
import torch
import random

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

def whole_word_masking(text, mlm_prob=0.15):
    """Apply whole-word masking to tokenized text."""
    tokens = tokenizer.tokenize(text)

    # Group subtokens into words
    word_groups = []
    current_word = []
    for i, token in enumerate(tokens):
        if token.startswith("##"):
            current_word.append(i)
        else:
            if current_word:
                word_groups.append(current_word)
            current_word = [i]
    if current_word:
        word_groups.append(current_word)

    # Select ~15% of words to mask
    num_to_mask = max(1, int(len(word_groups) * mlm_prob))
    selected = random.sample(word_groups, num_to_mask)
    mask_indices = set(idx for group in selected for idx in group)

    # Apply 80/10/10 rule
    masked_tokens = tokens.copy()
    labels = [-100] * len(tokens)
    for idx in mask_indices:
        labels[idx] = tokenizer.convert_tokens_to_ids(tokens[idx])
        r = random.random()
        if r < 0.8:
            masked_tokens[idx] = "[MASK]"
        elif r < 0.9:
            masked_tokens[idx] = tokenizer.convert_ids_to_tokens(
                random.randint(0, tokenizer.vocab_size - 1)
            )
    return masked_tokens, labels

# Example
text = "The unbelievable performance surprised everyone"
masked, labels = whole_word_masking(text, mlm_prob=0.3)
print(f"Original tokens: {tokenizer.tokenize(text)}")
print(f"Masked tokens:   {masked}")
print(f"Labels (masked):  {[l for l in labels if l != -100]}")

# SpanBERT-style span masking
import numpy as np

def span_masking(tokens, mlm_prob=0.15, max_span=10, p=0.2):
    """Sample geometric-distribution spans for masking."""
    n = len(tokens)
    budget = int(n * mlm_prob)
    mask_indices = set()

    while len(mask_indices) < budget:
        span_len = min(np.random.geometric(p), max_span)
        start = random.randint(0, n - 1)
        for i in range(start, min(start + span_len, n)):
            mask_indices.add(i)
    return mask_indices

tokens = tokenizer.tokenize("The quick brown fox jumps over the lazy dog")
spans = span_masking(tokens, mlm_prob=0.30)
print(f"\\nSpan masking indices: {sorted(spans)}")
result = [("[MASK]" if i in spans else t) for i, t in enumerate(tokens)]
print(f"Span masked: {result}")`}
        id="wwm-code"
      />

      <WarningBlock
        title="Token Budget with WWM"
        content="With whole-word masking, the actual percentage of masked tokens may exceed 15% because masking one subtoken forces all sibling subtokens to be masked. Implementations typically adjust the word selection probability to keep the total masked token count near the target budget."
        id="wwm-budget-warning"
      />
    </div>
  )
}
