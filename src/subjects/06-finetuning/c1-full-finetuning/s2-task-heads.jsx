import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function TaskHeads() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Task Heads: Classification and Token Classification</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        A task head is a lightweight neural network layer added on top of a pretrained backbone.
        Different downstream tasks require different head architectures. The most common are
        sequence classification heads (sentiment, topic) and token classification heads (NER,
        POS tagging). The backbone provides rich contextualized representations while the head
        maps them to task-specific outputs.
      </p>

      <DefinitionBlock
        title="Task Head"
        definition="A task head is a small neural network (typically one or two linear layers) appended to a pretrained encoder or decoder. It projects the hidden representations $h \in \mathbb{R}^d$ to the label space $\mathbb{R}^C$ where $C$ is the number of classes. For sequence classification, the [CLS] token representation is used; for token classification, every token's representation is projected independently."
        id="def-task-head"
      />

      <h2 className="text-2xl font-semibold">Sequence Classification</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Sequence classification assigns a single label to an entire input sequence. The model
        typically uses a pooled representation (the [CLS] token in BERT-style models) and feeds
        it through a linear classifier.
      </p>

      <ExampleBlock
        title="Classification Head Architecture"
        problem="Describe the forward pass of a sequence classification head on top of BERT."
        steps={[
          { formula: 'h_{\\text{CLS}} = \\text{BERT}(x)[0][:,0,:]', explanation: 'Extract the [CLS] token hidden state from the last layer, shape (batch, hidden_dim).' },
          { formula: 'h_{\\text{drop}} = \\text{Dropout}(h_{\\text{CLS}}, p=0.1)', explanation: 'Apply dropout for regularization during finetuning.' },
          { formula: '\\text{logits} = W h_{\\text{drop}} + b, \\quad W \\in \\mathbb{R}^{C \\times d}', explanation: 'Project to the number of classes C via a linear layer.' },
          { formula: '\\mathcal{L} = \\text{CrossEntropy}(\\text{logits}, y)', explanation: 'Compute cross-entropy loss against ground-truth labels.' },
        ]}
        id="example-cls-head"
      />

      <PythonCode
        title="sequence_classification.py"
        code={`from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2-class sentiment classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

# Inspect the classification head
print(model.classifier)
# Linear(in_features=768, out_features=2, bias=True)

# Forward pass
inputs = tokenizer("This movie was fantastic!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    print(f"Logits: {logits}")
    print(f"Probabilities: {probs}")
    print(f"Predicted class: {torch.argmax(probs, dim=-1).item()}")`}
        id="code-seq-cls"
      />

      <h2 className="text-2xl font-semibold">Token Classification</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Token classification assigns a label to each token independently. This is used for
        Named Entity Recognition (NER), Part-of-Speech tagging, and extractive question
        answering. Every token's hidden state is projected through the classification head.
      </p>

      <PythonCode
        title="token_classification_ner.py"
        code={`from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline

model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# The token classification head: projects each token to label space
print(model.classifier)
# Linear(in_features=1024, out_features=9, bias=True)
# 9 labels: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC

# Use pipeline for easy inference
ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
results = ner("Hugging Face is based in New York City.")
for entity in results:
    print(f"  {entity['word']:15s} | {entity['entity_group']:5s} | score={entity['score']:.3f}")`}
        id="code-token-cls"
      />

      <NoteBlock
        type="tip"
        title="Subword Alignment for Token Classification"
        content="When using subword tokenizers, a single word may be split into multiple tokens. For token classification, labels are typically assigned only to the first subword of each word. The remaining subwords are ignored in loss computation using a special label index (commonly -100 in PyTorch)."
        id="note-subword-alignment"
      />

      <WarningBlock
        title="Head Initialization Matters"
        content="The task head is initialized randomly while the backbone is pretrained. This mismatch means the randomly-initialized head produces large, noisy gradients early in training that can disrupt the pretrained backbone. Using a warmup schedule and small learning rate for the backbone helps mitigate this."
        id="warning-head-init"
      />
    </div>
  )
}
