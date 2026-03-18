import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function NamedEntityRecognition() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Named Entity Recognition</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Named Entity Recognition (NER) identifies and classifies named entities in text into
        predefined categories such as persons, organizations, locations, dates, and more.
        It is a core information extraction task used in search, question answering, and
        knowledge graph construction.
      </p>

      <DefinitionBlock
        title="Named Entity Recognition (NER)"
        definition="NER is a sequence labeling task that identifies spans of text referring to real-world entities and classifies them into categories. Standard entity types include PERSON, ORGANIZATION (ORG), LOCATION (LOC), DATE, TIME, MONEY, and PERCENT."
        id="def-ner"
      />

      <h2 className="text-2xl font-semibold">BIO Tagging Scheme</h2>
      <p className="text-gray-700 dark:text-gray-300">
        NER is typically framed as a token-level classification task using the BIO (Beginning,
        Inside, Outside) tagging scheme. Each token receives a tag indicating whether it begins
        an entity (B-TYPE), continues one (I-TYPE), or is outside any entity (O).
      </p>

      <ExampleBlock
        title="BIO Tagging"
        problem="Apply BIO tags to: 'Barack Obama visited New York City on Friday'"
        steps={[
          { formula: 'Barack -> B-PER', explanation: 'Begins a PERSON entity.' },
          { formula: 'Obama -> I-PER', explanation: 'Continues the PERSON entity.' },
          { formula: 'visited -> O', explanation: 'Not part of any entity.' },
          { formula: 'New -> B-LOC', explanation: 'Begins a LOCATION entity.' },
          { formula: 'York -> I-LOC', explanation: 'Continues the LOCATION entity.' },
          { formula: 'City -> I-LOC', explanation: 'Still part of the LOCATION entity.' },
        ]}
        id="example-bio"
      />

      <PythonCode
        title="ner_spacy.py"
        code={`import spacy

nlp = spacy.load("en_core_web_sm")

text = """Apple Inc. was founded by Steve Jobs in Cupertino, California
in 1976. The company is now worth over $2.8 trillion and employs
more than 160,000 people worldwide."""

doc = nlp(text)

# Extract named entities
print(f"{'Entity':<25} {'Label':<12} {'Description'}")
print("-" * 60)
for ent in doc.ents:
    print(f"{ent.text:<25} {ent.label_:<12} {spacy.explain(ent.label_)}")

# Visualize entity spans with character offsets
print("\\nEntity spans:")
for ent in doc.ents:
    print(f"  [{ent.start_char}:{ent.end_char}] '{ent.text}' ({ent.label_})")

# Count entity types
from collections import Counter
type_counts = Counter(ent.label_ for ent in doc.ents)
print("\\nEntity type distribution:")
for label, count in type_counts.most_common():
    print(f"  {label}: {count}")`}
        id="code-ner-spacy"
      />

      <h2 className="text-2xl font-semibold">NER as Sequence Labeling</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Formally, NER finds the tag sequence <InlineMath math="\hat{y}_1^n" /> that maximizes:
      </p>

      <div className="my-4">
        <BlockMath math="\hat{y}_1^n = \arg\max_{y_1^n} P(y_1, \ldots, y_n \mid x_1, \ldots, x_n)" />
      </div>

      <p className="text-gray-700 dark:text-gray-300">
        Classical approaches use Conditional Random Fields (CRFs) to model the joint probability,
        capturing dependencies between adjacent tags (e.g., I-PER should not follow B-LOC).
      </p>

      <PythonCode
        title="ner_custom_rules.py"
        code={`import re

# Simple rule-based NER using regex patterns
patterns = {
    'EMAIL': r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b',
    'PHONE': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'URL': r'https?://[\w/\-?=%.]+\.[\w/\-?=%.]+',
    'DATE': r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
    'MONEY': r'\$[\d,]+\.?\d*',
}

text = """Contact john.doe@example.com or call 555-123-4567.
Visit https://openai.com for details. Meeting on 3/15/2024.
Budget: $1,250,000 allocated for Q2."""

print("Rule-based entity extraction:")
for entity_type, pattern in patterns.items():
    matches = re.findall(pattern, text)
    if matches:
        print(f"  {entity_type}: {matches}")

# Compare: spaCy finds semantic entities, regex finds patterns
# Both approaches are useful and often combined in practice`}
        id="code-ner-rules"
      />

      <NoteBlock
        type="tip"
        title="Evaluation Metrics for NER"
        content="NER is evaluated using entity-level precision, recall, and F1 score. An entity is correct only if both the span boundaries AND the type label match exactly. Token-level accuracy can be misleadingly high because most tokens are 'O' (outside any entity)."
        id="note-ner-eval"
      />

      <WarningBlock
        title="Nested and Overlapping Entities"
        content="Standard BIO tagging cannot handle nested entities. In 'New York University', 'New York' is a LOC inside the ORG 'New York University'. Specialized approaches like span-based models or multi-layer tagging are needed for nested NER."
        id="warning-nested"
      />

      <NoteBlock
        type="historical"
        title="NER Timeline"
        content="NER emerged from the MUC conferences (1990s). Early systems were rule-based. CRF-based taggers (Lafferty et al., 2001) dominated for a decade. BiLSTM-CRF models (Lample et al., 2016) set new benchmarks. Modern Transformer-based NER (BERT fine-tuning) achieves F1 scores above 92% on CoNLL-2003."
        id="note-ner-history"
      />
    </div>
  )
}
