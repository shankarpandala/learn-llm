import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function RelationExtraction() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Relation Extraction with LLMs</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Relation extraction (RE) identifies semantic relationships between entities mentioned
        in text, producing structured triples that can populate knowledge graphs. LLMs have
        dramatically simplified this task, enabling zero-shot extraction across domains without
        predefined relation schemas.
      </p>

      <DefinitionBlock
        title="Relation Extraction"
        definition="Given a sentence $s$ containing entity mentions $e_1$ and $e_2$, relation extraction identifies the relation $r \in \mathcal{R} \cup \{\text{NoRelation}\}$ that holds between them, producing a triple $(e_1, r, e_2)$. In open RE, the relation set $\mathcal{R}$ is not predefined. In closed RE, $\mathcal{R}$ is a fixed schema."
        notation="s = sentence, e_1, e_2 = entities, r = relation, \mathcal{R} = relation set"
        id="def-relation-extraction"
      />

      <h2 className="text-2xl font-semibold">Closed-Domain Relation Extraction</h2>
      <p className="text-gray-700 dark:text-gray-300">
        In closed-domain RE, the model must classify entity pairs into a predefined set of
        relations. This is common in domain-specific applications like biomedical literature
        mining or financial analysis.
      </p>

      <PythonCode
        title="closed_re.py"
        code={`import json

# Predefined relation schema
RELATIONS = [
    "founded_by", "headquartered_in", "CEO_of", "acquired_by",
    "subsidiary_of", "partnership_with", "competitor_of",
    "invested_in", "no_relation"
]

def closed_re_prompt(text, entity1, entity2, relations):
    return f"""Extract the relationship between the two entities.

Text: "{text}"
Entity 1: {entity1}
Entity 2: {entity2}

Possible relations: {', '.join(relations)}

Return JSON with "entity1", "relation", "entity2", "confidence".
JSON:"""

# Examples
examples = [
    ("Microsoft acquired Activision Blizzard for $69 billion in 2023.",
     "Microsoft", "Activision Blizzard"),
    ("Sundar Pichai serves as the CEO of Alphabet Inc.",
     "Sundar Pichai", "Alphabet Inc."),
    ("Amazon and Google are major competitors in cloud computing.",
     "Amazon", "Google"),
]

for text, e1, e2 in examples:
    prompt = closed_re_prompt(text, e1, e2, RELATIONS)
    print(f"Text: {text}")
    print(f"  {e1} --[?]--> {e2}\\n")

# Expected outputs:
# Microsoft --[acquired_by]--> Activision Blizzard (or reverse)
# Sundar Pichai --[CEO_of]--> Alphabet Inc.
# Amazon --[competitor_of]--> Google`}
        id="code-closed-re"
      />

      <h2 className="text-2xl font-semibold">Open Information Extraction</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Open information extraction (OpenIE) does not require a predefined relation schema.
        The model extracts arbitrary relation phrases directly from the text, enabling
        knowledge discovery from unstructured sources.
      </p>

      <PythonCode
        title="open_ie_with_llm.py"
        code={`import json

def open_ie_prompt(text):
    return f"""Extract all factual relationships from the text as triples.

Text: "{text}"

Return a JSON array of triples. Each triple has:
- "subject": the entity performing or being described
- "relation": the relationship phrase
- "object": the entity being related to
- "confidence": float between 0 and 1

Extract ALL relationships, including implicit ones.
JSON:"""

text = """Marie Curie was born in Warsaw, Poland in 1867. She moved to
Paris to study at the Sorbonne. In 1903, she became the first woman
to win a Nobel Prize, sharing it with her husband Pierre Curie and
Henri Becquerel for their work on radioactivity."""

prompt = open_ie_prompt(text)
print(prompt)

# Expected LLM output:
expected_triples = [
    {"subject": "Marie Curie", "relation": "born_in", "object": "Warsaw, Poland", "confidence": 0.99},
    {"subject": "Marie Curie", "relation": "birth_year", "object": "1867", "confidence": 0.99},
    {"subject": "Marie Curie", "relation": "moved_to", "object": "Paris", "confidence": 0.98},
    {"subject": "Marie Curie", "relation": "studied_at", "object": "Sorbonne", "confidence": 0.97},
    {"subject": "Marie Curie", "relation": "won", "object": "Nobel Prize", "confidence": 0.99},
    {"subject": "Marie Curie", "relation": "spouse", "object": "Pierre Curie", "confidence": 0.95},
    {"subject": "Marie Curie", "relation": "first_woman_to", "object": "win Nobel Prize", "confidence": 0.98},
    {"subject": "Nobel Prize", "relation": "shared_with", "object": "Henri Becquerel", "confidence": 0.96},
    {"subject": "Nobel Prize", "relation": "field", "object": "radioactivity", "confidence": 0.94},
]

print("\\nExtracted triples:")
for t in expected_triples:
    print(f"  ({t['subject']}, {t['relation']}, {t['object']}) [{t['confidence']:.2f}]")`}
        id="code-open-ie"
      />

      <ExampleBlock
        title="Document-Level Relation Extraction"
        problem="Extract relations that span multiple sentences in a document, requiring coreference resolution."
        steps={[
          { formula: '\\text{S1: "Elon Musk founded SpaceX in 2002."}', explanation: 'Direct relation: (Elon Musk, founded, SpaceX).' },
          { formula: '\\text{S2: "The company launched Falcon 9 in 2010."}', explanation: '"The company" refers to SpaceX (coreference). Extract: (SpaceX, launched, Falcon 9).' },
          { formula: '\\text{S3: "He also leads Tesla, which produces electric vehicles."}', explanation: '"He" refers to Elon Musk. Extract: (Elon Musk, leads, Tesla), (Tesla, produces, electric vehicles).' },
          { formula: '\\text{Cross-sentence: (Elon Musk, founded, SpaceX) + (SpaceX, launched, Falcon 9)}', explanation: 'Chaining relations across sentences enables multi-hop reasoning in the resulting KG.' },
        ]}
        id="example-doc-level"
      />

      <PythonCode
        title="re_with_constraints.py"
        code={`# Constrained relation extraction with type checking
ENTITY_TYPES = {
    "PERSON": ["founded_by", "CEO_of", "born_in", "spouse_of"],
    "ORGANIZATION": ["headquartered_in", "subsidiary_of", "acquired_by"],
    "LOCATION": [],  # locations are typically objects, not subjects
}

RELATION_CONSTRAINTS = {
    "founded_by": {"subject": "ORGANIZATION", "object": "PERSON"},
    "CEO_of": {"subject": "PERSON", "object": "ORGANIZATION"},
    "born_in": {"subject": "PERSON", "object": "LOCATION"},
    "headquartered_in": {"subject": "ORGANIZATION", "object": "LOCATION"},
    "acquired_by": {"subject": "ORGANIZATION", "object": "ORGANIZATION"},
}

def validate_triple(subject, relation, obj, entity_types):
    """Check if a triple satisfies type constraints."""
    if relation not in RELATION_CONSTRAINTS:
        return True  # no constraints defined
    constraints = RELATION_CONSTRAINTS[relation]
    subj_type = entity_types.get(subject)
    obj_type = entity_types.get(obj)
    valid = True
    if subj_type and subj_type != constraints["subject"]:
        valid = False
    if obj_type and obj_type != constraints["object"]:
        valid = False
    return valid

# Validate extracted triples
entity_types = {
    "Apple Inc.": "ORGANIZATION",
    "Tim Cook": "PERSON",
    "Cupertino": "LOCATION",
}

triples = [
    ("Tim Cook", "CEO_of", "Apple Inc."),       # valid
    ("Apple Inc.", "headquartered_in", "Cupertino"),  # valid
    ("Cupertino", "CEO_of", "Tim Cook"),         # invalid
]

for s, r, o in triples:
    valid = validate_triple(s, r, o, entity_types)
    status = "VALID" if valid else "INVALID"
    print(f"  [{status}] ({s}, {r}, {o})")`}
        id="code-constraints"
      />

      <NoteBlock
        type="note"
        title="Benchmarks and State of the Art"
        content="TACRED and DocRED are the primary benchmarks for sentence-level and document-level RE respectively. GPT-4 achieves ~75% F1 on TACRED in zero-shot, compared to ~72% for supervised BERT-based models. On DocRED (requiring cross-sentence reasoning), supervised models still lead at ~65% F1, with LLMs at ~58% due to the long-context challenge."
        id="note-benchmarks"
      />

      <WarningBlock
        title="Relation Hallucination"
        content="LLMs can hallucinate plausible-sounding but incorrect relations, especially for entities they have strong prior knowledge about. For example, given text about a lesser-known 'John Smith', the model might infer relations from a famous John Smith. Always require textual evidence for extracted relations and implement confidence thresholds."
        id="warning-hallucination"
      />

      <NoteBlock
        type="tip"
        title="Iterative Extraction"
        content="For complex documents, use iterative extraction: first pass identifies entities, second pass extracts pairwise relations for high-confidence entity pairs, third pass resolves conflicts and fills gaps. This multi-pass approach improves recall without sacrificing precision, and each pass can use a different prompt tailored to its stage."
        id="note-iterative"
      />
    </div>
  )
}
