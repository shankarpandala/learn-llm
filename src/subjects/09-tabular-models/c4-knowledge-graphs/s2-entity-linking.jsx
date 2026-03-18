import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function EntityLinking() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Entity Linking with LLMs</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Entity linking (EL) maps mentions in text to their corresponding entries in a knowledge
        base. It resolves ambiguity -- "Apple" could refer to the company, the fruit, or the
        record label -- by considering the surrounding context. LLMs have transformed this task
        by leveraging their world knowledge for disambiguation.
      </p>

      <DefinitionBlock
        title="Entity Linking"
        definition="Entity linking takes a text $d$ containing entity mentions $M = \{m_1, \ldots, m_k\}$ and maps each mention $m_i$ to an entity $e_i$ in a knowledge base $\mathcal{KB}$, or to NIL if the entity is not in the KB. Formally: $\text{EL}(m_i, d) \to e_i \in \mathcal{KB} \cup \{\text{NIL}\}$. The task combines mention detection, candidate generation, and entity disambiguation."
        notation="d = document, m_i = mention, e_i = entity, \mathcal{KB} = knowledge base"
        id="def-entity-linking"
      />

      <h2 className="text-2xl font-semibold">Entity Linking Pipeline</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Traditional entity linking follows a three-stage pipeline: mention detection (finding
        entity spans in text), candidate generation (retrieving possible KB entries), and
        disambiguation (selecting the correct entity based on context).
      </p>

      <PythonCode
        title="entity_linking_pipeline.py"
        code={`import re
from difflib import get_close_matches

# Simple knowledge base
knowledge_base = {
    "Q312": {"name": "Apple Inc.", "type": "company",
             "aliases": ["Apple", "Apple Computer"],
             "description": "American technology company"},
    "Q89": {"name": "apple", "type": "fruit",
            "aliases": ["apple fruit", "malus"],
            "description": "Edible fruit from apple trees"},
    "Q484523": {"name": "Apple Records", "type": "company",
                "aliases": ["Apple"],
                "description": "Record label founded by the Beatles"},
    "Q937": {"name": "Albert Einstein", "type": "person",
             "aliases": ["Einstein", "A. Einstein"],
             "description": "Theoretical physicist"},
    "Q7186": {"name": "Marie Curie", "type": "person",
              "aliases": ["Curie", "Maria Sklodowska"],
              "description": "Physicist and chemist, Nobel laureate"},
}

# Stage 1: Mention Detection (simplified with NER-like rules)
def detect_mentions(text):
    """Find capitalized phrases as potential entity mentions."""
    pattern = r'\\b[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*\\b'
    return [(m.group(), m.start(), m.end())
            for m in re.finditer(pattern, text)]

# Stage 2: Candidate Generation
def generate_candidates(mention, kb, max_candidates=5):
    """Find KB entries matching the mention."""
    candidates = []
    mention_lower = mention.lower()
    for qid, entity in kb.items():
        all_names = [entity["name"].lower()] + [a.lower() for a in entity["aliases"]]
        for name in all_names:
            if mention_lower in name or name in mention_lower:
                candidates.append((qid, entity))
                break
    return candidates

# Stage 3: Disambiguation (context-based scoring)
def disambiguate(mention, candidates, context, kb):
    """Score candidates based on context overlap."""
    context_words = set(context.lower().split())
    scores = []
    for qid, entity in candidates:
        desc_words = set(entity["description"].lower().split())
        overlap = len(context_words & desc_words)
        scores.append((qid, entity["name"], overlap))
    return sorted(scores, key=lambda x: -x[2])

# Full pipeline
text = "Apple announced new products at their Cupertino headquarters. Einstein would have loved the physics simulations."
mentions = detect_mentions(text)
print(f"Text: {text}\\n")

for mention_text, start, end in mentions:
    candidates = generate_candidates(mention_text, knowledge_base)
    if candidates:
        ranked = disambiguate(mention_text, candidates, text, knowledge_base)
        best = ranked[0] if ranked else None
        print(f"Mention: '{mention_text}' -> {best[1]} ({best[0]})")
    else:
        print(f"Mention: '{mention_text}' -> NIL (not in KB)")`}
        id="code-pipeline"
      />

      <ExampleBlock
        title="LLM-Based Entity Disambiguation"
        problem="Disambiguate 'Mercury' in the sentence: 'Mercury's orbit has the highest eccentricity of any planet.'"
        steps={[
          { formula: '\\text{Candidates: Mercury (planet), Mercury (element), Mercury (mythology)}', explanation: 'KB lookup returns multiple entities matching "Mercury".' },
          { formula: '\\text{Context clues: "orbit", "eccentricity", "planet"}', explanation: 'Surrounding words strongly suggest an astronomical context.' },
          { formula: '\\text{LLM prompt: "Which Mercury? planet/element/mythology. Context: orbit, planet"}', explanation: 'Frame disambiguation as a classification task for the LLM.' },
          { formula: '\\text{Output: Mercury (planet) with high confidence}', explanation: 'The LLM leverages both context and world knowledge to select the correct entity.' },
        ]}
        id="example-disambiguation"
      />

      <PythonCode
        title="llm_entity_linking.py"
        code={`import json

def llm_entity_linking_prompt(text, mentions, candidates_per_mention):
    """Build a prompt for LLM-based entity linking."""
    prompt = f"""Link each entity mention to the correct knowledge base entry.

Text: "{text}"

Mentions to link:
"""
    for mention, candidates in zip(mentions, candidates_per_mention):
        prompt += f"\\nMention: '{mention}'\\nCandidates:\\n"
        for qid, name, desc in candidates:
            prompt += f"  - {qid}: {name} ({desc})\\n"

    prompt += """
Return a JSON array of objects with "mention", "qid", and "confidence" fields.
If no candidate matches, use "NIL" for qid.
JSON:"""
    return prompt

# Example usage
text = "Apple CEO Tim Cook presented at WWDC in San Jose"
mentions = ["Apple", "Tim Cook", "WWDC", "San Jose"]
candidates = [
    [("Q312", "Apple Inc.", "tech company"),
     ("Q89", "apple", "fruit")],
    [("Q265", "Tim Cook", "CEO of Apple Inc.")],
    [("Q1630", "WWDC", "Apple developer conference")],
    [("Q16553", "San Jose", "city in California"),
     ("Q79984", "San Jose", "city in Costa Rica")],
]

prompt = llm_entity_linking_prompt(text, mentions, candidates)
print(prompt)

# Expected LLM output:
expected = [
    {"mention": "Apple", "qid": "Q312", "confidence": 0.98},
    {"mention": "Tim Cook", "qid": "Q265", "confidence": 0.99},
    {"mention": "WWDC", "qid": "Q1630", "confidence": 0.97},
    {"mention": "San Jose", "qid": "Q16553", "confidence": 0.95},
]
print("\\nExpected output:")
print(json.dumps(expected, indent=2))`}
        id="code-llm-el"
      />

      <NoteBlock
        type="note"
        title="Zero-Shot Entity Linking"
        content="LLMs enable zero-shot entity linking -- resolving entities without task-specific training data. By prompting with entity descriptions from the KB, GPT-4 achieves 85%+ accuracy on standard EL benchmarks (AIDA-CoNLL), approaching the performance of specialized fine-tuned models like BLINK and GENRE that require extensive training."
        id="note-zero-shot"
      />

      <WarningBlock
        title="NIL Entity Challenge"
        content="A significant fraction of entity mentions in real text refer to entities not present in any knowledge base (emerging entities, rare proper nouns). Models must learn to output NIL rather than forcing a match. Fine-tuned models often default to the most popular candidate; LLMs handle NIL detection better through reasoning but can still hallucinate non-existent KB entries."
        id="warning-nil"
      />

      <NoteBlock
        type="tip"
        title="Bi-Encoder for Scalable Candidate Generation"
        content="For large KBs with millions of entities, use a bi-encoder architecture: encode the mention with context and each entity independently, then use approximate nearest neighbor search (FAISS) for fast candidate retrieval. The LLM is only used for the final disambiguation step over the top-k candidates, keeping the system efficient."
        id="note-scalable"
      />
    </div>
  )
}
