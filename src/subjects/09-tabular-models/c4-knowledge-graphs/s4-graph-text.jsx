import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function GraphText() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Graph-to-Text and Text-to-Graph Generation</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Graph-to-text (G2T) converts structured graph data into fluent natural language, while
        text-to-graph (T2G) extracts structured graph representations from text. Together, these
        tasks form a bidirectional bridge between structured and unstructured knowledge, enabling
        knowledge graph construction, data-to-text generation, and graph-grounded dialogue.
      </p>

      <DefinitionBlock
        title="Graph-to-Text Generation"
        definition="Graph-to-text generation maps a subgraph $g = \{(h_1, r_1, t_1), \ldots, (h_k, r_k, t_k)\}$ to a natural language text $y$ that faithfully verbalizes all triples. The generation must be faithful ($y$ entails all triples), fluent (grammatically correct), and concise (no unnecessary repetition)."
        notation="g = subgraph, (h, r, t) = triple, y = generated text"
        id="def-g2t"
      />

      <h2 className="text-2xl font-semibold">Graph-to-Text with LLMs</h2>
      <p className="text-gray-700 dark:text-gray-300">
        LLMs can verbalize graph triples into fluent text by serializing the triples and
        prompting the model to generate a natural language description. The challenge is
        ensuring all triples are covered without hallucinating additional facts.
      </p>

      <PythonCode
        title="graph_to_text.py"
        code={`import json

# Graph-to-text: verbalize triples as natural language
def graph_to_text_prompt(triples, style="paragraph"):
    triple_str = "\\n".join(
        f"  ({h}, {r}, {t})" for h, r, t in triples
    )
    return f"""Convert the following knowledge graph triples into a natural
language {style}. Include ALL facts from the triples. Do not add
information not present in the triples.

Triples:
{triple_str}

Text:"""

# Example: biographical triples
bio_triples = [
    ("Ada Lovelace", "born_in", "London"),
    ("Ada Lovelace", "birth_year", "1815"),
    ("Ada Lovelace", "occupation", "mathematician"),
    ("Ada Lovelace", "known_for", "first computer program"),
    ("Ada Lovelace", "worked_with", "Charles Babbage"),
    ("Charles Babbage", "invented", "Analytical Engine"),
    ("Ada Lovelace", "parent", "Lord Byron"),
]

prompt = graph_to_text_prompt(bio_triples)
print(prompt)

# Expected output:
expected = (
    "Ada Lovelace was a mathematician born in London in 1815. "
    "She is known for writing the first computer program, developed "
    "during her work with Charles Babbage, the inventor of the "
    "Analytical Engine. She was the daughter of Lord Byron."
)
print(f"\\nExpected output:\\n{expected}")

# Graph-to-text for different styles
for style in ["paragraph", "bullet points", "formal biography"]:
    p = graph_to_text_prompt(bio_triples[:3], style=style)
    print(f"\\n--- Style: {style} ---")
    print(p[:100] + "...")`}
        id="code-g2t"
      />

      <ExampleBlock
        title="Faithfulness Evaluation"
        problem="Verify that a generated text faithfully represents the source triples without hallucination."
        steps={[
          { formula: '\\text{Triples: (Paris, capital\\_of, France), (Paris, population, 2.1M)}', explanation: 'Source graph contains exactly two facts.' },
          { formula: '\\text{Generated: "Paris, the capital of France, has 2.1M residents and is known for the Eiffel Tower."}', explanation: 'The text mentions the Eiffel Tower, which is not in the triples.' },
          { formula: '\\text{Faithfulness check: "Eiffel Tower" } \\notin \\text{ triples } \\to \\text{ hallucination}', explanation: 'Any fact in the text not traceable to a source triple is a hallucination.' },
          { formula: '\\text{Coverage check: both triples mentioned } \\to \\text{ complete}', explanation: 'All source triples must appear in the generated text for full coverage.' },
        ]}
        id="example-faithfulness"
      />

      <h2 className="text-2xl font-semibold">Text-to-Graph Extraction</h2>

      <PythonCode
        title="text_to_graph.py"
        code={`import json
import re

def text_to_graph_prompt(text, entity_types=None, relation_types=None):
    """Build a prompt for text-to-graph extraction."""
    constraints = ""
    if entity_types:
        constraints += f"Entity types: {', '.join(entity_types)}\\n"
    if relation_types:
        constraints += f"Relation types: {', '.join(relation_types)}\\n"

    return f"""Extract a knowledge graph from the text.

Text: "{text}"

{constraints}
Return JSON with:
- "entities": list of {{"name": str, "type": str}}
- "triples": list of {{"subject": str, "relation": str, "object": str}}

Ensure entity names in triples match exactly those in the entities list.
JSON:"""

text = """The University of Oxford, founded in 1096, is located in Oxford,
England. It is one of the oldest universities in the world. Stephen
Hawking studied at Oxford before moving to Cambridge for his PhD.
He later became Lucasian Professor of Mathematics at Cambridge."""

prompt = text_to_graph_prompt(
    text,
    entity_types=["PERSON", "ORGANIZATION", "LOCATION", "DATE"],
    relation_types=["located_in", "founded_in", "studied_at",
                    "role_at", "moved_to"]
)

# Expected output
expected = {
    "entities": [
        {"name": "University of Oxford", "type": "ORGANIZATION"},
        {"name": "Oxford", "type": "LOCATION"},
        {"name": "England", "type": "LOCATION"},
        {"name": "Stephen Hawking", "type": "PERSON"},
        {"name": "Cambridge", "type": "ORGANIZATION"},
    ],
    "triples": [
        {"subject": "University of Oxford", "relation": "founded_in", "object": "1096"},
        {"subject": "University of Oxford", "relation": "located_in", "object": "Oxford"},
        {"subject": "Oxford", "relation": "located_in", "object": "England"},
        {"subject": "Stephen Hawking", "relation": "studied_at", "object": "University of Oxford"},
        {"subject": "Stephen Hawking", "relation": "moved_to", "object": "Cambridge"},
        {"subject": "Stephen Hawking", "relation": "role_at", "object": "Cambridge"},
    ]
}
print(json.dumps(expected, indent=2))`}
        id="code-t2g"
      />

      <PythonCode
        title="graph_roundtrip.py"
        code={`# Round-trip evaluation: Text -> Graph -> Text
# Tests both extraction and generation quality

def roundtrip_eval(original_text, extracted_graph, regenerated_text):
    """Evaluate round-trip consistency."""
    # Count triples preserved
    num_triples = len(extracted_graph)

    # Simple word overlap as a proxy for semantic similarity
    orig_words = set(original_text.lower().split())
    regen_words = set(regenerated_text.lower().split())
    overlap = len(orig_words & regen_words)
    total = len(orig_words | regen_words)
    jaccard = overlap / total if total > 0 else 0

    return {
        "num_triples": num_triples,
        "jaccard_similarity": round(jaccard, 3),
        "original_length": len(original_text),
        "regenerated_length": len(regenerated_text),
    }

# Simulate round-trip
original = "Marie Curie won the Nobel Prize in Physics in 1903 and Chemistry in 1911."
graph = [
    ("Marie Curie", "won", "Nobel Prize in Physics"),
    ("Nobel Prize in Physics", "year", "1903"),
    ("Marie Curie", "won", "Nobel Prize in Chemistry"),
    ("Nobel Prize in Chemistry", "year", "1911"),
]
regenerated = "Marie Curie was awarded the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911."

metrics = roundtrip_eval(original, graph, regenerated)
print(f"Round-trip metrics: {json.dumps(metrics, indent=2)}")

print("Graph adjacency:", {h: [(r,t)] for h,r,t in graph})`}
        id="code-roundtrip"
      />

      <NoteBlock
        type="note"
        title="Benchmarks for Graph-Text Tasks"
        content="WebNLG is the primary benchmark for graph-to-text, containing 25,000+ (graph, text) pairs from DBpedia. KELM evaluates text-to-graph with Wikidata triples. GenWiki tests both directions. On WebNLG, fine-tuned T5-large achieves BLEU scores of ~65, while GPT-4 zero-shot reaches ~55. The gap is narrower on faithfulness metrics where LLMs excel."
        id="note-benchmarks"
      />

      <WarningBlock
        title="Graph Serialization Order Matters"
        content="The order in which triples are serialized affects generation quality. Grouping triples by subject entity produces more coherent text than random ordering. For text-to-graph, the extraction order can create cascading errors -- if an entity is missed early, all its relations will also be missed. Consider multiple extraction passes with different orderings."
        id="warning-order"
      />

    </div>
  )
}
