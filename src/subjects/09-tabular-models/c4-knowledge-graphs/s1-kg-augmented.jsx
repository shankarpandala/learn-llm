import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function KGAugmentedLLMs() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Knowledge Graph-Augmented LLMs</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Knowledge graphs (KGs) store factual knowledge as structured triples, while LLMs
        excel at language understanding and reasoning. KG-augmented LLMs combine these
        strengths: the KG provides a grounded, verifiable knowledge base, and the LLM
        provides flexible natural language reasoning over that knowledge.
      </p>

      <DefinitionBlock
        title="Knowledge Graph"
        definition="A knowledge graph $\mathcal{G} = (\mathcal{E}, \mathcal{R}, \mathcal{T})$ consists of entities $\mathcal{E}$, relations $\mathcal{R}$, and triples $\mathcal{T} \subseteq \mathcal{E} \times \mathcal{R} \times \mathcal{E}$. Each triple $(h, r, t)$ represents a fact: head entity $h$ is related to tail entity $t$ by relation $r$. For example, (Einstein, bornIn, Ulm)."
        notation="\mathcal{E} = entities, \mathcal{R} = relations, \mathcal{T} = triples, (h, r, t) = head-relation-tail"
        id="def-kg"
      />

      <h2 className="text-2xl font-semibold">Retrieval-Augmented KG Integration</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The most common pattern retrieves relevant subgraph from the KG based on the query,
        serializes the triples into text, and includes them in the LLM prompt. This grounds
        the model's response in factual knowledge.
      </p>

      <PythonCode
        title="kg_augmented_qa.py"
        code={`import json
from collections import defaultdict

# Simple knowledge graph representation
class KnowledgeGraph:
    def __init__(self):
        self.triples = []
        self.entity_index = defaultdict(list)

    def add_triple(self, head, relation, tail):
        triple = (head, relation, tail)
        self.triples.append(triple)
        self.entity_index[head.lower()].append(triple)
        self.entity_index[tail.lower()].append(triple)

    def query_entity(self, entity, max_hops=2):
        """Retrieve subgraph around an entity up to max_hops."""
        visited = set()
        result = []
        frontier = {entity.lower()}

        for hop in range(max_hops):
            next_frontier = set()
            for e in frontier:
                if e in visited:
                    continue
                visited.add(e)
                for h, r, t in self.entity_index.get(e, []):
                    result.append((h, r, t))
                    next_frontier.add(h.lower())
                    next_frontier.add(t.lower())
            frontier = next_frontier - visited

        return result

# Build a small KG
kg = KnowledgeGraph()
kg.add_triple("Albert Einstein", "bornIn", "Ulm")
kg.add_triple("Albert Einstein", "field", "Theoretical Physics")
kg.add_triple("Albert Einstein", "award", "Nobel Prize in Physics")
kg.add_triple("Albert Einstein", "knownFor", "Theory of Relativity")
kg.add_triple("Ulm", "country", "Germany")
kg.add_triple("Ulm", "locatedIn", "Baden-Württemberg")
kg.add_triple("Nobel Prize in Physics", "year", "1921")

# Retrieve relevant triples
question = "Where was Einstein born and in which country?"
subgraph = kg.query_entity("Albert Einstein", max_hops=2)

# Serialize triples for the LLM prompt
def triples_to_text(triples):
    return "\\n".join(f"- {h} --[{r}]--> {t}" for h, r, t in triples)

prompt = f"""Use the following knowledge to answer the question.

Knowledge:
{triples_to_text(subgraph)}

Question: {question}
Answer:"""

print(prompt)
# The LLM can now answer: "Einstein was born in Ulm, Germany"
# grounded in the KG triples`}
        id="code-kg-qa"
      />

      <ExampleBlock
        title="KG-Augmented Reasoning Pipeline"
        problem="Answer 'Did Einstein win a Nobel Prize before or after publishing general relativity?' using a KG."
        steps={[
          { formula: '\\text{Entity linking: "Einstein" } \\to \\text{Albert Einstein}', explanation: 'Map the question mention to the canonical KG entity.' },
          { formula: '\\text{Retrieve: (Einstein, award, Nobel Prize), (Nobel Prize, year, 1921)}', explanation: 'Fetch triples about Einstein and Nobel Prize.' },
          { formula: '\\text{Retrieve: (Einstein, published, General Relativity), (General Relativity, year, 1915)}', explanation: 'Fetch triples about general relativity publication.' },
          { formula: '\\text{LLM reasons: 1915 < 1921 } \\to \\text{"after publishing general relativity"}', explanation: 'The LLM performs temporal reasoning over the retrieved facts.' },
        ]}
        id="example-reasoning"
      />

      <PythonCode
        title="kg_with_embeddings.py"
        code={`import numpy as np

# KG entity/relation embeddings for semantic retrieval
class KGEmbeddings:
    def __init__(self, dim=64):
        self.dim = dim
        self.entity_embeddings = {}
        self.relation_embeddings = {}

    def add_entity(self, name, embedding=None):
        if embedding is None:
            embedding = np.random.randn(self.dim)
            embedding = embedding / np.linalg.norm(embedding)
        self.entity_embeddings[name] = embedding

    def add_relation(self, name, embedding=None):
        if embedding is None:
            embedding = np.random.randn(self.dim)
            embedding = embedding / np.linalg.norm(embedding)
        self.relation_embeddings[name] = embedding

    def find_similar_entities(self, query_embedding, top_k=5):
        scores = {}
        for name, emb in self.entity_embeddings.items():
            scores[name] = np.dot(query_embedding, emb)
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]

# Build KG embeddings (in practice, trained with TransE/RotatE)
kge = KGEmbeddings(dim=64)
entities = ["Albert Einstein", "Isaac Newton", "Marie Curie",
            "Physics", "Chemistry", "Nobel Prize"]
for e in entities:
    kge.add_entity(e)

# Semantic search: find entities related to a query
query = kge.entity_embeddings["Albert Einstein"]
similar = kge.find_similar_entities(query, top_k=3)
print("Entities similar to Einstein:")
for name, score in similar:
    print(f"  {name}: {score:.3f}")

# TransE scoring: h + r ≈ t for valid triples
# score(h, r, t) = -||h + r - t||
h = kge.entity_embeddings["Albert Einstein"]
r = np.random.randn(64)  # "field" relation embedding
t = kge.entity_embeddings["Physics"]
score = -np.linalg.norm(h + r - t)
print(f"\\nTransE score (Einstein, field, Physics): {score:.3f}")`}
        id="code-embeddings"
      />

      <NoteBlock
        type="note"
        title="Major Knowledge Graphs"
        content="Wikidata contains 100M+ entities and 1.5B+ triples covering general knowledge. Freebase (now deprecated, absorbed into Wikidata) powered Google's Knowledge Graph. Domain-specific KGs include UMLS (medical), Gene Ontology (biology), and ConceptNet (commonsense). These structured resources complement LLMs' parametric knowledge with explicit, updatable facts."
        id="note-major-kgs"
      />

      <WarningBlock
        title="KG Incompleteness"
        content="Knowledge graphs are inherently incomplete -- they only contain explicitly stated facts. The open-world assumption means that a missing triple does not imply the fact is false. When augmenting LLMs with KGs, the model should be able to reason under uncertainty and not treat KG absence as negative evidence."
        id="warning-incompleteness"
      />

      <NoteBlock
        type="intuition"
        title="KGs vs. RAG with Text"
        content="KG-augmented generation and text-based RAG serve different needs. KGs excel at multi-hop factual reasoning (following chains of relations), provide precise, structured facts, and support explicit provenance. Text-based RAG provides richer context, handles nuance and qualification better, and requires no upfront structuring. The best systems often combine both: KG for factual grounding and text for context."
        id="note-vs-rag"
      />
    </div>
  )
}
