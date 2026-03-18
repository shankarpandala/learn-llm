import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function DependencyParsing() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Dependency Parsing</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Dependency parsing reveals the grammatical structure of a sentence by establishing
        directed relationships between words. Each word depends on exactly one other word (its head),
        forming a tree structure that captures who does what to whom.
      </p>

      <DefinitionBlock
        title="Dependency Tree"
        definition="A dependency tree is a directed tree where each node is a word in the sentence, and each edge represents a grammatical relation (dependency) from a head word to a dependent word. The root of the tree is typically the main verb."
        notation="A dependency relation is written as $\text{head} \xrightarrow{\text{rel}} \text{dependent}$. For example, $\text{chased} \xrightarrow{\text{nsubj}} \text{cat}$ means 'cat' is the nominal subject of 'chased'."
        id="def-dep-tree"
      />

      <h2 className="text-2xl font-semibold">Universal Dependencies</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The Universal Dependencies (UD) framework defines a cross-linguistically consistent set
        of dependency relations. Key relations include:
      </p>
      <ul className="ml-6 list-disc space-y-1 text-gray-700 dark:text-gray-300">
        <li><strong>nsubj</strong> - nominal subject</li>
        <li><strong>obj</strong> - direct object</li>
        <li><strong>amod</strong> - adjectival modifier</li>
        <li><strong>det</strong> - determiner</li>
        <li><strong>advmod</strong> - adverbial modifier</li>
        <li><strong>prep / obl</strong> - prepositional/oblique modifier</li>
      </ul>

      <ExampleBlock
        title="Dependency Parse Example"
        problem="Parse the dependencies in: 'The large cat quickly chased the small mouse'"
        steps={[
          { formula: 'chased is the ROOT', explanation: 'The main verb is the root of the tree.' },
          { formula: 'chased -nsubj-> cat', explanation: '"cat" is the nominal subject of "chased".' },
          { formula: 'cat -det-> The', explanation: '"The" is the determiner of "cat".' },
          { formula: 'cat -amod-> large', explanation: '"large" is an adjectival modifier of "cat".' },
          { formula: 'chased -obj-> mouse', explanation: '"mouse" is the direct object of "chased".' },
          { formula: 'chased -advmod-> quickly', explanation: '"quickly" is an adverbial modifier of "chased".' },
        ]}
        id="example-dep-parse"
      />

      <PythonCode
        title="dependency_parsing_spacy.py"
        code={`import spacy

nlp = spacy.load("en_core_web_sm")

sentence = "The quick brown fox jumps over the lazy dog"
doc = nlp(sentence)

# Display dependency tree
print(f"{'Token':<10} {'Dep':<10} {'Head':<10} {'Children'}")
print("-" * 55)
for token in doc:
    children = [child.text for child in token.children]
    print(f"{token.text:<10} {token.dep_:<10} {token.head.text:<10} {children}")

# Find the root and traverse the tree
root = [token for token in doc if token.dep_ == "ROOT"][0]
print(f"\\nRoot: '{root.text}' ({root.pos_})")

# Extract subject-verb-object triples
def extract_svo(doc):
    """Extract subject-verb-object triples from a parsed sentence."""
    triples = []
    for token in doc:
        if token.dep_ == "ROOT":
            verb = token
            subj = None
            obj = None
            for child in verb.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj = child
                elif child.dep_ in ("dobj", "obj"):
                    obj = child
            if subj and obj:
                triples.append((subj.text, verb.text, obj.text))
    return triples

sentences = [
    "The cat chased the mouse",
    "Scientists discovered a new species",
    "The student wrote an excellent paper",
]

for sent in sentences:
    doc = nlp(sent)
    triples = extract_svo(doc)
    print(f"'{sent}' -> SVO: {triples}")`}
        id="code-dep-parsing"
      />

      <h2 className="text-2xl font-semibold">Parsing Algorithms</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Two main families of algorithms exist for dependency parsing:
      </p>
      <ul className="ml-6 list-disc space-y-2 text-gray-700 dark:text-gray-300">
        <li>
          <strong>Transition-based parsing</strong> uses a stack and buffer with shift/reduce
          actions. It runs in <InlineMath math="O(n)" /> time but makes greedy local decisions.
          The arc-standard and arc-eager systems are popular variants.
        </li>
        <li>
          <strong>Graph-based parsing</strong> scores all possible edges and finds the
          maximum spanning tree. Eisner's algorithm runs in <InlineMath math="O(n^3)" /> for
          projective trees. This approach considers the global structure but is slower.
        </li>
      </ul>

      <NoteBlock
        type="intuition"
        title="Why Dependencies Matter for LLMs"
        content="While LLMs do not explicitly build dependency trees, they implicitly learn syntactic structure through attention patterns. Research has shown that specific attention heads in BERT and GPT models correspond to dependency relations, suggesting that understanding syntax is a natural byproduct of language modeling."
        id="note-dep-llm"
      />

      <WarningBlock
        title="Non-Projective Dependencies"
        content="In some languages (Czech, Dutch, German), dependency arcs can cross each other (non-projective trees). Standard shift-reduce parsers cannot produce non-projective trees. Special algorithms like the Chu-Liu-Edmonds algorithm or swap-based transitions are needed."
        id="warning-nonprojective"
      />

      <NoteBlock
        type="historical"
        title="Parsing History"
        content="Dependency grammar dates to Lucien Tesniere (1959). Computational dependency parsing took off with Nivre's arc-eager parser (2003) and McDonald's MST parser (2005). Chen and Manning's neural dependency parser (2014) showed that neural networks could replace hand-crafted features, paving the way for modern parsers."
        id="note-parsing-history"
      />
    </div>
  )
}
