import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function PosTagging() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Part-of-Speech Tagging</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Part-of-speech (POS) tagging assigns a grammatical category (noun, verb, adjective, etc.)
        to each word in a sentence. It is one of the fundamental tasks in NLP, serving as a
        building block for parsing, information extraction, and text understanding.
      </p>

      <DefinitionBlock
        title="Part-of-Speech Tag"
        definition="A POS tag is a label assigned to a word indicating its syntactic role. Common tagsets include the Penn Treebank tagset (45 tags) and the Universal Dependencies tagset (17 tags). For example, 'NN' = singular noun, 'VB' = base verb, 'JJ' = adjective."
        id="def-pos"
      />

      <ExampleBlock
        title="POS Tagging Example"
        problem="Tag each word in: 'The quick brown fox jumps over the lazy dog'"
        steps={[
          { formula: 'The -> DT (Determiner)', explanation: 'Articles like "the", "a", "an" are determiners.' },
          { formula: 'quick -> JJ (Adjective)', explanation: 'Describes a property of the noun.' },
          { formula: 'brown -> JJ (Adjective)', explanation: 'Another adjective modifying "fox".' },
          { formula: 'fox -> NN (Noun, singular)', explanation: 'The subject of the sentence.' },
          { formula: 'jumps -> VBZ (Verb, 3rd person singular present)', explanation: 'The main verb, conjugated for "fox".' },
          { formula: 'over -> IN (Preposition)', explanation: 'Introduces a prepositional phrase.' },
        ]}
        id="example-pos"
      />

      <h2 className="text-2xl font-semibold">Hidden Markov Models for POS Tagging</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Classical POS taggers use Hidden Markov Models (HMMs). The hidden states are the POS tags,
        and the observations are the words. The model estimates two probability distributions:
      </p>

      <div className="my-4 space-y-2">
        <BlockMath math="P(\text{tag}_i \mid \text{tag}_{i-1}) \quad \text{(transition probability)}" />
        <BlockMath math="P(\text{word}_i \mid \text{tag}_i) \quad \text{(emission probability)}" />
      </div>

      <p className="text-gray-700 dark:text-gray-300">
        The Viterbi algorithm finds the most likely tag sequence by dynamic programming:
      </p>

      <div className="my-4">
        <BlockMath math="\hat{t}_1^n = \arg\max_{t_1^n} \prod_{i=1}^{n} P(w_i \mid t_i) \cdot P(t_i \mid t_{i-1})" />
      </div>

      <PythonCode
        title="pos_tagging_spacy.py"
        code={`import spacy

# Load the English model (small)
nlp = spacy.load("en_core_web_sm")

text = "The quick brown fox jumps over the lazy dog"
doc = nlp(text)

# Display POS tags
print(f"{'Token':<12} {'POS':<8} {'Fine POS':<8} {'Description'}")
print("-" * 55)
for token in doc:
    print(f"{token.text:<12} {token.pos_:<8} {token.tag_:<8} {spacy.explain(token.tag_)}")

# POS tag distribution in a longer text
long_text = """
Natural language processing enables computers to understand human language.
Machine learning models learn patterns from large datasets of text.
Transformers revolutionized the field with attention mechanisms.
"""
doc2 = nlp(long_text)
from collections import Counter
pos_counts = Counter(token.pos_ for token in doc2 if not token.is_space)
print("\\nPOS distribution:")
for pos, count in pos_counts.most_common():
    print(f"  {pos:<8} {count}")`}
        id="code-pos-spacy"
      />

      <NoteBlock
        type="intuition"
        title="Why POS Tagging Is Hard"
        content="Many words are ambiguous: 'run' can be a verb ('I run daily') or a noun ('a morning run'). 'Back' can be a noun, verb, adjective, or adverb. Context is essential. The word 'flies' in 'time flies like an arrow' vs 'fruit flies like a banana' has completely different tags."
        id="note-ambiguity"
      />

      <PythonCode
        title="pos_ambiguity.py"
        code={`import spacy
nlp = spacy.load("en_core_web_sm")

# The same word gets different POS tags in different contexts
examples = [
    "I need to book a flight",        # book = VERB
    "I read an interesting book",      # book = NOUN
    "They will fly to Paris",          # fly = VERB
    "A fly landed on the table",       # fly = NOUN
    "The old man the boats",           # old = NOUN (garden path!)
]

for sent in examples:
    doc = nlp(sent)
    tags = [(t.text, t.pos_) for t in doc]
    print(f"{sent}")
    print(f"  Tags: {tags}\\n")`}
        id="code-pos-ambiguity"
      />

      <WarningBlock
        title="Tagset Differences"
        content="Different POS tagsets exist: Penn Treebank (PTB) uses fine-grained tags like VBZ, VBD, VBG for verb forms, while Universal Dependencies (UD) uses coarser tags like VERB. Always check which tagset your tools use, as downstream tasks may expect a specific one."
        id="warning-tagsets"
      />

      <NoteBlock
        type="historical"
        title="From Rules to Neural Models"
        content="Early POS taggers (1960s) used hand-written rules. HMM taggers (1980s-90s) achieved ~96% accuracy. The Brill tagger (1992) used transformation-based learning. Modern neural taggers using BiLSTMs and Transformers achieve ~97.5% accuracy, approaching the human inter-annotator agreement ceiling of ~97%."
        id="note-pos-history"
      />
    </div>
  )
}
