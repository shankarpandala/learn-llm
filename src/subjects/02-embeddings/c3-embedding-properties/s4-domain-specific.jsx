import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function DomainSpecific() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Domain-Specific Embeddings</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        General-purpose embeddings trained on Wikipedia or news corpora often perform poorly on
        specialized domains like biomedicine, law, or finance. Domain-specific vocabulary, jargon,
        and word sense distributions differ significantly from general text. Training or adapting
        embeddings on domain corpora can yield substantial improvements for specialized NLP tasks.
      </p>

      <DefinitionBlock
        title="Domain Adaptation of Embeddings"
        definition="Domain adaptation for embeddings involves either (1) training from scratch on a domain-specific corpus, (2) fine-tuning pre-trained general embeddings on domain text, or (3) combining general and domain embeddings via concatenation or retrofitting. The goal is to capture domain-specific semantics: in medicine, 'discharge' means leaving a hospital, not electrical discharge."
        id="def-domain-adapt"
      />

      <ExampleBlock
        title="Domain-Specific Nearest Neighbors"
        problem="The word 'cell' has very different neighbors in general vs. biomedical embeddings."
        steps={[
          { formula: 'General embeddings: cell -> phone, battery, jail, prison', explanation: 'General text is dominated by "cell phone" and "prison cell" contexts.' },
          { formula: 'BioWordVec: cell -> cells, cellular, tissue, lymphocyte', explanation: 'Biomedical text is dominated by biological cell contexts.' },
          { formula: 'Financial embeddings: cell -> spreadsheet, table, column, row', explanation: 'In financial documents, "cell" refers to spreadsheet cells.' },
        ]}
        id="ex-neighbors"
      />

      <NoteBlock
        type="note"
        title="Notable Domain-Specific Embeddings"
        content="BioWordVec/BioSentVec: trained on PubMed + MIMIC-III clinical notes for biomedical NLP. SciBERT: BERT pre-trained on Semantic Scholar papers for scientific text. LegalBERT: trained on legal corpora (court opinions, legislation). FinBERT: trained on financial news and SEC filings. ClinicalBERT: trained on clinical notes for healthcare applications."
        id="note-examples"
      />

      <PythonCode
        title="domain_embedding_training.py"
        id="code-domain"
        code={`import numpy as np
from gensim.models import Word2Vec
from collections import Counter

# Simulate a domain-specific corpus (biomedical)
bio_corpus = [
    "the patient presented with acute myocardial infarction".split(),
    "echocardiography revealed left ventricular dysfunction".split(),
    "treatment included aspirin and beta blockers".split(),
    "the cell culture showed abnormal proliferation".split(),
    "biopsy confirmed malignant cell growth in tissue".split(),
    "blood pressure was elevated at admission".split(),
    "the patient was discharged after seven days".split(),
    "lab results showed elevated white blood cell count".split(),
    "ct scan revealed no metastatic lesions".split(),
    "follow up mri confirmed treatment response".split(),
]

# Train domain-specific Word2Vec
bio_model = Word2Vec(
    sentences=bio_corpus,
    vector_size=50,
    window=5,
    min_count=1,
    sg=1,
    epochs=200,
    seed=42,
)

# General corpus for comparison
general_corpus = [
    "the cat sat on the mat in the room".split(),
    "she went to the bank to deposit money".split(),
    "the cell phone battery was dead".split(),
    "he was discharged from the army last year".split(),
    "blood is thicker than water they say".split(),
    "the patient customer waited in line".split(),
    "treatment of guests was excellent at the hotel".split(),
    "the growth of the company exceeded expectations".split(),
]

general_model = Word2Vec(
    sentences=general_corpus,
    vector_size=50,
    window=5,
    min_count=1,
    sg=1,
    epochs=200,
    seed=42,
)

# Compare nearest neighbors
def show_neighbors(model, word, topn=3, label=""):
    if word in model.wv:
        neighbors = model.wv.most_similar(word, topn=topn)
        nlist = ", ".join(f"{w}({s:.2f})" for w, s in neighbors)
        print(f"  [{label}] {word} -> {nlist}")
    else:
        print(f"  [{label}] {word} -> not in vocabulary")

print("Nearest neighbors comparison:")
for word in ["cell", "patient", "treatment", "blood"]:
    show_neighbors(bio_model, word, label="Bio")
    show_neighbors(general_model, word, label="Gen")
    print()`}
      />

      <PythonCode
        title="embedding_retrofit.py"
        id="code-retrofit"
        code={`import numpy as np

# Retrofitting: adjust pre-trained embeddings using a domain lexicon
# (Faruqui et al., 2015)

np.random.seed(42)
d = 50

# Simulated pre-trained embeddings
vocab = ["aspirin", "ibuprofen", "tylenol", "headache", "fever",
         "inflammation", "pain", "tablet", "dose", "prescription"]
original = {w: np.random.randn(d) * 0.5 for w in vocab}

# Domain knowledge: synonym pairs from medical ontology (e.g., UMLS)
synonyms = [
    ("aspirin", "ibuprofen"),     # both NSAIDs
    ("headache", "pain"),          # related symptoms
    ("fever", "inflammation"),     # related conditions
    ("tablet", "dose"),            # drug administration
    ("tylenol", "aspirin"),        # both analgesics
]

def retrofit(embeddings, lexicon, n_iters=10, alpha=1.0):
    """Retrofit embeddings to bring synonyms closer together."""
    new_vecs = {w: v.copy() for w, v in embeddings.items()}

    # Build adjacency list
    neighbors = {w: [] for w in embeddings}
    for w1, w2 in lexicon:
        if w1 in neighbors and w2 in neighbors:
            neighbors[w1].append(w2)
            neighbors[w2].append(w1)

    for iteration in range(n_iters):
        for word in embeddings:
            if not neighbors[word]:
                continue
            # Average of original vector and synonym vectors
            neighbor_sum = sum(new_vecs[n] for n in neighbors[word])
            num_neighbors = len(neighbors[word])
            new_vecs[word] = (alpha * embeddings[word] + neighbor_sum) / (alpha + num_neighbors)

    return new_vecs

cos = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

retrofitted = retrofit(original, synonyms)

print("Similarity changes after retrofitting:")
print(f"{'Pair':30s} {'Before':>8s} {'After':>8s} {'Change':>8s}")
print("-" * 58)
for w1, w2 in synonyms:
    before = cos(original[w1], original[w2])
    after = cos(retrofitted[w1], retrofitted[w2])
    print(f"{w1 + ' / ' + w2:30s} {before:8.3f} {after:8.3f} {after-before:+8.3f}")`}
      />

      <WarningBlock
        title="Domain Data Scarcity"
        content="Domain-specific corpora are often much smaller than general corpora (millions vs billions of tokens). Training embeddings from scratch on small corpora produces noisy vectors. Prefer fine-tuning pre-trained embeddings or using retrofitting when domain data is limited. For very specialized domains, consider using pre-trained contextual models (e.g., SciBERT, BioBERT) instead of static embeddings."
        id="warn-scarcity"
      />

      <NoteBlock
        type="tip"
        title="Practical Recommendations"
        content="Start with a pre-trained model closest to your domain (e.g., PubMedBERT for biomedical). If no domain model exists, fine-tune a general model on your corpus with a low learning rate. Use domain ontologies (UMLS for medicine, FIBO for finance) for retrofitting. Always evaluate on domain-specific benchmarks, not general ones like SimLex-999."
        id="note-practical"
      />
    </div>
  )
}
