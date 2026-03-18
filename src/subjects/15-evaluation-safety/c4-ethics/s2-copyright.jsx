import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Copyright() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Copyright, Training Data, and Fair Use</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The relationship between copyright law and LLM training data is one of the most
        actively litigated areas in AI. Questions about what data can be used for training,
        whether model outputs infringe copyright, and how fair use applies remain largely
        unresolved.
      </p>

      <DefinitionBlock
        title="Training Data Copyright"
        definition="The legal question of whether using copyrighted works to train AI models constitutes copyright infringement or is protected by fair use (US) / text and data mining exceptions (EU). Key cases include NYT v. OpenAI, Authors Guild v. OpenAI, and Getty v. Stability AI."
        id="def-training-copyright"
      />

      <DefinitionBlock
        title="Fair Use (US)"
        definition="A legal doctrine that permits limited use of copyrighted material without permission for purposes such as commentary, education, or research. Courts evaluate four factors: (1) purpose and character of use, (2) nature of the copyrighted work, (3) amount used, and (4) effect on the market."
        id="def-fair-use"
      />

      <h2 className="text-2xl font-semibold">Key Legal Questions</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The debate centers on several interrelated questions: Is training a "transformative use"?
        Can models memorize and reproduce copyrighted content? Does generated content that
        resembles copyrighted works constitute infringement? The concept of memorization
        can be quantified:
      </p>
      <BlockMath math="\text{Memorization rate} = \frac{|\{x \in D_{\text{train}} : P(x|\text{prefix}) > \tau\}|}{|D_{\text{train}}|}" />
      <p className="text-gray-700 dark:text-gray-300">
        where <InlineMath math="\tau" /> is a threshold probability and{' '}
        <InlineMath math="D_{\text{train}}" /> is the training dataset.
      </p>

      <ExampleBlock
        title="Fair Use Analysis for LLM Training"
        problem="Analyze whether training an LLM on web-scraped books qualifies as fair use."
        steps={[
          { formula: '\\text{Factor 1: Transformative? Training \\neq reproduction}', explanation: 'The model learns statistical patterns, not storing/reproducing full works. This is argued as transformative.' },
          { formula: '\\text{Factor 2: Creative works (novels) get stronger protection}', explanation: 'Factual works receive less protection than creative/fictional works.' },
          { formula: '\\text{Factor 3: Models process entire works during training}', explanation: 'The full work is ingested, even if not fully memorized. This weighs against fair use.' },
          { formula: '\\text{Factor 4: Does the model substitute for the original?}', explanation: 'If users can extract book content from the model, it competes with the original market.' },
        ]}
        id="example-fair-use"
      />

      <PythonCode
        title="memorization_detection.py"
        code={`# Detecting memorization in language models
# This helps assess copyright risk in model outputs

import numpy as np
from openai import OpenAI

client = OpenAI()

def check_memorization(prefix: str, known_continuation: str,
                       model: str = "gpt-4o-mini", n_tokens: int = 100) -> dict:
    """Check if a model has memorized a specific text passage."""
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",  # Completion API for raw continuation
        prompt=prefix,
        max_tokens=n_tokens,
        temperature=0,
    )
    generated = response.choices[0].text

    # Calculate overlap metrics
    gen_words = generated.lower().split()
    ref_words = known_continuation.lower().split()

    # Longest common subsequence ratio
    def lcs_length(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    lcs = lcs_length(gen_words, ref_words)
    overlap = lcs / max(len(ref_words), 1)

    # Exact match ratio (n-gram overlap)
    def ngram_overlap(a, b, n=5):
        a_ngrams = set(tuple(a[i:i+n]) for i in range(len(a)-n+1))
        b_ngrams = set(tuple(b[i:i+n]) for i in range(len(b)-n+1))
        if not b_ngrams:
            return 0.0
        return len(a_ngrams & b_ngrams) / len(b_ngrams)

    return {
        "generated_preview": generated[:200],
        "lcs_overlap": overlap,
        "5gram_overlap": ngram_overlap(gen_words, ref_words),
        "likely_memorized": overlap > 0.5 or ngram_overlap(gen_words, ref_words) > 0.3,
    }

# Data provenance tracking
class DataProvenanceTracker:
    """Track data sources and licensing for training datasets."""

    LICENSE_TYPES = {
        "public_domain": {"commercial": True, "attribution": False},
        "cc_by": {"commercial": True, "attribution": True},
        "cc_by_sa": {"commercial": True, "attribution": True, "share_alike": True},
        "cc_by_nc": {"commercial": False, "attribution": True},
        "copyrighted": {"commercial": False, "attribution": True, "permission_needed": True},
    }

    def __init__(self):
        self.sources = []

    def add_source(self, name: str, license_type: str, size_gb: float, url: str = ""):
        self.sources.append({
            "name": name, "license": license_type,
            "size_gb": size_gb, "url": url,
            "permissions": self.LICENSE_TYPES.get(license_type, {}),
        })

    def commercial_safe(self) -> list:
        return [s for s in self.sources if s["permissions"].get("commercial", False)]

    def report(self):
        total = sum(s["size_gb"] for s in self.sources)
        safe = sum(s["size_gb"] for s in self.commercial_safe())
        print(f"Total data: {total:.1f} GB")
        print(f"Commercially safe: {safe:.1f} GB ({100*safe/total:.1f}%)")
        for s in self.sources:
            status = "OK" if s["permissions"].get("commercial") else "RESTRICTED"
            print(f"  [{status}] {s['name']}: {s['size_gb']:.1f} GB ({s['license']})")

tracker = DataProvenanceTracker()
tracker.add_source("Wikipedia", "cc_by_sa", 20.0)
tracker.add_source("ArXiv papers", "cc_by", 50.0)
tracker.add_source("Web scrape", "copyrighted", 500.0)
tracker.add_source("Project Gutenberg", "public_domain", 10.0)
tracker.report()`}
        id="code-memorization"
      />

      <WarningBlock
        title="Rapidly Evolving Legal Landscape"
        content="Copyright law as applied to AI training is evolving rapidly. Court decisions in the US (NYT v. OpenAI, Thomson Reuters v. ROSS), EU AI Act provisions, and new legislation may fundamentally change what is permissible. The information here reflects the state of debate, not settled law. Consult legal counsel for compliance."
        id="warning-evolving"
      />

      <NoteBlock
        type="note"
        title="Opt-Out and Consent Frameworks"
        content="Some approaches to the copyright question include: robots.txt and AI-specific opt-out mechanisms (like the proposed TDM Reservation Protocol), data licensing marketplaces, revenue-sharing models with content creators, and training only on permissively licensed or public domain data."
        id="note-opt-out"
      />

      <NoteBlock
        type="historical"
        title="Precedents and Milestones"
        content="Google Books (Authors Guild v. Google, 2015) established that scanning books for a search index is fair use. The US Copyright Office (2023) ruled that purely AI-generated content cannot be copyrighted. The EU AI Act (2024) requires transparency about training data. These precedents shape the evolving framework."
        id="note-precedents"
      />
    </div>
  )
}
