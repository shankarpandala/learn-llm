import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Cleaning() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Cleaning and Normalization</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Real-world text is messy. Web scrapes contain HTML tags, social media has irregular
        spelling, and documents mix encodings. Text cleaning and normalization transform raw
        text into a consistent format before tokenization or model training. The quality of
        your data pipeline directly determines the quality of your model.
      </p>

      <DefinitionBlock
        title="Text Normalization"
        definition="Text normalization is the process of transforming text into a canonical (standard) form. This includes case folding, Unicode normalization, whitespace standardization, and removal of irrelevant content. The goal is to reduce surface variation while preserving semantic content."
        id="def-normalization"
      />

      <h2 className="text-2xl font-semibold">Common Cleaning Steps</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The specific cleaning steps depend on your data source and task, but common operations include:
      </p>

      <PythonCode
        title="text_cleaning.py"
        code={`import re
import unicodedata
import html

def clean_text(text):
    """Comprehensive text cleaning pipeline."""
    # 1. Decode HTML entities
    text = html.unescape(text)  # &amp; -> &, &lt; -> <

    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # 3. Unicode normalization (NFKC: compatibility decomposition + canonical composition)
    text = unicodedata.normalize('NFKC', text)

    # 4. Replace common Unicode variants
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # Smart quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2014', '--')  # Em dash
    text = text.replace('\u00a0', ' ')   # Non-breaking space

    # 5. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Remove control characters (keep newlines optionally)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cc' or c in '\n\t')

    return text

# Example: messy web-scraped text
raw = """<p>This is &ldquo;great&rdquo; content!&nbsp;&nbsp;
   <b>Bold</b> claims   with\u00a0weird\u2014spacing.</p>
   Visit us at <a href="http://example.com">our site</a>."""

cleaned = clean_text(raw)
print(f"Raw:     {repr(raw[:80])}")
print(f"Cleaned: {repr(cleaned)}")

# Case normalization (context-dependent!)
text = "Apple released the iPhone in San Francisco"
print(f"\\nLowered: {text.lower()}")
# Warning: lowercasing loses entity info ("Apple" company vs "apple" fruit)`}
        id="code-cleaning"
      />

      <h2 className="text-2xl font-semibold">Unicode Normalization Forms</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Unicode provides four normalization forms. The choice matters because visually identical
        strings may have different byte representations:
      </p>

      <PythonCode
        title="unicode_normalization.py"
        code={`import unicodedata

# The accented 'e' has two representations
# Composed: single code point U+00E9
e_composed = '\u00e9'  # e with acute accent
# Decomposed: base 'e' + combining acute accent
e_decomposed = 'e\u0301'

print(f"Composed:   '{e_composed}' (len={len(e_composed)}, bytes={e_composed.encode('utf-8').hex()})")
print(f"Decomposed: '{e_decomposed}' (len={len(e_decomposed)}, bytes={e_decomposed.encode('utf-8').hex()})")
print(f"Look same?  {e_composed} == {e_decomposed}? {e_composed == e_decomposed}")

# NFC (Canonical Decomposition + Canonical Composition) - recommended default
nfc = unicodedata.normalize('NFC', e_decomposed)
print(f"\\nAfter NFC:  '{nfc}' == composed? {nfc == e_composed}")

# NFKC also converts compatibility characters
text = '\ufb01'  # fi ligature
print(f"\\nLigature: '{text}' -> NFKC: '{unicodedata.normalize('NFKC', text)}'")

# Practical example: searching for "cafe" should match "cafe\u0301"
def normalize_for_search(text):
    return unicodedata.normalize('NFKC', text).lower()

query = "cafe"
documents = ["caf\u00e9 latte", "cafe\u0301 mocha", "cafe americano"]
for doc in documents:
    match = normalize_for_search(query) in normalize_for_search(doc)
    print(f"  '{query}' in '{doc}': {match}")`}
        id="code-unicode-norm"
      />

      <ExampleBlock
        title="Cleaning Pipeline Order Matters"
        problem="Given HTML text: '&lt;p&gt;The caf&amp;eacute; is GREAT!!!&lt;/p&gt;', apply cleaning steps in the correct order."
        steps={[
          { formula: "Step 1: HTML unescape -> '<p>The caf\\u00e9 is GREAT!!!</p>'", explanation: 'Decode HTML entities first, before removing tags.' },
          { formula: "Step 2: Strip HTML tags -> 'The caf\\u00e9 is GREAT!!!'", explanation: 'Remove markup after unescaping.' },
          { formula: "Step 3: Unicode normalize (NFC) -> 'The caf\\u00e9 is GREAT!!!'", explanation: 'Normalize Unicode representations.' },
          { formula: "Step 4: Lowercase (if needed) -> 'the caf\\u00e9 is great!!!'", explanation: 'Case folding depends on your task.' },
        ]}
        id="example-cleaning-order"
      />

      <WarningBlock
        title="Do Not Over-Clean"
        content="Aggressive cleaning can destroy useful signal. For LLM training, preserving case, punctuation, and formatting is often important because the model should learn to handle real text. Modern LLMs trained on lightly-cleaned data outperform those trained on heavily normalized data. Clean just enough to remove true noise (broken HTML, encoding errors) without removing linguistic variation."
        id="warning-over-clean"
      />

      <NoteBlock
        type="tip"
        title="Language-Specific Cleaning"
        content="Different languages need different normalization. Chinese and Japanese text has no spaces between words and needs segmentation. Arabic requires handling diacritics. German has compound words. Hindi uses Devanagari conjuncts. Always consider the target language when designing your cleaning pipeline."
        id="note-language-specific"
      />
    </div>
  )
}
