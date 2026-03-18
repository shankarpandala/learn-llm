import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function UnicodeEncoding() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Unicode and Text Encoding</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Before text reaches a tokenizer, it exists as a sequence of bytes governed by encoding
        standards. Understanding Unicode and UTF-8 is essential for building robust NLP systems
        that handle multilingual text, emojis, and special characters.
      </p>

      <DefinitionBlock
        title="Unicode Code Point"
        definition="A Unicode code point is a unique integer assigned to every character in the Unicode standard. Code points are written as U+XXXX (hexadecimal). Unicode 15.0 defines over 149,000 characters across 161 scripts."
        notation="A character like 'A' has code point $U{+}0041$, stored as integer $65_{10}$."
        id="def-unicode"
      />

      <DefinitionBlock
        title="UTF-8 Encoding"
        definition="UTF-8 is a variable-length encoding that represents each Unicode code point as 1 to 4 bytes. ASCII characters (U+0000 to U+007F) use 1 byte, making UTF-8 backward-compatible with ASCII."
        id="def-utf8"
      />

      <ExampleBlock
        title="UTF-8 Byte Sequences"
        problem="How is the emoji character (U+1F600) encoded in UTF-8?"
        steps={[
          { formula: 'Code point: U+1F600 = 128512 in decimal', explanation: 'The grinning face emoji has a high code point requiring 4 bytes.' },
          { formula: 'Binary: 0001 1111 0110 0000 0000', explanation: 'Convert the code point to binary (21 bits needed).' },
          { formula: 'UTF-8 pattern: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx', explanation: '4-byte UTF-8 uses this bit pattern (leading byte starts with 11110).' },
          { formula: 'Result: F0 9F 98 80 (4 bytes)', explanation: 'Fill the x positions with the binary bits of the code point.' },
        ]}
        id="example-utf8"
      />

      <PythonCode
        title="unicode_exploration.py"
        code={`# Exploring Unicode and encodings in Python
text = "Hello, \u4e16\u754c! \ud83d\ude00"

# Code points
for char in text:
    print(f"'{char}' -> U+{ord(char):04X} (decimal {ord(char)})")

# UTF-8 encoding
utf8_bytes = text.encode('utf-8')
print(f"\\nUTF-8 bytes: {utf8_bytes}")
print(f"UTF-8 length: {len(utf8_bytes)} bytes")
print(f"Character count: {len(text)} characters")

# Different encodings produce different byte lengths
encodings = ['utf-8', 'utf-16', 'utf-32', 'ascii']
for enc in encodings:
    try:
        encoded = text.encode(enc)
        print(f"{enc:8s}: {len(encoded)} bytes")
    except UnicodeEncodeError:
        print(f"{enc:8s}: Cannot encode (characters out of range)")

# Byte-level view of a CJK character
char = '\u4e16'  # Chinese character for "world"
print(f"\\n'{char}' in UTF-8: {char.encode('utf-8').hex(' ')}")
print(f"'{char}' in UTF-16: {char.encode('utf-16-be').hex(' ')}")`}
        id="code-unicode"
      />

      <h2 className="text-2xl font-semibold">Byte-Level Tokenization</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Modern LLMs like GPT-2 and beyond use byte-level BPE, which operates on raw UTF-8 bytes
        rather than Unicode characters. This guarantees that any input string can be tokenized
        without unknown tokens, since all 256 byte values are in the base vocabulary.
      </p>

      <PythonCode
        title="byte_level_tokenization.py"
        code={`# Byte-level BPE: the foundation of GPT tokenizers
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

# Multilingual text: same tokenizer handles everything
examples = {
    "English": "Machine learning is powerful.",
    "Chinese": "\u673a\u5668\u5b66\u4e60\u5f88\u5f3a\u5927\u3002",
    "Arabic":  "\u0627\u0644\u062a\u0639\u0644\u0645 \u0627\u0644\u0622\u0644\u064a",
    "Emoji":   "\ud83e\udd16\ud83d\udcac\ud83c\udf0d",
}

for lang, text in examples.items():
    tokens = enc.encode(text)
    utf8_len = len(text.encode('utf-8'))
    print(f"{lang:10s}: {len(tokens):3d} tokens, {utf8_len:3d} UTF-8 bytes")
    # Decode each token to see the subwords
    decoded = [enc.decode([t]) for t in tokens]
    print(f"{'':10s}  Tokens: {decoded}")`}
        id="code-byte-bpe"
      />

      <NoteBlock
        type="tip"
        title="Normalization Matters"
        content="Unicode has multiple ways to represent the same visual character. For example, the accented 'e' can be a single code point (U+00E9) or a base 'e' plus a combining accent (U+0065 + U+0301). Always normalize text (NFC or NFKC) before tokenization to avoid treating identical-looking text differently."
        id="note-normalization"
      />

      <WarningBlock
        title="Encoding Mismatches Cause Silent Bugs"
        content="If you read a file as latin-1 but it was written as UTF-8, multi-byte characters will be split incorrectly. Always be explicit about encodings when reading data: use open(file, encoding='utf-8') in Python."
        id="warning-encoding"
      />

      <NoteBlock
        type="historical"
        title="From ASCII to Unicode"
        content="ASCII (1963) defined 128 characters for English. Various incompatible extensions (Latin-1, Shift-JIS, GB2312) arose for other languages. Unicode (1991) unified all scripts into a single standard. UTF-8 (1993), designed by Ken Thompson and Rob Pike, became the dominant encoding on the web, covering over 98% of web pages today."
        id="note-ascii-history"
      />
    </div>
  )
}
