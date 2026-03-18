import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function GrammarConstrained() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Grammar-Constrained Generation</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Grammar-constrained generation restricts an LLM's output to conform to a formal grammar,
        such as a context-free grammar (CFG) or regular expression. This guarantees syntactic
        validity of the output, enabling reliable generation of code, structured data formats,
        and domain-specific languages.
      </p>

      <DefinitionBlock
        title="Grammar-Constrained Decoding"
        definition="Given a context-free grammar $G = (V, \Sigma, R, S)$ and a language model $P(x_t \mid x_{<t})$, grammar-constrained decoding produces tokens from the modified distribution $P'(x_t \mid x_{<t}) \propto P(x_t \mid x_{<t}) \cdot \mathbb{1}[x_t \text{ is valid given } G \text{ and } x_{<t}]$. At each step, only tokens that can lead to a complete valid string in $\mathcal{L}(G)$ are allowed."
        notation="G = grammar, V = variables, \Sigma = terminals, R = rules, S = start symbol"
        id="def-grammar"
      />

      <h2 className="text-2xl font-semibold">GBNF: Grammar BNF for llama.cpp</h2>
      <p className="text-gray-700 dark:text-gray-300">
        GBNF (Grammar BNF) is the grammar format used by llama.cpp for constrained generation.
        It extends BNF notation with character classes, repetition operators, and alternation
        to define the set of valid outputs.
      </p>

      <PythonCode
        title="gbnf_grammars.py"
        code={`# GBNF grammar definitions for constrained generation

# Grammar for valid JSON objects
json_grammar = r"""
root   ::= object
value  ::= object | array | string | number | "true" | "false" | "null"

object ::= "{" ws (pair ("," ws pair)*)? ws "}"
pair   ::= string ws ":" ws value
array  ::= "[" ws (value ("," ws value)*)? ws "]"

string ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""
number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws     ::= [ \\t\\n]*
"""

# Grammar for SQL SELECT statements
sql_select_grammar = r"""
root       ::= "SELECT " columns " FROM " table where? groupby? orderby? limit? ";"
columns    ::= column ("," ws column)*
column     ::= (aggregate "(" colname ")") | colname | "*"
aggregate  ::= "COUNT" | "SUM" | "AVG" | "MAX" | "MIN"
colname    ::= [a-zA-Z_] [a-zA-Z0-9_]*
table      ::= [a-zA-Z_] [a-zA-Z0-9_]* (ws join)*
join       ::= "JOIN " [a-zA-Z_]+ " ON " condition
where      ::= " WHERE " condition (" AND " condition)*
condition  ::= colname ws comparator ws value
comparator ::= "=" | "!=" | ">" | "<" | ">=" | "<="
value      ::= number | string
groupby    ::= " GROUP BY " colname ("," ws colname)*
orderby    ::= " ORDER BY " colname (" ASC" | " DESC")?
limit      ::= " LIMIT " [0-9]+
number     ::= [0-9]+ ("." [0-9]+)?
string     ::= "'" [^']* "'"
ws         ::= " "*
"""

# Grammar for structured entity extraction
entity_grammar = r"""
root    ::= "{" ws
            "\\"entities\\":" ws "[" (entity ("," entity)*)? "]" ws
            "}"
entity  ::= "{" ws
            "\\"name\\":" ws string "," ws
            "\\"type\\":" ws type "," ws
            "\\"confidence\\":" ws number ws
            "}"
type    ::= "\\"PERSON\\"" | "\\"ORG\\"" | "\\"LOCATION\\"" | "\\"DATE\\""
string  ::= "\\"" [^"\\\\]* "\\""
number  ::= "0." [0-9]+
ws      ::= [ \\t\\n]*
"""

print("JSON Grammar:")
print(json_grammar)
print("\\nSQL Grammar (excerpt):")
print(sql_select_grammar[:200] + "...")`}
        id="code-gbnf"
      />

      <ExampleBlock
        title="Grammar-Constrained Token Selection"
        problem="Show how grammar constraints filter tokens during generation of a JSON boolean field."
        steps={[
          { formula: '\\text{Generated so far: } \\{\\text{"active": }', explanation: 'The model has produced the key and is about to generate the value.' },
          { formula: '\\text{Grammar state: expecting } value \\to \\text{"true"} \\mid \\text{"false"} \\mid \\text{number} \\mid \\ldots', explanation: 'The grammar parser tracks the current valid continuations.' },
          { formula: '\\text{Token mask: allow } [\\text{"true", "false", "null", "\\"", digits, "[", "\\{"}]', explanation: 'Only tokens that begin a valid JSON value are permitted.' },
          { formula: '\\text{If schema says boolean: mask reduces to } [\\text{"true", "false"}]', explanation: 'With additional schema constraints, the allowed set narrows further.' },
        ]}
        id="example-filtering"
      />

      <PythonCode
        title="llama_cpp_grammar.py"
        code={`# Using grammar-constrained generation with llama-cpp-python
from llama_cpp import Llama, LlamaGrammar

# Load model
llm = Llama(model_path="./models/llama-3-8b.gguf", n_ctx=2048)

# Define a grammar for structured output
grammar_text = r"""
root   ::= "{" ws
           "\"sentiment\":" ws sentiment "," ws
           "\"score\":" ws score "," ws
           "\"keywords\":" ws "[" ws (string ("," ws string)*)? ws "]" ws
           "}"
sentiment ::= "\"positive\"" | "\"negative\"" | "\"neutral\""
score     ::= "0." [0-9] [0-9]
string    ::= "\"" [a-zA-Z ]+ "\""
ws        ::= [ \t\n]*
"""

grammar = LlamaGrammar.from_string(grammar_text)

# Generate with grammar constraint
output = llm(
    "Analyze the sentiment of this review: 'The product works great "
    "but shipping was slow.' Respond as JSON:\n",
    grammar=grammar,
    max_tokens=200,
    temperature=0.7,
)

print(output["choices"][0]["text"])
# {"sentiment": "positive", "score": 0.72, "keywords": ["great", "slow shipping"]}

# The output is GUARANTEED to match the grammar
# No parsing errors possible`}
        id="code-llama-grammar"
      />

      <TheoremBlock
        title="Completeness of Grammar-Constrained Decoding"
        statement="For any context-free grammar G and any string s in the language of G, grammar-constrained decoding with a sufficiently expressive language model can generate s, provided that for each valid token at each position, the model assigns non-zero probability."
        proof="At each step, the constraint only masks tokens that would lead to strings outside the language. Since s is in the language, each token of s is always in the allowed set. With non-zero probability on all valid tokens, greedy or sampling-based decoding can produce s."
        id="theorem-completeness"
      />

      <WarningBlock
        title="Grammar Complexity and Performance"
        content="Complex grammars with many production rules can slow down generation because the parser must check validity at each token. Ambiguous grammars (where the same string can be derived multiple ways) can cause the parser to track multiple states simultaneously. Keep grammars as simple as possible and prefer deterministic rules."
        id="warning-complexity"
      />

      <NoteBlock
        type="note"
        title="Grammar Libraries"
        content="llama.cpp pioneered GBNF grammars for local models. The Outlines library provides Python-native grammar support for Hugging Face models. Guidance from Microsoft uses a hybrid template/grammar approach. vLLM integrates with Outlines for production serving. All achieve the same goal: guaranteed structural validity of LLM outputs."
        id="note-libraries"
      />
    </div>
  )
}
