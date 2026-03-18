import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function TableSerialization() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Table Serialization: Markdown, CSV, and JSON Formats</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Before an LLM can reason about tabular data, the table must be converted into a linear
        text sequence. The choice of serialization format significantly affects how well the model
        understands structure, relationships between columns, and the data types within each cell.
      </p>

      <DefinitionBlock
        title="Table Serialization"
        definition="Table serialization is the process of converting a two-dimensional table with $m$ rows and $n$ columns into a linear string representation that preserves structural information. The serialized form $S(T)$ must encode both schema (column names, types) and cell values."
        notation="T = table, m = rows, n = columns, S(T) = serialized string"
        id="def-serialization"
      />

      <h2 className="text-2xl font-semibold">Markdown Format</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Markdown tables use pipes and dashes to create a human-readable grid layout. This format
        preserves column alignment visually and is common in LLM prompts because models have seen
        extensive markdown during pretraining.
      </p>

      <PythonCode
        title="markdown_serialization.py"
        code={`import pandas as pd

df = pd.DataFrame({
    "Product": ["Widget A", "Widget B", "Gadget C"],
    "Price": [29.99, 49.99, 19.99],
    "Stock": [150, 80, 320]
})

# Markdown serialization
def to_markdown(df):
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\\n".join([header, sep] + rows)

print(to_markdown(df))
# | Product  | Price | Stock |
# | ---      | ---   | ---   |
# | Widget A | 29.99 | 150   |
# | Widget B | 49.99 | 80    |
# | Gadget C | 19.99 | 320   |

# Pandas built-in (equivalent)
print(df.to_markdown(index=False))`}
        id="code-markdown"
      />

      <h2 className="text-2xl font-semibold">CSV Format</h2>
      <p className="text-gray-700 dark:text-gray-300">
        CSV (Comma-Separated Values) is the most compact serialization. It uses fewer tokens than
        markdown but loses visual alignment cues. Each row maps directly to one line of text.
      </p>

      <PythonCode
        title="csv_serialization.py"
        code={`import pandas as pd
import io

df = pd.DataFrame({
    "City": ["Tokyo", "London", "New York"],
    "Population_M": [13.96, 8.98, 8.34],
    "Country": ["Japan", "UK", "USA"]
})

# CSV serialization
csv_str = df.to_csv(index=False)
print(csv_str)
# City,Population_M,Country
# Tokyo,13.96,Japan
# London,8.98,UK
# New York,8.34,USA

# JSON Lines - one JSON object per row
jsonl_str = df.to_json(orient="records", lines=True)
print(jsonl_str)

# Compare token counts (approximate by character length)
md_str = df.to_markdown(index=False)
print(f"Markdown: {len(md_str)} chars")
print(f"CSV:      {len(csv_str)} chars")
print(f"JSON-L:   {len(jsonl_str)} chars")`}
        id="code-csv"
      />

      <ExampleBlock
        title="JSON Serialization Variants"
        problem="Serialize a 2-row table as JSON using records, columns, and split orientations."
        steps={[
          { formula: '\\text{records: } [\\{\\text{"a": 1, "b": 2}\\}, \\{\\text{"a": 3, "b": 4}\\}]', explanation: 'Each row becomes an independent JSON object. Best for row-level reasoning.' },
          { formula: '\\text{columns: } \\{\\text{"a": [1, 3], "b": [2, 4]}\\}', explanation: 'Grouped by column. Best for column-level aggregation questions.' },
          { formula: '\\text{split: } \\{\\text{columns: ["a","b"], data: [[1,2],[3,4]]}\\}', explanation: 'Separates schema from data. Most compact for large tables.' },
        ]}
        id="example-json-variants"
      />

      <NoteBlock
        type="tip"
        title="Choosing the Right Format"
        content="Empirical studies show that markdown tables work best for small tables (under 20 rows) where visual alignment helps. CSV is more token-efficient for larger tables. JSON records format is preferred when the model needs to reason about individual rows or when column names contain special characters."
        id="note-choosing-format"
      />

      <WarningBlock
        title="Token Budget and Table Size"
        content="A table with 100 rows and 10 columns can easily consume 3,000-5,000 tokens in markdown format. Always estimate token usage before serializing large tables. Consider truncation strategies: sample representative rows, include only relevant columns, or summarize numeric columns with statistics."
        id="warning-token-budget"
      />

      <NoteBlock
        type="historical"
        title="Evolution of Table Serialization"
        content="Early table-QA systems like TAPAS operated directly on structured table inputs. The shift to serialization-based approaches began with GPT-3 (2020), where researchers discovered that simply pasting tables as text into prompts yielded surprisingly good results. Sui et al. (2023) systematically benchmarked serialization formats, finding that the optimal choice depends on both table size and task type."
        id="note-history"
      />
    </div>
  )
}
