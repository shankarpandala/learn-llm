import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function TAPAS() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">TAPAS: Weakly Supervised Table Parsing</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        TAPAS (Table Parser) is a BERT-based model specifically designed for table understanding.
        Unlike serialization-based approaches, TAPAS encodes tables using special positional
        embeddings that preserve row and column structure, enabling it to reason over tables
        without converting them to flat text.
      </p>

      <DefinitionBlock
        title="TAPAS Architecture"
        definition="TAPAS extends BERT with additional embedding layers for tabular structure. Each token receives embeddings for: token position, segment (question vs. table), column index $c_i \in \{0, \ldots, C\}$, row index $r_j \in \{0, \ldots, R\}$, and numeric rank within the column. The final embedding is $e = e_{token} + e_{position} + e_{segment} + e_{column} + e_{row} + e_{rank}$."
        notation="e = embedding, C = max columns, R = max rows"
        id="def-tapas"
      />

      <h2 className="text-2xl font-semibold">Structural Embeddings</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The key innovation of TAPAS is its structural embeddings that encode the 2D position
        of each token within the table. This allows the transformer to distinguish between
        tokens in different cells without relying on delimiters or formatting.
      </p>

      <ExampleBlock
        title="TAPAS Input Encoding"
        problem="Show how a question + 2x2 table is encoded with TAPAS positional embeddings."
        steps={[
          { formula: '\\text{Input: [CLS] question tokens [SEP] header1 header2 cell11 cell12 cell21 cell22}', explanation: 'Flatten the table row-by-row after the question, separated by [SEP].' },
          { formula: '\\text{Column IDs: } [0, 0, \\ldots, 0, 1, 2, 1, 2, 1, 2]', explanation: 'Question tokens get column=0, then each cell gets its column index.' },
          { formula: '\\text{Row IDs: } [0, 0, \\ldots, 0, 0, 0, 1, 1, 2, 2]', explanation: 'Question tokens and headers get row=0, data rows are numbered 1, 2, ...' },
          { formula: 'e_i = e_{token_i} + e_{pos_i} + e_{seg_i} + e_{col_i} + e_{row_i}', explanation: 'All embeddings are summed to produce the final input representation.' },
        ]}
        id="example-encoding"
      />

      <PythonCode
        title="tapas_inference.py"
        code={`from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd

# Load pretrained TAPAS model
model_name = "google/tapas-base-finetuned-wtq"
tokenizer = TapasTokenizer.from_pretrained(model_name)
model = TapasForQuestionAnswering.from_pretrained(model_name)

# Create a table as a DataFrame
table = pd.DataFrame({
    "City": ["Paris", "London", "Berlin", "Madrid"],
    "Country": ["France", "UK", "Germany", "Spain"],
    "Population": ["2148000", "8982000", "3645000", "3224000"],
    "Area_km2": ["105", "1572", "892", "604"]
})

# Ask questions about the table
queries = [
    "What is the population of Berlin?",
    "Which city has the largest area?",
    "How many cities have population over 3 million?",
]

for query in queries:
    inputs = tokenizer(
        table=table, queries=query,
        padding="max_length", return_tensors="pt"
    )
    outputs = model(**inputs)

    # Get predicted answer coordinates and aggregation
    predicted_answer = tokenizer.convert_logits_to_predictions(
        inputs, outputs.logits.detach(),
        outputs.logits_aggregation.detach()
    )
    coords, agg_indices = predicted_answer
    print(f"Q: {query}")
    print(f"  Selected cells: {coords[0]}")
    print(f"  Aggregation: {['NONE','SUM','AVG','COUNT'][agg_indices[0]]}\\n")`}
        id="code-tapas"
      />

      <NoteBlock
        type="historical"
        title="TAPAS Development Timeline"
        content="TAPAS was introduced by Herzig et al. (2020) at Google Research. It was pretrained on millions of Wikipedia tables using a masked language modeling objective adapted for tables. TAPAS achieved state-of-the-art on WTQ (WikiTableQuestions), SQA, and TabFact. Later versions (TaPEx, 2022) showed that pretraining on synthetic SQL execution traces could further improve table reasoning."
        id="note-history"
      />

      <h2 className="text-2xl font-semibold">Aggregation Operations</h2>
      <p className="text-gray-700 dark:text-gray-300">
        TAPAS jointly predicts which cells are relevant and what aggregation operation to apply.
        This allows it to handle questions that require counting, summing, or averaging without
        generating SQL.
      </p>

      <PythonCode
        title="tapas_cell_selection.py"
        code={`import torch
import torch.nn.functional as F

# Simulated TAPAS cell selection logic
# The model outputs logits for each cell and aggregation type
num_rows, num_cols = 4, 3
cell_logits = torch.randn(num_rows, num_cols)  # per-cell scores
agg_logits = torch.randn(4)  # NONE, SUM, AVERAGE, COUNT

# Cell selection probabilities
cell_probs = torch.sigmoid(cell_logits)
print("Cell selection probabilities:")
print(cell_probs.numpy().round(3))

# Aggregation prediction
agg_ops = ["NONE", "SUM", "AVERAGE", "COUNT"]
agg_pred = torch.argmax(agg_logits).item()
print(f"\\nPredicted aggregation: {agg_ops[agg_pred]}")

# For NONE: answer is the selected cell values directly
# For SUM/AVG: apply aggregation to numeric values in selected cells
selected = cell_probs > 0.5
print(f"Selected cell mask:\\n{selected.numpy()}")

# Weakly supervised training: only need (question, answer) pairs
# No need for cell-level annotations
# Loss = cell_selection_loss + aggregation_loss
# cell_selection_loss uses marginal likelihood over all
# cell combinations that produce the correct answer`}
        id="code-cell-selection"
      />

      <WarningBlock
        title="TAPAS Limitations"
        content="TAPAS is limited to tables that fit within BERT's 512-token context window, which typically means tables with fewer than 50-100 cells. It also requires all cell values to be strings (numbers must be stringified), and struggles with tables containing merged cells, multi-level headers, or hierarchical row indices."
        id="warning-limitations"
      />

      <NoteBlock
        type="tip"
        title="When to Use TAPAS vs. LLM-Based Approaches"
        content="Use TAPAS when you need fast, consistent inference on small-to-medium tables with well-defined schemas. Use LLM-based approaches when tables are large (with truncation strategies), questions require world knowledge beyond the table, or you need natural language explanations alongside answers."
        id="note-when-to-use"
      />
    </div>
  )
}
