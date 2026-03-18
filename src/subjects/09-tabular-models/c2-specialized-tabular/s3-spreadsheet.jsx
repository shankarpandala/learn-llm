import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function SpreadsheetIntegration() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Spreadsheet Integration with LLMs</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Spreadsheets are the most ubiquitous form of structured data, with billions of Excel
        and Google Sheets files in active use. Integrating LLMs with spreadsheets enables
        natural language formula generation, data analysis, and automated transformations
        that make structured data manipulation accessible to non-technical users.
      </p>

      <DefinitionBlock
        title="Spreadsheet LLM Integration"
        definition="Spreadsheet-LLM integration maps a spreadsheet state $S = (C, F, V)$ -- where $C$ is the cell grid, $F$ is the formula set, and $V$ are computed values -- with a natural language instruction $I$ to produce an action $A$ that modifies the spreadsheet. Actions include formula insertion, data transformation, chart creation, and formatting."
        notation="S = spreadsheet state, C = cells, F = formulas, V = values, I = instruction, A = action"
        id="def-spreadsheet"
      />

      <h2 className="text-2xl font-semibold">Formula Generation</h2>
      <p className="text-gray-700 dark:text-gray-300">
        One of the most impactful applications is generating spreadsheet formulas from natural
        language descriptions. LLMs can produce Excel formulas, Google Sheets functions,
        and even complex array formulas from plain English instructions.
      </p>

      <PythonCode
        title="formula_generation.py"
        code={`# LLM-powered spreadsheet formula generation
import json

# Spreadsheet context representation
spreadsheet_context = {
    "sheet_name": "Sales Report",
    "headers": {
        "A": "Date", "B": "Product", "C": "Quantity",
        "D": "Unit Price", "E": "Total", "F": "Region"
    },
    "sample_rows": [
        {"A": "2024-01-15", "B": "Widget", "C": 10, "D": 29.99, "F": "East"},
        {"A": "2024-01-16", "B": "Gadget", "C": 5, "D": 49.99, "F": "West"},
    ],
    "num_rows": 500,
}

def build_formula_prompt(context, request):
    return f"""You are a spreadsheet formula expert.

Sheet: {context['sheet_name']}
Columns: {json.dumps(context['headers'])}
Data rows: 2 to {context['num_rows'] + 1}
Sample data: {json.dumps(context['sample_rows'][:2])}

User request: {request}

Return ONLY the Excel formula. Use absolute references where appropriate."""

# Example requests and expected formulas
requests = [
    ("Calculate the total for each row (quantity * unit price)",
     "=C2*D2"),
    ("Sum all quantities for the East region",
     '=SUMIFS(C2:C501,F2:F501,"East")'),
    ("Find the date with the highest total sales",
     "=INDEX(A2:A501,MATCH(MAX(E2:E501),E2:E501,0))"),
    ("Calculate a 7-day moving average of daily totals",
     "=AVERAGE(OFFSET(E2,ROW()-ROW($E$2)-6,0,7,1))"),
]

for request, expected in requests:
    prompt = build_formula_prompt(spreadsheet_context, request)
    print(f"Request: {request}")
    print(f"Expected: {expected}\\n")`}
        id="code-formula"
      />

      <ExampleBlock
        title="Spreadsheet Context Serialization"
        problem="Represent a spreadsheet region for an LLM to understand cell references and data layout."
        steps={[
          { formula: '\\text{Headers: A1="Name", B1="Score", C1="Grade"}', explanation: 'Include column headers with their cell references for formula generation.' },
          { formula: '\\text{Sample: A2="Alice", B2=92, C2="A"}', explanation: 'Provide 2-3 sample rows so the model understands data types and patterns.' },
          { formula: '\\text{Range: A2:C101 (100 data rows)}', explanation: 'Specify the data range so formulas reference the correct extent.' },
          { formula: '\\text{Existing formulas: C2=IF(B2>=90,"A",IF(B2>=80,"B","C"))}', explanation: 'Show existing formulas to help the model maintain consistency.' },
        ]}
        id="example-context"
      />

      <h2 className="text-2xl font-semibold">Programmatic Spreadsheet Manipulation</h2>

      <PythonCode
        title="spreadsheet_automation.py"
        code={`import openpyxl
from openpyxl.utils import get_column_letter

# Create a spreadsheet programmatically
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Analysis"

# Write headers and data
headers = ["Month", "Revenue", "Costs", "Profit", "Margin"]
data = [
    ["Jan", 50000, 35000, None, None],
    ["Feb", 62000, 38000, None, None],
    ["Mar", 58000, 36000, None, None],
]

for col, h in enumerate(headers, 1):
    ws.cell(row=1, column=col, value=h)

for row_idx, row_data in enumerate(data, 2):
    for col_idx, value in enumerate(row_data, 1):
        ws.cell(row=row_idx, column=col_idx, value=value)

# LLM-generated formulas inserted programmatically
for row in range(2, len(data) + 2):
    # Profit = Revenue - Costs
    ws.cell(row=row, column=4, value=f"=B{row}-C{row}")
    # Margin = Profit / Revenue
    ws.cell(row=row, column=5, value=f"=D{row}/B{row}")

# Add summary formulas
summary_row = len(data) + 3
ws.cell(row=summary_row, column=1, value="Total")
for col in range(2, 5):
    letter = get_column_letter(col)
    ws.cell(row=summary_row, column=col,
            value=f"=SUM({letter}2:{letter}{len(data)+1})")

ws.cell(row=summary_row, column=5,
        value=f"=D{summary_row}/B{summary_row}")

wb.save("/tmp/analysis.xlsx")
print("Spreadsheet created with LLM-generated formulas")`}
        id="code-automation"
      />

      <NoteBlock
        type="note"
        title="SheetCopilot and Copilot in Excel"
        content="Microsoft's Copilot in Excel and research prototypes like SheetCopilot demonstrate end-to-end spreadsheet automation. These systems combine LLMs with spreadsheet APIs to execute multi-step operations: creating pivot tables, generating charts, applying conditional formatting, and writing VBA macros -- all from natural language commands."
        id="note-copilot"
      />

      <WarningBlock
        title="Formula Correctness Risks"
        content="LLM-generated formulas can contain subtle errors: off-by-one in ranges (A2:A100 vs A2:A101), incorrect absolute/relative references ($A$1 vs A1), wrong function arguments order, or logical errors in nested IF statements. Always verify generated formulas against expected results on sample data before applying to the full dataset."
        id="warning-correctness"
      />

      <NoteBlock
        type="intuition"
        title="Why Spreadsheets Are Hard for LLMs"
        content="Spreadsheets combine three reasoning modalities: spatial (cell A3 is below A2 and left of B3), formulaic (understanding precedent/dependent cell relationships), and semantic (knowing that column B contains revenue). LLMs must juggle all three simultaneously, which is why specialized context representations that encode spatial layout alongside data content significantly outperform naive serialization."
        id="note-challenge"
      />
    </div>
  )
}
