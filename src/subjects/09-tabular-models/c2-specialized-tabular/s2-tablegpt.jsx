import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function TableGPT() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">TableGPT: Unified Table Understanding</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        TableGPT represents a family of approaches that fine-tune GPT-style language models
        specifically for table tasks. By training on diverse table operations -- including
        completion, transformation, QA, and data cleaning -- TableGPT creates a unified model
        that handles the full spectrum of table manipulation through natural language instructions.
      </p>

      <DefinitionBlock
        title="TableGPT Framework"
        definition="TableGPT fine-tunes a pretrained language model $M$ on a corpus of table-instruction pairs $\{(T_i, I_i, O_i)\}$ where $T_i$ is a serialized table, $I_i$ is a natural language instruction, and $O_i$ is the expected output (a modified table, answer, or code). The model learns $P(O \mid T, I; \theta)$ across multiple table tasks simultaneously."
        notation="T = table, I = instruction, O = output, \theta = model parameters"
        id="def-tablegpt"
      />

      <h2 className="text-2xl font-semibold">Multi-Task Table Operations</h2>
      <p className="text-gray-700 dark:text-gray-300">
        TableGPT handles a wide range of table tasks through a unified interface. Each task
        is framed as a text-to-text problem where the input includes the table and an instruction,
        and the output is the result.
      </p>

      <PythonCode
        title="tablegpt_tasks.py"
        code={`import pandas as pd
import json

# TableGPT handles multiple table tasks through natural language

# Task 1: Column Type Detection
table_str = """| Name    | Age | Salary  | Start Date |
| ---     | --- | ---     | ---        |
| Alice   | 30  | 85000   | 2020-01-15 |
| Bob     | 25  | 62000   | 2021-06-01 |"""

instruction_1 = "Detect the data type of each column."
# Expected: Name=string, Age=integer, Salary=integer, Start_Date=date

# Task 2: Data Imputation
table_with_missing = """| Product | Price | Rating |
| Widget  | 9.99  |        |
| Gadget  |       | 4.2    |
| Tool    | 15.99 | 3.8    |"""

instruction_2 = "Fill in the missing values based on patterns in the data."
# Expected: Rating for Widget ≈ 4.0, Price for Gadget ≈ 12.99

# Task 3: Table Transformation
instruction_3 = "Pivot this table so that quarters become columns."
source_table = pd.DataFrame({
    "Product": ["A", "A", "B", "B"],
    "Quarter": ["Q1", "Q2", "Q1", "Q2"],
    "Revenue": [100, 150, 200, 180]
})
print("Source table:")
print(source_table.to_markdown(index=False))

# Expected output: pivot table
pivot = source_table.pivot(index="Product", columns="Quarter", values="Revenue")
print("\\nExpected output:")
print(pivot.to_markdown())

# Task 4: Schema Matching
instruction_4 = "Map columns from Table A to Table B."
table_a_cols = ["emp_name", "emp_salary", "dept"]
table_b_cols = ["employee_name", "annual_pay", "department"]
mapping = dict(zip(table_a_cols, table_b_cols))
print(f"\\nSchema mapping: {json.dumps(mapping, indent=2)}")`}
        id="code-tasks"
      />

      <ExampleBlock
        title="TableGPT Training Data Construction"
        problem="Create a training example for the table completion task."
        steps={[
          { formula: '\\text{Select a complete table } T \\text{ from the corpus}', explanation: 'Start with a clean, complete table from web crawls or databases.' },
          { formula: '\\text{Mask cells: } T_{masked} = \\text{mask}(T, p=0.15)', explanation: 'Randomly mask 15% of cells to create the input table with missing values.' },
          { formula: '\\text{Create instruction } I = \\text{"Complete the missing cells"}', explanation: 'Generate a natural language instruction for the task.' },
          { formula: '\\text{Training pair: } (T_{masked}, I) \\to T', explanation: 'The model learns to reconstruct the original table from the masked version.' },
        ]}
        id="example-training"
      />

      <PythonCode
        title="tablegpt_inference.py"
        code={`# Simulating TableGPT-style inference with an LLM
from transformers import AutoTokenizer, AutoModelForCausalLM

def tablegpt_prompt(table_str, instruction):
    """Build a TableGPT-style prompt."""
    return f"""You are TableGPT, a model specialized in table operations.

Table:
{table_str}

Instruction: {instruction}

Output:"""

# Example: Data cleaning task
dirty_table = """| Name      | Email              | Phone        |
| ---       | ---                | ---          |
| J. Smith  | jsmith@email       | 555-1234     |
| MARY DOE  | mary@company.com   | (555)5678    |
| bob jones | BOB@COMPANY.COM    | 555.9012     |"""

instruction = "Clean this table: standardize name capitalization, validate emails, and normalize phone numbers to XXX-XXXX format."

prompt = tablegpt_prompt(dirty_table, instruction)
print(prompt)

# Expected output:
# | Name       | Email             | Phone    |
# | J. Smith   | jsmith@email.com  | 555-1234 |
# | Mary Doe   | mary@company.com  | 555-5678 |
# | Bob Jones  | bob@company.com   | 555-9012 |

# In practice, load a fine-tuned model:
# tokenizer = AutoTokenizer.from_pretrained("tablegpt/TableGPT2-7B")
# model = AutoModelForCausalLM.from_pretrained("tablegpt/TableGPT2-7B")
# inputs = tokenizer(prompt, return_tensors="pt")
# outputs = model.generate(**inputs, max_new_tokens=256)
# result = tokenizer.decode(outputs[0], skip_special_tokens=True)`}
        id="code-inference"
      />

      <NoteBlock
        type="note"
        title="TableGPT vs. General-Purpose LLMs"
        content="TableGPT models are fine-tuned on 2M+ table-instruction pairs covering 18 task categories. On benchmarks like table QA and data imputation, TableGPT-7B outperforms GPT-3.5 while being significantly smaller. However, general-purpose models like GPT-4 still lead on tasks requiring external knowledge or complex multi-step reasoning."
        id="note-comparison"
      />

      <WarningBlock
        title="Training Data Bias"
        content="TableGPT models can inherit biases from their training tables. If the training data predominantly contains tables from specific domains (e.g., finance, sports), the model may underperform on tables from underrepresented domains. Always evaluate on domain-specific data before deployment."
        id="warning-bias"
      />

      <NoteBlock
        type="tip"
        title="Combining TableGPT with Agents"
        content="TableGPT works well as a specialized tool within an agent framework. The agent routes table-specific tasks (cleaning, transformation, QA) to TableGPT while handling broader reasoning, planning, and integration with external data sources through general-purpose LLMs."
        id="note-agents"
      />
    </div>
  )
}
