import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function TableQA() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Table Question Answering</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Table Question Answering (Table QA) is the task of answering natural language questions
        about the contents of a table. Unlike Text-to-SQL, Table QA can operate without generating
        an intermediate formal query, instead reasoning directly over the serialized table to
        produce an answer in natural language.
      </p>

      <DefinitionBlock
        title="Table Question Answering"
        definition="Given a table $T$ with headers $H = \{h_1, \ldots, h_n\}$ and $m$ rows, and a natural language question $q$, Table QA produces an answer $a$ that may be a cell value, an aggregation result, or a derived statement. The answer can be extractive ($a \in T$) or abstractive ($a$ synthesized from $T$)."
        notation="T = table, H = headers, q = question, a = answer"
        id="def-table-qa"
      />

      <h2 className="text-2xl font-semibold">Direct Table QA with LLMs</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The simplest approach serializes the table into the prompt and asks the model to answer
        the question directly. This avoids SQL generation entirely and works well for lookup
        questions, simple aggregations, and comparison tasks.
      </p>

      <PythonCode
        title="table_qa_prompt.py"
        code={`import pandas as pd

# Sample table
df = pd.DataFrame({
    "Country": ["USA", "China", "India", "Germany", "Brazil"],
    "GDP_Trillion": [25.46, 17.96, 3.39, 4.26, 1.92],
    "Population_M": [331, 1412, 1408, 83, 214],
    "Continent": ["N. America", "Asia", "Asia", "Europe", "S. America"]
})

def table_qa_prompt(df, question):
    table_str = df.to_markdown(index=False)
    return f"""Answer the question based on the table below.

Table:
{table_str}

Question: {question}
Answer:"""

# Different question types
questions = [
    "Which country has the highest GDP?",           # Lookup / argmax
    "What is the total GDP of Asian countries?",    # Filter + aggregate
    "Is Germany's GDP higher than Brazil's?",       # Comparison
    "How many countries have population over 300M?", # Count + filter
]

for q in questions:
    prompt = table_qa_prompt(df, q)
    print(f"Q: {q}")
    print(f"Prompt length: {len(prompt)} chars\\n")

# Expected answers:
# USA, 21.35 trillion, Yes, 3 countries`}
        id="code-qa-prompt"
      />

      <ExampleBlock
        title="Table QA Operation Types"
        problem="Categorize a question into the required table operation to plan the answering strategy."
        steps={[
          { formula: '\\text{Lookup: "What is X for row Y?"}', explanation: 'Direct cell retrieval. The model finds the row matching Y and returns the value in column X.' },
          { formula: '\\text{Aggregation: "What is the total/average/max of X?"}', explanation: 'Requires computing a function over a column. Models must perform arithmetic reasoning.' },
          { formula: '\\text{Comparison: "Is A greater than B?"}', explanation: 'Requires retrieving two values and comparing them. Answer is typically boolean.' },
          { formula: '\\text{Multi-hop: "What is X for the row with max Y?"}', explanation: 'Requires chaining operations: first find the row with max Y, then retrieve X.' },
        ]}
        id="example-operations"
      />

      <h2 className="text-2xl font-semibold">Program-Aided Table QA</h2>
      <p className="text-gray-700 dark:text-gray-300">
        For questions requiring complex aggregations or multi-step reasoning, having the LLM
        generate executable Python code (rather than answering directly) significantly improves
        accuracy, especially for numeric computations.
      </p>

      <PythonCode
        title="program_aided_table_qa.py"
        code={`# Program-aided approach: LLM generates Python code for the answer
import pandas as pd

df = pd.DataFrame({
    "Quarter": ["Q1", "Q2", "Q3", "Q4"],
    "Revenue": [1200000, 1450000, 1380000, 1620000],
    "Expenses": [980000, 1100000, 1050000, 1200000],
})

question = "What is the quarter-over-quarter revenue growth rate?"

# LLM would generate this code:
generated_code = '''
df["Revenue_Growth"] = df["Revenue"].pct_change() * 100
result = df[["Quarter", "Revenue_Growth"]].dropna().to_string(index=False)
'''

# Execute the generated code safely
exec(generated_code)
print(f"Question: {question}")
print(f"Result:\\n{result}")
# Quarter  Revenue_Growth
#      Q2       20.833333
#      Q3       -4.827586
#      Q4       17.391304

# Chain-of-Table approach: decompose into sub-operations
def chain_of_table(df, question):
    """Simulate Chain-of-Table reasoning."""
    steps = []

    # Step 1: Select relevant columns
    relevant = df[["Quarter", "Revenue"]]
    steps.append(("SELECT columns", relevant.columns.tolist()))

    # Step 2: Add derived column
    relevant = relevant.copy()
    relevant["Growth_%"] = relevant["Revenue"].pct_change() * 100
    steps.append(("ADD derived column", "Growth_%"))

    # Step 3: Extract answer
    answer = relevant.dropna().to_dict("records")
    steps.append(("EXTRACT answer", answer))

    return steps

for step_name, detail in chain_of_table(df, question):
    print(f"  {step_name}: {detail}")`}
        id="code-program-aided"
      />

      <NoteBlock
        type="intuition"
        title="Direct Answer vs. Code Generation"
        content="Direct answering is like mental math -- fast but error-prone for complex calculations. Code generation is like using a calculator -- the LLM's job is reduced to understanding the question and writing correct code, while the actual computation is handled by a Python interpreter. This separation of concerns is why program-aided approaches outperform direct answering on numeric-heavy table QA."
        id="note-intuition"
      />

      <WarningBlock
        title="Table QA Failure Modes"
        content="LLMs commonly fail on: (1) large tables that exceed context length, (2) questions requiring precise arithmetic over many cells, (3) ambiguous column names where the model picks the wrong column, and (4) tables with merged cells or hierarchical headers. Always validate answers for critical applications."
        id="warning-failures"
      />

      <NoteBlock
        type="note"
        title="Benchmarks for Table QA"
        content="WikiTableQuestions (WTQ) contains 22,033 question-answer pairs over Wikipedia tables. SQA (Sequential Question Answering) tests multi-turn reasoning. HybridQA combines tables with linked text passages. The TableFact dataset evaluates fact verification against tables. State-of-the-art models achieve 70-80% accuracy on WTQ, with program-aided approaches leading."
        id="note-benchmarks"
      />
    </div>
  )
}
