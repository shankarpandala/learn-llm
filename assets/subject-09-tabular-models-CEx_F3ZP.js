import{j as e}from"./vendor-DWbzdFaj.js";import"./vendor-katex-BYl39Yo6.js";import{D as n,P as t,E as i,N as a,W as o,T as r}from"./subject-01-text-fundamentals-DG6tAvii.js";function s(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Table Serialization: Markdown, CSV, and JSON Formats"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Before an LLM can reason about tabular data, the table must be converted into a linear text sequence. The choice of serialization format significantly affects how well the model understands structure, relationships between columns, and the data types within each cell."}),e.jsx(n,{title:"Table Serialization",definition:"Table serialization is the process of converting a two-dimensional table with $m$ rows and $n$ columns into a linear string representation that preserves structural information. The serialized form $S(T)$ must encode both schema (column names, types) and cell values.",notation:"T = table, m = rows, n = columns, S(T) = serialized string",id:"def-serialization"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Markdown Format"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Markdown tables use pipes and dashes to create a human-readable grid layout. This format preserves column alignment visually and is common in LLM prompts because models have seen extensive markdown during pretraining."}),e.jsx(t,{title:"markdown_serialization.py",code:`import pandas as pd

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
print(df.to_markdown(index=False))`,id:"code-markdown"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"CSV Format"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"CSV (Comma-Separated Values) is the most compact serialization. It uses fewer tokens than markdown but loses visual alignment cues. Each row maps directly to one line of text."}),e.jsx(t,{title:"csv_serialization.py",code:`import pandas as pd
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
print(f"JSON-L:   {len(jsonl_str)} chars")`,id:"code-csv"}),e.jsx(i,{title:"JSON Serialization Variants",problem:"Serialize a 2-row table as JSON using records, columns, and split orientations.",steps:[{formula:'\\text{records: } [\\{\\text{"a": 1, "b": 2}\\}, \\{\\text{"a": 3, "b": 4}\\}]',explanation:"Each row becomes an independent JSON object. Best for row-level reasoning."},{formula:'\\text{columns: } \\{\\text{"a": [1, 3], "b": [2, 4]}\\}',explanation:"Grouped by column. Best for column-level aggregation questions."},{formula:'\\text{split: } \\{\\text{columns: ["a","b"], data: [[1,2],[3,4]]}\\}',explanation:"Separates schema from data. Most compact for large tables."}],id:"example-json-variants"}),e.jsx(a,{type:"tip",title:"Choosing the Right Format",content:"Empirical studies show that markdown tables work best for small tables (under 20 rows) where visual alignment helps. CSV is more token-efficient for larger tables. JSON records format is preferred when the model needs to reason about individual rows or when column names contain special characters.",id:"note-choosing-format"}),e.jsx(o,{title:"Token Budget and Table Size",content:"A table with 100 rows and 10 columns can easily consume 3,000-5,000 tokens in markdown format. Always estimate token usage before serializing large tables. Consider truncation strategies: sample representative rows, include only relevant columns, or summarize numeric columns with statistics.",id:"warning-token-budget"}),e.jsx(a,{type:"historical",title:"Evolution of Table Serialization",content:"Early table-QA systems like TAPAS operated directly on structured table inputs. The shift to serialization-based approaches began with GPT-3 (2020), where researchers discovered that simply pasting tables as text into prompts yielded surprisingly good results. Sui et al. (2023) systematically benchmarked serialization formats, finding that the optimal choice depends on both table size and task type.",id:"note-history"})]})}const j=Object.freeze(Object.defineProperty({__proto__:null,default:s},Symbol.toStringTag,{value:"Module"}));function l(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"TabLLM: Few-Shot Tabular Classification with LLMs"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"TabLLM demonstrates that large language models can perform tabular classification by converting each row into a natural language description. This text-serialization approach enables few-shot learning on structured data, often matching or exceeding traditional ML models like gradient-boosted trees when labeled data is scarce."}),e.jsx(n,{title:"TabLLM",definition:"TabLLM is a framework that serializes each tabular data row into a natural language sentence, then uses an LLM for classification. Given a row $x = (x_1, \\ldots, x_n)$ with column names $(c_1, \\ldots, c_n)$, the serialization produces a text $t(x)$ and the model predicts $P(y \\mid t(x))$.",notation:"x = feature vector, c_i = column name, t(x) = text serialization, y = class label",id:"def-tabllm"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Serialization Templates"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"TabLLM explores multiple serialization strategies to convert tabular rows into text. The text template approach constructs natural language sentences from column-value pairs, which leverages the language model's understanding of natural language semantics."}),e.jsx(t,{title:"tabllm_serialization.py",code:`# TabLLM serialization templates for a single data row
row = {
    "age": 35, "income": 75000, "education": "Bachelor's",
    "marital_status": "Married", "occupation": "Engineer"
}
label_col = "high_earner"  # binary classification target

# Template 1: Text template (natural language)
def text_template(row):
    parts = [f"The {k.replace('_', ' ')} is {v}" for k, v in row.items()]
    return ". ".join(parts) + "."

# Template 2: List template
def list_template(row):
    parts = [f"- {k}: {v}" for k, v in row.items()]
    return "\\n".join(parts)

# Template 3: Key-value serialization
def kv_template(row):
    return " | ".join(f"{k}={v}" for k, v in row.items())

print("=== Text Template ===")
print(text_template(row))
# The age is 35. The income is 75000. The education is Bachelor's...

print("\\n=== List Template ===")
print(list_template(row))

print("\\n=== Key-Value Template ===")
print(kv_template(row))

# Few-shot prompt construction
def build_prompt(train_examples, test_row, template_fn):
    prompt = "Classify whether the person is a high earner.\\n\\n"
    for ex_row, ex_label in train_examples:
        prompt += template_fn(ex_row) + f"\\nLabel: {ex_label}\\n\\n"
    prompt += template_fn(test_row) + "\\nLabel:"
    return prompt

print("\\n=== Few-Shot Prompt ===")
print(build_prompt(
    [(row, "Yes")],
    {"age": 22, "income": 35000, "education": "High School",
     "marital_status": "Single", "occupation": "Retail"},
    text_template
))`,id:"code-serialization"}),e.jsx(i,{title:"TabLLM Classification Pipeline",problem:"Classify a loan applicant as low/high risk using TabLLM with 4 labeled examples.",steps:[{formula:"\\text{Serialize each row } x_i \\to t(x_i)",explanation:"Convert the tabular row into a natural language description using a template."},{formula:"\\text{Construct prompt: } [t(x_1), y_1, \\ldots, t(x_4), y_4, t(x_{\\text{test}})]",explanation:"Build a few-shot prompt with serialized examples and their labels."},{formula:"P(y \\mid \\text{prompt}) = \\text{LLM}(\\text{prompt})",explanation:"The LLM generates the predicted label by completing the prompt."},{formula:"\\hat{y} = \\arg\\max_{y} P(y \\mid \\text{prompt})",explanation:"Select the class with the highest probability as the prediction."}],id:"example-pipeline"}),e.jsx(a,{type:"intuition",title:"Why Text Serialization Helps",content:"LLMs have been pretrained on vast text corpora containing descriptions of people, products, and events. When a tabular row is serialized as 'The age is 35 and the income is 75000', the model can leverage its world knowledge about typical income levels for different ages, effectively performing implicit feature engineering.",id:"note-intuition"}),e.jsx(t,{title:"tabllm_with_t5.py",code:`# TabLLM-style classification using a T5 model
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Serialize a row as natural language
def serialize_row(row):
    return ". ".join(f"The {k} is {v}" for k, v in row.items())

# Build classification prompt
row = {"age": "45", "job": "technician", "balance": "1200"}
text = f"Classify subscription: {serialize_row(row)}. Will subscribe?"

inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)
outputs = model.generate(**inputs, max_new_tokens=5)
prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prediction: {prediction}")

# For proper TabLLM: fine-tune T5 on serialized tabular data
# model.train()
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# for row_text, label_text in train_loader:
#     inputs = tokenizer(row_text, return_tensors="pt", padding=True)
#     labels = tokenizer(label_text, return_tensors="pt", padding=True)
#     loss = model(**inputs, labels=labels.input_ids).loss
#     loss.backward()
#     optimizer.step()`,id:"code-t5"}),e.jsx(o,{title:"When TabLLM Falls Short",content:"TabLLM struggles with high-dimensional numeric data where feature interactions matter more than semantic meaning. For datasets with 50+ numeric columns (e.g., sensor readings), gradient-boosted trees like XGBoost still dominate. TabLLM is most effective when column names carry semantic meaning and labeled data is limited (under 100 examples).",id:"warning-limitations"}),e.jsx(a,{type:"note",title:"Key Results from TabLLM (Hegselmann et al., 2023)",content:"On 9 benchmark datasets, TabLLM with T0 (11B parameters) matched or exceeded XGBoost performance in 4-shot and 8-shot settings. The text template serialization consistently outperformed list and CSV formats. Fine-tuning on serialized data further improved results, especially with LoRA adapters to keep training efficient.",id:"note-results"})]})}const L=Object.freeze(Object.defineProperty({__proto__:null,default:l},Symbol.toStringTag,{value:"Module"}));function d(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Text-to-SQL with Large Language Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Text-to-SQL converts natural language questions into executable SQL queries. LLMs have transformed this task from a specialized semantic parsing problem into a prompting challenge, achieving state-of-the-art accuracy on benchmarks like Spider and BIRD by leveraging in-context learning with database schema information."}),e.jsx(n,{title:"Text-to-SQL",definition:"Text-to-SQL is the task of mapping a natural language question $q$ and a database schema $S = \\{(t_i, \\{c_{i,1}, \\ldots, c_{i,k}\\})\\}$ to a valid SQL query $Q$ such that executing $Q$ on the database returns the answer to $q$. Formally: $f(q, S) \\to Q$ where $\\text{exec}(Q, D) = \\text{answer}(q)$.",notation:"q = natural language question, S = schema, t_i = table name, c_{i,j} = column, Q = SQL query, D = database",id:"def-text-to-sql"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Schema-Aware Prompting"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The key to accurate Text-to-SQL is providing the model with a clear representation of the database schema, including table names, column names, data types, and foreign key relationships."}),e.jsx(t,{title:"text_to_sql_prompt.py",code:`# Building a Text-to-SQL prompt with schema information
schema = """
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER,
    salary DECIMAL(10,2),
    hire_date DATE,
    FOREIGN KEY (department_id) REFERENCES departments(id)
);

CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    budget DECIMAL(12,2),
    manager_id INTEGER
);

CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    title TEXT,
    department_id INTEGER,
    start_date DATE,
    end_date DATE,
    FOREIGN KEY (department_id) REFERENCES departments(id)
);
"""

def build_text_to_sql_prompt(question, schema, examples=None):
    prompt = "Given the following database schema:\\n"
    prompt += schema + "\\n"
    prompt += "Write a SQL query to answer the question.\\n"
    prompt += "Return ONLY the SQL query, no explanations.\\n\\n"

    if examples:
        for q, sql in examples:
            prompt += f"Question: {q}\\nSQL: {sql}\\n\\n"

    prompt += f"Question: {question}\\nSQL:"
    return prompt

# Few-shot examples
examples = [
    ("How many employees are there?",
     "SELECT COUNT(*) FROM employees;"),
    ("What is the average salary by department?",
     "SELECT d.name, AVG(e.salary) FROM employees e "
     "JOIN departments d ON e.department_id = d.id GROUP BY d.name;"),
]

question = "Which departments have more than 5 employees with salary above 80000?"
prompt = build_text_to_sql_prompt(question, schema, examples)
print(prompt)

# Expected SQL output:
# SELECT d.name FROM departments d
# JOIN employees e ON e.department_id = d.id
# WHERE e.salary > 80000
# GROUP BY d.name
# HAVING COUNT(*) > 5;`,id:"code-prompt"}),e.jsx(i,{title:"Multi-Step Text-to-SQL with Chain-of-Thought",problem:"Generate SQL for: 'Find the department with the highest total salary that also has an active project.'",steps:[{formula:"\\text{Step 1: Identify relevant tables: employees, departments, projects}",explanation:"Parse the question to determine which tables are needed."},{formula:'\\text{Step 2: "highest total salary" } \\to \\text{SUM(salary) + ORDER BY + LIMIT}',explanation:"Map the phrase to SQL aggregation and ordering constructs."},{formula:'\\text{Step 3: "active project" } \\to \\text{WHERE end\\_date >= CURRENT\\_DATE}',explanation:"Interpret the filter condition from natural language."},{formula:"\\text{Step 4: Combine with JOIN on department\\_id}",explanation:"Link tables through foreign key relationships to form the complete query."}],id:"example-cot"}),e.jsx(t,{title:"text_to_sql_execution.py",code:`import sqlite3

# Create an in-memory database and execute generated SQL
conn = sqlite3.connect(":memory:")
cursor = conn.cursor()

# Set up schema
cursor.executescript("""
CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, budget REAL);
CREATE TABLE employees (
    id INTEGER PRIMARY KEY, name TEXT,
    department_id INTEGER, salary REAL
);
INSERT INTO departments VALUES (1, 'Engineering', 500000);
INSERT INTO departments VALUES (2, 'Marketing', 200000);
INSERT INTO employees VALUES (1, 'Alice', 1, 95000);
INSERT INTO employees VALUES (2, 'Bob', 1, 88000);
INSERT INTO employees VALUES (3, 'Carol', 2, 72000);
""")

# Simulated LLM-generated SQL
question = "What is the average salary per department?"
generated_sql = """
SELECT d.name, ROUND(AVG(e.salary), 2) as avg_salary
FROM employees e
JOIN departments d ON e.department_id = d.id
GROUP BY d.name
ORDER BY avg_salary DESC;
"""

# Execute with safety checks
def safe_execute(cursor, sql, read_only=True):
    sql_upper = sql.strip().upper()
    if read_only and not sql_upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")
    try:
        cursor.execute(sql)
        return cursor.fetchall()
    except sqlite3.Error as e:
        return f"SQL Error: {e}"

results = safe_execute(cursor, generated_sql)
print(f"Question: {question}")
print(f"Results: {results}")
# [('Engineering', 91500.0), ('Marketing', 72000.0)]
conn.close()`,id:"code-execution"}),e.jsx(o,{title:"SQL Injection and Safety",content:"Never execute LLM-generated SQL without validation. Always restrict to SELECT queries when the intent is read-only, use parameterized queries for user inputs, and run generated SQL in sandboxed environments with limited permissions. Consider using query validation libraries to parse and check the SQL AST before execution.",id:"warning-safety"}),e.jsx(a,{type:"note",title:"Benchmark Performance",content:"On the Spider benchmark (cross-database Text-to-SQL), GPT-4 with few-shot prompting achieves ~85% execution accuracy, compared to ~72% for fine-tuned T5-3B. The BIRD benchmark adds real-world complexity with dirty data and external knowledge, where GPT-4 scores ~55%. The gap highlights that schema understanding and SQL generation are largely solved, but handling messy real data remains challenging.",id:"note-benchmarks"}),e.jsx(a,{type:"tip",title:"Self-Correction Improves Accuracy",content:"A powerful technique is to execute the generated SQL, then show the model both the query and the results (or error message), and ask it to verify or fix the query. This 'generate-execute-refine' loop can boost accuracy by 5-10% on complex queries involving multiple joins and subqueries.",id:"note-self-correction"})]})}const A=Object.freeze(Object.defineProperty({__proto__:null,default:d},Symbol.toStringTag,{value:"Module"}));function c(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Table Question Answering"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Table Question Answering (Table QA) is the task of answering natural language questions about the contents of a table. Unlike Text-to-SQL, Table QA can operate without generating an intermediate formal query, instead reasoning directly over the serialized table to produce an answer in natural language."}),e.jsx(n,{title:"Table Question Answering",definition:"Given a table $T$ with headers $H = \\{h_1, \\ldots, h_n\\}$ and $m$ rows, and a natural language question $q$, Table QA produces an answer $a$ that may be a cell value, an aggregation result, or a derived statement. The answer can be extractive ($a \\in T$) or abstractive ($a$ synthesized from $T$).",notation:"T = table, H = headers, q = question, a = answer",id:"def-table-qa"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Direct Table QA with LLMs"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The simplest approach serializes the table into the prompt and asks the model to answer the question directly. This avoids SQL generation entirely and works well for lookup questions, simple aggregations, and comparison tasks."}),e.jsx(t,{title:"table_qa_prompt.py",code:`import pandas as pd

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
# USA, 21.35 trillion, Yes, 3 countries`,id:"code-qa-prompt"}),e.jsx(i,{title:"Table QA Operation Types",problem:"Categorize a question into the required table operation to plan the answering strategy.",steps:[{formula:'\\text{Lookup: "What is X for row Y?"}',explanation:"Direct cell retrieval. The model finds the row matching Y and returns the value in column X."},{formula:'\\text{Aggregation: "What is the total/average/max of X?"}',explanation:"Requires computing a function over a column. Models must perform arithmetic reasoning."},{formula:'\\text{Comparison: "Is A greater than B?"}',explanation:"Requires retrieving two values and comparing them. Answer is typically boolean."},{formula:'\\text{Multi-hop: "What is X for the row with max Y?"}',explanation:"Requires chaining operations: first find the row with max Y, then retrieve X."}],id:"example-operations"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Program-Aided Table QA"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"For questions requiring complex aggregations or multi-step reasoning, having the LLM generate executable Python code (rather than answering directly) significantly improves accuracy, especially for numeric computations."}),e.jsx(t,{title:"program_aided_table_qa.py",code:`# Program-aided approach: LLM generates Python code for the answer
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
    print(f"  {step_name}: {detail}")`,id:"code-program-aided"}),e.jsx(a,{type:"intuition",title:"Direct Answer vs. Code Generation",content:"Direct answering is like mental math -- fast but error-prone for complex calculations. Code generation is like using a calculator -- the LLM's job is reduced to understanding the question and writing correct code, while the actual computation is handled by a Python interpreter. This separation of concerns is why program-aided approaches outperform direct answering on numeric-heavy table QA.",id:"note-intuition"}),e.jsx(o,{title:"Table QA Failure Modes",content:"LLMs commonly fail on: (1) large tables that exceed context length, (2) questions requiring precise arithmetic over many cells, (3) ambiguous column names where the model picks the wrong column, and (4) tables with merged cells or hierarchical headers. Always validate answers for critical applications.",id:"warning-failures"}),e.jsx(a,{type:"note",title:"Benchmarks for Table QA",content:"WikiTableQuestions (WTQ) contains 22,033 question-answer pairs over Wikipedia tables. SQA (Sequential Question Answering) tests multi-turn reasoning. HybridQA combines tables with linked text passages. The TableFact dataset evaluates fact verification against tables. State-of-the-art models achieve 70-80% accuracy on WTQ, with program-aided approaches leading.",id:"note-benchmarks"})]})}const E=Object.freeze(Object.defineProperty({__proto__:null,default:c},Symbol.toStringTag,{value:"Module"}));function m(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"TAPAS: Weakly Supervised Table Parsing"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"TAPAS (Table Parser) is a BERT-based model specifically designed for table understanding. Unlike serialization-based approaches, TAPAS encodes tables using special positional embeddings that preserve row and column structure, enabling it to reason over tables without converting them to flat text."}),e.jsx(n,{title:"TAPAS Architecture",definition:"TAPAS extends BERT with additional embedding layers for tabular structure. Each token receives embeddings for: token position, segment (question vs. table), column index $c_i \\in \\{0, \\ldots, C\\}$, row index $r_j \\in \\{0, \\ldots, R\\}$, and numeric rank within the column. The final embedding is $e = e_{token} + e_{position} + e_{segment} + e_{column} + e_{row} + e_{rank}$.",notation:"e = embedding, C = max columns, R = max rows",id:"def-tapas"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Structural Embeddings"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The key innovation of TAPAS is its structural embeddings that encode the 2D position of each token within the table. This allows the transformer to distinguish between tokens in different cells without relying on delimiters or formatting."}),e.jsx(i,{title:"TAPAS Input Encoding",problem:"Show how a question + 2x2 table is encoded with TAPAS positional embeddings.",steps:[{formula:"\\text{Input: [CLS] question tokens [SEP] header1 header2 cell11 cell12 cell21 cell22}",explanation:"Flatten the table row-by-row after the question, separated by [SEP]."},{formula:"\\text{Column IDs: } [0, 0, \\ldots, 0, 1, 2, 1, 2, 1, 2]",explanation:"Question tokens get column=0, then each cell gets its column index."},{formula:"\\text{Row IDs: } [0, 0, \\ldots, 0, 0, 0, 1, 1, 2, 2]",explanation:"Question tokens and headers get row=0, data rows are numbered 1, 2, ..."},{formula:"e_i = e_{token_i} + e_{pos_i} + e_{seg_i} + e_{col_i} + e_{row_i}",explanation:"All embeddings are summed to produce the final input representation."}],id:"example-encoding"}),e.jsx(t,{title:"tapas_inference.py",code:`from transformers import TapasTokenizer, TapasForQuestionAnswering
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
    print(f"  Aggregation: {['NONE','SUM','AVG','COUNT'][agg_indices[0]]}\\n")`,id:"code-tapas"}),e.jsx(a,{type:"historical",title:"TAPAS Development Timeline",content:"TAPAS was introduced by Herzig et al. (2020) at Google Research. It was pretrained on millions of Wikipedia tables using a masked language modeling objective adapted for tables. TAPAS achieved state-of-the-art on WTQ (WikiTableQuestions), SQA, and TabFact. Later versions (TaPEx, 2022) showed that pretraining on synthetic SQL execution traces could further improve table reasoning.",id:"note-history"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Aggregation Operations"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"TAPAS jointly predicts which cells are relevant and what aggregation operation to apply. This allows it to handle questions that require counting, summing, or averaging without generating SQL."}),e.jsx(t,{title:"tapas_cell_selection.py",code:`import torch
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
# cell combinations that produce the correct answer`,id:"code-cell-selection"}),e.jsx(o,{title:"TAPAS Limitations",content:"TAPAS is limited to tables that fit within BERT's 512-token context window, which typically means tables with fewer than 50-100 cells. It also requires all cell values to be strings (numbers must be stringified), and struggles with tables containing merged cells, multi-level headers, or hierarchical row indices.",id:"warning-limitations"}),e.jsx(a,{type:"tip",title:"When to Use TAPAS vs. LLM-Based Approaches",content:"Use TAPAS when you need fast, consistent inference on small-to-medium tables with well-defined schemas. Use LLM-based approaches when tables are large (with truncation strategies), questions require world knowledge beyond the table, or you need natural language explanations alongside answers.",id:"note-when-to-use"})]})}const N=Object.freeze(Object.defineProperty({__proto__:null,default:m},Symbol.toStringTag,{value:"Module"}));function p(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"TableGPT: Unified Table Understanding"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"TableGPT represents a family of approaches that fine-tune GPT-style language models specifically for table tasks. By training on diverse table operations -- including completion, transformation, QA, and data cleaning -- TableGPT creates a unified model that handles the full spectrum of table manipulation through natural language instructions."}),e.jsx(n,{title:"TableGPT Framework",definition:"TableGPT fine-tunes a pretrained language model $M$ on a corpus of table-instruction pairs $\\{(T_i, I_i, O_i)\\}$ where $T_i$ is a serialized table, $I_i$ is a natural language instruction, and $O_i$ is the expected output (a modified table, answer, or code). The model learns $P(O \\mid T, I; \\theta)$ across multiple table tasks simultaneously.",notation:"T = table, I = instruction, O = output, \\theta = model parameters",id:"def-tablegpt"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Multi-Task Table Operations"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"TableGPT handles a wide range of table tasks through a unified interface. Each task is framed as a text-to-text problem where the input includes the table and an instruction, and the output is the result."}),e.jsx(t,{title:"tablegpt_tasks.py",code:`import pandas as pd
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
print(f"\\nSchema mapping: {json.dumps(mapping, indent=2)}")`,id:"code-tasks"}),e.jsx(i,{title:"TableGPT Training Data Construction",problem:"Create a training example for the table completion task.",steps:[{formula:"\\text{Select a complete table } T \\text{ from the corpus}",explanation:"Start with a clean, complete table from web crawls or databases."},{formula:"\\text{Mask cells: } T_{masked} = \\text{mask}(T, p=0.15)",explanation:"Randomly mask 15% of cells to create the input table with missing values."},{formula:'\\text{Create instruction } I = \\text{"Complete the missing cells"}',explanation:"Generate a natural language instruction for the task."},{formula:"\\text{Training pair: } (T_{masked}, I) \\to T",explanation:"The model learns to reconstruct the original table from the masked version."}],id:"example-training"}),e.jsx(t,{title:"tablegpt_inference.py",code:`# Simulating TableGPT-style inference with an LLM
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
# result = tokenizer.decode(outputs[0], skip_special_tokens=True)`,id:"code-inference"}),e.jsx(a,{type:"note",title:"TableGPT vs. General-Purpose LLMs",content:"TableGPT models are fine-tuned on 2M+ table-instruction pairs covering 18 task categories. On benchmarks like table QA and data imputation, TableGPT-7B outperforms GPT-3.5 while being significantly smaller. However, general-purpose models like GPT-4 still lead on tasks requiring external knowledge or complex multi-step reasoning.",id:"note-comparison"}),e.jsx(o,{title:"Training Data Bias",content:"TableGPT models can inherit biases from their training tables. If the training data predominantly contains tables from specific domains (e.g., finance, sports), the model may underperform on tables from underrepresented domains. Always evaluate on domain-specific data before deployment.",id:"warning-bias"}),e.jsx(a,{type:"tip",title:"Combining TableGPT with Agents",content:"TableGPT works well as a specialized tool within an agent framework. The agent routes table-specific tasks (cleaning, transformation, QA) to TableGPT while handling broader reasoning, planning, and integration with external data sources through general-purpose LLMs.",id:"note-agents"})]})}const C=Object.freeze(Object.defineProperty({__proto__:null,default:p},Symbol.toStringTag,{value:"Module"}));function u(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Spreadsheet Integration with LLMs"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Spreadsheets are the most ubiquitous form of structured data, with billions of Excel and Google Sheets files in active use. Integrating LLMs with spreadsheets enables natural language formula generation, data analysis, and automated transformations that make structured data manipulation accessible to non-technical users."}),e.jsx(n,{title:"Spreadsheet LLM Integration",definition:"Spreadsheet-LLM integration maps a spreadsheet state $S = (C, F, V)$ -- where $C$ is the cell grid, $F$ is the formula set, and $V$ are computed values -- with a natural language instruction $I$ to produce an action $A$ that modifies the spreadsheet. Actions include formula insertion, data transformation, chart creation, and formatting.",notation:"S = spreadsheet state, C = cells, F = formulas, V = values, I = instruction, A = action",id:"def-spreadsheet"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Formula Generation"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"One of the most impactful applications is generating spreadsheet formulas from natural language descriptions. LLMs can produce Excel formulas, Google Sheets functions, and even complex array formulas from plain English instructions."}),e.jsx(t,{title:"formula_generation.py",code:`# LLM-powered spreadsheet formula generation
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
    print(f"Expected: {expected}\\n")`,id:"code-formula"}),e.jsx(i,{title:"Spreadsheet Context Serialization",problem:"Represent a spreadsheet region for an LLM to understand cell references and data layout.",steps:[{formula:'\\text{Headers: A1="Name", B1="Score", C1="Grade"}',explanation:"Include column headers with their cell references for formula generation."},{formula:'\\text{Sample: A2="Alice", B2=92, C2="A"}',explanation:"Provide 2-3 sample rows so the model understands data types and patterns."},{formula:"\\text{Range: A2:C101 (100 data rows)}",explanation:"Specify the data range so formulas reference the correct extent."},{formula:'\\text{Existing formulas: C2=IF(B2>=90,"A",IF(B2>=80,"B","C"))}',explanation:"Show existing formulas to help the model maintain consistency."}],id:"example-context"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Programmatic Spreadsheet Manipulation"}),e.jsx(t,{title:"spreadsheet_automation.py",code:`import openpyxl
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
print("Spreadsheet created with LLM-generated formulas")`,id:"code-automation"}),e.jsx(a,{type:"note",title:"SheetCopilot and Copilot in Excel",content:"Microsoft's Copilot in Excel and research prototypes like SheetCopilot demonstrate end-to-end spreadsheet automation. These systems combine LLMs with spreadsheet APIs to execute multi-step operations: creating pivot tables, generating charts, applying conditional formatting, and writing VBA macros -- all from natural language commands.",id:"note-copilot"}),e.jsx(o,{title:"Formula Correctness Risks",content:"LLM-generated formulas can contain subtle errors: off-by-one in ranges (A2:A100 vs A2:A101), incorrect absolute/relative references ($A$1 vs A1), wrong function arguments order, or logical errors in nested IF statements. Always verify generated formulas against expected results on sample data before applying to the full dataset.",id:"warning-correctness"}),e.jsx(a,{type:"intuition",title:"Why Spreadsheets Are Hard for LLMs",content:"Spreadsheets combine three reasoning modalities: spatial (cell A3 is below A2 and left of B3), formulaic (understanding precedent/dependent cell relationships), and semantic (knowing that column B contains revenue). LLMs must juggle all three simultaneously, which is why specialized context representations that encode spatial layout alongside data content significantly outperform naive serialization.",id:"note-challenge"})]})}const M=Object.freeze(Object.defineProperty({__proto__:null,default:u},Symbol.toStringTag,{value:"Module"}));function h(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Schema Linking for SQL Generation"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Schema linking is the critical step of mapping mentions in a natural language question to the corresponding tables, columns, and values in a database schema. It bridges the gap between how users describe data and how it is actually structured, and is often the bottleneck in Text-to-SQL accuracy."}),e.jsx(n,{title:"Schema Linking",definition:"Schema linking is the process of identifying correspondences between spans in a natural language question $q$ and elements of a database schema $S$. Formally, it produces a set of alignments $L = \\{(s_i, e_j)\\}$ where $s_i$ is a span in $q$ and $e_j \\in \\{tables, columns, values\\}$ is a schema element. The alignment can be exact match, partial match, or semantic match.",notation:"q = question, S = schema, L = links, s_i = question span, e_j = schema element",id:"def-schema-linking"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Types of Schema Links"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Schema linking must handle multiple types of correspondences, from trivial exact matches to challenging semantic matches where the question uses completely different terminology than the schema."}),e.jsx(i,{title:"Schema Linking Categories",problem:"Given the question 'How many employees earn more than 100k in the engineering team?' and schema tables employees(id, name, salary, dept_id) and departments(id, dept_name), identify all schema links.",steps:[{formula:'\\text{"employees"} \\to \\texttt{employees} \\text{ (exact match)}',explanation:'The word "employees" directly matches the table name.'},{formula:'\\text{"earn"} \\to \\texttt{salary} \\text{ (semantic match)}',explanation:'"Earn" implies salary/compensation -- requires world knowledge.'},{formula:'\\text{"100k"} \\to \\texttt{100000} \\text{ (value match)}',explanation:'The shorthand "100k" must be interpreted as the numeric value 100000.'},{formula:'\\text{"engineering team"} \\to \\texttt{departments.dept\\_name} \\text{ (semantic match)}',explanation:'"Engineering team" maps to a value in the dept_name column, requiring a JOIN.'}],id:"example-categories"}),e.jsx(t,{title:"schema_linking.py",code:`import re
from difflib import SequenceMatcher

# Schema representation
schema = {
    "employees": ["id", "name", "salary", "department_id", "hire_date"],
    "departments": ["id", "dept_name", "budget", "location"],
    "projects": ["id", "title", "lead_id", "dept_id", "deadline"],
}

# Column descriptions for semantic matching
column_descriptions = {
    "employees.salary": ["pay", "earn", "income", "compensation", "wage"],
    "employees.name": ["employee", "person", "worker", "staff"],
    "employees.hire_date": ["hired", "joined", "started", "tenure"],
    "departments.dept_name": ["department", "team", "division", "group"],
    "departments.budget": ["funding", "allocation", "spend"],
    "projects.deadline": ["due date", "finish", "completion"],
}

def exact_match(question_tokens, schema):
    """Find exact matches between question tokens and schema elements."""
    links = []
    q_lower = question_tokens.lower()
    for table, columns in schema.items():
        if table in q_lower:
            links.append(("exact_table", table, table))
        for col in columns:
            if col.replace("_", " ") in q_lower:
                links.append(("exact_column", col, f"{table}.{col}"))
    return links

def semantic_match(question, column_descriptions):
    """Find semantic matches using synonym lists."""
    links = []
    q_lower = question.lower()
    for col_path, synonyms in column_descriptions.items():
        for syn in synonyms:
            if syn in q_lower:
                links.append(("semantic", syn, col_path))
                break
    return links

# Example
question = "How many workers earn over 100k in the engineering team?"
print(f"Question: {question}\\n")

for link in exact_match(question, schema):
    print(f"  Exact: '{link[1]}' -> {link[2]}")
for link in semantic_match(question, column_descriptions):
    print(f"  Semantic: '{link[1]}' -> {link[2]}")
`,id:"code-linking"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"LLM-Based Schema Linking"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Modern approaches use LLMs to perform schema linking as part of the SQL generation pipeline. The model identifies relevant tables and columns before generating the query, which reduces hallucination and improves accuracy on complex schemas."}),e.jsx(t,{title:"llm_schema_linking.py",code:`# LLM-based schema linking with structured output
import json

schema_ddl = """
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    full_name VARCHAR(100),
    email VARCHAR(100),
    signup_date DATE,
    plan_type VARCHAR(20)  -- 'free', 'basic', 'premium'
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    product_name VARCHAR(100),
    amount DECIMAL(10,2),
    order_date DATE,
    status VARCHAR(20)  -- 'pending', 'shipped', 'delivered'
);
"""

def schema_linking_prompt(question, schema_ddl):
    return f"""Given the database schema and question, identify the relevant
tables, columns, and any value mappings needed.

Schema:
{schema_ddl}

Question: {question}

Return JSON with:
- "tables": list of relevant table names
- "columns": list of "table.column" that are needed
- "value_maps": dict mapping question phrases to DB values
- "joins": list of join conditions needed

JSON:"""

question = "Show me the names and emails of premium customers who spent over $500 last month"
prompt = schema_linking_prompt(question, schema_ddl)

# Expected LLM output:
expected_output = {
    "tables": ["customers", "orders"],
    "columns": [
        "customers.full_name", "customers.email",
        "customers.plan_type", "orders.amount", "orders.order_date"
    ],
    "value_maps": {
        "premium": "customers.plan_type = 'premium'",
        "$500": "orders.amount > 500",
        "last month": "orders.order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)"
    },
    "joins": ["customers.customer_id = orders.customer_id"]
}

print(json.dumps(expected_output, indent=2))

# The schema linking output guides SQL generation
print("\\nSQL guided by schema links:")
print("SELECT c.full_name, c.email FROM customers c ...")
`,id:"code-llm-linking"}),e.jsx(a,{type:"intuition",title:"Schema Linking as Information Retrieval",content:"Schema linking can be viewed as a retrieval problem: given a question, retrieve the most relevant schema elements. This framing allows using embedding-based similarity search over schema elements, which scales better to large databases with hundreds of tables than prompting the LLM with the full schema.",id:"note-retrieval"}),e.jsx(o,{title:"Ambiguous Schema Elements",content:"Real-world databases often have ambiguous column names like 'id', 'name', 'type', or 'status' that appear in multiple tables. Without proper schema linking, the model may reference the wrong table's column. Always include foreign key relationships and sample values in the schema representation to disambiguate.",id:"warning-ambiguity"})]})}const O=Object.freeze(Object.defineProperty({__proto__:null,default:h},Symbol.toStringTag,{value:"Module"}));function g(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"JSON Mode and Structured Outputs"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Structured output generation ensures LLMs produce valid, parseable data formats like JSON. This is essential for integrating LLMs into software systems where downstream code expects well-formed data structures rather than free-form text. Modern APIs offer JSON mode, JSON Schema enforcement, and structured output guarantees."}),e.jsx(n,{title:"Structured Output",definition:"Structured output constrains an LLM's generation to produce text conforming to a formal grammar or schema $G$. Given a prompt $p$, the model generates $y$ such that $y \\in \\mathcal{L}(G)$ where $\\mathcal{L}(G)$ is the language defined by grammar $G$. For JSON Schema validation: $\\text{validate}(y, \\text{schema}) = \\text{true}$.",notation:"G = grammar/schema, \\mathcal{L}(G) = valid strings, y = output",id:"def-structured-output"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"JSON Mode Basics"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"JSON mode guarantees the model output is valid JSON, but does not enforce a specific schema. The model may produce any valid JSON object. For schema enforcement, you need structured outputs with a JSON Schema definition."}),e.jsx(t,{title:"json_mode.py",code:`from openai import OpenAI
import json

client = OpenAI()

# Basic JSON mode - guarantees valid JSON, no schema enforcement
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": "You extract data as JSON."},
        {"role": "user", "content": (
            "Extract the entities from this text: "
            "'Apple Inc. reported $94.8B revenue in Q1 2024, "
            "up 2% from the previous year.'"
        )}
    ]
)

result = json.loads(response.choices[0].message.content)
print(json.dumps(result, indent=2))
# {
#   "company": "Apple Inc.",
#   "revenue": "$94.8B",
#   "period": "Q1 2024",
#   "change": "+2%",
#   "comparison": "previous year"
# }`,id:"code-json-mode"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Schema-Enforced Structured Outputs"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Structured outputs with JSON Schema enforcement guarantee that every response matches your exact schema definition, including required fields, types, enums, and nested structures."}),e.jsx(t,{title:"structured_outputs.py",code:`from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import json

client = OpenAI()

# Define the output schema using Pydantic
class FinancialEntity(BaseModel):
    name: str
    entity_type: str  # "company", "person", "product"
    revenue: Optional[float] = None
    currency: Optional[str] = None

class ExtractionResult(BaseModel):
    entities: list[FinancialEntity]
    time_period: str
    sentiment: str  # "positive", "negative", "neutral"
    confidence: float

# Use structured outputs with schema enforcement
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    response_format=ExtractionResult,
    messages=[
        {"role": "system", "content": "Extract structured data from financial text."},
        {"role": "user", "content": (
            "Tesla reported Q3 revenue of $25.2 billion, beating "
            "analyst expectations. The Model Y was the best-selling vehicle."
        )}
    ]
)

result = response.choices[0].message.parsed
print(f"Entities: {len(result.entities)}")
for entity in result.entities:
    print(f"  {entity.name} ({entity.entity_type}): {entity.revenue} {entity.currency}")
print(f"Sentiment: {result.sentiment}")
print(f"Confidence: {result.confidence}")

# The result is guaranteed to match the ExtractionResult schema
# No need for try/except around JSON parsing`,id:"code-structured"}),e.jsx(i,{title:"JSON Schema for API Responses",problem:"Design a JSON Schema for a product search API response that an LLM must produce.",steps:[{formula:"\\text{Define root object with required fields: products, total\\_count, query}",explanation:"The top-level schema specifies what fields must always be present."},{formula:"\\text{products: array of objects with name, price, category, in\\_stock}",explanation:"Each product has typed fields -- string, number, string, boolean."},{formula:"\\text{Add constraints: price > 0, category } \\in \\text{ enum values}",explanation:"JSON Schema supports numeric ranges and enumerated string values."},{formula:"\\text{Schema validation guarantees type safety at generation time}",explanation:"Unlike post-hoc parsing, schema-constrained generation never produces invalid output."}],id:"example-schema"}),e.jsx(t,{title:"json_schema_definition.py",code:`import json

# Explicit JSON Schema definition (alternative to Pydantic)
product_search_schema = {
    "type": "object",
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "price": {"type": "number", "minimum": 0},
                    "category": {
                        "type": "string",
                        "enum": ["electronics", "clothing", "food", "books"]
                    },
                    "in_stock": {"type": "boolean"},
                    "rating": {"type": "number", "minimum": 0, "maximum": 5}
                },
                "required": ["name", "price", "category", "in_stock"],
                "additionalProperties": False
            }
        },
        "total_count": {"type": "integer", "minimum": 0},
        "query": {"type": "string"}
    },
    "required": ["products", "total_count", "query"],
    "additionalProperties": False
}

# Use with OpenAI API
# response = client.chat.completions.create(
#     model="gpt-4o",
#     response_format={
#         "type": "json_schema",
#         "json_schema": {
#             "name": "product_search",
#             "strict": True,
#             "schema": product_search_schema
#         }
#     },
#     messages=[...]
# )

print(json.dumps(product_search_schema, indent=2))`,id:"code-schema"}),e.jsx(a,{type:"tip",title:"Pydantic vs. Raw JSON Schema",content:"Use Pydantic models when working with Python -- they provide type checking, validation, and auto-generate JSON Schema. Use raw JSON Schema when you need cross-language compatibility or when the schema is dynamically constructed. Both approaches produce identical constraints for the LLM.",id:"note-pydantic"}),e.jsx(o,{title:"Structured Output Limitations",content:"Schema enforcement slightly increases latency (5-15%) because the model must respect constraints at each token. Very complex schemas with deep nesting or many enum values can degrade output quality. Keep schemas as flat as possible and limit enum lists to under 50 values for best results.",id:"warning-limitations"})]})}const q=Object.freeze(Object.defineProperty({__proto__:null,default:g},Symbol.toStringTag,{value:"Module"}));function f(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Function Calling and Tool Use"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Function calling enables LLMs to invoke external tools by generating structured function calls with the correct arguments. Rather than producing a final answer directly, the model outputs a JSON object specifying which function to call and with what parameters, enabling integration with APIs, databases, calculators, and other external systems."}),e.jsx(n,{title:"Function Calling",definition:"Function calling is a structured output mode where the model selects a function $f_i$ from a set of available tools $\\mathcal{F} = \\{f_1, \\ldots, f_k\\}$ and generates arguments $\\text{args}_i$ conforming to the function's parameter schema. The system executes $f_i(\\text{args}_i)$, returns the result, and the model incorporates it into its response.",notation:"\\mathcal{F} = tool set, f_i = function, \\text{args}_i = arguments, \\text{schema}(f_i) = parameter JSON Schema",id:"def-function-calling"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Defining Tool Schemas"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Each tool is described by a JSON Schema that specifies its name, description, and parameter types. The model uses these descriptions to decide which tool to call and how to construct valid arguments."}),e.jsx(t,{title:"tool_definitions.py",code:`import json

# Define tools as JSON Schema objects
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g. 'San Francisco, CA'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Query a SQL database with a natural language question",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT query to execute"
                    },
                    "database": {
                        "type": "string",
                        "enum": ["customers", "products", "orders"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to return",
                        "default": 10
                    }
                },
                "required": ["query", "database"]
            }
        }
    },
]

print(json.dumps(tools[0], indent=2))`,id:"code-tools"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Function Calling Flow"}),e.jsx(t,{title:"function_calling_flow.py",code:`from openai import OpenAI
import json

client = OpenAI()

# Tool implementations
def get_weather(location, unit="celsius"):
    # Simulated API call
    return {"temp": 22, "unit": unit, "condition": "sunny", "location": location}

def search_database(query, database, limit=10):
    # Simulated DB query
    return [{"id": 1, "name": "Widget", "price": 29.99}]

def calculate(expression):
    return {"result": eval(expression)}  # use safe eval in production

# Map function names to implementations
tool_map = {
    "get_weather": get_weather,
    "search_database": search_database,
    "calculate": calculate,
}

# Conversation with tool use
messages = [
    {"role": "user", "content":
     "What's the weather in Tokyo, and how much is 15% tip on a $85 dinner?"}
]

# Step 1: Model decides which tools to call
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,  # from previous example
    tool_choice="auto",  # model decides whether to call tools
)

# Step 2: Execute tool calls
message = response.choices[0].message
if message.tool_calls:
    messages.append(message)  # add assistant message with tool calls

    for tool_call in message.tool_calls:
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)
        print(f"Calling: {fn_name}({fn_args})")

        # Execute the function
        result = tool_map[fn_name](**fn_args)

        # Add result back to conversation
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

    # Step 3: Model generates final response using tool results
    final = client.chat.completions.create(
        model="gpt-4o", messages=messages
    )
    print(f"\\nFinal answer: {final.choices[0].message.content}")`,id:"code-flow"}),e.jsx(i,{title:"Parallel Function Calling",problem:"The model can call multiple tools simultaneously when the question requires independent pieces of information.",steps:[{formula:'\\text{User: "Compare weather in NYC and London, convert 100 USD to EUR"}',explanation:"This question requires three independent tool calls."},{formula:'\\text{Model emits: } [\\text{get\\_weather}(\\text{"NYC"}), \\text{get\\_weather}(\\text{"London"}), \\text{convert\\_currency}(\\ldots)]',explanation:"All three tool calls are generated in a single response turn."},{formula:"\\text{System executes all three in parallel}",explanation:"Since the calls are independent, they can run concurrently for lower latency."},{formula:"\\text{Model receives all results and composes final answer}",explanation:"The model synthesizes information from all tool results into a coherent response."}],id:"example-parallel"}),e.jsx(a,{type:"tip",title:"Tool Choice Control",content:"Use tool_choice='auto' to let the model decide when to use tools. Use tool_choice='required' to force a tool call. Use tool_choice={'type':'function','function':{'name':'X'}} to force a specific tool. Setting tool_choice='none' disables tools entirely for that request.",id:"note-tool-choice"}),e.jsx(o,{title:"Security: Validate Tool Arguments",content:"Never blindly execute function arguments from the model. For database queries, validate that generated SQL is read-only. For API calls, check that URLs and parameters are within expected bounds. For file operations, restrict paths to allowed directories. The model can be prompt-injected into generating malicious tool calls.",id:"warning-security"})]})}const R=Object.freeze(Object.defineProperty({__proto__:null,default:f},Symbol.toStringTag,{value:"Module"}));function x(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Grammar-Constrained Generation"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Grammar-constrained generation restricts an LLM's output to conform to a formal grammar, such as a context-free grammar (CFG) or regular expression. This guarantees syntactic validity of the output, enabling reliable generation of code, structured data formats, and domain-specific languages."}),e.jsx(n,{title:"Grammar-Constrained Decoding",definition:"Given a context-free grammar $G = (V, \\Sigma, R, S)$ and a language model $P(x_t \\mid x_{<t})$, grammar-constrained decoding produces tokens from the modified distribution $P'(x_t \\mid x_{<t}) \\propto P(x_t \\mid x_{<t}) \\cdot \\mathbb{1}[x_t \\text{ is valid given } G \\text{ and } x_{<t}]$. At each step, only tokens that can lead to a complete valid string in $\\mathcal{L}(G)$ are allowed.",notation:"G = grammar, V = variables, \\Sigma = terminals, R = rules, S = start symbol",id:"def-grammar"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"GBNF: Grammar BNF for llama.cpp"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"GBNF (Grammar BNF) is the grammar format used by llama.cpp for constrained generation. It extends BNF notation with character classes, repetition operators, and alternation to define the set of valid outputs."}),e.jsx(t,{title:"gbnf_grammars.py",code:`# GBNF grammar definitions for constrained generation

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
print(sql_select_grammar[:200] + "...")`,id:"code-gbnf"}),e.jsx(i,{title:"Grammar-Constrained Token Selection",problem:"Show how grammar constraints filter tokens during generation of a JSON boolean field.",steps:[{formula:'\\text{Generated so far: } \\{\\text{"active": }',explanation:"The model has produced the key and is about to generate the value."},{formula:'\\text{Grammar state: expecting } value \\to \\text{"true"} \\mid \\text{"false"} \\mid \\text{number} \\mid \\ldots',explanation:"The grammar parser tracks the current valid continuations."},{formula:'\\text{Token mask: allow } [\\text{"true", "false", "null", "\\"", digits, "[", "\\{"}]',explanation:"Only tokens that begin a valid JSON value are permitted."},{formula:'\\text{If schema says boolean: mask reduces to } [\\text{"true", "false"}]',explanation:"With additional schema constraints, the allowed set narrows further."}],id:"example-filtering"}),e.jsx(t,{title:"llama_cpp_grammar.py",code:`# Using grammar-constrained generation with llama-cpp-python
from llama_cpp import Llama, LlamaGrammar

# Load model
llm = Llama(model_path="./models/llama-3-8b.gguf", n_ctx=2048)

# Define a grammar for structured output
grammar_text = r"""
root   ::= "{" ws
           ""sentiment":" ws sentiment "," ws
           ""score":" ws score "," ws
           ""keywords":" ws "[" ws (string ("," ws string)*)? ws "]" ws
           "}"
sentiment ::= ""positive"" | ""negative"" | ""neutral""
score     ::= "0." [0-9] [0-9]
string    ::= """ [a-zA-Z ]+ """
ws        ::= [ 	
]*
"""

grammar = LlamaGrammar.from_string(grammar_text)

# Generate with grammar constraint
output = llm(
    "Analyze the sentiment of this review: 'The product works great "
    "but shipping was slow.' Respond as JSON:
",
    grammar=grammar,
    max_tokens=200,
    temperature=0.7,
)

print(output["choices"][0]["text"])
# {"sentiment": "positive", "score": 0.72, "keywords": ["great", "slow shipping"]}

# The output is GUARANTEED to match the grammar
# No parsing errors possible`,id:"code-llama-grammar"}),e.jsx(r,{title:"Completeness of Grammar-Constrained Decoding",statement:"For any context-free grammar G and any string s in the language of G, grammar-constrained decoding with a sufficiently expressive language model can generate s, provided that for each valid token at each position, the model assigns non-zero probability.",proof:"At each step, the constraint only masks tokens that would lead to strings outside the language. Since s is in the language, each token of s is always in the allowed set. With non-zero probability on all valid tokens, greedy or sampling-based decoding can produce s.",id:"theorem-completeness"}),e.jsx(o,{title:"Grammar Complexity and Performance",content:"Complex grammars with many production rules can slow down generation because the parser must check validity at each token. Ambiguous grammars (where the same string can be derived multiple ways) can cause the parser to track multiple states simultaneously. Keep grammars as simple as possible and prefer deterministic rules.",id:"warning-complexity"}),e.jsx(a,{type:"note",title:"Grammar Libraries",content:"llama.cpp pioneered GBNF grammars for local models. The Outlines library provides Python-native grammar support for Hugging Face models. Guidance from Microsoft uses a hybrid template/grammar approach. vLLM integrates with Outlines for production serving. All achieve the same goal: guaranteed structural validity of LLM outputs.",id:"note-libraries"})]})}const G=Object.freeze(Object.defineProperty({__proto__:null,default:x},Symbol.toStringTag,{value:"Module"}));function b(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Constrained Decoding Algorithms"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Constrained decoding modifies the token selection process during LLM generation to enforce structural, lexical, or semantic constraints. Rather than relying on post-hoc validation, constraints are applied at each decoding step, guaranteeing that the final output satisfies all requirements."}),e.jsx(n,{title:"Constrained Decoding",definition:"Constrained decoding modifies the standard autoregressive generation process by applying a token mask $M_t \\in \\{0, 1\\}^{|V|}$ at each step $t$. The effective distribution becomes $P'(x_t \\mid x_{<t}) = \\frac{P(x_t \\mid x_{<t}) \\cdot M_t(x_t)}{\\sum_{x' \\in V} P(x' \\mid x_{<t}) \\cdot M_t(x')}$ where $M_t$ is determined by the constraint specification and previously generated tokens.",notation:"V = vocabulary, M_t = mask at step t, P = base model, P' = constrained distribution",id:"def-constrained"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Token Masking Approach"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The simplest constrained decoding technique masks invalid tokens by setting their logits to negative infinity before the softmax. This effectively removes them from consideration without changing the relative probabilities of valid tokens."}),e.jsx(t,{title:"token_masking.py",code:`import torch
import torch.nn.functional as F

# Simulate constrained decoding with token masking
vocab_size = 50000
logits = torch.randn(vocab_size)  # raw model logits

# Example: force output to be a valid integer
# Only allow digit tokens and end-of-sequence
digit_token_ids = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57]  # '0'-'9'
eos_token_id = 2

# Create mask: 1 for valid tokens, 0 for invalid
mask = torch.zeros(vocab_size)
mask[digit_token_ids] = 1
mask[eos_token_id] = 1

# Apply mask by setting invalid logits to -inf
masked_logits = logits.clone()
masked_logits[mask == 0] = float('-inf')

# Compare distributions
probs_unconstrained = F.softmax(logits, dim=0)
probs_constrained = F.softmax(masked_logits, dim=0)

print(f"Unconstrained: top token prob = {probs_unconstrained.max():.4f}")
print(f"Constrained:   top token prob = {probs_constrained.max():.4f}")
print(f"Constrained:   valid tokens get all probability mass")
print(f"Sum of constrained probs: {probs_constrained.sum():.6f}")  # 1.0

# The relative ranking among valid tokens is preserved
valid_probs = probs_constrained[digit_token_ids + [eos_token_id]]
print(f"Valid token probs: {valid_probs.numpy().round(4)}")`,id:"code-masking"}),e.jsx(i,{title:"Finite State Machine for Constrained Decoding",problem:"Design an FSM that constrains output to valid email addresses.",steps:[{formula:"\\text{State 0: local part} \\to [a-zA-Z0-9.]+",explanation:"Accept alphanumeric characters and dots for the local part of the email."},{formula:"\\text{State 0} \\xrightarrow{@} \\text{State 1: domain}",explanation:"Transition to domain state when @ is generated."},{formula:"\\text{State 1: domain} \\to [a-zA-Z0-9-]+ \\xrightarrow{.} \\text{State 2: TLD}",explanation:"Accept domain characters, transition on dot to TLD state."},{formula:"\\text{State 2: TLD} \\to [a-zA-Z]\\{2,\\} \\to \\text{Accept}",explanation:"Accept 2+ alpha characters for TLD, then transition to accept state."}],id:"example-fsm"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Outlines: Token-Level Constrained Generation"}),e.jsx(t,{title:"outlines_constrained.py",code:`import outlines

# Load a model through Outlines
model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")

# Regex-constrained generation: valid US phone number
phone_generator = outlines.generate.regex(
    model,
    r"\\(\\d{3}\\) \\d{3}-\\d{4}"
)
phone = phone_generator("Generate a phone number: ")
print(f"Phone: {phone}")  # e.g., (555) 123-4567

# JSON Schema-constrained generation
from pydantic import BaseModel
from typing import List

class MovieReview(BaseModel):
    title: str
    year: int
    rating: float
    genres: List[str]
    recommend: bool

review_generator = outlines.generate.json(model, MovieReview)
review = review_generator(
    "Write a review for a sci-fi movie as JSON:\\n"
)
print(f"Title: {review.title}")
print(f"Rating: {review.rating}")
print(f"Recommend: {review.recommend}")

# Choice-constrained generation: pick from allowed values
sentiment = outlines.generate.choice(
    model,
    ["positive", "negative", "neutral"]
)
result = sentiment("The movie was okay but not great. Sentiment: ")
print(f"Sentiment: {result}")  # "neutral"`,id:"code-outlines"}),e.jsx(t,{title:"constrained_beam_search.py",code:`import torch

def constrained_beam_search(model, tokenizer, prompt, constraint_fn,
                            beam_width=4, max_length=50):
    """Beam search with constraint function.

    constraint_fn(generated_ids) -> list of valid next token ids
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Initialize beams: (score, token_ids)
    beams = [(0.0, input_ids[0].tolist())]

    for step in range(max_length):
        candidates = []
        for score, ids in beams:
            # Get model logits for this beam
            with torch.no_grad():
                outputs = model(torch.tensor([ids]))
                logits = outputs.logits[0, -1, :]

            # Apply constraints
            valid_tokens = constraint_fn(ids[len(input_ids[0]):])
            mask = torch.full_like(logits, float('-inf'))
            mask[valid_tokens] = 0
            constrained_logits = logits + mask

            # Get top-k valid continuations
            log_probs = torch.log_softmax(constrained_logits, dim=0)
            top_k = torch.topk(log_probs, beam_width)

            for log_p, token_id in zip(top_k.values, top_k.indices):
                new_score = score + log_p.item()
                new_ids = ids + [token_id.item()]
                candidates.append((new_score, new_ids))

        # Keep top beams
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

        # Check for completion
        if all(b[1][-1] == tokenizer.eos_token_id for b in beams):
            break

    return tokenizer.decode(beams[0][1])

# Example constraint: output must be valid Python variable name
# def python_var_constraint(generated_ids):
#     if len(generated_ids) == 0:
#         return letter_tokens + underscore_token
#     return letter_tokens + digit_tokens + underscore_token + [eos]`,id:"code-beam"}),e.jsx(a,{type:"intuition",title:"Why Constrained Decoding Preserves Quality",content:"Constrained decoding does not change the model's learned distribution -- it only renormalizes it over valid tokens. If the model already assigns high probability to valid outputs (as well-trained models do), the constraint primarily eliminates low-probability invalid tokens. The quality impact is minimal because the model was already 'trying' to produce valid output; the constraint just ensures it succeeds.",id:"note-quality"})]})}const P=Object.freeze(Object.defineProperty({__proto__:null,default:b},Symbol.toStringTag,{value:"Module"}));function y(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Knowledge Graph-Augmented LLMs"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Knowledge graphs (KGs) store factual knowledge as structured triples, while LLMs excel at language understanding and reasoning. KG-augmented LLMs combine these strengths: the KG provides a grounded, verifiable knowledge base, and the LLM provides flexible natural language reasoning over that knowledge."}),e.jsx(n,{title:"Knowledge Graph",definition:"A knowledge graph $\\mathcal{G} = (\\mathcal{E}, \\mathcal{R}, \\mathcal{T})$ consists of entities $\\mathcal{E}$, relations $\\mathcal{R}$, and triples $\\mathcal{T} \\subseteq \\mathcal{E} \\times \\mathcal{R} \\times \\mathcal{E}$. Each triple $(h, r, t)$ represents a fact: head entity $h$ is related to tail entity $t$ by relation $r$. For example, (Einstein, bornIn, Ulm).",notation:"\\mathcal{E} = entities, \\mathcal{R} = relations, \\mathcal{T} = triples, (h, r, t) = head-relation-tail",id:"def-kg"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Retrieval-Augmented KG Integration"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The most common pattern retrieves relevant subgraph from the KG based on the query, serializes the triples into text, and includes them in the LLM prompt. This grounds the model's response in factual knowledge."}),e.jsx(t,{title:"kg_augmented_qa.py",code:`import json
from collections import defaultdict

# Simple knowledge graph representation
class KnowledgeGraph:
    def __init__(self):
        self.triples = []
        self.entity_index = defaultdict(list)

    def add_triple(self, head, relation, tail):
        triple = (head, relation, tail)
        self.triples.append(triple)
        self.entity_index[head.lower()].append(triple)
        self.entity_index[tail.lower()].append(triple)

    def query_entity(self, entity, max_hops=2):
        """Retrieve subgraph around an entity up to max_hops."""
        visited = set()
        result = []
        frontier = {entity.lower()}

        for hop in range(max_hops):
            next_frontier = set()
            for e in frontier:
                if e in visited:
                    continue
                visited.add(e)
                for h, r, t in self.entity_index.get(e, []):
                    result.append((h, r, t))
                    next_frontier.add(h.lower())
                    next_frontier.add(t.lower())
            frontier = next_frontier - visited

        return result

# Build a small KG
kg = KnowledgeGraph()
kg.add_triple("Albert Einstein", "bornIn", "Ulm")
kg.add_triple("Albert Einstein", "field", "Theoretical Physics")
kg.add_triple("Albert Einstein", "award", "Nobel Prize in Physics")
kg.add_triple("Albert Einstein", "knownFor", "Theory of Relativity")
kg.add_triple("Ulm", "country", "Germany")
kg.add_triple("Ulm", "locatedIn", "Baden-Württemberg")
kg.add_triple("Nobel Prize in Physics", "year", "1921")

# Retrieve relevant triples
question = "Where was Einstein born and in which country?"
subgraph = kg.query_entity("Albert Einstein", max_hops=2)

# Serialize triples for the LLM prompt
def triples_to_text(triples):
    return "\\n".join(f"- {h} --[{r}]--> {t}" for h, r, t in triples)

prompt = f"""Use the following knowledge to answer the question.

Knowledge:
{triples_to_text(subgraph)}

Question: {question}
Answer:"""

print(prompt)
# The LLM can now answer: "Einstein was born in Ulm, Germany"
# grounded in the KG triples`,id:"code-kg-qa"}),e.jsx(i,{title:"KG-Augmented Reasoning Pipeline",problem:"Answer 'Did Einstein win a Nobel Prize before or after publishing general relativity?' using a KG.",steps:[{formula:'\\text{Entity linking: "Einstein" } \\to \\text{Albert Einstein}',explanation:"Map the question mention to the canonical KG entity."},{formula:"\\text{Retrieve: (Einstein, award, Nobel Prize), (Nobel Prize, year, 1921)}",explanation:"Fetch triples about Einstein and Nobel Prize."},{formula:"\\text{Retrieve: (Einstein, published, General Relativity), (General Relativity, year, 1915)}",explanation:"Fetch triples about general relativity publication."},{formula:'\\text{LLM reasons: 1915 < 1921 } \\to \\text{"after publishing general relativity"}',explanation:"The LLM performs temporal reasoning over the retrieved facts."}],id:"example-reasoning"}),e.jsx(t,{title:"kg_with_embeddings.py",code:`import numpy as np

# KG entity/relation embeddings for semantic retrieval
class KGEmbeddings:
    def __init__(self, dim=64):
        self.dim = dim
        self.entity_embeddings = {}
        self.relation_embeddings = {}

    def add_entity(self, name, embedding=None):
        if embedding is None:
            embedding = np.random.randn(self.dim)
            embedding = embedding / np.linalg.norm(embedding)
        self.entity_embeddings[name] = embedding

    def add_relation(self, name, embedding=None):
        if embedding is None:
            embedding = np.random.randn(self.dim)
            embedding = embedding / np.linalg.norm(embedding)
        self.relation_embeddings[name] = embedding

    def find_similar_entities(self, query_embedding, top_k=5):
        scores = {}
        for name, emb in self.entity_embeddings.items():
            scores[name] = np.dot(query_embedding, emb)
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]

# Build KG embeddings (in practice, trained with TransE/RotatE)
kge = KGEmbeddings(dim=64)
entities = ["Albert Einstein", "Isaac Newton", "Marie Curie",
            "Physics", "Chemistry", "Nobel Prize"]
for e in entities:
    kge.add_entity(e)

# Semantic search: find entities related to a query
query = kge.entity_embeddings["Albert Einstein"]
similar = kge.find_similar_entities(query, top_k=3)
print("Entities similar to Einstein:")
for name, score in similar:
    print(f"  {name}: {score:.3f}")

# TransE scoring: h + r ≈ t for valid triples
# score(h, r, t) = -||h + r - t||
h = kge.entity_embeddings["Albert Einstein"]
r = np.random.randn(64)  # "field" relation embedding
t = kge.entity_embeddings["Physics"]
score = -np.linalg.norm(h + r - t)
print(f"\\nTransE score (Einstein, field, Physics): {score:.3f}")`,id:"code-embeddings"}),e.jsx(a,{type:"note",title:"Major Knowledge Graphs",content:"Wikidata contains 100M+ entities and 1.5B+ triples covering general knowledge. Freebase (now deprecated, absorbed into Wikidata) powered Google's Knowledge Graph. Domain-specific KGs include UMLS (medical), Gene Ontology (biology), and ConceptNet (commonsense). These structured resources complement LLMs' parametric knowledge with explicit, updatable facts.",id:"note-major-kgs"}),e.jsx(o,{title:"KG Incompleteness",content:"Knowledge graphs are inherently incomplete -- they only contain explicitly stated facts. The open-world assumption means that a missing triple does not imply the fact is false. When augmenting LLMs with KGs, the model should be able to reason under uncertainty and not treat KG absence as negative evidence.",id:"warning-incompleteness"}),e.jsx(a,{type:"intuition",title:"KGs vs. RAG with Text",content:"KG-augmented generation and text-based RAG serve different needs. KGs excel at multi-hop factual reasoning (following chains of relations), provide precise, structured facts, and support explicit provenance. Text-based RAG provides richer context, handles nuance and qualification better, and requires no upfront structuring. The best systems often combine both: KG for factual grounding and text for context.",id:"note-vs-rag"})]})}const I=Object.freeze(Object.defineProperty({__proto__:null,default:y},Symbol.toStringTag,{value:"Module"}));function _(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Entity Linking with LLMs"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:'Entity linking (EL) maps mentions in text to their corresponding entries in a knowledge base. It resolves ambiguity -- "Apple" could refer to the company, the fruit, or the record label -- by considering the surrounding context. LLMs have transformed this task by leveraging their world knowledge for disambiguation.'}),e.jsx(n,{title:"Entity Linking",definition:"Entity linking takes a text $d$ containing entity mentions $M = \\{m_1, \\ldots, m_k\\}$ and maps each mention $m_i$ to an entity $e_i$ in a knowledge base $\\mathcal{KB}$, or to NIL if the entity is not in the KB. Formally: $\\text{EL}(m_i, d) \\to e_i \\in \\mathcal{KB} \\cup \\{\\text{NIL}\\}$. The task combines mention detection, candidate generation, and entity disambiguation.",notation:"d = document, m_i = mention, e_i = entity, \\mathcal{KB} = knowledge base",id:"def-entity-linking"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Entity Linking Pipeline"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Traditional entity linking follows a three-stage pipeline: mention detection (finding entity spans in text), candidate generation (retrieving possible KB entries), and disambiguation (selecting the correct entity based on context)."}),e.jsx(t,{title:"entity_linking_pipeline.py",code:`import re
from difflib import get_close_matches

# Simple knowledge base
knowledge_base = {
    "Q312": {"name": "Apple Inc.", "type": "company",
             "aliases": ["Apple", "Apple Computer"],
             "description": "American technology company"},
    "Q89": {"name": "apple", "type": "fruit",
            "aliases": ["apple fruit", "malus"],
            "description": "Edible fruit from apple trees"},
    "Q484523": {"name": "Apple Records", "type": "company",
                "aliases": ["Apple"],
                "description": "Record label founded by the Beatles"},
    "Q937": {"name": "Albert Einstein", "type": "person",
             "aliases": ["Einstein", "A. Einstein"],
             "description": "Theoretical physicist"},
    "Q7186": {"name": "Marie Curie", "type": "person",
              "aliases": ["Curie", "Maria Sklodowska"],
              "description": "Physicist and chemist, Nobel laureate"},
}

# Stage 1: Mention Detection (simplified with NER-like rules)
def detect_mentions(text):
    """Find capitalized phrases as potential entity mentions."""
    pattern = r'\\b[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*\\b'
    return [(m.group(), m.start(), m.end())
            for m in re.finditer(pattern, text)]

# Stage 2: Candidate Generation
def generate_candidates(mention, kb, max_candidates=5):
    """Find KB entries matching the mention."""
    candidates = []
    mention_lower = mention.lower()
    for qid, entity in kb.items():
        all_names = [entity["name"].lower()] + [a.lower() for a in entity["aliases"]]
        for name in all_names:
            if mention_lower in name or name in mention_lower:
                candidates.append((qid, entity))
                break
    return candidates

# Stage 3: Disambiguation (context-based scoring)
def disambiguate(mention, candidates, context, kb):
    """Score candidates based on context overlap."""
    context_words = set(context.lower().split())
    scores = []
    for qid, entity in candidates:
        desc_words = set(entity["description"].lower().split())
        overlap = len(context_words & desc_words)
        scores.append((qid, entity["name"], overlap))
    return sorted(scores, key=lambda x: -x[2])

# Full pipeline
text = "Apple announced new products at their Cupertino headquarters. Einstein would have loved the physics simulations."
mentions = detect_mentions(text)
print(f"Text: {text}\\n")

for mention_text, start, end in mentions:
    candidates = generate_candidates(mention_text, knowledge_base)
    if candidates:
        ranked = disambiguate(mention_text, candidates, text, knowledge_base)
        best = ranked[0] if ranked else None
        print(f"Mention: '{mention_text}' -> {best[1]} ({best[0]})")
    else:
        print(f"Mention: '{mention_text}' -> NIL (not in KB)")`,id:"code-pipeline"}),e.jsx(i,{title:"LLM-Based Entity Disambiguation",problem:"Disambiguate 'Mercury' in the sentence: 'Mercury's orbit has the highest eccentricity of any planet.'",steps:[{formula:"\\text{Candidates: Mercury (planet), Mercury (element), Mercury (mythology)}",explanation:'KB lookup returns multiple entities matching "Mercury".'},{formula:'\\text{Context clues: "orbit", "eccentricity", "planet"}',explanation:"Surrounding words strongly suggest an astronomical context."},{formula:'\\text{LLM prompt: "Which Mercury? planet/element/mythology. Context: orbit, planet"}',explanation:"Frame disambiguation as a classification task for the LLM."},{formula:"\\text{Output: Mercury (planet) with high confidence}",explanation:"The LLM leverages both context and world knowledge to select the correct entity."}],id:"example-disambiguation"}),e.jsx(t,{title:"llm_entity_linking.py",code:`import json

def llm_entity_linking_prompt(text, mentions, candidates_per_mention):
    """Build a prompt for LLM-based entity linking."""
    prompt = f"""Link each entity mention to the correct knowledge base entry.

Text: "{text}"

Mentions to link:
"""
    for mention, candidates in zip(mentions, candidates_per_mention):
        prompt += f"\\nMention: '{mention}'\\nCandidates:\\n"
        for qid, name, desc in candidates:
            prompt += f"  - {qid}: {name} ({desc})\\n"

    prompt += """
Return a JSON array of objects with "mention", "qid", and "confidence" fields.
If no candidate matches, use "NIL" for qid.
JSON:"""
    return prompt

# Example usage
text = "Apple CEO Tim Cook presented at WWDC in San Jose"
mentions = ["Apple", "Tim Cook", "WWDC", "San Jose"]
candidates = [
    [("Q312", "Apple Inc.", "tech company"),
     ("Q89", "apple", "fruit")],
    [("Q265", "Tim Cook", "CEO of Apple Inc.")],
    [("Q1630", "WWDC", "Apple developer conference")],
    [("Q16553", "San Jose", "city in California"),
     ("Q79984", "San Jose", "city in Costa Rica")],
]

prompt = llm_entity_linking_prompt(text, mentions, candidates)
print(prompt)

# Expected LLM output:
expected = [
    {"mention": "Apple", "qid": "Q312", "confidence": 0.98},
    {"mention": "Tim Cook", "qid": "Q265", "confidence": 0.99},
    {"mention": "WWDC", "qid": "Q1630", "confidence": 0.97},
    {"mention": "San Jose", "qid": "Q16553", "confidence": 0.95},
]
print("\\nExpected output:")
print(json.dumps(expected, indent=2))`,id:"code-llm-el"}),e.jsx(a,{type:"note",title:"Zero-Shot Entity Linking",content:"LLMs enable zero-shot entity linking -- resolving entities without task-specific training data. By prompting with entity descriptions from the KB, GPT-4 achieves 85%+ accuracy on standard EL benchmarks (AIDA-CoNLL), approaching the performance of specialized fine-tuned models like BLINK and GENRE that require extensive training.",id:"note-zero-shot"}),e.jsx(o,{title:"NIL Entity Challenge",content:"A significant fraction of entity mentions in real text refer to entities not present in any knowledge base (emerging entities, rare proper nouns). Models must learn to output NIL rather than forcing a match. Fine-tuned models often default to the most popular candidate; LLMs handle NIL detection better through reasoning but can still hallucinate non-existent KB entries.",id:"warning-nil"}),e.jsx(a,{type:"tip",title:"Bi-Encoder for Scalable Candidate Generation",content:"For large KBs with millions of entities, use a bi-encoder architecture: encode the mention with context and each entity independently, then use approximate nearest neighbor search (FAISS) for fast candidate retrieval. The LLM is only used for the final disambiguation step over the top-k candidates, keeping the system efficient.",id:"note-scalable"})]})}const $=Object.freeze(Object.defineProperty({__proto__:null,default:_},Symbol.toStringTag,{value:"Module"}));function w(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Relation Extraction with LLMs"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Relation extraction (RE) identifies semantic relationships between entities mentioned in text, producing structured triples that can populate knowledge graphs. LLMs have dramatically simplified this task, enabling zero-shot extraction across domains without predefined relation schemas."}),e.jsx(n,{title:"Relation Extraction",definition:"Given a sentence $s$ containing entity mentions $e_1$ and $e_2$, relation extraction identifies the relation $r \\in \\mathcal{R} \\cup \\{\\text{NoRelation}\\}$ that holds between them, producing a triple $(e_1, r, e_2)$. In open RE, the relation set $\\mathcal{R}$ is not predefined. In closed RE, $\\mathcal{R}$ is a fixed schema.",notation:"s = sentence, e_1, e_2 = entities, r = relation, \\mathcal{R} = relation set",id:"def-relation-extraction"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Closed-Domain Relation Extraction"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"In closed-domain RE, the model must classify entity pairs into a predefined set of relations. This is common in domain-specific applications like biomedical literature mining or financial analysis."}),e.jsx(t,{title:"closed_re.py",code:`import json

# Predefined relation schema
RELATIONS = [
    "founded_by", "headquartered_in", "CEO_of", "acquired_by",
    "subsidiary_of", "partnership_with", "competitor_of",
    "invested_in", "no_relation"
]

def closed_re_prompt(text, entity1, entity2, relations):
    return f"""Extract the relationship between the two entities.

Text: "{text}"
Entity 1: {entity1}
Entity 2: {entity2}

Possible relations: {', '.join(relations)}

Return JSON with "entity1", "relation", "entity2", "confidence".
JSON:"""

# Examples
examples = [
    ("Microsoft acquired Activision Blizzard for $69 billion in 2023.",
     "Microsoft", "Activision Blizzard"),
    ("Sundar Pichai serves as the CEO of Alphabet Inc.",
     "Sundar Pichai", "Alphabet Inc."),
    ("Amazon and Google are major competitors in cloud computing.",
     "Amazon", "Google"),
]

for text, e1, e2 in examples:
    prompt = closed_re_prompt(text, e1, e2, RELATIONS)
    print(f"Text: {text}")
    print(f"  {e1} --[?]--> {e2}\\n")

# Expected outputs:
# Microsoft --[acquired_by]--> Activision Blizzard (or reverse)
# Sundar Pichai --[CEO_of]--> Alphabet Inc.
# Amazon --[competitor_of]--> Google`,id:"code-closed-re"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Open Information Extraction"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Open information extraction (OpenIE) does not require a predefined relation schema. The model extracts arbitrary relation phrases directly from the text, enabling knowledge discovery from unstructured sources."}),e.jsx(t,{title:"open_ie_with_llm.py",code:`import json

def open_ie_prompt(text):
    return f"""Extract all factual relationships from the text as triples.

Text: "{text}"

Return a JSON array of triples. Each triple has:
- "subject": the entity performing or being described
- "relation": the relationship phrase
- "object": the entity being related to
- "confidence": float between 0 and 1

Extract ALL relationships, including implicit ones.
JSON:"""

text = """Marie Curie was born in Warsaw, Poland in 1867. She moved to
Paris to study at the Sorbonne. In 1903, she became the first woman
to win a Nobel Prize, sharing it with her husband Pierre Curie and
Henri Becquerel for their work on radioactivity."""

prompt = open_ie_prompt(text)
print(prompt)

# Expected LLM output:
expected_triples = [
    {"subject": "Marie Curie", "relation": "born_in", "object": "Warsaw, Poland", "confidence": 0.99},
    {"subject": "Marie Curie", "relation": "birth_year", "object": "1867", "confidence": 0.99},
    {"subject": "Marie Curie", "relation": "moved_to", "object": "Paris", "confidence": 0.98},
    {"subject": "Marie Curie", "relation": "studied_at", "object": "Sorbonne", "confidence": 0.97},
    {"subject": "Marie Curie", "relation": "won", "object": "Nobel Prize", "confidence": 0.99},
    {"subject": "Marie Curie", "relation": "spouse", "object": "Pierre Curie", "confidence": 0.95},
    {"subject": "Marie Curie", "relation": "first_woman_to", "object": "win Nobel Prize", "confidence": 0.98},
    {"subject": "Nobel Prize", "relation": "shared_with", "object": "Henri Becquerel", "confidence": 0.96},
    {"subject": "Nobel Prize", "relation": "field", "object": "radioactivity", "confidence": 0.94},
]

print("\\nExtracted triples:")
for t in expected_triples:
    print(f"  ({t['subject']}, {t['relation']}, {t['object']}) [{t['confidence']:.2f}]")`,id:"code-open-ie"}),e.jsx(t,{title:"re_with_constraints.py",code:`# Constrained relation extraction with type checking
ENTITY_TYPES = {
    "PERSON": ["founded_by", "CEO_of", "born_in", "spouse_of"],
    "ORGANIZATION": ["headquartered_in", "subsidiary_of", "acquired_by"],
    "LOCATION": [],  # locations are typically objects, not subjects
}

RELATION_CONSTRAINTS = {
    "founded_by": {"subject": "ORGANIZATION", "object": "PERSON"},
    "CEO_of": {"subject": "PERSON", "object": "ORGANIZATION"},
    "born_in": {"subject": "PERSON", "object": "LOCATION"},
    "headquartered_in": {"subject": "ORGANIZATION", "object": "LOCATION"},
    "acquired_by": {"subject": "ORGANIZATION", "object": "ORGANIZATION"},
}

def validate_triple(subject, relation, obj, entity_types):
    """Check if a triple satisfies type constraints."""
    if relation not in RELATION_CONSTRAINTS:
        return True  # no constraints defined
    constraints = RELATION_CONSTRAINTS[relation]
    subj_type = entity_types.get(subject)
    obj_type = entity_types.get(obj)
    valid = True
    if subj_type and subj_type != constraints["subject"]:
        valid = False
    if obj_type and obj_type != constraints["object"]:
        valid = False
    return valid

# Validate extracted triples
entity_types = {
    "Apple Inc.": "ORGANIZATION",
    "Tim Cook": "PERSON",
    "Cupertino": "LOCATION",
}

triples = [
    ("Tim Cook", "CEO_of", "Apple Inc."),       # valid
    ("Apple Inc.", "headquartered_in", "Cupertino"),  # valid
    ("Cupertino", "CEO_of", "Tim Cook"),         # invalid
]

for s, r, o in triples:
    valid = validate_triple(s, r, o, entity_types)
    status = "VALID" if valid else "INVALID"
    print(f"  [{status}] ({s}, {r}, {o})")`,id:"code-constraints"}),e.jsx(o,{title:"Relation Hallucination",content:"LLMs can hallucinate plausible-sounding but incorrect relations, especially for entities they have strong prior knowledge about. For example, given text about a lesser-known 'John Smith', the model might infer relations from a famous John Smith. Always require textual evidence for extracted relations and implement confidence thresholds.",id:"warning-hallucination"})]})}const z=Object.freeze(Object.defineProperty({__proto__:null,default:w},Symbol.toStringTag,{value:"Module"}));function v(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Graph-to-Text and Text-to-Graph Generation"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Graph-to-text (G2T) converts structured graph data into fluent natural language, while text-to-graph (T2G) extracts structured graph representations from text. Together, these tasks form a bidirectional bridge between structured and unstructured knowledge, enabling knowledge graph construction, data-to-text generation, and graph-grounded dialogue."}),e.jsx(n,{title:"Graph-to-Text Generation",definition:"Graph-to-text generation maps a subgraph $g = \\{(h_1, r_1, t_1), \\ldots, (h_k, r_k, t_k)\\}$ to a natural language text $y$ that faithfully verbalizes all triples. The generation must be faithful ($y$ entails all triples), fluent (grammatically correct), and concise (no unnecessary repetition).",notation:"g = subgraph, (h, r, t) = triple, y = generated text",id:"def-g2t"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Graph-to-Text with LLMs"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"LLMs can verbalize graph triples into fluent text by serializing the triples and prompting the model to generate a natural language description. The challenge is ensuring all triples are covered without hallucinating additional facts."}),e.jsx(t,{title:"graph_to_text.py",code:`import json

# Graph-to-text: verbalize triples as natural language
def graph_to_text_prompt(triples, style="paragraph"):
    triple_str = "\\n".join(
        f"  ({h}, {r}, {t})" for h, r, t in triples
    )
    return f"""Convert the following knowledge graph triples into a natural
language {style}. Include ALL facts from the triples. Do not add
information not present in the triples.

Triples:
{triple_str}

Text:"""

# Example: biographical triples
bio_triples = [
    ("Ada Lovelace", "born_in", "London"),
    ("Ada Lovelace", "birth_year", "1815"),
    ("Ada Lovelace", "occupation", "mathematician"),
    ("Ada Lovelace", "known_for", "first computer program"),
    ("Ada Lovelace", "worked_with", "Charles Babbage"),
    ("Charles Babbage", "invented", "Analytical Engine"),
    ("Ada Lovelace", "parent", "Lord Byron"),
]

prompt = graph_to_text_prompt(bio_triples)
print(prompt)

# Expected output:
expected = (
    "Ada Lovelace was a mathematician born in London in 1815. "
    "She is known for writing the first computer program, developed "
    "during her work with Charles Babbage, the inventor of the "
    "Analytical Engine. She was the daughter of Lord Byron."
)
print(f"\\nExpected output:\\n{expected}")

# Graph-to-text for different styles
for style in ["paragraph", "bullet points", "formal biography"]:
    p = graph_to_text_prompt(bio_triples[:3], style=style)
    print(f"\\n--- Style: {style} ---")
    print(p[:100] + "...")`,id:"code-g2t"}),e.jsx(i,{title:"Faithfulness Evaluation",problem:"Verify that a generated text faithfully represents the source triples without hallucination.",steps:[{formula:"\\text{Triples: (Paris, capital\\_of, France), (Paris, population, 2.1M)}",explanation:"Source graph contains exactly two facts."},{formula:'\\text{Generated: "Paris, the capital of France, has 2.1M residents and is known for the Eiffel Tower."}',explanation:"The text mentions the Eiffel Tower, which is not in the triples."},{formula:'\\text{Faithfulness check: "Eiffel Tower" } \\notin \\text{ triples } \\to \\text{ hallucination}',explanation:"Any fact in the text not traceable to a source triple is a hallucination."},{formula:"\\text{Coverage check: both triples mentioned } \\to \\text{ complete}",explanation:"All source triples must appear in the generated text for full coverage."}],id:"example-faithfulness"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Text-to-Graph Extraction"}),e.jsx(t,{title:"text_to_graph.py",code:`import json
import re

def text_to_graph_prompt(text, entity_types=None, relation_types=None):
    """Build a prompt for text-to-graph extraction."""
    constraints = ""
    if entity_types:
        constraints += f"Entity types: {', '.join(entity_types)}\\n"
    if relation_types:
        constraints += f"Relation types: {', '.join(relation_types)}\\n"

    return f"""Extract a knowledge graph from the text.

Text: "{text}"

{constraints}
Return JSON with:
- "entities": list of {{"name": str, "type": str}}
- "triples": list of {{"subject": str, "relation": str, "object": str}}

Ensure entity names in triples match exactly those in the entities list.
JSON:"""

text = """The University of Oxford, founded in 1096, is located in Oxford,
England. It is one of the oldest universities in the world. Stephen
Hawking studied at Oxford before moving to Cambridge for his PhD.
He later became Lucasian Professor of Mathematics at Cambridge."""

prompt = text_to_graph_prompt(
    text,
    entity_types=["PERSON", "ORGANIZATION", "LOCATION", "DATE"],
    relation_types=["located_in", "founded_in", "studied_at",
                    "role_at", "moved_to"]
)

# Expected output
expected = {
    "entities": [
        {"name": "University of Oxford", "type": "ORGANIZATION"},
        {"name": "Oxford", "type": "LOCATION"},
        {"name": "England", "type": "LOCATION"},
        {"name": "Stephen Hawking", "type": "PERSON"},
        {"name": "Cambridge", "type": "ORGANIZATION"},
    ],
    "triples": [
        {"subject": "University of Oxford", "relation": "founded_in", "object": "1096"},
        {"subject": "University of Oxford", "relation": "located_in", "object": "Oxford"},
        {"subject": "Oxford", "relation": "located_in", "object": "England"},
        {"subject": "Stephen Hawking", "relation": "studied_at", "object": "University of Oxford"},
        {"subject": "Stephen Hawking", "relation": "moved_to", "object": "Cambridge"},
        {"subject": "Stephen Hawking", "relation": "role_at", "object": "Cambridge"},
    ]
}
print(json.dumps(expected, indent=2))`,id:"code-t2g"}),e.jsx(t,{title:"graph_roundtrip.py",code:`# Round-trip evaluation: Text -> Graph -> Text
# Tests both extraction and generation quality

# Round-trip: verify text -> graph -> text preserves facts
original = "Marie Curie won the Nobel Prize in Physics in 1903 and Chemistry in 1911."
graph = [
    ("Marie Curie", "won", "Nobel Prize in Physics"),
    ("Nobel Prize in Physics", "year", "1903"),
    ("Marie Curie", "won", "Nobel Prize in Chemistry"),
    ("Nobel Prize in Chemistry", "year", "1911"),
]
regenerated = "Marie Curie was awarded the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911."

metrics = roundtrip_eval(original, graph, regenerated)
print(f"Round-trip metrics: {json.dumps(metrics, indent=2)}")

print("Graph adjacency:", {h: [(r,t)] for h,r,t in graph})`,id:"code-roundtrip"}),e.jsx(a,{type:"note",title:"Benchmarks for Graph-Text Tasks",content:"WebNLG is the primary benchmark for graph-to-text, containing 25,000+ (graph, text) pairs from DBpedia. KELM evaluates text-to-graph with Wikidata triples. GenWiki tests both directions. On WebNLG, fine-tuned T5-large achieves BLEU scores of ~65, while GPT-4 zero-shot reaches ~55. The gap is narrower on faithfulness metrics where LLMs excel.",id:"note-benchmarks"})]})}const F=Object.freeze(Object.defineProperty({__proto__:null,default:v},Symbol.toStringTag,{value:"Module"}));export{L as a,A as b,E as c,N as d,C as e,M as f,O as g,q as h,R as i,G as j,P as k,I as l,$ as m,z as n,F as o,j as s};
