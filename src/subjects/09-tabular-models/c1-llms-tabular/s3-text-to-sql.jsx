import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function TextToSQL() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Text-to-SQL with Large Language Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Text-to-SQL converts natural language questions into executable SQL queries. LLMs have
        transformed this task from a specialized semantic parsing problem into a prompting challenge,
        achieving state-of-the-art accuracy on benchmarks like Spider and BIRD by leveraging
        in-context learning with database schema information.
      </p>

      <DefinitionBlock
        title="Text-to-SQL"
        definition="Text-to-SQL is the task of mapping a natural language question $q$ and a database schema $S = \{(t_i, \{c_{i,1}, \ldots, c_{i,k}\})\}$ to a valid SQL query $Q$ such that executing $Q$ on the database returns the answer to $q$. Formally: $f(q, S) \to Q$ where $\text{exec}(Q, D) = \text{answer}(q)$."
        notation="q = natural language question, S = schema, t_i = table name, c_{i,j} = column, Q = SQL query, D = database"
        id="def-text-to-sql"
      />

      <h2 className="text-2xl font-semibold">Schema-Aware Prompting</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The key to accurate Text-to-SQL is providing the model with a clear representation of the
        database schema, including table names, column names, data types, and foreign key relationships.
      </p>

      <PythonCode
        title="text_to_sql_prompt.py"
        code={`# Building a Text-to-SQL prompt with schema information
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
# HAVING COUNT(*) > 5;`}
        id="code-prompt"
      />

      <ExampleBlock
        title="Multi-Step Text-to-SQL with Chain-of-Thought"
        problem="Generate SQL for: 'Find the department with the highest total salary that also has an active project.'"
        steps={[
          { formula: '\\text{Step 1: Identify relevant tables: employees, departments, projects}', explanation: 'Parse the question to determine which tables are needed.' },
          { formula: '\\text{Step 2: "highest total salary" } \\to \\text{SUM(salary) + ORDER BY + LIMIT}', explanation: 'Map the phrase to SQL aggregation and ordering constructs.' },
          { formula: '\\text{Step 3: "active project" } \\to \\text{WHERE end\\_date >= CURRENT\\_DATE}', explanation: 'Interpret the filter condition from natural language.' },
          { formula: '\\text{Step 4: Combine with JOIN on department\\_id}', explanation: 'Link tables through foreign key relationships to form the complete query.' },
        ]}
        id="example-cot"
      />

      <PythonCode
        title="text_to_sql_execution.py"
        code={`import sqlite3

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
conn.close()`}
        id="code-execution"
      />

      <WarningBlock
        title="SQL Injection and Safety"
        content="Never execute LLM-generated SQL without validation. Always restrict to SELECT queries when the intent is read-only, use parameterized queries for user inputs, and run generated SQL in sandboxed environments with limited permissions. Consider using query validation libraries to parse and check the SQL AST before execution."
        id="warning-safety"
      />

      <NoteBlock
        type="note"
        title="Benchmark Performance"
        content="On the Spider benchmark (cross-database Text-to-SQL), GPT-4 with few-shot prompting achieves ~85% execution accuracy, compared to ~72% for fine-tuned T5-3B. The BIRD benchmark adds real-world complexity with dirty data and external knowledge, where GPT-4 scores ~55%. The gap highlights that schema understanding and SQL generation are largely solved, but handling messy real data remains challenging."
        id="note-benchmarks"
      />

      <NoteBlock
        type="tip"
        title="Self-Correction Improves Accuracy"
        content="A powerful technique is to execute the generated SQL, then show the model both the query and the results (or error message), and ask it to verify or fix the query. This 'generate-execute-refine' loop can boost accuracy by 5-10% on complex queries involving multiple joins and subqueries."
        id="note-self-correction"
      />
    </div>
  )
}
