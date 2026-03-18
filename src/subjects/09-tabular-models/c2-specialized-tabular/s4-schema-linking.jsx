import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function SchemaLinking() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Schema Linking for SQL Generation</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Schema linking is the critical step of mapping mentions in a natural language question
        to the corresponding tables, columns, and values in a database schema. It bridges the
        gap between how users describe data and how it is actually structured, and is often
        the bottleneck in Text-to-SQL accuracy.
      </p>

      <DefinitionBlock
        title="Schema Linking"
        definition="Schema linking is the process of identifying correspondences between spans in a natural language question $q$ and elements of a database schema $S$. Formally, it produces a set of alignments $L = \{(s_i, e_j)\}$ where $s_i$ is a span in $q$ and $e_j \in \{tables, columns, values\}$ is a schema element. The alignment can be exact match, partial match, or semantic match."
        notation="q = question, S = schema, L = links, s_i = question span, e_j = schema element"
        id="def-schema-linking"
      />

      <h2 className="text-2xl font-semibold">Types of Schema Links</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Schema linking must handle multiple types of correspondences, from trivial exact matches
        to challenging semantic matches where the question uses completely different terminology
        than the schema.
      </p>

      <ExampleBlock
        title="Schema Linking Categories"
        problem="Given the question 'How many employees earn more than 100k in the engineering team?' and schema tables employees(id, name, salary, dept_id) and departments(id, dept_name), identify all schema links."
        steps={[
          { formula: '\\text{"employees"} \\to \\texttt{employees} \\text{ (exact match)}', explanation: 'The word "employees" directly matches the table name.' },
          { formula: '\\text{"earn"} \\to \\texttt{salary} \\text{ (semantic match)}', explanation: '"Earn" implies salary/compensation -- requires world knowledge.' },
          { formula: '\\text{"100k"} \\to \\texttt{100000} \\text{ (value match)}', explanation: 'The shorthand "100k" must be interpreted as the numeric value 100000.' },
          { formula: '\\text{"engineering team"} \\to \\texttt{departments.dept\\_name} \\text{ (semantic match)}', explanation: '"Engineering team" maps to a value in the dept_name column, requiring a JOIN.' },
        ]}
        id="example-categories"
      />

      <PythonCode
        title="schema_linking.py"
        code={`import re
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

def fuzzy_match(question_tokens, schema, threshold=0.7):
    """Find fuzzy matches using string similarity."""
    links = []
    words = question_tokens.lower().split()
    for table, columns in schema.items():
        for col in columns:
            for word in words:
                ratio = SequenceMatcher(None, word, col).ratio()
                if ratio >= threshold and ratio < 1.0:
                    links.append(("fuzzy", word, f"{table}.{col}", ratio))
    return links

# Example
question = "How many workers earn over 100k in the engineering team?"
print(f"Question: {question}\\n")

for link in exact_match(question, schema):
    print(f"  Exact: '{link[1]}' -> {link[2]}")
for link in semantic_match(question, column_descriptions):
    print(f"  Semantic: '{link[1]}' -> {link[2]}")
for link in fuzzy_match(question, schema):
    print(f"  Fuzzy: '{link[1]}' -> {link[2]} ({link[3]:.2f})")`}
        id="code-linking"
      />

      <h2 className="text-2xl font-semibold">LLM-Based Schema Linking</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Modern approaches use LLMs to perform schema linking as part of the SQL generation
        pipeline. The model identifies relevant tables and columns before generating the query,
        which reduces hallucination and improves accuracy on complex schemas.
      </p>

      <PythonCode
        title="llm_schema_linking.py"
        code={`# LLM-based schema linking with structured output
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

# The schema linking output guides SQL generation:
sql = """SELECT c.full_name, c.email
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE c.plan_type = 'premium'
  AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
GROUP BY c.customer_id, c.full_name, c.email
HAVING SUM(o.amount) > 500;"""
print(f"\\nGenerated SQL:\\n{sql}")`}
        id="code-llm-linking"
      />

      <NoteBlock
        type="intuition"
        title="Schema Linking as Information Retrieval"
        content="Schema linking can be viewed as a retrieval problem: given a question, retrieve the most relevant schema elements. This framing allows using embedding-based similarity search over schema elements, which scales better to large databases with hundreds of tables than prompting the LLM with the full schema."
        id="note-retrieval"
      />

      <WarningBlock
        title="Ambiguous Schema Elements"
        content="Real-world databases often have ambiguous column names like 'id', 'name', 'type', or 'status' that appear in multiple tables. Without proper schema linking, the model may reference the wrong table's column. Always include foreign key relationships and sample values in the schema representation to disambiguate."
        id="warning-ambiguity"
      />

      <NoteBlock
        type="note"
        title="Impact on SQL Accuracy"
        content="Research shows that schema linking accounts for 30-40% of Text-to-SQL errors. On the Spider benchmark, providing oracle schema links (telling the model exactly which tables and columns are needed) improves GPT-4 accuracy from ~85% to ~93%. This gap represents the practical ceiling that better schema linking methods can close."
        id="note-impact"
      />
    </div>
  )
}
