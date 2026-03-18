import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function JSONGeneration() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">JSON Mode and Structured Outputs</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Structured output generation ensures LLMs produce valid, parseable data formats
        like JSON. This is essential for integrating LLMs into software systems where
        downstream code expects well-formed data structures rather than free-form text.
        Modern APIs offer JSON mode, JSON Schema enforcement, and structured output guarantees.
      </p>

      <DefinitionBlock
        title="Structured Output"
        definition="Structured output constrains an LLM's generation to produce text conforming to a formal grammar or schema $G$. Given a prompt $p$, the model generates $y$ such that $y \in \mathcal{L}(G)$ where $\mathcal{L}(G)$ is the language defined by grammar $G$. For JSON Schema validation: $\text{validate}(y, \text{schema}) = \text{true}$."
        notation="G = grammar/schema, \mathcal{L}(G) = valid strings, y = output"
        id="def-structured-output"
      />

      <h2 className="text-2xl font-semibold">JSON Mode Basics</h2>
      <p className="text-gray-700 dark:text-gray-300">
        JSON mode guarantees the model output is valid JSON, but does not enforce a specific
        schema. The model may produce any valid JSON object. For schema enforcement, you need
        structured outputs with a JSON Schema definition.
      </p>

      <PythonCode
        title="json_mode.py"
        code={`from openai import OpenAI
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
# }`}
        id="code-json-mode"
      />

      <h2 className="text-2xl font-semibold">Schema-Enforced Structured Outputs</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Structured outputs with JSON Schema enforcement guarantee that every response matches
        your exact schema definition, including required fields, types, enums, and nested structures.
      </p>

      <PythonCode
        title="structured_outputs.py"
        code={`from openai import OpenAI
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
# No need for try/except around JSON parsing`}
        id="code-structured"
      />

      <ExampleBlock
        title="JSON Schema for API Responses"
        problem="Design a JSON Schema for a product search API response that an LLM must produce."
        steps={[
          { formula: '\\text{Define root object with required fields: products, total\\_count, query}', explanation: 'The top-level schema specifies what fields must always be present.' },
          { formula: '\\text{products: array of objects with name, price, category, in\\_stock}', explanation: 'Each product has typed fields -- string, number, string, boolean.' },
          { formula: '\\text{Add constraints: price > 0, category } \\in \\text{ enum values}', explanation: 'JSON Schema supports numeric ranges and enumerated string values.' },
          { formula: '\\text{Schema validation guarantees type safety at generation time}', explanation: 'Unlike post-hoc parsing, schema-constrained generation never produces invalid output.' },
        ]}
        id="example-schema"
      />

      <PythonCode
        title="json_schema_definition.py"
        code={`import json

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

print(json.dumps(product_search_schema, indent=2))`}
        id="code-schema"
      />

      <NoteBlock
        type="tip"
        title="Pydantic vs. Raw JSON Schema"
        content="Use Pydantic models when working with Python -- they provide type checking, validation, and auto-generate JSON Schema. Use raw JSON Schema when you need cross-language compatibility or when the schema is dynamically constructed. Both approaches produce identical constraints for the LLM."
        id="note-pydantic"
      />

      <WarningBlock
        title="Structured Output Limitations"
        content="Schema enforcement slightly increases latency (5-15%) because the model must respect constraints at each token. Very complex schemas with deep nesting or many enum values can degrade output quality. Keep schemas as flat as possible and limit enum lists to under 50 values for best results."
        id="warning-limitations"
      />

    </div>
  )
}
