import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function StructuredGeneration() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Structured & Constrained Generation</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        LLMs generate free-form text, but many applications require structured outputs like JSON,
        SQL, or XML. Constrained generation techniques guarantee that model output conforms to a
        specified schema or grammar, eliminating parsing failures entirely.
      </p>

      <DefinitionBlock
        title="Constrained Decoding"
        definition="Constrained decoding modifies the sampling distribution at each step to only allow tokens that are valid according to a grammar or schema. Invalid tokens receive $-\infty$ logits (are masked out), ensuring 100% conformance."
        id="def-constrained"
      />

      <NoteBlock
        type="intuition"
        title="How It Works"
        content={'Think of it like autocomplete with strict rules. At each token position, a finite state machine or parser tracks what tokens are valid next. If you have generated \'{"name": "\' so far, the only valid continuations are string characters -- not a closing brace or number. The model still chooses among valid tokens using its learned distribution.'}
        id="note-how-it-works"
      />

      <PythonCode
        title="outlines_json.py"
        code={`# Outlines: structured generation library
# pip install outlines
import outlines

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")

# Define a JSON schema
from pydantic import BaseModel
from typing import List

class MovieReview(BaseModel):
    title: str
    year: int
    rating: float  # 0-10
    genres: List[str]
    summary: str

# Create a generator constrained to this schema
generator = outlines.generate.json(model, MovieReview)

prompt = """Review the movie Inception and provide a structured review.
Output your review as JSON:"""

# Output is GUARANTEED to be valid JSON matching the schema
review = generator(prompt)
print(review)
# MovieReview(title='Inception', year=2010, rating=9.2,
#   genres=['Sci-Fi', 'Thriller'], summary='A mind-bending...')`}
        id="code-outlines-json"
      />

      <PythonCode
        title="outlines_regex.py"
        code={`import outlines

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")

# Constrain to a regex pattern -- e.g., US phone number
phone_generator = outlines.generate.regex(
    model,
    r"\\(\\d{3}\\) \\d{3}-\\d{4}"
)

result = phone_generator("Generate a US phone number: ")
print(result)  # Always matches (XXX) XXX-XXXX format

# Constrain to specific choices
sentiment_generator = outlines.generate.choice(
    model,
    ["positive", "negative", "neutral"]
)

sentiment = sentiment_generator(
    "Classify the sentiment: 'This movie was absolutely terrible.' -> "
)
print(sentiment)  # Always one of the three choices

# Grammar-based generation (e.g., arithmetic expressions)
grammar = r\"""
    start: expr
    expr: term (("+"|"-") term)*
    term: NUMBER
    NUMBER: /[0-9]+/
\"""
math_gen = outlines.generate.cfg(model, grammar)
result = math_gen("Write a math expression: ")
print(result)  # e.g., "42+17-3"`}
        id="code-outlines-regex"
      />

      <ExampleBlock
        title="Guidance Template Example"
        problem="Use Microsoft Guidance to create a structured character profile."
        steps={[
          { formula: 'Define template with {{gen}} blocks', explanation: 'Guidance uses template syntax where {{gen}} marks places the LLM fills in.' },
          { formula: 'Add constraints: stop tokens, regex, choices', explanation: 'Each gen block can have constraints like stop="\\n" or pattern="[0-9]+".' },
          { formula: 'LLM fills in constrained slots', explanation: 'The model generates text that must satisfy all constraints, producing a valid structured output.' },
        ]}
        id="example-guidance"
      />

      <PythonCode
        title="instructor_openai.py"
        code={`# Instructor: structured outputs from any OpenAI-compatible API
# pip install instructor openai
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

# Patch OpenAI client with instructor
client = instructor.from_openai(OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama
    api_key="ollama",
))

class ExtractedEntity(BaseModel):
    name: str
    entity_type: str = Field(description="person, org, location, etc.")
    confidence: float = Field(ge=0.0, le=1.0)

class ExtractionResult(BaseModel):
    entities: List[ExtractedEntity]
    summary: str

# Instructor ensures the response matches the schema
result = client.chat.completions.create(
    model="llama3.2",
    response_model=ExtractionResult,
    messages=[{
        "role": "user",
        "content": "Extract entities: Apple CEO Tim Cook announced new AI features at WWDC in Cupertino."
    }],
)

for entity in result.entities:
    print(f"  {entity.name} ({entity.entity_type}): {entity.confidence:.0%}")`}
        id="code-instructor"
      />

      <NoteBlock
        type="note"
        title="Performance Impact"
        content="Constrained generation adds minimal overhead. The grammar/schema check at each step is fast (typically microseconds) compared to the model forward pass (milliseconds). The main cost is that masking tokens can sometimes force the model into lower-quality continuations."
        id="note-performance"
      />

      <WarningBlock
        title="Structured Output Does Not Mean Correct Output"
        content="Constrained generation guarantees syntactic validity (valid JSON, matching schema) but not semantic correctness. The model can still hallucinate values, provide wrong numbers, or fill fields with nonsensical content. Always validate the content, not just the structure."
        id="warning-correctness"
      />
    </div>
  )
}
