import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function CustomModels() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Creating Custom Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Beyond basic Modelfiles, Ollama lets you create specialized models for specific domains,
        chain models together, and manage a library of purpose-built assistants. This section
        covers practical patterns for building and organizing custom models.
      </p>

      <ExampleBlock
        title="Common Custom Model Patterns"
        problem="What kinds of custom models are most useful?"
        steps={[
          { formula: 'Domain expert: system prompt + low temperature', explanation: 'Medical, legal, financial assistants with domain-specific instructions.' },
          { formula: 'Output formatter: strict output format instructions', explanation: 'JSON-only responses, markdown tables, specific report formats.' },
          { formula: 'Persona: character + style instructions', explanation: 'Customer service agents, tutors, creative writing helpers.' },
          { formula: 'Chain-of-thought: reasoning instructions', explanation: 'Step-by-step problem solving with explicit reasoning.' },
        ]}
        id="example-patterns"
      />

      <PythonCode
        title="Terminal"
        code={`# Pattern 1: JSON-only responder
cat > Modelfile-json << 'EOF'
FROM llama3.2

SYSTEM """You are a JSON API. You ONLY respond with valid JSON.
Never include explanations, markdown, or any text outside the JSON object.
Always use double quotes for keys and string values.
If you cannot fulfill the request, respond with {"error": "description"}."""

PARAMETER temperature 0.1
PARAMETER top_p 0.8
EOF

ollama create json-api -f Modelfile-json
ollama run json-api "Extract entities from: Apple CEO Tim Cook visited Tokyo"
# {"entities": [{"name": "Apple", "type": "organization"},
#  {"name": "Tim Cook", "type": "person"}, {"name": "Tokyo", "type": "location"}]}

# Pattern 2: Code reviewer
cat > Modelfile-reviewer << 'EOF'
FROM llama3.1:8b

SYSTEM """You are a strict code reviewer. For each code snippet:
1. List bugs and issues (CRITICAL, WARNING, INFO)
2. Suggest specific fixes with corrected code
3. Rate overall quality (1-10)
Keep reviews concise and actionable."""

PARAMETER temperature 0.2
PARAMETER num_predict 2048
EOF

ollama create code-reviewer -f Modelfile-reviewer`}
        id="code-patterns"
      />

      <PythonCode
        title="model_manager.py"
        code={`import ollama
import json

# Define a library of custom models
MODEL_CONFIGS = {
    "sql-expert": {
        "base": "llama3.1:8b",
        "system": (
            "You are a SQL expert. Convert natural language to SQL queries. "
            "Always specify the assumed table schema before the query. "
            "Use standard SQL syntax compatible with PostgreSQL."
        ),
        "params": {"temperature": 0.1, "top_p": 0.85},
    },
    "eli5": {
        "base": "llama3.2",
        "system": (
            "Explain everything as if talking to a 5-year-old. Use simple words, "
            "fun analogies, and short sentences. Never use jargon or technical terms."
        ),
        "params": {"temperature": 0.8, "top_p": 0.95},
    },
    "summarizer": {
        "base": "llama3.2",
        "system": (
            "You are a summarization engine. Provide concise bullet-point summaries. "
            "Maximum 5 bullet points. Each bullet should be one sentence. "
            "Capture the key facts and insights only."
        ),
        "params": {"temperature": 0.2, "top_p": 0.8},
    },
}

def create_all_models():
    for name, config in MODEL_CONFIGS.items():
        params = "\\n".join(
            f"PARAMETER {k} {v}" for k, v in config["params"].items()
        )
        modelfile = f'''FROM {config["base"]}

SYSTEM """{config["system"]}"""

{params}
'''
        print(f"Creating {name}...")
        ollama.create(model=name, modelfile=modelfile)
        print(f"  Done!")

def test_model(name, prompt):
    response = ollama.generate(model=name, prompt=prompt)
    print(f"[{name}] {response['response'][:300]}")

create_all_models()
test_model("sql-expert", "Find all users who signed up last month")
test_model("eli5", "What is quantum entanglement?")
test_model("summarizer", "Summarize the concept of machine learning")`}
        id="code-manager"
      />

      <NoteBlock
        type="tip"
        title="Version Your Modelfiles"
        content="Store Modelfiles in version control alongside your application code. This ensures reproducibility -- anyone can recreate your exact model configuration. Include a README documenting each model's purpose and expected behavior."
        id="note-version-control"
      />

      <WarningBlock
        title="System Prompts Are Not Security Boundaries"
        content="Users can override or extract system prompts through prompt injection. Do not rely on system prompts for access control or to hide sensitive information. Treat them as behavioral guidelines, not security mechanisms."
        id="warning-security"
      />

      <NoteBlock
        type="note"
        title="Model Inheritance"
        content="Custom models built with FROM reference the base model's weights. Deleting the base model will break custom models that depend on it. Use 'ollama show <model>' to see the full dependency chain."
        id="note-inheritance"
      />
    </div>
  )
}
