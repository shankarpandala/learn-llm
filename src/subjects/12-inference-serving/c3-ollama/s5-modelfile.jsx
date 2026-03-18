import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Modelfile() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">The Modelfile</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        A Modelfile is Ollama's equivalent of a Dockerfile. It defines a custom model configuration
        including the base model, system prompt, parameters, and chat template. Modelfiles let you
        create purpose-built assistants without any fine-tuning.
      </p>

      <DefinitionBlock
        title="Modelfile"
        definition="A text file with instructions for creating a custom Ollama model. It specifies the base model (FROM), system prompt (SYSTEM), generation parameters (PARAMETER), chat template (TEMPLATE), and optional adapter weights (ADAPTER)."
        id="def-modelfile"
      />

      <PythonCode
        title="Terminal"
        code={`# Create a basic Modelfile
cat > Modelfile << 'EOF'
# Base model
FROM llama3.2

# System prompt
SYSTEM """You are a senior Python developer. You write clean, well-documented
code following PEP 8 conventions. You always include type hints and docstrings.
When asked to write code, provide complete, runnable examples."""

# Generation parameters
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 1024
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
EOF

# Build the custom model
ollama create python-dev -f Modelfile
# transferring model data
# creating model layer
# writing manifest
# success

# Run it
ollama run python-dev "Write a function to merge two sorted lists"`}
        id="code-basic-modelfile"
      />

      <ExampleBlock
        title="Modelfile Instructions Reference"
        problem="What instructions are available in a Modelfile?"
        steps={[
          { formula: 'FROM <model> — base model (required)', explanation: 'Specifies the parent model: FROM llama3.2, FROM ./model.gguf, etc.' },
          { formula: 'SYSTEM <text> — system prompt', explanation: 'Sets the system message that defines the model personality.' },
          { formula: 'PARAMETER <key> <value> — generation settings', explanation: 'temperature, top_p, top_k, num_predict, stop, repeat_penalty, etc.' },
          { formula: 'TEMPLATE <template> — chat format template', explanation: 'Go template defining how messages are formatted for the model.' },
          { formula: 'ADAPTER <path> — LoRA adapter', explanation: 'Path to a GGUF LoRA adapter file to apply on top of the base model.' },
          { formula: 'LICENSE <text> — license information', explanation: 'Embeds license text in the model metadata.' },
        ]}
        id="example-instructions"
      />

      <PythonCode
        title="Terminal"
        code={`# Advanced Modelfile with custom template
cat > Modelfile-analyst << 'MODELFILE'
FROM llama3.1:8b

SYSTEM """You are a data analyst. You communicate findings clearly using
bullet points and tables. Always cite your reasoning and acknowledge
uncertainty. Format numbers with appropriate precision."""

PARAMETER temperature 0.2
PARAMETER top_p 0.85
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 8192
PARAMETER num_predict 2048

# Custom template (Go template syntax)
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
MODELFILE

ollama create data-analyst -f Modelfile-analyst
ollama run data-analyst "Analyze the trend: Q1=$1.2M, Q2=$1.5M, Q3=$1.1M, Q4=$1.8M"`}
        id="code-advanced-modelfile"
      />

      <PythonCode
        title="create_models.py"
        code={`import ollama

# Create a model programmatically
modelfile_content = '''FROM llama3.2

SYSTEM """You are a friendly tutor who explains concepts step by step.
Use analogies and examples. Ask follow-up questions to check understanding."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
'''

# Create the model via Python API
ollama.create(model="tutor", modelfile=modelfile_content)

# Test it
response = ollama.chat(
    model="tutor",
    messages=[{"role": "user", "content": "Explain recursion to a beginner"}]
)
print(response["message"]["content"])`}
        id="code-python-create"
      />

      <NoteBlock
        type="tip"
        title="Iterate Quickly"
        content="Creating a model from a Modelfile is instant (no training involved). Change the system prompt, rebuild with 'ollama create', and test immediately. This makes Modelfiles ideal for rapid prototyping of different assistant personalities."
        id="note-iterate"
      />

      <WarningBlock
        title="Template Compatibility"
        content="Custom TEMPLATE instructions must match the base model's expected chat format. Using the wrong template causes garbled output. When in doubt, omit TEMPLATE and let Ollama use the model's default. Only customize it when you need to change how messages are structured."
        id="warning-template"
      />
    </div>
  )
}
