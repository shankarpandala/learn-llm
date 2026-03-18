import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ModelParams() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Model Selection & Parameter Tuning</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Open WebUI lets you select models and tune generation parameters directly from the chat
        interface. Understanding these parameters helps you get better results for different tasks
        without writing any code.
      </p>

      <DefinitionBlock
        title="Generation Parameters in Open WebUI"
        definition="Open WebUI exposes key generation parameters through the chat settings panel: temperature (creativity), top-p (nucleus sampling threshold), top-k (vocabulary restriction), max tokens (response length), and frequency/presence penalties (repetition control)."
        id="def-params"
      />

      <ExampleBlock
        title="Parameter Presets by Task"
        problem="What parameters work best for different tasks?"
        steps={[
          { formula: 'Coding: T=0.1, top_p=0.9, max_tokens=2048', explanation: 'Low temperature for precise, deterministic code generation.' },
          { formula: 'Creative writing: T=0.9, top_p=0.95, freq_penalty=0.5', explanation: 'High temperature for diverse, creative output with reduced repetition.' },
          { formula: 'Factual Q&A: T=0.0, top_p=1.0, max_tokens=500', explanation: 'Zero temperature for the most likely (factual) response.' },
          { formula: 'Brainstorming: T=1.2, top_p=0.98, presence_penalty=0.8', explanation: 'High temperature and presence penalty for maximum diversity.' },
        ]}
        id="example-presets"
      />

      <PythonCode
        title="Terminal"
        code={`# Access model parameters through the API
curl -X POST http://localhost:3000/api/chat/completions \\
    -H "Authorization: Bearer YOUR_TOKEN" \\
    -H "Content-Type: application/json" \\
    -d '{
        "model": "llama3.2",
        "messages": [
            {"role": "user", "content": "Write a haiku about programming"}
        ],
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": 100,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "stream": false
    }'

# In the Open WebUI interface:
# 1. Click the gear icon next to the model selector
# 2. Adjust sliders for temperature, top-p, etc.
# 3. Settings persist per conversation

# Or set defaults in the Modelfile for Ollama models:
# PARAMETER temperature 0.7
# PARAMETER top_p 0.9`}
        id="code-params-api"
      />

      <PythonCode
        title="param_comparison.py"
        code={`import requests
import json

API_URL = "http://localhost:3000/api/chat/completions"
HEADERS = {
    "Authorization": "Bearer YOUR_TOKEN",
    "Content-Type": "application/json",
}

def generate_with_params(prompt, model="llama3.2", **params):
    """Test different parameter settings."""
    resp = requests.post(API_URL, headers=HEADERS, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "stream": False,
        **params,
    })
    return resp.json()["choices"][0]["message"]["content"]

prompt = "Describe a sunset."

# Compare different temperature settings
for temp in [0.0, 0.5, 1.0, 1.5]:
    result = generate_with_params(prompt, temperature=temp)
    print(f"T={temp}: {result[:80]}...")
    print()

# Compare top-p settings
for top_p in [0.5, 0.8, 0.95]:
    result = generate_with_params(prompt, temperature=0.8, top_p=top_p)
    print(f"top_p={top_p}: {result[:80]}...")
    print()`}
        id="code-comparison"
      />

      <NoteBlock
        type="tip"
        title="Save Parameter Presets"
        content="Create model presets in Open WebUI for common task types. Each preset saves the model choice, system prompt, and all parameter settings. Switch between presets with one click instead of reconfiguring parameters each time."
        id="note-presets"
      />

      <WarningBlock
        title="Parameters Interact with Each Other"
        content="Temperature, top-p, and top-k all affect the same sampling distribution. Setting temperature=2.0 with top_p=0.1 produces confusing results because temperature flattens the distribution while top_p aggressively filters it. Start with one parameter and adjust gradually."
        id="warning-interaction"
      />
    </div>
  )
}
