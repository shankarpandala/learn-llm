import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function GPT4Capabilities() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">GPT-4: Capabilities and Multimodal Reasoning</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        GPT-4 (March 2023) represented a qualitative leap in capability. While OpenAI did not
        disclose architectural details, GPT-4 demonstrated human-level performance on professional
        exams, robust multimodal understanding (text and images), and significantly improved
        reasoning, factuality, and steerability compared to GPT-3.5.
      </p>

      <DefinitionBlock
        title="Multimodal Language Model"
        definition="A model that accepts and reasons over multiple input modalities (e.g., text and images). GPT-4V processes visual inputs through a vision encoder whose representations are projected into the language model's embedding space, enabling unified text-image reasoning."
        id="def-multimodal"
      />

      <h2 className="text-2xl font-semibold">Benchmark Performance</h2>
      <p className="text-gray-700 dark:text-gray-300">
        GPT-4 scored in the 90th percentile on the bar exam (vs 10th for GPT-3.5), 99th percentile
        on the GRE Verbal, and achieved 86.4% on MMLU (5-shot), up from 70% for GPT-3.5. These
        gains came from better pre-training data, RLHF refinement, and likely architectural
        improvements including mixture-of-experts.
      </p>

      <ExampleBlock
        title="GPT-4 Exam Performance"
        problem="Compare GPT-3.5 and GPT-4 on key professional and academic benchmarks."
        steps={[
          { formula: '\\text{Bar Exam}: \\text{GPT-3.5} \\approx 10\\text{th pctile} \\to \\text{GPT-4} \\approx 90\\text{th pctile}', explanation: 'A massive jump, demonstrating strong legal reasoning capabilities.' },
          { formula: '\\text{MMLU (5-shot)}: 70.0\\% \\to 86.4\\%', explanation: 'Broad improvement across 57 subjects from STEM to humanities.' },
          { formula: '\\text{HumanEval (code)}: 48.1\\% \\to 67.0\\%', explanation: 'Significant gains in code generation and understanding.' },
          { formula: '\\text{AP Calculus BC}: 43\\% \\to 76\\%', explanation: 'Major improvement in complex mathematical reasoning.' },
        ]}
        id="example-gpt4-benchmarks"
      />

      <h2 className="text-2xl font-semibold">Rumored Architecture</h2>
      <p className="text-gray-700 dark:text-gray-300">
        While not officially confirmed, credible reports suggest GPT-4 uses a Mixture-of-Experts
        (MoE) architecture with 8 experts of roughly 220B parameters each, for a total of ~1.76T
        parameters but only ~280B active per forward pass. This allows much greater total capacity
        while keeping inference cost manageable.
      </p>

      <NoteBlock
        type="intuition"
        title="Why MoE for GPT-4?"
        content="A dense 1.76T parameter model would be prohibitively expensive to run. MoE lets you store vast knowledge across experts but only activate a small subset per token. Think of it like a company with many specialists: for any given question, only the relevant experts are consulted, keeping response time fast."
        id="note-moe-intuition"
      />

      <PythonCode
        title="gpt4_api_usage.py"
        code={`from openai import OpenAI

client = OpenAI()  # uses OPENAI_API_KEY env var

# Text completion with GPT-4
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the key architectural differences "
         "between GPT-3 and GPT-4 in 3 bullet points."},
    ],
    temperature=0.7,
    max_tokens=300,
)
print(response.choices[0].message.content)

# GPT-4 Vision (multimodal)
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Example: analyzing an architecture diagram
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this neural network architecture."},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{encode_image('arch.png')}"
            }},
        ]},
    ],
)
print(response.choices[0].message.content)

# Token usage tracking
usage = response.usage
print(f"Prompt tokens: {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total tokens: {usage.total_tokens}")`}
        id="code-gpt4-api"
      />

      <NoteBlock
        type="note"
        title="Predictable Scaling"
        content="The GPT-4 technical report revealed that OpenAI could predict GPT-4's final loss and benchmark performance from much smaller training runs. They trained models at 1/10,000th and 1/1,000th the compute and extrapolated, demonstrating that scaling laws enable reliable planning of training runs worth tens of millions of dollars."
        id="note-predictable-scaling"
      />

      <WarningBlock
        title="Closed-Source Limitations"
        content="GPT-4's architecture, training data, and parameter count are not officially disclosed. This limits reproducibility and independent analysis. The research community has increasingly pushed for open models (LLaMA, Mistral) partly as a response to this opacity."
        id="warning-closed-source"
      />
    </div>
  )
}
