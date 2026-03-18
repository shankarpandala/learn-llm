import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function CommandR() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Cohere Command R: RAG-Optimized Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Cohere's Command R (March 2024) and Command R+ (April 2024) were purpose-built for
        retrieval-augmented generation (RAG) and enterprise tool use. With 35B and 104B parameters
        respectively, they introduced built-in grounded generation with inline citations, native
        tool calling, and strong multilingual performance across 10 languages.
      </p>

      <DefinitionBlock
        title="Grounded Generation"
        definition="A generation paradigm where the model produces responses that are explicitly grounded in provided documents. The model generates inline citations (e.g., [doc1], [doc2]) pointing to specific source passages, enabling verifiable outputs. Formally, the model learns $P(y, c \mid x, D)$ where $c$ is the citation set and $D$ is the document collection."
        id="def-grounded-gen"
      />

      <h2 className="text-2xl font-semibold">Architecture and Training</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Command R uses a standard decoder-only transformer with GQA, RoPE, SwiGLU, and a 128K
        context window. What makes it distinctive is not the base architecture but the post-training:
        extensive fine-tuning on RAG tasks, tool use, structured output generation, and
        multi-turn conversations with document context.
      </p>

      <ExampleBlock
        title="Command R Model Family"
        problem="Compare Command R and Command R+ specifications."
        steps={[
          { formula: '\\text{Command R (35B)}: L{=}40, d{=}8192, h_q{=}64, h_{kv}{=}8', explanation: 'Optimized for RAG workloads. 128K context. 10 supported languages. Apache 2.0 licensed.' },
          { formula: '\\text{Command R+ (104B)}: L{=}64, d{=}12288, h_q{=}96, h_{kv}{=}8', explanation: 'Larger variant for complex enterprise tasks. Approaches GPT-4 on RAG benchmarks.' },
          { formula: '\\text{Context}: 128\\text{K tokens with RAG-specific training}', explanation: 'Trained specifically to handle long documents with accurate citation and attribution.' },
        ]}
        id="example-command-r-specs"
      />

      <h2 className="text-2xl font-semibold">Built-in Tool Use</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Command R models have native support for tool/function calling. They can generate
        structured JSON tool invocations, process tool results, and synthesize multi-step
        tool-assisted answers. This is trained directly into the model rather than relying on
        prompt engineering.
      </p>

      <PythonCode
        title="command_r_rag_example.py"
        code={`from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Command R (35B) - requires significant GPU memory
model_name = "CohereForAI/c4ai-command-r-v01"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# For demonstration, inspect the chat template with RAG
documents = [
    {"title": "LLM Scaling", "text": "Scaling laws show that model performance improves "
     "predictably with increased parameters, data, and compute."},
    {"title": "RAG Overview", "text": "Retrieval-Augmented Generation combines a retriever "
     "with a generator to produce grounded, factual responses."},
]

# Command R uses a special RAG-aware chat format
conversation = [
    {"role": "user", "content": "What is RAG and how does it relate to scaling?"},
]

# Apply Command R's grounded generation template
grounded_prompt = tokenizer.apply_grounded_generation_template(
    conversation,
    documents=documents,
    citation_mode="accurate",  # generates inline citations
    tokenize=False,
    add_generation_prompt=True,
)
print("=== Grounded Generation Prompt ===")
print(grounded_prompt[:500])

# Tool use example with Command R
tools = [
    {"name": "web_search", "description": "Search the web for information",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "The search query"}
     }, "required": ["query"]}},
    {"name": "calculator", "description": "Perform mathematical calculations",
     "parameters": {"type": "object", "properties": {
         "expression": {"type": "string", "description": "Math expression"}
     }, "required": ["expression"]}},
]

tool_prompt = tokenizer.apply_tool_use_template(
    [{"role": "user", "content": "What is 42 * 17 and who invented the transformer?"}],
    tools=tools,
    tokenize=False,
    add_generation_prompt=True,
)
print("\\n=== Tool Use Prompt ===")
print(tool_prompt[:500])`}
        id="code-command-r"
      />

      <NoteBlock
        type="note"
        title="Enterprise RAG Focus"
        content="Command R's RAG training includes learning to correctly attribute claims to source documents, refuse to answer when documents don't contain relevant information, and distinguish between information from documents versus parametric knowledge. This makes it particularly suited for enterprise deployments where factual grounding and auditability are critical."
        id="note-enterprise-rag"
      />

      <NoteBlock
        type="tip"
        title="Using Command R via API"
        content="While the open-weight models are available on HuggingFace, Cohere's API provides the most optimized inference with built-in RAG pipeline support (including document ingestion, retrieval, and grounded generation) through the /chat endpoint with the 'documents' parameter."
        id="note-api-tip"
      />

      <WarningBlock
        title="Citation Accuracy"
        content="While Command R is trained for grounded generation, citation accuracy is not perfect. The model may occasionally hallucinate citations or attribute information to the wrong document. Always implement verification logic that checks whether cited passages actually support the generated claims in production RAG systems."
        id="warning-citation-accuracy"
      />
    </div>
  )
}
