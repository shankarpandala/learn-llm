import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function PromptChaining() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Prompt Chaining and Pipelines</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Complex tasks often exceed what a single prompt can reliably accomplish. Prompt chaining
        decomposes a task into sequential steps, where each step's output feeds into the next.
        This yields more reliable results and enables debugging at each stage.
      </p>

      <DefinitionBlock
        title="Prompt Chaining"
        definition="A technique where a complex task is broken into a sequence of simpler subtasks, each handled by a separate LLM call. The output of step $i$ is processed and fed as input to step $i+1$, forming a pipeline: $y_i = f_i(y_{i-1}, p_i)$ where $p_i$ is the prompt for step $i$."
        id="def-chaining"
      />

      <h2 className="text-2xl font-semibold">Chain Architectures</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Chains can be sequential (linear pipeline), branching (parallel subtasks merged later),
        or conditional (different paths based on intermediate results). The overall success
        probability of a sequential chain is:
      </p>
      <BlockMath math="P(\text{chain}) = \prod_{i=1}^{n} P(\text{step}_i \text{ succeeds})" />
      <p className="text-gray-700 dark:text-gray-300">
        This means each step must be highly reliable. A 5-step chain where each step succeeds
        95% of the time only succeeds 77% overall.
      </p>

      <ExampleBlock
        title="Research Report Pipeline"
        problem="Generate a well-structured research summary from a raw document."
        steps={[
          { formula: '\\text{Step 1: Extract key claims and findings}', explanation: 'First LLM call focuses purely on information extraction.' },
          { formula: '\\text{Step 2: Fact-check claims against source}', explanation: 'Second call verifies extracted claims, flagging unsupported ones.' },
          { formula: '\\text{Step 3: Organize into structured outline}', explanation: 'Third call arranges verified claims into a logical structure.' },
          { formula: '\\text{Step 4: Generate polished summary}', explanation: 'Final call produces the finished report from the outline.' },
        ]}
        id="example-pipeline"
      />

      <PythonCode
        title="prompt_chaining_pipeline.py"
        code={`from openai import OpenAI
import json

client = OpenAI()

def llm_call(prompt, system="You are a helpful assistant.", model="gpt-4o-mini",
             temperature=0, json_mode=False):
    """Wrapper for a single LLM call in the chain."""
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content

# --- Sequential Chain: Document Analysis Pipeline ---

def analyze_document(document: str) -> dict:
    """Multi-step document analysis chain."""

    # Step 1: Extract key information
    extracted = llm_call(
        f"Extract the main claims, entities, and statistics from this document. "
        f"Return as JSON with keys: claims (list), entities (list), statistics (list).\\n\\n"
        f"Document:\\n{document}",
        json_mode=True,
    )
    data = json.loads(extracted)
    print(f"Step 1: Extracted {len(data['claims'])} claims")

    # Step 2: Classify and prioritize
    classified = llm_call(
        f"Given these extracted claims, classify each by importance (high/medium/low) "
        f"and topic category. Return as JSON with key 'classified_claims'.\\n\\n"
        f"Claims: {json.dumps(data['claims'])}",
        json_mode=True,
    )
    priorities = json.loads(classified)
    print(f"Step 2: Classified claims by priority")

    # Step 3: Generate summary from structured data
    summary = llm_call(
        f"Write a concise executive summary (3-5 paragraphs) based on these "
        f"prioritized findings. Focus on high-importance items first.\\n\\n"
        f"Data: {json.dumps(priorities)}\\n"
        f"Entities: {json.dumps(data['entities'])}\\n"
        f"Statistics: {json.dumps(data['statistics'])}",
    )
    print(f"Step 3: Generated summary ({len(summary.split())} words)")

    return {"extracted": data, "classified": priorities, "summary": summary}

# --- Conditional Chain: Route based on intent ---

def conditional_chain(user_query: str) -> str:
    """Route to different chains based on detected intent."""

    # Step 1: Classify intent
    intent = llm_call(
        f"Classify this query into exactly one category: "
        f"'technical', 'billing', 'general'. Return JSON: {{\"intent\": \"...\"}}.\\n\\n"
        f"Query: {user_query}",
        json_mode=True,
    )
    intent_type = json.loads(intent)["intent"]
    print(f"Detected intent: {intent_type}")

    # Step 2: Route to specialized prompt
    prompts = {
        "technical": "You are a senior engineer. Provide detailed technical guidance.",
        "billing": "You are a billing specialist. Help with account and payment questions.",
        "general": "You are a friendly assistant. Provide helpful general information.",
    }
    result = llm_call(user_query, system=prompts.get(intent_type, prompts["general"]))
    return result

# Example usage
print(conditional_chain("Why is my API returning 429 errors?"))`}
        id="code-chaining"
      />

      <NoteBlock
        type="tip"
        title="Validation Between Steps"
        content="Add validation gates between chain steps: check JSON parses correctly, verify expected fields exist, ensure outputs meet length/format constraints. Retry failed steps with modified prompts before failing the entire chain. Log intermediate results for debugging."
        id="note-validation"
      />

      <WarningBlock
        title="Latency and Cost Accumulate"
        content="Each chain step adds latency (typically 0.5-3s) and token costs. A 4-step chain is 4x slower and more expensive than a single call. Use chaining only when a single prompt genuinely cannot handle the task. Consider parallelizing independent steps."
        id="warning-latency"
      />

      <NoteBlock
        type="note"
        title="Frameworks for Chaining"
        content="LangChain, LlamaIndex, and Haystack provide abstractions for building prompt chains. However, simple Python functions (as shown above) often provide more control and debuggability. Start simple, adopt a framework only when the complexity justifies it."
        id="note-frameworks"
      />
    </div>
  )
}
