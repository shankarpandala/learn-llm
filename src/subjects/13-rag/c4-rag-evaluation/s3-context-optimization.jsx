import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ContextOptimization() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Context Window Optimization</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The context window is a finite and precious resource. Stuffing it with too many retrieved
        documents dilutes relevance and can degrade generation quality. Context optimization
        techniques maximize the signal-to-noise ratio of the information passed to the LLM,
        ensuring every token in the context contributes to a better answer.
      </p>

      <DefinitionBlock
        title="Context Window Budget"
        definition="The context budget $B$ is the maximum tokens available for retrieved context: $B = W - T_{\text{system}} - T_{\text{query}} - T_{\text{output}}$ where $W$ is the model's context window, and the other terms are the system prompt, query, and reserved output tokens respectively."
        id="def-context-budget"
      />

      <h2 className="text-2xl font-semibold">The Lost in the Middle Problem</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Research by Liu et al. (2023) showed that LLMs pay less attention to information in
        the middle of long contexts. Relevant information placed at the beginning or end of
        the context is used more effectively than information buried in the middle.
      </p>
      <BlockMath math="P(\text{use info at position } i) \propto \begin{cases} \text{high} & i \text{ near start or end} \\ \text{low} & i \text{ in middle} \end{cases}" />

      <ExampleBlock
        title="Context Window Optimization Strategies"
        problem="Optimize context usage for a model with 8K context window."
        steps={[
          { formula: 'B = 8192 - 500 - 100 - 1000 = 6592 \\text{ tokens}', explanation: 'Calculate available budget after system prompt, query, and output reservation.' },
          { formula: '\\text{With 512-token chunks: } \\lfloor 6592/512 \\rfloor = 12 \\text{ chunks max}', explanation: 'Maximum number of full chunks that fit in the budget.' },
          { formula: '\\text{Order: most relevant at start and end}', explanation: 'Place the highest-relevance chunks at positions 1 and n, not in the middle.' },
          { formula: '\\text{Compress: remove redundant information}', explanation: 'Use LLM-based compression to extract only the relevant sentences from each chunk.' },
        ]}
        id="example-context-optimization"
      />

      <PythonCode
        title="context_optimization.py"
        code={`# Context window optimization techniques
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(enc.encode(text))

# 1. Token budget calculation
MODEL_CONTEXT = 8192
SYSTEM_TOKENS = 500
QUERY_TOKENS = 100
OUTPUT_RESERVE = 1000
BUDGET = MODEL_CONTEXT - SYSTEM_TOKENS - QUERY_TOKENS - OUTPUT_RESERVE
print(f"Context budget: {BUDGET} tokens")

# 2. Smart context stuffing with budget awareness
def stuff_context(docs, max_tokens):
    """Add documents until budget is exhausted."""
    context_parts = []
    total_tokens = 0
    for doc in docs:
        doc_tokens = count_tokens(doc.page_content)
        if total_tokens + doc_tokens > max_tokens:
            break
        context_parts.append(doc.page_content)
        total_tokens += doc_tokens
    return "\\n\\n".join(context_parts), total_tokens

# 3. Lost-in-the-middle reordering
def reorder_for_attention(docs):
    """Place most relevant docs at start and end (avoid middle)."""
    if len(docs) <= 2:
        return docs
    # docs assumed sorted by relevance (most relevant first)
    reordered = []
    for i, doc in enumerate(docs):
        if i % 2 == 0:
            reordered.insert(0, doc)   # Prepend (start)
        else:
            reordered.append(doc)      # Append (end)
    return reordered

# 4. Context compression with LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

compress_prompt = ChatPromptTemplate.from_template(
    """Extract ONLY the sentences relevant to the question from this text.
Remove all irrelevant information. Keep extracted sentences verbatim.

Question: {question}
Text: {text}
Relevant sentences:"""
)

def compress_context(question, docs, max_tokens):
    """Compress each document to keep only relevant sentences."""
    compressed = []
    total = 0
    for doc in docs:
        result = (compress_prompt | llm).invoke({
            "question": question,
            "text": doc.page_content,
        })
        tokens = count_tokens(result.content)
        if total + tokens <= max_tokens:
            compressed.append(result.content)
            total += tokens
    return "\\n\\n".join(compressed)

# Example: compress 5 chunks into budget
original_tokens = sum(count_tokens(d.page_content) for d in docs)
compressed = compress_context("How does attention work?", docs, BUDGET)
print(f"Original: {original_tokens} tokens")
print(f"Compressed: {count_tokens(compressed)} tokens")`}
        id="code-context-optimization"
      />

      <NoteBlock
        type="intuition"
        title="Quality Over Quantity"
        content="Passing 10 mediocre chunks is often worse than passing 3 highly relevant ones. The LLM must parse all provided context, and irrelevant information can confuse it or cause it to hallucinate by mixing up details from different chunks. Think of it as providing a research assistant with a focused brief rather than a stack of loosely related papers."
        id="note-quality-over-quantity"
      />

      <WarningBlock
        title="Compression Can Lose Information"
        content="LLM-based context compression adds latency and cost, and the compressor LLM may accidentally remove information that turns out to be important for answering the question. Always evaluate compressed vs. uncompressed performance on your specific use case before deploying compression in production."
        id="warning-compression"
      />

      <NoteBlock
        type="tip"
        title="Long-Context Models Are Not a Free Pass"
        content="Models with 128K or 1M token context windows can fit more documents, but retrieval quality still matters. Studies show that even long-context models degrade when filled with irrelevant information. Use the extra context for more relevant documents, not more total documents."
        id="note-long-context"
      />
    </div>
  )
}
