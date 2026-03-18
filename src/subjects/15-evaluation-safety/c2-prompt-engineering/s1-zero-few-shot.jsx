import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ZeroFewShot() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Zero-Shot and Few-Shot Prompting</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Zero-shot and few-shot prompting are foundational techniques that leverage a pretrained
        model's in-context learning ability. By providing zero or a few examples in the prompt,
        we can steer model behavior without any parameter updates.
      </p>

      <DefinitionBlock
        title="Zero-Shot Prompting"
        definition="Querying a model with only a task description and no examples. The model must rely entirely on its pretraining knowledge to understand and execute the task."
        id="def-zero-shot"
      />

      <DefinitionBlock
        title="Few-Shot Prompting"
        definition="Providing $k$ input-output demonstration examples in the prompt before the actual query. The model uses in-context learning to infer the task pattern from the demonstrations. Performance typically follows $P(k) \approx P_\infty - c \cdot k^{-\alpha}$, improving with more shots."
        id="def-few-shot"
      />

      <h2 className="text-2xl font-semibold">In-Context Learning</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Few-shot prompting works because of in-context learning (ICL). Given demonstrations{' '}
        <InlineMath math="\{(x_1, y_1), \ldots, (x_k, y_k)\}" /> and a new input{' '}
        <InlineMath math="x_{k+1}" />, the model computes:
      </p>
      <BlockMath math="P(y_{k+1} | x_1, y_1, \ldots, x_k, y_k, x_{k+1})" />
      <p className="text-gray-700 dark:text-gray-300">
        without updating any weights. Research suggests ICL implicitly performs gradient
        descent in the model's hidden representations.
      </p>

      <ExampleBlock
        title="Zero-Shot vs Few-Shot Comparison"
        problem="Classify movie reviews as positive or negative."
        steps={[
          { formula: '\\text{Zero-shot: Classify the sentiment: \\"Great movie!\\"} \\to \\text{Positive}', explanation: 'Direct instruction with no examples. Works for simple, well-understood tasks.' },
          { formula: '\\text{1-shot: \\"Terrible film\\" → Negative. \\"Great movie!\\" →}', explanation: 'One example establishes the expected format and label space.' },
          { formula: '\\text{3-shot: 3 diverse examples covering edge cases}', explanation: 'More examples help with ambiguous cases and consistent formatting.' },
        ]}
        id="example-zero-few"
      />

      <PythonCode
        title="zero_few_shot_prompting.py"
        code={`from openai import OpenAI

client = OpenAI()

def zero_shot_classify(text):
    """Zero-shot sentiment classification."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Classify the sentiment of this text as 'positive' or 'negative'.\\n\\nText: {text}\\nSentiment:"
        }],
        temperature=0,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()

def few_shot_classify(text, examples):
    """Few-shot sentiment classification with demonstrations."""
    demo_str = "\\n".join(
        f"Text: {ex['text']}\\nSentiment: {ex['label']}" for ex in examples
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Classify the sentiment of each text as 'positive' or 'negative'.\\n\\n{demo_str}\\n\\nText: {text}\\nSentiment:"
        }],
        temperature=0,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()

# Few-shot examples
examples = [
    {"text": "This movie was absolutely wonderful!", "label": "positive"},
    {"text": "Waste of time, terrible acting.", "label": "negative"},
    {"text": "A masterpiece of modern cinema.", "label": "positive"},
]

# Compare approaches on ambiguous input
test_text = "It was okay, nothing special but not bad either."
print(f"Zero-shot: {zero_shot_classify(test_text)}")
print(f"Few-shot:  {few_shot_classify(test_text, examples)}")

# Chain-of-thought zero-shot (most powerful zero-shot variant)
def zero_shot_cot(question):
    """Zero-shot chain-of-thought prompting."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"{question}\\n\\nLet's think step by step."
        }],
        temperature=0,
    )
    return response.choices[0].message.content

result = zero_shot_cot("If a store has 3 shelves with 8 books each, and 5 books are removed, how many remain?")
print(result)`}
        id="code-zero-few-shot"
      />

      <NoteBlock
        type="intuition"
        title="Why Few-Shot Examples Matter"
        content="Examples serve multiple purposes: they define the label space, establish the output format, demonstrate the difficulty level expected, and prime the model's attention toward relevant features. Even random labels in examples improve formatting consistency, though correct labels significantly boost accuracy."
        id="note-why-fewshot"
      />

      <WarningBlock
        title="Example Selection and Order Matter"
        content="Few-shot performance is sensitive to which examples are chosen and their order. Selecting diverse, representative examples outperforms random selection. Placing similar examples closer to the query can help. Recent work on retrieval-augmented few-shot selects examples dynamically based on similarity to the query."
        id="warning-selection"
      />

      <NoteBlock
        type="historical"
        title="The GPT-3 Moment"
        content="Brown et al. (2020) demonstrated that GPT-3's few-shot performance rivaled fine-tuned models on many tasks, establishing in-context learning as a viable alternative to fine-tuning. This paper coined the terms 'zero-shot', 'one-shot', and 'few-shot' in the context of LLM prompting."
        id="note-gpt3-history"
      />
    </div>
  )
}
