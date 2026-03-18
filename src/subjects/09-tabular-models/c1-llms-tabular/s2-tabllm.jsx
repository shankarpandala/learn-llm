import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function TabLLM() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">TabLLM: Few-Shot Tabular Classification with LLMs</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        TabLLM demonstrates that large language models can perform tabular classification by
        converting each row into a natural language description. This text-serialization approach
        enables few-shot learning on structured data, often matching or exceeding traditional
        ML models like gradient-boosted trees when labeled data is scarce.
      </p>

      <DefinitionBlock
        title="TabLLM"
        definition="TabLLM is a framework that serializes each tabular data row into a natural language sentence, then uses an LLM for classification. Given a row $x = (x_1, \ldots, x_n)$ with column names $(c_1, \ldots, c_n)$, the serialization produces a text $t(x)$ and the model predicts $P(y \mid t(x))$."
        notation="x = feature vector, c_i = column name, t(x) = text serialization, y = class label"
        id="def-tabllm"
      />

      <h2 className="text-2xl font-semibold">Serialization Templates</h2>
      <p className="text-gray-700 dark:text-gray-300">
        TabLLM explores multiple serialization strategies to convert tabular rows into text.
        The text template approach constructs natural language sentences from column-value pairs,
        which leverages the language model's understanding of natural language semantics.
      </p>

      <PythonCode
        title="tabllm_serialization.py"
        code={`# TabLLM serialization templates for a single data row
row = {
    "age": 35, "income": 75000, "education": "Bachelor's",
    "marital_status": "Married", "occupation": "Engineer"
}
label_col = "high_earner"  # binary classification target

# Template 1: Text template (natural language)
def text_template(row):
    parts = [f"The {k.replace('_', ' ')} is {v}" for k, v in row.items()]
    return ". ".join(parts) + "."

# Template 2: List template
def list_template(row):
    parts = [f"- {k}: {v}" for k, v in row.items()]
    return "\\n".join(parts)

# Template 3: Key-value serialization
def kv_template(row):
    return " | ".join(f"{k}={v}" for k, v in row.items())

print("=== Text Template ===")
print(text_template(row))
# The age is 35. The income is 75000. The education is Bachelor's...

print("\\n=== List Template ===")
print(list_template(row))

print("\\n=== Key-Value Template ===")
print(kv_template(row))

# Few-shot prompt construction
def build_prompt(train_examples, test_row, template_fn):
    prompt = "Classify whether the person is a high earner.\\n\\n"
    for ex_row, ex_label in train_examples:
        prompt += template_fn(ex_row) + f"\\nLabel: {ex_label}\\n\\n"
    prompt += template_fn(test_row) + "\\nLabel:"
    return prompt

print("\\n=== Few-Shot Prompt ===")
print(build_prompt(
    [(row, "Yes")],
    {"age": 22, "income": 35000, "education": "High School",
     "marital_status": "Single", "occupation": "Retail"},
    text_template
))`}
        id="code-serialization"
      />

      <ExampleBlock
        title="TabLLM Classification Pipeline"
        problem="Classify a loan applicant as low/high risk using TabLLM with 4 labeled examples."
        steps={[
          { formula: '\\text{Serialize each row } x_i \\to t(x_i)', explanation: 'Convert the tabular row into a natural language description using a template.' },
          { formula: '\\text{Construct prompt: } [t(x_1), y_1, \\ldots, t(x_4), y_4, t(x_{\\text{test}})]', explanation: 'Build a few-shot prompt with serialized examples and their labels.' },
          { formula: 'P(y \\mid \\text{prompt}) = \\text{LLM}(\\text{prompt})', explanation: 'The LLM generates the predicted label by completing the prompt.' },
          { formula: '\\hat{y} = \\arg\\max_{y} P(y \\mid \\text{prompt})', explanation: 'Select the class with the highest probability as the prediction.' },
        ]}
        id="example-pipeline"
      />

      <NoteBlock
        type="intuition"
        title="Why Text Serialization Helps"
        content="LLMs have been pretrained on vast text corpora containing descriptions of people, products, and events. When a tabular row is serialized as 'The age is 35 and the income is 75000', the model can leverage its world knowledge about typical income levels for different ages, effectively performing implicit feature engineering."
        id="note-intuition"
      />

      <PythonCode
        title="tabllm_with_t5.py"
        code={`# TabLLM-style classification using a T5 model
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Serialize a row as natural language
def serialize_row(row):
    return ". ".join(f"The {k} is {v}" for k, v in row.items())

# Build classification prompt
row = {"age": "45", "job": "technician", "balance": "1200"}
text = f"Classify subscription: {serialize_row(row)}. Will subscribe?"

inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)
outputs = model.generate(**inputs, max_new_tokens=5)
prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prediction: {prediction}")

# For proper TabLLM: fine-tune T5 on serialized tabular data
# model.train()
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# for row_text, label_text in train_loader:
#     inputs = tokenizer(row_text, return_tensors="pt", padding=True)
#     labels = tokenizer(label_text, return_tensors="pt", padding=True)
#     loss = model(**inputs, labels=labels.input_ids).loss
#     loss.backward()
#     optimizer.step()`}
        id="code-t5"
      />

      <WarningBlock
        title="When TabLLM Falls Short"
        content="TabLLM struggles with high-dimensional numeric data where feature interactions matter more than semantic meaning. For datasets with 50+ numeric columns (e.g., sensor readings), gradient-boosted trees like XGBoost still dominate. TabLLM is most effective when column names carry semantic meaning and labeled data is limited (under 100 examples)."
        id="warning-limitations"
      />

      <NoteBlock
        type="note"
        title="Key Results from TabLLM (Hegselmann et al., 2023)"
        content="On 9 benchmark datasets, TabLLM with T0 (11B parameters) matched or exceeded XGBoost performance in 4-shot and 8-shot settings. The text template serialization consistently outperformed list and CSV formats. Fine-tuning on serialized data further improved results, especially with LoRA adapters to keep training efficient."
        id="note-results"
      />
    </div>
  )
}
