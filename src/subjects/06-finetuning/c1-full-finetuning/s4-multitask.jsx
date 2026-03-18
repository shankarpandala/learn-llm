import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function MultitaskLearning() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Multi-Task Learning for Language Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Multi-task learning (MTL) trains a single model on multiple tasks simultaneously,
        allowing tasks to share representations and benefit from each other's training signal.
        Models like T5 and FLAN demonstrated that casting diverse NLP tasks into a unified
        text-to-text format enables powerful multi-task generalization.
      </p>

      <DefinitionBlock
        title="Multi-Task Learning"
        definition="Multi-task learning is a training paradigm where a model is optimized on multiple objectives simultaneously. The combined loss is $\mathcal{L} = \sum_{t=1}^{T} \alpha_t \mathcal{L}_t$ where $\alpha_t$ are task weights and $\mathcal{L}_t$ is the loss for task $t$. Shared parameters learn representations that generalize across tasks."
        id="def-mtl"
      />

      <h2 className="text-2xl font-semibold">The Text-to-Text Framework</h2>
      <p className="text-gray-700 dark:text-gray-300">
        T5 (Raffel et al., 2020) unified all NLP tasks into a text-to-text format. Classification
        becomes generating a label word, translation becomes generating the translated text, and
        summarization becomes generating a summary. This eliminates the need for task-specific heads.
      </p>

      <ExampleBlock
        title="Text-to-Text Task Formatting"
        problem="Show how different NLP tasks are formatted as text-to-text problems."
        steps={[
          { formula: '\\text{Sentiment: } \\texttt{"classify: I love this!"} \\rightarrow \\texttt{"positive"}', explanation: 'Classification is cast as generating the label token.' },
          { formula: '\\text{Translation: } \\texttt{"translate en-fr: Hello"} \\rightarrow \\texttt{"Bonjour"}', explanation: 'Translation uses a language-pair prefix.' },
          { formula: '\\text{Summarize: } \\texttt{"summarize: [article]"} \\rightarrow \\texttt{"[summary]"}', explanation: 'Summarization generates a condensed version.' },
          { formula: '\\text{NLI: } \\texttt{"nli premise: ... hypothesis: ..."} \\rightarrow \\texttt{"entailment"}', explanation: 'Natural Language Inference outputs the relationship label.' },
        ]}
        id="example-t2t"
      />

      <NoteBlock
        type="historical"
        title="From T5 to FLAN"
        content="T5 (2020) showed the power of the text-to-text framework. FLAN (2022) scaled this to 1,836 tasks and demonstrated that instruction-tuned models generalize to unseen tasks far better than models trained on individual tasks. FLAN-T5 and FLAN-PaLM became strong baselines showing that multi-task instruction tuning is a critical step in building capable LLMs."
        id="note-t5-flan"
      />

      <PythonCode
        title="multitask_training.py"
        code={`from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, concatenate_datasets

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load multiple tasks
sst2 = load_dataset("glue", "sst2", split="train[:2000]")
mnli = load_dataset("glue", "mnli", split="train[:2000]")

# Format as text-to-text
def format_sst2(example):
    label = "positive" if example["label"] == 1 else "negative"
    return {"input_text": f"classify sentiment: {example['sentence']}",
            "target_text": label}

def format_mnli(example):
    labels = ["entailment", "neutral", "contradiction"]
    return {"input_text": f"nli premise: {example['premise']} hypothesis: {example['hypothesis']}",
            "target_text": labels[example["label"]]}

sst2_fmt = sst2.map(format_sst2, remove_columns=sst2.column_names)
mnli_fmt = mnli.map(format_mnli, remove_columns=mnli.column_names)

# Combine datasets (interleave for balanced training)
combined = concatenate_datasets([sst2_fmt, mnli_fmt]).shuffle(seed=42)

def tokenize_fn(examples):
    model_inputs = tokenizer(examples["input_text"], max_length=256, truncation=True)
    labels = tokenizer(examples["target_text"], max_length=32, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = combined.map(tokenize_fn, batched=True)

training_args = TrainingArguments(
    output_dir="./multitask-flan-t5",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=3e-5,
    warmup_steps=100,
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
trainer.train()`}
        id="code-multitask"
      />

      <WarningBlock
        title="Task Balancing is Critical"
        content="When tasks have very different dataset sizes, the model tends to overfit on smaller tasks and underfit on larger ones. Strategies include temperature-based sampling (sampling each task with probability proportional to its size raised to a temperature), equal mixing ratios, or dynamic task weighting based on validation performance."
        id="warning-task-balance"
      />

      <NoteBlock
        type="intuition"
        title="Why Multi-Task Helps"
        content="Multi-task learning acts as a regularizer: each task provides a different training signal that prevents the model from overfitting to any single task's idiosyncrasies. Tasks that require similar skills (e.g., NLI and reading comprehension both need reasoning) provide complementary supervision that improves shared representations."
        id="note-why-mtl"
      />
    </div>
  )
}
