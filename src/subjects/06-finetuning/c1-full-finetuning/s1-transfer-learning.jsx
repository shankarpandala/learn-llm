import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function TransferLearning() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Transfer Learning: The Pretrain-Finetune Paradigm</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Transfer learning is the foundational idea behind modern LLM development. A model is first
        pretrained on a massive unsupervised corpus to learn general language representations, then
        finetuned on a smaller task-specific dataset. This two-stage paradigm dramatically reduces
        the data and compute required for downstream tasks.
      </p>

      <DefinitionBlock
        title="Transfer Learning"
        definition="Transfer learning is the technique of reusing a model trained on one task (the source task) as the starting point for a model on a different task (the target task). In NLP, the source task is typically language modeling on a large corpus, and the target task is a specific downstream application."
        id="def-transfer-learning"
      />

      <h2 className="text-2xl font-semibold">The Two-Stage Pipeline</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Pretraining captures syntactic structure, world knowledge, and reasoning patterns from
        billions of tokens. Finetuning adapts these general capabilities to a narrow task using
        orders of magnitude less data. The pretrained weights serve as an informed initialization
        rather than random weights.
      </p>

      <ExampleBlock
        title="Pretraining vs. Finetuning Costs"
        problem="Compare the resources needed for pretraining versus finetuning a 7B parameter model."
        steps={[
          { formula: '\\text{Pretraining: } \\sim 1\\text{T tokens}, \\sim 100\\text{K GPU-hours}', explanation: 'Pretraining requires massive compute, data, and weeks of training on large clusters.' },
          { formula: '\\text{Finetuning: } \\sim 10\\text{K-100K examples}, \\sim 10\\text{-100 GPU-hours}', explanation: 'Finetuning uses a tiny fraction of the pretraining budget and can often run on a single node.' },
          { formula: '\\text{Ratio} \\approx 1000\\times \\text{ less compute}', explanation: 'This massive efficiency gain is why transfer learning transformed NLP.' },
        ]}
        id="example-cost-comparison"
      />

      <NoteBlock
        type="historical"
        title="History of Transfer Learning in NLP"
        content="Transfer learning in NLP gained momentum with ULMFiT (Howard & Ruder, 2018), which introduced discriminative fine-tuning and gradual unfreezing. BERT (Devlin et al., 2018) and GPT (Radford et al., 2018) cemented the pretrain-finetune paradigm. GPT-2 and GPT-3 later showed that sufficiently large pretrained models could perform tasks zero-shot."
        id="note-history"
      />

      <h2 className="text-2xl font-semibold">Finetuning Strategies</h2>
      <p className="text-gray-700 dark:text-gray-300">
        There are several strategies for finetuning: updating all parameters (full finetuning),
        freezing lower layers and only training upper layers, or using learning rate schedules
        that treat different layers differently (discriminative finetuning).
      </p>

      <PythonCode
        title="full_finetuning_hf.py"
        code={`from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

# Load pretrained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

# Load downstream dataset
dataset = load_dataset("imdb")

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized = dataset.map(tokenize_fn, batched=True)

# Finetuning configuration
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,          # Much smaller LR than pretraining
    weight_decay=0.01,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

# Finetune: all parameters are updated
trainer.train()`}
        id="code-full-finetuning"
      />

      <WarningBlock
        title="Learning Rate Sensitivity"
        content="Finetuning requires a much smaller learning rate than pretraining (typically 1e-5 to 5e-5). Using a pretraining-scale learning rate (e.g., 1e-3) will destroy the pretrained representations in just a few steps, a phenomenon known as catastrophic forgetting."
        id="warning-lr"
      />

      <NoteBlock
        type="intuition"
        title="Why Transfer Learning Works"
        content="Lower layers of neural networks learn general features (syntax, word relationships) while upper layers learn task-specific patterns. By preserving the lower layers' knowledge, finetuning lets the model build on a rich foundation rather than learning from scratch. This is analogous to how a medical student builds on general biology knowledge."
        id="note-intuition"
      />
    </div>
  )
}
