import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function SupervisedFineTuning() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Supervised Fine-Tuning (SFT)</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Supervised Fine-Tuning is the process of training a pretrained language model on
        curated instruction-response pairs using standard cross-entropy loss. SFT is the
        first step after pretraining in the modern LLM alignment pipeline, transforming a
        base model that merely predicts next tokens into one that helpfully follows instructions.
      </p>

      <DefinitionBlock
        title="Supervised Fine-Tuning (SFT)"
        definition="SFT optimizes a pretrained model $\pi_\theta$ on a dataset of demonstrations $\mathcal{D} = \{(x_i, y_i)\}$ where $x_i$ is an instruction and $y_i$ is the desired response. The objective is to minimize the negative log-likelihood: $\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t | x, y_{<t}) \right]$ where the loss is computed only over response tokens $y$."
        id="def-sft"
      />

      <h2 className="text-2xl font-semibold">The SFT Training Pipeline</h2>
      <p className="text-gray-700 dark:text-gray-300">
        A complete SFT pipeline involves data preparation, formatting with chat templates,
        tokenization with proper label masking, and training with carefully chosen hyperparameters.
        The TRL library from HuggingFace provides a streamlined SFTTrainer for this workflow.
      </p>

      <ExampleBlock
        title="SFT Hyperparameter Guidelines"
        problem="What are typical hyperparameters for SFT of a 7B model?"
        steps={[
          { formula: '\\text{Learning rate: } 1 \\times 10^{-5} \\text{ to } 2 \\times 10^{-5}', explanation: 'Lower than pretraining LR to preserve pretrained knowledge.' },
          { formula: '\\text{Epochs: } 1 \\text{ to } 3', explanation: 'SFT datasets are small; more epochs risk overfitting and losing generalization.' },
          { formula: '\\text{Batch size: effective } 128 \\text{ via gradient accumulation}', explanation: 'Large effective batches stabilize training on diverse instruction types.' },
          { formula: '\\text{Warmup: } 3\\text{-}10\\% \\text{ of total steps}', explanation: 'Gradual warmup prevents early instability from the randomly-initialized output shift.' },
        ]}
        id="example-sft-hparams"
      />

      <PythonCode
        title="sft_with_trl.py"
        code={`from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load instruction dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Format into chat-style prompts
def format_instruction(example):
    if example["input"]:
        text = (f"<|im_start|>user\\n{example['instruction']}\\n"
                f"Input: {example['input']}<|im_end|>\\n"
                f"<|im_start|>assistant\\n{example['output']}<|im_end|>")
    else:
        text = (f"<|im_start|>user\\n{example['instruction']}<|im_end|>\\n"
                f"<|im_start|>assistant\\n{example['output']}<|im_end|>")
    return {"text": text}

dataset = dataset.map(format_instruction)

# Optional: use LoRA for parameter efficiency
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# Configure SFT training
sft_config = SFTConfig(
    output_dir="./sft-llama2",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=32,    # Effective batch = 128
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_seq_length=2048,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    bf16=True,
    gradient_checkpointing=True,       # Save memory
    dataset_text_field="text",
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,           # Pass LoRA config for PEFT training
)

# Train
trainer.train()

# Save the model
trainer.save_model("./sft-llama2-final")`}
        id="code-sft-trl"
      />

      <NoteBlock
        type="intuition"
        title="SFT as Behavior Cloning"
        content="SFT is essentially behavior cloning: the model learns to imitate the demonstrations in the dataset. This means the model can only be as good as the data. If the demonstrations contain errors, the model learns to reproduce those errors. This is why data quality is paramount and why SFT alone is insufficient: the model needs further alignment (RLHF/DPO) to go beyond mimicking demonstrations."
        id="note-behavior-cloning"
      />

      <WarningBlock
        title="Overfitting on Small Datasets"
        content="SFT datasets are typically small (1K-100K examples) compared to pretraining data (trillions of tokens). Training for too many epochs causes the model to memorize responses verbatim rather than learning to generalize. Monitor the validation loss carefully; it often starts increasing after 1-2 epochs. Use dropout, weight decay, and early stopping."
        id="warning-overfitting"
      />

      <NoteBlock
        type="tip"
        title="Packing for Efficiency"
        content="Short examples waste compute due to padding. Packing concatenates multiple examples into a single sequence (separated by EOS tokens) to fill the context window. TRL's SFTTrainer supports this with the packing=True option. This can speed up training by 2-5x on datasets with variable-length examples."
        id="note-packing"
      />
    </div>
  )
}
