import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function RetrainVsAdjust() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">When to Retrain vs Adjust</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        After evaluating your fine-tuned model, you face a decision: retrain from scratch
        with different data or hyperparameters, or make adjustments to the existing model.
        This section provides a decision framework and practical strategies for iterating
        on fine-tuned models.
      </p>

      <DefinitionBlock
        title="Iterative Fine-tuning"
        definition="Iterative fine-tuning involves making incremental improvements to a model through successive rounds of training, evaluation, and adjustment. The key trade-off is between the cost of retraining (time, compute) and the expected improvement from each iteration strategy."
        id="def-iterative"
      />

      <ExampleBlock
        title="Decision Framework"
        problem="Should you retrain from scratch or adjust the existing model?"
        steps={[
          { formula: '\\text{Data quality issues} \\Rightarrow \\text{Fix data, retrain}', explanation: 'If training data has errors, duplicates, or wrong formats, fix the data and retrain from the base model.' },
          { formula: '\\text{Underfitting (high loss)} \\Rightarrow \\text{Adjust hyperparams}', explanation: 'Increase LoRA rank, learning rate, or epochs. Can continue training from checkpoint.' },
          { formula: '\\text{Overfitting (gap between train/val loss)} \\Rightarrow \\text{Adjust}', explanation: 'Reduce epochs, increase dropout, or add regularization. Resume from an earlier checkpoint.' },
          { formula: '\\text{Wrong task behavior} \\Rightarrow \\text{Fix data, retrain}', explanation: 'If the model learned the wrong behavior, the data or format needs changing.' },
          { formula: '\\text{Needs more capability} \\Rightarrow \\text{Add data, continue}', explanation: 'If the model is good but needs more coverage, add examples and continue training.' },
        ]}
        id="example-decision-framework"
      />

      <PythonCode
        title="continue_training.py"
        code={`from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel, get_peft_model, LoraConfig
from trl import SFTTrainer
from datasets import load_dataset

# Strategy 1: Continue training from checkpoint
def continue_from_checkpoint(checkpoint_dir, new_data_path, extra_epochs=1):
    """Resume training from a saved checkpoint with additional data."""
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    dataset = load_dataset("json", data_files=new_data_path, split="train")

    training_args = TrainingArguments(
        output_dir="./continued-training",
        num_train_epochs=extra_epochs,
        learning_rate=5e-5,  # lower LR for continued training
        per_device_train_batch_size=4,
        save_strategy="steps",
        save_steps=100,
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=dataset, args=training_args,
    )
    trainer.train()
    return model

# Strategy 2: Merge and re-adapt
def merge_and_readapt(base_model_path, adapter_path, new_data_path):
    """Merge existing LoRA, then train a new adapter."""
    import torch

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()

    new_lora = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
    )
    model = get_peft_model(model, new_lora)
    print(f"New trainable params: {model.print_trainable_parameters()}")
    return model

# Strategy 3: Selective data refinement
def refine_dataset(original_data, model_path, threshold=0.8):
    """Remove low-quality samples based on model confidence."""
    from transformers import pipeline

    pipe = pipeline("text-generation", model=model_path, device_map="auto")
    refined = []

    for sample in original_data:
        refined.append(sample)  # Add filtering logic based on loss

    print(f"Refined: {len(refined)}/{len(original_data)} samples kept")
    return refined`}
        id="code-continue-training"
      />

      <PythonCode
        title="hyperparameter_iteration.py"
        code={`# Quick reference for hyperparameter adjustments

ITERATION_PLAYBOOK = {
    "model_repeats_itself": {
        "diagnosis": "Learning rate too high or too many epochs",
        "adjustments": [
            "Reduce learning_rate by 2-5x",
            "Reduce num_train_epochs",
            "Add repetition_penalty=1.1 at inference",
            "Check for duplicate training samples",
        ]
    },
    "model_ignores_finetuning": {
        "diagnosis": "Learning rate too low or too few epochs",
        "adjustments": [
            "Increase learning_rate by 2-5x",
            "Increase LoRA rank (r=16 -> r=32 -> r=64)",
            "Add more target modules (include MLP layers)",
            "Increase num_train_epochs",
        ]
    },
    "model_forgets_general_knowledge": {
        "diagnosis": "Catastrophic forgetting from over-training",
        "adjustments": [
            "Reduce epochs (try 1 epoch first)",
            "Lower learning_rate",
            "Mix in 10-20% general instruction data",
            "Use lower LoRA rank to limit capacity",
        ]
    },
    "high_train_loss_wont_decrease": {
        "diagnosis": "Data format mismatch or model capacity issue",
        "adjustments": [
            "Verify chat template matches training format exactly",
            "Check that labels are not masked incorrectly",
            "Increase LoRA rank or add target modules",
            "Try a larger base model",
        ]
    },
}

for issue, info in ITERATION_PLAYBOOK.items():
    print(f"\\n{'='*60}")
    print(f"Issue: {issue}")
    print(f"Diagnosis: {info['diagnosis']}")
    print("Adjustments:")
    for adj in info["adjustments"]:
        print(f"  - {adj}")`}
        id="code-playbook"
      />

      <NoteBlock
        type="intuition"
        title="The 80/20 Rule of Fine-tuning"
        content="80% of improvement comes from data quality, 20% from hyperparameters. If your first attempt does not work well, spend your time improving the training data before running grid searches over learning rates and LoRA configurations."
        id="note-8020"
      />

      <WarningBlock
        title="Do Not Stack Too Many LoRA Adapters"
        content="While you can merge one LoRA and train another on top, stacking more than 2-3 rounds of fine-tuning often degrades quality. Each round adds noise. If you find yourself iterating many times, consider retraining from the base model with a curated combined dataset."
        id="warning-stacking"
      />

      <NoteBlock
        type="tip"
        title="Keep a Training Log"
        content="Document every training run: dataset version, hyperparameters, evaluation metrics, and qualitative observations. This log becomes invaluable when deciding what to try next. Tools like Weights & Biases (wandb) automate much of this tracking."
        id="note-training-log"
      />
    </div>
  )
}
