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
        After evaluating your finetuned model, you will inevitably find areas for improvement. The
        key decision is whether to retrain from scratch, continue training, adjust hyperparameters,
        or fix the data. This section provides a decision framework for iteration.
      </p>

      <DefinitionBlock
        title="Continued Training"
        definition="Continued training loads a previously saved checkpoint and resumes training with new or additional data. This preserves existing learning while adding new capabilities. It avoids the cost of retraining from scratch but risks overfitting if the new data is small."
        id="def-continued-training"
      />

      <ExampleBlock
        title="Decision Framework"
        problem="When should you retrain from scratch vs continue training vs adjust other factors?"
        steps={[
          { formula: '\\text{Wrong format/template} \\Rightarrow \\text{Fix data, retrain}', explanation: 'Fundamental data issues require a complete retrain. No shortcut.' },
          { formula: '\\text{Mostly good, weak on specific task} \\Rightarrow \\text{Add data, continue}', explanation: 'Add 200-500 targeted examples and continue training for 0.5 epochs.' },
          { formula: '\\text{Overfitting (memorization)} \\Rightarrow \\text{Reduce epochs, retrain}', explanation: 'Lower epochs, increase dropout, or reduce rank. Then retrain.' },
          { formula: '\\text{Good quality, wrong tone/style} \\Rightarrow \\text{Adjust data, retrain}', explanation: 'Rewrite a subset of examples with the desired style and retrain.' },
          { formula: '\\text{Model too verbose/terse} \\Rightarrow \\text{Adjust data lengths}', explanation: 'The model learns average response length from training data. Adjust accordingly.' },
        ]}
        id="example-decision"
      />

      <PythonCode
        title="continue_training.py"
        code={`from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Resume from a checkpoint
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=new_dataset,       # Can be new/additional data
    args=SFTConfig(
        output_dir="./continued-training",
        num_train_epochs=1,           # Fewer epochs for continuation
        learning_rate=5e-5,           # Lower LR for stability
        warmup_ratio=0.05,            # Brief warmup
        save_strategy="steps",
        save_steps=100,
    ),
)

# Resume from checkpoint
trainer.train(resume_from_checkpoint="./previous-output/checkpoint-500")

# Or: load the final model and train with new data
# This is "continued finetuning" - adding new capabilities
# model = AutoModelForCausalLM.from_pretrained("./previous-output")
# trainer = SFTTrainer(model=model, train_dataset=additional_data, ...)`}
        id="code-continue-training"
      />

      <PythonCode
        title="iteration_strategy.py"
        code={`def plan_next_iteration(eval_results):
    """Suggest next steps based on evaluation results."""

    suggestions = []

    # Check benchmark regression
    if eval_results.get("mmlu_delta", 0) < -0.03:
        suggestions.append({
            "priority": "HIGH",
            "action": "Reduce training - catastrophic forgetting detected",
            "fix": "Lower epochs (1), lower LR, or reduce LoRA rank",
        })

    # Check task performance
    if eval_results.get("task_accuracy", 1.0) < 0.7:
        suggestions.append({
            "priority": "HIGH",
            "action": "Check data quality and formatting",
            "fix": "Review 50 random training examples manually",
        })

    # Check for repetition issues
    if eval_results.get("repetition_rate", 0) > 0.1:
        suggestions.append({
            "priority": "MEDIUM",
            "action": "Model is generating repetitive text",
            "fix": "Reduce epochs, add diverse data, or use repetition penalty",
        })

    # Check response quality
    if eval_results.get("judge_score", 5) < 3.5:
        suggestions.append({
            "priority": "MEDIUM",
            "action": "Response quality below threshold",
            "fix": "Improve training data quality, consider DPO alignment",
        })

    # Print action plan
    print("=== Iteration Plan ===")
    for i, s in enumerate(suggestions, 1):
        print(f"\\n{i}. [{s['priority']}] {s['action']}")
        print(f"   Fix: {s['fix']}")

    if not suggestions:
        print("Model looks good! Consider:")
        print("- Running full benchmark suite")
        print("- DPO alignment for preference tuning")
        print("- Deployment preparation (quantization, serving)")

    return suggestions

# Example
plan_next_iteration({
    "mmlu_delta": -0.01,
    "task_accuracy": 0.85,
    "repetition_rate": 0.02,
    "judge_score": 4.1,
})`}
        id="code-iteration"
      />

      <NoteBlock
        type="tip"
        title="Keep an Experiment Log"
        content="Maintain a spreadsheet or WandB project tracking every training run: dataset version, hyperparameters, training time, key metrics, and qualitative observations. This prevents repeating failed experiments and helps identify which changes produced improvements."
        id="note-experiment-log"
      />

      <WarningBlock
        title="Do Not Chain Too Many LoRA Adapters"
        content="Training LoRA on top of a merged LoRA on top of another LoRA compounds approximation errors. If you need to iterate multiple times, merge and retrain from the merged checkpoint rather than stacking adapters. Alternatively, retrain a fresh LoRA on your accumulated dataset."
        id="warning-stacking"
      />

      <NoteBlock
        type="note"
        title="The Data Flywheel"
        content="The most effective iteration strategy is the data flywheel: deploy your model, collect real user interactions, identify failure cases, add corrective examples, and retrain. Each cycle improves the model on real-world usage patterns that synthetic data cannot fully capture."
        id="note-flywheel"
      />
    </div>
  )
}
