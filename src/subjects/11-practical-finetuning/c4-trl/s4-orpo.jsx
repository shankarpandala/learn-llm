import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function OrpoTraining() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">ORPO: Odds Ratio Preference Optimization</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        ORPO combines supervised finetuning and preference alignment into a single training stage.
        Unlike DPO, which requires a separate SFT step first, ORPO modifies the SFT loss to
        incorporate preference information directly, simplifying the training pipeline.
      </p>

      <DefinitionBlock
        title="ORPO Loss"
        definition="ORPO adds an odds ratio penalty to the standard SFT loss: $L_{\\text{ORPO}} = L_{\\text{SFT}}(y_w) + \\lambda \\cdot L_{\\text{OR}}$ where $L_{\\text{OR}} = -\\log \\sigma\\left(\\log \\frac{\\text{odds}_\\theta(y_w|x)}{\\text{odds}_\\theta(y_l|x)}\\right)$ and $\\text{odds}(y|x) = \\frac{P(y|x)}{1 - P(y|x)}$."
        notation="L_{ORPO} = L_{SFT}(y_w) + \lambda \cdot L_{OR}"
        id="def-orpo"
      />

      <ExampleBlock
        title="ORPO vs DPO Pipeline"
        problem="Compare the training pipeline for DPO vs ORPO."
        steps={[
          { formula: '\\text{DPO: SFT} \\rightarrow \\text{DPO (2 stages)}', explanation: 'DPO requires a separate SFT training run first, then DPO alignment.' },
          { formula: '\\text{ORPO: Single stage}', explanation: 'ORPO handles both SFT and alignment simultaneously, halving training time.' },
          { formula: '\\text{DPO needs reference model}', explanation: 'DPO computes KL divergence against a frozen reference model (doubles memory).' },
          { formula: '\\text{ORPO: no reference model}', explanation: 'ORPO uses odds ratios, eliminating the need for a reference model.' },
        ]}
        id="example-orpo-vs-dpo"
      />

      <PythonCode
        title="orpo_training.py"
        code={`from trl import ORPOTrainer, ORPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from datasets import load_dataset
import torch

# Load base model (NOT an SFT model - ORPO does SFT internally)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",  # Base model, not Instruct
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
tokenizer.pad_token = tokenizer.eos_token

# LoRA config
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# Load preference dataset (same format as DPO)
dataset = load_dataset(
    "argilla/ultrafeedback-binarized-preferences-cleaned",
    split="train[:10000]"
)

# ORPO configuration
orpo_config = ORPOConfig(
    output_dir="./orpo-output",
    beta=0.1,                        # Odds ratio weight (lambda)
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=8e-6,              # ORPO uses lower LR
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()

# Save the model
trainer.save_model("./orpo-final")`}
        id="code-orpo-training"
      />

      <PythonCode
        title="orpo_metrics.py"
        code={`# Key metrics to monitor during ORPO training
# These are logged automatically by ORPOTrainer

metrics_guide = {
    "train/loss": "Total ORPO loss (SFT + lambda * OR loss)",
    "train/sft_loss": "Standard next-token prediction loss on chosen",
    "train/odds_ratio_loss": "Preference alignment loss component",
    "train/log_odds_ratio": "Log odds ratio between chosen and rejected",
    "train/log_odds_chosen": "Log odds of chosen responses",
    "train/log_odds_rejected": "Log odds of rejected responses",
    "train/reward_margin": "Difference in implicit reward (chosen - rejected)",
}

for metric, description in metrics_guide.items():
    print(f"{metric}:")
    print(f"  {description}")

# Healthy training indicators:
# - sft_loss decreasing smoothly
# - log_odds_ratio increasing (chosen becoming more likely vs rejected)
# - reward_margin positive and increasing
# - odds_ratio_loss decreasing`}
        id="code-orpo-metrics"
      />

      <NoteBlock
        type="tip"
        title="When to Use ORPO"
        content="ORPO is ideal when: (1) you are starting from a base model (not instruction-tuned), (2) you want to minimize training stages, (3) GPU memory is limited (no reference model needed). ORPO is particularly effective when combined with high-quality preference data like UltraFeedback."
        id="note-when-orpo"
      />

      <WarningBlock
        title="ORPO Learning Rate Sensitivity"
        content="ORPO is more sensitive to the learning rate than DPO. Too high and the model collapses; too low and it does not align. Start with 5e-6 to 1e-5 and adjust based on the reward margin metric. If reward_margin is not increasing after 100 steps, try a higher LR."
        id="warning-orpo-lr"
      />

      <NoteBlock
        type="note"
        title="SimPO and KTO Alternatives"
        content="SimPO (Simple Preference Optimization) and KTO (Kahneman-Tversky Optimization) are newer alternatives. SimPO uses length-normalized log probabilities. KTO works with binary feedback (good/bad) without needing paired preferences. Both are available in TRL."
        id="note-alternatives"
      />
    </div>
  )
}
