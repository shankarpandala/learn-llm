import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function DpoTraining() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">DPO Training with TRL</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Direct Preference Optimization (DPO) aligns language models with human preferences without
        training a separate reward model. It directly optimizes the policy using pairs of preferred
        and rejected responses, making it simpler and more stable than traditional RLHF.
      </p>

      <DefinitionBlock
        title="DPO Loss"
        definition="Given a prompt $x$, chosen response $y_w$, and rejected response $y_l$, DPO optimizes: $L_{\\text{DPO}} = -\\log \\sigma\\left(\\beta \\left[\\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)}\\right]\\right)$ where $\\pi_{\\text{ref}}$ is the reference (SFT) model and $\\beta$ controls the deviation penalty."
        notation="L_{\text{DPO}} = -\log \sigma(\beta[\log r_w - \log r_l])"
        id="def-dpo"
      />

      <h2 className="text-2xl font-semibold">DPO Dataset Format</h2>

      <PythonCode
        title="dpo_dataset.py"
        code={`from datasets import Dataset

# DPO requires: prompt, chosen response, rejected response
dpo_data = [
    {
        "prompt": "Explain recursion in programming.",
        "chosen": "Recursion is when a function calls itself to solve smaller subproblems. For example, calculating factorial: fact(n) = n * fact(n-1), with base case fact(0) = 1. Each call reduces the problem until hitting the base case, then results build back up.",
        "rejected": "Recursion is a programming concept. It means something calls itself. It's used in many algorithms.",
    },
    {
        "prompt": "What causes seasons on Earth?",
        "chosen": "Seasons are caused by Earth's 23.5-degree axial tilt. As Earth orbits the Sun, different hemispheres receive more direct sunlight at different times. When the Northern Hemisphere tilts toward the Sun, it experiences summer while the Southern Hemisphere has winter.",
        "rejected": "Seasons happen because Earth gets closer to or farther from the Sun during its orbit.",
    },
]

# Or load an existing preference dataset
from datasets import load_dataset
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned",
                       split="train[:5000]")
print(f"Columns: {dataset.column_names}")
# Typically: prompt, chosen, rejected (may need reformatting)`}
        id="code-dpo-dataset"
      />

      <h2 className="text-2xl font-semibold">Training with DPOTrainer</h2>

      <PythonCode
        title="dpo_training.py"
        code={`from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
import torch

# Load the SFT model (already finetuned with SFTTrainer)
model_name = "your-sft-model-path"  # or HF model id

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# DPO uses the same model as both policy and reference
# The reference model is created internally by DPOTrainer
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

dpo_config = DPOConfig(
    output_dir="./dpo-output",
    beta=0.1,                        # KL penalty strength
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-5,              # Lower LR than SFT
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    logging_steps=10,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    # DPO-specific
    loss_type="sigmoid",             # Standard DPO loss
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()`}
        id="code-dpo-training"
      />

      <ExampleBlock
        title="DPO Hyperparameter Selection"
        problem="How to choose the key DPO hyperparameters?"
        steps={[
          { formula: '\\beta = 0.1\\text{-}0.5', explanation: 'Controls how much the model can deviate from the reference. Lower = more deviation. Start with 0.1.' },
          { formula: '\\text{LR} = 5 \\times 10^{-6}\\text{ to }5 \\times 10^{-5}', explanation: 'DPO needs lower learning rates than SFT to avoid forgetting.' },
          { formula: '\\text{Epochs} = 1\\text{-}3', explanation: 'DPO is sensitive to overfitting. Usually 1 epoch suffices.' },
          { formula: '\\text{loss\\_type} = \\text{"sigmoid"}', explanation: 'Standard DPO. Alternatives: "hinge" (more robust), "ipo" (identity preference optimization).' },
        ]}
        id="example-dpo-hyperparams"
      />

      <NoteBlock
        type="intuition"
        title="Why DPO Over RLHF?"
        content="Traditional RLHF requires training a separate reward model and then optimizing with PPO -- a complex, unstable process. DPO shows that the optimal policy has a closed-form solution given preferences, eliminating the need for RL entirely. The result is simpler code, faster training, and comparable quality."
        id="note-dpo-vs-rlhf"
      />

      <WarningBlock
        title="DPO Data Quality"
        content="DPO is very sensitive to data quality. The chosen/rejected pairs must have meaningful quality differences. If the rejected responses are nearly as good as the chosen ones, DPO learns little. Similarly, if rejected responses are absurdly bad, the signal is too easy and the model does not improve on realistic edge cases."
        id="warning-dpo-data"
      />
    </div>
  )
}
