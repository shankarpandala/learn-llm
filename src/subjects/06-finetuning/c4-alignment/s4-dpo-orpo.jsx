import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function DPOAndORPO() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">DPO and ORPO: RL-Free Alignment</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Direct Preference Optimization (DPO) and Odds Ratio Preference Optimization (ORPO)
        bypass the complexity of training a separate reward model and running RL. They directly
        optimize the language model on preference data using modified supervised learning objectives.
        DPO has become the dominant alignment method due to its simplicity and stability.
      </p>

      <DefinitionBlock
        title="Direct Preference Optimization (DPO)"
        definition="DPO (Rafailov et al., 2023) derives a closed-form solution to the RLHF objective by reparameterizing the reward function in terms of the optimal policy. Instead of training a reward model and running PPO, DPO directly optimizes the policy on preference pairs using a classification-like loss that implicitly defines the reward."
        id="def-dpo"
      />

      <TheoremBlock
        title="DPO Loss Function"
        statement="The DPO loss is derived by substituting the optimal policy from the KL-constrained reward maximization into the Bradley-Terry preference model:
$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$
where $\pi_{\text{ref}}$ is the reference (SFT) model and $\beta$ controls the deviation from the reference. The implicit reward is $r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$."
        proof="Starting from the RLHF objective $\max_\pi \mathbb{E}[r(x,y)] - \beta \text{KL}(\pi \| \pi_{\text{ref}})$, the optimal policy is $\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)$. Rearranging gives $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$. Substituting into the Bradley-Terry model and canceling the partition function $Z(x)$ (which appears in both chosen and rejected) yields the DPO loss."
        id="thm-dpo-loss"
      />

      <ExampleBlock
        title="DPO vs. RLHF Comparison"
        problem="Compare the computational requirements of DPO and RLHF."
        steps={[
          { formula: '\\text{RLHF: 3 models (policy + value + reward) in memory}', explanation: 'PPO requires the policy, value head, reference model, and reward model simultaneously.' },
          { formula: '\\text{DPO: 2 models (policy + reference) in memory}', explanation: 'DPO only needs the policy being trained and a frozen reference model.' },
          { formula: '\\text{RLHF: online generation + reward scoring + PPO}', explanation: 'Each PPO step requires generating responses, scoring them, and running multiple optimization epochs.' },
          { formula: '\\text{DPO: single forward-backward pass on preference pairs}', explanation: 'DPO is a standard supervised learning loop, making it much simpler to implement and debug.' },
        ]}
        id="example-dpo-vs-rlhf"
      />

      <PythonCode
        title="dpo_training.py"
        code={`from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset
from peft import LoraConfig

# Load SFT model as starting point
model_name = "./sft-model"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Reference model (frozen copy of SFT model)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

# Load preference dataset
# Each example needs: 'prompt', 'chosen', 'rejected'
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:5000]")

def format_preference(example):
    return {
        "prompt": example["chosen"].split("\\n\\nAssistant:")[0],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

dataset = dataset.map(format_preference)

# Optional LoRA for memory efficiency
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# DPO training configuration
dpo_config = DPOConfig(
    output_dir="./dpo-model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,           # Very low LR for DPO
    beta=0.1,                     # KL penalty strength
    max_length=512,
    max_prompt_length=256,
    bf16=True,
    logging_steps=10,
    gradient_checkpointing=True,
    loss_type="sigmoid",          # Standard DPO loss
)

# Train with DPO
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,          # None if using LoRA (implicit reference)
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model("./dpo-model-final")`}
        id="code-dpo-training"
      />

      <h2 className="text-2xl font-semibold">ORPO: Odds Ratio Preference Optimization</h2>
      <p className="text-gray-700 dark:text-gray-300">
        ORPO (Hong et al., 2024) eliminates the need for a reference model entirely by
        combining SFT and preference optimization into a single loss. It uses the odds ratio
        of generating chosen vs. rejected responses as the preference signal.
      </p>

      <NoteBlock
        type="note"
        title="ORPO Loss"
        content="The ORPO objective combines an SFT loss on the chosen response with an odds-ratio penalty: L_ORPO = L_SFT(y_w) + lambda * L_OR where L_OR = -log sigma(log odds(y_w) - log odds(y_l)) and odds(y) = p(y|x) / (1 - p(y|x)). This means ORPO does not require a reference model and can start from a base model rather than an SFT model."
        id="note-orpo"
      />

      <WarningBlock
        title="DPO Hyperparameter Sensitivity"
        content="DPO is sensitive to the beta parameter and learning rate. Beta too high makes the model stay too close to the reference (underfitting preferences). Beta too low allows the model to deviate too far (reward hacking without an explicit reward model). Start with beta = 0.1 and LR = 5e-7, and adjust based on the chosen/rejected reward margin during training."
        id="warning-dpo-hparams"
      />

      <NoteBlock
        type="tip"
        title="When to Use DPO vs. RLHF"
        content="DPO is recommended as the default for most practitioners: it is simpler, more stable, and produces comparable results. Use RLHF (PPO) when you need online learning (generating and scoring new responses during training), when the preference landscape is complex, or when you have a strong reward model you want to leverage. ORPO is best when you want to skip the SFT stage entirely and train directly from a base model."
        id="note-when-to-use"
      />
    </div>
  )
}
