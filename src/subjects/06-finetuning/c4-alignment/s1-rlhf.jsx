import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function RLHF() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">RLHF: Reinforcement Learning from Human Feedback</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        RLHF is a training methodology that aligns language models with human preferences
        by using reinforcement learning. After supervised fine-tuning, RLHF trains the model
        to generate responses that humans prefer, going beyond what can be achieved with
        static demonstration data alone. This is the technique that transformed GPT-3 into
        ChatGPT.
      </p>

      <DefinitionBlock
        title="RLHF Pipeline"
        definition="RLHF consists of three stages: (1) Supervised Fine-Tuning (SFT) on demonstrations to create a policy $\pi_{\text{SFT}}$, (2) Training a reward model $r_\phi(x, y)$ on human preference comparisons, and (3) Optimizing the policy via RL (typically PPO) to maximize the reward while staying close to $\pi_{\text{SFT}}$: $\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \, \text{KL}(\pi_\theta \| \pi_{\text{SFT}}) \right]$"
        id="def-rlhf"
      />

      <h2 className="text-2xl font-semibold">The Three Stages</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Each stage of RLHF serves a distinct purpose and requires different data, objectives,
        and computational resources.
      </p>

      <ExampleBlock
        title="RLHF Pipeline Stages"
        problem="Describe the data requirements and objectives for each RLHF stage."
        steps={[
          { formula: '\\text{Stage 1 - SFT: } \\mathcal{D}_{\\text{demo}} = \\{(x_i, y_i^*)\\}', explanation: 'Collect expert demonstrations. Train with cross-entropy loss on 10K-100K examples.' },
          { formula: '\\text{Stage 2 - RM: } \\mathcal{D}_{\\text{pref}} = \\{(x_i, y_i^w, y_i^l)\\}', explanation: 'Collect human preference pairs (chosen vs. rejected). Train reward model on 50K-500K comparisons.' },
          { formula: '\\text{Stage 3 - RL: } \\max_\\theta \\mathbb{E}[r_\\phi(x,y)] - \\beta \\text{KL}(\\pi_\\theta \\| \\pi_{\\text{ref}})', explanation: 'Optimize policy with PPO to maximize reward while constraining divergence from the SFT model.' },
        ]}
        id="example-rlhf-stages"
      />

      <PythonCode
        title="rlhf_pipeline_overview.py"
        code={`from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
import torch

# Stage 1: SFT (done separately, see s3-sft.jsx)
sft_model_path = "./sft-llama2-final"

# Stage 2: Reward Model (done separately, see s2-reward-modeling.jsx)
reward_model_path = "./reward-model-final"

# Stage 3: PPO Training
# Load the SFT model as the starting policy
model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model_path)
tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
tokenizer.pad_token = tokenizer.eos_token

# Reference model (frozen copy of SFT model for KL penalty)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model_path)

# Load reward model for scoring
reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path)

# PPO configuration
ppo_config = PPOConfig(
    batch_size=64,
    mini_batch_size=16,
    learning_rate=1.41e-5,
    log_with="wandb",
    kl_penalty="kl",           # KL divergence type
    init_kl_coeff=0.2,         # Initial beta for KL penalty
    target=6.0,                # Target KL divergence
    adap_kl_ctrl=True,         # Adaptive KL coefficient
)

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# Training loop sketch
prompts = load_dataset("your/prompt-dataset", split="train")
for batch in prompts:
    # Generate responses from current policy
    query_tensors = tokenizer(batch["prompt"], return_tensors="pt").input_ids
    response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=256)

    # Score with reward model
    rewards = compute_rewards(reward_model, query_tensors, response_tensors)

    # PPO update step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    print(f"Reward mean: {stats['ppo/mean_scores']:.3f}")`}
        id="code-rlhf-pipeline"
      />

      <NoteBlock
        type="historical"
        title="Origins of RLHF"
        content="RLHF was first applied to language models by Stiennon et al. (2020) for summarization. Ouyang et al. (2022) scaled it to InstructGPT, showing RLHF made a 1.3B model preferred over the 175B GPT-3. Anthropic's Constitutional AI (2022) extended RLHF with AI-generated feedback. The technique became widely known when ChatGPT launched in November 2022."
        id="note-history"
      />

      <WarningBlock
        title="RLHF is Complex and Unstable"
        content="RLHF involves training three separate models (SFT, reward, policy) with complex interactions. PPO is notoriously sensitive to hyperparameters, and reward hacking (the model exploiting reward model weaknesses) is a persistent problem. Small errors in the reward model get amplified during RL optimization. This complexity motivated simpler alternatives like DPO."
        id="warning-complexity"
      />

      <NoteBlock
        type="intuition"
        title="Why RL Over More Supervised Learning?"
        content="SFT can only teach the model to imitate demonstrations. But human preferences encode information that is hard to demonstrate: 'response A is better than response B because it is more nuanced.' RL allows the model to explore the response space and learn from comparative feedback, discovering response strategies that might not appear in any demonstration dataset."
        id="note-why-rl"
      />
    </div>
  )
}
