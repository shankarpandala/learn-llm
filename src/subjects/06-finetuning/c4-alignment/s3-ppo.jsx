import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function PPOForRLHF() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">PPO for RLHF</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Proximal Policy Optimization (PPO) is the reinforcement learning algorithm most commonly
        used in RLHF. It optimizes the language model policy to maximize rewards from the reward
        model while constraining updates to be small, preventing the policy from diverging too
        far from the reference model. PPO balances exploration (finding better responses) with
        stability (not destroying existing capabilities).
      </p>

      <DefinitionBlock
        title="PPO for Language Models"
        definition="In the RLHF context, PPO treats text generation as a sequential decision process. The policy $\pi_\theta$ generates tokens autoregressively. The reward is the reward model score of the full response minus a KL penalty: $R(x,y) = r_\phi(x,y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$. PPO updates the policy using clipped surrogate objectives to ensure stable optimization."
        id="def-ppo-rlhf"
      />

      <TheoremBlock
        title="PPO Clipped Objective"
        statement="The PPO clipped surrogate objective is:
$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( \rho_t(\theta) \hat{A}_t, \; \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
where $\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio and $\hat{A}_t$ is the estimated advantage. The clipping with $\epsilon$ (typically 0.2) prevents large policy updates."
        id="thm-ppo-clip"
      />

      <h2 className="text-2xl font-semibold">PPO in the RLHF Loop</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The PPO training loop for RLHF generates responses, scores them with the reward model,
        computes advantages using a value head, and updates the policy with the clipped objective.
        A KL penalty against the reference (SFT) model prevents mode collapse.
      </p>

      <ExampleBlock
        title="PPO Training Step"
        problem="Describe one iteration of the PPO training loop for RLHF."
        steps={[
          { formula: 'y \\sim \\pi_{\\theta_{\\text{old}}}(\\cdot | x)', explanation: 'Sample responses from the current policy for a batch of prompts.' },
          { formula: 'r = r_\\phi(x, y) - \\beta \\, \\text{KL}(\\pi_\\theta \\| \\pi_{\\text{ref}})', explanation: 'Compute reward minus KL penalty to form the total reward signal.' },
          { formula: '\\hat{A}_t = \\text{GAE}(r, V_{\\psi})', explanation: 'Estimate per-token advantages using Generalized Advantage Estimation with the value head.' },
          { formula: '\\theta \\leftarrow \\theta + \\alpha \\nabla_\\theta \\mathcal{L}^{\\text{CLIP}}(\\theta)', explanation: 'Update policy parameters using the clipped surrogate objective over multiple mini-batches.' },
        ]}
        id="example-ppo-step"
      />

      <PythonCode
        title="ppo_rlhf_training.py"
        code={`from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load SFT model with value head for PPO
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "./sft-model", torch_dtype=torch.bfloat16, device_map="auto"
)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "./sft-model", torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./sft-model")
tokenizer.pad_token = tokenizer.eos_token

# Load reward model
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "./reward-model", torch_dtype=torch.bfloat16, device_map="auto"
)
reward_tokenizer = AutoTokenizer.from_pretrained("./reward-model")

# PPO configuration
ppo_config = PPOConfig(
    batch_size=128,
    mini_batch_size=32,
    gradient_accumulation_steps=4,
    learning_rate=1.41e-5,
    init_kl_coeff=0.2,         # Initial KL penalty coefficient beta
    target=6.0,                # Target KL divergence
    adap_kl_ctrl=True,         # Adaptively adjust beta
    cliprange=0.2,             # PPO clip epsilon
    cliprange_value=0.2,       # Value function clip range
    vf_coef=0.1,               # Value loss coefficient
    ppo_epochs=4,              # PPO epochs per batch
    max_grad_norm=0.5,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

def get_reward(prompt_ids, response_ids):
    """Score prompt-response pairs with reward model."""
    full_ids = torch.cat([prompt_ids, response_ids], dim=-1)
    with torch.no_grad():
        rewards = reward_model(full_ids).logits.squeeze(-1)
    return rewards

# Training loop
for epoch in range(3):
    for batch in prompt_dataloader:
        query_tensors = [tokenizer.encode(q, return_tensors="pt").squeeze()
                        for q in batch["prompt"]]

        # Generate responses
        response_tensors = ppo_trainer.generate(
            query_tensors, max_new_tokens=256,
            temperature=0.7, top_p=0.9,
        )

        # Compute rewards
        rewards = [get_reward(q, r) for q, r in zip(query_tensors, response_tensors)]

        # PPO step: computes advantages, clips ratios, updates policy + value head
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        print(f"reward/mean: {stats['ppo/mean_scores']:.3f}, "
              f"kl: {stats['objective/kl']:.3f}")`}
        id="code-ppo-rlhf"
      />

      <WarningBlock
        title="PPO Instability"
        content="PPO for RLHF is notoriously difficult to stabilize. Common failure modes include: reward hacking (exploiting reward model weaknesses), mode collapse (generating repetitive responses), KL divergence explosion, and training instability from the value head. Careful monitoring of reward, KL, entropy, and response quality metrics is essential."
        id="warning-instability"
      />

      <NoteBlock
        type="tip"
        title="Practical PPO Tips"
        content="Key stabilization techniques: (1) Use adaptive KL penalty that increases beta when KL exceeds the target, (2) Clip both policy and value function, (3) Use multiple PPO epochs (2-4) per batch for sample efficiency, (4) Normalize advantages within each mini-batch, (5) Use gradient clipping (max_grad_norm = 0.5-1.0), (6) Start with a small learning rate and warm up."
        id="note-ppo-tips"
      />
    </div>
  )
}
