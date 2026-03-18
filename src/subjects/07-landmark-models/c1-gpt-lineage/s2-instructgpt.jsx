import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function InstructGPT() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">InstructGPT and RLHF</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        InstructGPT (Ouyang et al., 2022) bridged the gap between raw language model capability and
        human-aligned behavior. By combining supervised fine-tuning with reinforcement learning from
        human feedback (RLHF), OpenAI created a 1.3B parameter model that was preferred over the
        175B GPT-3 by human evaluators.
      </p>

      <DefinitionBlock
        title="Reinforcement Learning from Human Feedback (RLHF)"
        definition="A training paradigm that aligns language models with human preferences using three stages: (1) supervised fine-tuning on demonstrations, (2) training a reward model on human comparisons, and (3) optimizing the policy with PPO against the reward model while constraining divergence from the SFT model via a KL penalty: $\max_\pi \mathbb{E}_{x \sim D}[\mathbb{E}_{y \sim \pi(\cdot|x)}[R(x,y)] - \beta \, \text{KL}(\pi \| \pi_{\text{SFT}})]$."
        id="def-rlhf"
      />

      <h2 className="text-2xl font-semibold">Stage 1: Supervised Fine-Tuning (SFT)</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Human labelers wrote high-quality responses to a set of prompts. A pre-trained GPT-3
        model was fine-tuned on these demonstrations using standard cross-entropy loss. This created
        the SFT model, which already showed improved instruction following over the base model.
      </p>

      <h2 className="text-2xl font-semibold">Stage 2: Reward Model Training</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Labelers ranked multiple model outputs for the same prompt. These rankings trained a reward
        model (RM) to predict human preferences. The RM takes a prompt-response pair and outputs a
        scalar reward score.
      </p>

      <TheoremBlock
        title="Bradley-Terry Preference Model"
        statement="Given two responses $y_w$ (preferred) and $y_l$ (dispreferred) to prompt $x$, the reward model is trained by maximizing the log-likelihood of the observed preference under the Bradley-Terry model."
        proof={<BlockMath math="\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \sigma\left(R_\theta(x, y_w) - R_\theta(x, y_l)\right)\right]" />}
        id="theorem-bradley-terry"
      />

      <h2 className="text-2xl font-semibold">Stage 3: PPO Optimization</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The SFT model was further optimized using Proximal Policy Optimization (PPO) to maximize
        the reward model's scores while staying close to the SFT distribution. The KL divergence
        penalty prevents reward hacking, where the model exploits loopholes in the reward model.
      </p>

      <ExampleBlock
        title="RLHF Training Pipeline"
        problem="Outline the data requirements for each RLHF stage in InstructGPT."
        steps={[
          { formula: '\\text{SFT}: \\sim 13{,}000 \\text{ demonstrations}', explanation: 'Human labelers wrote ideal responses to sampled prompts from the OpenAI API.' },
          { formula: '\\text{RM}: \\sim 33{,}000 \\text{ comparisons}', explanation: 'Labelers ranked 4-9 outputs per prompt, yielding pairwise comparison data.' },
          { formula: '\\text{PPO}: \\sim 31{,}000 \\text{ prompts}', explanation: 'Additional prompts without labels, optimized against the frozen reward model.' },
        ]}
        id="example-rlhf-data"
      />

      <PythonCode
        title="reward_model_concept.py"
        code={`import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """Simplified reward model for RLHF."""
    def __init__(self, base_model_name="gpt2"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        # Use the last token's hidden state as the sequence representation
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)

# Bradley-Terry preference loss
def preference_loss(reward_chosen, reward_rejected):
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()

# Example usage
rm = RewardModel("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

chosen = tokenizer("Explain gravity: Gravity is the force...", return_tensors="pt", padding=True)
rejected = tokenizer("Explain gravity: I don't know lol", return_tensors="pt", padding=True)

r_chosen = rm(chosen.input_ids, chosen.attention_mask)
r_rejected = rm(rejected.input_ids, rejected.attention_mask)

loss = preference_loss(r_chosen, r_rejected)
print(f"Reward chosen: {r_chosen.item():.4f}")
print(f"Reward rejected: {r_rejected.item():.4f}")
print(f"Preference loss: {loss.item():.4f}")`}
        id="code-reward-model"
      />

      <NoteBlock
        type="note"
        title="InstructGPT vs ChatGPT"
        content="ChatGPT (November 2022) used the same RLHF methodology as InstructGPT but was built on top of a more capable base model (GPT-3.5). The conversational format and dialogue-specific training data made it the fastest-growing consumer application in history, reaching 100 million users in two months."
        id="note-chatgpt"
      />

      <WarningBlock
        title="Reward Hacking"
        content="Without the KL penalty, the policy can find adversarial responses that achieve high reward model scores without actually being helpful. For example, the model might learn to produce verbose, confident-sounding but incorrect answers that the reward model rates highly. Careful calibration of the KL coefficient beta is essential."
        id="warning-reward-hacking"
      />
    </div>
  )
}
