import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function SwitchTransformer() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Switch Transformer: Simplified MoE</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The Switch Transformer (Fedus et al., 2022) simplified MoE by routing each token to
        exactly one expert (top-1 routing), rather than combining outputs from multiple experts.
        This simplified routing, reduced communication costs, and scaled to 1.6 trillion parameters
        while maintaining training stability through careful auxiliary loss design.
      </p>

      <DefinitionBlock
        title="Switch Routing"
        definition="A top-1 routing mechanism where each token is sent to exactly one expert: $\text{expert}(x) = \arg\max_i (W_g \cdot x)_i$. The output is scaled by the router probability: $y = g_i(x) \cdot \text{FFN}_i(x)$ where $g_i(x) = \text{softmax}(W_g \cdot x)_i$ for the selected expert $i$. This is the simplest possible sparse routing strategy."
        id="def-switch-routing"
      />

      <h2 className="text-2xl font-semibold">Key Design Decisions</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The Switch Transformer made several crucial simplifications over prior MoE work: top-1
        routing (vs top-2 in GShard), expert capacity factor to handle imbalanced routing, and
        selective precision training where the router operates in float32 while experts use bfloat16.
      </p>

      <TheoremBlock
        title="Load Balancing Loss"
        statement="To prevent expert collapse (all tokens routed to a few experts), the Switch Transformer adds an auxiliary loss that encourages uniform routing across experts."
        proof={<BlockMath math="\mathcal{L}_{\text{balance}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot p_i" />}
        corollaries={[
          'f_i is the fraction of tokens routed to expert i (discrete, non-differentiable).',
          'p_i is the average router probability assigned to expert i (differentiable).',
          'The product f_i * p_i is minimized when both are uniform at 1/E, encouraging balanced routing.',
          'Alpha is typically set to 0.01 to avoid overwhelming the primary language modeling loss.',
        ]}
        id="theorem-balance-loss"
      />

      <ExampleBlock
        title="Expert Capacity"
        problem="Calculate the expert capacity for a batch of 32 tokens with 8 experts and capacity factor 1.25."
        steps={[
          { formula: '\\text{tokens\\_per\\_expert} = \\frac{T}{E} = \\frac{32}{8} = 4', explanation: 'With uniform routing, each expert would get exactly 4 tokens.' },
          { formula: '\\text{capacity} = \\lceil CF \\times \\frac{T}{E} \\rceil = \\lceil 1.25 \\times 4 \\rceil = 5', explanation: 'The capacity factor (CF=1.25) adds a 25% buffer for routing imbalance.' },
          { formula: '\\text{overflow tokens are dropped}', explanation: 'If more than 5 tokens route to one expert, extras pass through unchanged (skip the FFN).' },
        ]}
        id="example-capacity"
      />

      <h2 className="text-2xl font-semibold">Scaling Results</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The Switch Transformer scaled from 32 to 2048 experts, reaching 1.6T parameters. Despite
        having 1000x more parameters than T5-Base, the Switch-Base model used the same compute
        per token and achieved significant speedups: 7x faster pre-training to reach the same
        loss as T5-Base.
      </p>

      <PythonCode
        title="switch_transformer_concept.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class SwitchLayer(nn.Module):
    """Switch Transformer MoE layer with top-1 routing and load balancing."""
    def __init__(self, dim, hidden_dim, num_experts=8, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        T = x_flat.shape[0]

        # Router (use float32 for stability)
        router_logits = self.gate(x_flat.float())
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-1 routing
        expert_indices = router_probs.argmax(dim=-1)  # (T,)
        expert_weights = router_probs.gather(1, expert_indices.unsqueeze(-1)).squeeze(-1)

        # Compute load balancing loss
        f = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            f[i] = (expert_indices == i).float().mean()
        p = router_probs.mean(dim=0)
        balance_loss = self.num_experts * (f * p).sum()

        # Expert capacity
        capacity = int(self.capacity_factor * T / self.num_experts)

        # Dispatch to experts
        output = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask = (expert_indices == i)
            if mask.sum() == 0:
                continue
            # Apply capacity limit
            token_indices = mask.nonzero(as_tuple=True)[0][:capacity]
            expert_input = x_flat[token_indices]
            expert_output = self.experts[i](expert_input)
            output[token_indices] = expert_weights[token_indices].unsqueeze(-1) * expert_output

        return output.view(B, S, D), balance_loss

# Test
switch = SwitchLayer(dim=512, hidden_dim=2048, num_experts=8)
x = torch.randn(2, 32, 512)
out, loss = switch(x)
print(f"Output shape: {out.shape}")
print(f"Balance loss: {loss.item():.4f}")

# Check routing distribution
with torch.no_grad():
    logits = switch.gate(x.view(-1, 512))
    indices = logits.argmax(dim=-1)
    for i in range(8):
        count = (indices == i).sum().item()
        print(f"  Expert {i}: {count} tokens ({count/64*100:.0f}%)")`}
        id="code-switch"
      />

      <NoteBlock
        type="historical"
        title="From GShard to Switch"
        content="GShard (Lepikhin et al., 2021) used top-2 routing and was the first MoE to scale to 600B parameters for machine translation. The Switch Transformer showed that top-1 routing was not only simpler but also more efficient, as it halved the communication cost in distributed settings while maintaining quality."
        id="note-gshard-history"
      />

      <WarningBlock
        title="Token Dropping"
        content="When experts overflow their capacity, excess tokens skip the expert layer entirely. During training, this acts as implicit regularization. During inference with capacity_factor=1.0, up to ~10% of tokens can be dropped in poorly balanced models, degrading output quality. Always monitor the fraction of dropped tokens."
        id="warning-token-dropping"
      />
    </div>
  )
}
