import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function LoadBalancing() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Expert Load Balancing</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Load balancing is the central challenge in MoE training. Without proper balancing mechanisms,
        models suffer from expert collapse (most tokens routed to a few experts) or routing
        oscillation. Multiple strategies have been developed, from auxiliary losses to
        auxiliary-loss-free approaches.
      </p>

      <DefinitionBlock
        title="Expert Collapse"
        definition="A failure mode in MoE training where the router converges to sending most or all tokens to a small subset of experts, leaving the rest untrained. This creates a positive feedback loop: popular experts improve, increasing their routing probability further. The result is wasted capacity equivalent to a much smaller dense model."
        id="def-expert-collapse"
      />

      <h2 className="text-2xl font-semibold">Auxiliary Loss Approaches</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The most common approach adds a differentiable loss term that penalizes imbalanced routing.
        The Switch Transformer's balance loss computes the product of token fractions and router
        probabilities per expert, which is minimized when routing is uniform.
      </p>

      <TheoremBlock
        title="Z-Loss for Router Stability"
        statement="The Z-loss (Zoph et al., 2022, ST-MoE) penalizes large router logits to prevent the router from becoming too confident, which helps maintain exploration of all experts during training."
        proof={<BlockMath math="\mathcal{L}_z = \frac{1}{T} \sum_{t=1}^{T} \left(\log \sum_{i=1}^{E} e^{z_i^{(t)}}\right)^2" />}
        corollaries={[
          'z_i^(t) are the raw router logits before softmax for token t.',
          'The Z-loss is zero when all logits are equal (uniform routing) and grows as logits diverge.',
          'Used in conjunction with the balance loss, not as a replacement.',
        ]}
        id="theorem-zloss"
      />

      <h2 className="text-2xl font-semibold">Auxiliary-Loss-Free Balancing</h2>
      <p className="text-gray-700 dark:text-gray-300">
        DeepSeek-V3 introduced an auxiliary-loss-free approach where each expert maintains a bias
        term that is adjusted based on the expert's current load. Overloaded experts get their bias
        decreased (making them less likely to be selected), while underloaded experts get increased
        bias. This achieves better balancing without the gradient interference of auxiliary losses.
      </p>

      <ExampleBlock
        title="Balancing Strategies Comparison"
        problem="Compare three load balancing approaches used in major MoE models."
        steps={[
          { formula: '\\mathcal{L}_{\\text{Switch}} = \\alpha E \\sum_i f_i \\cdot p_i', explanation: 'Switch Transformer: product of token fraction and router probability. Alpha ~0.01. Simple but can interfere with language modeling loss.' },
          { formula: '\\mathcal{L}_{\\text{ST-MoE}} = \\mathcal{L}_{\\text{Switch}} + \\beta \\mathcal{L}_z', explanation: 'ST-MoE adds Z-loss for logit regularization. Better stability for very large models.' },
          { formula: 'b_i \\leftarrow b_i + \\gamma \\cdot (\\bar{f} - f_i)', explanation: 'DeepSeek-V3: bias adjustment. No gradient interference. Expert bias updated proportionally to load deficit.' },
        ]}
        id="example-balancing"
      />

      <PythonCode
        title="load_balancing_strategies.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedMoERouter(nn.Module):
    """MoE router with multiple load balancing strategies."""
    def __init__(self, dim, num_experts=8, top_k=2, balance_type="switch"):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_type = balance_type
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # For auxiliary-loss-free balancing (DeepSeek-V3 style)
        if balance_type == "bias":
            self.expert_bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)
            self.bias_update_rate = 0.001

    def forward(self, x):
        # x: (T, D)
        logits = self.gate(x.float())  # (T, E)

        # Apply expert bias if using bias-based balancing
        if self.balance_type == "bias":
            logits = logits + self.expert_bias.unsqueeze(0)

        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Renormalize top-k weights
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute auxiliary loss
        aux_loss = self._compute_balance_loss(logits, probs, top_k_indices)

        # Update bias (for bias-based method, during training)
        if self.balance_type == "bias" and self.training:
            self._update_bias(top_k_indices)

        return top_k_weights, top_k_indices, aux_loss

    def _compute_balance_loss(self, logits, probs, indices):
        T = probs.shape[0]
        E = self.num_experts

        if self.balance_type == "switch":
            # Token fraction per expert
            f = torch.zeros(E, device=probs.device)
            for k in range(self.top_k):
                for i in range(E):
                    f[i] += (indices[:, k] == i).float().sum() / T
            f = f / self.top_k
            p = probs.mean(dim=0)
            return E * (f * p).sum()

        elif self.balance_type == "zloss":
            # Z-loss: penalize large router logits
            z_loss = torch.logsumexp(logits, dim=-1).square().mean()
            # Plus standard balance loss
            f = torch.zeros(E, device=probs.device)
            for k in range(self.top_k):
                for i in range(E):
                    f[i] += (indices[:, k] == i).float().sum() / T
            f = f / self.top_k
            p = probs.mean(dim=0)
            return E * (f * p).sum() + 0.001 * z_loss

        return torch.tensor(0.0, device=probs.device)

    def _update_bias(self, indices):
        with torch.no_grad():
            counts = torch.zeros(self.num_experts, device=indices.device)
            for k in range(self.top_k):
                counts.scatter_add_(0, indices[:, k], torch.ones_like(indices[:, k], dtype=torch.float))
            avg = counts.mean()
            self.expert_bias += self.bias_update_rate * (avg - counts)

# Compare strategies
x = torch.randn(256, 512)
for strategy in ["switch", "zloss", "bias"]:
    router = BalancedMoERouter(512, num_experts=8, top_k=2, balance_type=strategy)
    router.train()
    weights, indices, loss = router(x)
    counts = torch.zeros(8)
    for k in range(2):
        for i in range(8):
            counts[i] += (indices[:, k] == i).sum()
    cv = counts.std() / counts.mean()  # coefficient of variation
    print(f"{strategy:8s} | balance_loss={loss.item():.4f} | CV={cv.item():.3f} | "
          f"counts={counts.int().tolist()}")`}
        id="code-balancing"
      />

      <NoteBlock
        type="intuition"
        title="Why Auxiliary Losses Are Problematic"
        content="Auxiliary balance losses add gradients to the router that conflict with the language modeling objective. The router must simultaneously route tokens to the best expert AND maintain balance -- these goals often conflict. The DeepSeek-V3 bias approach elegantly sidesteps this by moving balancing out of the gradient computation entirely, letting the router optimize purely for routing quality."
        id="note-aux-loss-problem"
      />

      <WarningBlock
        title="Balance vs Specialization Tradeoff"
        content="Perfect load balance means every expert processes the same number of tokens, but this may not be optimal. Some token types (e.g., code, math) may genuinely benefit from more expert capacity. Over-aggressive balancing can prevent useful specialization. Modern approaches aim for approximately balanced (within 10-20%) rather than perfectly balanced routing."
        id="warning-balance-tradeoff"
      />
    </div>
  )
}
