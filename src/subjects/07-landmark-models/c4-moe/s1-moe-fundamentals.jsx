import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function MoEFundamentals() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Mixture of Experts: Fundamentals</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Mixture of Experts (MoE) is a technique that decouples the total number of model parameters
        from the computation required per input. By routing each token to only a subset of "expert"
        sub-networks, MoE models can have trillions of parameters while keeping per-token compute
        similar to a much smaller dense model.
      </p>

      <DefinitionBlock
        title="Mixture of Experts Layer"
        definition="An MoE layer replaces the standard feed-forward network (FFN) with $E$ parallel expert networks and a gating (router) network. For input $x$, the router produces weights $g(x) \in \mathbb{R}^E$ and the output is a weighted sum of expert outputs: $\text{MoE}(x) = \sum_{i=1}^{E} g_i(x) \cdot \text{FFN}_i(x)$. With top-$K$ routing, only $K \ll E$ experts are activated per token."
        id="def-moe"
      />

      <h2 className="text-2xl font-semibold">The Router Network</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The router (or gating network) is typically a simple linear layer followed by a softmax
        that produces a probability distribution over experts. Only the top-K experts with the
        highest probabilities are activated, and the output is the weighted sum of their outputs.
      </p>

      <TheoremBlock
        title="Top-K Sparse Gating"
        statement="For a token representation $x \in \mathbb{R}^d$ and router weights $W_g \in \mathbb{R}^{E \times d}$, the sparse gating function selects the top-K experts and renormalizes their weights."
        proof={<BlockMath math="g(x) = \text{Softmax}(\text{TopK}(W_g \cdot x, K))" />}
        corollaries={[
          'Only K out of E experts compute their output, giving a speedup of E/K.',
          'The router itself adds minimal overhead: just a single matrix multiply plus top-K selection.',
        ]}
        id="theorem-topk-gating"
      />

      <ExampleBlock
        title="MoE Compute Savings"
        problem="Compare FLOPs for a dense model vs an MoE model with E=8 experts and K=2."
        steps={[
          { formula: '\\text{Dense FFN}: 2 \\times d \\times d_{ff} \\times 2 = 4 \\cdot d \\cdot d_{ff}', explanation: 'Two linear layers (up-project and down-project), each costing 2*d*d_ff FLOPs.' },
          { formula: '\\text{MoE FFN}: K \\times 4 \\cdot d \\cdot d_{ff} = 2 \\times 4 \\cdot d \\cdot d_{ff}', explanation: 'Only K=2 experts compute. Same per-expert cost but only 2 out of 8 are active.' },
          { formula: '\\text{Speedup} = \\frac{E}{K} = \\frac{8}{2} = 4\\times', explanation: 'MoE uses 4x fewer FLOPs than an equivalently-sized dense model (ignoring router cost).' },
          { formula: '\\text{Total params} = E \\times \\text{expert\\_params} = 8 \\times \\text{FFN}', explanation: 'But the model stores 8x more parameters in memory, enabling more capacity.' },
        ]}
        id="example-moe-flops"
      />

      <PythonCode
        title="moe_layer_implementation.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """A single expert: standard FFN with SwiGLU."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoELayer(nn.Module):
    """Mixture of Experts with top-K routing."""
    def __init__(self, dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(dim, hidden_dim) for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # (B*S, D)

        # Router: compute expert scores
        router_logits = self.gate(x_flat)  # (B*S, E)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (B*S, K)

        # Compute expert outputs (naive loop - real implementations use scatter/gather)
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  # (B*S,)
            weight = top_k_weights[:, i].unsqueeze(-1)  # (B*S, 1)
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += weight[mask] * expert_output

        return output.view(batch_size, seq_len, dim)

# Test
moe = MoELayer(dim=512, hidden_dim=1024, num_experts=8, top_k=2)
x = torch.randn(2, 16, 512)
out = moe(x)
print(f"Input shape: {x.shape}, Output shape: {out.shape}")
print(f"Total params: {sum(p.numel() for p in moe.parameters()):,}")
print(f"Active params per token: ~{sum(p.numel() for p in moe.experts[0].parameters()) * 2 + 512 * 8:,}")`}
        id="code-moe-layer"
      />

      <NoteBlock
        type="intuition"
        title="Why MoE Works"
        content="Different tokens need different types of processing. A code token, a math token, and a natural language token may benefit from different learned representations. MoE lets the model specialize: some experts may focus on syntax, others on semantics, others on reasoning patterns. The router learns to dispatch each token to the most relevant experts."
        id="note-moe-intuition"
      />

      <WarningBlock
        title="Memory vs Compute Tradeoff"
        content="MoE reduces FLOPs but not memory. All expert parameters must be stored in GPU memory even though only K are used per token. A model with 8 experts of 7B each requires 56B parameters in memory but only computes 14B FLOPs (top-2). This makes MoE models memory-bound rather than compute-bound."
        id="warning-moe-memory"
      />
    </div>
  )
}
