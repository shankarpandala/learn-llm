import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function JambaHybrid() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Jamba: Hybrid Attention-Mamba Architecture</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Jamba combines Transformer attention layers with Mamba SSM layers in a single model,
        getting the best of both worlds: attention's precise token retrieval and Mamba's
        efficient long-range processing. It further incorporates Mixture-of-Experts (MoE)
        to increase model capacity without proportionally increasing compute.
      </p>

      <DefinitionBlock
        title="Jamba Architecture"
        definition="Jamba interleaves Mamba layers, attention layers, and MoE-FFN layers in a repeating pattern. A typical block ratio is 7:1 (7 Mamba layers per 1 attention layer). Each attention layer uses grouped-query attention (GQA), and the FFN in select layers is replaced with a MoE layer containing $E$ experts with top-$k$ routing."
        notation="For a 52B total parameter model with 12B active: $E=16$ experts, $k=2$ active per token. Active FLOPs match a 12B dense model while having 52B total capacity."
        id="def-jamba"
      />

      <ExampleBlock
        title="Jamba Layer Configuration"
        problem="Design a Jamba model with 32 layers, 7:1 Mamba-to-attention ratio, and MoE on every other Mamba layer."
        steps={[
          {
            formula: '\\text{Attention layers: } \\{8, 16, 24, 32\\} \\quad (4 \\text{ layers})',
            explanation: 'Place attention every 8th layer for global token interaction.'
          },
          {
            formula: '\\text{Mamba layers: remaining 28 layers}',
            explanation: 'Mamba handles the bulk of processing with linear complexity.'
          },
          {
            formula: '\\text{MoE-FFN: layers } \\{2, 4, 6, 10, 12, 14, ...\\}',
            explanation: 'Every other Mamba layer uses MoE-FFN instead of dense FFN for capacity.'
          },
          {
            formula: '\\text{KV cache: only 4 layers} \\rightarrow 87.5\\% \\text{ memory reduction}',
            explanation: 'Only attention layers need KV cache. Mamba uses fixed-size recurrent state.'
          }
        ]}
        id="example-jamba-config"
      />

      <PythonCode
        title="jamba_hybrid_model.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMambaLayer(nn.Module):
    """Simplified Mamba layer for the hybrid model."""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(d_model, d_model, 4, padding=3, groups=d_model)
        self.ssm_proj = nn.Linear(d_model, d_state * 2 + 1)
        self.out_proj = nn.Linear(d_model, d_model)
        self.d_state = d_state

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        x_conv = self.conv(x_inner.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        x_conv = F.silu(x_conv)
        out = x_conv * F.silu(z)
        return residual + self.out_proj(out)

class SimpleAttentionLayer(nn.Module):
    """Standard attention layer with GQA."""
    def __init__(self, d_model, n_heads=8, n_kv_heads=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.norm(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        return x + self.ffn(self.norm2(x))

class MoEFFN(nn.Module):
    """Mixture-of-Experts FFN layer."""
    def __init__(self, d_model, n_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            ) for _ in range(n_experts)
        ])
        self.gate = nn.Linear(d_model, n_experts)
        self.top_k = top_k

    def forward(self, x):
        # Route tokens to top-k experts
        scores = self.gate(x)
        topk_vals, topk_idx = scores.topk(self.top_k, dim=-1)
        weights = F.softmax(topk_vals, dim=-1)

        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x[mask])
                weight = weights[topk_idx == i].unsqueeze(-1)
                output[mask] += expert_out * weight[:expert_out.shape[0]]
        return output

class JambaModel(nn.Module):
    """Hybrid Jamba architecture."""
    def __init__(self, d_model=512, n_layers=16, attn_every=8, moe_every=2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            if (i + 1) % attn_every == 0:
                layers.append(SimpleAttentionLayer(d_model))
            else:
                layers.append(SimpleMambaLayer(d_model))
        self.layers = nn.ModuleList(layers)

        # Count layer types
        n_attn = sum(1 for l in layers if isinstance(l, SimpleAttentionLayer))
        n_mamba = sum(1 for l in layers if isinstance(l, SimpleMambaLayer))
        print(f"Jamba: {n_attn} attention + {n_mamba} Mamba layers")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Build and analyze
model = JambaModel(d_model=512, n_layers=16, attn_every=8)
x = torch.randn(2, 1024, 512)
y = model(x)
print(f"Output shape: {y.shape}")

# Memory analysis: KV cache savings
seq_len, d, n_kv_heads, head_dim = 256_000, 512, 2, 64
full_attn_kv = 16 * 2 * seq_len * n_kv_heads * head_dim * 2  # All 16 layers
jamba_kv = 2 * 2 * seq_len * n_kv_heads * head_dim * 2        # Only 2 attn layers
mamba_state = 14 * d * 16 * 4  # 14 Mamba layers, d_state=16
print(f"\\nKV cache at {seq_len:,} tokens:")
print(f"  Full attention: {full_attn_kv / 1e9:.2f} GB")
print(f"  Jamba (attn):   {jamba_kv / 1e9:.2f} GB")
print(f"  Jamba (mamba):  {mamba_state / 1e6:.2f} MB (constant!)")
print(f"  Total savings:  {(1 - (jamba_kv + mamba_state) / full_attn_kv):.0%}")`}
        id="code-jamba"
      />

      <NoteBlock
        type="intuition"
        title="Why Hybrid Works"
        content="Mamba layers efficiently propagate information across long contexts through their recurrent state, handling the bulk of computation cheaply. The sparse attention layers act as 'checkpoints' where the model can perform precise token-to-token operations like copying, retrieval, and comparison. This division of labor gives near-linear scaling with periodic exact-attention capabilities."
        id="note-hybrid-intuition"
      />

      <NoteBlock
        type="note"
        title="Jamba's Impact"
        content="AI21's Jamba (2024) was the first production-quality hybrid SSM-attention model. At 52B total parameters (12B active), it fits a 256K context in a single 80GB GPU — impossible for a pure attention model of similar quality. This demonstrated that hybrid architectures are a practical path to efficient long-context LLMs."
        id="note-jamba-impact"
      />

      <WarningBlock
        title="Architecture Design Sensitivity"
        content="The ratio of attention to Mamba layers, placement of MoE layers, and expert count all significantly affect performance. Too few attention layers degrades retrieval tasks; too many negates the efficiency gains. The optimal configuration depends on the target task mix and hardware constraints."
        id="warning-jamba-design"
      />
    </div>
  )
}
