import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Flamingo() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Flamingo: Perceiver Resampler for Vision-Language</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        DeepMind's Flamingo introduced a cross-attention fusion approach that interleaves
        visual information into a frozen LLM via gated cross-attention layers and a Perceiver
        Resampler. This design allows the LLM to process arbitrarily many images while keeping
        a fixed number of visual tokens per image, enabling powerful few-shot multimodal learning.
      </p>

      <DefinitionBlock
        title="Perceiver Resampler"
        definition="The Perceiver Resampler compresses variable-length visual features into a fixed set of $M$ learnable query vectors via cross-attention. Given visual features $\mathbf{V} \in \mathbb{R}^{N \times D}$ and $M$ learned queries $\mathbf{Q} \in \mathbb{R}^{M \times D}$, it outputs $\mathbf{O} = \text{CrossAttn}(\mathbf{Q}, \mathbf{V}, \mathbf{V}) \in \mathbb{R}^{M \times D}$ where typically $M \ll N$."
        id="def-perceiver"
      />

      <h2 className="text-2xl font-semibold">Gated Cross-Attention</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Flamingo inserts gated cross-attention layers between existing frozen LLM layers. The
        gating mechanism uses a learnable scalar <InlineMath math="\alpha" /> initialized to
        zero, ensuring that the model starts as the original LLM and gradually learns to
        incorporate visual information.
      </p>
      <BlockMath math="\mathbf{h} = \mathbf{h} + \tanh(\alpha) \cdot \text{CrossAttn}(\mathbf{h}, \mathbf{O}_{\text{vis}}, \mathbf{O}_{\text{vis}})" />

      <ExampleBlock
        title="Perceiver Resampler Compression"
        problem="An image has 576 visual tokens (ViT with 24x24 patches). The Perceiver uses 64 queries. What is the compression ratio?"
        steps={[
          { formula: '\\text{Compression} = \\frac{576}{64} = 9\\times', explanation: 'Each image is represented by only 64 tokens regardless of input resolution.' },
          { formula: '\\text{Cross-attn cost} = 64 \\times 576 = 36{,}864', explanation: 'Much cheaper than self-attention on full 576 tokens (331,776).' },
          { formula: '\\text{For 4 images: } 4 \\times 64 = 256 \\text{ visual tokens total}', explanation: 'Fixed cost per image enables multi-image conversations without explosion.' },
        ]}
        id="example-perceiver-compression"
      />

      <PythonCode
        title="perceiver_resampler.py"
        code={`import torch
import torch.nn as nn

class PerceiverResampler(nn.Module):
    """Compress variable-length visual features to fixed-length queries."""
    def __init__(self, dim=768, num_queries=64, num_layers=6, num_heads=12):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(dim, num_heads, batch_first=True),
                'cross_norm': nn.LayerNorm(dim),
                'ff': nn.Sequential(
                    nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
                ),
                'ff_norm': nn.LayerNorm(dim),
            }))

    def forward(self, visual_features):
        B = visual_features.shape[0]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            # Cross-attention: queries attend to visual features
            q_norm = layer['cross_norm'](queries)
            attended, _ = layer['cross_attn'](query=q_norm, key=visual_features,
                                               value=visual_features)
            queries = queries + attended

            # Feed-forward
            queries = queries + layer['ff'](layer['ff_norm'](queries))

        return queries

class GatedCrossAttention(nn.Module):
    """Gated cross-attention inserted between frozen LLM layers."""
    def __init__(self, dim=4096, num_heads=32):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(1))  # Initialize to 0

    def forward(self, text_hidden, visual_tokens):
        residual = text_hidden
        text_norm = self.norm(text_hidden)
        attended, _ = self.cross_attn(query=text_norm, key=visual_tokens,
                                       value=visual_tokens)
        return residual + torch.tanh(self.gate) * attended

# Demo
perceiver = PerceiverResampler(dim=768, num_queries=64)
vis_features = torch.randn(2, 576, 768)  # Variable-length visual features
compressed = perceiver(vis_features)
print(f"Input: {vis_features.shape} -> Output: {compressed.shape}")
# (2, 576, 768) -> (2, 64, 768)

gated = GatedCrossAttention(dim=768, num_heads=12)
text_h = torch.randn(2, 128, 768)
vis_tokens = torch.randn(2, 64, 768)
out = gated(text_h, vis_tokens)
print(f"Gated cross-attn: {out.shape}")  # (2, 128, 768)
print(f"Initial gate value: {torch.tanh(gated.gate).item():.4f}")  # ~0.0`}
        id="code-perceiver"
      />

      <NoteBlock
        type="historical"
        title="From Flamingo to Open Source"
        content="Flamingo (Alayrac et al., 2022) was proprietary. OpenFlamingo replicated it using open components (CLIP + LLaMA). The Perceiver Resampler idea was adopted by BLIP-2's Q-Former and Qwen-VL's visual resampler. The core insight -- compress visual features with learnable queries -- remains widely used."
        id="note-flamingo-history"
      />

      <WarningBlock
        title="Information Bottleneck"
        content="The Perceiver Resampler acts as an information bottleneck. With too few queries (e.g., 4-8), fine-grained visual details like small text in images are lost. Tasks requiring detailed visual understanding may need more queries (64-256) or direct token injection like LLaVA."
        id="warning-bottleneck"
      />
    </div>
  )
}
