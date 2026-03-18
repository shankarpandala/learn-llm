import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function FusionStrategies() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Multimodal Fusion Strategies</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Fusion refers to how information from different modalities (vision and language) is
        combined. The three main strategies -- early fusion, late fusion, and cross-attention
        fusion -- each present different trade-offs in expressiveness, efficiency, and
        modularity. The choice of fusion strategy is one of the most important architectural
        decisions in vision-language models.
      </p>

      <DefinitionBlock
        title="Early Fusion"
        definition="In early fusion, visual and textual tokens are concatenated into a single sequence and processed together through shared transformer layers. The model learns cross-modal interactions from the first layer. Formally, $\mathbf{Z} = \text{Transformer}([\mathbf{V}_1, \ldots, \mathbf{V}_m, \mathbf{T}_1, \ldots, \mathbf{T}_n])$."
        id="def-early-fusion"
      />

      <DefinitionBlock
        title="Late Fusion"
        definition="In late fusion, each modality is processed independently by its own encoder, and representations are combined only at the final layers (e.g., via dot product or MLP). CLIP is a classic late fusion model: $\text{score} = \mathbf{f}_{\text{img}}(\mathbf{x})^\top \mathbf{f}_{\text{txt}}(\mathbf{t})$."
        id="def-late-fusion"
      />

      <DefinitionBlock
        title="Cross-Attention Fusion"
        definition="Cross-attention fusion uses dedicated cross-attention layers where one modality attends to the other. Text tokens query visual features: $\text{CrossAttn}(\mathbf{Q}_{\text{text}}, \mathbf{K}_{\text{image}}, \mathbf{V}_{\text{image}})$. This enables rich interaction while keeping encoders modular."
        id="def-cross-fusion"
      />

      <ExampleBlock
        title="Comparing Fusion Strategies"
        problem="For an image with 196 visual tokens and text with 77 tokens, compare computational costs."
        steps={[
          { formula: '\\text{Early: self-attn on } (196 + 77)^2 = 273^2 = 74{,}529', explanation: 'All tokens attend to all others; quadratic in total length.' },
          { formula: '\\text{Late: } 196^2 + 77^2 = 38{,}416 + 5{,}929 = 44{,}345', explanation: 'Each modality processes independently; cheaper but no cross-modal interaction.' },
          { formula: '\\text{Cross-attn: } 77 \\times 196 = 15{,}092 \\text{ (per cross-attn layer)}', explanation: 'Text queries attend to visual keys/values; more efficient than early fusion.' },
        ]}
        id="example-fusion-cost"
      />

      <PythonCode
        title="fusion_strategies.py"
        code={`import torch
import torch.nn as nn

class EarlyFusion(nn.Module):
    """Concatenate visual + text tokens, process jointly."""
    def __init__(self, d_model=768, nhead=12, num_layers=6):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, vis_tokens, txt_tokens):
        # Simply concatenate along sequence dimension
        combined = torch.cat([vis_tokens, txt_tokens], dim=1)
        return self.encoder(combined)

class LateFusion(nn.Module):
    """Process modalities independently, combine at the end."""
    def __init__(self, d_model=768):
        super().__init__()
        self.vis_proj = nn.Linear(d_model, 512)
        self.txt_proj = nn.Linear(d_model, 512)

    def forward(self, vis_cls, txt_cls):
        # Project both to shared space
        vis_emb = nn.functional.normalize(self.vis_proj(vis_cls), dim=-1)
        txt_emb = nn.functional.normalize(self.txt_proj(txt_cls), dim=-1)
        return vis_emb @ txt_emb.T  # similarity matrix

class CrossAttentionFusion(nn.Module):
    """Text queries attend to visual features."""
    def __init__(self, d_model=768, nhead=12, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, txt_tokens, vis_tokens):
        x = txt_tokens
        for attn, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            x, _ = attn(query=x, key=vis_tokens, value=vis_tokens)
            x = x + residual
        return x

# Compare all three
B, D = 2, 768
vis = torch.randn(B, 196, D)   # 14x14 patches
txt = torch.randn(B, 77, D)    # text tokens

early = EarlyFusion()
late = LateFusion()
cross = CrossAttentionFusion()

out_early = early(vis, txt)
print(f"Early fusion output: {out_early.shape}")      # (2, 273, 768)

out_late = late(vis[:, 0], txt[:, 0])  # CLS tokens
print(f"Late fusion output: {out_late.shape}")         # (2, 2)

out_cross = cross(txt, vis)
print(f"Cross-attention output: {out_cross.shape}")    # (2, 77, 768)`}
        id="code-fusion"
      />

      <NoteBlock
        type="intuition"
        title="Which Fusion to Choose?"
        content="Early fusion (LLaVA, GPT-4V) is most expressive but costly. It works well when the LLM is large enough to learn cross-modal reasoning. Late fusion (CLIP, SigLIP) is most efficient for retrieval tasks. Cross-attention (Flamingo, Qwen-VL) offers a middle ground with modularity -- you can swap the vision encoder without retraining the LLM."
        id="note-fusion-choice"
      />

      <WarningBlock
        title="Early Fusion Context Length"
        content="Early fusion models must fit both visual and text tokens within the context window. A single 224x224 image at patch size 14 produces 256 visual tokens. Higher-resolution images or multiple images can quickly exhaust the context budget, leaving less room for text generation."
        id="warning-context-length"
      />
    </div>
  )
}
