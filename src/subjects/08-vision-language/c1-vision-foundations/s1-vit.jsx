import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function VisionTransformer() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Vision Transformer (ViT)</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The Vision Transformer (ViT) demonstrated that a pure transformer architecture, originally
        designed for NLP, can achieve state-of-the-art results on image classification when trained
        on sufficient data. ViT splits an image into fixed-size patches and treats them as a
        sequence of tokens, just like words in a sentence.
      </p>

      <DefinitionBlock
        title="Vision Transformer"
        definition="A Vision Transformer (ViT) divides an image of size $H \times W$ into a grid of $N = \frac{H \times W}{P^2}$ non-overlapping patches of size $P \times P$, linearly embeds each patch into a $D$-dimensional vector, prepends a learnable [CLS] token, adds positional embeddings, and processes the resulting sequence through a standard transformer encoder."
        notation="Given image \( \mathbf{x} \in \mathbb{R}^{H \times W \times C} \), patches \( \mathbf{x}_p^i \in \mathbb{R}^{P^2 \cdot C} \) for \( i = 1, \ldots, N \)"
        id="def-vit"
      />

      <h2 className="text-2xl font-semibold">Patch Embedding</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Each patch is flattened from a <InlineMath math="P \times P \times C" /> tensor into a
        vector and projected through a linear layer to produce a <InlineMath math="D" />-dimensional
        embedding. Positional embeddings are added to retain spatial information.
      </p>
      <BlockMath math="\mathbf{z}_0 = [\mathbf{x}_{\text{class}};\; \mathbf{x}_p^1 \mathbf{E};\; \mathbf{x}_p^2 \mathbf{E};\; \ldots;\; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}}, \quad \mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}" />

      <ExampleBlock
        title="ViT-Base Patch Calculation"
        problem="For a 224x224 RGB image with patch size 16, how many patches and what is the sequence length?"
        steps={[
          { formula: 'N = \\frac{224 \\times 224}{16 \\times 16} = \\frac{50176}{256} = 196', explanation: 'Divide total pixels by pixels per patch.' },
          { formula: '\\text{seq\\_len} = N + 1 = 197', explanation: 'Add 1 for the prepended [CLS] token.' },
          { formula: '\\text{Each patch: } 16 \\times 16 \\times 3 = 768 \\text{ values}', explanation: 'Flattened patch dimension matches ViT-Base hidden size D=768.' },
        ]}
        id="example-vit-patches"
      />

      <PythonCode
        title="vit_from_scratch.py"
        code={`import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # Conv2d with kernel=stride=patch_size acts as patch extraction + linear projection
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, embed_dim, N) -> (B, N, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x)                          # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)                 # (B, 1, D)
        z = torch.cat([cls, patches], dim=1) + self.pos_embed  # (B, N+1, D)
        z = self.encoder(z)
        z = self.norm(z[:, 0])  # [CLS] token output
        return self.head(z)

# Test
model = ViT()
img = torch.randn(2, 3, 224, 224)
logits = model(img)
print(f"Output shape: {logits.shape}")  # (2, 1000)
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")`}
        id="code-vit"
      />

      <NoteBlock
        type="historical"
        title="ViT Origin"
        content="Dosovitskiy et al. (2020) introduced ViT in 'An Image Is Worth 16x16 Words.' The key finding was that with large-scale pretraining (JFT-300M), ViT surpassed CNNs, but with smaller datasets like ImageNet alone, CNNs still performed better due to their inductive biases (translation equivariance, locality)."
        id="note-vit-history"
      />

      <WarningBlock
        title="Positional Embedding Interpolation"
        content="ViT learns fixed positional embeddings for a specific resolution. When fine-tuning at higher resolutions, you must interpolate the positional embeddings (typically with bicubic interpolation), which can degrade performance if the resolution gap is too large."
        id="warning-pos-embed"
      />

      <NoteBlock
        type="intuition"
        title="Why Patches Work"
        content="Treating patches as tokens lets ViT leverage the transformer's global attention from the first layer. Unlike CNNs that build up receptive fields gradually, ViT can attend to distant image regions immediately, which helps with tasks requiring global context like scene understanding."
        id="note-patches-intuition"
      />
    </div>
  )
}
