import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ImageTokenization() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Patch Embedding and Image Tokenization</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Image tokenization converts continuous pixel data into discrete or continuous token
        representations that transformers can process. Beyond simple linear patch projection,
        modern approaches use convolutional stems, VQ-VAE codebooks, or learned visual
        vocabularies to create richer image tokens.
      </p>

      <DefinitionBlock
        title="Patch Embedding"
        definition="Patch embedding maps each image patch $\mathbf{x}_p^i \in \mathbb{R}^{P^2 \cdot C}$ to a $D$-dimensional vector via a learnable projection $\mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$. This is equivalent to applying a convolution with kernel size and stride equal to the patch size $P$."
        id="def-patch-embed"
      />

      <h2 className="text-2xl font-semibold">Linear Patch Projection</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The simplest patch embedding is a linear layer applied to flattened patches. A
        <InlineMath math="P \times P" /> patch from a <InlineMath math="C" />-channel image
        produces a <InlineMath math="P^2 C" />-dimensional vector, projected to dimension{' '}
        <InlineMath math="D" />.
      </p>
      <BlockMath math="\mathbf{e}_i = \text{flatten}(\mathbf{x}_p^i) \cdot \mathbf{E} + \mathbf{b}, \quad \mathbf{e}_i \in \mathbb{R}^D" />

      <h2 className="text-2xl font-semibold">Convolutional Patch Embedding</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Using a convolutional stem instead of a single projection provides overlapping
        receptive fields, better handling of edges, and improved training stability. Many modern
        ViT variants (e.g., ConvNeXt-based stems) replace the linear projection with 2-4
        convolutional layers with decreasing stride.
      </p>

      <PythonCode
        title="patch_embedding_variants.py"
        code={`import torch
import torch.nn as nn

# Method 1: Linear projection (original ViT)
class LinearPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

# Method 2: Convolutional stem (better for smaller datasets)
class ConvStemPatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=768):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1),  # total stride=16
        )

    def forward(self, x):
        return self.stem(x).flatten(2).transpose(1, 2)

img = torch.randn(1, 3, 224, 224)
for name, module in [("Linear", LinearPatchEmbed()), ("ConvStem", ConvStemPatchEmbed())]:
    out = module(img)
    print(f"{name}: {out.shape}")  # Both: (1, 196, 768)`}
        id="code-patch-variants"
      />

      <h2 className="text-2xl font-semibold">VQ-VAE Image Tokenization</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Vector-Quantized VAEs convert images into discrete codebook indices, enabling image
        generation via autoregressive token prediction. The encoder maps patches to the nearest
        codebook vector using nearest-neighbor lookup.
      </p>
      <BlockMath math="z_q = \text{argmin}_{e_k \in \mathcal{C}} \| z_e(\mathbf{x}) - e_k \|_2" />

      <ExampleBlock
        title="VQ-VAE Tokenization"
        problem="A 256x256 image encoded by a VQ-VAE with downsampling factor 16 and codebook size 8192. How many tokens?"
        steps={[
          { formula: '\\text{Latent grid} = \\frac{256}{16} \\times \\frac{256}{16} = 16 \\times 16 = 256', explanation: 'Spatial dimensions reduced by factor 16.' },
          { formula: '\\text{Each position} \\to \\text{index} \\in \\{0, 1, \\ldots, 8191\\}', explanation: 'Each spatial position maps to one of 8192 codebook entries.' },
          { formula: '\\text{Total: 256 discrete tokens from vocabulary of 8192}', explanation: 'The image becomes a sequence of 256 integer tokens, like text.' },
        ]}
        id="example-vqvae"
      />

      <PythonCode
        title="vq_tokenization.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """Simple VQ layer for image tokenization."""
    def __init__(self, num_embeddings=8192, embedding_dim=256):
        super().__init__()
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e):
        # z_e: (B, D, H, W) -> (B, H, W, D)
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e.view(-1, z_e.shape[-1])

        # Nearest neighbor lookup
        distances = torch.cdist(flat, self.codebook.weight)
        indices = distances.argmin(dim=-1)
        z_q = self.codebook(indices).view(z_e.shape)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Commitment loss
        commitment_loss = F.mse_loss(z_e.detach(), z_q) + F.mse_loss(z_e, z_q.detach())

        return z_q_st.permute(0, 3, 1, 2), indices, commitment_loss

vq = VectorQuantizer(num_embeddings=8192, embedding_dim=256)
z_encoded = torch.randn(2, 256, 16, 16)  # Pretend encoder output
z_quantized, token_ids, loss = vq(z_encoded)
print(f"Quantized shape: {z_quantized.shape}")   # (2, 256, 16, 16)
print(f"Token IDs shape: {token_ids.shape}")      # (512,) = 2*16*16
print(f"Unique tokens used: {token_ids.unique().shape[0]}")`}
        id="code-vq"
      />

      <NoteBlock
        type="tip"
        title="Patch Size Trade-offs"
        content="Smaller patches (e.g., 8x8) produce longer sequences with finer detail but quadratically increase attention cost. Larger patches (e.g., 32x32) are faster but lose fine-grained information. Most modern ViTs use 14x14 or 16x16 patches as a good balance."
        id="note-patch-tradeoff"
      />

      <WarningBlock
        title="Codebook Collapse"
        content="VQ-VAEs frequently suffer from codebook collapse where only a small fraction of codebook entries are used. Techniques like EMA updates, codebook reset, and entropy regularization help maintain codebook utilization."
        id="warning-codebook-collapse"
      />
    </div>
  )
}
