import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function VideoDiffusion() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Video Diffusion Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Video diffusion extends image diffusion to the temporal dimension, generating sequences
        of coherent frames. The key challenge is maintaining temporal consistency -- objects
        should move smoothly, lighting should be stable, and physics should be plausible across
        frames. Video diffusion models typically add temporal attention layers to an image
        diffusion backbone.
      </p>

      <DefinitionBlock
        title="Video Diffusion"
        definition="Video diffusion operates on 3D latent tensors $\mathbf{z} \in \mathbb{R}^{F \times H \times W \times C}$ where $F$ is the number of frames. The denoising network includes both spatial attention (within each frame) and temporal attention (across frames at the same spatial position) to ensure spatial quality and temporal coherence."
        id="def-video-diffusion"
      />

      <h2 className="text-2xl font-semibold">Temporal Attention</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The standard approach inserts temporal attention layers after spatial attention layers.
        Spatial attention operates on <InlineMath math="H \times W" /> tokens per frame, while
        temporal attention operates on <InlineMath math="F" /> tokens per spatial position.
      </p>
      <BlockMath math="\text{Spatial: } \text{Attn}(\mathbf{z}_{f,:,:}) \text{ for each frame } f \quad | \quad \text{Temporal: } \text{Attn}(\mathbf{z}_{:,h,w}) \text{ for each position } (h,w)" />

      <ExampleBlock
        title="Video Latent Dimensions"
        problem="A 16-frame 512x512 video encoded with SD's VAE (f=8) and 4 latent channels."
        steps={[
          { formula: '\\text{Spatial: } 512/8 = 64 \\implies 64 \\times 64 = 4{,}096 \\text{ tokens/frame}', explanation: 'Same as image latent diffusion per frame.' },
          { formula: '\\text{Total latent: } 16 \\times 64 \\times 64 \\times 4 = 262{,}144 \\text{ values}', explanation: 'Full 3D latent tensor for the video.' },
          { formula: '\\text{Spatial attn: } 4096^2 \\times 16 \\approx 268M \\text{ ops per layer}', explanation: 'Spatial attention is expensive but parallelizable across frames.' },
          { formula: '\\text{Temporal attn: } 16^2 \\times 4096 \\approx 1M \\text{ ops per layer}', explanation: 'Temporal attention is cheap since F is small.' },
        ]}
        id="example-video-latent"
      />

      <PythonCode
        title="temporal_attention.py"
        code={`import torch
import torch.nn as nn
from einops import rearrange

class TemporalAttentionBlock(nn.Module):
    """Temporal self-attention across video frames."""
    def __init__(self, dim=320, num_heads=8, num_frames=16):
        super().__init__()
        self.num_frames = num_frames
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B*F, H*W, D) from spatial attention
        BF, HW, D = x.shape
        B = BF // self.num_frames
        F = self.num_frames

        # Reshape: (B*F, HW, D) -> (B*HW, F, D) for temporal attention
        x_temporal = rearrange(x, '(b f) hw d -> (b hw) f d', b=B, f=F)

        # Temporal self-attention
        residual = x_temporal
        x_normed = self.norm(x_temporal)
        attended, _ = self.attn(x_normed, x_normed, x_normed)
        x_temporal = residual + self.proj(attended)

        # Reshape back: (B*HW, F, D) -> (B*F, HW, D)
        return rearrange(x_temporal, '(b hw) f d -> (b f) hw d', b=B, hw=HW)

class SpatioTemporalBlock(nn.Module):
    """Combined spatial + temporal attention block."""
    def __init__(self, dim=320, num_heads=8, num_frames=16):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.spatial_norm = nn.LayerNorm(dim)
        self.temporal_attn = TemporalAttentionBlock(dim, num_heads, num_frames)

    def forward(self, x):
        # Spatial attention (within each frame)
        residual = x
        x = self.spatial_norm(x)
        x, _ = self.spatial_attn(x, x, x)
        x = x + residual

        # Temporal attention (across frames)
        x = self.temporal_attn(x)
        return x

# Test
B, F, H, W, D = 2, 16, 8, 8, 320  # Small test
block = SpatioTemporalBlock(dim=D, num_frames=F)
x = torch.randn(B * F, H * W, D)
out = block(x)
print(f"Input: {x.shape} -> Output: {out.shape}")  # (32, 64, 320)`}
        id="code-temporal-attn"
      />

      <NoteBlock
        type="intuition"
        title="Factored Attention"
        content="Full 3D attention over all frames and spatial positions is prohibitively expensive (O(F^2 * H^2 * W^2)). Factoring into separate spatial and temporal attention reduces this to O(F * H^2 * W^2 + F^2 * H * W), making video generation tractable. Some models add a third factored dimension for cross-frame spatial attention."
        id="note-factored"
      />

      <WarningBlock
        title="Temporal Consistency"
        content="Even with temporal attention, video diffusion models can produce flickering, object deformation, and inconsistent backgrounds. These artifacts are most visible in long videos (>4 seconds) and fast motion. Post-processing with optical flow-based smoothing or video interpolation can help."
        id="warning-consistency"
      />
    </div>
  )
}
