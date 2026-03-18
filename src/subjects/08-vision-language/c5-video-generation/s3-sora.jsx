import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Sora() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Sora: Scaling Video Generation</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        OpenAI's Sora represents a paradigm shift in video generation by treating videos as
        sequences of spacetime patches processed by a Diffusion Transformer. Sora can generate
        up to 60 seconds of high-fidelity video with complex scenes, consistent characters,
        and realistic physics. Its key insight is that video generation benefits from the same
        scaling laws as language models.
      </p>

      <DefinitionBlock
        title="Spacetime Patches"
        definition="Sora decomposes video into spacetime patches: 3D chunks of size $t_p \times h_p \times w_p$ from the compressed video latent. These patches are flattened and linearly projected to form tokens, analogous to ViT's image patches but extended to the temporal dimension. This enables variable duration, resolution, and aspect ratio."
        id="def-spacetime-patches"
      />

      <h2 className="text-2xl font-semibold">Architecture Principles</h2>
      <p className="text-gray-700 dark:text-gray-300">
        While Sora's full architecture is not public, the technical report reveals key design
        principles: a video compression network (temporal VAE), a DiT backbone operating on
        spacetime patches, and training on native resolution videos without cropping.
      </p>
      <BlockMath math="\text{Spacetime tokens: } N = \frac{F}{t_p} \times \frac{H}{h_p} \times \frac{W}{w_p}" />

      <ExampleBlock
        title="Sora Token Count Estimation"
        problem="Estimate tokens for a 10-second 1080p video at 24fps with temporal VAE compression 4x and spatial compression 8x, using 2x2x2 spacetime patches."
        steps={[
          { formula: '\\text{Frames: } 10 \\times 24 = 240, \\text{ after temporal VAE: } 240 / 4 = 60', explanation: 'Temporal compression reduces frame count.' },
          { formula: '\\text{Spatial: } 1080/8 = 135, \\; 1920/8 = 240 \\text{ (latent spatial)}', explanation: 'Spatial VAE compression.' },
          { formula: '\\text{Spacetime patches: } (60/2) \\times (135/2) \\times (240/2) \\approx 2.4M', explanation: 'Massive token count requires efficient attention.' },
        ]}
        id="example-sora-tokens"
      />

      <PythonCode
        title="spacetime_patch_embedding.py"
        code={`import torch
import torch.nn as nn

class SpacetimePatchEmbed(nn.Module):
    """Convert video latents to spacetime patch tokens (Sora-style)."""
    def __init__(self, in_channels=4, embed_dim=1024,
                 patch_size_t=2, patch_size_h=2, patch_size_w=2):
        super().__init__()
        self.patch_size = (patch_size_t, patch_size_h, patch_size_w)
        # 3D convolution acts as patch extraction + projection
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        # x: (B, C, T, H, W) - video latent
        x = self.proj(x)  # (B, D, T', H', W')
        B, D, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, T'*H'*W', D)
        return x, (T, H, W)

class VideoTemporalVAE(nn.Module):
    """Simplified temporal VAE for video compression."""
    def __init__(self, in_channels=3, latent_channels=4,
                 spatial_downsample=8, temporal_downsample=4):
        super().__init__()
        self.spatial_down = spatial_downsample
        self.temporal_down = temporal_downsample

        # Simplified: in practice, this is a 3D convolutional encoder
        self.encoder = nn.Conv3d(
            in_channels, latent_channels,
            kernel_size=(temporal_downsample, spatial_downsample, spatial_downsample),
            stride=(temporal_downsample, spatial_downsample, spatial_downsample),
        )

    def encode(self, video):
        # video: (B, C, T, H, W)
        return self.encoder(video)

# Demonstrate the pipeline
B, C, T, H, W = 1, 3, 48, 256, 256

# Step 1: Temporal VAE compression
vae = VideoTemporalVAE()
video = torch.randn(B, C, T, H, W)
latent = vae.encode(video)
print(f"Video:  {video.shape}")   # (1, 3, 48, 256, 256)
print(f"Latent: {latent.shape}")  # (1, 4, 12, 32, 32)

# Step 2: Spacetime patch embedding
patch_embed = SpacetimePatchEmbed(in_channels=4, embed_dim=1024)
tokens, grid = patch_embed(latent)
print(f"Tokens: {tokens.shape}")  # (1, 6*16*16=1536, 1024)
print(f"Grid:   {grid}")          # (6, 16, 16)

# Step 3: Process with DiT (self-attention over all spacetime tokens)
dit_layer = nn.TransformerEncoderLayer(
    d_model=1024, nhead=16, dim_feedforward=4096,
    batch_first=True, norm_first=True
)
output = dit_layer(tokens)
print(f"DiT output: {output.shape}")  # (1, 1536, 1024)`}
        id="code-sora"
      />

      <NoteBlock
        type="intuition"
        title="Videos as World Simulators"
        content="Sora's technical report describes the model as a 'world simulator' -- by learning to predict video frames, the model implicitly learns about 3D geometry, physics, object permanence, and even basic cause-and-effect. This emergent understanding improves with scale, similar to how LLMs develop reasoning abilities."
        id="note-world-simulator"
      />

      <NoteBlock
        type="historical"
        title="Open Source Alternatives"
        content="Since Sora is not publicly available, the community has developed open alternatives: Open-Sora (HPC-AI Tech), Open-Sora-Plan, and CogVideoX (Tsinghua/Zhipu). These replicate core ideas (spacetime DiT, temporal VAE) at smaller scale, enabling research and experimentation."
        id="note-open-sora"
      />

      <WarningBlock
        title="Compute Requirements"
        content="Sora-scale video generation likely requires thousands of GPUs for training and significant resources even for inference (a 60-second 1080p video may take minutes on high-end hardware). The compute gap between image and video generation is roughly 100-1000x."
        id="warning-sora-compute"
      />
    </div>
  )
}
