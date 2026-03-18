import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function StableVideoDiffusion() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Stable Video Diffusion (SVD)</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Stable Video Diffusion from Stability AI adapts the Stable Diffusion architecture for
        video generation. It takes a single image as input and generates a short video clip
        (14-25 frames) showing plausible motion. SVD is trained in three stages: image
        pretraining, video pretraining on large data, and high-quality video fine-tuning.
      </p>

      <DefinitionBlock
        title="Stable Video Diffusion"
        definition="SVD extends SD's U-Net with temporal convolution and temporal attention layers. Given a conditioning image $\mathbf{x}_{\text{cond}}$, it generates $F$ frames by denoising a 3D latent tensor $\mathbf{z} \in \mathbb{R}^{F \times h \times w \times c}$. The conditioning image is encoded by the VAE and concatenated channel-wise with the noisy latent at each denoising step."
        id="def-svd"
      />

      <h2 className="text-2xl font-semibold">Image-to-Video Pipeline</h2>
      <p className="text-gray-700 dark:text-gray-300">
        SVD operates as an image-to-video model. The input image conditions the generation,
        and additional parameters control motion magnitude and frame rate. The model learns
        plausible motion patterns from its video training data.
      </p>

      <ExampleBlock
        title="SVD Architecture Components"
        problem="What are the added components in SVD compared to SD 2.1?"
        steps={[
          { formula: '\\text{Temporal Conv: 1D convolution along time axis}', explanation: 'Inserted after each spatial 2D convolution to capture local temporal patterns.' },
          { formula: '\\text{Temporal Attn: self-attention across } F \\text{ frames}', explanation: 'Inserted after each spatial self-attention for global temporal coherence.' },
          { formula: '\\text{Image conditioning: } \\text{concat}(\\mathbf{z}_t, \\text{VAE}(\\mathbf{x}_{\\text{cond}}))', explanation: 'Conditioning image latent concatenated channel-wise with noisy latent.' },
          { formula: '\\text{Motion bucket: FPS + motion magnitude embedding}', explanation: 'Learnable embeddings control speed and amount of motion.' },
        ]}
        id="example-svd-components"
      />

      <PythonCode
        title="svd_generation.py"
        code={`from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import torch

# Load SVD pipeline
# pipe = StableVideoDiffusionPipeline.from_pretrained(
#     "stabilityai/stable-video-diffusion-img2vid-xt",  # 25 frames
#     torch_dtype=torch.float16,
#     variant="fp16",
# ).to("cuda")
# pipe.enable_model_cpu_offload()  # Required for most GPUs

# # Generate video from image
# image = Image.open("landscape.png").resize((1024, 576))
# frames = pipe(
#     image,
#     num_frames=25,           # Number of output frames
#     decode_chunk_size=8,     # Decode frames in chunks to save memory
#     motion_bucket_id=127,    # 0-255, higher = more motion
#     fps=7,                   # Frames per second
#     noise_aug_strength=0.02, # Noise augmentation on conditioning image
# ).frames[0]

# # Save as GIF
# frames[0].save("output.gif", save_all=True, append_images=frames[1:],
#                 duration=1000//7, loop=0)

# SVD temporal layer structure
import torch.nn as nn

class TemporalConvLayer(nn.Module):
    """1D temporal convolution for local temporal modeling."""
    def __init__(self, channels, num_frames):
        super().__init__()
        self.num_frames = num_frames
        self.conv = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        # Zero-init for residual connection
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, x):
        # x: (B*F, C, H, W)
        BF, C, H, W = x.shape
        B = BF // self.num_frames

        residual = x
        # Reshape for temporal conv: (B*H*W, C, F)
        x = x.reshape(B, self.num_frames, C, H, W)
        x = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, C, self.num_frames)

        x = self.conv(x)

        # Reshape back
        x = x.reshape(B, H, W, C, self.num_frames)
        x = x.permute(0, 4, 3, 1, 2).reshape(BF, C, H, W)

        return residual + x

# Test
layer = TemporalConvLayer(channels=320, num_frames=14)
x = torch.randn(2 * 14, 320, 64, 64)
out = layer(x)
print(f"Temporal conv: {x.shape} -> {out.shape}")
print(f"SVD-XT: 25 frames @ 576x1024, ~3.5B params")`}
        id="code-svd"
      />

      <NoteBlock
        type="note"
        title="SVD Training Stages"
        content="Stage 1: Image pretraining (SD 2.1 weights). Stage 2: Video pretraining on Large Video Dataset (580M clips, filtered to 152M). Stage 3: High-quality fine-tuning on 1M carefully curated clips. This staged approach is critical -- training on video from scratch is extremely expensive."
        id="note-svd-stages"
      />

      <WarningBlock
        title="SVD Limitations"
        content="SVD generates short clips (2-4 seconds) without camera control, text conditioning, or multi-shot consistency. It often produces subtle 'breathing' artifacts where the image gently warps. For longer or more controlled videos, consider AnimateDiff, CogVideo, or Sora-type approaches."
        id="warning-svd-limitations"
      />
    </div>
  )
}
