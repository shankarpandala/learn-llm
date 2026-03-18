import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function CogVideo() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">CogVideo and CogVideoX</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        CogVideo from Tsinghua University and Zhipu AI is a family of text-to-video models.
        CogVideoX, the latest iteration, uses a 3D VAE and expert transformer blocks with
        3D full attention over spacetime, achieving high-quality open-source video generation
        that approaches proprietary systems in quality and coherence.
      </p>

      <DefinitionBlock
        title="CogVideoX Architecture"
        definition="CogVideoX uses a 3D causal VAE that compresses videos by 4x temporally and 8x spatially. The denoising backbone is a DiT with expert adaptive LayerNorm, processing joint text-video token sequences with full 3D attention. Text tokens from T5-XXL are concatenated with video latent patch tokens."
        id="def-cogvideox"
      />

      <h2 className="text-2xl font-semibold">3D Causal VAE</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The 3D VAE in CogVideoX uses causal convolutions along the temporal axis, ensuring
        that frame <InlineMath math="t" /> only depends on frames <InlineMath math="\leq t" />.
        This enables autoregressive extension of videos beyond the training length.
      </p>

      <ExampleBlock
        title="CogVideoX Generation Specs"
        problem="CogVideoX-5B generates 6-second 720p videos at 8fps. What are the latent dimensions?"
        steps={[
          { formula: '\\text{Frames: } 6 \\times 8 = 48, \\text{ after 4x temporal: } 48 / 4 = 12', explanation: 'Temporal compression yields 12 latent frames.' },
          { formula: '\\text{Spatial: } 720/8 \\times 1280/8 = 90 \\times 160', explanation: 'Spatial compression from pixel to latent space.' },
          { formula: '\\text{With 16 latent channels: } 12 \\times 90 \\times 160 \\times 16 \\approx 2.76M \\text{ values}', explanation: 'Total latent tensor size for denoising.' },
        ]}
        id="example-cogvideox-specs"
      />

      <PythonCode
        title="cogvideox_generation.py"
        code={`from diffusers import CogVideoXPipeline
import torch

# CogVideoX-5B text-to-video
# pipe = CogVideoXPipeline.from_pretrained(
#     "THUDM/CogVideoX-5b",
#     torch_dtype=torch.bfloat16,
# )
# pipe.enable_model_cpu_offload()
# pipe.vae.enable_tiling()  # Tile-based VAE decoding to save memory

# prompt = "A golden retriever running through a field of sunflowers, slow motion"
# video_frames = pipe(
#     prompt=prompt,
#     num_inference_steps=50,
#     guidance_scale=6.0,
#     num_frames=48,  # 6 seconds at 8fps
# ).frames[0]

# # Save as MP4
# from diffusers.utils import export_to_video
# export_to_video(video_frames, "dog_sunflowers.mp4", fps=8)

# CogVideoX model comparison
models = {
    "CogVideoX-2B": {
        "params": "2B",
        "resolution": "720x480",
        "frames": 48,
        "fps": 8,
        "vae_channels": 16,
    },
    "CogVideoX-5B": {
        "params": "5B",
        "resolution": "1280x720",
        "frames": 48,
        "fps": 8,
        "vae_channels": 16,
    },
}

for name, cfg in models.items():
    print(f"\\n{name}:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

# 3D causal convolution concept
import torch.nn as nn

class CausalConv3d(nn.Module):
    """3D convolution with causal padding along temporal axis."""
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.temporal_pad = kernel_size - 1  # Causal: pad only past
        self.spatial_pad = kernel_size // 2  # Symmetric spatial padding
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, padding=0)

    def forward(self, x):
        # x: (B, C, T, H, W)
        # Pad: (W_left, W_right, H_top, H_bottom, T_past, T_future)
        x = nn.functional.pad(x, (
            self.spatial_pad, self.spatial_pad,   # W
            self.spatial_pad, self.spatial_pad,   # H
            self.temporal_pad, 0,                 # T: only past padding
        ))
        return self.conv(x)

# Test causal conv
conv = CausalConv3d(4, 16, kernel_size=3)
x = torch.randn(1, 4, 12, 90, 160)
out = conv(x)
print(f"\\nCausal Conv3D: {x.shape} -> {out.shape}")  # Same temporal dim`}
        id="code-cogvideo"
      />

      <NoteBlock
        type="note"
        title="Progressive Training"
        content="CogVideoX uses progressive training: first low-resolution short videos, then gradually increasing to full resolution and length. This curriculum learning approach stabilizes training and is more efficient than directly training at the target resolution."
        id="note-progressive"
      />

      <WarningBlock
        title="Memory for Video Generation"
        content="CogVideoX-5B requires ~30GB VRAM even with model offloading and VAE tiling. The 3D attention over all spacetime tokens is memory-intensive. For consumer GPUs, use the 2B variant or apply further optimizations like quantization and attention slicing."
        id="warning-cogvideo-memory"
      />
    </div>
  )
}
