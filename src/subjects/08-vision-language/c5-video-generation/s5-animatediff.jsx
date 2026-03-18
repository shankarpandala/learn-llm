import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function AnimateDiff() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">AnimateDiff: Plug-and-Play Motion Modules</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        AnimateDiff introduces a modular approach to video generation by training a standalone
        motion module that can be plugged into any personalized Stable Diffusion model (including
        LoRA-customized checkpoints). This means you can animate any image style without
        retraining the base model -- the motion module provides temporal coherence while the
        base model controls visual style.
      </p>

      <DefinitionBlock
        title="Motion Module"
        definition="A motion module is a set of temporal attention layers that are inserted into a frozen Stable Diffusion U-Net. Each motion module block operates on features across frames at a given spatial resolution, learning general motion priors from video data. The key property is compatibility: the same motion module works with any SD 1.5 checkpoint."
        id="def-motion-module"
      />

      <h2 className="text-2xl font-semibold">Architecture Design</h2>
      <p className="text-gray-700 dark:text-gray-300">
        AnimateDiff inserts temporal transformer blocks after each spatial self-attention block
        in the U-Net. During training, only the motion module parameters are updated while the
        SD U-Net remains frozen. At inference, the motion module can be combined with any
        compatible SD checkpoint.
      </p>

      <ExampleBlock
        title="AnimateDiff Modularity"
        problem="How does AnimateDiff enable animating a custom art style?"
        steps={[
          { formula: '\\text{Step 1: User has custom SD model (e.g., anime LoRA)}', explanation: 'Any SD 1.5 checkpoint or LoRA that generates desired visual style.' },
          { formula: '\\text{Step 2: Insert motion module into U-Net}', explanation: 'Temporal attention layers are added between existing spatial layers.' },
          { formula: '\\text{Step 3: Generate video with combined model}', explanation: 'Base model controls style, motion module controls movement.' },
          { formula: '\\text{No retraining needed!}', explanation: 'The same motion module generalizes across different style checkpoints.' },
        ]}
        id="example-animatediff-modularity"
      />

      <PythonCode
        title="animatediff_generation.py"
        code={`from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif
import torch

# Load motion module adapter
# adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-3")

# # Combine with any SD 1.5 model
# pipe = AnimateDiffPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",  # Can swap with any SD 1.5 model
#     motion_adapter=adapter,
#     torch_dtype=torch.float16,
# ).to("cuda")
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# # Generate animation
# output = pipe(
#     prompt="A cat playing with a ball of yarn, detailed fur, soft lighting",
#     negative_prompt="blurry, low quality, static",
#     num_frames=16,
#     num_inference_steps=25,
#     guidance_scale=7.5,
# )
# export_to_gif(output.frames[0], "cat_yarn.gif")

# # Use with a custom LoRA for style
# pipe.load_lora_weights("custom_anime_lora", adapter_name="anime")
# pipe.set_adapters(["anime"], [0.8])  # LoRA weight
# output_anime = pipe(
#     prompt="anime girl walking in a garden, cherry blossoms",
#     num_frames=16,
#     num_inference_steps=25,
# )
# export_to_gif(output_anime.frames[0], "anime_walk.gif")

# Simplified motion module structure
import torch.nn as nn

class MotionModuleBlock(nn.Module):
    """Single AnimateDiff motion module block."""
    def __init__(self, channels, num_frames=16, num_heads=8):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(
            channels, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(channels)
        self.pos_embed = nn.Parameter(torch.randn(1, num_frames, channels) * 0.02)
        # Zero-init output projection for clean integration
        self.proj_out = nn.Linear(channels, channels)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, num_frames):
        # x: (B*F, HW, C) from spatial attention
        BF, HW, C = x.shape
        B = BF // num_frames

        # Reshape for temporal attention
        x_t = x.reshape(B, num_frames, HW, C)
        x_t = x_t.permute(0, 2, 1, 3).reshape(B * HW, num_frames, C)

        # Add temporal positional embedding
        x_t = x_t + self.pos_embed[:, :num_frames]

        residual = x_t
        x_t = self.norm(x_t)
        x_t, _ = self.temporal_attn(x_t, x_t, x_t)
        x_t = residual + self.proj_out(x_t)

        # Reshape back
        x_t = x_t.reshape(B, HW, num_frames, C)
        x_t = x_t.permute(0, 2, 1, 3).reshape(BF, HW, C)
        return x_t

# Test
block = MotionModuleBlock(channels=320, num_frames=16)
x = torch.randn(32, 64, 320)  # B*F=32, HW=64
out = block(x, num_frames=16)
print(f"Motion module: {x.shape} -> {out.shape}")
print(f"Motion module params: {sum(p.numel() for p in block.parameters()) / 1e6:.1f}M")`}
        id="code-animatediff"
      />

      <NoteBlock
        type="tip"
        title="AnimateDiff Versions"
        content="AnimateDiff v1 produces basic motion, v2 improves quality, and v3 adds sparse controlnet for motion guidance. AnimateDiff-Lightning uses distillation for faster generation (4-8 steps). SparseCtrl allows controlling specific frames as keyframes for more directed animation."
        id="note-versions"
      />

      <WarningBlock
        title="SD Version Compatibility"
        content="AnimateDiff motion modules are architecture-specific: v1-v3 modules work only with SD 1.5 models. They are NOT compatible with SDXL, FLUX, or SD 2.x without dedicated motion modules trained for those architectures. Always check version compatibility before combining components."
        id="warning-compatibility"
      />
    </div>
  )
}
