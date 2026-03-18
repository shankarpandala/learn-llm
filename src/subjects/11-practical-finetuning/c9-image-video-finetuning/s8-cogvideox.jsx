import React from 'react';
import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx';
import NoteBlock from '../../../components/content/NoteBlock.jsx';
import WarningBlock from '../../../components/content/WarningBlock.jsx';
import PythonCode from '../../../components/content/PythonCode.jsx';

const SETUP_CODE = `# CogVideoX Fine-tuning Setup
# Requires: diffusers >= 0.30.0, accelerate, peft, decord
pip install diffusers accelerate peft transformers
pip install decord imageio[ffmpeg]

# CogVideoX-2b requires ~40GB VRAM for fine-tuning
# CogVideoX-5b requires ~80GB VRAM (multi-GPU recommended)

# Clone the training scripts
# git clone https://github.com/THUDM/CogVideo.git
# cd CogVideo/finetune`;

const TRAINING_CODE = `# CogVideoX LoRA Fine-tuning with diffusers
from diffusers import CogVideoXPipeline
from peft import LoraConfig, get_peft_model
import torch

# 1. Load the base model
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.bfloat16,
)

# 2. Configure LoRA for the transformer
lora_config = LoraConfig(
    r=64,
    lora_alpha=64,
    target_modules=[
        "to_q", "to_k", "to_v", "to_out.0",  # attention
        "proj_in", "proj_out",  # projections
    ],
    lora_dropout=0.0,
)

# 3. Dataset preparation
# Organize videos as:
# dataset/
#   videos/
#     001.mp4  # 6 seconds, 480x720 or 720x480
#     002.mp4
#   metadata.json  # [{"file": "001.mp4", "text": "description"}]

# 4. Training command with accelerate
# accelerate launch train_cogvideox_lora.py \\
#   --pretrained_model_name_or_path="THUDM/CogVideoX-2b" \\
#   --data_root="./dataset" \\
#   --output_dir="./cogvideox-lora" \\
#   --height=480 --width=720 \\
#   --num_frames=49 --fps=8 \\
#   --train_batch_size=1 \\
#   --gradient_accumulation_steps=4 \\
#   --learning_rate=1e-4 \\
#   --lr_scheduler="cosine" \\
#   --max_train_steps=1000 \\
#   --lora_rank=64 \\
#   --mixed_precision="bf16" \\
#   --gradient_checkpointing`;

const INFERENCE_CODE = `# Generate video with fine-tuned CogVideoX LoRA
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import torch

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Load fine-tuned LoRA
pipe.load_lora_weights("./cogvideox-lora")

video_frames = pipe(
    prompt="A golden retriever playing in autumn leaves",
    num_frames=49,
    guidance_scale=6.0,
    num_inference_steps=50,
    generator=torch.Generator("cuda").manual_seed(42),
).frames[0]

export_to_video(video_frames, "output.mp4", fps=8)`;

export default function CogVideoXFinetune() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <div>
        <h1 className="text-3xl font-extrabold tracking-tight text-gray-900 dark:text-white">
          Fine-tuning CogVideoX
        </h1>
        <p className="mt-3 text-lg text-gray-600 dark:text-gray-400">
          CogVideoX is an open-source video generation model using a 3D causal VAE and
          expert transformer blocks. Fine-tuning with LoRA lets you specialize it for
          specific visual styles or motion patterns.
        </p>
      </div>

      <DefinitionBlock
        title="CogVideoX Architecture"
        definition="CogVideoX uses a 3D causal VAE that compresses video spatiotemporally (4× temporal, 8× spatial), followed by an expert transformer with full 3D attention across all frames. The text encoder is T5-XXL."
        notation="Input video $V \in \mathbb{R}^{T \times H \times W \times 3}$ is encoded to latent $z \in \mathbb{R}^{T/4 \times H/8 \times W/8 \times C}$"
      />

      <PythonCode code={SETUP_CODE} title="Terminal — Setup" />

      <NoteBlock type="note" title="Dataset Guidelines">
        <p>
          Prepare <strong>50-200 video clips</strong>, each 6 seconds at 8 fps (49 frames).
          Resolution should be 480×720 or 720×480 (landscape/portrait).
          Include diverse scenes with consistent style. Each video needs a text caption
          describing the content and motion.
        </p>
      </NoteBlock>

      <PythonCode code={TRAINING_CODE} title="train_cogvideox_lora.py" />

      <WarningBlock title="Compute Requirements">
        <p>
          CogVideoX-2b LoRA fine-tuning needs <strong>~40GB VRAM</strong> with gradient checkpointing
          and bf16. For CogVideoX-5b, use multi-GPU with DeepSpeed ZeRO Stage 2.
          Training typically takes 4-8 hours on a single A100 for 1000 steps.
        </p>
      </WarningBlock>

      <PythonCode code={INFERENCE_CODE} title="generate_video.py" />

      <NoteBlock type="tip" title="Quality Tips">
        <p>
          Start with a small learning rate (1e-4) and monitor validation videos every 200 steps.
          CogVideoX is sensitive to caption quality — use detailed, specific descriptions.
          If motion becomes jittery, reduce the learning rate or increase LoRA rank.
        </p>
      </NoteBlock>
    </div>
  );
}
