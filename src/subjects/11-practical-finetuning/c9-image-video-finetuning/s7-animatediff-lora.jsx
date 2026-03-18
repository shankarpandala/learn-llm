import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function AnimateDiffLora() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">AnimateDiff LoRA Training</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        AnimateDiff adds temporal motion modules to Stable Diffusion to generate short
        video clips. Training LoRA adapters for AnimateDiff lets you create custom motion
        styles and domain-specific animations without full model training. This section
        covers the practical workflow for training motion LoRAs.
      </p>

      <DefinitionBlock
        title="AnimateDiff"
        definition="AnimateDiff inserts temporal attention layers (motion modules) into a pre-trained Stable Diffusion UNet. These modules learn to generate temporally coherent frames. A motion LoRA fine-tunes these temporal layers to learn specific motion patterns while keeping spatial generation frozen."
        id="def-animatediff"
      />

      <ExampleBlock
        title="AnimateDiff LoRA Training Setup"
        problem="What data and configuration do you need for motion LoRA training?"
        steps={[
          { formula: '\\text{Training data: 50-200 short video clips (2-4 seconds)}', explanation: 'Clips should show the target motion pattern. Extract at 8 fps for 16-frame sequences.' },
          { formula: '\\text{Resolution: 256 or 512 pixels}', explanation: 'Lower resolution is common for video training due to the memory multiplier per frame.' },
          { formula: '\\text{Frames per sample: 16}', explanation: 'Standard AnimateDiff uses 16 frames. This is the temporal window size.' },
          { formula: '\\text{VRAM: 24-40 GB depending on resolution}', explanation: 'Video training multiplies memory by the number of frames. Use gradient checkpointing.' },
        ]}
        id="example-animatediff-setup"
      />

      <PythonCode
        title="prepare_video_dataset.py"
        code={`import os
import subprocess
from pathlib import Path

def extract_training_clips(video_path, output_dir, fps=8, num_frames=16,
                           resolution=512):
    """Extract training clips from a video file."""
    os.makedirs(output_dir, exist_ok=True)
    clip_idx = 0

    # Get video duration
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True
    )
    duration = float(result.stdout.strip())
    clip_duration = num_frames / fps

    for start in range(0, int(duration - clip_duration), int(clip_duration // 2)):
        clip_dir = os.path.join(output_dir, f"clip_{clip_idx:04d}")
        os.makedirs(clip_dir, exist_ok=True)

        subprocess.run([
            "ffmpeg", "-y", "-ss", str(start),
            "-i", video_path,
            "-t", str(clip_duration),
            "-vf", f"fps={fps},scale={resolution}:{resolution}:"
                   f"force_original_aspect_ratio=decrease,"
                   f"pad={resolution}:{resolution}:-1:-1",
            "-frames:v", str(num_frames),
            os.path.join(clip_dir, "frame_%04d.png")
        ], capture_output=True)

        frames = list(Path(clip_dir).glob("*.png"))
        if len(frames) == num_frames:
            clip_idx += 1
        else:
            import shutil
            shutil.rmtree(clip_dir)

    print(f"Extracted {clip_idx} training clips from {video_path}")
    return clip_idx

video_dir = "./raw_videos"
output_dir = "./animatediff_training_data"
total_clips = 0

for video in Path(video_dir).glob("*.mp4"):
    clips = extract_training_clips(str(video), output_dir, fps=8, num_frames=16)
    total_clips += clips

print(f"Total training clips: {total_clips}")`}
        id="code-prepare-video"
      />

      <PythonCode
        title="train_animatediff_lora.py"
        code={`# AnimateDiff LoRA training using diffusers

TRAIN_CMD = """
accelerate launch diffusers/examples/animatediff/train_animatediff.py \\
    --pretrained_model_name_or_path="SG161222/Realistic_Vision_V5.1_noVAE" \\
    --motion_module="guoyww/animatediff-motion-adapter-v1-5-3" \\
    --train_data_dir="./animatediff_training_data" \\
    --output_dir="./animatediff-lora-output" \\
    --resolution=512 \\
    --train_batch_size=1 \\
    --gradient_accumulation_steps=4 \\
    --max_train_steps=2000 \\
    --learning_rate=1e-4 \\
    --lr_scheduler="cosine" \\
    --lr_warmup_steps=100 \\
    --rank=32 \\
    --seed=42 \\
    --mixed_precision="bf16" \\
    --gradient_checkpointing \\
    --enable_xformers_memory_efficient_attention \\
    --num_frames=16 \\
    --checkpointing_steps=500
"""

# Inference with trained motion LoRA
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

def generate_animation(motion_lora_path, prompt, num_frames=16):
    """Generate animation using trained motion LoRA."""
    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-3",
        torch_dtype=torch.float16,
    )

    pipe = AnimateDiffPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    ).to("cuda")

    pipe.load_lora_weights(motion_lora_path, adapter_name="motion_lora")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    output = pipe(
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=25,
        guidance_scale=7.5,
    )

    export_to_gif(output.frames[0], "animation_output.gif")
    print("Saved animation to animation_output.gif")

print(TRAIN_CMD)`}
        id="code-train-animatediff"
      />

      <NoteBlock
        type="tip"
        title="Start with Short Low-Res Clips"
        content="Begin training at 256x256 resolution with 16 frames to iterate quickly. Once you have good motion patterns, scale up to 512x512. The motion patterns learned at lower resolution transfer well to higher resolution inference."
        id="note-start-small"
      />

      <WarningBlock
        title="Temporal Flickering"
        content="The most common AnimateDiff failure mode is temporal flickering where frames are individually good but inconsistent with each other. This is usually caused by too few training clips, too high a learning rate, or insufficient training steps. Reduce LR to 5e-5 and ensure at least 50 diverse clips."
        id="warning-flickering"
      />

      <NoteBlock
        type="note"
        title="Motion Module Compatibility"
        content="AnimateDiff motion LoRAs are tied to a specific motion module version. A LoRA trained with motion-adapter-v1-5-3 may not work with v1-5-2. Always document which motion module version was used during training."
        id="note-compatibility"
      />
    </div>
  )
}
