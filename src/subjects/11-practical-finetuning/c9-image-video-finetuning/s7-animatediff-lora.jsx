import React from 'react';
import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx';
import ExampleBlock from '../../../components/content/ExampleBlock.jsx';
import NoteBlock from '../../../components/content/NoteBlock.jsx';
import WarningBlock from '../../../components/content/WarningBlock.jsx';
import PythonCode from '../../../components/content/PythonCode.jsx';

const TRAINING_CODE = `# AnimateDiff LoRA Training with diffusers
from diffusers import AnimateDiffPipeline, MotionAdapter
from diffusers.training_utils import EMAModel
import torch

# 1. Prepare your video dataset
# Videos should be 16-24 frames, 512x512, same motion style
# Store as folders of PNG frames or MP4 files

# Dataset structure:
# data/
#   video_001/ frame_00.png, frame_01.png, ...
#   video_002/ frame_00.png, frame_01.png, ...

# 2. Training config
config = {
    "pretrained_model": "runwayml/stable-diffusion-v1-5",
    "motion_module": "guoyww/animatediff-motion-adapter-v1-5-3",
    "output_dir": "./animatediff-lora-output",
    "train_data": "./data",
    "resolution": 512,
    "num_frames": 16,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "max_train_steps": 2000,
    "lora_rank": 32,
    "validation_prompt": "a cat walking, smooth motion",
}

# 3. Launch training (via accelerate)
# accelerate launch train_animatediff_lora.py \\
#   --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\
#   --motion_module="guoyww/animatediff-motion-adapter-v1-5-3" \\
#   --train_data_dir="./data" \\
#   --output_dir="./animatediff-lora-output" \\
#   --resolution=512 --num_frames=16 \\
#   --train_batch_size=1 --learning_rate=1e-4 \\
#   --max_train_steps=2000 --lora_rank=32`;

const INFERENCE_CODE = `# Inference with trained AnimateDiff LoRA
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
import torch

# Load base pipeline
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-3"
)
pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float16,
).to("cuda")

# Load your trained LoRA weights
pipe.load_lora_weights("./animatediff-lora-output", adapter_name="motion_lora")
pipe.set_adapters(["motion_lora"], adapter_weights=[0.8])

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

# Generate animation
output = pipe(
    prompt="a cat walking gracefully, smooth motion",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cuda").manual_seed(42),
)
export_to_gif(output.frames[0], "animation.gif")`;

export default function AnimateDiffLora() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <div>
        <h1 className="text-3xl font-extrabold tracking-tight text-gray-900 dark:text-white">
          AnimateDiff LoRA Training
        </h1>
        <p className="mt-3 text-lg text-gray-600 dark:text-gray-400">
          Train motion-specific LoRA adapters for AnimateDiff to create custom animation
          styles — from character movements to camera motions — while keeping the base
          model frozen.
        </p>
      </div>

      <DefinitionBlock
        title="AnimateDiff Motion LoRA"
        definition="A low-rank adapter applied to the temporal attention layers of AnimateDiff's motion module, enabling fine-tuning of specific motion patterns with only 5-20MB of additional weights."
        notation="Motion LoRA adapts only the temporal self-attention: $W_{temporal} = W_0 + BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$"
      />

      <NoteBlock type="tip" title="Dataset Requirements">
        <p>
          For AnimateDiff LoRA training, prepare <strong>50-200 short video clips</strong> (2-3 seconds each)
          showing the target motion style. Extract frames at the model's native frame rate (typically 8 fps).
          All videos should be the same resolution (512×512) and frame count (16 frames).
        </p>
      </NoteBlock>

      <PythonCode code={TRAINING_CODE} title="train_animatediff_lora.py" />

      <WarningBlock title="VRAM Requirements">
        <p>
          AnimateDiff LoRA training requires <strong>24GB+ VRAM</strong> due to temporal attention layers.
          Use gradient checkpointing and mixed precision (fp16/bf16) to fit on a single GPU.
          Reduce <code>num_frames</code> to 8 if memory is tight.
        </p>
      </WarningBlock>

      <ExampleBlock
        title="LoRA Weight Merging for Motion Styles"
        problem="Combine a camera-pan LoRA (weight 0.6) with a zoom LoRA (weight 0.4) for a pan-and-zoom effect."
        steps={[
          { formula: '$w_{combined} = 0.6 \\cdot w_{pan} + 0.4 \\cdot w_{zoom}$', explanation: 'Linear combination of LoRA weights' },
          { explanation: 'Use pipe.set_adapters(["pan", "zoom"], adapter_weights=[0.6, 0.4])' },
          { explanation: 'Adjust weights at inference time — no retraining needed' },
        ]}
      />

      <PythonCode code={INFERENCE_CODE} title="inference.py" />
    </div>
  );
}
