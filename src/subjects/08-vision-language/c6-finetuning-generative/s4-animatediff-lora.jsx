import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function AnimateDiffLoRA() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">AnimateDiff LoRA: Training Custom Motion</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        AnimateDiff LoRA applies LoRA to the temporal attention layers of AnimateDiff's motion
        module, enabling custom motion patterns (e.g., specific camera movements, character
        animations, or physics behaviors) without full retraining. This combines the modularity
        of AnimateDiff with the efficiency of LoRA fine-tuning.
      </p>

      <DefinitionBlock
        title="Motion LoRA"
        definition="Motion LoRA applies low-rank adaptation specifically to the temporal attention layers of AnimateDiff's motion module. Given temporal attention weights $\mathbf{W}_{\text{temp}}$, the motion LoRA modifies them as $\mathbf{W}_{\text{temp}} + \alpha \mathbf{B}_{\text{motion}}\mathbf{A}_{\text{motion}}$ to encode specific motion patterns from reference video clips."
        id="def-motion-lora"
      />

      <h2 className="text-2xl font-semibold">Training Pipeline</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Motion LoRA training uses short video clips (2-4 seconds) that demonstrate the desired
        motion. The base motion module and SD U-Net are frozen; only the LoRA layers on
        temporal attention are trained. This requires significantly less data and compute
        than training a full motion module.
      </p>

      <ExampleBlock
        title="Motion LoRA Training Setup"
        problem="Train a motion LoRA for smooth camera pan-left movement."
        steps={[
          { formula: '\\text{Collect 50-100 short clips with leftward camera pans}', explanation: 'Curated dataset of the specific motion pattern.' },
          { formula: '\\text{Freeze: SD U-Net + base motion module}', explanation: 'Only temporal attention LoRA layers are trainable.' },
          { formula: '\\text{Train: rank 64, 2000 steps, lr=1e-4}', explanation: 'Short training on temporal layers only.' },
          { formula: '\\text{Output: ~5-15MB LoRA file}', explanation: 'Compact motion control that works with any SD 1.5 model.' },
        ]}
        id="example-motion-lora-setup"
      />

      <PythonCode
        title="animatediff_lora_usage.py"
        code={`from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
import torch

# Load base AnimateDiff pipeline
# adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-3")
# pipe = AnimateDiffPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     motion_adapter=adapter,
#     torch_dtype=torch.float16,
# ).to("cuda")
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Load motion LoRA for specific motion
# pipe.load_lora_weights(
#     "guoyww/animatediff-motion-lora-zoom-out",
#     adapter_name="zoom_out"
# )
# pipe.set_adapters(["zoom_out"], [1.0])

# # Stack with a style LoRA
# pipe.load_lora_weights("custom_style_lora", adapter_name="style")
# pipe.set_adapters(["zoom_out", "style"], [1.0, 0.7])

# # Generate video with controlled motion
# output = pipe(
#     prompt="A beautiful forest clearing, sunlight through trees",
#     num_frames=16,
#     num_inference_steps=25,
#     guidance_scale=7.5,
# )
# from diffusers.utils import export_to_gif
# export_to_gif(output.frames[0], "forest_zoom_out.gif")

# Available motion LoRAs from the community
motion_loras = {
    "zoom-out":      "Camera smoothly zooms out",
    "zoom-in":       "Camera smoothly zooms in",
    "pan-left":      "Camera pans to the left",
    "pan-right":     "Camera pans to the right",
    "tilt-up":       "Camera tilts upward",
    "tilt-down":     "Camera tilts downward",
    "rolling":       "Rolling/rotating camera motion",
}

print("Available motion LoRAs:")
for name, desc in motion_loras.items():
    print(f"  {name}: {desc}")

# Custom motion LoRA training config
training_config = {
    "base_model": "runwayml/stable-diffusion-v1-5",
    "motion_module": "animatediff-motion-adapter-v1-5-3",
    "train_data": "50-100 video clips showing desired motion",
    "target_layers": "temporal_attention (Q, K, V projections)",
    "rank": 64,
    "learning_rate": 1e-4,
    "train_steps": 2000,
    "batch_size": 1,
    "num_frames": 16,
    "resolution": 256,  # Lower res for training efficiency
    "gradient_checkpointing": True,
}

print("\\nTraining config:")
for k, v in training_config.items():
    print(f"  {k}: {v}")`}
        id="code-motion-lora"
      />

      <NoteBlock
        type="tip"
        title="Combining Motion and Style LoRAs"
        content="AnimateDiff's modularity shines when stacking LoRAs: use a motion LoRA for camera control, a style LoRA for visual appearance, and even a character LoRA for consistent subjects. Adjust individual weights to balance each effect. Start with all weights at 1.0 and reduce if effects conflict."
        id="note-stacking-loras"
      />

      <WarningBlock
        title="Training Data Quality"
        content="Motion LoRA quality depends heavily on training data consistency. Clips should show the same type of motion with minimal variation. Mixing different motions (e.g., panning mixed with zooming) will produce a confused LoRA. Use optical flow analysis to filter and curate motion-consistent clips."
        id="warning-data-quality"
      />
    </div>
  )
}
