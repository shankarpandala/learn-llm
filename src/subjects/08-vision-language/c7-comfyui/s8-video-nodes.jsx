import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function VideoNodes() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Video Generation Nodes in ComfyUI</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        ComfyUI supports video generation through AnimateDiff nodes, SVD (Stable Video
        Diffusion) nodes, and CogVideoX integration. Video workflows extend image workflows
        by adding temporal dimensions to the latent space, requiring specialized loaders,
        samplers, and output nodes for handling frame sequences.
      </p>

      <DefinitionBlock
        title="Video Workflow in ComfyUI"
        definition="A video workflow in ComfyUI generates a batch of temporally coherent frames by processing a 3D latent tensor (frames x height x width x channels). The AnimateDiff approach injects motion modules into the standard SD pipeline, while SVD uses a dedicated image-to-video model. Output nodes combine frames into GIF, MP4, or image sequences."
        id="def-video-workflow"
      />

      <h2 className="text-2xl font-semibold">AnimateDiff Nodes</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The AnimateDiff custom node pack adds motion module loading, frame control, and video
        output nodes. It integrates with the standard SD 1.5 pipeline, adding temporal
        attention to the existing U-Net.
      </p>

      <ExampleBlock
        title="AnimateDiff Node Chain"
        problem="Build an AnimateDiff video workflow in ComfyUI."
        steps={[
          { formula: '\\text{CheckpointLoader} \\to \\text{AnimateDiffLoader (motion module)}', explanation: 'Load base SD model, then inject motion module.' },
          { formula: '\\text{CLIPTextEncode} \\to \\text{KSampler (batch\\_size=16)}', explanation: 'Standard prompt encoding, but latent batch = number of frames.' },
          { formula: '\\text{VAEDecode} \\to \\text{AnimateDiffCombine (output)}', explanation: 'Decode all frames, combine into video file.' },
        ]}
        id="example-animatediff-nodes"
      />

      <PythonCode
        title="video_workflow.py"
        code={`# AnimateDiff video workflow for ComfyUI
def build_animatediff_workflow(
    checkpoint="sd_v1-5.safetensors",
    motion_module="mm_sd_v15_v3.safetensors",
    prompt="a cat walking on a garden path, cinematic",
    negative="blurry, static, low quality",
    num_frames=16, fps=8,
    steps=25, cfg=7.5, seed=42,
):
    """AnimateDiff text-to-video workflow."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": checkpoint}},
        # Load and apply motion module
        "2": {"class_type": "ADE_AnimateDiffLoaderWithContext",
              "inputs": {
                  "model": ["1", 0],
                  "model_name": motion_module,
                  "context_options": ["3", 0],
              }},
        # Context options for sliding window
        "3": {"class_type": "ADE_StandardStaticContextOptions",
              "inputs": {
                  "context_length": 16,
                  "context_overlap": 4,
              }},
        # Prompts
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["1", 1]}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["1", 1]}},
        # Empty latent with frame count as batch size
        "6": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 512, "height": 512, "batch_size": num_frames}},
        # Sample
        "7": {"class_type": "KSampler",
              "inputs": {
                  "model": ["2", 0],  # Motion-module-enhanced model
                  "positive": ["4", 0], "negative": ["5", 0],
                  "latent_image": ["6", 0],
                  "seed": seed, "steps": steps, "cfg": cfg,
                  "sampler_name": "euler_ancestral", "scheduler": "normal",
                  "denoise": 1.0,
              }},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        # Combine frames into video
        "9": {"class_type": "ADE_AnimateDiffCombine",
              "inputs": {
                  "images": ["8", 0],
                  "frame_rate": fps,
                  "format": "video/h264-mp4",
                  "pingpong": False,
              }},
    }

# SVD (Stable Video Diffusion) workflow
def build_svd_workflow(
    svd_model="svd_xt_1_1.safetensors",
    input_image="input/landscape.png",
    num_frames=25, fps=7,
    motion_bucket=127, augmentation=0.02,
    steps=25, cfg=2.5, seed=42,
):
    """SVD image-to-video workflow."""
    return {
        "1": {"class_type": "ImageOnlyCheckpointLoader",
              "inputs": {"ckpt_name": svd_model}},
        "2": {"class_type": "LoadImage",
              "inputs": {"image": input_image}},
        "3": {"class_type": "SVD_img2vid_Conditioning",
              "inputs": {
                  "init_image": ["2", 0],
                  "vae": ["1", 2],
                  "width": 1024, "height": 576,
                  "video_frames": num_frames,
                  "motion_bucket_id": motion_bucket,
                  "fps": fps,
                  "augmentation_level": augmentation,
              }},
        "4": {"class_type": "KSampler",
              "inputs": {
                  "model": ["1", 0],
                  "positive": ["3", 0], "negative": ["3", 1],
                  "latent_image": ["3", 2],
                  "seed": seed, "steps": steps, "cfg": cfg,
                  "sampler_name": "euler", "scheduler": "karras",
                  "denoise": 1.0,
              }},
        "5": {"class_type": "VAEDecode",
              "inputs": {"samples": ["4", 0], "vae": ["1", 2]}},
        "6": {"class_type": "SaveAnimatedWEBP",
              "inputs": {"images": ["5", 0], "fps": fps,
                         "filename_prefix": "svd_output"}},
    }

print(f"AnimateDiff workflow: {len(build_animatediff_workflow())} nodes")
print(f"SVD workflow: {len(build_svd_workflow())} nodes")`}
        id="code-video-workflow"
      />

      <NoteBlock
        type="tip"
        title="Sliding Window for Longer Videos"
        content="AnimateDiff's context_length limits frames processed simultaneously (typically 16). For longer videos, use sliding window context (context_overlap=4) to generate 32+ frames by processing overlapping windows and blending them. This enables longer videos without excessive VRAM."
        id="note-sliding-window"
      />

      <WarningBlock
        title="Video VRAM Usage"
        content="Video generation requires significantly more VRAM than single images. AnimateDiff with 16 frames at 512x512 needs ~8GB. SVD at 1024x576 with 25 frames needs ~12-16GB. Always enable VAE tiling and consider reducing resolution for initial tests."
        id="warning-video-vram"
      />
    </div>
  )
}
