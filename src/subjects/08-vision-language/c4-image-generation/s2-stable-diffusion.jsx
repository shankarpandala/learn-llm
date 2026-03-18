import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function StableDiffusion() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Stable Diffusion Architecture</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Stable Diffusion performs diffusion in a compressed latent space rather than pixel space,
        making it dramatically more efficient. The architecture consists of three main components:
        a Variational Autoencoder (VAE) for image compression, a U-Net for noise prediction in
        latent space, and a CLIP text encoder for conditioning on text prompts.
      </p>

      <DefinitionBlock
        title="Latent Diffusion Model"
        definition="A Latent Diffusion Model (LDM) operates in the latent space of a pretrained autoencoder. Given an image $\mathbf{x}$, the encoder produces $\mathbf{z} = \mathcal{E}(\mathbf{x})$ with spatial compression factor $f$ (typically $f=8$). Diffusion is applied to $\mathbf{z}$, and the decoder reconstructs: $\hat{\mathbf{x}} = \mathcal{D}(\hat{\mathbf{z}})$."
        notation="\( \mathbf{z} \in \mathbb{R}^{h/f \times w/f \times c} \) where typically \( c = 4 \) latent channels"
        id="def-ldm"
      />

      <h2 className="text-2xl font-semibold">The Three Components</h2>

      <h3 className="text-xl font-medium">1. VAE (Autoencoder)</h3>
      <p className="text-gray-700 dark:text-gray-300">
        The VAE compresses a 512x512x3 image to a 64x64x4 latent representation (64x compression).
        This makes diffusion computationally feasible on consumer GPUs.
      </p>
      <BlockMath math="\text{Encoder: } \mathbb{R}^{512 \times 512 \times 3} \to \mathbb{R}^{64 \times 64 \times 4}, \quad \text{Decoder: } \mathbb{R}^{64 \times 64 \times 4} \to \mathbb{R}^{512 \times 512 \times 3}" />

      <h3 className="text-xl font-medium">2. U-Net (Noise Predictor)</h3>
      <p className="text-gray-700 dark:text-gray-300">
        The U-Net predicts noise <InlineMath math="\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c})" /> in
        latent space, conditioned on timestep <InlineMath math="t" /> and text embedding{' '}
        <InlineMath math="\mathbf{c}" />. Cross-attention layers inject text conditioning
        into the U-Net's intermediate representations.
      </p>

      <h3 className="text-xl font-medium">3. CLIP Text Encoder</h3>
      <p className="text-gray-700 dark:text-gray-300">
        The CLIP text encoder converts the text prompt into a sequence of embeddings that
        condition the U-Net via cross-attention at multiple resolution levels.
      </p>

      <ExampleBlock
        title="Stable Diffusion Generation Pipeline"
        problem="Trace the steps to generate a 512x512 image from a text prompt."
        steps={[
          { formula: '\\text{Encode text: } \\mathbf{c} = \\text{CLIP}_{\\text{text}}(\\text{prompt}) \\in \\mathbb{R}^{77 \\times 768}', explanation: 'Text prompt tokenized and encoded to 77 token embeddings.' },
          { formula: '\\mathbf{z}_T \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I}), \\quad \\mathbf{z}_T \\in \\mathbb{R}^{64 \\times 64 \\times 4}', explanation: 'Start with random noise in latent space.' },
          { formula: '\\text{For } t = T, T-1, \\ldots, 1: \\mathbf{z}_{t-1} = \\text{denoise}(\\mathbf{z}_t, t, \\mathbf{c})', explanation: 'Iteratively denoise using U-Net predictions (typically 20-50 steps).' },
          { formula: '\\hat{\\mathbf{x}} = \\text{VAE}_{\\text{dec}}(\\mathbf{z}_0) \\in \\mathbb{R}^{512 \\times 512 \\times 3}', explanation: 'Decode the clean latent to pixel space.' },
        ]}
        id="example-sd-pipeline"
      />

      <PythonCode
        title="stable_diffusion_pipeline.py"
        code={`from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch

# Load SD 1.5 pipeline
# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     torch_dtype=torch.float16,
#     safety_checker=None,
# ).to("cuda")

# # Use DDIM scheduler for faster inference (50 steps instead of 1000)
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# # Generate image
# prompt = "A serene mountain lake at sunset, photorealistic, 8k"
# negative_prompt = "blurry, low quality, distorted"
# image = pipe(
#     prompt,
#     negative_prompt=negative_prompt,
#     num_inference_steps=50,
#     guidance_scale=7.5,      # Classifier-free guidance strength
#     height=512, width=512,
#     generator=torch.Generator("cuda").manual_seed(42),
# ).images[0]
# image.save("mountain_lake.png")

# Understanding classifier-free guidance (CFG)
# The model runs twice: conditioned and unconditioned
# noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

# SD component sizes (v1.5):
components = {
    "CLIP Text Encoder": {"params": "123M", "role": "Text -> 77x768 embeddings"},
    "VAE Encoder":       {"params": "34M",  "role": "512x512x3 -> 64x64x4 latent"},
    "VAE Decoder":       {"params": "49M",  "role": "64x64x4 latent -> 512x512x3"},
    "U-Net":             {"params": "860M", "role": "Noise prediction in latent space"},
}
total = 0
for name, info in components.items():
    params = float(info['params'].rstrip('M'))
    total += params
    print(f"{name}: {info['params']} - {info['role']}")
print(f"\\nTotal: ~{total:.0f}M parameters")`}
        id="code-sd-pipeline"
      />

      <NoteBlock
        type="intuition"
        title="Why Latent Space?"
        content="Diffusing in pixel space for a 512x512 image means operating on 786,432 dimensions. In latent space (64x64x4), it is only 16,384 dimensions -- a 48x reduction. The VAE preserves perceptual quality while compressing away imperceptible high-frequency details, making diffusion both faster and more focused on meaningful image structure."
        id="note-latent-intuition"
      />

      <WarningBlock
        title="VAE Quality Limitations"
        content="The VAE is a bottleneck for image quality. Fine details like small text, faces at low resolution, and intricate patterns can be lost during VAE encoding/decoding. This is why Stable Diffusion sometimes produces blurry faces or garbled text, independent of the diffusion quality."
        id="warning-vae-quality"
      />
    </div>
  )
}
