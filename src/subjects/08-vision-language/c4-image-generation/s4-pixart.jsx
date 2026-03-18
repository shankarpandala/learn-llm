import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function PixArt() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">PixArt-Alpha: Efficient Training of Diffusion Transformers</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        PixArt-Alpha demonstrates that high-quality text-to-image generation is possible with
        dramatically less training compute than Stable Diffusion or DALL-E. Through a three-stage
        training strategy and efficient DiT architecture, PixArt-Alpha achieves competitive
        quality at approximately 2% of the training cost of SD, making it an important milestone
        for accessible image generation research.
      </p>

      <DefinitionBlock
        title="PixArt-Alpha"
        definition="PixArt-Alpha is a Diffusion Transformer (DiT) model that uses cross-attention for T5 text conditioning and adaLN-single for timestep conditioning. It achieves efficient training through: (1) decomposed training with a pretrained class-conditional DiT as initialization, (2) efficient T5 text conditioning, and (3) a high-quality curated dataset (SAM-LLaVA-Captions)."
        id="def-pixart"
      />

      <h2 className="text-2xl font-semibold">Three-Stage Training</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Instead of training from scratch on text-image pairs (expensive), PixArt decomposes
        training into learning pixel distributions, text-image alignment, and aesthetic quality
        in separate stages.
      </p>

      <ExampleBlock
        title="PixArt Training Efficiency"
        problem="Compare training costs of PixArt-Alpha vs Stable Diffusion v1.5."
        steps={[
          { formula: '\\text{SD 1.5: } 6{,}250 \\text{ A100 GPU-days}', explanation: 'Stable Diffusion required massive compute for training from scratch.' },
          { formula: '\\text{PixArt-}\\alpha\\text{: } 133 \\text{ A100 GPU-days}', explanation: 'PixArt achieves comparable quality at ~2% of the cost.' },
          { formula: '\\text{CO}_2 \\text{ reduction: } 90\\% \\text{ lower emissions}', explanation: 'Efficient training directly reduces environmental impact.' },
        ]}
        id="example-pixart-efficiency"
      />

      <PythonCode
        title="pixart_generation.py"
        code={`from diffusers import PixArtAlphaPipeline
import torch

# PixArt-Alpha (600M params DiT)
# pipe = PixArtAlphaPipeline.from_pretrained(
#     "PixArt-alpha/PixArt-XL-2-1024-MS",
#     torch_dtype=torch.float16,
# ).to("cuda")

# # Generate with PixArt-Alpha
# image = pipe(
#     prompt="A small cactus with a happy face in a terracotta pot, pixel art style",
#     num_inference_steps=20,
#     guidance_scale=4.5,
# ).images[0]
# image.save("pixart_cactus.png")

# PixArt-Sigma (improved version)
# pipe_sigma = PixArtAlphaPipeline.from_pretrained(
#     "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
#     torch_dtype=torch.float16,
# ).to("cuda")

# Architecture details
import torch.nn as nn

class AdaLNSingle(nn.Module):
    """Adaptive LayerNorm with single set of parameters (PixArt style).
    More parameter-efficient than adaLN-Zero used in original DiT."""
    def __init__(self, dim, cond_dim=None):
        super().__init__()
        cond_dim = cond_dim or dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim),  # scale, shift for pre-norm, gate for attn & ff
        )

    def forward(self, x, cond):
        params = self.proj(cond).unsqueeze(1)  # (B, 1, 6*D)
        shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff = params.chunk(6, dim=-1)
        # Pre-attention norm
        x_normed = self.norm(x) * (1 + scale_attn) + shift_attn
        return x_normed, gate_attn, shift_ff, scale_ff, gate_ff

# Compare DiT variants
models = {
    "DiT (original)":  {"conditioning": "adaLN-Zero", "text": "Class labels", "params": "675M"},
    "PixArt-Alpha":    {"conditioning": "adaLN-Single", "text": "T5 cross-attn", "params": "600M"},
    "PixArt-Sigma":    {"conditioning": "adaLN-Single", "text": "T5 cross-attn", "params": "600M"},
    "FLUX":            {"conditioning": "adaLN + joint", "text": "T5 + CLIP joint", "params": "12B"},
}
for name, info in models.items():
    print(f"{name}: {info}")`}
        id="code-pixart"
      />

      <NoteBlock
        type="note"
        title="PixArt-Sigma Improvements"
        content="PixArt-Sigma improves upon Alpha with better VAE (SDXL's VAE), support for various aspect ratios, and a token compression mechanism that reduces computation for higher resolutions. It also introduces a 'weak-to-strong' training paradigm for efficiently scaling resolution."
        id="note-pixart-sigma"
      />

      <NoteBlock
        type="tip"
        title="Efficient Training Recipe"
        content="PixArt's key insight is decomposed training: first learn to generate images (class-conditional pretraining is cheap), then learn text alignment (fine-tuning is cheaper than joint training). This principle applies broadly -- always leverage pretrained components when possible."
        id="note-efficient-recipe"
      />

      <WarningBlock
        title="T5 Text Encoder Memory"
        content="PixArt uses T5-XXL (4.7B params) as text encoder, which alone requires ~10GB in float16. For inference on limited hardware, precompute and cache T5 embeddings, or use the smaller T5-large variant with some quality trade-off."
        id="warning-t5-memory"
      />
    </div>
  )
}
