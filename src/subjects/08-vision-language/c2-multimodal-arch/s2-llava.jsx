import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LLaVA() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LLaVA: Large Language-and-Vision Assistant</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        LLaVA is a simple yet effective architecture that connects a pretrained CLIP vision
        encoder to a pretrained LLM via a lightweight projection layer. It follows the early
        fusion paradigm where visual tokens are injected directly into the LLM's input sequence
        alongside text tokens, enabling the LLM to reason about images using its existing
        language understanding capabilities.
      </p>

      <DefinitionBlock
        title="LLaVA Architecture"
        definition="LLaVA consists of three components: (1) a frozen CLIP ViT-L/14 vision encoder that produces visual features $\mathbf{Z}_v \in \mathbb{R}^{N \times D_v}$, (2) a trainable projection layer $\mathbf{W} \in \mathbb{R}^{D_v \times D_l}$ that maps visual features to the LLM's embedding space, and (3) a pretrained LLM (e.g., LLaMA/Vicuna) that processes the combined sequence."
        notation="\( \mathbf{H}_v = \mathbf{Z}_v \mathbf{W}, \quad \text{Input} = [\mathbf{H}_v; \mathbf{H}_{\text{text}}] \)"
        id="def-llava"
      />

      <h2 className="text-2xl font-semibold">Two-Stage Training</h2>
      <p className="text-gray-700 dark:text-gray-300">
        LLaVA uses a two-stage training procedure. Stage 1 (pretraining) trains only the
        projection layer on image-caption pairs to align visual features to the LLM's embedding
        space. Stage 2 (instruction tuning) fine-tunes both the projection and the LLM on
        multimodal instruction-following data.
      </p>

      <ExampleBlock
        title="LLaVA Token Sequence"
        problem="Trace how a user query 'What is in this image?' with an attached 224x224 image flows through LLaVA."
        steps={[
          { formula: '\\text{CLIP ViT}: 224^2 / 14^2 = 256 \\text{ visual tokens} \\in \\mathbb{R}^{256 \\times 1024}', explanation: 'CLIP ViT-L/14 produces 256 patch features of dimension 1024.' },
          { formula: '\\text{Projection}: \\mathbb{R}^{256 \\times 1024} \\to \\mathbb{R}^{256 \\times 4096}', explanation: 'Linear projection maps visual dim (1024) to LLM dim (4096 for LLaMA-7B).' },
          { formula: '\\text{Concat}: [\\text{sys\\_tokens}, \\mathbf{H}_v, \\text{query\\_tokens}]', explanation: 'Visual tokens are inserted at the image placeholder position in the prompt.' },
          { formula: '\\text{LLM generates response autoregressively}', explanation: 'The LLM sees visual tokens as if they were regular embeddings and generates text.' },
        ]}
        id="example-llava-flow"
      />

      <PythonCode
        title="llava_architecture.py"
        code={`import torch
import torch.nn as nn

class LLaVAProjector(nn.Module):
    """Simple MLP projector used in LLaVA-1.5."""
    def __init__(self, vis_dim=1024, llm_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vis_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),  # Two-layer MLP in LLaVA-1.5
        )

    def forward(self, visual_features):
        return self.proj(visual_features)

class SimpleLLaVA(nn.Module):
    """Simplified LLaVA showing the data flow."""
    def __init__(self, vision_encoder, projector, llm):
        super().__init__()
        self.vision_encoder = vision_encoder  # Frozen CLIP ViT
        self.projector = projector            # Trainable
        self.llm = llm                        # Fine-tuned in stage 2

    def forward(self, images, input_ids, image_positions):
        # 1. Extract visual features
        with torch.no_grad():
            vis_features = self.vision_encoder(images)  # (B, N_patches, D_vis)

        # 2. Project to LLM space
        vis_tokens = self.projector(vis_features)       # (B, N_patches, D_llm)

        # 3. Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # 4. Insert visual tokens at image placeholder positions
        # (simplified - real impl handles variable positions)
        combined = torch.cat([
            text_embeds[:, :image_positions[0]],
            vis_tokens,
            text_embeds[:, image_positions[0]+1:],
        ], dim=1)

        # 5. Forward through LLM
        return self.llm(inputs_embeds=combined)

# Projector parameter count comparison
proj_linear = nn.Linear(1024, 4096)
proj_mlp = LLaVAProjector(1024, 4096)
print(f"Linear projector: {sum(p.numel() for p in proj_linear.parameters()) / 1e6:.1f}M")
print(f"MLP projector:    {sum(p.numel() for p in proj_mlp.parameters()) / 1e6:.1f}M")`}
        id="code-llava"
      />

      <PythonCode
        title="llava_inference.py"
        code={`# Using LLaVA with transformers library
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch

# Load LLaVA-1.5 (requires ~14GB GPU memory for 7B)
model_id = "llava-hf/llava-1.5-7b-hf"
# processor = AutoProcessor.from_pretrained(model_id)
# model = LlavaForConditionalGeneration.from_pretrained(
#     model_id, torch_dtype=torch.float16, device_map="auto"
# )

# Example inference
# image = Image.open("photo.jpg")
# prompt = "USER: <image>\\nDescribe this image in detail.\\nASSISTANT:"
# inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
# output = model.generate(**inputs, max_new_tokens=200)
# print(processor.decode(output[0], skip_special_tokens=True))

# Key hyperparameters for LLaVA variants:
configs = {
    "LLaVA-1.5-7B":  {"vision": "CLIP ViT-L/14@336", "llm": "Vicuna-7B",  "proj": "2-layer MLP"},
    "LLaVA-1.5-13B": {"vision": "CLIP ViT-L/14@336", "llm": "Vicuna-13B", "proj": "2-layer MLP"},
    "LLaVA-NeXT":    {"vision": "CLIP ViT-L/14@672", "llm": "Various",     "proj": "2-layer MLP"},
}
for name, cfg in configs.items():
    print(f"{name}: {cfg}")`}
        id="code-llava-inference"
      />

      <NoteBlock
        type="note"
        title="LLaVA Versions"
        content="LLaVA-1.0 (2023) used a single linear projection. LLaVA-1.5 improved with a 2-layer MLP projector and higher resolution (336px). LLaVA-NeXT further increased resolution with dynamic tiling, splitting high-res images into multiple 336px tiles to preserve fine detail."
        id="note-llava-versions"
      />

      <WarningBlock
        title="Projection Layer Alignment"
        content="The projection layer must be pretrained before instruction tuning. Skipping Stage 1 (visual-language alignment) and going directly to instruction tuning results in significantly worse performance because the LLM cannot interpret raw CLIP features."
        id="warning-llava-alignment"
      />
    </div>
  )
}
