import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function TextualInversion() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Textual Inversion</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Textual Inversion learns a new "word" in the text encoder's embedding space to
        represent a specific concept (object, style, or texture) from a few example images.
        Unlike DreamBooth or LoRA, it does not modify any model weights -- only a single
        embedding vector is optimized. This produces extremely small files (a few KB) but
        has limited expressiveness.
      </p>

      <DefinitionBlock
        title="Textual Inversion"
        definition="Textual Inversion optimizes a new embedding vector $v^* \in \mathbb{R}^{D}$ in the text encoder's token embedding space to represent a target concept. The text encoder and diffusion model remain frozen. The training objective is: $v^* = \arg\min_v \mathbb{E}_{t, \epsilon}[\| \epsilon - \epsilon_\theta(\mathbf{x}_t, t, c_\theta(\text{'a photo of } S^*\text{'})) \|^2]$ where $S^*$ maps to $v^*$."
        id="def-textual-inversion"
      />

      <h2 className="text-2xl font-semibold">How It Works</h2>
      <p className="text-gray-700 dark:text-gray-300">
        A new token (e.g., {"<my-concept>"}) is added to the tokenizer's vocabulary. Its
        embedding vector is initialized (randomly or from a related word) and optimized via
        gradient descent while all other parameters remain frozen. The learned embedding
        captures the concept in the text encoder's latent space.
      </p>

      <ExampleBlock
        title="Textual Inversion vs DreamBooth vs LoRA"
        problem="Compare the three personalization methods on key dimensions."
        steps={[
          { formula: '\\text{Textual Inv: 1 vector (768D)} \\approx 3\\text{KB file}', explanation: 'Only learns an embedding. Smallest, least expressive.' },
          { formula: '\\text{LoRA: rank-decomposed matrices} \\approx 2\\text{-}200\\text{MB}', explanation: 'Modifies attention layers. Good balance of quality and size.' },
          { formula: '\\text{DreamBooth: full model fine-tune} \\approx 2\\text{-}7\\text{GB}', explanation: 'Highest quality but largest storage and most prone to overfitting.' },
        ]}
        id="example-comparison"
      />

      <PythonCode
        title="textual_inversion_training.py"
        code={`# Textual inversion training with diffusers
# accelerate launch textual_inversion.py \\
#   --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\
#   --train_data_dir="./concept_images" \\
#   --learnable_property="object" \\
#   --placeholder_token="<my-cat>" \\
#   --initializer_token="cat" \\
#   --resolution=512 \\
#   --train_batch_size=4 \\
#   --max_train_steps=3000 \\
#   --learning_rate=5e-4

import torch
import torch.nn as nn

class TextualInversionTrainer:
    """Simplified textual inversion training logic."""

    def __init__(self, tokenizer, text_encoder, placeholder="<my-concept>",
                 initializer="object"):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        # Add new token to tokenizer
        num_added = tokenizer.add_tokens([placeholder])
        assert num_added == 1, "Token already exists"
        self.placeholder_id = tokenizer.convert_tokens_to_ids(placeholder)

        # Resize text encoder embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialize from existing token
        init_id = tokenizer.encode(initializer, add_special_tokens=False)[0]
        with torch.no_grad():
            embeddings = text_encoder.get_input_embeddings()
            embeddings.weight[self.placeholder_id] = embeddings.weight[init_id].clone()

        # Freeze everything except the new embedding
        for param in text_encoder.parameters():
            param.requires_grad = False
        embeddings.weight[self.placeholder_id].requires_grad = True

    def get_trainable_params(self):
        embeddings = self.text_encoder.get_input_embeddings()
        return [embeddings.weight[self.placeholder_id]]

    def save_embedding(self, path):
        embeddings = self.text_encoder.get_input_embeddings()
        learned = embeddings.weight[self.placeholder_id].detach().cpu()
        torch.save({"embedding": learned}, path)
        print(f"Saved embedding: {learned.shape} ({learned.numel() * 4 / 1024:.1f} KB)")

# Demo: show embedding size comparison
embedding_dim = 768  # CLIP / SD 1.5
sdxl_dim = 1280      # SDXL CLIP-G

print("Storage comparison:")
print(f"  Textual Inversion (SD 1.5): {embedding_dim * 4 / 1024:.1f} KB")
print(f"  Textual Inversion (SDXL):   {sdxl_dim * 4 / 1024:.1f} KB")
print(f"  LoRA (rank 32, SD 1.5):     ~10-50 MB")
print(f"  DreamBooth (SD 1.5):        ~2 GB")
print(f"  DreamBooth (SDXL):          ~7 GB")`}
        id="code-textual-inversion"
      />

      <NoteBlock
        type="intuition"
        title="Embedding Space Geometry"
        content="Textual inversion works because the CLIP text embedding space is structured: similar concepts cluster together. The learned embedding positions itself among related concepts, inheriting compositional properties. You can use 'a painting of <my-cat> in a garden' and the model compositionally understands both the learned concept and the context."
        id="note-embedding-geometry"
      />

      <WarningBlock
        title="Expressiveness Limitations"
        content="A single embedding vector cannot capture complex appearances with high fidelity. Textual inversion works well for styles and textures but struggles with specific faces or intricate objects. For high-fidelity subject reproduction, combine textual inversion with LoRA or use DreamBooth."
        id="warning-expressiveness"
      />
    </div>
  )
}
