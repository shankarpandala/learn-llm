import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function CLIP() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">CLIP: Contrastive Language-Image Pretraining</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        CLIP jointly trains an image encoder and a text encoder to align visual and textual
        representations in a shared embedding space. By training on 400 million image-text
        pairs from the internet, CLIP learns highly transferable visual features that enable
        zero-shot image classification and serve as the backbone for many vision-language models.
      </p>

      <DefinitionBlock
        title="CLIP Contrastive Loss"
        definition="CLIP uses a symmetric contrastive loss (InfoNCE) over a batch of $N$ image-text pairs. For image embeddings $\mathbf{I}_i$ and text embeddings $\mathbf{T}_j$, the loss maximizes cosine similarity of matching pairs $(i, i)$ while minimizing similarity of non-matching pairs $(i, j)$ where $i \neq j$."
        notation="\( \mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(\mathbf{I}_i, \mathbf{T}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{I}_i, \mathbf{T}_j) / \tau)} + \log \frac{\exp(\text{sim}(\mathbf{T}_i, \mathbf{I}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{T}_j, \mathbf{I}_i) / \tau)} \right] \)"
        id="def-clip-loss"
      />

      <h2 className="text-2xl font-semibold">Architecture</h2>
      <p className="text-gray-700 dark:text-gray-300">
        CLIP consists of two encoders: a ViT (or ResNet) for images and a transformer for text.
        Both encoders project their outputs to a shared <InlineMath math="d" />-dimensional space
        where cosine similarity measures alignment. A learnable temperature parameter{' '}
        <InlineMath math="\tau" /> scales the logits.
      </p>
      <BlockMath math="\text{sim}(\mathbf{I}_i, \mathbf{T}_j) = \frac{\mathbf{I}_i \cdot \mathbf{T}_j}{\|\mathbf{I}_i\| \|\mathbf{T}_j\|}" />

      <ExampleBlock
        title="CLIP Contrastive Matrix"
        problem="In a batch of 4 image-text pairs, what does the similarity matrix look like?"
        steps={[
          { formula: '\\mathbf{S}_{ij} = \\text{sim}(\\mathbf{I}_i, \\mathbf{T}_j) / \\tau', explanation: 'Compute cosine similarity between every image and every text, scaled by temperature.' },
          { formula: '\\text{Diagonal entries } S_{ii} \\text{ are positive pairs}', explanation: 'These should be high (matched image-text pairs).' },
          { formula: '\\text{Off-diagonal } S_{ij}, i \\neq j \\text{ are negative pairs}', explanation: 'These should be low (mismatched pairs). With N=4, each sample has 3 negatives.' },
          { formula: '\\text{Apply softmax row-wise (image->text) and column-wise (text->image)}', explanation: 'Two cross-entropy losses: image-to-text and text-to-image, averaged.' },
        ]}
        id="example-clip-matrix"
      />

      <PythonCode
        title="clip_contrastive_loss.py"
        code={`import torch
import torch.nn.functional as F

def clip_loss(image_embeds, text_embeds, temperature=0.07):
    """Compute symmetric CLIP contrastive loss."""
    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # Cosine similarity matrix: (N, N)
    logits = image_embeds @ text_embeds.T / temperature

    # Labels: diagonal entries are positive pairs
    N = logits.shape[0]
    labels = torch.arange(N, device=logits.device)

    # Symmetric cross-entropy loss
    loss_i2t = F.cross_entropy(logits, labels)       # image-to-text
    loss_t2i = F.cross_entropy(logits.T, labels)     # text-to-image

    return (loss_i2t + loss_t2i) / 2

# Simulate a batch of 8 image-text pairs with 512-dim embeddings
image_emb = torch.randn(8, 512)
text_emb = torch.randn(8, 512)

loss = clip_loss(image_emb, text_emb)
print(f"CLIP loss: {loss.item():.4f}")

# With perfectly aligned embeddings, loss should be low
aligned_emb = F.normalize(torch.randn(8, 512), dim=-1)
loss_aligned = clip_loss(aligned_emb, aligned_emb)
print(f"Aligned loss: {loss_aligned.item():.4f}")`}
        id="code-clip-loss"
      />

      <PythonCode
        title="clip_zero_shot.py"
        code={`# Zero-shot classification with OpenAI CLIP
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Classify an image against text prompts
image = Image.new("RGB", (224, 224), color="blue")  # placeholder
labels = ["a photo of a cat", "a photo of a dog", "a blue square", "a red circle"]

inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Cosine similarity -> probabilities
logits = outputs.logits_per_image  # (1, num_labels)
probs = logits.softmax(dim=-1)

for label, prob in zip(labels, probs[0]):
    print(f"  {label}: {prob.item():.3f}")`}
        id="code-clip-zeroshot"
      />

      <NoteBlock
        type="historical"
        title="CLIP Impact"
        content="Radford et al. (2021) at OpenAI showed that natural language supervision scales better than fixed label sets. CLIP's image encoder became foundational: Stable Diffusion uses CLIP text encoder for conditioning, LLaVA uses CLIP's ViT as its vision backbone, and CLIP embeddings power most modern image search systems."
        id="note-clip-history"
      />

      <WarningBlock
        title="Batch Size Requirements"
        content="CLIP's contrastive loss requires very large batch sizes (32,768 in the original paper) to provide enough negatives for effective learning. Small batch sizes lead to poor representations because the model sees too few negative examples per step."
        id="warning-clip-batch"
      />
    </div>
  )
}
