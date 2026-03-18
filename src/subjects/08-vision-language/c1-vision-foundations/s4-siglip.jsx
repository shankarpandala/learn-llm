import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function SigLIP() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">SigLIP: Sigmoid Loss for Language-Image Pretraining</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        SigLIP replaces CLIP's softmax-based contrastive loss with a pairwise sigmoid loss.
        This eliminates the need for a global softmax normalization across the batch, enabling
        better scaling to larger batch sizes and more efficient distributed training. SigLIP
        achieves comparable or better performance than CLIP with simpler training dynamics.
      </p>

      <DefinitionBlock
        title="SigLIP Sigmoid Loss"
        definition="Instead of computing a softmax over the full batch, SigLIP treats each $(i, j)$ pair independently with a binary sigmoid loss. For matching pairs $(i, i)$, the target is $y_{ij} = 1$; for non-matching pairs, $y_{ij} = -1$. The loss per pair is $\ell_{ij} = -\log \sigma(y_{ij} \cdot z_{ij})$ where $z_{ij} = t \cdot \text{sim}(\mathbf{I}_i, \mathbf{T}_j) + b$."
        notation="\( \mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} \log \sigma(y_{ij}(t \cdot \mathbf{I}_i^\top \mathbf{T}_j + b)) \)"
        id="def-siglip"
      />

      <h2 className="text-2xl font-semibold">Why Sigmoid over Softmax?</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The softmax normalization in CLIP requires computing the partition function across the
        entire batch, which couples all samples together. This makes distributed training
        difficult because gradients must be synchronized across all devices. The sigmoid loss
        decomposes into independent pairwise terms, enabling efficient chunked computation.
      </p>

      <ExampleBlock
        title="CLIP vs SigLIP Loss Comparison"
        problem="Compare the gradient flow for a batch of 4 pairs."
        steps={[
          { formula: '\\text{CLIP: } \\frac{\\partial \\mathcal{L}}{\\partial z_{11}} \\text{ depends on } z_{12}, z_{13}, z_{14}', explanation: 'Softmax couples all logits in a row; every negative affects the positive gradient.' },
          { formula: '\\text{SigLIP: } \\frac{\\partial \\mathcal{L}}{\\partial z_{11}} = \\sigma(-z_{11}) - 1', explanation: 'Each pair is independent; gradient depends only on the pair itself.' },
          { formula: '\\text{Communication: CLIP needs all-gather, SigLIP only needs chunks}', explanation: 'SigLIP can process sub-batches on different GPUs without full synchronization.' },
        ]}
        id="example-comparison"
      />

      <PythonCode
        title="siglip_loss.py"
        code={`import torch
import torch.nn.functional as F

def siglip_loss(image_embeds, text_embeds, temperature=10.0, bias=-10.0):
    """SigLIP pairwise sigmoid contrastive loss."""
    # Normalize
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # Pairwise cosine similarities: (N, N)
    logits = image_embeds @ text_embeds.T * temperature + bias

    # Labels: +1 for diagonal (matching), -1 for off-diagonal
    N = logits.shape[0]
    labels = 2 * torch.eye(N, device=logits.device) - 1  # +1 on diag, -1 elsewhere

    # Pairwise sigmoid loss: -log(sigmoid(y * z))
    loss = -F.logsigmoid(labels * logits).mean()

    return loss

# Compare CLIP and SigLIP losses
def clip_loss(img_emb, txt_emb, temp=0.07):
    img_emb = F.normalize(img_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)
    logits = img_emb @ txt_emb.T / temp
    labels = torch.arange(logits.shape[0])
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

batch = 16
img_e = torch.randn(batch, 512)
txt_e = torch.randn(batch, 512)

print(f"CLIP loss:   {clip_loss(img_e, txt_e).item():.4f}")
print(f"SigLIP loss: {siglip_loss(img_e, txt_e).item():.4f}")

# SigLIP scales better: chunk computation
def siglip_chunked(img_emb, txt_emb, chunk_size=4, temp=10.0, bias=-10.0):
    """Process SigLIP loss in chunks for memory efficiency."""
    img_emb = F.normalize(img_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)
    N = img_emb.shape[0]
    total_loss = 0.0
    for i in range(0, N, chunk_size):
        for j in range(0, N, chunk_size):
            logits_chunk = img_emb[i:i+chunk_size] @ txt_emb[j:j+chunk_size].T
            logits_chunk = logits_chunk * temp + bias
            labels_chunk = 2 * torch.eye(chunk_size, device=logits_chunk.device) - 1
            if i != j:
                labels_chunk = -torch.ones_like(logits_chunk)
            total_loss += -F.logsigmoid(labels_chunk * logits_chunk).sum()
    return total_loss / (N * N)

print(f"Chunked SigLIP: {siglip_chunked(img_e, txt_e).item():.4f}")`}
        id="code-siglip"
      />

      <NoteBlock
        type="intuition"
        title="Temperature and Bias in SigLIP"
        content="SigLIP uses learnable temperature t and bias b parameters. The bias b acts as a threshold: when t*sim(I,T) + b > 0, the model predicts the pair matches. The bias is typically initialized negative (e.g., -10) so that most pairs default to non-matching, reflecting the true distribution."
        id="note-siglip-temp"
      />

      <NoteBlock
        type="note"
        title="SigLIP in Practice"
        content="Google's PaLI-X and Gemini use SigLIP-trained vision encoders. The SigLIP ViT-SO400M (400M params, trained on WebLI) is widely used as a vision backbone in open models like PaliGemma and LLaVA-NeXT, often outperforming CLIP-based encoders."
        id="note-siglip-usage"
      />

      <WarningBlock
        title="Negative-Positive Imbalance"
        content="In a batch of N pairs, there are N positives but N^2 - N negatives. SigLIP averages over all N^2 pairs, so negatives dominate. This is by design (most pairs should not match), but careful initialization of the bias term is critical to avoid training instability."
        id="warning-siglip-imbalance"
      />
    </div>
  )
}
