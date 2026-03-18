import{j as e}from"./vendor-DWbzdFaj.js";import{r as s}from"./vendor-katex-BYl39Yo6.js";import{D as i,E as o,P as a,N as t,W as n,T as r}from"./subject-01-text-fundamentals-DG6tAvii.js";function l(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Vision Transformer (ViT)"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The Vision Transformer (ViT) demonstrated that a pure transformer architecture, originally designed for NLP, can achieve state-of-the-art results on image classification when trained on sufficient data. ViT splits an image into fixed-size patches and treats them as a sequence of tokens, just like words in a sentence."}),e.jsx(i,{title:"Vision Transformer",definition:"A Vision Transformer (ViT) divides an image of size $H \\times W$ into a grid of $N = \\frac{H \\times W}{P^2}$ non-overlapping patches of size $P \\times P$, linearly embeds each patch into a $D$-dimensional vector, prepends a learnable [CLS] token, adds positional embeddings, and processes the resulting sequence through a standard transformer encoder.",notation:"Given image \\( \\mathbf{x} \\in \\mathbb{R}^{H \\times W \\times C} \\), patches \\( \\mathbf{x}_p^i \\in \\mathbb{R}^{P^2 \\cdot C} \\) for \\( i = 1, \\ldots, N \\)",id:"def-vit"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Patch Embedding"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["Each patch is flattened from a ",e.jsx(s.InlineMath,{math:"P \\times P \\times C"})," tensor into a vector and projected through a linear layer to produce a ",e.jsx(s.InlineMath,{math:"D"}),"-dimensional embedding. Positional embeddings are added to retain spatial information."]}),e.jsx(s.BlockMath,{math:"\\mathbf{z}_0 = [\\mathbf{x}_{\\text{class}};\\; \\mathbf{x}_p^1 \\mathbf{E};\\; \\mathbf{x}_p^2 \\mathbf{E};\\; \\ldots;\\; \\mathbf{x}_p^N \\mathbf{E}] + \\mathbf{E}_{\\text{pos}}, \\quad \\mathbf{E} \\in \\mathbb{R}^{(P^2 \\cdot C) \\times D}"}),e.jsx(o,{title:"ViT-Base Patch Calculation",problem:"For a 224x224 RGB image with patch size 16, how many patches and what is the sequence length?",steps:[{formula:"N = \\frac{224 \\times 224}{16 \\times 16} = \\frac{50176}{256} = 196",explanation:"Divide total pixels by pixels per patch."},{formula:"\\text{seq\\_len} = N + 1 = 197",explanation:"Add 1 for the prepended [CLS] token."},{formula:"\\text{Each patch: } 16 \\times 16 \\times 3 = 768 \\text{ values}",explanation:"Flattened patch dimension matches ViT-Base hidden size D=768."}],id:"example-vit-patches"}),e.jsx(a,{title:"vit_from_scratch.py",code:`import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # Conv2d with kernel=stride=patch_size acts as patch extraction + linear projection
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, embed_dim, N) -> (B, N, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x)                          # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)                 # (B, 1, D)
        z = torch.cat([cls, patches], dim=1) + self.pos_embed  # (B, N+1, D)
        z = self.encoder(z)
        z = self.norm(z[:, 0])  # [CLS] token output
        return self.head(z)

# Test
model = ViT()
img = torch.randn(2, 3, 224, 224)
logits = model(img)
print(f"Output shape: {logits.shape}")  # (2, 1000)
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")`,id:"code-vit"}),e.jsx(t,{type:"historical",title:"ViT Origin",content:"Dosovitskiy et al. (2020) introduced ViT in 'An Image Is Worth 16x16 Words.' The key finding was that with large-scale pretraining (JFT-300M), ViT surpassed CNNs, but with smaller datasets like ImageNet alone, CNNs still performed better due to their inductive biases (translation equivariance, locality).",id:"note-vit-history"}),e.jsx(n,{title:"Positional Embedding Interpolation",content:"ViT learns fixed positional embeddings for a specific resolution. When fine-tuning at higher resolutions, you must interpolate the positional embeddings (typically with bicubic interpolation), which can degrade performance if the resolution gap is too large.",id:"warning-pos-embed"}),e.jsx(t,{type:"intuition",title:"Why Patches Work",content:"Treating patches as tokens lets ViT leverage the transformer's global attention from the first layer. Unlike CNNs that build up receptive fields gradually, ViT can attend to distant image regions immediately, which helps with tasks requiring global context like scene understanding.",id:"note-patches-intuition"})]})}const Q=Object.freeze(Object.defineProperty({__proto__:null,default:l},Symbol.toStringTag,{value:"Module"}));function d(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Patch Embedding and Image Tokenization"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Image tokenization converts continuous pixel data into discrete or continuous token representations that transformers can process. Beyond simple linear patch projection, modern approaches use convolutional stems, VQ-VAE codebooks, or learned visual vocabularies to create richer image tokens."}),e.jsx(i,{title:"Patch Embedding",definition:"Patch embedding maps each image patch $\\mathbf{x}_p^i \\in \\mathbb{R}^{P^2 \\cdot C}$ to a $D$-dimensional vector via a learnable projection $\\mathbf{E} \\in \\mathbb{R}^{(P^2 C) \\times D}$. This is equivalent to applying a convolution with kernel size and stride equal to the patch size $P$.",id:"def-patch-embed"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Linear Patch Projection"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The simplest patch embedding is a linear layer applied to flattened patches. A",e.jsx(s.InlineMath,{math:"P \\times P"})," patch from a ",e.jsx(s.InlineMath,{math:"C"}),"-channel image produces a ",e.jsx(s.InlineMath,{math:"P^2 C"}),"-dimensional vector, projected to dimension"," ",e.jsx(s.InlineMath,{math:"D"}),"."]}),e.jsx(s.BlockMath,{math:"\\mathbf{e}_i = \\text{flatten}(\\mathbf{x}_p^i) \\cdot \\mathbf{E} + \\mathbf{b}, \\quad \\mathbf{e}_i \\in \\mathbb{R}^D"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Convolutional Patch Embedding"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Using a convolutional stem instead of a single projection provides overlapping receptive fields, better handling of edges, and improved training stability. Many modern ViT variants (e.g., ConvNeXt-based stems) replace the linear projection with 2-4 convolutional layers with decreasing stride."}),e.jsx(a,{title:"patch_embedding_variants.py",code:`import torch
import torch.nn as nn

# Method 1: Linear projection (original ViT)
class LinearPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

# Method 2: Convolutional stem (better for smaller datasets)
class ConvStemPatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=768):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1),  # total stride=16
        )

    def forward(self, x):
        return self.stem(x).flatten(2).transpose(1, 2)

img = torch.randn(1, 3, 224, 224)
for name, module in [("Linear", LinearPatchEmbed()), ("ConvStem", ConvStemPatchEmbed())]:
    out = module(img)
    print(f"{name}: {out.shape}")  # Both: (1, 196, 768)`,id:"code-patch-variants"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"VQ-VAE Image Tokenization"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Vector-Quantized VAEs convert images into discrete codebook indices, enabling image generation via autoregressive token prediction. The encoder maps patches to the nearest codebook vector using nearest-neighbor lookup."}),e.jsx(s.BlockMath,{math:"z_q = \\text{argmin}_{e_k \\in \\mathcal{C}} \\| z_e(\\mathbf{x}) - e_k \\|_2"}),e.jsx(o,{title:"VQ-VAE Tokenization",problem:"A 256x256 image encoded by a VQ-VAE with downsampling factor 16 and codebook size 8192. How many tokens?",steps:[{formula:"\\text{Latent grid} = \\frac{256}{16} \\times \\frac{256}{16} = 16 \\times 16 = 256",explanation:"Spatial dimensions reduced by factor 16."},{formula:"\\text{Each position} \\to \\text{index} \\in \\{0, 1, \\ldots, 8191\\}",explanation:"Each spatial position maps to one of 8192 codebook entries."},{formula:"\\text{Total: 256 discrete tokens from vocabulary of 8192}",explanation:"The image becomes a sequence of 256 integer tokens, like text."}],id:"example-vqvae"}),e.jsx(a,{title:"vq_tokenization.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """Simple VQ layer for image tokenization."""
    def __init__(self, num_embeddings=8192, embedding_dim=256):
        super().__init__()
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e):
        # z_e: (B, D, H, W) -> (B, H, W, D)
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e.view(-1, z_e.shape[-1])

        # Nearest neighbor lookup
        distances = torch.cdist(flat, self.codebook.weight)
        indices = distances.argmin(dim=-1)
        z_q = self.codebook(indices).view(z_e.shape)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # Commitment loss
        commitment_loss = F.mse_loss(z_e.detach(), z_q) + F.mse_loss(z_e, z_q.detach())

        return z_q_st.permute(0, 3, 1, 2), indices, commitment_loss

vq = VectorQuantizer(num_embeddings=8192, embedding_dim=256)
z_encoded = torch.randn(2, 256, 16, 16)  # Pretend encoder output
z_quantized, token_ids, loss = vq(z_encoded)
print(f"Quantized shape: {z_quantized.shape}")   # (2, 256, 16, 16)
print(f"Token IDs shape: {token_ids.shape}")      # (512,) = 2*16*16
print(f"Unique tokens used: {token_ids.unique().shape[0]}")`,id:"code-vq"}),e.jsx(t,{type:"tip",title:"Patch Size Trade-offs",content:"Smaller patches (e.g., 8x8) produce longer sequences with finer detail but quadratically increase attention cost. Larger patches (e.g., 32x32) are faster but lose fine-grained information. Most modern ViTs use 14x14 or 16x16 patches as a good balance.",id:"note-patch-tradeoff"}),e.jsx(n,{title:"Codebook Collapse",content:"VQ-VAEs frequently suffer from codebook collapse where only a small fraction of codebook entries are used. Techniques like EMA updates, codebook reset, and entropy regularization help maintain codebook utilization.",id:"warning-codebook-collapse"})]})}const K=Object.freeze(Object.defineProperty({__proto__:null,default:d},Symbol.toStringTag,{value:"Module"}));function m(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"CLIP: Contrastive Language-Image Pretraining"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"CLIP jointly trains an image encoder and a text encoder to align visual and textual representations in a shared embedding space. By training on 400 million image-text pairs from the internet, CLIP learns highly transferable visual features that enable zero-shot image classification and serve as the backbone for many vision-language models."}),e.jsx(i,{title:"CLIP Contrastive Loss",definition:"CLIP uses a symmetric contrastive loss (InfoNCE) over a batch of $N$ image-text pairs. For image embeddings $\\mathbf{I}_i$ and text embeddings $\\mathbf{T}_j$, the loss maximizes cosine similarity of matching pairs $(i, i)$ while minimizing similarity of non-matching pairs $(i, j)$ where $i \\neq j$.",notation:"\\( \\mathcal{L} = -\\frac{1}{2N}\\sum_{i=1}^{N} \\left[ \\log \\frac{\\exp(\\text{sim}(\\mathbf{I}_i, \\mathbf{T}_i) / \\tau)}{\\sum_{j=1}^{N} \\exp(\\text{sim}(\\mathbf{I}_i, \\mathbf{T}_j) / \\tau)} + \\log \\frac{\\exp(\\text{sim}(\\mathbf{T}_i, \\mathbf{I}_i) / \\tau)}{\\sum_{j=1}^{N} \\exp(\\text{sim}(\\mathbf{T}_j, \\mathbf{I}_i) / \\tau)} \\right] \\)",id:"def-clip-loss"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Architecture"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["CLIP consists of two encoders: a ViT (or ResNet) for images and a transformer for text. Both encoders project their outputs to a shared ",e.jsx(s.InlineMath,{math:"d"}),"-dimensional space where cosine similarity measures alignment. A learnable temperature parameter"," ",e.jsx(s.InlineMath,{math:"\\tau"})," scales the logits."]}),e.jsx(s.BlockMath,{math:"\\text{sim}(\\mathbf{I}_i, \\mathbf{T}_j) = \\frac{\\mathbf{I}_i \\cdot \\mathbf{T}_j}{\\|\\mathbf{I}_i\\| \\|\\mathbf{T}_j\\|}"}),e.jsx(o,{title:"CLIP Contrastive Matrix",problem:"In a batch of 4 image-text pairs, what does the similarity matrix look like?",steps:[{formula:"\\mathbf{S}_{ij} = \\text{sim}(\\mathbf{I}_i, \\mathbf{T}_j) / \\tau",explanation:"Compute cosine similarity between every image and every text, scaled by temperature."},{formula:"\\text{Diagonal entries } S_{ii} \\text{ are positive pairs}",explanation:"These should be high (matched image-text pairs)."},{formula:"\\text{Off-diagonal } S_{ij}, i \\neq j \\text{ are negative pairs}",explanation:"These should be low (mismatched pairs). With N=4, each sample has 3 negatives."},{formula:"\\text{Apply softmax row-wise (image->text) and column-wise (text->image)}",explanation:"Two cross-entropy losses: image-to-text and text-to-image, averaged."}],id:"example-clip-matrix"}),e.jsx(a,{title:"clip_contrastive_loss.py",code:`import torch
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
print(f"Aligned loss: {loss_aligned.item():.4f}")`,id:"code-clip-loss"}),e.jsx(a,{title:"clip_zero_shot.py",code:`# Zero-shot classification with OpenAI CLIP
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
    print(f"  {label}: {prob.item():.3f}")`,id:"code-clip-zeroshot"}),e.jsx(t,{type:"historical",title:"CLIP Impact",content:"Radford et al. (2021) at OpenAI showed that natural language supervision scales better than fixed label sets. CLIP's image encoder became foundational: Stable Diffusion uses CLIP text encoder for conditioning, LLaVA uses CLIP's ViT as its vision backbone, and CLIP embeddings power most modern image search systems.",id:"note-clip-history"}),e.jsx(n,{title:"Batch Size Requirements",content:"CLIP's contrastive loss requires very large batch sizes (32,768 in the original paper) to provide enough negatives for effective learning. Small batch sizes lead to poor representations because the model sees too few negative examples per step.",id:"warning-clip-batch"})]})}const J=Object.freeze(Object.defineProperty({__proto__:null,default:m},Symbol.toStringTag,{value:"Module"}));function c(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"SigLIP: Sigmoid Loss for Language-Image Pretraining"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"SigLIP replaces CLIP's softmax-based contrastive loss with a pairwise sigmoid loss. This eliminates the need for a global softmax normalization across the batch, enabling better scaling to larger batch sizes and more efficient distributed training. SigLIP achieves comparable or better performance than CLIP with simpler training dynamics."}),e.jsx(i,{title:"SigLIP Sigmoid Loss",definition:"Instead of computing a softmax over the full batch, SigLIP treats each $(i, j)$ pair independently with a binary sigmoid loss. For matching pairs $(i, i)$, the target is $y_{ij} = 1$; for non-matching pairs, $y_{ij} = -1$. The loss per pair is $\\ell_{ij} = -\\log \\sigma(y_{ij} \\cdot z_{ij})$ where $z_{ij} = t \\cdot \\text{sim}(\\mathbf{I}_i, \\mathbf{T}_j) + b$.",notation:"\\( \\mathcal{L}_{\\text{SigLIP}} = -\\frac{1}{N^2} \\sum_{i=1}^{N} \\sum_{j=1}^{N} \\log \\sigma(y_{ij}(t \\cdot \\mathbf{I}_i^\\top \\mathbf{T}_j + b)) \\)",id:"def-siglip"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Why Sigmoid over Softmax?"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The softmax normalization in CLIP requires computing the partition function across the entire batch, which couples all samples together. This makes distributed training difficult because gradients must be synchronized across all devices. The sigmoid loss decomposes into independent pairwise terms, enabling efficient chunked computation."}),e.jsx(o,{title:"CLIP vs SigLIP Loss Comparison",problem:"Compare the gradient flow for a batch of 4 pairs.",steps:[{formula:"\\text{CLIP: } \\frac{\\partial \\mathcal{L}}{\\partial z_{11}} \\text{ depends on } z_{12}, z_{13}, z_{14}",explanation:"Softmax couples all logits in a row; every negative affects the positive gradient."},{formula:"\\text{SigLIP: } \\frac{\\partial \\mathcal{L}}{\\partial z_{11}} = \\sigma(-z_{11}) - 1",explanation:"Each pair is independent; gradient depends only on the pair itself."},{formula:"\\text{Communication: CLIP needs all-gather, SigLIP only needs chunks}",explanation:"SigLIP can process sub-batches on different GPUs without full synchronization."}],id:"example-comparison"}),e.jsx(a,{title:"siglip_loss.py",code:`import torch
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

print(f"Chunked SigLIP: {siglip_chunked(img_e, txt_e).item():.4f}")`,id:"code-siglip"}),e.jsx(t,{type:"intuition",title:"Temperature and Bias in SigLIP",content:"SigLIP uses learnable temperature t and bias b parameters. The bias b acts as a threshold: when t*sim(I,T) + b > 0, the model predicts the pair matches. The bias is typically initialized negative (e.g., -10) so that most pairs default to non-matching, reflecting the true distribution.",id:"note-siglip-temp"}),e.jsx(t,{type:"note",title:"SigLIP in Practice",content:"Google's PaLI-X and Gemini use SigLIP-trained vision encoders. The SigLIP ViT-SO400M (400M params, trained on WebLI) is widely used as a vision backbone in open models like PaliGemma and LLaVA-NeXT, often outperforming CLIP-based encoders.",id:"note-siglip-usage"}),e.jsx(n,{title:"Negative-Positive Imbalance",content:"In a batch of N pairs, there are N positives but N^2 - N negatives. SigLIP averages over all N^2 pairs, so negatives dominate. This is by design (most pairs should not match), but careful initialization of the bias term is critical to avoid training instability.",id:"warning-siglip-imbalance"})]})}const Y=Object.freeze(Object.defineProperty({__proto__:null,default:c},Symbol.toStringTag,{value:"Module"}));function p(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Multimodal Fusion Strategies"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Fusion refers to how information from different modalities (vision and language) is combined. The three main strategies -- early fusion, late fusion, and cross-attention fusion -- each present different trade-offs in expressiveness, efficiency, and modularity. The choice of fusion strategy is one of the most important architectural decisions in vision-language models."}),e.jsx(i,{title:"Early Fusion",definition:"In early fusion, visual and textual tokens are concatenated into a single sequence and processed together through shared transformer layers. The model learns cross-modal interactions from the first layer. Formally, $\\mathbf{Z} = \\text{Transformer}([\\mathbf{V}_1, \\ldots, \\mathbf{V}_m, \\mathbf{T}_1, \\ldots, \\mathbf{T}_n])$.",id:"def-early-fusion"}),e.jsx(i,{title:"Late Fusion",definition:"In late fusion, each modality is processed independently by its own encoder, and representations are combined only at the final layers (e.g., via dot product or MLP). CLIP is a classic late fusion model: $\\text{score} = \\mathbf{f}_{\\text{img}}(\\mathbf{x})^\\top \\mathbf{f}_{\\text{txt}}(\\mathbf{t})$.",id:"def-late-fusion"}),e.jsx(i,{title:"Cross-Attention Fusion",definition:"Cross-attention fusion uses dedicated cross-attention layers where one modality attends to the other. Text tokens query visual features: $\\text{CrossAttn}(\\mathbf{Q}_{\\text{text}}, \\mathbf{K}_{\\text{image}}, \\mathbf{V}_{\\text{image}})$. This enables rich interaction while keeping encoders modular.",id:"def-cross-fusion"}),e.jsx(o,{title:"Comparing Fusion Strategies",problem:"For an image with 196 visual tokens and text with 77 tokens, compare computational costs.",steps:[{formula:"\\text{Early: self-attn on } (196 + 77)^2 = 273^2 = 74{,}529",explanation:"All tokens attend to all others; quadratic in total length."},{formula:"\\text{Late: } 196^2 + 77^2 = 38{,}416 + 5{,}929 = 44{,}345",explanation:"Each modality processes independently; cheaper but no cross-modal interaction."},{formula:"\\text{Cross-attn: } 77 \\times 196 = 15{,}092 \\text{ (per cross-attn layer)}",explanation:"Text queries attend to visual keys/values; more efficient than early fusion."}],id:"example-fusion-cost"}),e.jsx(a,{title:"fusion_strategies.py",code:`import torch
import torch.nn as nn

class EarlyFusion(nn.Module):
    """Concatenate visual + text tokens, process jointly."""
    def __init__(self, d_model=768, nhead=12, num_layers=6):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, vis_tokens, txt_tokens):
        # Simply concatenate along sequence dimension
        combined = torch.cat([vis_tokens, txt_tokens], dim=1)
        return self.encoder(combined)

class LateFusion(nn.Module):
    """Process modalities independently, combine at the end."""
    def __init__(self, d_model=768):
        super().__init__()
        self.vis_proj = nn.Linear(d_model, 512)
        self.txt_proj = nn.Linear(d_model, 512)

    def forward(self, vis_cls, txt_cls):
        # Project both to shared space
        vis_emb = nn.functional.normalize(self.vis_proj(vis_cls), dim=-1)
        txt_emb = nn.functional.normalize(self.txt_proj(txt_cls), dim=-1)
        return vis_emb @ txt_emb.T  # similarity matrix

class CrossAttentionFusion(nn.Module):
    """Text queries attend to visual features."""
    def __init__(self, d_model=768, nhead=12, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, txt_tokens, vis_tokens):
        x = txt_tokens
        for attn, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            x, _ = attn(query=x, key=vis_tokens, value=vis_tokens)
            x = x + residual
        return x

# Compare all three
B, D = 2, 768
vis = torch.randn(B, 196, D)   # 14x14 patches
txt = torch.randn(B, 77, D)    # text tokens

early = EarlyFusion()
late = LateFusion()
cross = CrossAttentionFusion()

out_early = early(vis, txt)
print(f"Early fusion output: {out_early.shape}")      # (2, 273, 768)

out_late = late(vis[:, 0], txt[:, 0])  # CLS tokens
print(f"Late fusion output: {out_late.shape}")         # (2, 2)

out_cross = cross(txt, vis)
print(f"Cross-attention output: {out_cross.shape}")    # (2, 77, 768)`,id:"code-fusion"}),e.jsx(t,{type:"intuition",title:"Which Fusion to Choose?",content:"Early fusion (LLaVA, GPT-4V) is most expressive but costly. It works well when the LLM is large enough to learn cross-modal reasoning. Late fusion (CLIP, SigLIP) is most efficient for retrieval tasks. Cross-attention (Flamingo, Qwen-VL) offers a middle ground with modularity -- you can swap the vision encoder without retraining the LLM.",id:"note-fusion-choice"}),e.jsx(n,{title:"Early Fusion Context Length",content:"Early fusion models must fit both visual and text tokens within the context window. A single 224x224 image at patch size 14 produces 256 visual tokens. Higher-resolution images or multiple images can quickly exhaust the context budget, leaving less room for text generation.",id:"warning-context-length"})]})}const Z=Object.freeze(Object.defineProperty({__proto__:null,default:p},Symbol.toStringTag,{value:"Module"}));function f(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"LLaVA: Large Language-and-Vision Assistant"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"LLaVA is a simple yet effective architecture that connects a pretrained CLIP vision encoder to a pretrained LLM via a lightweight projection layer. It follows the early fusion paradigm where visual tokens are injected directly into the LLM's input sequence alongside text tokens, enabling the LLM to reason about images using its existing language understanding capabilities."}),e.jsx(i,{title:"LLaVA Architecture",definition:"LLaVA consists of three components: (1) a frozen CLIP ViT-L/14 vision encoder that produces visual features $\\mathbf{Z}_v \\in \\mathbb{R}^{N \\times D_v}$, (2) a trainable projection layer $\\mathbf{W} \\in \\mathbb{R}^{D_v \\times D_l}$ that maps visual features to the LLM's embedding space, and (3) a pretrained LLM (e.g., LLaMA/Vicuna) that processes the combined sequence.",notation:"\\( \\mathbf{H}_v = \\mathbf{Z}_v \\mathbf{W}, \\quad \\text{Input} = [\\mathbf{H}_v; \\mathbf{H}_{\\text{text}}] \\)",id:"def-llava"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Two-Stage Training"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"LLaVA uses a two-stage training procedure. Stage 1 (pretraining) trains only the projection layer on image-caption pairs to align visual features to the LLM's embedding space. Stage 2 (instruction tuning) fine-tunes both the projection and the LLM on multimodal instruction-following data."}),e.jsx(o,{title:"LLaVA Token Sequence",problem:"Trace how a user query 'What is in this image?' with an attached 224x224 image flows through LLaVA.",steps:[{formula:"\\text{CLIP ViT}: 224^2 / 14^2 = 256 \\text{ visual tokens} \\in \\mathbb{R}^{256 \\times 1024}",explanation:"CLIP ViT-L/14 produces 256 patch features of dimension 1024."},{formula:"\\text{Projection}: \\mathbb{R}^{256 \\times 1024} \\to \\mathbb{R}^{256 \\times 4096}",explanation:"Linear projection maps visual dim (1024) to LLM dim (4096 for LLaMA-7B)."},{formula:"\\text{Concat}: [\\text{sys\\_tokens}, \\mathbf{H}_v, \\text{query\\_tokens}]",explanation:"Visual tokens are inserted at the image placeholder position in the prompt."},{formula:"\\text{LLM generates response autoregressively}",explanation:"The LLM sees visual tokens as if they were regular embeddings and generates text."}],id:"example-llava-flow"}),e.jsx(a,{title:"llava_architecture.py",code:`import torch
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
print(f"MLP projector:    {sum(p.numel() for p in proj_mlp.parameters()) / 1e6:.1f}M")`,id:"code-llava"}),e.jsx(a,{title:"llava_inference.py",code:`# Using LLaVA with transformers library
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
    print(f"{name}: {cfg}")`,id:"code-llava-inference"}),e.jsx(t,{type:"note",title:"LLaVA Versions",content:"LLaVA-1.0 (2023) used a single linear projection. LLaVA-1.5 improved with a 2-layer MLP projector and higher resolution (336px). LLaVA-NeXT further increased resolution with dynamic tiling, splitting high-res images into multiple 336px tiles to preserve fine detail.",id:"note-llava-versions"}),e.jsx(n,{title:"Projection Layer Alignment",content:"The projection layer must be pretrained before instruction tuning. Skipping Stage 1 (visual-language alignment) and going directly to instruction tuning results in significantly worse performance because the LLM cannot interpret raw CLIP features.",id:"warning-llava-alignment"})]})}const ee=Object.freeze(Object.defineProperty({__proto__:null,default:f},Symbol.toStringTag,{value:"Module"}));function u(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Flamingo: Perceiver Resampler for Vision-Language"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"DeepMind's Flamingo introduced a cross-attention fusion approach that interleaves visual information into a frozen LLM via gated cross-attention layers and a Perceiver Resampler. This design allows the LLM to process arbitrarily many images while keeping a fixed number of visual tokens per image, enabling powerful few-shot multimodal learning."}),e.jsx(i,{title:"Perceiver Resampler",definition:"The Perceiver Resampler compresses variable-length visual features into a fixed set of $M$ learnable query vectors via cross-attention. Given visual features $\\mathbf{V} \\in \\mathbb{R}^{N \\times D}$ and $M$ learned queries $\\mathbf{Q} \\in \\mathbb{R}^{M \\times D}$, it outputs $\\mathbf{O} = \\text{CrossAttn}(\\mathbf{Q}, \\mathbf{V}, \\mathbf{V}) \\in \\mathbb{R}^{M \\times D}$ where typically $M \\ll N$.",id:"def-perceiver"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Gated Cross-Attention"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["Flamingo inserts gated cross-attention layers between existing frozen LLM layers. The gating mechanism uses a learnable scalar ",e.jsx(s.InlineMath,{math:"\\alpha"})," initialized to zero, ensuring that the model starts as the original LLM and gradually learns to incorporate visual information."]}),e.jsx(s.BlockMath,{math:"\\mathbf{h} = \\mathbf{h} + \\tanh(\\alpha) \\cdot \\text{CrossAttn}(\\mathbf{h}, \\mathbf{O}_{\\text{vis}}, \\mathbf{O}_{\\text{vis}})"}),e.jsx(o,{title:"Perceiver Resampler Compression",problem:"An image has 576 visual tokens (ViT with 24x24 patches). The Perceiver uses 64 queries. What is the compression ratio?",steps:[{formula:"\\text{Compression} = \\frac{576}{64} = 9\\times",explanation:"Each image is represented by only 64 tokens regardless of input resolution."},{formula:"\\text{Cross-attn cost} = 64 \\times 576 = 36{,}864",explanation:"Much cheaper than self-attention on full 576 tokens (331,776)."},{formula:"\\text{For 4 images: } 4 \\times 64 = 256 \\text{ visual tokens total}",explanation:"Fixed cost per image enables multi-image conversations without explosion."}],id:"example-perceiver-compression"}),e.jsx(a,{title:"perceiver_resampler.py",code:`import torch
import torch.nn as nn

class PerceiverResampler(nn.Module):
    """Compress variable-length visual features to fixed-length queries."""
    def __init__(self, dim=768, num_queries=64, num_layers=6, num_heads=12):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(dim, num_heads, batch_first=True),
                'cross_norm': nn.LayerNorm(dim),
                'ff': nn.Sequential(
                    nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
                ),
                'ff_norm': nn.LayerNorm(dim),
            }))

    def forward(self, visual_features):
        B = visual_features.shape[0]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            # Cross-attention: queries attend to visual features
            q_norm = layer['cross_norm'](queries)
            attended, _ = layer['cross_attn'](query=q_norm, key=visual_features,
                                               value=visual_features)
            queries = queries + attended

            # Feed-forward
            queries = queries + layer['ff'](layer['ff_norm'](queries))

        return queries

class GatedCrossAttention(nn.Module):
    """Gated cross-attention inserted between frozen LLM layers."""
    def __init__(self, dim=4096, num_heads=32):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(1))  # Initialize to 0

    def forward(self, text_hidden, visual_tokens):
        residual = text_hidden
        text_norm = self.norm(text_hidden)
        attended, _ = self.cross_attn(query=text_norm, key=visual_tokens,
                                       value=visual_tokens)
        return residual + torch.tanh(self.gate) * attended

# Demo
perceiver = PerceiverResampler(dim=768, num_queries=64)
vis_features = torch.randn(2, 576, 768)  # Variable-length visual features
compressed = perceiver(vis_features)
print(f"Input: {vis_features.shape} -> Output: {compressed.shape}")
# (2, 576, 768) -> (2, 64, 768)

gated = GatedCrossAttention(dim=768, num_heads=12)
text_h = torch.randn(2, 128, 768)
vis_tokens = torch.randn(2, 64, 768)
out = gated(text_h, vis_tokens)
print(f"Gated cross-attn: {out.shape}")  # (2, 128, 768)
print(f"Initial gate value: {torch.tanh(gated.gate).item():.4f}")  # ~0.0`,id:"code-perceiver"}),e.jsx(t,{type:"historical",title:"From Flamingo to Open Source",content:"Flamingo (Alayrac et al., 2022) was proprietary. OpenFlamingo replicated it using open components (CLIP + LLaMA). The Perceiver Resampler idea was adopted by BLIP-2's Q-Former and Qwen-VL's visual resampler. The core insight -- compress visual features with learnable queries -- remains widely used.",id:"note-flamingo-history"}),e.jsx(n,{title:"Information Bottleneck",content:"The Perceiver Resampler acts as an information bottleneck. With too few queries (e.g., 4-8), fine-grained visual details like small text in images are lost. Tasks requiring detailed visual understanding may need more queries (64-256) or direct token injection like LLaVA.",id:"warning-bottleneck"})]})}const te=Object.freeze(Object.defineProperty({__proto__:null,default:u},Symbol.toStringTag,{value:"Module"}));function h(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Qwen-VL: Versatile Vision-Language Model"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Qwen-VL from Alibaba combines a ViT vision encoder with the Qwen LLM through a cross-attention resampler. It supports diverse visual tasks including image captioning, visual question answering, grounding (bounding box output), and OCR. Qwen-VL demonstrates that careful multi-task training with structured outputs enables a single model to handle both understanding and localization."}),e.jsx(i,{title:"Qwen-VL Architecture",definition:"Qwen-VL uses a ViT-bigG vision encoder (1.9B params) with a single-layer cross-attention module that compresses visual features from 256 tokens to a fixed set of 256 compressed visual tokens. These are prepended to text tokens and processed by the Qwen-7B LLM with position-aware bounding box tokens.",id:"def-qwen-vl"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Bounding Box as Text"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"A key innovation in Qwen-VL is representing bounding boxes as normalized coordinate tokens within the text vocabulary. This allows the model to output grounding information as part of its natural language response without requiring a separate detection head."}),e.jsx(o,{title:"Qwen-VL Grounding Format",problem:"How does Qwen-VL represent the bounding box of a cat at coordinates (120, 80, 340, 290) in a 640x480 image?",steps:[{formula:"\\text{Normalize: } (\\frac{120}{640}, \\frac{80}{480}, \\frac{340}{640}, \\frac{290}{480})",explanation:"Convert pixel coordinates to [0, 1] range."},{formula:"\\text{Quantize to 1000 bins: } (187, 166, 531, 604)",explanation:"Multiply by 1000 and round to get integer tokens."},{formula:"\\text{Output: } \\texttt{<ref>cat</ref><box>(187,166),(531,604)</box>}",explanation:"Special tokens wrap the object name and coordinates in the text output."}],id:"example-grounding"}),e.jsx(a,{title:"qwen_vl_inference.py",code:`# Qwen-VL inference with transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# model_id = "Qwen/Qwen-VL-Chat"
# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, device_map="auto", trust_remote_code=True
# ).eval()

# Qwen-VL uses special image tags in the conversation
# query = tokenizer.from_list_format([
#     {'image': 'https://example.com/photo.jpg'},
#     {'text': 'Describe this image and locate all objects.'},
# ])
# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)

# Simulate Qwen-VL bounding box tokenization
def normalize_bbox(bbox, img_w, img_h, num_bins=1000):
    """Convert pixel bbox to Qwen-VL format."""
    x1, y1, x2, y2 = bbox
    # Normalize and quantize
    nx1 = round(x1 / img_w * num_bins)
    ny1 = round(y1 / img_h * num_bins)
    nx2 = round(x2 / img_w * num_bins)
    ny2 = round(y2 / img_h * num_bins)
    return f"<box>({nx1},{ny1}),({nx2},{ny2})</box>"

def parse_bbox(bbox_str, img_w, img_h, num_bins=1000):
    """Parse Qwen-VL bbox string back to pixel coordinates."""
    import re
    match = re.search(r'\\((d+),(d+)\\),\\((d+),(d+)\\)', bbox_str)
    if match:
        coords = [int(x) for x in match.groups()]
        return [
            coords[0] / num_bins * img_w, coords[1] / num_bins * img_h,
            coords[2] / num_bins * img_w, coords[3] / num_bins * img_h,
        ]

# Example
bbox = (120, 80, 340, 290)
img_w, img_h = 640, 480
bbox_str = normalize_bbox(bbox, img_w, img_h)
print(f"Bbox string: {bbox_str}")
# <box>(188,167),(531,604)</box>

# Qwen-VL-2 architecture improvements
configs = {
    "Qwen-VL":   {"vision": "ViT-bigG", "resampler": "256 tokens", "llm": "Qwen-7B"},
    "Qwen2-VL":  {"vision": "ViT + NaViT", "resampler": "Dynamic", "llm": "Qwen2-72B"},
}
for name, cfg in configs.items():
    print(f"{name}: {cfg}")`,id:"code-qwen-vl"}),e.jsx(t,{type:"note",title:"Qwen2-VL Improvements",content:"Qwen2-VL introduced dynamic resolution support via NaViT-style packing, allowing images of any aspect ratio without distortion. It also added video understanding by treating video frames as a sequence of images with temporal position encoding. The 72B variant achieves GPT-4V-level performance on many benchmarks.",id:"note-qwen2-vl"}),e.jsx(t,{type:"tip",title:"Multi-Image and Interleaved Input",content:"Qwen-VL natively supports multiple images in a single conversation by inserting image tokens at different positions. This enables tasks like image comparison, multi-step visual reasoning, and document processing where multiple pages must be analyzed together.",id:"note-multi-image"}),e.jsx(n,{title:"Trust Remote Code",content:"Qwen-VL requires trust_remote_code=True because it uses custom model architectures not yet integrated into the core transformers library. Always review the remote code before running in production environments, as it can execute arbitrary Python code.",id:"warning-trust-code"})]})}const ae=Object.freeze(Object.defineProperty({__proto__:null,default:h},Symbol.toStringTag,{value:"Module"}));function g(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"LayoutLM: Document Understanding with Layout"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"LayoutLM extends BERT-style models to understand documents by incorporating 2D positional information (bounding box coordinates) alongside text tokens. This enables the model to reason about the spatial layout of text on a page, which is critical for understanding forms, invoices, receipts, and other structured documents."}),e.jsx(i,{title:"LayoutLM Embedding",definition:"LayoutLM augments the standard token embedding with 2D position embeddings derived from OCR bounding boxes. For each token, the input representation is $\\mathbf{e}_i = \\mathbf{e}_{\\text{token}} + \\mathbf{e}_{\\text{1D-pos}} + \\mathbf{e}_{\\text{x0}} + \\mathbf{e}_{\\text{y0}} + \\mathbf{e}_{\\text{x1}} + \\mathbf{e}_{\\text{y1}} + \\mathbf{e}_{\\text{w}} + \\mathbf{e}_{\\text{h}}$ where $(x_0, y_0, x_1, y_1)$ are normalized bounding box coordinates.",id:"def-layoutlm"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"LayoutLM Evolution"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The LayoutLM family progressed from text + layout (v1) to incorporating the document image (v2, v3). LayoutLMv3 uses a unified architecture that jointly processes text tokens, layout positions, and image patches within a single transformer."}),e.jsx(o,{title:"LayoutLM Coordinate Encoding",problem:"A word 'Total' appears at pixel coordinates (320, 580, 410, 610) in a 1000x1000 normalized document. How is this encoded?",steps:[{formula:"(x_0, y_0, x_1, y_1) = (320, 580, 410, 610)",explanation:"Coordinates normalized to 0-1000 range."},{formula:"w = x_1 - x_0 = 90, \\quad h = y_1 - y_0 = 30",explanation:"Width and height of the bounding box."},{formula:"\\mathbf{e} = \\text{Embed}_{x0}(320) + \\text{Embed}_{y0}(580) + \\ldots",explanation:"Each coordinate has its own embedding table (size 1001 x D)."}],id:"example-layout-coords"}),e.jsx(a,{title:"layoutlm_document.py",code:`from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch

# LayoutLMv3 for document token classification (e.g., form field extraction)
# processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
# model = LayoutLMv3ForTokenClassification.from_pretrained(
#     "microsoft/layoutlmv3-base", num_labels=7
# )

# Simulated OCR output for a receipt
words = ["RECEIPT", "Item", "Qty", "Price", "Coffee", "2", "$5.00",
         "Sandwich", "1", "$8.50", "Total", "$13.50"]
# Bounding boxes: [x0, y0, x1, y1] normalized to 0-1000
boxes = [
    [350, 50, 650, 100],   # RECEIPT (header, centered)
    [50, 150, 200, 180],   # Item
    [300, 150, 400, 180],  # Qty
    [600, 150, 750, 180],  # Price
    [50, 200, 200, 230],   # Coffee
    [300, 200, 350, 230],  # 2
    [600, 200, 750, 230],  # $5.00
    [50, 250, 250, 280],   # Sandwich
    [300, 250, 350, 280],  # 1
    [600, 250, 750, 280],  # $8.50
    [50, 350, 200, 380],   # Total
    [600, 350, 750, 380],  # $13.50
]

# Labels: 0=Other, 1=Header, 2=Key, 3=Value
labels = [1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 3]

print("Document layout:")
for word, box, label in zip(words, boxes, labels):
    label_name = ["Other", "Header", "Key", "Value"][label]
    print(f"  '{word}' at ({box[0]},{box[1]})-({box[2]},{box[3]}) -> {label_name}")

# LayoutLMv3 processes words + boxes + image together
# encoding = processor(
#     Image.new("RGB", (1000, 1000)),
#     words, boxes=boxes, return_tensors="pt"
# )
# outputs = model(**encoding)
# predictions = outputs.logits.argmax(-1)

# Key LayoutLM versions
versions = {
    "LayoutLM v1": "Text + 2D layout (no image)",
    "LayoutLM v2": "Text + layout + image (separate encoders)",
    "LayoutLM v3": "Unified text + layout + image patches",
}
for v, desc in versions.items():
    print(f"{v}: {desc}")`,id:"code-layoutlm"}),e.jsx(t,{type:"tip",title:"OCR Preprocessing",content:"LayoutLM requires OCR-extracted text with bounding boxes as input. For best results, use a high-quality OCR engine like Tesseract, PaddleOCR, or cloud APIs (Google Vision, Azure Document Intelligence). The OCR quality directly bounds LayoutLM's performance.",id:"note-ocr-preprocessing"}),e.jsx(n,{title:"Coordinate Normalization",content:"LayoutLM expects bounding box coordinates normalized to the range [0, 1000]. Mismatched normalization (e.g., using raw pixel coordinates or [0, 1] range) is a common source of poor results. Always verify coordinate ranges match what the model expects.",id:"warning-normalization"})]})}const ie=Object.freeze(Object.defineProperty({__proto__:null,default:g},Symbol.toStringTag,{value:"Module"}));function x(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"OCR-Free Document Understanding"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"OCR-free models process document images directly without requiring a separate OCR preprocessing step. Models like Donut, Nougat, and Pix2Struct learn to read text from pixels end-to-end, avoiding OCR errors and simplifying the pipeline. This approach is particularly effective for complex layouts, handwriting, and non-Latin scripts."}),e.jsx(i,{title:"OCR-Free Document Model",definition:"An OCR-free document model is an encoder-decoder architecture where the encoder processes raw document images (as patches) and the decoder generates structured text output (JSON, markdown, etc.) directly. The model implicitly learns character recognition, layout understanding, and information extraction in a unified framework.",id:"def-ocr-free"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Donut Architecture"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Donut (Document Understanding Transformer) uses a Swin Transformer encoder for images and a BART-style decoder to generate structured JSON from document images. It is pretrained with a synthetic document reading task and fine-tuned for specific extraction tasks."}),e.jsx(o,{title:"OCR Pipeline vs OCR-Free",problem:"Compare the processing steps for extracting invoice total from a document image.",steps:[{formula:"\\text{OCR pipeline: Image} \\to \\text{OCR} \\to \\text{Text+Boxes} \\to \\text{LayoutLM} \\to \\text{Extract}",explanation:"Traditional approach requires OCR as a separate step, introducing potential errors."},{formula:"\\text{OCR-free: Image} \\to \\text{Donut/Nougat} \\to \\text{JSON output}",explanation:"Single model directly outputs structured data from pixels."},{formula:"\\text{Error propagation: OCR@95\\% acc} \\to \\text{NER@90\\%} = 85.5\\% \\text{ end-to-end}",explanation:"OCR pipeline compounds errors; OCR-free avoids this cascade."}],id:"example-pipeline-comparison"}),e.jsx(a,{title:"donut_inference.py",code:`from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Donut for document parsing
# processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
# model = VisionEncoderDecoderModel.from_pretrained(
#     "naver-clova-ix/donut-base-finetuned-cord-v2"
# )

# Example: Parse a receipt
# image = Image.open("receipt.png").convert("RGB")
# pixel_values = processor(image, return_tensors="pt").pixel_values

# # Generate structured output
# task_prompt = "<s_cord-v2>"  # Task-specific prompt token
# decoder_input_ids = processor.tokenizer(
#     task_prompt, add_special_tokens=False, return_tensors="pt"
# ).input_ids

# outputs = model.generate(
#     pixel_values,
#     decoder_input_ids=decoder_input_ids,
#     max_length=model.decoder.config.max_position_embeddings,
#     pad_token_id=processor.tokenizer.pad_token_id,
#     eos_token_id=processor.tokenizer.eos_token_id,
# )
# result = processor.token2json(processor.decode(outputs[0]))
# print(result)  # {"menu": [{"nm": "Coffee", "price": "5.00"}, ...], "total": "13.50"}

# Nougat for academic papers (LaTeX output)
# nougat_processor = DonutProcessor.from_pretrained("facebook/nougat-base")
# nougat_model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

# OCR-free model comparison
models = {
    "Donut":     {"encoder": "Swin-B", "decoder": "BART", "output": "JSON", "params": "200M"},
    "Pix2Struct": {"encoder": "ViT", "decoder": "T5", "output": "HTML/Text", "params": "300M"},
    "Nougat":    {"encoder": "Swin-B", "decoder": "mBART", "output": "LaTeX/MD", "params": "250M"},
    "Florence-2": {"encoder": "DaViT", "decoder": "BART", "output": "Multi-task", "params": "770M"},
}
for name, info in models.items():
    print(f"{name}: {info['encoder']} -> {info['decoder']} ({info['params']}) -> {info['output']}")`,id:"code-donut"}),e.jsx(t,{type:"intuition",title:"Why OCR-Free Works",content:"OCR is fundamentally a vision task -- recognizing text from pixels. By training an end-to-end model, the vision encoder learns to extract textual features that are optimized for the downstream task, not just generic character recognition. This is especially powerful for degraded documents, handwriting, and complex layouts where traditional OCR struggles.",id:"note-ocr-free-intuition"}),e.jsx(t,{type:"note",title:"Resolution Matters",content:"OCR-free models are highly sensitive to input resolution. Small text in documents may be unreadable at 224x224. Most document models use higher resolutions (1024x1024 or more) or dynamic tiling to preserve text legibility. This significantly increases computation but is necessary for practical document understanding.",id:"note-resolution"}),e.jsx(n,{title:"Structured Output Reliability",content:"OCR-free models generate structured output (JSON, markdown) autoregressively, which means they can produce malformed outputs. Always validate the generated structure and implement fallback parsing. For production systems, consider constrained decoding or grammar-guided generation.",id:"warning-structured-output"})]})}const oe=Object.freeze(Object.defineProperty({__proto__:null,default:x},Symbol.toStringTag,{value:"Module"}));function _(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Chart and Table Understanding"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Charts and tables encode data visually in structured formats that require specialized understanding beyond basic OCR. Models must interpret axes, legends, data relationships, cell structures, and spans. This section covers approaches for extracting structured data from visual charts and tables in documents."}),e.jsx(i,{title:"Chart Understanding",definition:"Chart understanding involves extracting the underlying data table from a chart image, identifying chart type (bar, line, pie, scatter), reading axis labels and values, and answering questions about trends and comparisons. This requires both visual perception and numerical reasoning.",id:"def-chart-understanding"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Table Structure Recognition"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Table structure recognition (TSR) detects the grid structure of tables including rows, columns, merged cells, and headers. The task is often decomposed into cell detection and relationship classification."}),e.jsx(o,{title:"Table Structure Recognition Pipeline",problem:"Extract a 3x4 table with one merged header from a document image.",steps:[{formula:"\\text{Step 1: Detect table region in the document}",explanation:"Use an object detector (DETR, YOLO) to localize the table bounding box."},{formula:"\\text{Step 2: Identify rows and columns}",explanation:"Detect horizontal and vertical separators to establish the grid structure."},{formula:"\\text{Step 3: Detect merged cells (spans)}",explanation:"Classify cell relationships to find colspan/rowspan attributes."},{formula:"\\text{Step 4: Extract cell content via OCR or VLM}",explanation:"Read text from each detected cell region."}],id:"example-table-pipeline"}),e.jsx(a,{title:"chart_table_extraction.py",code:`# Table detection and extraction using transformers
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from PIL import Image
import torch

# Microsoft Table Transformer for table detection
# processor = AutoImageProcessor.from_pretrained(
#     "microsoft/table-transformer-detection"
# )
# model = TableTransformerForObjectDetection.from_pretrained(
#     "microsoft/table-transformer-detection"
# )

# image = Image.open("document.png")
# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)

# # Post-process detections
# target_sizes = torch.tensor([image.size[::-1]])
# results = processor.post_process_object_detection(
#     outputs, threshold=0.7, target_sizes=target_sizes
# )[0]

# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     print(f"Table detected: score={score:.3f}, box={box.tolist()}")

# Chart QA with Pix2Struct
# from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
# chart_model = Pix2StructForConditionalGeneration.from_pretrained(
#     "google/deplot"  # DePlot: chart -> data table
# )
# chart_processor = Pix2StructProcessor.from_pretrained("google/deplot")

# Simulated chart data extraction
def parse_chart_description(chart_type, data_points):
    """Convert extracted chart data to structured format."""
    result = {
        "chart_type": chart_type,
        "data": data_points,
        "summary": {}
    }

    values = [d["value"] for d in data_points]
    result["summary"] = {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "trend": "increasing" if values[-1] > values[0] else "decreasing"
    }
    return result

# Example: bar chart extraction
chart_data = parse_chart_description("bar", [
    {"label": "Q1", "value": 150},
    {"label": "Q2", "value": 230},
    {"label": "Q3", "value": 180},
    {"label": "Q4", "value": 310},
])
print(f"Chart type: {chart_data['chart_type']}")
print(f"Data: {chart_data['data']}")
print(f"Summary: {chart_data['summary']}")`,id:"code-chart-table"}),e.jsx(t,{type:"note",title:"DePlot and ChartQA",content:"Google's DePlot converts chart images directly to linearized data tables, which can then be processed by LLMs for question answering. ChartQA benchmarks show that combining a chart-to-table model with an LLM reasoning module significantly outperforms end-to-end approaches.",id:"note-deplot"}),e.jsx(n,{title:"Numerical Precision",content:"Chart understanding models often struggle with precise numerical reading, especially for values between grid lines or in dense charts. Always validate extracted numbers against any available textual annotations in the chart, and consider using OCR on axis labels as a cross-check.",id:"warning-precision"})]})}const ne=Object.freeze(Object.defineProperty({__proto__:null,default:_},Symbol.toStringTag,{value:"Module"}));function y(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Multi-Page Document Understanding"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Real-world documents like contracts, reports, and academic papers span multiple pages. Multi-page understanding requires models to maintain context across page boundaries, resolve cross-page references, and aggregate information from different sections. This extends single-page models with long-context strategies and page-aware architectures."}),e.jsx(i,{title:"Multi-Page Document Understanding",definition:"Multi-page document understanding processes a sequence of page images $\\{P_1, P_2, \\ldots, P_K\\}$ to answer questions or extract information that may require reasoning across pages. The challenge is encoding all pages within a manageable context while preserving layout and cross-page relationships.",id:"def-multipage"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Strategies for Multi-Page Processing"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Three main approaches handle multi-page documents: (1) process pages independently and aggregate, (2) concatenate all page tokens into one long sequence, or (3) use hierarchical encoding with page-level and document-level representations."}),e.jsx(o,{title:"Token Budget for Multi-Page Documents",problem:"A 10-page document with 1024x1024 images at patch size 16. How many visual tokens total?",steps:[{formula:"\\text{Tokens per page} = (1024 / 16)^2 = 64^2 = 4{,}096",explanation:"Each page produces 4096 visual patch tokens."},{formula:"\\text{Total} = 10 \\times 4{,}096 = 40{,}960 \\text{ visual tokens}",explanation:"All pages together exceed most context windows."},{formula:"\\text{With Perceiver (64 queries)}: 10 \\times 64 = 640 \\text{ tokens}",explanation:"Compression via Perceiver Resampler makes multi-page feasible."}],id:"example-multipage-tokens"}),e.jsx(a,{title:"multipage_processing.py",code:`import torch
import torch.nn as nn
from typing import List

class MultiPageDocEncoder(nn.Module):
    """Hierarchical multi-page document encoder."""
    def __init__(self, page_dim=768, doc_dim=768, num_queries=64, max_pages=20):
        super().__init__()
        # Per-page compression (Perceiver-style)
        self.page_queries = nn.Parameter(torch.randn(num_queries, page_dim) * 0.02)
        self.page_cross_attn = nn.MultiheadAttention(page_dim, 12, batch_first=True)
        self.page_norm = nn.LayerNorm(page_dim)

        # Page position embedding
        self.page_pos = nn.Embedding(max_pages, page_dim)

        # Document-level transformer
        doc_layer = nn.TransformerEncoderLayer(
            doc_dim, 12, doc_dim * 4, batch_first=True, norm_first=True
        )
        self.doc_encoder = nn.TransformerEncoder(doc_layer, num_layers=4)

    def encode_page(self, page_features):
        """Compress single page features to fixed-size representation."""
        B = page_features.shape[0]
        queries = self.page_queries.unsqueeze(0).expand(B, -1, -1)
        queries = self.page_norm(queries)
        compressed, _ = self.page_cross_attn(queries, page_features, page_features)
        return compressed  # (B, num_queries, D)

    def forward(self, pages: List[torch.Tensor]):
        """Process list of page features into document representation.

        Args:
            pages: List of K tensors, each (B, N_patches, D)
        Returns:
            (B, K * num_queries, D) document representation
        """
        page_reps = []
        for i, page in enumerate(pages):
            compressed = self.encode_page(page)  # (B, Q, D)
            # Add page position embedding
            compressed = compressed + self.page_pos(
                torch.tensor(i, device=compressed.device)
            )
            page_reps.append(compressed)

        # Concatenate all page representations
        doc_tokens = torch.cat(page_reps, dim=1)  # (B, K*Q, D)
        return self.doc_encoder(doc_tokens)

# Example: 5-page document
encoder = MultiPageDocEncoder(num_queries=32, max_pages=20)
pages = [torch.randn(2, 4096, 768) for _ in range(5)]  # 5 pages of 4096 tokens each

doc_repr = encoder(pages)
print(f"Input: 5 pages x 4096 tokens = {5 * 4096} total tokens")
print(f"Output: {doc_repr.shape}")  # (2, 160, 768) = 5 pages x 32 queries
print(f"Compression ratio: {5 * 4096 / doc_repr.shape[1]:.0f}x")`,id:"code-multipage"}),e.jsx(t,{type:"tip",title:"Practical Multi-Page Approaches",content:"For production systems, consider hybrid approaches: use OCR + text-based LLM for text-heavy pages, and VLM for pages with complex layouts, figures, or charts. Route pages through different pipelines based on content type to balance cost and accuracy.",id:"note-practical-multipage"}),e.jsx(t,{type:"note",title:"Long-Context VLMs",content:"Models like Qwen2-VL and InternVL2 support processing multiple images within their extended context windows (32K+ tokens). Combined with dynamic resolution, they can handle 10-20 page documents directly, though at significant computational cost.",id:"note-long-context"}),e.jsx(n,{title:"Cross-Page Reference Resolution",content:"Multi-page processing with per-page compression can lose cross-page references like 'see Table 2 on page 5' or running totals. Ensure your approach maintains enough information for cross-page reasoning, potentially with explicit page linking mechanisms.",id:"warning-cross-page"})]})}const se=Object.freeze(Object.defineProperty({__proto__:null,default:y},Symbol.toStringTag,{value:"Module"}));function b(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Diffusion Process Basics"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Diffusion models generate data by learning to reverse a gradual noising process. Starting from pure Gaussian noise, the model iteratively denoises to produce realistic images. The mathematical framework connects forward noise addition (a Markov chain) with a learned reverse process, yielding state-of-the-art generative quality."}),e.jsx(i,{title:"Forward Diffusion Process",definition:"The forward process gradually adds Gaussian noise to data $\\mathbf{x}_0$ over $T$ timesteps according to a variance schedule $\\{\\beta_t\\}_{t=1}^T$. At each step: $q(\\mathbf{x}_t | \\mathbf{x}_{t-1}) = \\mathcal{N}(\\mathbf{x}_t; \\sqrt{1 - \\beta_t}\\,\\mathbf{x}_{t-1}, \\beta_t \\mathbf{I})$.",notation:"Let \\( \\alpha_t = 1 - \\beta_t \\) and \\( \\bar{\\alpha}_t = \\prod_{s=1}^{t} \\alpha_s \\). Then \\( q(\\mathbf{x}_t | \\mathbf{x}_0) = \\mathcal{N}(\\mathbf{x}_t; \\sqrt{\\bar{\\alpha}_t}\\,\\mathbf{x}_0, (1 - \\bar{\\alpha}_t)\\mathbf{I}) \\)",id:"def-forward-diffusion"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The Reparameterization Trick"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["A key insight is that we can sample ",e.jsx(s.InlineMath,{math:"\\mathbf{x}_t"})," directly from"," ",e.jsx(s.InlineMath,{math:"\\mathbf{x}_0"})," without iterating through all intermediate steps:"]}),e.jsx(s.BlockMath,{math:"\\mathbf{x}_t = \\sqrt{\\bar{\\alpha}_t}\\,\\mathbf{x}_0 + \\sqrt{1 - \\bar{\\alpha}_t}\\,\\boldsymbol{\\epsilon}, \\quad \\boldsymbol{\\epsilon} \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I})"}),e.jsx(r,{title:"DDPM Training Objective",statement:"The simplified DDPM loss trains a network $\\epsilon_\\theta$ to predict the noise added at timestep $t$:",proof:"\\mathcal{L}_{\\text{simple}} = \\mathbb{E}_{t, \\mathbf{x}_0, \\boldsymbol{\\epsilon}} \\left[ \\| \\boldsymbol{\\epsilon} - \\boldsymbol{\\epsilon}_\\theta(\\mathbf{x}_t, t) \\|^2 \\right]",corollaries:["This is equivalent to learning the score function (gradient of log probability) of the noised data distribution.","At inference, the model iteratively predicts and removes noise starting from pure Gaussian noise."],id:"thm-ddpm-loss"}),e.jsx(o,{title:"Forward Process Step-by-Step",problem:"Given x_0 (a clean image), compute x_t at t=500 with alpha_bar_500 = 0.05.",steps:[{formula:"\\bar{\\alpha}_{500} = 0.05 \\implies \\sqrt{\\bar{\\alpha}_{500}} \\approx 0.224",explanation:"The signal scaling factor is very small at t=500."},{formula:"\\sqrt{1 - \\bar{\\alpha}_{500}} = \\sqrt{0.95} \\approx 0.975",explanation:"The noise component dominates."},{formula:"\\mathbf{x}_{500} = 0.224 \\cdot \\mathbf{x}_0 + 0.975 \\cdot \\boldsymbol{\\epsilon}",explanation:"At t=500 of 1000, the image is mostly noise with faint signal."}],id:"example-forward-process"}),e.jsx(a,{title:"ddpm_forward_reverse.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDDPM:
    """Simplified DDPM scheduler for forward and reverse processes."""
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.T = num_timesteps
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def forward_process(self, x_0, t):
        """Add noise to x_0 at timestep t: q(x_t | x_0)."""
        noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return x_t, noise

    def training_loss(self, model, x_0):
        """Compute simplified DDPM loss."""
        B = x_0.shape[0]
        t = torch.randint(0, self.T, (B,))
        x_t, noise = self.forward_process(x_0, t)
        noise_pred = model(x_t, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def reverse_step(self, model, x_t, t):
        """Single reverse diffusion step: p(x_{t-1} | x_t)."""
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]

        noise_pred = model(x_t, torch.tensor([t]))
        # Predicted mean
        mu = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
        )
        # Add noise (except at t=0)
        if t > 0:
            sigma = torch.sqrt(beta_t)
            mu = mu + sigma * torch.randn_like(x_t)
        return mu

# Visualize noise schedule
ddpm = SimpleDDPM()
timesteps = [0, 100, 250, 500, 750, 999]
for t in timesteps:
    signal = ddpm.alpha_bar[t].sqrt().item()
    noise = (1 - ddpm.alpha_bar[t]).sqrt().item()
    print(f"t={t:4d}: signal={signal:.3f}, noise={noise:.3f}, SNR={signal/noise:.3f}")`,id:"code-ddpm"}),e.jsx(t,{type:"historical",title:"Diffusion Model History",content:"Sohl-Dickstein et al. (2015) first proposed diffusion for generative modeling. Ho et al. (2020) made it practical with DDPM. Song et al. (2021) unified the framework with score-based SDEs. Dhariwal & Nichol (2021) showed diffusion beats GANs on ImageNet, marking a paradigm shift in generative AI.",id:"note-diffusion-history"}),e.jsx(n,{title:"Slow Sampling",content:"DDPM requires iterating through all T=1000 steps during generation, making it much slower than GANs or VAEs. DDIM (Song et al., 2020) reduces this to ~50 steps, and modern schedulers (DPM-Solver, Euler) can achieve good quality in 20-30 steps.",id:"warning-slow-sampling"})]})}const re=Object.freeze(Object.defineProperty({__proto__:null,default:b},Symbol.toStringTag,{value:"Module"}));function w(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Stable Diffusion Architecture"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Stable Diffusion performs diffusion in a compressed latent space rather than pixel space, making it dramatically more efficient. The architecture consists of three main components: a Variational Autoencoder (VAE) for image compression, a U-Net for noise prediction in latent space, and a CLIP text encoder for conditioning on text prompts."}),e.jsx(i,{title:"Latent Diffusion Model",definition:"A Latent Diffusion Model (LDM) operates in the latent space of a pretrained autoencoder. Given an image $\\mathbf{x}$, the encoder produces $\\mathbf{z} = \\mathcal{E}(\\mathbf{x})$ with spatial compression factor $f$ (typically $f=8$). Diffusion is applied to $\\mathbf{z}$, and the decoder reconstructs: $\\hat{\\mathbf{x}} = \\mathcal{D}(\\hat{\\mathbf{z}})$.",notation:"\\( \\mathbf{z} \\in \\mathbb{R}^{h/f \\times w/f \\times c} \\) where typically \\( c = 4 \\) latent channels",id:"def-ldm"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The Three Components"}),e.jsx("h3",{className:"text-xl font-medium",children:"1. VAE (Autoencoder)"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The VAE compresses a 512x512x3 image to a 64x64x4 latent representation (64x compression). This makes diffusion computationally feasible on consumer GPUs."}),e.jsx(s.BlockMath,{math:"\\text{Encoder: } \\mathbb{R}^{512 \\times 512 \\times 3} \\to \\mathbb{R}^{64 \\times 64 \\times 4}, \\quad \\text{Decoder: } \\mathbb{R}^{64 \\times 64 \\times 4} \\to \\mathbb{R}^{512 \\times 512 \\times 3}"}),e.jsx("h3",{className:"text-xl font-medium",children:"2. U-Net (Noise Predictor)"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The U-Net predicts noise ",e.jsx(s.InlineMath,{math:"\\boldsymbol{\\epsilon}_\\theta(\\mathbf{z}_t, t, \\mathbf{c})"})," in latent space, conditioned on timestep ",e.jsx(s.InlineMath,{math:"t"})," and text embedding"," ",e.jsx(s.InlineMath,{math:"\\mathbf{c}"}),". Cross-attention layers inject text conditioning into the U-Net's intermediate representations."]}),e.jsx("h3",{className:"text-xl font-medium",children:"3. CLIP Text Encoder"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The CLIP text encoder converts the text prompt into a sequence of embeddings that condition the U-Net via cross-attention at multiple resolution levels."}),e.jsx(o,{title:"Stable Diffusion Generation Pipeline",problem:"Trace the steps to generate a 512x512 image from a text prompt.",steps:[{formula:"\\text{Encode text: } \\mathbf{c} = \\text{CLIP}_{\\text{text}}(\\text{prompt}) \\in \\mathbb{R}^{77 \\times 768}",explanation:"Text prompt tokenized and encoded to 77 token embeddings."},{formula:"\\mathbf{z}_T \\sim \\mathcal{N}(\\mathbf{0}, \\mathbf{I}), \\quad \\mathbf{z}_T \\in \\mathbb{R}^{64 \\times 64 \\times 4}",explanation:"Start with random noise in latent space."},{formula:"\\text{For } t = T, T-1, \\ldots, 1: \\mathbf{z}_{t-1} = \\text{denoise}(\\mathbf{z}_t, t, \\mathbf{c})",explanation:"Iteratively denoise using U-Net predictions (typically 20-50 steps)."},{formula:"\\hat{\\mathbf{x}} = \\text{VAE}_{\\text{dec}}(\\mathbf{z}_0) \\in \\mathbb{R}^{512 \\times 512 \\times 3}",explanation:"Decode the clean latent to pixel space."}],id:"example-sd-pipeline"}),e.jsx(a,{title:"stable_diffusion_pipeline.py",code:`from diffusers import StableDiffusionPipeline, DDIMScheduler
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
print(f"\\nTotal: ~{total:.0f}M parameters")`,id:"code-sd-pipeline"}),e.jsx(t,{type:"intuition",title:"Why Latent Space?",content:"Diffusing in pixel space for a 512x512 image means operating on 786,432 dimensions. In latent space (64x64x4), it is only 16,384 dimensions -- a 48x reduction. The VAE preserves perceptual quality while compressing away imperceptible high-frequency details, making diffusion both faster and more focused on meaningful image structure.",id:"note-latent-intuition"}),e.jsx(n,{title:"VAE Quality Limitations",content:"The VAE is a bottleneck for image quality. Fine details like small text, faces at low resolution, and intricate patterns can be lost during VAE encoding/decoding. This is why Stable Diffusion sometimes produces blurry faces or garbled text, independent of the diffusion quality.",id:"warning-vae-quality"})]})}const le=Object.freeze(Object.defineProperty({__proto__:null,default:w},Symbol.toStringTag,{value:"Module"}));function v(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"SDXL and FLUX: Next-Generation Image Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"SDXL and FLUX represent major architectural advances beyond Stable Diffusion 1.5. SDXL scales the U-Net and adds a refiner model for high-resolution generation. FLUX replaces the U-Net entirely with a Diffusion Transformer (DiT), bringing the scaling properties of transformers to image generation with superior prompt adherence and image quality."}),e.jsx(i,{title:"Diffusion Transformer (DiT)",definition:"A Diffusion Transformer replaces the U-Net with a transformer architecture for noise prediction. Instead of convolutional downsampling/upsampling, DiT processes latent patches as a flat token sequence with self-attention. The timestep and class/text conditioning are injected via adaptive layer normalization (adaLN).",id:"def-dit"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"SDXL Architecture"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"SDXL uses a 2.6B parameter U-Net (3x larger than SD 1.5), dual text encoders (CLIP ViT-L + OpenCLIP ViT-bigG), and generates at 1024x1024 base resolution. An optional refiner model improves fine details in a second pass."}),e.jsx("h2",{className:"text-2xl font-semibold",children:"FLUX Architecture"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"FLUX from Black Forest Labs uses a 12B parameter DiT with rectified flow matching instead of DDPM noise scheduling. It processes both text and image tokens in a unified transformer with joint attention, achieving remarkable text rendering and prompt following."}),e.jsx(s.BlockMath,{math:"\\text{FLUX loss: } \\mathcal{L} = \\mathbb{E}_{t, \\mathbf{x}_0, \\mathbf{x}_1} \\left[ \\| \\mathbf{v}_\\theta(\\mathbf{x}_t, t) - (\\mathbf{x}_1 - \\mathbf{x}_0) \\|^2 \\right]"}),e.jsx(o,{title:"Architecture Comparison",problem:"Compare SD 1.5, SDXL, and FLUX architectures.",steps:[{formula:"\\text{SD 1.5: 860M U-Net, 1 CLIP, 512px, DDPM}",explanation:"Original Stable Diffusion with single text encoder."},{formula:"\\text{SDXL: 2.6B U-Net, 2 CLIPs, 1024px, DDPM + refiner}",explanation:"Larger model with dual text encoders and optional refinement stage."},{formula:"\\text{FLUX: 12B DiT, T5-XXL + CLIP, 1024px, Flow Matching}",explanation:"Transformer-based with flow matching for straighter sampling trajectories."}],id:"example-arch-comparison"}),e.jsx(a,{title:"sdxl_flux_generation.py",code:`from diffusers import (
    StableDiffusionXLPipeline,
    # FluxPipeline,
    EulerDiscreteScheduler,
)
import torch

# === SDXL ===
# pipe_xl = StableDiffusionXLPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     torch_dtype=torch.float16,
#     variant="fp16",
# ).to("cuda")

# image = pipe_xl(
#     prompt="An astronaut riding a horse on Mars, cinematic lighting",
#     negative_prompt="blurry, low quality",
#     num_inference_steps=30,
#     guidance_scale=7.0,
#     height=1024, width=1024,
# ).images[0]

# === FLUX ===
# pipe_flux = FluxPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-dev",
#     torch_dtype=torch.bfloat16,
# ).to("cuda")

# # FLUX uses guidance-distilled model (no negative prompt needed)
# image = pipe_flux(
#     prompt="A cat wearing a tiny hat, sitting in a teacup, watercolor style",
#     num_inference_steps=28,
#     guidance_scale=3.5,
#     height=1024, width=1024,
# ).images[0]

# Key differences
print("=== Architecture Comparison ===")
comparison = {
    "Component":     ["Backbone",    "Text Encoder",          "Resolution", "Scheduler",    "Params"],
    "SD 1.5":        ["U-Net",       "CLIP ViT-L",            "512px",      "DDPM/DDIM",    "~1B"],
    "SDXL":          ["U-Net",       "CLIP-L + OpenCLIP-G",   "1024px",     "DDPM/Euler",   "~3.5B"],
    "FLUX.1-dev":    ["DiT (MMDiT)", "T5-XXL + CLIP-L",       "1024px",     "Flow Matching", "~12B"],
}
header = comparison["Component"]
for key in ["SD 1.5", "SDXL", "FLUX.1-dev"]:
    row = comparison[key]
    print(f"\\n{key}:")
    for h, v in zip(header, row):
        print(f"  {h}: {v}")

# Flow matching vs DDPM
print("\\n=== Flow Matching ===")
print("DDPM: curved noise trajectories, many steps needed")
print("Flow: straight paths from noise to data, fewer steps")
print("FLUX achieves good quality in 4 steps (with guidance distillation)")`,id:"code-sdxl-flux"}),e.jsx(t,{type:"intuition",title:"Why DiT Beats U-Net",content:"U-Nets rely on hand-designed skip connections and resolution hierarchies. DiTs treat the problem as sequence modeling, leveraging transformers' proven scaling laws. As compute increases, DiTs improve more predictably. FLUX's joint text-image attention also eliminates the cross-attention bottleneck, giving the model better access to text conditioning.",id:"note-dit-advantage"}),e.jsx(n,{title:"VRAM Requirements",content:"FLUX.1-dev requires ~24GB VRAM in float16, making it impractical for consumer GPUs without quantization. SDXL needs ~7GB in float16. Consider using model offloading (pipe.enable_model_cpu_offload()) or quantized versions (GGUF, NF4) for limited hardware.",id:"warning-vram"})]})}const de=Object.freeze(Object.defineProperty({__proto__:null,default:v},Symbol.toStringTag,{value:"Module"}));function k(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"PixArt-Alpha: Efficient Training of Diffusion Transformers"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"PixArt-Alpha demonstrates that high-quality text-to-image generation is possible with dramatically less training compute than Stable Diffusion or DALL-E. Through a three-stage training strategy and efficient DiT architecture, PixArt-Alpha achieves competitive quality at approximately 2% of the training cost of SD, making it an important milestone for accessible image generation research."}),e.jsx(i,{title:"PixArt-Alpha",definition:"PixArt-Alpha is a Diffusion Transformer (DiT) model that uses cross-attention for T5 text conditioning and adaLN-single for timestep conditioning. It achieves efficient training through: (1) decomposed training with a pretrained class-conditional DiT as initialization, (2) efficient T5 text conditioning, and (3) a high-quality curated dataset (SAM-LLaVA-Captions).",id:"def-pixart"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Three-Stage Training"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Instead of training from scratch on text-image pairs (expensive), PixArt decomposes training into learning pixel distributions, text-image alignment, and aesthetic quality in separate stages."}),e.jsx(o,{title:"PixArt Training Efficiency",problem:"Compare training costs of PixArt-Alpha vs Stable Diffusion v1.5.",steps:[{formula:"\\text{SD 1.5: } 6{,}250 \\text{ A100 GPU-days}",explanation:"Stable Diffusion required massive compute for training from scratch."},{formula:"\\text{PixArt-}\\alpha\\text{: } 133 \\text{ A100 GPU-days}",explanation:"PixArt achieves comparable quality at ~2% of the cost."},{formula:"\\text{CO}_2 \\text{ reduction: } 90\\% \\text{ lower emissions}",explanation:"Efficient training directly reduces environmental impact."}],id:"example-pixart-efficiency"}),e.jsx(a,{title:"pixart_generation.py",code:`from diffusers import PixArtAlphaPipeline
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
    print(f"{name}: {info}")`,id:"code-pixart"}),e.jsx(t,{type:"note",title:"PixArt-Sigma Improvements",content:"PixArt-Sigma improves upon Alpha with better VAE (SDXL's VAE), support for various aspect ratios, and a token compression mechanism that reduces computation for higher resolutions. It also introduces a 'weak-to-strong' training paradigm for efficiently scaling resolution.",id:"note-pixart-sigma"}),e.jsx(t,{type:"tip",title:"Efficient Training Recipe",content:"PixArt's key insight is decomposed training: first learn to generate images (class-conditional pretraining is cheap), then learn text alignment (fine-tuning is cheaper than joint training). This principle applies broadly -- always leverage pretrained components when possible.",id:"note-efficient-recipe"}),e.jsx(n,{title:"T5 Text Encoder Memory",content:"PixArt uses T5-XXL (4.7B params) as text encoder, which alone requires ~10GB in float16. For inference on limited hardware, precompute and cache T5 embeddings, or use the smaller T5-large variant with some quality trade-off.",id:"warning-t5-memory"})]})}const me=Object.freeze(Object.defineProperty({__proto__:null,default:k},Symbol.toStringTag,{value:"Module"}));function L(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Video Diffusion Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Video diffusion extends image diffusion to the temporal dimension, generating sequences of coherent frames. The key challenge is maintaining temporal consistency -- objects should move smoothly, lighting should be stable, and physics should be plausible across frames. Video diffusion models typically add temporal attention layers to an image diffusion backbone."}),e.jsx(i,{title:"Video Diffusion",definition:"Video diffusion operates on 3D latent tensors $\\mathbf{z} \\in \\mathbb{R}^{F \\times H \\times W \\times C}$ where $F$ is the number of frames. The denoising network includes both spatial attention (within each frame) and temporal attention (across frames at the same spatial position) to ensure spatial quality and temporal coherence.",id:"def-video-diffusion"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Temporal Attention"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The standard approach inserts temporal attention layers after spatial attention layers. Spatial attention operates on ",e.jsx(s.InlineMath,{math:"H \\times W"})," tokens per frame, while temporal attention operates on ",e.jsx(s.InlineMath,{math:"F"})," tokens per spatial position."]}),e.jsx(s.BlockMath,{math:"\\text{Spatial: } \\text{Attn}(\\mathbf{z}_{f,:,:}) \\text{ for each frame } f \\quad | \\quad \\text{Temporal: } \\text{Attn}(\\mathbf{z}_{:,h,w}) \\text{ for each position } (h,w)"}),e.jsx(o,{title:"Video Latent Dimensions",problem:"A 16-frame 512x512 video encoded with SD's VAE (f=8) and 4 latent channels.",steps:[{formula:"\\text{Spatial: } 512/8 = 64 \\implies 64 \\times 64 = 4{,}096 \\text{ tokens/frame}",explanation:"Same as image latent diffusion per frame."},{formula:"\\text{Total latent: } 16 \\times 64 \\times 64 \\times 4 = 262{,}144 \\text{ values}",explanation:"Full 3D latent tensor for the video."},{formula:"\\text{Spatial attn: } 4096^2 \\times 16 \\approx 268M \\text{ ops per layer}",explanation:"Spatial attention is expensive but parallelizable across frames."},{formula:"\\text{Temporal attn: } 16^2 \\times 4096 \\approx 1M \\text{ ops per layer}",explanation:"Temporal attention is cheap since F is small."}],id:"example-video-latent"}),e.jsx(a,{title:"temporal_attention.py",code:`import torch
import torch.nn as nn
from einops import rearrange

class TemporalAttentionBlock(nn.Module):
    """Temporal self-attention across video frames."""
    def __init__(self, dim=320, num_heads=8, num_frames=16):
        super().__init__()
        self.num_frames = num_frames
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B*F, H*W, D) from spatial attention
        BF, HW, D = x.shape
        B = BF // self.num_frames
        F = self.num_frames

        # Reshape: (B*F, HW, D) -> (B*HW, F, D) for temporal attention
        x_temporal = rearrange(x, '(b f) hw d -> (b hw) f d', b=B, f=F)

        # Temporal self-attention
        residual = x_temporal
        x_normed = self.norm(x_temporal)
        attended, _ = self.attn(x_normed, x_normed, x_normed)
        x_temporal = residual + self.proj(attended)

        # Reshape back: (B*HW, F, D) -> (B*F, HW, D)
        return rearrange(x_temporal, '(b hw) f d -> (b f) hw d', b=B, hw=HW)

class SpatioTemporalBlock(nn.Module):
    """Combined spatial + temporal attention block."""
    def __init__(self, dim=320, num_heads=8, num_frames=16):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.spatial_norm = nn.LayerNorm(dim)
        self.temporal_attn = TemporalAttentionBlock(dim, num_heads, num_frames)

    def forward(self, x):
        # Spatial attention (within each frame)
        residual = x
        x = self.spatial_norm(x)
        x, _ = self.spatial_attn(x, x, x)
        x = x + residual

        # Temporal attention (across frames)
        x = self.temporal_attn(x)
        return x

# Test
B, F, H, W, D = 2, 16, 8, 8, 320  # Small test
block = SpatioTemporalBlock(dim=D, num_frames=F)
x = torch.randn(B * F, H * W, D)
out = block(x)
print(f"Input: {x.shape} -> Output: {out.shape}")  # (32, 64, 320)`,id:"code-temporal-attn"}),e.jsx(t,{type:"intuition",title:"Factored Attention",content:"Full 3D attention over all frames and spatial positions is prohibitively expensive (O(F^2 * H^2 * W^2)). Factoring into separate spatial and temporal attention reduces this to O(F * H^2 * W^2 + F^2 * H * W), making video generation tractable. Some models add a third factored dimension for cross-frame spatial attention.",id:"note-factored"}),e.jsx(n,{title:"Temporal Consistency",content:"Even with temporal attention, video diffusion models can produce flickering, object deformation, and inconsistent backgrounds. These artifacts are most visible in long videos (>4 seconds) and fast motion. Post-processing with optical flow-based smoothing or video interpolation can help.",id:"warning-consistency"})]})}const ce=Object.freeze(Object.defineProperty({__proto__:null,default:L},Symbol.toStringTag,{value:"Module"}));function j(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Stable Video Diffusion (SVD)"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Stable Video Diffusion from Stability AI adapts the Stable Diffusion architecture for video generation. It takes a single image as input and generates a short video clip (14-25 frames) showing plausible motion. SVD is trained in three stages: image pretraining, video pretraining on large data, and high-quality video fine-tuning."}),e.jsx(i,{title:"Stable Video Diffusion",definition:"SVD extends SD's U-Net with temporal convolution and temporal attention layers. Given a conditioning image $\\mathbf{x}_{\\text{cond}}$, it generates $F$ frames by denoising a 3D latent tensor $\\mathbf{z} \\in \\mathbb{R}^{F \\times h \\times w \\times c}$. The conditioning image is encoded by the VAE and concatenated channel-wise with the noisy latent at each denoising step.",id:"def-svd"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Image-to-Video Pipeline"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"SVD operates as an image-to-video model. The input image conditions the generation, and additional parameters control motion magnitude and frame rate. The model learns plausible motion patterns from its video training data."}),e.jsx(o,{title:"SVD Architecture Components",problem:"What are the added components in SVD compared to SD 2.1?",steps:[{formula:"\\text{Temporal Conv: 1D convolution along time axis}",explanation:"Inserted after each spatial 2D convolution to capture local temporal patterns."},{formula:"\\text{Temporal Attn: self-attention across } F \\text{ frames}",explanation:"Inserted after each spatial self-attention for global temporal coherence."},{formula:"\\text{Image conditioning: } \\text{concat}(\\mathbf{z}_t, \\text{VAE}(\\mathbf{x}_{\\text{cond}}))",explanation:"Conditioning image latent concatenated channel-wise with noisy latent."},{formula:"\\text{Motion bucket: FPS + motion magnitude embedding}",explanation:"Learnable embeddings control speed and amount of motion."}],id:"example-svd-components"}),e.jsx(a,{title:"svd_generation.py",code:`from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import torch

# Load SVD pipeline
# pipe = StableVideoDiffusionPipeline.from_pretrained(
#     "stabilityai/stable-video-diffusion-img2vid-xt",  # 25 frames
#     torch_dtype=torch.float16,
#     variant="fp16",
# ).to("cuda")
# pipe.enable_model_cpu_offload()  # Required for most GPUs

# # Generate video from image
# image = Image.open("landscape.png").resize((1024, 576))
# frames = pipe(
#     image,
#     num_frames=25,           # Number of output frames
#     decode_chunk_size=8,     # Decode frames in chunks to save memory
#     motion_bucket_id=127,    # 0-255, higher = more motion
#     fps=7,                   # Frames per second
#     noise_aug_strength=0.02, # Noise augmentation on conditioning image
# ).frames[0]

# # Save as GIF
# frames[0].save("output.gif", save_all=True, append_images=frames[1:],
#                 duration=1000//7, loop=0)

# SVD temporal layer structure
import torch.nn as nn

class TemporalConvLayer(nn.Module):
    """1D temporal convolution for local temporal modeling."""
    def __init__(self, channels, num_frames):
        super().__init__()
        self.num_frames = num_frames
        self.conv = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        # Zero-init for residual connection
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, x):
        # x: (B*F, C, H, W)
        BF, C, H, W = x.shape
        B = BF // self.num_frames

        residual = x
        # Reshape for temporal conv: (B*H*W, C, F)
        x = x.reshape(B, self.num_frames, C, H, W)
        x = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, C, self.num_frames)

        x = self.conv(x)

        # Reshape back
        x = x.reshape(B, H, W, C, self.num_frames)
        x = x.permute(0, 4, 3, 1, 2).reshape(BF, C, H, W)

        return residual + x

# Test
layer = TemporalConvLayer(channels=320, num_frames=14)
x = torch.randn(2 * 14, 320, 64, 64)
out = layer(x)
print(f"Temporal conv: {x.shape} -> {out.shape}")
print(f"SVD-XT: 25 frames @ 576x1024, ~3.5B params")`,id:"code-svd"}),e.jsx(t,{type:"note",title:"SVD Training Stages",content:"Stage 1: Image pretraining (SD 2.1 weights). Stage 2: Video pretraining on Large Video Dataset (580M clips, filtered to 152M). Stage 3: High-quality fine-tuning on 1M carefully curated clips. This staged approach is critical -- training on video from scratch is extremely expensive.",id:"note-svd-stages"}),e.jsx(n,{title:"SVD Limitations",content:"SVD generates short clips (2-4 seconds) without camera control, text conditioning, or multi-shot consistency. It often produces subtle 'breathing' artifacts where the image gently warps. For longer or more controlled videos, consider AnimateDiff, CogVideo, or Sora-type approaches.",id:"warning-svd-limitations"})]})}const pe=Object.freeze(Object.defineProperty({__proto__:null,default:j},Symbol.toStringTag,{value:"Module"}));function C(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Sora: Scaling Video Generation"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"OpenAI's Sora represents a paradigm shift in video generation by treating videos as sequences of spacetime patches processed by a Diffusion Transformer. Sora can generate up to 60 seconds of high-fidelity video with complex scenes, consistent characters, and realistic physics. Its key insight is that video generation benefits from the same scaling laws as language models."}),e.jsx(i,{title:"Spacetime Patches",definition:"Sora decomposes video into spacetime patches: 3D chunks of size $t_p \\times h_p \\times w_p$ from the compressed video latent. These patches are flattened and linearly projected to form tokens, analogous to ViT's image patches but extended to the temporal dimension. This enables variable duration, resolution, and aspect ratio.",id:"def-spacetime-patches"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Architecture Principles"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"While Sora's full architecture is not public, the technical report reveals key design principles: a video compression network (temporal VAE), a DiT backbone operating on spacetime patches, and training on native resolution videos without cropping."}),e.jsx(s.BlockMath,{math:"\\text{Spacetime tokens: } N = \\frac{F}{t_p} \\times \\frac{H}{h_p} \\times \\frac{W}{w_p}"}),e.jsx(o,{title:"Sora Token Count Estimation",problem:"Estimate tokens for a 10-second 1080p video at 24fps with temporal VAE compression 4x and spatial compression 8x, using 2x2x2 spacetime patches.",steps:[{formula:"\\text{Frames: } 10 \\times 24 = 240, \\text{ after temporal VAE: } 240 / 4 = 60",explanation:"Temporal compression reduces frame count."},{formula:"\\text{Spatial: } 1080/8 = 135, \\; 1920/8 = 240 \\text{ (latent spatial)}",explanation:"Spatial VAE compression."},{formula:"\\text{Spacetime patches: } (60/2) \\times (135/2) \\times (240/2) \\approx 2.4M",explanation:"Massive token count requires efficient attention."}],id:"example-sora-tokens"}),e.jsx(a,{title:"spacetime_patch_embedding.py",code:`import torch
import torch.nn as nn

class SpacetimePatchEmbed(nn.Module):
    """Convert video latents to spacetime patch tokens (Sora-style)."""
    def __init__(self, in_channels=4, embed_dim=1024,
                 patch_size_t=2, patch_size_h=2, patch_size_w=2):
        super().__init__()
        self.patch_size = (patch_size_t, patch_size_h, patch_size_w)
        # 3D convolution acts as patch extraction + projection
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        # x: (B, C, T, H, W) - video latent
        x = self.proj(x)  # (B, D, T', H', W')
        B, D, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, T'*H'*W', D)
        return x, (T, H, W)

class VideoTemporalVAE(nn.Module):
    """Simplified temporal VAE for video compression."""
    def __init__(self, in_channels=3, latent_channels=4,
                 spatial_downsample=8, temporal_downsample=4):
        super().__init__()
        self.spatial_down = spatial_downsample
        self.temporal_down = temporal_downsample

        # Simplified: in practice, this is a 3D convolutional encoder
        self.encoder = nn.Conv3d(
            in_channels, latent_channels,
            kernel_size=(temporal_downsample, spatial_downsample, spatial_downsample),
            stride=(temporal_downsample, spatial_downsample, spatial_downsample),
        )

    def encode(self, video):
        # video: (B, C, T, H, W)
        return self.encoder(video)

# Demonstrate the pipeline
B, C, T, H, W = 1, 3, 48, 256, 256

# Step 1: Temporal VAE compression
vae = VideoTemporalVAE()
video = torch.randn(B, C, T, H, W)
latent = vae.encode(video)
print(f"Video:  {video.shape}")   # (1, 3, 48, 256, 256)
print(f"Latent: {latent.shape}")  # (1, 4, 12, 32, 32)

# Step 2: Spacetime patch embedding
patch_embed = SpacetimePatchEmbed(in_channels=4, embed_dim=1024)
tokens, grid = patch_embed(latent)
print(f"Tokens: {tokens.shape}")  # (1, 6*16*16=1536, 1024)
print(f"Grid:   {grid}")          # (6, 16, 16)

# Step 3: Process with DiT (self-attention over all spacetime tokens)
dit_layer = nn.TransformerEncoderLayer(
    d_model=1024, nhead=16, dim_feedforward=4096,
    batch_first=True, norm_first=True
)
output = dit_layer(tokens)
print(f"DiT output: {output.shape}")  # (1, 1536, 1024)`,id:"code-sora"}),e.jsx(t,{type:"intuition",title:"Videos as World Simulators",content:"Sora's technical report describes the model as a 'world simulator' -- by learning to predict video frames, the model implicitly learns about 3D geometry, physics, object permanence, and even basic cause-and-effect. This emergent understanding improves with scale, similar to how LLMs develop reasoning abilities.",id:"note-world-simulator"}),e.jsx(t,{type:"historical",title:"Open Source Alternatives",content:"Since Sora is not publicly available, the community has developed open alternatives: Open-Sora (HPC-AI Tech), Open-Sora-Plan, and CogVideoX (Tsinghua/Zhipu). These replicate core ideas (spacetime DiT, temporal VAE) at smaller scale, enabling research and experimentation.",id:"note-open-sora"}),e.jsx(n,{title:"Compute Requirements",content:"Sora-scale video generation likely requires thousands of GPUs for training and significant resources even for inference (a 60-second 1080p video may take minutes on high-end hardware). The compute gap between image and video generation is roughly 100-1000x.",id:"warning-sora-compute"})]})}const fe=Object.freeze(Object.defineProperty({__proto__:null,default:C},Symbol.toStringTag,{value:"Module"}));function A(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"CogVideo and CogVideoX"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"CogVideo from Tsinghua University and Zhipu AI is a family of text-to-video models. CogVideoX, the latest iteration, uses a 3D VAE and expert transformer blocks with 3D full attention over spacetime, achieving high-quality open-source video generation that approaches proprietary systems in quality and coherence."}),e.jsx(i,{title:"CogVideoX Architecture",definition:"CogVideoX uses a 3D causal VAE that compresses videos by 4x temporally and 8x spatially. The denoising backbone is a DiT with expert adaptive LayerNorm, processing joint text-video token sequences with full 3D attention. Text tokens from T5-XXL are concatenated with video latent patch tokens.",id:"def-cogvideox"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"3D Causal VAE"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The 3D VAE in CogVideoX uses causal convolutions along the temporal axis, ensuring that frame ",e.jsx(s.InlineMath,{math:"t"})," only depends on frames ",e.jsx(s.InlineMath,{math:"\\leq t"}),". This enables autoregressive extension of videos beyond the training length."]}),e.jsx(o,{title:"CogVideoX Generation Specs",problem:"CogVideoX-5B generates 6-second 720p videos at 8fps. What are the latent dimensions?",steps:[{formula:"\\text{Frames: } 6 \\times 8 = 48, \\text{ after 4x temporal: } 48 / 4 = 12",explanation:"Temporal compression yields 12 latent frames."},{formula:"\\text{Spatial: } 720/8 \\times 1280/8 = 90 \\times 160",explanation:"Spatial compression from pixel to latent space."},{formula:"\\text{With 16 latent channels: } 12 \\times 90 \\times 160 \\times 16 \\approx 2.76M \\text{ values}",explanation:"Total latent tensor size for denoising."}],id:"example-cogvideox-specs"}),e.jsx(a,{title:"cogvideox_generation.py",code:`from diffusers import CogVideoXPipeline
import torch

# CogVideoX-5B text-to-video
# pipe = CogVideoXPipeline.from_pretrained(
#     "THUDM/CogVideoX-5b",
#     torch_dtype=torch.bfloat16,
# )
# pipe.enable_model_cpu_offload()
# pipe.vae.enable_tiling()  # Tile-based VAE decoding to save memory

# prompt = "A golden retriever running through a field of sunflowers, slow motion"
# video_frames = pipe(
#     prompt=prompt,
#     num_inference_steps=50,
#     guidance_scale=6.0,
#     num_frames=48,  # 6 seconds at 8fps
# ).frames[0]

# # Save as MP4
# from diffusers.utils import export_to_video
# export_to_video(video_frames, "dog_sunflowers.mp4", fps=8)

# CogVideoX model comparison
models = {
    "CogVideoX-2B": {
        "params": "2B",
        "resolution": "720x480",
        "frames": 48,
        "fps": 8,
        "vae_channels": 16,
    },
    "CogVideoX-5B": {
        "params": "5B",
        "resolution": "1280x720",
        "frames": 48,
        "fps": 8,
        "vae_channels": 16,
    },
}

for name, cfg in models.items():
    print(f"\\n{name}:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

# 3D causal convolution concept
import torch.nn as nn

class CausalConv3d(nn.Module):
    """3D convolution with causal padding along temporal axis."""
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.temporal_pad = kernel_size - 1  # Causal: pad only past
        self.spatial_pad = kernel_size // 2  # Symmetric spatial padding
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, padding=0)

    def forward(self, x):
        # x: (B, C, T, H, W)
        # Pad: (W_left, W_right, H_top, H_bottom, T_past, T_future)
        x = nn.functional.pad(x, (
            self.spatial_pad, self.spatial_pad,   # W
            self.spatial_pad, self.spatial_pad,   # H
            self.temporal_pad, 0,                 # T: only past padding
        ))
        return self.conv(x)

# Test causal conv
conv = CausalConv3d(4, 16, kernel_size=3)
x = torch.randn(1, 4, 12, 90, 160)
out = conv(x)
print(f"\\nCausal Conv3D: {x.shape} -> {out.shape}")  # Same temporal dim`,id:"code-cogvideo"}),e.jsx(t,{type:"note",title:"Progressive Training",content:"CogVideoX uses progressive training: first low-resolution short videos, then gradually increasing to full resolution and length. This curriculum learning approach stabilizes training and is more efficient than directly training at the target resolution.",id:"note-progressive"}),e.jsx(n,{title:"Memory for Video Generation",content:"CogVideoX-5B requires ~30GB VRAM even with model offloading and VAE tiling. The 3D attention over all spacetime tokens is memory-intensive. For consumer GPUs, use the 2B variant or apply further optimizations like quantization and attention slicing.",id:"warning-cogvideo-memory"})]})}const ue=Object.freeze(Object.defineProperty({__proto__:null,default:A},Symbol.toStringTag,{value:"Module"}));function T(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"AnimateDiff: Plug-and-Play Motion Modules"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"AnimateDiff introduces a modular approach to video generation by training a standalone motion module that can be plugged into any personalized Stable Diffusion model (including LoRA-customized checkpoints). This means you can animate any image style without retraining the base model -- the motion module provides temporal coherence while the base model controls visual style."}),e.jsx(i,{title:"Motion Module",definition:"A motion module is a set of temporal attention layers that are inserted into a frozen Stable Diffusion U-Net. Each motion module block operates on features across frames at a given spatial resolution, learning general motion priors from video data. The key property is compatibility: the same motion module works with any SD 1.5 checkpoint.",id:"def-motion-module"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Architecture Design"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"AnimateDiff inserts temporal transformer blocks after each spatial self-attention block in the U-Net. During training, only the motion module parameters are updated while the SD U-Net remains frozen. At inference, the motion module can be combined with any compatible SD checkpoint."}),e.jsx(o,{title:"AnimateDiff Modularity",problem:"How does AnimateDiff enable animating a custom art style?",steps:[{formula:"\\text{Step 1: User has custom SD model (e.g., anime LoRA)}",explanation:"Any SD 1.5 checkpoint or LoRA that generates desired visual style."},{formula:"\\text{Step 2: Insert motion module into U-Net}",explanation:"Temporal attention layers are added between existing spatial layers."},{formula:"\\text{Step 3: Generate video with combined model}",explanation:"Base model controls style, motion module controls movement."},{formula:"\\text{No retraining needed!}",explanation:"The same motion module generalizes across different style checkpoints."}],id:"example-animatediff-modularity"}),e.jsx(a,{title:"animatediff_generation.py",code:`from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif
import torch

# Load motion module adapter
# adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-3")

# # Combine with any SD 1.5 model
# pipe = AnimateDiffPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",  # Can swap with any SD 1.5 model
#     motion_adapter=adapter,
#     torch_dtype=torch.float16,
# ).to("cuda")
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# # Generate animation
# output = pipe(
#     prompt="A cat playing with a ball of yarn, detailed fur, soft lighting",
#     negative_prompt="blurry, low quality, static",
#     num_frames=16,
#     num_inference_steps=25,
#     guidance_scale=7.5,
# )
# export_to_gif(output.frames[0], "cat_yarn.gif")

# # Use with a custom LoRA for style
# pipe.load_lora_weights("custom_anime_lora", adapter_name="anime")
# pipe.set_adapters(["anime"], [0.8])  # LoRA weight
# output_anime = pipe(
#     prompt="anime girl walking in a garden, cherry blossoms",
#     num_frames=16,
#     num_inference_steps=25,
# )
# export_to_gif(output_anime.frames[0], "anime_walk.gif")

# Simplified motion module structure
import torch.nn as nn

class MotionModuleBlock(nn.Module):
    """Single AnimateDiff motion module block."""
    def __init__(self, channels, num_frames=16, num_heads=8):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(
            channels, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(channels)
        self.pos_embed = nn.Parameter(torch.randn(1, num_frames, channels) * 0.02)
        # Zero-init output projection for clean integration
        self.proj_out = nn.Linear(channels, channels)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, num_frames):
        # x: (B*F, HW, C) from spatial attention
        BF, HW, C = x.shape
        B = BF // num_frames

        # Reshape for temporal attention
        x_t = x.reshape(B, num_frames, HW, C)
        x_t = x_t.permute(0, 2, 1, 3).reshape(B * HW, num_frames, C)

        # Add temporal positional embedding
        x_t = x_t + self.pos_embed[:, :num_frames]

        residual = x_t
        x_t = self.norm(x_t)
        x_t, _ = self.temporal_attn(x_t, x_t, x_t)
        x_t = residual + self.proj_out(x_t)

        # Reshape back
        x_t = x_t.reshape(B, HW, num_frames, C)
        x_t = x_t.permute(0, 2, 1, 3).reshape(BF, HW, C)
        return x_t

# Test
block = MotionModuleBlock(channels=320, num_frames=16)
x = torch.randn(32, 64, 320)  # B*F=32, HW=64
out = block(x, num_frames=16)
print(f"Motion module: {x.shape} -> {out.shape}")
print(f"Motion module params: {sum(p.numel() for p in block.parameters()) / 1e6:.1f}M")`,id:"code-animatediff"}),e.jsx(t,{type:"tip",title:"AnimateDiff Versions",content:"AnimateDiff v1 produces basic motion, v2 improves quality, and v3 adds sparse controlnet for motion guidance. AnimateDiff-Lightning uses distillation for faster generation (4-8 steps). SparseCtrl allows controlling specific frames as keyframes for more directed animation.",id:"note-versions"}),e.jsx(n,{title:"SD Version Compatibility",content:"AnimateDiff motion modules are architecture-specific: v1-v3 modules work only with SD 1.5 models. They are NOT compatible with SDXL, FLUX, or SD 2.x without dedicated motion modules trained for those architectures. Always check version compatibility before combining components.",id:"warning-compatibility"})]})}const he=Object.freeze(Object.defineProperty({__proto__:null,default:T},Symbol.toStringTag,{value:"Module"}));function S(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"DreamBooth: Personalizing Diffusion Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:`DreamBooth fine-tunes the entire diffusion model to learn a specific subject from just 3-5 images. By binding the subject to a unique identifier token (e.g., "sks"), the model can generate novel images of that subject in diverse contexts, poses, and styles while preserving the subject's key visual features.`}),e.jsx(i,{title:"DreamBooth",definition:"DreamBooth fine-tunes a text-to-image diffusion model to associate a unique token identifier $V^*$ (e.g., 'sks') with a specific subject. The training objective is: $\\mathcal{L} = \\mathbb{E}_{t, \\epsilon} [\\| \\epsilon - \\epsilon_\\theta(\\mathbf{x}_t, t, \\text{'a photo of } V^* \\text{ [class]'}) \\|^2]$ where [class] is the subject's class (e.g., 'dog', 'person').",id:"def-dreambooth"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Prior Preservation Loss"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"A critical component is the prior preservation loss that prevents catastrophic forgetting. The model generates class-specific images with the frozen original model, and uses these as regularization during fine-tuning."}),e.jsx(s.BlockMath,{math:"\\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{reconstruction}} + \\lambda \\cdot \\mathcal{L}_{\\text{prior}}"}),e.jsx(o,{title:"DreamBooth Training Setup",problem:"Fine-tune SD to learn your pet dog 'Rex' from 5 photos.",steps:[{formula:'\\text{Identifier: } V^* = \\text{"sks"}, \\text{ Class: "dog"}',explanation:"Choose a rare token as identifier, paired with the class noun."},{formula:'\\text{Training prompts: "a photo of sks dog"}',explanation:"All 5 training images use this caption."},{formula:'\\text{Prior: Generate 200 images of "a photo of dog"}',explanation:"Use the original model to create class regularization images."},{formula:"\\text{Fine-tune 800-1200 steps, lr=5e-6}",explanation:"Short fine-tuning with low learning rate to avoid overfitting."}],id:"example-dreambooth-setup"}),e.jsx(a,{title:"dreambooth_training.py",code:`# DreamBooth training with diffusers
# Command-line approach (recommended):
# accelerate launch train_dreambooth.py \\
#   --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\
#   --instance_data_dir="./my_dog_photos" \\
#   --class_data_dir="./class_dog_photos" \\
#   --output_dir="./dreambooth_dog" \\
#   --instance_prompt="a photo of sks dog" \\
#   --class_prompt="a photo of dog" \\
#   --with_prior_preservation --prior_loss_weight=1.0 \\
#   --num_class_images=200 \\
#   --resolution=512 \\
#   --train_batch_size=1 \\
#   --gradient_accumulation_steps=1 \\
#   --learning_rate=5e-6 \\
#   --lr_scheduler="constant" \\
#   --max_train_steps=800 \\
#   --mixed_precision="fp16"

# Simplified DreamBooth training loop
import torch
import torch.nn.functional as F

def dreambooth_training_step(
    unet, vae, text_encoder, tokenizer, noise_scheduler,
    instance_batch, class_batch, instance_prompt, class_prompt,
    prior_loss_weight=1.0
):
    """Single DreamBooth training step with prior preservation."""

    def compute_loss(images, prompt):
        # Encode images to latent
        latents = vae.encode(images).latent_dist.sample() * 0.18215

        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=latents.device)
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)

        # Text conditioning
        tokens = tokenizer(prompt, padding=True, return_tensors="pt").input_ids
        text_emb = text_encoder(tokens.to(latents.device))[0]

        # Predict noise
        noise_pred = unet(noisy, timesteps, text_emb).sample
        return F.mse_loss(noise_pred, noise)

    # Instance loss (subject-specific)
    loss_instance = compute_loss(instance_batch, instance_prompt)

    # Prior preservation loss (class-specific)
    loss_prior = compute_loss(class_batch, class_prompt)

    # Combined loss
    total_loss = loss_instance + prior_loss_weight * loss_prior
    return total_loss, loss_instance.item(), loss_prior.item()

# Hyperparameter guidelines
print("DreamBooth Hyperparameters:")
configs = {
    "Learning rate": "1e-6 to 5e-6 (lower for faces)",
    "Training steps": "800-1200 (more steps = more overfitting)",
    "Instance images": "3-5 (more is better, up to ~20)",
    "Prior images": "200-300 per class",
    "Prior loss weight": "1.0 (standard)",
    "Resolution": "512 (SD 1.5) or 1024 (SDXL)",
}
for k, v in configs.items():
    print(f"  {k}: {v}")`,id:"code-dreambooth"}),e.jsx(t,{type:"tip",title:"DreamBooth Tips",content:"Use diverse training images (different angles, lighting, backgrounds). Avoid images that are too similar. For human faces, 10-20 high-quality photos work best. The 'sks' identifier is commonly used, but any rare token works. Check results every 200 steps to catch overfitting early.",id:"note-dreambooth-tips"}),e.jsx(n,{title:"Overfitting and Language Drift",content:"Without prior preservation, DreamBooth quickly overfits: the model can only generate the training images and loses diversity. Language drift is another risk where the class word ('dog') becomes synonymous with only your specific subject. Prior preservation loss mitigates both issues.",id:"warning-overfitting"})]})}const ge=Object.freeze(Object.defineProperty({__proto__:null,default:S},Symbol.toStringTag,{value:"Module"}));function I(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"LoRA for Stable Diffusion"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Low-Rank Adaptation (LoRA) is the most popular method for fine-tuning Stable Diffusion models. Instead of updating all model weights (860M+ parameters), LoRA trains small rank-decomposed matrices that are added to specific layers. This reduces training memory by 3x and produces small checkpoint files (2-200MB vs 2-7GB for full models) that can be easily shared, stacked, and swapped at inference time."}),e.jsx(i,{title:"LoRA for Diffusion",definition:"For a pretrained weight matrix $\\mathbf{W}_0 \\in \\mathbb{R}^{d \\times k}$, LoRA learns a low-rank update $\\Delta\\mathbf{W} = \\mathbf{B}\\mathbf{A}$ where $\\mathbf{B} \\in \\mathbb{R}^{d \\times r}$ and $\\mathbf{A} \\in \\mathbb{R}^{r \\times k}$ with rank $r \\ll \\min(d, k)$. The forward pass becomes: $\\mathbf{h} = \\mathbf{W}_0\\mathbf{x} + \\alpha \\cdot \\mathbf{B}\\mathbf{A}\\mathbf{x}$ where $\\alpha$ is a scaling factor.",id:"def-lora-sd"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Where to Apply LoRA"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"In Stable Diffusion, LoRA is typically applied to the attention layers (Q, K, V, O projections) and optionally to the cross-attention layers. Some trainers also apply LoRA to the text encoder for better prompt understanding."}),e.jsx(s.BlockMath,{math:"\\text{Trainable params} = r \\times (d + k) \\times N_{\\text{layers}} \\ll d \\times k \\times N_{\\text{layers}}"}),e.jsx(o,{title:"LoRA Parameter Count",problem:"SD 1.5 U-Net has attention layers with d=k=320 to 1280. With rank 32, how many LoRA parameters?",steps:[{formula:"\\text{Per layer (rank 32, d=k=768): } 32 \\times 768 \\times 2 = 49{,}152",explanation:"Each LoRA pair (A, B) for one projection layer."},{formula:"\\text{4 projections (Q,K,V,O) per attention block} \\times 49{,}152 \\approx 200K",explanation:"All attention projections in one block."},{formula:"\\text{Total across U-Net: } \\sim 3\\text{-}50M \\text{ params}",explanation:"Depends on which layers and rank. ~1-5% of full model."}],id:"example-lora-params"}),e.jsx(a,{title:"lora_sd_training.py",code:`# LoRA training for Stable Diffusion with diffusers + PEFT
# Command-line training:
# accelerate launch train_text_to_image_lora.py \\
#   --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\
#   --dataset_name="my_dataset" \\
#   --output_dir="./lora_output" \\
#   --resolution=512 \\
#   --train_batch_size=4 \\
#   --gradient_accumulation_steps=4 \\
#   --num_train_epochs=100 \\
#   --learning_rate=1e-4 \\
#   --lr_scheduler="cosine" \\
#   --rank=32 \\
#   --mixed_precision="fp16"

from diffusers import StableDiffusionPipeline
import torch

# Load base model and apply LoRA
# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
# ).to("cuda")

# # Load a single LoRA
# pipe.load_lora_weights("path/to/lora", adapter_name="style_lora")
# pipe.set_adapters(["style_lora"], [0.8])  # weight=0.8

# # Stack multiple LoRAs
# pipe.load_lora_weights("path/to/lora2", adapter_name="character_lora")
# pipe.set_adapters(
#     ["style_lora", "character_lora"],
#     [0.7, 0.5]  # Different weights for each
# )

# # Generate with combined LoRAs
# image = pipe("a portrait of sks person in anime style",
#              num_inference_steps=30).images[0]

# LoRA weight merging for inference speed
# pipe.fuse_lora(lora_scale=0.8)  # Merge LoRA into base weights
# pipe.unload_lora_weights()       # Free LoRA adapter memory

# LoRA rank comparison
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scale

# Compare ranks
for rank in [4, 8, 16, 32, 64, 128]:
    lora = LoRALayer(768, 768, rank=rank)
    params = sum(p.numel() for p in lora.parameters())
    full_params = 768 * 768
    print(f"Rank {rank:3d}: {params:>8,} params ({params/full_params*100:.1f}% of full)")`,id:"code-lora-sd"}),e.jsx(t,{type:"tip",title:"Rank Selection Guide",content:"Rank 4-8: Style transfer, simple concepts. Rank 16-32: Character consistency, specific art styles. Rank 64-128: Complex subjects, photorealistic faces. Higher ranks capture more detail but risk overfitting and produce larger files. Start with rank 32 as a default.",id:"note-rank-guide"}),e.jsx(n,{title:"LoRA Compatibility",content:"LoRAs are model-specific: a LoRA trained on SD 1.5 will not work with SDXL or vice versa. The architecture, attention dimensions, and layer names must match exactly. SDXL LoRAs are typically 2-4x larger than SD 1.5 LoRAs due to the larger U-Net.",id:"warning-lora-compat"})]})}const xe=Object.freeze(Object.defineProperty({__proto__:null,default:I},Symbol.toStringTag,{value:"Module"}));function D(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Textual Inversion"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:`Textual Inversion learns a new "word" in the text encoder's embedding space to represent a specific concept (object, style, or texture) from a few example images. Unlike DreamBooth or LoRA, it does not modify any model weights -- only a single embedding vector is optimized. This produces extremely small files (a few KB) but has limited expressiveness.`}),e.jsx(i,{title:"Textual Inversion",definition:"Textual Inversion optimizes a new embedding vector $v^* \\in \\mathbb{R}^{D}$ in the text encoder's token embedding space to represent a target concept. The text encoder and diffusion model remain frozen. The training objective is: $v^* = \\arg\\min_v \\mathbb{E}_{t, \\epsilon}[\\| \\epsilon - \\epsilon_\\theta(\\mathbf{x}_t, t, c_\\theta(\\text{'a photo of } S^*\\text{'})) \\|^2]$ where $S^*$ maps to $v^*$.",id:"def-textual-inversion"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"How It Works"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["A new token (e.g., ","<my-concept>",") is added to the tokenizer's vocabulary. Its embedding vector is initialized (randomly or from a related word) and optimized via gradient descent while all other parameters remain frozen. The learned embedding captures the concept in the text encoder's latent space."]}),e.jsx(o,{title:"Textual Inversion vs DreamBooth vs LoRA",problem:"Compare the three personalization methods on key dimensions.",steps:[{formula:"\\text{Textual Inv: 1 vector (768D)} \\approx 3\\text{KB file}",explanation:"Only learns an embedding. Smallest, least expressive."},{formula:"\\text{LoRA: rank-decomposed matrices} \\approx 2\\text{-}200\\text{MB}",explanation:"Modifies attention layers. Good balance of quality and size."},{formula:"\\text{DreamBooth: full model fine-tune} \\approx 2\\text{-}7\\text{GB}",explanation:"Highest quality but largest storage and most prone to overfitting."}],id:"example-comparison"}),e.jsx(a,{title:"textual_inversion_training.py",code:`# Textual inversion training with diffusers
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
print(f"  DreamBooth (SDXL):          ~7 GB")`,id:"code-textual-inversion"}),e.jsx(t,{type:"intuition",title:"Embedding Space Geometry",content:"Textual inversion works because the CLIP text embedding space is structured: similar concepts cluster together. The learned embedding positions itself among related concepts, inheriting compositional properties. You can use 'a painting of <my-cat> in a garden' and the model compositionally understands both the learned concept and the context.",id:"note-embedding-geometry"}),e.jsx(n,{title:"Expressiveness Limitations",content:"A single embedding vector cannot capture complex appearances with high fidelity. Textual inversion works well for styles and textures but struggles with specific faces or intricate objects. For high-fidelity subject reproduction, combine textual inversion with LoRA or use DreamBooth.",id:"warning-expressiveness"})]})}const _e=Object.freeze(Object.defineProperty({__proto__:null,default:D},Symbol.toStringTag,{value:"Module"}));function P(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"AnimateDiff LoRA: Training Custom Motion"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"AnimateDiff LoRA applies LoRA to the temporal attention layers of AnimateDiff's motion module, enabling custom motion patterns (e.g., specific camera movements, character animations, or physics behaviors) without full retraining. This combines the modularity of AnimateDiff with the efficiency of LoRA fine-tuning."}),e.jsx(i,{title:"Motion LoRA",definition:"Motion LoRA applies low-rank adaptation specifically to the temporal attention layers of AnimateDiff's motion module. Given temporal attention weights $\\mathbf{W}_{\\text{temp}}$, the motion LoRA modifies them as $\\mathbf{W}_{\\text{temp}} + \\alpha \\mathbf{B}_{\\text{motion}}\\mathbf{A}_{\\text{motion}}$ to encode specific motion patterns from reference video clips.",id:"def-motion-lora"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Training Pipeline"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Motion LoRA training uses short video clips (2-4 seconds) that demonstrate the desired motion. The base motion module and SD U-Net are frozen; only the LoRA layers on temporal attention are trained. This requires significantly less data and compute than training a full motion module."}),e.jsx(o,{title:"Motion LoRA Training Setup",problem:"Train a motion LoRA for smooth camera pan-left movement.",steps:[{formula:"\\text{Collect 50-100 short clips with leftward camera pans}",explanation:"Curated dataset of the specific motion pattern."},{formula:"\\text{Freeze: SD U-Net + base motion module}",explanation:"Only temporal attention LoRA layers are trainable."},{formula:"\\text{Train: rank 64, 2000 steps, lr=1e-4}",explanation:"Short training on temporal layers only."},{formula:"\\text{Output: ~5-15MB LoRA file}",explanation:"Compact motion control that works with any SD 1.5 model."}],id:"example-motion-lora-setup"}),e.jsx(a,{title:"animatediff_lora_usage.py",code:`from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
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
    print(f"  {k}: {v}")`,id:"code-motion-lora"}),e.jsx(t,{type:"tip",title:"Combining Motion and Style LoRAs",content:"AnimateDiff's modularity shines when stacking LoRAs: use a motion LoRA for camera control, a style LoRA for visual appearance, and even a character LoRA for consistent subjects. Adjust individual weights to balance each effect. Start with all weights at 1.0 and reduce if effects conflict.",id:"note-stacking-loras"}),e.jsx(n,{title:"Training Data Quality",content:"Motion LoRA quality depends heavily on training data consistency. Clips should show the same type of motion with minimal variation. Mixing different motions (e.g., panning mixed with zooming) will produce a confused LoRA. Use optical flow analysis to filter and curate motion-consistent clips.",id:"warning-data-quality"})]})}const ye=Object.freeze(Object.defineProperty({__proto__:null,default:P},Symbol.toStringTag,{value:"Module"}));function N(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Testing Fine-Tuned Models in ComfyUI"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"After fine-tuning models with DreamBooth, LoRA, or textual inversion, ComfyUI provides a visual node-based interface for testing and iterating on results. Its workflow system makes it easy to compare different checkpoints, LoRA combinations, and generation parameters without writing code."}),e.jsx(i,{title:"ComfyUI Workflow Testing",definition:"A ComfyUI testing workflow connects model loading nodes (checkpoints, LoRAs, embeddings) with sampling and preview nodes to evaluate fine-tuned models. Workflows can be saved as JSON and shared, ensuring reproducible evaluation across different setups.",id:"def-comfyui-testing"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Loading Custom Models"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"ComfyUI supports loading DreamBooth checkpoints, LoRA files, and textual inversion embeddings through dedicated nodes. Files are placed in specific directories within the ComfyUI installation."}),e.jsx(o,{title:"Model File Placement",problem:"Where do fine-tuned model files go in ComfyUI?",steps:[{formula:"\\text{DreamBooth: models/checkpoints/my\\_model.safetensors}",explanation:"Full model checkpoints in the checkpoints folder."},{formula:"\\text{LoRA: models/loras/my\\_style.safetensors}",explanation:"LoRA adapters in the loras folder."},{formula:"\\text{Textual Inv: models/embeddings/my\\_concept.pt}",explanation:"Embedding files in the embeddings folder."},{formula:"\\text{VAE: models/vae/custom\\_vae.safetensors}",explanation:"Custom VAE files for improved decoding."}],id:"example-file-placement"}),e.jsx(a,{title:"comfyui_testing_workflow.py",code:`# ComfyUI workflow as Python API (JSON equivalent)
# This demonstrates the workflow structure for testing fine-tuned models

def create_lora_test_workflow():
    """Build a ComfyUI-style workflow for LoRA A/B testing."""
    workflow = {
        "nodes": {
            # Load base checkpoint
            "checkpoint_loader": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_v1-5.safetensors"}
            },
            # Load LoRA adapter
            "lora_loader": {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["checkpoint_loader", "MODEL"],
                    "clip": ["checkpoint_loader", "CLIP"],
                    "lora_name": "my_style_lora.safetensors",
                    "strength_model": 0.8,
                    "strength_clip": 0.8,
                }
            },
            # Text prompts
            "positive": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["lora_loader", "CLIP"],
                    "text": "a beautiful landscape, <my-concept> style"
                }
            },
            "negative": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["lora_loader", "CLIP"],
                    "text": "blurry, low quality, distorted"
                }
            },
            # Sampler
            "sampler": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["lora_loader", "MODEL"],
                    "positive": ["positive", "CONDITIONING"],
                    "negative": ["negative", "CONDITIONING"],
                    "latent_image": ["empty_latent", "LATENT"],
                    "seed": 42,
                    "steps": 30,
                    "cfg": 7.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                }
            },
            "empty_latent": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 512, "height": 512, "batch_size": 1}
            },
            "decode": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["sampler", "LATENT"],
                    "vae": ["checkpoint_loader", "VAE"],
                }
            },
            "preview": {
                "class_type": "PreviewImage",
                "inputs": {"images": ["decode", "IMAGE"]}
            },
        }
    }
    return workflow

# A/B testing helper
def create_ab_test(lora_a, lora_b, prompt, seed=42):
    """Create side-by-side comparison workflow."""
    tests = []
    for lora_name, weight in [(lora_a, 0.8), (lora_b, 0.8)]:
        tests.append({
            "lora": lora_name,
            "weight": weight,
            "prompt": prompt,
            "seed": seed,  # Same seed for fair comparison
            "steps": 30,
            "cfg": 7.5,
        })
    return tests

# Example A/B test
tests = create_ab_test(
    "style_v1.safetensors",
    "style_v2.safetensors",
    "portrait of a woman in a garden, masterpiece"
)
for i, t in enumerate(tests):
    print(f"Test {i+1}: {t['lora']} @ weight {t['weight']}")

# LoRA weight sweep
print("\\nLoRA weight sweep:")
for w in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]:
    print(f"  weight={w}: {'undershoot' if w < 0.5 else 'good' if w <= 1.0 else 'overshoot'}")`,id:"code-comfyui-workflow"}),e.jsx(t,{type:"tip",title:"Systematic Testing",content:"Use fixed seeds and identical prompts when comparing models. Vary one parameter at a time: LoRA weight, CFG scale, or sampler. ComfyUI's batch processing lets you generate grids of images across parameter sweeps for visual comparison.",id:"note-systematic-testing"}),e.jsx(n,{title:"ComfyUI Version Compatibility",content:"Custom nodes and workflows may break between ComfyUI updates. Always pin your ComfyUI version during testing campaigns, and document the exact node versions used. Export workflows as JSON for reproducibility.",id:"warning-version-compat"})]})}const be=Object.freeze(Object.defineProperty({__proto__:null,default:N},Symbol.toStringTag,{value:"Module"}));function M(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Introduction to ComfyUI"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"ComfyUI is a powerful node-based visual interface for building and executing Stable Diffusion workflows. Unlike button-based UIs like Automatic1111, ComfyUI exposes the full diffusion pipeline as a graph of connected nodes, giving users complete control over every step of the generation process. This transparency makes it ideal for advanced workflows and experimentation."}),e.jsx(i,{title:"ComfyUI",definition:"ComfyUI is a modular, graph-based UI and backend for diffusion model inference. Each processing step (model loading, text encoding, sampling, VAE decoding) is represented as a node with typed inputs and outputs. Nodes are connected via edges to form a directed acyclic graph (DAG) that defines the generation pipeline.",id:"def-comfyui"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Why ComfyUI?"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"ComfyUI has become the de facto standard for advanced Stable Diffusion workflows because it offers full pipeline transparency, efficient memory management, workflow sharing via JSON, and a growing ecosystem of community custom nodes for every use case."}),e.jsx(o,{title:"ComfyUI vs Other UIs",problem:"Compare ComfyUI with Automatic1111 and Forge.",steps:[{formula:"\\text{A1111: Button-based, beginner-friendly, extension system}",explanation:"Easy to use but limited in complex pipeline customization."},{formula:"\\text{Forge: Optimized A1111 fork, better memory management}",explanation:"Faster than A1111 but still limited by the button-based paradigm."},{formula:"\\text{ComfyUI: Node graph, full control, API-ready, reproducible}",explanation:"Most flexible; workflows are exportable JSON DAGs."}],id:"example-ui-comparison"}),e.jsx(a,{title:"comfyui_setup.py",code:`# ComfyUI installation and setup
import os

# Installation steps
setup_commands = """
# 1. Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\\Scripts\\activate   # Windows

# 3. Install PyTorch (CUDA 12.1 example)
pip install torch torchvision torchaudio --index-url \\
    https://download.pytorch.org/whl/cu121

# 4. Install ComfyUI dependencies
pip install -r requirements.txt

# 5. Download models to correct directories
# models/checkpoints/  - SD, SDXL checkpoints
# models/loras/        - LoRA adapters
# models/vae/          - VAE models
# models/embeddings/   - Textual inversions
# models/controlnet/   - ControlNet models
# models/clip/         - CLIP models (for FLUX)

# 6. Run ComfyUI
python main.py --listen 0.0.0.0 --port 8188
"""
print(setup_commands)

# Directory structure
dirs = {
    "models/checkpoints":   "Base model files (.safetensors)",
    "models/loras":         "LoRA adapter files",
    "models/vae":           "VAE model files",
    "models/embeddings":    "Textual inversion embeddings",
    "models/controlnet":    "ControlNet model files",
    "models/clip":          "CLIP text encoder models",
    "models/unet":          "U-Net / DiT model files (FLUX)",
    "custom_nodes":         "Community node packages",
    "input":                "Input images for img2img, etc.",
    "output":               "Generated images output",
}

print("\\nComfyUI Directory Structure:")
for path, desc in dirs.items():
    print(f"  {path:30s} - {desc}")`,id:"code-comfyui-setup"}),e.jsx(t,{type:"tip",title:"Getting Started",content:"Start with the default workflow (drag an image onto the canvas to load its workflow metadata). The basic txt2img workflow has just 6 nodes: CheckpointLoader, CLIPTextEncode (positive/negative), EmptyLatentImage, KSampler, and VAEDecode. Understanding these 6 nodes gives you the foundation for everything else.",id:"note-getting-started"}),e.jsx(n,{title:"GPU Memory Management",content:"ComfyUI keeps models in VRAM between runs for speed. If you run out of memory, use the 'Free model memory' option in the queue menu, or add --lowvram or --novram flags when starting ComfyUI. For FLUX models, --lowvram is usually required on consumer GPUs.",id:"warning-memory"})]})}const we=Object.freeze(Object.defineProperty({__proto__:null,default:M},Symbol.toStringTag,{value:"Module"}));function z(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"ComfyUI Core Concepts"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Understanding ComfyUI's core concepts -- nodes, slots, links, and data types -- is essential for building effective workflows. Every generation pipeline is a directed acyclic graph (DAG) where data flows from loader nodes through processing nodes to output nodes. Each connection carries a specific data type that must match between source and destination."}),e.jsx(i,{title:"ComfyUI Data Types",definition:"ComfyUI uses typed connections between nodes. The primary types are: MODEL (diffusion model weights), CLIP (text encoder), VAE (autoencoder), CONDITIONING (encoded text embeddings), LATENT (latent space tensors), and IMAGE (decoded pixel tensors). Connections are color-coded: purple for MODEL, yellow for CLIP, red for VAE, orange for CONDITIONING, pink for LATENT, green for IMAGE.",id:"def-data-types"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Node Anatomy"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Each node has input slots (left side), output slots (right side), and widget parameters (internal controls like dropdowns, sliders, text fields). Inputs can be either connected from another node or set via widgets."}),e.jsx(o,{title:"KSampler Node Inputs",problem:"What inputs does the KSampler node require?",steps:[{formula:"\\text{model: MODEL} \\leftarrow \\text{CheckpointLoader or LoRA output}",explanation:"The diffusion model to use for denoising."},{formula:"\\text{positive/negative: CONDITIONING} \\leftarrow \\text{CLIPTextEncode}",explanation:"Encoded text prompts for guidance."},{formula:"\\text{latent\\_image: LATENT} \\leftarrow \\text{EmptyLatent or VAEEncode}",explanation:"The starting noise or encoded image to denoise."},{formula:"\\text{Widgets: seed, steps, cfg, sampler, scheduler, denoise}",explanation:"Parameters controlling the sampling process."}],id:"example-ksampler"}),e.jsx(a,{title:"comfyui_workflow_structure.py",code:`# ComfyUI workflow as JSON (the format used for saving/sharing)
import json

# Minimal txt2img workflow
workflow = {
    "3": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_v1-5.safetensors"},
        "_meta": {"title": "Load Checkpoint"}
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "beautiful sunset over mountains, masterpiece",
            "clip": ["3", 1]  # [node_id, output_index]
        },
        "_meta": {"title": "Positive Prompt"}
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "ugly, blurry, low quality",
            "clip": ["3", 1]
        },
        "_meta": {"title": "Negative Prompt"}
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
        "_meta": {"title": "Empty Latent"}
    },
    "8": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["3", 0],          # MODEL output
            "positive": ["6", 0],        # CONDITIONING
            "negative": ["7", 0],        # CONDITIONING
            "latent_image": ["5", 0],    # LATENT
            "seed": 42,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
        },
        "_meta": {"title": "KSampler"}
    },
    "9": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["8", 0],   # LATENT from KSampler
            "vae": ["3", 2],       # VAE from checkpoint
        },
        "_meta": {"title": "VAE Decode"}
    },
    "10": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["9", 0],    # IMAGE from decode
            "filename_prefix": "ComfyUI_output"
        },
        "_meta": {"title": "Save Image"}
    }
}

# Print the workflow DAG
print("Workflow DAG:")
for node_id, node in sorted(workflow.items()):
    title = node.get("_meta", {}).get("title", node["class_type"])
    connections = []
    for key, val in node["inputs"].items():
        if isinstance(val, list):
            src_id, src_slot = val
            src_title = workflow[str(src_id)].get("_meta", {}).get("title", "?")
            connections.append(f"{key} <- {src_title}[{src_slot}]")
    conn_str = ", ".join(connections) if connections else "(no connections)"
    print(f"  [{node_id}] {title}: {conn_str}")

# Save workflow
print(f"\\nWorkflow JSON size: {len(json.dumps(workflow))} bytes")`,id:"code-workflow-structure"}),e.jsx(t,{type:"note",title:"Checkpoint Outputs",content:"The CheckpointLoaderSimple node outputs three things: MODEL (index 0), CLIP (index 1), and VAE (index 2). These three outputs feed the rest of the workflow. Understanding this triple output is key to understanding all ComfyUI workflows.",id:"note-checkpoint-outputs"}),e.jsx(n,{title:"Type Mismatches",content:"Connecting incompatible types (e.g., IMAGE to a LATENT input) will cause errors. ComfyUI color-codes connections to help prevent this, but the error messages can be cryptic. If a workflow fails silently, check that all connections have matching types.",id:"warning-type-mismatch"})]})}const ve=Object.freeze(Object.defineProperty({__proto__:null,default:z},Symbol.toStringTag,{value:"Module"}));function E(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Text-to-Image Workflows in ComfyUI"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The text-to-image (txt2img) workflow is the foundational ComfyUI pipeline. Starting from a text prompt and random noise, the model iteratively denoises a latent tensor to produce an image. Understanding this basic workflow is essential before building more complex pipelines with ControlNet, LoRA, or multi-pass refinement."}),e.jsx(i,{title:"Classifier-Free Guidance (CFG)",definition:"CFG steers generation toward the text prompt by combining conditioned and unconditioned noise predictions: $\\hat{\\epsilon} = \\epsilon_{\\text{uncond}} + s \\cdot (\\epsilon_{\\text{cond}} - \\epsilon_{\\text{uncond}})$ where $s$ is the guidance scale. Higher $s$ increases prompt adherence but can reduce diversity and cause artifacts.",id:"def-cfg"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Sampler and Scheduler Selection"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"ComfyUI separates the sampling algorithm (how noise is removed per step) from the scheduler (how noise levels are distributed across steps). Common samplers include Euler, DPM++ 2M, and UniPC. Common schedulers include normal, karras, and exponential."}),e.jsx(o,{title:"Sampler Comparison",problem:"Which sampler to use for different scenarios?",steps:[{formula:"\\text{euler: Fast, good at 20-30 steps, general purpose}",explanation:"Default choice for quick generation."},{formula:"\\text{dpmpp\\_2m + karras: Best quality, 20-30 steps}",explanation:"DPM++ 2M with Karras scheduler produces sharp, detailed results."},{formula:"\\text{euler\\_ancestral: More creative/varied, slightly noisy}",explanation:"Adds noise during sampling for more diverse outputs."},{formula:"\\text{uni\\_pc: Fast convergence, good at 10-15 steps}",explanation:"Best for quick previews or when speed matters most."}],id:"example-samplers"}),e.jsx(a,{title:"txt2img_workflow_api.py",code:`# Complete txt2img workflow with parameter exploration
import json

def build_txt2img_workflow(
    checkpoint="sd_v1-5.safetensors",
    positive_prompt="a serene lake surrounded by mountains at golden hour, photorealistic",
    negative_prompt="blurry, low quality, distorted, watermark",
    width=512, height=512,
    steps=25, cfg=7.0,
    sampler="dpmpp_2m", scheduler="karras",
    seed=42, batch_size=1
):
    """Build a complete txt2img ComfyUI workflow."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint}
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": ["1", 1]}
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["1", 1]}
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": batch_size}
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["2", 0],
                "negative": ["3", 0], "latent_image": ["4", 0],
                "seed": seed, "steps": steps, "cfg": cfg,
                "sampler_name": sampler, "scheduler": scheduler,
                "denoise": 1.0,
            }
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": "txt2img"}
        },
    }

# Generate workflow
workflow = build_txt2img_workflow()
print(f"Workflow nodes: {len(workflow)}")

# Parameter sweep for CFG scale
print("\\nCFG Scale Guide:")
cfg_guide = {
    1.0: "No guidance (random, ignores prompt)",
    3.0: "Subtle guidance, creative and loose",
    5.0: "Moderate, good balance for artistic styles",
    7.0: "Standard, good prompt adherence",
    10.0: "Strong, very literal interpretation",
    15.0: "Very strong, may cause saturation/artifacts",
    20.0: "Extreme, usually produces artifacts",
}
for cfg, desc in cfg_guide.items():
    print(f"  CFG {cfg:5.1f}: {desc}")

# Resolution guide for different models
print("\\nResolution Guide:")
resolutions = {
    "SD 1.5":  [(512, 512), (512, 768), (768, 512)],
    "SDXL":    [(1024, 1024), (896, 1152), (1152, 896)],
    "FLUX":    [(1024, 1024), (768, 1344), (1344, 768)],
}
for model, res_list in resolutions.items():
    dims = ", ".join(f"{w}x{h}" for w, h in res_list)
    print(f"  {model}: {dims}")`,id:"code-txt2img"}),e.jsx(t,{type:"tip",title:"Prompt Engineering Tips",content:"Place the most important elements at the beginning of the prompt. Use commas to separate concepts. Quality modifiers like 'masterpiece, best quality, detailed' help with many models. SDXL and FLUX respond better to natural language descriptions than keyword-style prompts.",id:"note-prompt-tips"}),e.jsx(n,{title:"Resolution Restrictions",content:"Each model has a native training resolution. Generating far outside this range produces poor results. SD 1.5 works best at 512px, SDXL at 1024px. For other resolutions, generate at native and then upscale. Dimensions should be multiples of 8 (or 64 for some models).",id:"warning-resolution"})]})}const ke=Object.freeze(Object.defineProperty({__proto__:null,default:E},Symbol.toStringTag,{value:"Module"}));function V(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Image-to-Image Workflows"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Image-to-image (img2img) generation starts from an existing image rather than pure noise. The image is encoded to latent space, noise is added to a specified level (controlled by the denoise parameter), and then the model denoises it guided by a text prompt. This enables style transfer, image editing, sketch-to-photo conversion, and iterative refinement of generated images."}),e.jsx(i,{title:"Denoise Strength",definition:"The denoise parameter $d \\in [0, 1]$ controls how much of the original image is preserved. At $d = 0$, the output is the original image. At $d = 1.0$, the image is fully replaced with noise (equivalent to txt2img). In latent space: $\\mathbf{z}_{\\text{start}} = \\sqrt{\\bar{\\alpha}_{t_d}} \\cdot \\mathbf{z}_0 + \\sqrt{1 - \\bar{\\alpha}_{t_d}} \\cdot \\boldsymbol{\\epsilon}$ where $t_d = \\lfloor d \\cdot T \\rfloor$.",id:"def-denoise"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"ComfyUI Img2Img Pipeline"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"In ComfyUI, img2img replaces the EmptyLatentImage node with a LoadImage + VAEEncode chain. The KSampler's denoise parameter is set below 1.0 to preserve the input image structure while applying the style/content from the prompt."}),e.jsx(o,{title:"Denoise Strength Effects",problem:"How does denoise strength affect the output?",steps:[{formula:"d = 0.2: \\text{Subtle changes, color correction, minor style}",explanation:"Mostly preserves the original, good for refinement."},{formula:"d = 0.5: \\text{Balanced, keeps composition but changes style}",explanation:"Good starting point for style transfer."},{formula:"d = 0.7: \\text{Major changes, only broad structure preserved}",explanation:"Good for converting sketches to detailed images."},{formula:"d = 1.0: \\text{Complete regeneration (same as txt2img)}",explanation:"Original image has no influence; only dimensions matter."}],id:"example-denoise"}),e.jsx(a,{title:"img2img_workflow.py",code:`# ComfyUI img2img workflow
def build_img2img_workflow(
    checkpoint="sd_v1-5.safetensors",
    image_path="input/photo.png",
    prompt="oil painting style, masterpiece",
    negative="ugly, blurry",
    denoise=0.6,
    steps=30, cfg=7.0,
    sampler="dpmpp_2m", scheduler="karras",
    seed=42
):
    """Build img2img workflow for ComfyUI."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint}
        },
        # Load input image instead of empty latent
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": image_path}
        },
        # Encode image to latent space
        "3": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["2", 0],  # IMAGE from LoadImage
                "vae": ["1", 2],     # VAE from checkpoint
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["1", 1]}
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 1]}
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["3", 0],  # Encoded input image
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": denoise,  # Key difference from txt2img
            }
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["6", 0], "vae": ["1", 2]}
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {"images": ["7", 0], "filename_prefix": "img2img"}
        },
    }

# Build workflow
workflow = build_img2img_workflow(denoise=0.6)
print(f"Img2Img workflow: {len(workflow)} nodes")

# Common img2img use cases
use_cases = {
    "Style Transfer":     {"denoise": 0.5, "desc": "Apply art style to photo"},
    "Sketch to Photo":    {"denoise": 0.75, "desc": "Convert rough sketch to detailed image"},
    "Color Correction":   {"denoise": 0.2, "desc": "Adjust colors/lighting"},
    "Inpainting Prep":    {"denoise": 0.4, "desc": "Refine specific areas"},
    "Upscale Refinement": {"denoise": 0.3, "desc": "Add detail to upscaled image"},
}

print("\\nImg2Img Use Cases:")
for name, info in use_cases.items():
    print(f"  {name}: denoise={info['denoise']} - {info['desc']}")`,id:"code-img2img"}),e.jsx(t,{type:"tip",title:"Iterative Refinement",content:"Chain multiple img2img passes with decreasing denoise strength (0.7 -> 0.5 -> 0.3) for progressive refinement. Each pass adds more detail while preserving the structure established in previous passes. This 'multi-pass' approach often produces better results than a single high-denoise pass.",id:"note-iterative"}),e.jsx(n,{title:"Resolution Matching",content:"The input image should match the model's native resolution or be a multiple of 64. If the input image has a different resolution, resize it before VAE encoding. Mismatched resolutions can cause the VAE to produce artifacts, especially at the image borders.",id:"warning-resolution-match"})]})}const Le=Object.freeze(Object.defineProperty({__proto__:null,default:V},Symbol.toStringTag,{value:"Module"}));function R(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"ControlNet in ComfyUI"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"ControlNet adds spatial conditioning to diffusion models through edge maps, depth maps, pose skeletons, segmentation maps, and other structural guides. It enables precise control over the generated image's composition while the text prompt controls style and content. In ComfyUI, ControlNet is applied via dedicated nodes that inject conditioning into the U-Net's encoder."}),e.jsx(i,{title:"ControlNet",definition:"ControlNet creates a trainable copy of the U-Net encoder blocks that processes a conditioning image $\\mathbf{c}_{\\text{spatial}}$ (e.g., Canny edges). The copy's outputs are added to the main U-Net's skip connections with a learned zero-initialized scaling: $\\mathbf{y}_i = \\mathbf{F}_i(\\mathbf{x}) + \\alpha_i \\cdot \\mathbf{G}_i(\\mathbf{c}_{\\text{spatial}})$ where $\\alpha_i$ starts at zero.",id:"def-controlnet"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"ControlNet Types"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Different ControlNet models accept different types of conditioning images, each controlling different aspects of the output."}),e.jsx(o,{title:"ControlNet Conditioning Types",problem:"What spatial conditions can ControlNet use?",steps:[{formula:"\\text{Canny: Edge detection} \\to \\text{controls object boundaries}",explanation:"Sharp edges from input image guide structural outlines."},{formula:"\\text{Depth: Monocular depth map} \\to \\text{controls spatial layout}",explanation:"MiDaS or Depth Anything depth estimation for 3D structure."},{formula:"\\text{OpenPose: Skeleton keypoints} \\to \\text{controls human pose}",explanation:"Stick figure skeletons guide body positions."},{formula:"\\text{Lineart/Scribble: Line drawing} \\to \\text{sketch to image}",explanation:"Convert rough drawings to detailed rendered images."}],id:"example-controlnet-types"}),e.jsx(a,{title:"controlnet_workflow.py",code:`# ControlNet workflow in ComfyUI
def build_controlnet_workflow(
    checkpoint="sd_v1-5.safetensors",
    controlnet_model="control_v11p_sd15_canny.safetensors",
    control_image="input/canny_edges.png",
    prompt="beautiful landscape painting, oil on canvas",
    negative="ugly, blurry",
    strength=1.0,
    steps=30, cfg=7.0, seed=42,
):
    """ControlNet txt2img workflow."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint}
        },
        # Load ControlNet model
        "2": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": controlnet_model}
        },
        # Load conditioning image
        "3": {
            "class_type": "LoadImage",
            "inputs": {"image": control_image}
        },
        # Text encoding
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["1", 1]}
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 1]}
        },
        # Apply ControlNet to positive conditioning
        "6": {
            "class_type": "ControlNetApply",
            "inputs": {
                "conditioning": ["4", 0],       # Positive CONDITIONING
                "control_net": ["2", 0],         # CONTROL_NET model
                "image": ["3", 0],               # Control IMAGE
                "strength": strength,            # How strongly to apply
            }
        },
        # Standard sampling
        "7": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1}
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["6", 0],   # ControlNet-enhanced conditioning
                "negative": ["5", 0],
                "latent_image": ["7", 0],
                "seed": seed, "steps": steps, "cfg": cfg,
                "sampler_name": "dpmpp_2m", "scheduler": "karras",
                "denoise": 1.0,
            }
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["1", 2]}
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {"images": ["9", 0], "filename_prefix": "controlnet"}
        },
    }

workflow = build_controlnet_workflow()
print(f"ControlNet workflow: {len(workflow)} nodes")

# ControlNet preprocessor nodes (from ControlNet Aux)
preprocessors = {
    "CannyEdgePreprocessor":    {"low": 100, "high": 200},
    "DepthAnythingPreprocessor": {"model": "depth_anything_vitl14"},
    "OpenposePreprocessor":     {"detect_body": True, "detect_hand": True},
    "LineartPreprocessor":      {"coarse": False},
    "ScribblePreprocessor":     {},
    "NormalMapPreprocessor":    {"bg_threshold": 0.4},
    "SegmentAnythingPreprocessor": {"model": "sam_vit_h"},
}

print("\\nAvailable preprocessors:")
for name, params in preprocessors.items():
    print(f"  {name}: {params}")`,id:"code-controlnet"}),e.jsx(t,{type:"tip",title:"Multi-ControlNet",content:"You can chain multiple ControlNet nodes: apply Canny for edges AND depth for 3D layout AND pose for characters. Each ControlNetApply takes the previous conditioning as input. Reduce individual strengths (0.5-0.7 each) when stacking to avoid over-constraining the generation.",id:"note-multi-controlnet"}),e.jsx(n,{title:"ControlNet Model Matching",content:"ControlNet models must match the base model version: SD 1.5 ControlNets only work with SD 1.5 checkpoints. SDXL has its own ControlNet models. Using mismatched versions produces garbage output without any error message. Always verify model compatibility.",id:"warning-model-match"})]})}const je=Object.freeze(Object.defineProperty({__proto__:null,default:R},Symbol.toStringTag,{value:"Module"}));function U(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"LoRA Stacking in ComfyUI"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"ComfyUI allows chaining multiple LoRA nodes to combine different fine-tuned effects. A style LoRA for art direction, a character LoRA for consistent faces, and a detail LoRA for texture quality can all be applied simultaneously. Understanding how to balance LoRA weights and resolve conflicts is key to effective multi-LoRA workflows."}),e.jsx(i,{title:"LoRA Stacking",definition:"LoRA stacking applies multiple low-rank updates sequentially to the model weights. Given base weights $\\mathbf{W}_0$ and $K$ LoRAs with weights $\\alpha_k$, the effective weight is: $\\mathbf{W}_{\\text{eff}} = \\mathbf{W}_0 + \\sum_{k=1}^{K} \\alpha_k \\cdot \\mathbf{B}_k \\mathbf{A}_k$. In ComfyUI, each LoraLoader node takes MODEL and CLIP inputs and outputs modified MODEL and CLIP.",id:"def-lora-stacking"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Chaining LoRA Nodes"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"In ComfyUI, LoRAs are stacked by chaining LoraLoader nodes: the output MODEL and CLIP of one LoraLoader feed into the input of the next. Each node applies its LoRA on top of the previous result."}),e.jsx(o,{title:"Multi-LoRA Setup",problem:"Stack a style LoRA, character LoRA, and detail LoRA.",steps:[{formula:"\\text{Checkpoint} \\to \\text{LoRA\\_style (0.7)} \\to \\text{LoRA\\_char (0.8)} \\to \\text{LoRA\\_detail (0.5)}",explanation:"Each LoRA node passes its modified MODEL/CLIP to the next."},{formula:"\\mathbf{W} = \\mathbf{W}_0 + 0.7\\mathbf{B}_1\\mathbf{A}_1 + 0.8\\mathbf{B}_2\\mathbf{A}_2 + 0.5\\mathbf{B}_3\\mathbf{A}_3",explanation:"Effective weight is base plus all weighted LoRA contributions."},{formula:"\\text{Total LoRA effect} \\leq 2.0 \\text{ recommended}",explanation:"Sum of weights above 2.0 often causes artifacts or instability."}],id:"example-multi-lora"}),e.jsx(a,{title:"lora_stacking_workflow.py",code:`# Multi-LoRA workflow for ComfyUI
def build_lora_stack_workflow(
    checkpoint="sd_v1-5.safetensors",
    loras=[
        ("anime_style.safetensors", 0.7, 0.7),    # (file, model_strength, clip_strength)
        ("character_face.safetensors", 0.8, 0.6),
        ("detail_enhancer.safetensors", 0.5, 0.3),
    ],
    prompt="1girl, walking in garden, cherry blossoms, masterpiece",
    negative="ugly, blurry, deformed",
    steps=30, cfg=7.0, seed=42,
):
    """Build LoRA stacking workflow."""
    nodes = {}
    node_id = 1

    # Checkpoint loader
    nodes[str(node_id)] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": checkpoint},
        "_meta": {"title": "Base Checkpoint"}
    }
    model_src = [str(node_id), 0]  # MODEL output
    clip_src = [str(node_id), 1]   # CLIP output
    vae_src = [str(node_id), 2]    # VAE output
    node_id += 1

    # Chain LoRA loaders
    for lora_file, model_str, clip_str in loras:
        nodes[str(node_id)] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_src,
                "clip": clip_src,
                "lora_name": lora_file,
                "strength_model": model_str,
                "strength_clip": clip_str,
            },
            "_meta": {"title": f"LoRA: {lora_file}"}
        }
        model_src = [str(node_id), 0]  # Updated MODEL
        clip_src = [str(node_id), 1]   # Updated CLIP
        node_id += 1

    # Rest of the pipeline uses final model_src and clip_src
    nodes[str(node_id)] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": prompt, "clip": clip_src}
    }
    pos_id = str(node_id)
    node_id += 1

    nodes[str(node_id)] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": negative, "clip": clip_src}
    }
    neg_id = str(node_id)
    node_id += 1

    nodes[str(node_id)] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 768, "batch_size": 1}
    }
    latent_id = str(node_id)
    node_id += 1

    nodes[str(node_id)] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_src, "positive": [pos_id, 0],
            "negative": [neg_id, 0], "latent_image": [latent_id, 0],
            "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "dpmpp_2m", "scheduler": "karras",
            "denoise": 1.0,
        }
    }
    sample_id = str(node_id)
    node_id += 1

    nodes[str(node_id)] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [sample_id, 0], "vae": vae_src}
    }
    decode_id = str(node_id)
    node_id += 1

    nodes[str(node_id)] = {
        "class_type": "SaveImage",
        "inputs": {"images": [decode_id, 0], "filename_prefix": "lora_stack"}
    }

    return nodes

workflow = build_lora_stack_workflow()
print(f"LoRA stack workflow: {len(workflow)} nodes")

# Print chain
print("\\nLoRA chain:")
for nid, node in sorted(workflow.items(), key=lambda x: int(x[0])):
    title = node.get("_meta", {}).get("title", node["class_type"])
    print(f"  [{nid}] {title}")`,id:"code-lora-stacking"}),e.jsx(t,{type:"tip",title:"Weight Balancing",content:"When LoRAs conflict (e.g., two style LoRAs), reduce both weights proportionally. A good starting approach: dominant LoRA at 0.7-0.9, supplementary LoRAs at 0.3-0.5. Model strength and CLIP strength can be set independently -- higher CLIP strength means stronger prompt influence from that LoRA's training.",id:"note-weight-balance"}),e.jsx(n,{title:"LoRA Conflicts",content:"Stacking too many LoRAs or using total weights above 2.0 often produces muddy, oversaturated, or artifact-prone images. If results degrade, reduce weights or remove conflicting LoRAs. Two LoRAs that modify the same concept (e.g., two face LoRAs) will typically interfere.",id:"warning-conflicts"})]})}const Ce=Object.freeze(Object.defineProperty({__proto__:null,default:U},Symbol.toStringTag,{value:"Module"}));function q(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"SDXL and FLUX Workflows in ComfyUI"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"SDXL and FLUX require different ComfyUI workflows than SD 1.5 due to their distinct architectures. SDXL uses dual text encoders and an optional refiner model. FLUX uses separate CLIP and T5 text encoders with a completely different model loading approach. Understanding these differences is essential for building effective workflows."}),e.jsx(i,{title:"SDXL Dual Encoding",definition:"SDXL uses two CLIP text encoders: CLIP-L (from SD 1.5) and OpenCLIP-G (larger, more capable). Both encoders' outputs are concatenated to form the conditioning tensor. In ComfyUI, the CLIPTextEncode node automatically handles both encoders when an SDXL checkpoint is loaded.",id:"def-sdxl-dual"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"SDXL Base + Refiner"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"SDXL optionally uses a two-stage pipeline: the base model generates for the first 80% of steps, then the refiner model takes over for the final 20% to add fine details. In ComfyUI, this requires two KSampler nodes with coordinated denoise ranges."}),e.jsx(o,{title:"FLUX Model Loading",problem:"How does FLUX model loading differ from SD/SDXL in ComfyUI?",steps:[{formula:"\\text{SD/SDXL: Single checkpoint contains MODEL + CLIP + VAE}",explanation:"CheckpointLoaderSimple outputs all three from one file."},{formula:"\\text{FLUX: Separate UNET, CLIP, T5, and VAE files}",explanation:"Each component loaded by its own node."},{formula:"\\text{FLUX uses DualCLIPLoader for CLIP-L + T5-XXL}",explanation:"Two text encoders loaded together into a combined CLIP output."}],id:"example-flux-loading"}),e.jsx(a,{title:"sdxl_flux_workflows.py",code:`# SDXL Base + Refiner workflow
def build_sdxl_refiner_workflow(
    base_ckpt="sd_xl_base_1.0.safetensors",
    refiner_ckpt="sd_xl_refiner_1.0.safetensors",
    prompt="A majestic eagle soaring over snow-capped mountains",
    negative="blurry, low quality",
    total_steps=30, switch_at=0.8,
    seed=42, cfg=7.0,
):
    """SDXL with base + refiner handoff."""
    base_steps = int(total_steps * switch_at)
    return {
        # Load base and refiner models
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": base_ckpt}},
        "2": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": refiner_ckpt}},
        # Text encoding (using base CLIP)
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["1", 1]}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["1", 1]}},
        # Refiner text encoding
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["2", 1]}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["2", 1]}},
        # Empty latent at SDXL resolution
        "7": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        # Base sampler (first 80% of steps)
        "8": {"class_type": "KSampler",
              "inputs": {
                  "model": ["1", 0], "positive": ["3", 0],
                  "negative": ["4", 0], "latent_image": ["7", 0],
                  "seed": seed, "steps": total_steps, "cfg": cfg,
                  "sampler_name": "dpmpp_2m", "scheduler": "karras",
                  "denoise": 1.0,
                  "start_at_step": 0, "end_at_step": base_steps,
              }},
        # Refiner sampler (final 20%)
        "9": {"class_type": "KSampler",
              "inputs": {
                  "model": ["2", 0], "positive": ["5", 0],
                  "negative": ["6", 0], "latent_image": ["8", 0],
                  "seed": seed, "steps": total_steps, "cfg": cfg,
                  "sampler_name": "dpmpp_2m", "scheduler": "karras",
                  "denoise": 1.0,
                  "start_at_step": base_steps, "end_at_step": total_steps,
              }},
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0], "filename_prefix": "sdxl_refiner"}},
    }

# FLUX workflow (separate model components)
def build_flux_workflow(
    unet="flux1-dev.safetensors",
    clip_l="clip_l.safetensors",
    t5xxl="t5xxl_fp16.safetensors",
    vae="ae.safetensors",
    prompt="A cat sitting on a windowsill watching rain",
    steps=28, guidance=3.5, seed=42,
):
    """FLUX.1-dev workflow with separate model loading."""
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet, "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": clip_l, "clip_name2": t5xxl,
                         "type": "flux"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": vae}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "6": {"class_type": "KSampler",
              "inputs": {
                  "model": ["1", 0], "positive": ["4", 0],
                  "negative": ["4", 0],  # FLUX: same or empty
                  "latent_image": ["5", 0],
                  "seed": seed, "steps": steps, "cfg": guidance,
                  "sampler_name": "euler", "scheduler": "simple",
                  "denoise": 1.0,
              }},
        "7": {"class_type": "VAEDecode",
              "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0], "filename_prefix": "flux"}},
    }

print("SDXL workflow:", len(build_sdxl_refiner_workflow()), "nodes")
print("FLUX workflow:", len(build_flux_workflow()), "nodes")`,id:"code-sdxl-flux"}),e.jsx(t,{type:"note",title:"FLUX Negative Prompts",content:"FLUX.1-dev was trained with guidance distillation, which means it does not use traditional classifier-free guidance with negative prompts. The guidance_scale parameter works differently -- it controls the distilled guidance. Setting it too high (>5) can cause artifacts. Typical range is 2.5-4.0.",id:"note-flux-cfg"}),e.jsx(n,{title:"FLUX Memory Requirements",content:"FLUX requires loading UNET (~12B params), T5-XXL (~4.7B), CLIP-L, and VAE separately. Total VRAM needed: ~24GB in fp16, ~12GB in fp8. Use fp8_e4m3fn weight dtype for the UNET in ComfyUI to fit on 16GB GPUs. T5 can be loaded in fp16 or fp8.",id:"warning-flux-memory"})]})}const Ae=Object.freeze(Object.defineProperty({__proto__:null,default:q},Symbol.toStringTag,{value:"Module"}));function B(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Video Generation Nodes in ComfyUI"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"ComfyUI supports video generation through AnimateDiff nodes, SVD (Stable Video Diffusion) nodes, and CogVideoX integration. Video workflows extend image workflows by adding temporal dimensions to the latent space, requiring specialized loaders, samplers, and output nodes for handling frame sequences."}),e.jsx(i,{title:"Video Workflow in ComfyUI",definition:"A video workflow in ComfyUI generates a batch of temporally coherent frames by processing a 3D latent tensor (frames x height x width x channels). The AnimateDiff approach injects motion modules into the standard SD pipeline, while SVD uses a dedicated image-to-video model. Output nodes combine frames into GIF, MP4, or image sequences.",id:"def-video-workflow"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"AnimateDiff Nodes"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The AnimateDiff custom node pack adds motion module loading, frame control, and video output nodes. It integrates with the standard SD 1.5 pipeline, adding temporal attention to the existing U-Net."}),e.jsx(o,{title:"AnimateDiff Node Chain",problem:"Build an AnimateDiff video workflow in ComfyUI.",steps:[{formula:"\\text{CheckpointLoader} \\to \\text{AnimateDiffLoader (motion module)}",explanation:"Load base SD model, then inject motion module."},{formula:"\\text{CLIPTextEncode} \\to \\text{KSampler (batch\\_size=16)}",explanation:"Standard prompt encoding, but latent batch = number of frames."},{formula:"\\text{VAEDecode} \\to \\text{AnimateDiffCombine (output)}",explanation:"Decode all frames, combine into video file."}],id:"example-animatediff-nodes"}),e.jsx(a,{title:"video_workflow.py",code:`# AnimateDiff video workflow for ComfyUI
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
print(f"SVD workflow: {len(build_svd_workflow())} nodes")`,id:"code-video-workflow"}),e.jsx(t,{type:"tip",title:"Sliding Window for Longer Videos",content:"AnimateDiff's context_length limits frames processed simultaneously (typically 16). For longer videos, use sliding window context (context_overlap=4) to generate 32+ frames by processing overlapping windows and blending them. This enables longer videos without excessive VRAM.",id:"note-sliding-window"}),e.jsx(n,{title:"Video VRAM Usage",content:"Video generation requires significantly more VRAM than single images. AnimateDiff with 16 frames at 512x512 needs ~8GB. SVD at 1024x576 with 25 frames needs ~12-16GB. Always enable VAE tiling and consider reducing resolution for initial tests.",id:"warning-video-vram"})]})}const Te=Object.freeze(Object.defineProperty({__proto__:null,default:B},Symbol.toStringTag,{value:"Module"}));function F(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Upscaling Workflows in ComfyUI"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Upscaling takes a generated or input image and increases its resolution while adding detail. ComfyUI supports two main approaches: model-based upscaling (ESRGAN, RealESRGAN) for direct super-resolution, and latent upscaling with img2img re-diffusion for adding AI-generated detail. The best results combine both in a multi-pass workflow."}),e.jsx(i,{title:"Upscaling Approaches",definition:"Model-based upscaling uses dedicated super-resolution neural networks (e.g., RealESRGAN 4x) that directly map low-res to high-res pixels. Latent upscaling encodes the upscaled image to latent space and runs a partial diffusion pass (denoise 0.3-0.5) to add AI-generated fine details that pure SR models cannot hallucinate.",id:"def-upscaling"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Two-Pass Upscaling"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The recommended approach: (1) upscale with an ESRGAN model for clean resolution increase, (2) run img2img with low denoise to refine details. This avoids the blurriness of pure SR while preventing the hallucination artifacts of pure re-diffusion."}),e.jsx(o,{title:"Upscaling Pipeline",problem:"Upscale a 512x512 image to 2048x2048 with detail enhancement.",steps:[{formula:"\\text{Step 1: ESRGAN 4x: } 512 \\to 2048 \\text{ pixels}",explanation:"Clean upscale preserving structure."},{formula:"\\text{Step 2: VAEEncode(2048x2048)} \\to \\text{latent 256x256}",explanation:"Encode upscaled image to latent space."},{formula:"\\text{Step 3: KSampler(denoise=0.35)} \\to \\text{refined latent}",explanation:"Add fine detail via partial re-diffusion."},{formula:"\\text{Step 4: VAEDecode} \\to \\text{2048x2048 detailed output}",explanation:"Final high-resolution image with enhanced detail."}],id:"example-upscale-pipeline"}),e.jsx(a,{title:"upscaling_workflow.py",code:`# Two-pass upscaling workflow for ComfyUI
def build_upscale_workflow(
    checkpoint="sd_v1-5.safetensors",
    upscale_model="RealESRGAN_x4plus.pth",
    input_image="input/generated.png",
    prompt="highly detailed, sharp focus, 8k uhd",
    negative="blurry, low quality, pixelated",
    denoise=0.35, steps=20, cfg=7.0, seed=42,
):
    """Two-pass upscaling: ESRGAN + img2img refinement."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": checkpoint}},
        # Load upscale model
        "2": {"class_type": "UpscaleModelLoader",
              "inputs": {"model_name": upscale_model}},
        # Load input image
        "3": {"class_type": "LoadImage",
              "inputs": {"image": input_image}},
        # Pass 1: ESRGAN upscale (pixel-space)
        "4": {"class_type": "ImageUpscaleWithModel",
              "inputs": {
                  "upscale_model": ["2", 0],
                  "image": ["3", 0],
              }},
        # Optionally resize to exact target (ESRGAN is fixed 4x)
        "5": {"class_type": "ImageScale",
              "inputs": {
                  "image": ["4", 0],
                  "upscale_method": "lanczos",
                  "width": 2048, "height": 2048,
                  "crop": "center",
              }},
        # Pass 2: Encode to latent for img2img refinement
        "6": {"class_type": "VAEEncode",
              "inputs": {"pixels": ["5", 0], "vae": ["1", 2]}},
        # Text encoding
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["1", 1]}},
        "8": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["1", 1]}},
        # Refinement sampling (low denoise)
        "9": {"class_type": "KSampler",
              "inputs": {
                  "model": ["1", 0],
                  "positive": ["7", 0], "negative": ["8", 0],
                  "latent_image": ["6", 0],
                  "seed": seed, "steps": steps, "cfg": cfg,
                  "sampler_name": "dpmpp_2m", "scheduler": "karras",
                  "denoise": denoise,
              }},
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0], "filename_prefix": "upscaled"}},
    }

workflow = build_upscale_workflow()
print(f"Upscaling workflow: {len(workflow)} nodes")

# Upscale model comparison
models = {
    "RealESRGAN_x4plus":     {"scale": "4x", "quality": "General purpose, good default"},
    "RealESRGAN_x4plus_anime": {"scale": "4x", "quality": "Optimized for anime/illustration"},
    "4x-UltraSharp":        {"scale": "4x", "quality": "Sharp, good for photos"},
    "4x_NMKD-Siax_200k":    {"scale": "4x", "quality": "Balanced, less artifacts"},
    "ESRGAN_4x":             {"scale": "4x", "quality": "Original ESRGAN, classic"},
}

print("\\nUpscale Models:")
for name, info in models.items():
    print(f"  {name}: {info['scale']} - {info['quality']}")

# Denoise guide for upscaling
print("\\nDenoise for Upscaling:")
for d in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    effect = ("minimal change" if d < 0.2 else
              "subtle detail" if d < 0.35 else
              "balanced" if d < 0.45 else
              "significant repainting")
    print(f"  {d:.1f}: {effect}")`,id:"code-upscaling"}),e.jsx(t,{type:"tip",title:"Tiled Upscaling",content:"For very large outputs (4K+), use tiled VAE encoding/decoding to avoid VRAM overflow. ComfyUI's VAETile node processes the image in overlapping tiles. Also consider the 'Ultimate SD Upscale' custom node which automates the tile-based img2img refinement process.",id:"note-tiled-upscale"}),e.jsx(n,{title:"Over-Sharpening",content:"Using too high a denoise value (>0.5) during upscale refinement can hallucinate new content that was not in the original image, changing faces or adding unwanted elements. Keep denoise at 0.25-0.4 for detail enhancement without content alteration.",id:"warning-oversharpening"})]})}const Se=Object.freeze(Object.defineProperty({__proto__:null,default:F},Symbol.toStringTag,{value:"Module"}));function O(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Custom Nodes in ComfyUI"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"ComfyUI's power comes from its extensible custom node ecosystem. Community-built nodes add capabilities like IP-Adapter for image-guided generation, face restoration, advanced schedulers, and integration with external tools. Understanding how to install, use, and even create custom nodes unlocks the full potential of ComfyUI."}),e.jsx(i,{title:"Custom Node",definition:"A ComfyUI custom node is a Python class that defines inputs, outputs, and a processing function. It is packaged in a directory under custom_nodes/ with an __init__.py that registers the node class. Each node exposes typed input/output slots and a category for the node menu.",id:"def-custom-node"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Essential Custom Node Packs"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The ComfyUI ecosystem has hundreds of custom node packs. A few are considered essential for most advanced workflows: ComfyUI-Manager for easy installation, ControlNet auxiliary preprocessors, AnimateDiff nodes, and IP-Adapter for image conditioning."}),e.jsx(o,{title:"Essential Custom Nodes",problem:"Which custom node packs should you install first?",steps:[{formula:"\\text{ComfyUI-Manager: GUI for installing other custom nodes}",explanation:"Install this first -- it provides a UI for managing all other nodes."},{formula:"\\text{comfyui\\_controlnet\\_aux: ControlNet preprocessors}",explanation:"Canny, depth, pose, lineart detectors for ControlNet conditioning."},{formula:"\\text{ComfyUI-AnimateDiff-Evolved: Video generation}",explanation:"AnimateDiff integration with motion modules and video output."},{formula:"\\text{ComfyUI\\_IPAdapter\\_plus: Image-guided generation}",explanation:"Use reference images to guide style and content."}],id:"example-essential-nodes"}),e.jsx(a,{title:"custom_node_creation.py",code:`# Creating a custom ComfyUI node
# File: custom_nodes/my_nodes/__init__.py

# Minimal custom node example
class ImageBrightnessAdjust:
    """Custom node to adjust image brightness."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),           # Typed input: IMAGE tensor
                "brightness": ("FLOAT", {      # Float parameter with constraints
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)          # Output type
    RETURN_NAMES = ("adjusted_image",) # Output name
    FUNCTION = "adjust"                # Method to call
    CATEGORY = "image/adjustments"     # Menu category

    def adjust(self, image, brightness):
        # image is a torch tensor: (B, H, W, C) in [0, 1]
        import torch
        adjusted = torch.clamp(image * brightness, 0.0, 1.0)
        return (adjusted,)  # Must return a tuple

# Register nodes
NODE_CLASS_MAPPINGS = {
    "ImageBrightnessAdjust": ImageBrightnessAdjust,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBrightnessAdjust": "Brightness Adjust",
}

# Installation via ComfyUI-Manager or git
install_methods = """
# Method 1: ComfyUI-Manager (GUI)
# Click Manager -> Install Custom Nodes -> Search

# Method 2: Git clone
cd ComfyUI/custom_nodes
git clone https://github.com/user/custom-node-pack.git
pip install -r custom-node-pack/requirements.txt

# Method 3: Manual
# Copy Python files to custom_nodes/my_nodes/
# Restart ComfyUI
"""
print(install_methods)

# Popular custom node packs
packs = {
    "ComfyUI-Manager":              "Node management GUI",
    "comfyui_controlnet_aux":       "ControlNet preprocessors",
    "ComfyUI-AnimateDiff-Evolved":  "AnimateDiff video generation",
    "ComfyUI_IPAdapter_plus":       "Image prompt adapter",
    "ComfyUI-Impact-Pack":          "Face detection, SAM, utilities",
    "ComfyUI_UltimateSDUpscale":    "Tiled upscaling workflow",
    "ComfyUI-KJNodes":              "Utility nodes (math, logic, etc)",
    "rgthree-comfy":                "Quality-of-life improvements",
}
for name, desc in packs.items():
    print(f"  {name}: {desc}")`,id:"code-custom-nodes"}),e.jsx(t,{type:"tip",title:"IP-Adapter for Style Transfer",content:"IP-Adapter (Image Prompt Adapter) is one of the most powerful custom nodes. It takes a reference image and extracts style/content features that condition the generation. Unlike img2img, IP-Adapter preserves the reference style without constraining the composition. Use strength 0.5-0.8 for style influence.",id:"note-ip-adapter"}),e.jsx(n,{title:"Custom Node Stability",content:"Custom nodes are community-maintained and may break with ComfyUI updates. Pin your ComfyUI version for production use. Some custom nodes may conflict with each other or have incompatible dependencies. Always test in isolation before adding to complex workflows.",id:"warning-stability"})]})}const Ie=Object.freeze(Object.defineProperty({__proto__:null,default:O},Symbol.toStringTag,{value:"Module"}));function G(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"ComfyUI API Mode"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"ComfyUI exposes a REST API that accepts workflow JSON and returns generated images programmatically. This enables integration into production applications, batch processing pipelines, and automated testing. The API accepts the same workflow format used internally, making it straightforward to convert visual workflows to API calls."}),e.jsx(i,{title:"ComfyUI API",definition:"The ComfyUI API server (default port 8188) exposes endpoints for queuing workflows (/prompt), checking status (/queue), retrieving results (/history), and uploading images (/upload/image). Workflows are submitted as JSON containing the node graph, and results are retrieved via WebSocket or polling.",id:"def-comfyui-api"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"API Workflow"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The typical API flow is: (1) submit a workflow JSON to /prompt, (2) receive a prompt_id, (3) monitor progress via WebSocket, (4) retrieve results from /history when complete."}),e.jsx(o,{title:"API Request Flow",problem:"Submit a txt2img workflow and retrieve the result.",steps:[{formula:'\\text{POST /prompt} \\to \\text{\\{prompt\\_id: "abc123"\\}}',explanation:"Submit workflow JSON, receive tracking ID."},{formula:"\\text{WebSocket /ws?clientId=xyz} \\to \\text{progress events}",explanation:"Monitor execution progress in real-time."},{formula:"\\text{GET /history/abc123} \\to \\text{output metadata}",explanation:"Retrieve completion status and output file references."},{formula:"\\text{GET /view?filename=output.png} \\to \\text{image data}",explanation:"Download the generated image."}],id:"example-api-flow"}),e.jsx(a,{title:"comfyui_api_client.py",code:`import json
import urllib.request
import urllib.parse
import uuid
import io

COMFYUI_URL = "http://127.0.0.1:8188"

class ComfyUIClient:
    """Simple ComfyUI API client."""

    def __init__(self, server_url=COMFYUI_URL):
        self.server = server_url
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, workflow):
        """Submit a workflow for execution."""
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.server}/prompt",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        response = json.loads(urllib.request.urlopen(req).read())
        return response["prompt_id"]

    def get_history(self, prompt_id):
        """Get execution results."""
        url = f"{self.server}/history/{prompt_id}"
        response = json.loads(urllib.request.urlopen(url).read())
        return response.get(prompt_id, {})

    def get_image(self, filename, subfolder="", folder_type="output"):
        """Download a generated image."""
        params = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type,
        })
        url = f"{self.server}/view?{params}"
        return urllib.request.urlopen(url).read()

    def generate(self, workflow, timeout=120):
        """Submit workflow and wait for results."""
        import time

        prompt_id = self.queue_prompt(workflow)
        print(f"Queued: {prompt_id}")

        # Poll for completion
        start = time.time()
        while time.time() - start < timeout:
            history = self.get_history(prompt_id)
            if history:
                outputs = history.get("outputs", {})
                images = []
                for node_output in outputs.values():
                    if "images" in node_output:
                        for img_info in node_output["images"]:
                            img_data = self.get_image(
                                img_info["filename"],
                                img_info.get("subfolder", ""),
                            )
                            images.append(img_data)
                return images
            time.sleep(1)
        raise TimeoutError(f"Generation timed out after {timeout}s")

# Usage example
# client = ComfyUIClient()
# workflow = {...}  # Your workflow JSON
# images = client.generate(workflow)
# with open("output.png", "wb") as f:
#     f.write(images[0])

# Batch generation with parameter sweeps
def parameter_sweep(base_workflow, param_node, param_name, values):
    """Generate images across a parameter sweep."""
    results = []
    for val in values:
        workflow = json.loads(json.dumps(base_workflow))  # Deep copy
        workflow[param_node]["inputs"][param_name] = val
        results.append({"value": val, "workflow": workflow})
    return results

# Example: CFG sweep
sweep = parameter_sweep(
    base_workflow={"8": {"class_type": "KSampler", "inputs": {"cfg": 7.0}}},
    param_node="8", param_name="cfg",
    values=[3.0, 5.0, 7.0, 10.0, 15.0]
)
print(f"Generated {len(sweep)} workflow variants for CFG sweep")`,id:"code-api-client"}),e.jsx(t,{type:"tip",title:"WebSocket for Real-Time Progress",content:"For production applications, use WebSocket connections instead of polling. Connect to ws://host:8188/ws?clientId=YOUR_ID to receive real-time progress events including current step, total steps, and preview images. This enables progress bars and live previews in your application.",id:"note-websocket"}),e.jsx(n,{title:"API Security",content:"ComfyUI's API has no built-in authentication. Never expose it directly to the internet. Use a reverse proxy (nginx, Caddy) with authentication for remote access. The API allows arbitrary code execution via custom nodes, so treat it as a privileged endpoint.",id:"warning-security"})]})}const De=Object.freeze(Object.defineProperty({__proto__:null,default:G},Symbol.toStringTag,{value:"Module"}));function $(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"ComfyUI Performance Optimization"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Optimizing ComfyUI performance involves reducing VRAM usage, accelerating inference, and efficiently managing model loading. Key techniques include model quantization, attention optimization, VAE tiling, and strategic use of CPU offloading. These optimizations can make the difference between running models at all on consumer hardware versus needing expensive cloud GPUs."}),e.jsx(i,{title:"Performance Dimensions",definition:"ComfyUI performance has three key dimensions: (1) VRAM usage -- how much GPU memory models and intermediates consume, (2) inference speed -- time per generation step, measured in iterations/second, and (3) model loading time -- how quickly checkpoints are swapped. Optimizing one often trades off against another.",id:"def-performance"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"VRAM Optimization"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"VRAM is the primary bottleneck for most users. ComfyUI provides several mechanisms to reduce memory usage at the cost of some speed."}),e.jsx(o,{title:"VRAM Usage by Model",problem:"How much VRAM do different models need in various precision modes?",steps:[{formula:"\\text{SD 1.5 (fp16): } \\sim 3.5\\text{GB model} + 1\\text{GB working} = 4.5\\text{GB}",explanation:"Fits on most modern GPUs."},{formula:"\\text{SDXL (fp16): } \\sim 6.5\\text{GB model} + 2\\text{GB working} = 8.5\\text{GB}",explanation:"Needs 10GB+ GPU for comfortable use."},{formula:"\\text{FLUX (fp8): } \\sim 12\\text{GB model} + 3\\text{GB working} = 15\\text{GB}",explanation:"Requires fp8 quantization for consumer GPUs."},{formula:"\\text{FLUX (fp16): } \\sim 24\\text{GB} \\to \\text{needs A5000/3090/4090}",explanation:"Full precision requires high-end GPU or offloading."}],id:"example-vram"}),e.jsx(a,{title:"performance_optimization.py",code:`# ComfyUI performance optimization techniques

# 1. Command-line flags for memory management
cli_flags = {
    "--lowvram":   "Aggressively offload to CPU, ~3x slower but uses minimal VRAM",
    "--novram":    "Everything on CPU except active computation, very slow",
    "--gpu-only":  "Keep everything in VRAM (fast, needs lots of VRAM)",
    "--highvram":  "Disable smart memory management (fastest if you have 24GB+)",
    "--fp8_e4m3fn-unet": "Load UNet in fp8 (halves UNet VRAM)",
    "--fp16-vae":  "Run VAE in fp16 instead of fp32",
    "--bf16-vae":  "Run VAE in bf16 (Ampere+ GPUs)",
}

print("ComfyUI CLI Flags:")
for flag, desc in cli_flags.items():
    print(f"  {flag:25s} {desc}")

# 2. Attention optimization
attention_modes = {
    "pytorch":     "Default PyTorch attention",
    "xformers":    "xformers memory-efficient attention (install separately)",
    "sdp":         "PyTorch 2.0+ scaled dot product attention (recommended)",
}

print("\\nAttention Backends:")
for mode, desc in attention_modes.items():
    print(f"  {mode}: {desc}")

# 3. VRAM estimation helper
def estimate_vram(model_params_B, precision="fp16", working_memory_GB=2.0):
    """Estimate VRAM needed for a model."""
    bytes_per_param = {
        "fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5
    }
    model_gb = model_params_B * bytes_per_param[precision] / 1e9
    total = model_gb + working_memory_GB
    return model_gb, total

models = [
    ("SD 1.5 UNet", 0.86),
    ("SDXL UNet", 2.6),
    ("FLUX DiT", 12.0),
    ("T5-XXL", 4.7),
    ("CLIP-L", 0.12),
]

print("\\nVRAM Estimates:")
for name, params in models:
    for prec in ["fp16", "fp8"]:
        model_gb, total = estimate_vram(params, prec)
        print(f"  {name} ({prec}): model={model_gb:.1f}GB, total~{total:.1f}GB")

# 4. Generation speed benchmarks (approximate)
print("\\nApprox. Generation Speed (RTX 4090, 512x512, 20 steps):")
benchmarks = {
    "SD 1.5 (fp16)":    "~2.5 seconds",
    "SD 1.5 + LoRA":    "~2.7 seconds",
    "SDXL (fp16)":      "~8 seconds at 1024x1024",
    "FLUX (fp8)":       "~15 seconds at 1024x1024",
    "AnimateDiff 16f":  "~20 seconds",
}
for model, speed in benchmarks.items():
    print(f"  {model}: {speed}")

# 5. Model caching strategy
print("\\nModel Caching Tips:")
tips = [
    "ComfyUI caches the last-used model in VRAM automatically",
    "Switching models triggers unload + reload (slow for large models)",
    "Use 'Keep Models in Memory' for workflows using multiple models",
    "For batch jobs with same model: queue all at once to avoid reloading",
    "FLUX: preload T5 and keep it; swap only the UNET if comparing versions",
]
for i, tip in enumerate(tips, 1):
    print(f"  {i}. {tip}")`,id:"code-performance"}),e.jsx(t,{type:"tip",title:"Quick Performance Wins",content:"(1) Use --fp16-vae flag to halve VAE memory. (2) Enable PyTorch 2.0 SDP attention (default in recent ComfyUI). (3) Use fp8 UNet loading for FLUX. (4) Enable VAE tiling for images above 1024px. (5) Reduce preview frequency in settings. These five changes alone can reduce VRAM usage by 30-50%.",id:"note-quick-wins"}),e.jsx(t,{type:"note",title:"torch.compile Support",content:"Recent ComfyUI versions support torch.compile() for the UNet/DiT, which can speed up inference by 10-30% after an initial compilation delay. Add --use-pytorch-cross-attention --force-channels-last for best results. This requires PyTorch 2.0+ and works best on Ampere/Ada GPUs.",id:"note-torch-compile"}),e.jsx(n,{title:"Quantization Artifacts",content:"FP8 and INT8 quantization can introduce subtle quality degradation, particularly in skin tones, fine textures, and color gradients. Always compare quantized vs full-precision outputs for quality-sensitive applications. NF4 quantization (4-bit) has more visible artifacts and is best for previews only.",id:"warning-quantization"})]})}const Pe=Object.freeze(Object.defineProperty({__proto__:null,default:$},Symbol.toStringTag,{value:"Module"}));export{we as A,ve as B,ke as C,Le as D,je as E,Ce as F,Ae as G,Te as H,Se as I,Ie as J,De as K,Pe as L,K as a,J as b,Y as c,Z as d,ee as e,te as f,ae as g,ie as h,oe as i,ne as j,se as k,re as l,le as m,de as n,me as o,ce as p,pe as q,fe as r,Q as s,ue as t,he as u,ge as v,xe as w,_e as x,ye as y,be as z};
