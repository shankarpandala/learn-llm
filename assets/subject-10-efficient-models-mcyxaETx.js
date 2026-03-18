import{j as e}from"./vendor-DWbzdFaj.js";import"./vendor-katex-BYl39Yo6.js";import{D as a,N as t,E as i,P as n,T as o,W as r}from"./subject-01-text-fundamentals-DG6tAvii.js";function s(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Knowledge Distillation: Teacher-Student Training"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:`Knowledge distillation compresses a large teacher model into a smaller student model by training the student to mimic the teacher's soft probability distribution. This transfers the teacher's "dark knowledge" — the relative probabilities it assigns to incorrect classes — which contains far more information than hard labels alone.`}),e.jsx(a,{title:"Knowledge Distillation",definition:"Knowledge distillation trains a compact student network to reproduce the behavior of a larger teacher network. The student learns from the teacher's softened output distribution at temperature $T$: $q_i = \\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)}$ where $z_i$ are the teacher's logits.",notation:"$L = \\alpha \\, L_{CE}(y, p_s) + (1 - \\alpha) \\, T^2 \\, L_{KD}(q_t, q_s)$ where $L_{CE}$ is cross-entropy with hard labels, $L_{KD}$ is KL divergence between teacher and student soft outputs, and $\\alpha$ balances the two terms.",id:"def-distillation"}),e.jsx(t,{type:"intuition",title:"Why Soft Labels Help",content:"When a teacher predicts 'cat' with 0.7, 'tiger' with 0.2, and 'dog' with 0.1, the student learns that cats look somewhat like tigers — structural knowledge absent from the hard label [1, 0, 0]. Temperature T > 1 further softens this distribution, amplifying these inter-class relationships.",id:"note-soft-labels"}),e.jsx(i,{title:"Distillation Loss Computation",problem:"Compute the distillation loss for a 3-class problem with temperature T=4, teacher logits [5, 3, 1], student logits [4, 2, 0.5], and true label class 0.",steps:[{formula:"q_t = \\text{softmax}([5/4, 3/4, 1/4]) = [0.506, 0.312, 0.182]",explanation:"Soften teacher logits by dividing by temperature T=4."},{formula:"q_s = \\text{softmax}([4/4, 2/4, 0.5/4]) = [0.509, 0.297, 0.194]",explanation:"Soften student logits similarly."},{formula:"L_{KD} = \\sum_i q_t^{(i)} \\log \\frac{q_t^{(i)}}{q_s^{(i)}} = 0.000457",explanation:"KL divergence between teacher and student soft distributions."},{formula:"L = 0.5 \\cdot L_{CE} + 0.5 \\cdot 16 \\cdot 0.000457",explanation:"Combine with hard-label loss. Multiply KD loss by T^2 = 16 to balance gradient magnitudes."}],id:"example-distillation-loss"}),e.jsx(n,{title:"knowledge_distillation.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation."""
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        # Hard-label cross-entropy loss
        ce_loss = F.cross_entropy(student_logits, labels)

        # Soft-label KL divergence loss
        student_soft = F.log_softmax(student_logits / self.T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=-1)
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')

        # Combine: multiply KD loss by T^2 to match gradient scale
        total = self.alpha * ce_loss + (1 - self.alpha) * (self.T ** 2) * kd_loss
        return total

# Example: distill a 6-layer teacher into a 2-layer student
teacher_logits = torch.tensor([[5.0, 3.0, 1.0], [2.0, 6.0, 0.5]])
student_logits = torch.tensor([[4.0, 2.0, 0.5], [1.5, 5.0, 0.3]])
labels = torch.tensor([0, 1])

criterion = DistillationLoss(temperature=4.0, alpha=0.5)
loss = criterion(student_logits, teacher_logits, labels)
print(f"Distillation loss: {loss.item():.4f}")

# Model size comparison
teacher_params = 110_000_000   # BERT-Base: 110M
student_params = 22_000_000    # DistilBERT: 22M (6 layers -> 3 layers, ~60% fewer params)
compression = teacher_params / student_params
print(f"Compression ratio: {compression:.1f}x")
print(f"Student retains ~97% of teacher performance on GLUE")`,id:"code-distillation"}),e.jsx(o,{title:"Gradient Scaling with Temperature",statement:"The gradients of the KL divergence loss with respect to student logits scale as $1/T^2$ when using temperature $T$. Therefore, the distillation loss must be multiplied by $T^2$ to keep gradient magnitudes comparable to the hard-label loss.",proof:"For softmax with temperature, $\\partial q_i / \\partial z_j = (q_i(\\delta_{ij} - q_j))/T$. Since KL divergence involves $\\partial / \\partial z$, the chain rule introduces two factors of $1/T$, yielding an overall $1/T^2$ scaling.",id:"thm-temperature-scaling"}),e.jsx(t,{type:"historical",title:"Distillation in Practice",content:"Hinton et al. (2015) formalized knowledge distillation. DistilBERT (Sanh et al., 2019) applied it to BERT, removing every other layer and achieving 97% of BERT's performance with 40% fewer parameters and 60% faster inference. TinyBERT further distills attention matrices and hidden states, not just output logits.",id:"note-distillation-history"}),e.jsx(r,{title:"Teacher Quality Matters",content:"Distillation can only compress knowledge the teacher actually has. A poorly trained teacher produces noisy soft labels that may hurt the student. Always validate teacher performance before distillation. Also, very large temperature values can over-smooth distributions, losing discriminative information.",id:"warning-teacher-quality"})]})}const q=Object.freeze(Object.defineProperty({__proto__:null,default:s},Symbol.toStringTag,{value:"Module"}));function l(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Pruning: Magnitude and Structured Approaches"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Pruning removes redundant parameters from neural networks, reducing model size and computation. The key insight is that large networks are over-parameterized: many weights contribute negligibly to the output and can be zeroed out with minimal accuracy loss."}),e.jsx(a,{title:"Weight Pruning",definition:"Pruning sets a subset of model weights to zero based on an importance criterion. Unstructured pruning removes individual weights: $w_{ij} = 0$ if $|w_{ij}| < \\theta$. Structured pruning removes entire neurons, attention heads, or layers, yielding direct speedups on hardware.",notation:"Sparsity $s = \\frac{\\text{number of zero weights}}{\\text{total weights}}$. A model with 90% sparsity has only 10% non-zero parameters.",id:"def-pruning"}),e.jsx(i,{title:"Magnitude Pruning",problem:"Given weight matrix W = [[0.8, -0.02, 0.5], [0.01, -0.7, 0.03]], prune to 50% sparsity using magnitude pruning.",steps:[{formula:"|W| = [[0.8, 0.02, 0.5], [0.01, 0.7, 0.03]]",explanation:"Compute absolute values of all weights."},{formula:"\\text{sorted} = [0.01, 0.02, 0.03, 0.5, 0.7, 0.8]",explanation:"Sort all magnitudes. For 50% sparsity, threshold is the 3rd value = 0.03."},{formula:"W_{\\text{pruned}} = [[0.8, 0, 0.5], [0, -0.7, 0]]",explanation:"Zero out all weights with magnitude <= 0.03. Three of six weights removed."}],id:"example-magnitude-pruning"}),e.jsx(n,{title:"pruning_methods.py",code:`import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Create a simple linear layer
layer = nn.Linear(768, 768, bias=False)
total_params = layer.weight.numel()
print(f"Total parameters: {total_params:,}")  # 589,824

# --- Unstructured magnitude pruning ---
prune.l1_unstructured(layer, name='weight', amount=0.5)
sparsity = (layer.weight == 0).sum().item() / total_params
print(f"Unstructured sparsity: {sparsity:.1%}")  # 50.0%

# Remove pruning reparameterization (make permanent)
prune.remove(layer, 'weight')

# --- Structured pruning: remove entire output neurons ---
layer2 = nn.Linear(768, 768, bias=False)
prune.ln_structured(layer2, name='weight', amount=0.3, n=2, dim=0)
# 30% of output neurons (rows) are zeroed out
zero_rows = (layer2.weight.sum(dim=1) == 0).sum().item()
print(f"Pruned neurons: {zero_rows}/{layer2.weight.shape[0]}")

# --- Iterative magnitude pruning (IMP) ---
# Key idea: prune, retrain, prune more, retrain...
def iterative_prune(model, target_sparsity=0.9, steps=10):
    """Gradually increase sparsity over multiple rounds."""
    per_step = 1 - (1 - target_sparsity) ** (1 / steps)
    for step in range(steps):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, 'weight', amount=per_step)
        # In practice: retrain for several epochs here
        current = sum((p == 0).sum().item() for p in model.parameters())
        total = sum(p.numel() for p in model.parameters())
        print(f"Step {step+1}: sparsity = {current/total:.1%}")
    return model

# Model size calculation at different sparsity levels
base_size_gb = 7.0  # 7B params at FP16 = ~14GB, but ~7GB with sparse storage
for s in [0.5, 0.8, 0.9, 0.95]:
    effective_gb = base_size_gb * (1 - s)
    print(f"Sparsity {s:.0%}: effective size = {effective_gb:.2f} GB")`,id:"code-pruning"}),e.jsx(o,{title:"Lottery Ticket Hypothesis",statement:"A randomly initialized dense network contains a subnetwork (the 'winning ticket') that, when trained in isolation with its original initialization, can match the full network's accuracy. Formally, there exists a mask $m$ such that training $f(x; m \\odot \\theta_0)$ achieves test accuracy comparable to training $f(x; \\theta_0)$.",proof:"Frankle & Carlin (2019) demonstrated this empirically: train a network, prune the smallest-magnitude weights, rewind remaining weights to their initial values, and retrain. The resulting sparse network matches or exceeds the dense network's performance at 10-20% of the original size.",corollaries:["Over-parameterization during training is beneficial even if the final model is sparse.","The specific initialization of surviving weights matters — random re-initialization fails."],id:"thm-lottery-ticket"}),e.jsx(t,{type:"tip",title:"Structured vs. Unstructured Trade-offs",content:"Unstructured pruning achieves higher sparsity at the same accuracy but requires sparse matrix libraries for speedup. Structured pruning (removing heads, layers, or channels) gives immediate speedup on standard hardware. For LLMs, removing 30-50% of attention heads often preserves 95%+ of performance.",id:"note-structured-vs-unstructured"}),e.jsx(r,{title:"Pruning Without Retraining",content:"One-shot pruning without fine-tuning degrades accuracy rapidly above 40-50% sparsity. SparseGPT (2023) addresses this by solving a layer-wise reconstruction problem, enabling 50-60% unstructured sparsity on GPT-scale models without any retraining.",id:"warning-pruning-retraining"})]})}const z=Object.freeze(Object.defineProperty({__proto__:null,default:l},Symbol.toStringTag,{value:"Module"}));function d(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Weight Sharing and Weight Tying"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Weight sharing reduces model size by reusing the same parameters across different parts of the network. The most common form in LLMs is embedding-output weight tying, where the input embedding matrix and the output projection share the same parameters, saving hundreds of millions of parameters in large-vocabulary models."}),e.jsx(a,{title:"Weight Tying",definition:"Weight tying constrains two or more parameter matrices to be identical. In language models, the input embedding $E \\in \\mathbb{R}^{V \\times d}$ and the output projection $W_o \\in \\mathbb{R}^{d \\times V}$ are tied: $W_o = E^\\top$. The output logits become $z = h \\cdot E^\\top$ where $h$ is the final hidden state.",notation:"Memory saved $= V \\times d$ parameters. For $V = 50{,}000$ and $d = 4{,}096$: savings = $204{,}800{,}000$ parameters ($\\approx 390$ MB at FP16).",id:"def-weight-tying"}),e.jsx(i,{title:"Embedding-Output Tying Savings",problem:"Calculate parameter savings from weight tying in a model with vocabulary V=32,000 and hidden dimension d=4,096.",steps:[{formula:"\\text{Embedding params} = V \\times d = 32{,}000 \\times 4{,}096 = 131{,}072{,}000",explanation:"Input embedding matrix size."},{formula:"\\text{Without tying: } 2 \\times 131{,}072{,}000 = 262{,}144{,}000",explanation:"Separate embedding and output projection would use 262M parameters."},{formula:"\\text{With tying: } 131{,}072{,}000 \\text{ (shared)}",explanation:"Weight tying halves this to 131M parameters."},{formula:"\\text{Savings} = 131M \\text{ params} = 250\\text{ MB at FP16}",explanation:"For a 7B model, this is roughly 1.9% of total parameters — small but free."}],id:"example-weight-tying"}),e.jsx(n,{title:"weight_sharing.py",code:`import torch
import torch.nn as nn

class TiedEmbeddingLM(nn.Module):
    """Language model with tied input/output embeddings."""
    def __init__(self, vocab_size, d_model, n_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        # Output projection — weight tied with embedding
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight  # Tie weights!

    def forward(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        h = self.ln(h)
        return self.output_proj(h)  # Uses embedding weights transposed

# Compare parameter counts
V, d = 32_000, 4_096

model_tied = TiedEmbeddingLM(V, d, n_layers=2)
model_untied = TiedEmbeddingLM(V, d, n_layers=2)
# Break the tie for comparison
model_untied.output_proj.weight = nn.Parameter(torch.randn(V, d))

tied_params = sum(p.numel() for p in model_tied.parameters())
untied_params = sum(p.numel() for p in model_untied.parameters())

print(f"Tied model:   {tied_params:>12,} params")
print(f"Untied model: {untied_params:>12,} params")
print(f"Savings:      {untied_params - tied_params:>12,} params")
print(f"Savings:      {(untied_params - tied_params) * 2 / 1e6:.0f} MB (FP16)")

# Cross-layer weight sharing (ALBERT-style)
class ALBERTStyleEncoder(nn.Module):
    """All transformer layers share the same parameters."""
    def __init__(self, d_model, n_virtual_layers=12):
        super().__init__()
        # Only ONE physical layer, applied n times
        self.shared_layer = nn.TransformerEncoderLayer(
            d_model, nhead=8, batch_first=True
        )
        self.n_layers = n_virtual_layers

    def forward(self, x):
        for _ in range(self.n_layers):
            x = self.shared_layer(x)
        return x

shared = ALBERTStyleEncoder(768, n_virtual_layers=12)
unshared_count = 12 * sum(p.numel() for p in shared.shared_layer.parameters())
shared_count = sum(p.numel() for p in shared.parameters())
print(f"\\nALBERT-style sharing: {shared_count:,} vs {unshared_count:,}")
print(f"Reduction: {unshared_count / shared_count:.0f}x fewer params")`,id:"code-weight-sharing"}),e.jsx(t,{type:"historical",title:"Weight Tying in Major Models",content:"Press & Wolf (2017) showed embedding-output tying improves perplexity while reducing parameters. GPT-2, T5, and LLaMA all use this technique. ALBERT (Lan et al., 2019) went further with cross-layer parameter sharing, achieving 18x fewer parameters than BERT-Large with competitive accuracy.",id:"note-weight-tying-history"}),e.jsx(t,{type:"note",title:"Beyond Simple Sharing",content:"Modern approaches include factorized embeddings (ALBERT decomposes V x d into V x e and e x d with e << d), grouped weight sharing across attention heads, and Universal Transformers that share weights across depth while using adaptive computation time.",id:"note-advanced-sharing"}),e.jsx(r,{title:"Cross-Layer Sharing Limitations",content:"While ALBERT-style full cross-layer sharing dramatically reduces parameters, it does not reduce computation (FLOPs) since every layer still runs a full forward pass. It also tends to underperform independent layers when the parameter budget is not the bottleneck.",id:"warning-cross-layer"})]})}const j=Object.freeze(Object.defineProperty({__proto__:null,default:d},Symbol.toStringTag,{value:"Module"}));function c(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Neural Architecture Search for LLMs"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Neural Architecture Search (NAS) automates the design of efficient model architectures by searching over configurations of depth, width, attention heads, and feed-forward dimensions. For LLMs, NAS finds Pareto-optimal trade-offs between accuracy and latency/memory that human designers often miss."}),e.jsx(a,{title:"Neural Architecture Search",definition:"NAS formulates architecture design as an optimization problem: $a^* = \\arg\\max_{a \\in \\mathcal{A}} \\text{Acc}(a)$ subject to $\\text{Cost}(a) \\leq B$, where $\\mathcal{A}$ is the search space, and $B$ is a resource budget (FLOPs, latency, or parameters).",notation:"Search space for Transformer LLMs typically includes: number of layers $L \\in [6, 48]$, hidden dimension $d \\in [256, 4096]$, heads $h \\in [4, 32]$, FFN ratio $r \\in [2, 8]$, yielding $|\\mathcal{A}| > 10^{12}$ candidate architectures.",id:"def-nas"}),e.jsx(i,{title:"Search Space Design for Efficient LLMs",problem:"Define a NAS search space for a Transformer targeting < 1B parameters and < 100ms latency on mobile.",steps:[{formula:"L \\in \\{6, 8, 12, 16, 20, 24\\}",explanation:"Search over different depths. Deeper is more capable but slower."},{formula:"d \\in \\{384, 512, 640, 768, 1024\\}",explanation:"Hidden dimension affects parameter count quadratically via FFN layers."},{formula:"\\text{Params} \\approx 12 L d^2 + 2Vd",explanation:"Rough parameter estimate: 12Ld^2 for transformer layers + 2Vd for embeddings."},{formula:"\\text{Filter: } 12 \\times 24 \\times 1024^2 + 2 \\times 32000 \\times 1024 \\approx 367M",explanation:"Even the largest candidate fits within the 1B budget."}],id:"example-nas-search-space"}),e.jsx(n,{title:"nas_for_transformers.py",code:`import itertools
import math

def estimate_params(n_layers, d_model, ffn_ratio=4, vocab_size=32000):
    """Estimate total parameters in a Transformer."""
    # Self-attention: Q, K, V, O projections
    attn_params = 4 * d_model * d_model * n_layers
    # FFN: two linear layers with expansion ratio
    ffn_params = 2 * d_model * (ffn_ratio * d_model) * n_layers
    # Layer norms (2 per layer)
    norm_params = 4 * d_model * n_layers
    # Embeddings (tied input/output)
    embed_params = vocab_size * d_model
    return attn_params + ffn_params + norm_params + embed_params

def estimate_flops(n_layers, d_model, seq_len=2048, ffn_ratio=4):
    """Estimate FLOPs for a single forward pass."""
    # Attention: 2 * seq * d^2 * 4 (QKV + O) + 2 * seq^2 * d (scores)
    attn_flops = n_layers * (8 * seq_len * d_model**2 + 2 * seq_len**2 * d_model)
    # FFN: 2 * seq * d * (r*d) * 2
    ffn_flops = n_layers * 4 * seq_len * d_model * (ffn_ratio * d_model)
    return attn_flops + ffn_flops

# Search space
layers_options = [6, 8, 12, 16, 20, 24]
dim_options = [384, 512, 640, 768, 1024, 1280]
ffn_ratios = [2, 4, 8]

# Brute-force search with constraints
budget_params = 1_000_000_000   # 1B parameters
budget_flops = 5e12             # 5 TFLOPs

candidates = []
for L, d, r in itertools.product(layers_options, dim_options, ffn_ratios):
    params = estimate_params(L, d, r)
    flops = estimate_flops(L, d, ffn_ratio=r)
    if params <= budget_params and flops <= budget_flops:
        # Proxy score: deeper + wider is generally better
        score = L * math.log(d)
        candidates.append((score, L, d, r, params, flops))

# Top-5 architectures by proxy score
candidates.sort(reverse=True)
print("Top-5 architectures under 1B params, 5 TFLOPs:")
print(f"{'Layers':>7} {'Dim':>6} {'FFN_r':>6} {'Params':>12} {'TFLOPs':>8}")
for score, L, d, r, params, flops in candidates[:5]:
    print(f"{L:>7} {d:>6} {r:>6} {params:>12,} {flops/1e12:>8.2f}")

# In practice, each candidate would be trained briefly (or use
# a supernet with weight-sharing) and evaluated on a validation set.`,id:"code-nas"}),e.jsx(t,{type:"note",title:"Efficient NAS Methods",content:"Early NAS required training thousands of models from scratch (Zoph & Le, 2017). Modern approaches use weight-sharing supernets (one-shot NAS), predictor-based methods that estimate accuracy without training, or hardware-aware search that directly optimizes for latency on target devices.",id:"note-efficient-nas"}),e.jsx(t,{type:"historical",title:"NAS-Designed LLMs",content:"AutoTinyBERT (2021) used NAS to find optimal BERT architectures for each downstream task. LiteTransformer (2020) discovered that splitting attention between local and global heads is more efficient. Primer (2022) by Google used NAS to find architectural modifications that improved training efficiency by 4x over vanilla Transformers.",id:"note-nas-history"}),e.jsx(r,{title:"Search Cost vs. Benefit",content:"NAS itself is computationally expensive. A single NAS run can cost thousands of GPU hours. For one-off models, hand-designed scaling laws (Chinchilla, LLaMA configurations) may be more practical. NAS shines when deploying across many hardware targets where one-size-fits-all architectures are suboptimal.",id:"warning-nas-cost"})]})}const L=Object.freeze(Object.defineProperty({__proto__:null,default:c},Symbol.toStringTag,{value:"Module"}));function m(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Post-Training Quantization (PTQ)"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Post-training quantization converts a pre-trained model's weights and activations from high-precision floating point (FP16/FP32) to lower-bit integer representations (INT8/INT4) without retraining. This is the fastest path to model compression, reducing memory by 2-4x and enabling inference on consumer hardware."}),e.jsx(a,{title:"Uniform Affine Quantization",definition:"Quantization maps a floating-point value $x$ to an integer $x_q$ using scale $s$ and zero-point $z$: $x_q = \\text{clamp}\\left(\\left\\lfloor \\frac{x}{s} \\right\\rceil + z, \\; 0, \\; 2^b - 1\\right)$ where $b$ is the bit-width, $s = \\frac{x_{\\max} - x_{\\min}}{2^b - 1}$, and $z = \\text{round}(-x_{\\min}/s)$.",notation:"Dequantization recovers the approximate value: $\\hat{x} = s \\cdot (x_q - z)$. The quantization error is $|x - \\hat{x}| \\leq s/2$.",id:"def-ptq"}),e.jsx(i,{title:"INT8 Quantization of a Weight Tensor",problem:"Quantize weights [-0.8, -0.2, 0.0, 0.3, 1.2] to INT8 (0-255 range).",steps:[{formula:"s = \\frac{1.2 - (-0.8)}{255} = \\frac{2.0}{255} \\approx 0.00784",explanation:"Scale maps the full float range to 256 integer levels."},{formula:"z = \\text{round}\\left(\\frac{0.8}{0.00784}\\right) = \\text{round}(102.04) = 102",explanation:"Zero-point ensures 0.0 maps to an exact integer."},{formula:"x_q = [0, 77, 102, 140, 255]",explanation:"Each value: round(x / 0.00784) + 102, clamped to [0, 255]."},{formula:"\\hat{x} = [-0.800, -0.196, 0.000, 0.298, 1.200]",explanation:"Dequantized values. Maximum error is ~0.004 — negligible for neural networks."}],id:"example-int8-quant"}),e.jsx(o,{title:"Model Size Under Quantization",statement:"A model with $N$ parameters stored at $b$ bits per parameter occupies $\\frac{N \\cdot b}{8}$ bytes. Quantizing from FP16 (16 bits) to INT4 (4 bits) yields a $4\\times$ size reduction.",proof:"FP16 model: $N \\times 16 / 8 = 2N$ bytes. INT4 model: $N \\times 4 / 8 = N/2$ bytes. Ratio: $2N / (N/2) = 4\\times$. A 7B-parameter model goes from 14 GB (FP16) to 3.5 GB (INT4).",corollaries:["A 70B model at INT4 (35 GB) fits on a single 48GB GPU.","INT8 quantization halves memory: 7B model goes from 14 GB to 7 GB."],id:"thm-model-size"}),e.jsx(n,{title:"post_training_quantization.py",code:`import torch
import numpy as np

def quantize_tensor(tensor, n_bits=8):
    """Symmetric quantization of a float tensor to n_bits."""
    qmin = -(2 ** (n_bits - 1))
    qmax = 2 ** (n_bits - 1) - 1

    # Compute scale from max absolute value
    abs_max = tensor.abs().max()
    scale = abs_max / qmax

    # Quantize
    quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax).to(torch.int8)

    return quantized, scale

def dequantize_tensor(quantized, scale):
    """Recover float values from quantized tensor."""
    return quantized.float() * scale

# Example: quantize a weight matrix
torch.manual_seed(42)
W = torch.randn(4096, 4096) * 0.02  # Typical LLM weight scale

W_q, scale = quantize_tensor(W, n_bits=8)
W_hat = dequantize_tensor(W_q, scale)

# Measure quantization error
mse = ((W - W_hat) ** 2).mean().item()
rel_error = (W - W_hat).abs().mean().item() / W.abs().mean().item()
print(f"Scale: {scale:.6f}")
print(f"MSE: {mse:.2e}")
print(f"Relative error: {rel_error:.4%}")

# Memory comparison
fp16_bytes = W.numel() * 2
int8_bytes = W_q.numel() * 1 + 4  # +4 bytes for scale
int4_bytes = W_q.numel() // 2 + 4  # 4 bits per weight
print(f"\\nFP16: {fp16_bytes / 1e6:.1f} MB")
print(f"INT8: {int8_bytes / 1e6:.1f} MB ({fp16_bytes/int8_bytes:.1f}x smaller)")
print(f"INT4: {int4_bytes / 1e6:.1f} MB ({fp16_bytes/int4_bytes:.1f}x smaller)")

# Full model size estimates
for params_b in [7, 13, 70]:
    fp16_gb = params_b * 2
    int8_gb = params_b * 1
    int4_gb = params_b * 0.5
    print(f"\\n{params_b}B model: FP16={fp16_gb}GB, INT8={int8_gb}GB, INT4={int4_gb}GB")`,id:"code-ptq"}),e.jsx(t,{type:"tip",title:"Calibration Data Matters",content:"PTQ methods like GPTQ and AWQ use a small calibration dataset (128-1024 samples) to determine optimal quantization parameters per layer. Using representative data for calibration is critical — the calibration set should match your deployment distribution.",id:"note-calibration"}),e.jsx(r,{title:"Accuracy Degradation at Low Bits",content:"INT8 PTQ is nearly lossless for most LLMs. At INT4, naive round-to-nearest quantization can degrade perplexity significantly (5-15% increase). This is why advanced methods like GPTQ and AWQ were developed — they minimize the layer-wise reconstruction error during quantization.",id:"warning-low-bit-accuracy"})]})}const M=Object.freeze(Object.defineProperty({__proto__:null,default:m},Symbol.toStringTag,{value:"Module"}));function p(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Quantization-Aware Training (QAT)"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Quantization-aware training simulates quantization during the forward pass while maintaining full-precision weights for gradient updates. This allows the model to adapt to quantization noise during training, achieving significantly better accuracy at low bit-widths than post-training quantization alone."}),e.jsx(a,{title:"Quantization-Aware Training",definition:"QAT inserts fake quantization operators into the computation graph: $\\hat{w} = s \\cdot \\text{clamp}\\left(\\left\\lfloor w/s \\right\\rceil, -2^{b-1}, 2^{b-1}-1\\right)$. The forward pass uses quantized weights $\\hat{w}$, but backpropagation updates the full-precision weights $w$ using the Straight-Through Estimator (STE).",notation:"STE: $\\frac{\\partial \\hat{w}}{\\partial w} \\approx 1$ when $w$ is within the clamp range, $0$ otherwise. This allows gradients to flow through the non-differentiable rounding operation.",id:"def-qat"}),e.jsx(i,{title:"STE Gradient Flow",problem:"Show how gradients pass through quantization using the Straight-Through Estimator.",steps:[{formula:"\\text{Forward: } \\hat{w} = Q(w) = s \\cdot \\text{round}(w/s)",explanation:"Quantize weight w to discrete value using scale s."},{formula:"\\frac{\\partial Q}{\\partial w} = 0 \\text{ (true gradient of rounding)}",explanation:"The rounding function has zero gradient almost everywhere — training would stall."},{formula:"\\text{STE: } \\frac{\\partial Q}{\\partial w} \\approx \\mathbb{1}_{|w| \\leq c}",explanation:"STE approximates the gradient as 1 within the clamp range, enabling learning."},{formula:"w \\leftarrow w - \\eta \\cdot \\frac{\\partial L}{\\partial \\hat{w}}",explanation:"Full-precision weights are updated using gradients computed at quantized values."}],id:"example-ste"}),e.jsx(o,{title:"QAT vs PTQ Accuracy Gap",statement:"For a model with quantization noise $\\epsilon_q \\sim \\mathcal{U}(-s/2, s/2)$ per weight, QAT reduces the effective noise by learning weight distributions that are robust to discretization. Empirically, QAT at INT4 achieves accuracy comparable to PTQ at INT8.",proof:"QAT optimizes the loss landscape including quantization noise: $\\min_w L(Q(w); \\mathcal{D})$. The model learns to place weights near quantization grid points and to reduce sensitivity to quantization in critical layers. This is analogous to training with weight noise regularization.",id:"thm-qat-vs-ptq"}),e.jsx(n,{title:"quantization_aware_training.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class FakeQuantize(torch.autograd.Function):
    """Simulates quantization in forward, passes gradients via STE."""
    @staticmethod
    def forward(ctx, x, n_bits=8):
        qmin = -(2 ** (n_bits - 1))
        qmax = 2 ** (n_bits - 1) - 1
        scale = x.abs().max() / qmax
        # Quantize and dequantize
        x_q = torch.clamp(torch.round(x / scale), qmin, qmax)
        x_hat = x_q * scale
        # Save mask for STE
        ctx.save_for_backward((x >= qmin * scale) & (x <= qmax * scale))
        return x_hat

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        # STE: pass gradient through where within clamp range
        return grad_output * mask.float(), None

fake_quantize = FakeQuantize.apply

class QATLinear(nn.Module):
    """Linear layer with fake quantization for QAT."""
    def __init__(self, in_features, out_features, n_bits=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.n_bits = n_bits

    def forward(self, x):
        # Quantize weights during forward pass
        w_q = fake_quantize(self.linear.weight, self.n_bits)
        return F.linear(x, w_q, self.linear.bias)

# Training loop comparison: QAT vs standard
model_qat = nn.Sequential(
    QATLinear(784, 256, n_bits=4),
    nn.ReLU(),
    QATLinear(256, 10, n_bits=4),
)

# Simulate one training step
x = torch.randn(32, 784)
y = torch.randint(0, 10, (32,))
optimizer = torch.optim.Adam(model_qat.parameters(), lr=1e-3)

logits = model_qat(x)
loss = F.cross_entropy(logits, y)
loss.backward()
optimizer.step()
print(f"QAT training loss: {loss.item():.4f}")

# After training, export to actual INT4 weights
for name, module in model_qat.named_modules():
    if isinstance(module, QATLinear):
        w = module.linear.weight.data
        scale = w.abs().max() / 7  # INT4: -8 to 7
        w_int = torch.clamp(torch.round(w / scale), -8, 7).to(torch.int8)
        print(f"{name}: scale={scale:.6f}, unique values={w_int.unique().numel()}")`,id:"code-qat"}),e.jsx(t,{type:"intuition",title:"Why QAT Outperforms PTQ",content:"Think of PTQ as forcing a dancer to perform in a straitjacket after learning to dance freely. QAT is like training the dancer in the straitjacket from the start — they learn movements that work within the constraints. The model learns weight distributions that are naturally quantization-friendly.",id:"note-qat-intuition"}),e.jsx(t,{type:"tip",title:"QAT for LLMs in Practice",content:"Full QAT of a 70B model is prohibitively expensive. Practical approaches include: (1) QLoRA-style QAT that only trains adapters, (2) applying QAT only to the most quantization-sensitive layers, and (3) short QAT fine-tuning (a few hundred steps) after initial PTQ to recover lost accuracy.",id:"note-qat-practical"}),e.jsx(r,{title:"STE Approximation Degrades at Low Bits",content:"The STE becomes a poor gradient approximation below 4 bits because the rounding error is too large relative to the step size. At 2-bit and below, more sophisticated gradient estimators or alternative training strategies (like learned step sizes via LSQ) are needed.",id:"warning-ste-limits"})]})}const S=Object.freeze(Object.defineProperty({__proto__:null,default:p},Symbol.toStringTag,{value:"Module"}));function u(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"GPTQ and AWQ: Advanced Weight Quantization"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"GPTQ and AWQ are state-of-the-art post-training quantization methods specifically designed for large language models. They achieve INT4/INT3 quantization with minimal accuracy loss by solving layer-wise optimization problems using calibration data."}),e.jsx(a,{title:"GPTQ (Generative Pre-Trained Quantization)",definition:"GPTQ quantizes weights column-by-column, minimizing the squared error $\\| W X - \\hat{W} X \\|_2^2$ where $X$ is the layer input from calibration data. It uses the inverse Hessian $H^{-1} = (X X^\\top)^{-1}$ to optimally adjust remaining weights after quantizing each column, compensating for quantization error.",notation:"For each column $j$: quantize $w_j$, compute error $\\delta_j = (w_j - \\hat{w}_j) / [H^{-1}]_{jj}$, update remaining weights: $W_{:, j+1:} \\leftarrow W_{:, j+1:} - \\delta_j \\cdot H^{-1}_{j, j+1:} / H^{-1}_{jj}$.",id:"def-gptq"}),e.jsx(a,{title:"AWQ (Activation-Aware Weight Quantization)",definition:"AWQ observes that a small fraction of weights (0.1-1%) are critical because they correspond to large activation magnitudes. Instead of mixed-precision, AWQ applies per-channel scaling: $s_j = \\left(\\frac{\\max|X_j|}{q_{\\max}}\\right)^\\alpha$ to protect salient channels before quantization, where $\\alpha \\in [0, 1]$ balances protection and quantization range.",id:"def-awq"}),e.jsx(i,{title:"GPTQ Column-wise Quantization",problem:"Quantize a 2x3 weight matrix W using GPTQ with calibration data.",steps:[{formula:"H = X X^\\top + \\lambda I \\quad \\text{(Hessian with damping)}",explanation:"Compute the Hessian from calibration inputs. Damping prevents numerical issues."},{formula:"\\text{Quantize column 0: } \\hat{w}_0 = Q(w_0)",explanation:"Quantize the first column using round-to-nearest."},{formula:"\\delta_0 = w_0 - \\hat{w}_0 \\quad \\text{(quantization error)}",explanation:"Measure the error introduced by quantizing column 0."},{formula:"W_{:, 1:} \\leftarrow W_{:, 1:} - \\delta_0 \\cdot (H^{-1}_{0, 1:} / H^{-1}_{00})",explanation:"Compensate: adjust remaining columns to minimize overall output error."}],id:"example-gptq"}),e.jsx(n,{title:"gptq_awq_quantization.py",code:`import torch
import torch.nn.functional as F

def gptq_quantize(W, X, n_bits=4, block_size=128):
    """Simplified GPTQ quantization of weight matrix W given input X."""
    rows, cols = W.shape
    qmin, qmax = -(2**(n_bits-1)), 2**(n_bits-1) - 1

    # Compute Hessian
    H = X @ X.T / X.shape[1]
    H += 1e-4 * torch.eye(cols)  # Damping
    H_inv = torch.linalg.inv(H)

    W_q = W.clone()
    scales = torch.zeros(rows)

    # Process in blocks for efficiency
    for j in range(0, cols, block_size):
        end = min(j + block_size, cols)
        for k in range(j, end):
            # Quantize column k
            w_col = W_q[:, k]
            scale = w_col.abs().max() / qmax
            scales_col = scale
            w_int = torch.clamp(torch.round(w_col / scale), qmin, qmax)
            w_hat = w_int * scale

            # Error compensation for remaining columns
            error = (w_col - w_hat) / H_inv[k, k]
            W_q[:, k] = w_hat
            if k + 1 < cols:
                W_q[:, k+1:] -= error.unsqueeze(1) * H_inv[k, k+1:].unsqueeze(0)

    return W_q

def awq_quantize(W, X, n_bits=4, alpha=0.5):
    """Simplified AWQ: scale salient channels before quantization."""
    qmax = 2**(n_bits-1) - 1

    # Find salient channels based on activation magnitudes
    act_scales = X.abs().mean(dim=1)  # Per-channel activation magnitude
    s = (act_scales / act_scales.max()).pow(alpha)
    s = s.clamp(min=1e-5)

    # Scale weights to protect salient channels
    W_scaled = W * s.unsqueeze(0)

    # Standard quantization on scaled weights
    scale = W_scaled.abs().max(dim=0).values / qmax
    W_int = torch.clamp(torch.round(W_scaled / scale), -qmax, qmax)
    W_hat = (W_int * scale) / s.unsqueeze(0)  # Undo scaling

    return W_hat

# Compare methods
torch.manual_seed(42)
W = torch.randn(256, 256) * 0.02
X = torch.randn(256, 1024) * 0.5  # Calibration activations

# Naive round-to-nearest
scale_naive = W.abs().max() / 7
W_naive = torch.clamp(torch.round(W / scale_naive), -8, 7) * scale_naive

# GPTQ
W_gptq = gptq_quantize(W, X, n_bits=4)

# AWQ
W_awq = awq_quantize(W, X, n_bits=4)

# Measure output error (what matters for accuracy)
Y_true = W @ X
for name, W_q in [("Naive", W_naive), ("GPTQ", W_gptq), ("AWQ", W_awq)]:
    Y_q = W_q @ X
    mse = ((Y_true - Y_q) ** 2).mean().item()
    print(f"{name:>6} INT4 output MSE: {mse:.6f}")`,id:"code-gptq-awq"}),e.jsx(o,{title:"Optimality of GPTQ Error Compensation",statement:"GPTQ's column-wise update rule minimizes the layer output error $\\|WX - \\hat{W}X\\|_F^2$ greedily. The compensation step is the optimal least-squares adjustment for remaining weights given the Hessian of the layer output.",proof:"After quantizing column $j$, the residual error is $\\delta_j \\cdot X_j$. The optimal adjustment to column $k > j$ minimizes $\\|\\delta_j X_j - \\Delta w_k X_k\\|^2$, giving $\\Delta w_k = \\delta_j \\cdot \\text{Cov}(X_j, X_k) / \\text{Var}(X_k) = \\delta_j \\cdot H^{-1}_{jk} / H^{-1}_{jj}$.",id:"thm-gptq-optimality"}),e.jsx(t,{type:"tip",title:"Practical Quantization Choice",content:"GPTQ is better when you need the absolute best INT4 quality and have calibration data. AWQ is faster to apply and more robust across different input distributions. Both methods quantize a 70B model in under an hour on a single GPU, producing models that fit in 35GB at INT4.",id:"note-practical-choice"}),e.jsx(r,{title:"Calibration Data Sensitivity",content:"Both GPTQ and AWQ assume the calibration data is representative. If calibration uses English Wikipedia but deployment involves code generation, the quantization parameters may be suboptimal. Use diverse, domain-appropriate calibration sets for best results.",id:"warning-calibration-sensitivity"})]})}const N=Object.freeze(Object.defineProperty({__proto__:null,default:u},Symbol.toStringTag,{value:"Module"}));function h(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Extreme Quantization: 1-Bit and Ternary Models"}),e.jsxs("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:["Extreme quantization pushes model compression to the limit: ternary weights use only ","{-1, 0, +1}"," and binary weights use ","{-1, +1}",". These enable replacing multiplications with additions and sign flips, offering dramatic speedups on specialized hardware. BitNet and related work show this is viable even for billion-parameter language models."]}),e.jsx(a,{title:"Ternary Quantization",definition:"Ternary quantization maps each weight to one of three values: $w_q \\in \\{-\\alpha, 0, +\\alpha\\}$ where $\\alpha$ is a learned or computed scaling factor. The quantization function is $w_q = \\alpha \\cdot \\text{sign}(w) \\cdot \\mathbb{1}(|w| > \\Delta)$ where $\\Delta$ is a threshold.",notation:"For a weight matrix $W$: $\\hat{W} = \\alpha \\cdot T$ where $T \\in \\{-1, 0, +1\\}^{m \\times n}$ and $\\alpha = \\frac{\\sum_{|w| > \\Delta} |w|}{|\\{w : |w| > \\Delta\\}|}$ (mean of non-zero magnitudes).",id:"def-ternary"}),e.jsx(a,{title:"BitNet b1.58",definition:"BitNet b1.58 (Ma et al., 2024) constrains every weight to $\\{-1, 0, +1\\}$, achieving 1.58 bits per weight ($\\log_2 3 \\approx 1.58$). Matrix multiplication $Y = WX$ becomes $Y = \\alpha \\cdot (T \\cdot X)$ where the ternary multiply reduces to additions and subtractions only.",id:"def-bitnet"}),e.jsx(i,{title:"Ternary Matrix Multiplication",problem:"Compute Y = W * x where W is ternary and x = [0.5, -0.3, 0.8, 0.1].",steps:[{formula:"T = [[1, 0, -1, 1], [-1, 1, 0, -1]], \\quad \\alpha = 0.45",explanation:"Weight matrix decomposed into ternary values and scale."},{formula:"T \\cdot x = [0.5 + 0 - 0.8 + 0.1, \\; -0.5 - 0.3 + 0 - 0.1]",explanation:"No multiplications! Just additions/subtractions based on sign."},{formula:"T \\cdot x = [-0.2, -0.9]",explanation:"Raw ternary output computed with zero multiplies."},{formula:"Y = 0.45 \\times [-0.2, -0.9] = [-0.09, -0.405]",explanation:"Apply scale factor alpha. Only one multiply per output element."}],id:"example-ternary-matmul"}),e.jsx(o,{title:"Compute Savings from Ternary Weights",statement:"For a matrix multiplication $Y = WX$ with $W \\in \\mathbb{R}^{m \\times n}$, standard FP16 requires $2mn$ FLOPs. With ternary weights, this reduces to $mn$ additions plus $m$ scalar multiplications (for the scale factor), a theoretical $2\\times$ compute reduction per layer.",proof:"Each element of $T \\cdot x$ involves $n$ operations, each of which is either +x_j, -x_j, or 0 (no-op for zeros). With typical 30% sparsity in ternary matrices, the effective operation count is $\\sim 0.7mn$ additions + $m$ multiplies, compared to $2mn$ FLOPs for FP16 matmul.",id:"thm-ternary-compute"}),e.jsx(n,{title:"extreme_quantization.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

def ternarize(W, threshold_ratio=0.7):
    """Quantize weights to {-alpha, 0, +alpha}."""
    abs_W = W.abs()
    threshold = threshold_ratio * abs_W.mean()

    # Create ternary mask
    T = torch.zeros_like(W)
    T[W > threshold] = 1.0
    T[W < -threshold] = -1.0

    # Compute optimal scale factor
    mask = T != 0
    alpha = abs_W[mask].mean() if mask.any() else torch.tensor(1.0)

    return T, alpha

def binary_quantize(W):
    """Quantize weights to {-alpha, +alpha} (1-bit)."""
    alpha = W.abs().mean()
    B = torch.sign(W)
    B[B == 0] = 1.0  # No zeros in binary
    return B, alpha

# Compare quantization levels
torch.manual_seed(42)
W = torch.randn(1024, 1024) * 0.02
x = torch.randn(1024, 128)

# Ground truth
Y_true = W @ x

# Ternary (1.58-bit)
T, alpha_t = ternarize(W)
Y_ternary = alpha_t * (T @ x)
sparsity = (T == 0).float().mean()
print(f"Ternary: alpha={alpha_t:.4f}, sparsity={sparsity:.1%}")

# Binary (1-bit)
B, alpha_b = binary_quantize(W)
Y_binary = alpha_b * (B @ x)

# INT4 baseline
scale_4 = W.abs().max() / 7
W_int4 = torch.clamp(torch.round(W / scale_4), -8, 7) * scale_4
Y_int4 = W_int4 @ x

# Compare errors and sizes
for name, Y_q, bits in [("INT4", Y_int4, 4), ("Ternary", Y_ternary, 1.58),
                          ("Binary", Y_binary, 1)]:
    mse = ((Y_true - Y_q)**2).mean().item()
    size_mb = 1024 * 1024 * bits / 8 / 1e6
    print(f"{name:>8}: MSE={mse:.6f}, size={size_mb:.2f} MB "
          f"(vs {1024*1024*16/8/1e6:.2f} MB FP16)")

# BitNet-style linear layer
class BitLinear(nn.Module):
    """1.58-bit linear layer (BitNet b1.58 style)."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def forward(self, x):
        # Activation quantization: absmax to INT8
        x_scale = x.abs().max()
        x_q = (x / x_scale * 127).round().clamp(-128, 127)

        # Weight ternarization
        T, alpha = ternarize(self.weight)

        # Compute: only additions and subtractions
        y = alpha * (T @ (x_q.float())) * (x_scale / 127)
        return y

layer = BitLinear(4096, 4096)
test_input = torch.randn(1, 128, 4096)
output = layer(test_input)
print(f"\\nBitLinear output shape: {output.shape}")
print(f"Weight memory: {4096*4096*1.58/8/1e6:.1f} MB (vs {4096*4096*2/1e6:.1f} MB FP16)")`,id:"code-extreme-quant"}),e.jsx(t,{type:"historical",title:"The Path to 1-Bit LLMs",content:"BinaryConnect (2015) first showed binary weights could work for small networks. TWN (2016) introduced ternary weights. BitNet (2023) scaled binary quantization to Transformer LLMs. BitNet b1.58 (2024) demonstrated that 1.58-bit LLMs can match FP16 performance starting at 3B parameters, with dramatic energy and latency improvements.",id:"note-extreme-history"}),e.jsx(r,{title:"Hardware Support Required",content:"Extreme quantization only delivers speedups with specialized kernels or hardware. Standard GPU CUDA cores cannot efficiently exploit ternary arithmetic. Custom kernels (like those in llama.cpp for 2-4 bit) and upcoming hardware with native low-bit support are necessary to realize the theoretical gains.",id:"warning-hardware-support"})]})}const A=Object.freeze(Object.defineProperty({__proto__:null,default:h},Symbol.toStringTag,{value:"Module"}));function f(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Mamba: Selective State Space Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Mamba introduces a selective state space model (SSM) that achieves Transformer-quality language modeling with linear-time complexity in sequence length. Unlike attention's quadratic cost, Mamba processes sequences in O(n) time and O(1) memory per step during generation, enabling efficient handling of very long contexts."}),e.jsx(a,{title:"Continuous State Space Model",definition:"A state space model maps an input sequence $x(t)$ to output $y(t)$ through a latent state $h(t) \\in \\mathbb{R}^N$: $h'(t) = A h(t) + B x(t)$ and $y(t) = C h(t) + D x(t)$, where $A \\in \\mathbb{R}^{N \\times N}$ is the state matrix, $B \\in \\mathbb{R}^{N \\times 1}$ the input matrix, and $C \\in \\mathbb{R}^{1 \\times N}$ the output matrix.",notation:"Discretized with step size $\\Delta$: $\\bar{A} = \\exp(\\Delta A)$, $\\bar{B} = (\\Delta A)^{-1}(\\exp(\\Delta A) - I) \\cdot \\Delta B$. Recurrence: $h_k = \\bar{A} h_{k-1} + \\bar{B} x_k$, $y_k = C h_k$.",id:"def-ssm"}),e.jsx(a,{title:"Selective State Space (Mamba)",definition:"Mamba makes SSM parameters input-dependent: $B_k = s_B(x_k)$, $C_k = s_C(x_k)$, and $\\Delta_k = \\text{softplus}(s_\\Delta(x_k))$, where $s_B$, $s_C$, $s_\\Delta$ are learned linear projections. This selectivity allows the model to filter information based on content, analogous to attention's content-based routing.",id:"def-mamba-selective"}),e.jsx(i,{title:"SSM Recurrence Step",problem:"Given state h_0 = [0.5, -0.3], A_bar = [[0.9, 0.1], [0, 0.8]], B_bar = [0.2, 0.1], C = [1, 1], compute output for input x_1 = 3.0.",steps:[{formula:"h_1 = \\bar{A} h_0 + \\bar{B} x_1 = \\begin{bmatrix} 0.9(0.5)+0.1(-0.3) \\\\ 0(0.5)+0.8(-0.3) \\end{bmatrix} + 3.0 \\begin{bmatrix} 0.2 \\\\ 0.1 \\end{bmatrix}",explanation:"Apply state transition and input injection."},{formula:"h_1 = \\begin{bmatrix} 0.42 \\\\ -0.24 \\end{bmatrix} + \\begin{bmatrix} 0.6 \\\\ 0.3 \\end{bmatrix} = \\begin{bmatrix} 1.02 \\\\ 0.06 \\end{bmatrix}",explanation:"New hidden state combines memory and new input."},{formula:"y_1 = C h_1 = [1, 1] \\cdot [1.02, 0.06] = 1.08",explanation:"Output is a linear readout of the state."}],id:"example-ssm-recurrence"}),e.jsx(o,{title:"Linear-Time Sequence Processing",statement:"An SSM processes a sequence of length $L$ in $O(L \\cdot N)$ time and $O(N)$ memory during autoregressive generation, compared to $O(L^2 \\cdot d)$ time for self-attention. For training, Mamba uses a parallel scan algorithm achieving $O(L \\cdot N \\cdot \\log L)$ time.",proof:"Each recurrence step $h_k = \\bar{A}_k h_{k-1} + \\bar{B}_k x_k$ is $O(N)$ for state dimension $N$. Over $L$ steps: $O(LN)$. The parallel scan exploits associativity of the linear recurrence, enabling GPU-efficient parallel computation in $O(L \\log L)$ parallel time.",id:"thm-ssm-complexity"}),e.jsx(n,{title:"mamba_selective_ssm.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    """Simplified Mamba selective state space model block."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner
        )

        # SSM parameters: input-dependent (selective)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # B, C, delta
        self.dt_proj = nn.Linear(1, d_inner, bias=True)  # Expand delta

        # State matrix A (structured, not input-dependent)
        A = torch.arange(1, d_state + 1).float()
        self.A_log = nn.Parameter(torch.log(A))  # Learn in log space

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.d_state = d_state

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        batch, seq_len, _ = x.shape

        # Project and split into two paths
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        # 1D convolution for local context
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Compute input-dependent SSM parameters (SELECTIVE)
        x_ssm = self.x_proj(x_conv)
        B = x_ssm[:, :, :self.d_state]           # (batch, seq, d_state)
        C = x_ssm[:, :, self.d_state:2*self.d_state]
        delta = F.softplus(x_ssm[:, :, -1:])      # (batch, seq, 1)

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_state,) - negative for stability
        # Simplified: run SSM recurrence
        h = torch.zeros(batch, x_conv.shape[-1], self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            dt = delta[:, t, :]  # (batch, 1)
            A_bar = torch.exp(dt * A)  # (batch, d_state)
            B_bar = dt * B[:, t, :]
            h = A_bar.unsqueeze(1) * h + B_bar.unsqueeze(1) * x_conv[:, t:t+1, :].transpose(1, 2)
            y_t = (C[:, t, :].unsqueeze(2) * h).sum(dim=1)  # (batch, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)

        # Gate and project output
        y = y * F.silu(z)
        return self.out_proj(y)

# Benchmark: Mamba vs Attention scaling
model = SelectiveSSM(d_model=256, d_state=16)
for seq_len in [128, 512, 1024, 2048]:
    x = torch.randn(2, seq_len, 256)
    import time
    start = time.time()
    y = model(x)
    elapsed = (time.time() - start) * 1000
    print(f"Seq={seq_len:>5}: output={y.shape}, time={elapsed:.1f}ms")
    # Time grows linearly, not quadratically!`,id:"code-mamba"}),e.jsx(t,{type:"intuition",title:"Selectivity as Learned Gating",content:"In standard SSMs, B and C are fixed, so the model treats all inputs identically. Mamba's selectivity makes these input-dependent — the model can 'choose' what to store in its state and what to read out. This is conceptually similar to how attention selects which tokens to attend to, but achieved through recurrent dynamics instead of pairwise comparisons.",id:"note-selectivity-intuition"}),e.jsx(r,{title:"Trade-offs vs. Attention",content:"While Mamba is faster for long sequences, it cannot perform exact token-to-token lookback like attention. Tasks requiring precise copying or retrieval from distant context may still favor Transformers. Hybrid architectures (like Jamba) combine both mechanisms to get the best of both worlds.",id:"warning-mamba-tradeoffs"})]})}const B=Object.freeze(Object.defineProperty({__proto__:null,default:f},Symbol.toStringTag,{value:"Module"}));function _(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"RWKV: Linear Attention with RNN Efficiency"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"RWKV (Receptance Weighted Key Value) combines the training parallelism of Transformers with the inference efficiency of RNNs. It replaces quadratic attention with a linear mechanism that can be computed either as a parallel scan during training or as a constant-memory recurrence during inference."}),e.jsx(a,{title:"RWKV Time-Mixing",definition:"RWKV's time-mixing block computes a weighted combination of current and previous token features. For token at position $t$: $r_t = W_r \\cdot (\\mu_r x_t + (1 - \\mu_r) x_{t-1})$, $k_t = W_k \\cdot (\\mu_k x_t + (1 - \\mu_k) x_{t-1})$, $v_t = W_v \\cdot (\\mu_v x_t + (1 - \\mu_v) x_{t-1})$, where $\\mu$ are learned interpolation parameters.",notation:"Output: $o_t = \\sigma(r_t) \\odot \\frac{\\sum_{i=1}^{t} e^{-(t-i)w + k_i} v_i}{\\sum_{i=1}^{t} e^{-(t-i)w + k_i}}$ where $w$ is a channel-wise decay factor.",id:"def-rwkv-time-mixing"}),e.jsx(a,{title:"RWKV Channel-Mixing",definition:"The channel-mixing block (replacing FFN) uses gated linear units: $r_t = W_r \\cdot (\\mu_r x_t + (1-\\mu_r) x_{t-1})$, $k_t = W_k \\cdot (\\mu_k x_t + (1-\\mu_k) x_{t-1})$, output $= \\sigma(r_t) \\odot (W_v \\cdot \\max(k_t, 0)^2)$. The squared ReLU activation provides nonlinearity without traditional MLP structure.",id:"def-rwkv-channel-mixing"}),e.jsx(i,{title:"RWKV Recurrent Inference",problem:"Show how RWKV maintains constant memory during autoregressive generation.",steps:[{formula:"a_t = e^{-w} \\cdot a_{t-1} + e^{k_t} \\cdot v_t",explanation:"Running numerator: exponentially decayed sum of weighted values."},{formula:"b_t = e^{-w} \\cdot b_{t-1} + e^{k_t}",explanation:"Running denominator: exponentially decayed sum of weights."},{formula:"wkv_t = a_t / b_t",explanation:"Weighted key-value output, computed from two scalar states per channel."},{formula:"o_t = \\sigma(r_t) \\odot wkv_t",explanation:"Gate the output with receptance. Memory: O(d) regardless of sequence length."}],id:"example-rwkv-recurrence"}),e.jsx(n,{title:"rwkv_mechanism.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class RWKVTimeMixing(nn.Module):
    """RWKV time-mixing block with linear attention."""
    def __init__(self, d_model, layer_id=0, n_layers=12):
        super().__init__()
        self.d_model = d_model

        # Learnable interpolation factors (token shift mixing)
        ratio = layer_id / max(n_layers - 1, 1)
        self.time_mix_r = nn.Parameter(torch.ones(d_model) * (1 - ratio))
        self.time_mix_k = nn.Parameter(torch.ones(d_model) * (1 - ratio))
        self.time_mix_v = nn.Parameter(torch.ones(d_model) * (1 - ratio))

        # Channel-wise decay (learned)
        self.time_decay = nn.Parameter(torch.randn(d_model) * 0.1 - 5.0)
        self.time_first = nn.Parameter(torch.randn(d_model) * 0.1)

        # Projections
        self.W_r = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, state=None):
        """x: (batch, seq_len, d_model)"""
        B, T, C = x.shape

        # Token shift: mix current and previous token
        x_prev = torch.zeros_like(x[:, :1, :]) if state is None else state
        x_shifted = torch.cat([x_prev, x[:, :-1, :]], dim=1)

        r = self.W_r(x * self.time_mix_r + x_shifted * (1 - self.time_mix_r))
        k = self.W_k(x * self.time_mix_k + x_shifted * (1 - self.time_mix_k))
        v = self.W_v(x * self.time_mix_v + x_shifted * (1 - self.time_mix_v))

        # WKV computation (parallel mode for training)
        w = -torch.exp(self.time_decay)  # Negative decay
        u = self.time_first

        # Recurrent WKV (simple version)
        wkv = torch.zeros(B, C, device=x.device)
        a = torch.zeros(B, C, device=x.device)
        b = torch.zeros(B, C, device=x.device)
        outputs = []

        for t in range(T):
            kt, vt = k[:, t], v[:, t]
            # Numerically stable computation
            ww = w + kt
            p = torch.max(a, ww)
            e1 = torch.exp(a - p)
            e2 = torch.exp(ww - p)
            wkv = (e1 * a + e2 * vt) / (e1 * b + e2)
            # For next step: include bonus for first token
            qq = torch.max(w + a, u + kt)
            e1 = torch.exp(w + a - qq)
            e2 = torch.exp(u + kt - qq)
            a = qq + torch.log(e1 + e2)
            b = e1 * b + e2
            outputs.append(wkv)

        wkv_out = torch.stack(outputs, dim=1)

        # Receptance gating
        out = torch.sigmoid(r) * wkv_out
        return self.W_o(out), x[:, -1:, :]  # Return state for next call

# Compare memory usage: RWKV vs Attention
d = 512
seq_lengths = [256, 1024, 4096, 16384]
print("Memory comparison (relative to d_model):")
print(f"{'Seq Len':>8} {'Attention KV':>14} {'RWKV State':>12}")
for L in seq_lengths:
    attn_memory = L * d * 2  # K and V cache
    rwkv_memory = d * 2      # Just a and b states (constant!)
    print(f"{L:>8} {attn_memory:>14,} {rwkv_memory:>12,}")`,id:"code-rwkv"}),e.jsx(t,{type:"historical",title:"RWKV Evolution",content:"RWKV was created by Bo Peng as an open-source project. RWKV-4 (2023) demonstrated competitive performance with Transformers up to 14B parameters. RWKV-5 (Eagle) and RWKV-6 (Finch) introduced multi-headed variants and improved the WKV mechanism. The model is fully open-source and has an active community.",id:"note-rwkv-history"}),e.jsx(t,{type:"tip",title:"When to Use RWKV",content:"RWKV excels in scenarios with very long sequences and streaming inference (chatbots, real-time processing). Its constant memory footprint during generation means a 14B RWKV model uses the same memory generating the 100th token as the 100,000th. Choose attention-based models when precise recall over long contexts is critical.",id:"note-rwkv-when"}),e.jsx(r,{title:"Finite State Capacity",content:"RWKV's recurrent state has fixed capacity (d_model floats). As sequences grow very long, earlier information is exponentially decayed. Unlike attention's KV cache which grows with context, RWKV must compress all history into a fixed-size state, which can lose fine-grained details from far back.",id:"warning-rwkv-state"})]})}const P=Object.freeze(Object.defineProperty({__proto__:null,default:_},Symbol.toStringTag,{value:"Module"}));function g(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Hyena Hierarchy: Long Convolutions for Attention-Free Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Hyena replaces self-attention with a hierarchy of long convolutions and element-wise gating. It achieves sub-quadratic complexity while maintaining the expressiveness needed for language modeling. The key insight is that interleaving data-controlled gating with long convolution filters can implicitly learn attention-like patterns."}),e.jsx(a,{title:"Hyena Operator",definition:"The Hyena operator of order $N$ applies $N$ stages of gated long convolution: $y = h_N * (x_N \\odot (h_{N-1} * (x_{N-1} \\odot \\cdots (h_1 * (x_1 \\odot v)) \\cdots)))$ where $h_i$ are implicitly parameterized long convolution filters, $x_i$ are data-dependent projections of the input, $*$ is convolution, and $\\odot$ is element-wise multiplication.",notation:"Complexity: $O(N \\cdot L \\log L)$ using FFT-based convolution, compared to $O(L^2 d)$ for attention.",id:"def-hyena"}),e.jsx(i,{title:"Hyena Order-2 Computation",problem:"Trace through a Hyena-2 block for a sequence of length 4 with d_model=2.",steps:[{formula:"v, x_1, x_2 = \\text{Linear}(\\text{input}) \\quad \\text{(3 projections)}",explanation:"Project input into value v and two gating signals, analogous to Q/K/V."},{formula:"z_1 = h_1 * (x_1 \\odot v)",explanation:"First stage: gate v by x_1, then apply long convolution filter h_1."},{formula:"z_2 = h_2 * (x_2 \\odot z_1)",explanation:"Second stage: gate first output by x_2, apply another long convolution."},{formula:"y = \\text{Linear}(z_2)",explanation:"Output projection. The nested gating creates data-dependent mixing of tokens."}],id:"example-hyena"}),e.jsx(n,{title:"hyena_operator.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ImplicitFilter(nn.Module):
    """Parameterize convolution filter implicitly with a small MLP."""
    def __init__(self, d_model, seq_len, n_hidden=64):
        super().__init__()
        self.seq_len = seq_len
        # Small MLP maps position -> filter value
        self.mlp = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, d_model),
        )
        # Exponential decay for positional bias
        self.decay = nn.Parameter(torch.linspace(-1, -5, d_model))

    def forward(self):
        t = torch.linspace(0, 1, self.seq_len).unsqueeze(-1).to(self.decay.device)
        h = self.mlp(t)  # (seq_len, d_model)
        # Apply exponential decay window
        window = torch.exp(self.decay.unsqueeze(0) * t * self.seq_len)
        return h * window

def fft_conv(u, h):
    """Efficient convolution via FFT: O(L log L) instead of O(L^2)."""
    L = u.shape[-2]
    # Pad to avoid circular convolution artifacts
    fft_len = 2 * L
    U = torch.fft.rfft(u, n=fft_len, dim=-2)
    H = torch.fft.rfft(h, n=fft_len, dim=-2)
    Y = U * H
    y = torch.fft.irfft(Y, n=fft_len, dim=-2)[:, :L, :]
    return y

class HyenaBlock(nn.Module):
    """Simplified Hyena operator (order 2)."""
    def __init__(self, d_model, seq_len, order=2):
        super().__init__()
        self.order = order
        # Input projections: value + order gating signals
        self.in_proj = nn.Linear(d_model, d_model * (order + 1))
        # Implicit convolution filters
        self.filters = nn.ModuleList([
            ImplicitFilter(d_model, seq_len) for _ in range(order)
        ])
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        # Project to value + gating signals
        projections = self.in_proj(x).chunk(self.order + 1, dim=-1)
        v = projections[0]
        gates = projections[1:]

        # Iterative gated convolution
        z = v
        for i in range(self.order):
            z = gates[i] * z  # Element-wise gating
            h = self.filters[i]()  # (seq_len, d_model)
            z = fft_conv(z, h)     # Long convolution via FFT

        return self.out_proj(z)

# Benchmark: compare FLOPs scaling
d_model = 256
for L in [256, 1024, 4096, 16384]:
    # Attention FLOPs: O(L^2 * d)
    attn_flops = 2 * L * L * d_model
    # Hyena FLOPs: O(order * L * log(L) * d) for FFT convolution
    hyena_flops = 2 * 2 * L * math.log2(L) * d_model
    ratio = attn_flops / hyena_flops
    print(f"L={L:>6}: Attention={attn_flops/1e6:.0f}M, "
          f"Hyena={hyena_flops/1e6:.0f}M, speedup={ratio:.1f}x")

# Test forward pass
block = HyenaBlock(d_model=256, seq_len=1024, order=2)
x = torch.randn(2, 1024, 256)
y = block(x)
print(f"\\nInput: {x.shape} -> Output: {y.shape}")`,id:"code-hyena"}),e.jsx(t,{type:"intuition",title:"Convolution as Implicit Attention",content:"A long convolution filter h can learn to weight nearby tokens heavily and distant tokens weakly — similar to a fixed attention pattern. By stacking multiple gated convolutions, Hyena builds data-dependent mixing that approximates the flexibility of attention. The gating signals act like learned soft masks that select which information the convolution should propagate.",id:"note-hyena-intuition"}),e.jsx(t,{type:"historical",title:"From S4 to Hyena",content:"Hyena (Poli et al., 2023) builds on the S4 line of work (Gu et al., 2022). While S4 used structured state spaces for long convolutions, Hyena uses implicitly parameterized filters via small MLPs. StripedHyena scaled this to 7B parameters, showing competitive results with similarly sized Transformers on language tasks.",id:"note-hyena-history"}),e.jsx(r,{title:"Fixed Sequence Length at Init",content:"The implicit filter is parameterized for a fixed maximum sequence length. Handling variable-length sequences requires either padding (wasteful) or filter interpolation. This is less flexible than attention, which naturally handles any sequence length. Newer variants address this with position-independent filter parameterizations.",id:"warning-hyena-seqlen"})]})}const W=Object.freeze(Object.defineProperty({__proto__:null,default:g},Symbol.toStringTag,{value:"Module"}));function x(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Jamba: Hybrid Attention-Mamba Architecture"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Jamba combines Transformer attention layers with Mamba SSM layers in a single model, getting the best of both worlds: attention's precise token retrieval and Mamba's efficient long-range processing. It further incorporates Mixture-of-Experts (MoE) to increase model capacity without proportionally increasing compute."}),e.jsx(a,{title:"Jamba Architecture",definition:"Jamba interleaves Mamba layers, attention layers, and MoE-FFN layers in a repeating pattern. A typical block ratio is 7:1 (7 Mamba layers per 1 attention layer). Each attention layer uses grouped-query attention (GQA), and the FFN in select layers is replaced with a MoE layer containing $E$ experts with top-$k$ routing.",notation:"For a 52B total parameter model with 12B active: $E=16$ experts, $k=2$ active per token. Active FLOPs match a 12B dense model while having 52B total capacity.",id:"def-jamba"}),e.jsx(i,{title:"Jamba Layer Configuration",problem:"Design a Jamba model with 32 layers, 7:1 Mamba-to-attention ratio, and MoE on every other Mamba layer.",steps:[{formula:"\\text{Attention layers: } \\{8, 16, 24, 32\\} \\quad (4 \\text{ layers})",explanation:"Place attention every 8th layer for global token interaction."},{formula:"\\text{Mamba layers: remaining 28 layers}",explanation:"Mamba handles the bulk of processing with linear complexity."},{formula:"\\text{MoE-FFN: layers } \\{2, 4, 6, 10, 12, 14, ...\\}",explanation:"Every other Mamba layer uses MoE-FFN instead of dense FFN for capacity."},{formula:"\\text{KV cache: only 4 layers} \\rightarrow 87.5\\% \\text{ memory reduction}",explanation:"Only attention layers need KV cache. Mamba uses fixed-size recurrent state."}],id:"example-jamba-config"}),e.jsx(n,{title:"jamba_hybrid_model.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMambaLayer(nn.Module):
    """Simplified Mamba layer for the hybrid model."""
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(d_model, d_model, 4, padding=3, groups=d_model)
        self.ssm_proj = nn.Linear(d_model, d_state * 2 + 1)
        self.out_proj = nn.Linear(d_model, d_model)
        self.d_state = d_state

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        x_conv = self.conv(x_inner.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        x_conv = F.silu(x_conv)
        out = x_conv * F.silu(z)
        return residual + self.out_proj(out)

class SimpleAttentionLayer(nn.Module):
    """Standard attention layer with GQA."""
    def __init__(self, d_model, n_heads=8, n_kv_heads=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.norm(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        return x + self.ffn(self.norm2(x))

class MoEFFN(nn.Module):
    """Mixture-of-Experts FFN layer."""
    def __init__(self, d_model, n_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            ) for _ in range(n_experts)
        ])
        self.gate = nn.Linear(d_model, n_experts)
        self.top_k = top_k

    def forward(self, x):
        # Route tokens to top-k experts
        scores = self.gate(x)
        topk_vals, topk_idx = scores.topk(self.top_k, dim=-1)
        weights = F.softmax(topk_vals, dim=-1)

        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x[mask])
                weight = weights[topk_idx == i].unsqueeze(-1)
                output[mask] += expert_out * weight[:expert_out.shape[0]]
        return output

class JambaModel(nn.Module):
    """Hybrid Jamba architecture."""
    def __init__(self, d_model=512, n_layers=16, attn_every=8, moe_every=2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            if (i + 1) % attn_every == 0:
                layers.append(SimpleAttentionLayer(d_model))
            else:
                layers.append(SimpleMambaLayer(d_model))
        self.layers = nn.ModuleList(layers)

        # Count layer types
        n_attn = sum(1 for l in layers if isinstance(l, SimpleAttentionLayer))
        n_mamba = sum(1 for l in layers if isinstance(l, SimpleMambaLayer))
        print(f"Jamba: {n_attn} attention + {n_mamba} Mamba layers")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Build and analyze
model = JambaModel(d_model=512, n_layers=16, attn_every=8)
x = torch.randn(2, 1024, 512)
y = model(x)
print(f"Output shape: {y.shape}")

# Memory analysis: KV cache savings
seq_len, d, n_kv_heads, head_dim = 256_000, 512, 2, 64
full_attn_kv = 16 * 2 * seq_len * n_kv_heads * head_dim * 2  # All 16 layers
jamba_kv = 2 * 2 * seq_len * n_kv_heads * head_dim * 2        # Only 2 attn layers
mamba_state = 14 * d * 16 * 4  # 14 Mamba layers, d_state=16
print(f"\\nKV cache at {seq_len:,} tokens:")
print(f"  Full attention: {full_attn_kv / 1e9:.2f} GB")
print(f"  Jamba (attn):   {jamba_kv / 1e9:.2f} GB")
print(f"  Jamba (mamba):  {mamba_state / 1e6:.2f} MB (constant!)")
print(f"  Total savings:  {(1 - (jamba_kv + mamba_state) / full_attn_kv):.0%}")`,id:"code-jamba"}),e.jsx(t,{type:"intuition",title:"Why Hybrid Works",content:"Mamba layers efficiently propagate information across long contexts through their recurrent state, handling the bulk of computation cheaply. The sparse attention layers act as 'checkpoints' where the model can perform precise token-to-token operations like copying, retrieval, and comparison. This division of labor gives near-linear scaling with periodic exact-attention capabilities.",id:"note-hybrid-intuition"}),e.jsx(t,{type:"note",title:"Jamba's Impact",content:"AI21's Jamba (2024) was the first production-quality hybrid SSM-attention model. At 52B total parameters (12B active), it fits a 256K context in a single 80GB GPU — impossible for a pure attention model of similar quality. This demonstrated that hybrid architectures are a practical path to efficient long-context LLMs.",id:"note-jamba-impact"}),e.jsx(r,{title:"Architecture Design Sensitivity",content:"The ratio of attention to Mamba layers, placement of MoE layers, and expert count all significantly affect performance. Too few attention layers degrades retrieval tasks; too many negates the efficiency gains. The optimal configuration depends on the target task mix and hardware constraints.",id:"warning-jamba-design"})]})}const F=Object.freeze(Object.defineProperty({__proto__:null,default:x},Symbol.toStringTag,{value:"Module"}));function y(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Small Language Models: Phi, Gemma, and Beyond"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Small language models (SLMs) with 1-3 billion parameters challenge the assumption that scale is everything. Through careful data curation, architectural innovations, and training recipes, models like Phi-2, Gemma-2B, and StableLM demonstrate that compact models can match or exceed much larger models on many benchmarks."}),e.jsx(a,{title:"Data Quality Over Quantity",definition:"The Phi series demonstrates that training data quality dominates model size for smaller models. Phi-1 (1.3B) was trained on 'textbook quality' data — a curated mix of filtered web data and synthetically generated textbook-style content. The key insight: $\\text{Performance} \\propto f(\\text{data quality}) \\cdot g(\\text{model size})$ where data quality has a larger effect at small scales.",id:"def-data-quality"}),e.jsx(i,{title:"Model Size and Memory Calculations",problem:"Compare memory requirements for running Phi-2 (2.7B), Gemma-2B, and LLaMA-7B.",steps:[{formula:"\\text{Phi-2: } 2.7B \\times 2 \\text{ bytes} = 5.4\\text{ GB (FP16)}",explanation:"At FP16, each parameter uses 2 bytes."},{formula:"\\text{Phi-2 INT4: } 2.7B \\times 0.5 = 1.35\\text{ GB}",explanation:"Quantized to 4-bit, Phi-2 fits comfortably in phone RAM."},{formula:"\\text{Gemma-2B INT4: } 2.0B \\times 0.5 = 1.0\\text{ GB}",explanation:"Even smaller, with architecture optimized for the 2B scale."},{formula:"\\text{LLaMA-7B INT4: } 7.0B \\times 0.5 = 3.5\\text{ GB}",explanation:"For comparison, LLaMA-7B needs 2.5x more memory even at INT4."}],id:"example-model-sizes"}),e.jsx(n,{title:"small_model_comparison.py",code:`import torch

# Model configurations for popular small LLMs
models = {
    "Phi-2 (2.7B)": {
        "layers": 32, "d_model": 2560, "heads": 32,
        "vocab": 51200, "context": 2048,
    },
    "Gemma-2B": {
        "layers": 18, "d_model": 2048, "heads": 8,
        "vocab": 256000, "context": 8192,
    },
    "StableLM-2-1.6B": {
        "layers": 24, "d_model": 2048, "heads": 32,
        "vocab": 100289, "context": 4096,
    },
    "TinyLlama-1.1B": {
        "layers": 22, "d_model": 2048, "heads": 32,
        "vocab": 32000, "context": 2048,
    },
    "LLaMA-7B (baseline)": {
        "layers": 32, "d_model": 4096, "heads": 32,
        "vocab": 32000, "context": 4096,
    },
}

def count_params(cfg):
    L, d, V = cfg["layers"], cfg["d_model"], cfg["vocab"]
    # Attention: Q, K, V, O
    attn = 4 * d * d * L
    # FFN: gate + up + down (SwiGLU style: 3 matrices of d x 4d/3*2)
    ffn = 3 * d * int(d * 8/3) * L  # Approximate SwiGLU
    # Embeddings (tied)
    embed = V * d
    # Layer norms
    norms = 2 * d * L
    return attn + ffn + embed + norms

def kv_cache_size(cfg, seq_len, n_kv_heads=None, dtype_bytes=2):
    d_head = cfg["d_model"] // cfg["heads"]
    kv_heads = n_kv_heads or cfg["heads"]
    # 2 (K+V) * layers * seq_len * kv_heads * d_head
    return 2 * cfg["layers"] * seq_len * kv_heads * d_head * dtype_bytes

print(f"{'Model':<22} {'Params':>10} {'FP16 GB':>8} {'INT4 GB':>8} {'KV@2K':>8}")
print("-" * 60)
for name, cfg in models.items():
    params = count_params(cfg)
    fp16_gb = params * 2 / 1e9
    int4_gb = params * 0.5 / 1e9
    kv_gb = kv_cache_size(cfg, 2048) / 1e9
    print(f"{name:<22} {params/1e9:>9.1f}B {fp16_gb:>7.1f} {int4_gb:>7.2f} {kv_gb:>7.3f}")

# Training data comparison
training_data = {
    "Phi-1 (1.3B)": "6B tokens (textbook quality + CodeExercises)",
    "Phi-2 (2.7B)": "1.4T tokens (filtered web + synthetic textbooks)",
    "Gemma-2B": "2T tokens (web, code, math)",
    "TinyLlama": "3T tokens (SlimPajama + StarCoder)",
    "LLaMA-7B": "1T tokens (CommonCrawl, C4, etc.)",
}
print("\\nTraining data:")
for name, data in training_data.items():
    print(f"  {name}: {data}")`,id:"code-small-models"}),e.jsx(t,{type:"intuition",title:"Why Small Models Can Punch Above Their Weight",content:"Large models trained on internet-scale data spend capacity learning low-quality patterns, duplicated knowledge, and noise. Small models trained on curated data focus their limited capacity on high-value knowledge. This is why Phi-2 (2.7B) can match LLaMA-7B on reasoning benchmarks — it learned more efficiently from better data.",id:"note-small-model-intuition"}),e.jsx(t,{type:"note",title:"Key Architectural Choices for Small Models",content:"Successful SLMs use: (1) Grouped-query attention (GQA) to reduce KV cache, (2) SwiGLU or GeGLU activations for better parameter efficiency, (3) RoPE positional embeddings for context extension, (4) deeper-and-narrower architectures over shallow-and-wide ones. Gemma-2B uses only 8 KV heads (vs 8 query heads) to minimize serving cost.",id:"note-arch-choices"}),e.jsx(r,{title:"Benchmark Caveats",content:"Small models often excel on academic benchmarks but struggle with complex multi-step reasoning, following nuanced instructions, and generating long coherent text. Their reduced capacity means they cannot store as much factual knowledge. Always evaluate on your specific use case, not just headline benchmark numbers.",id:"warning-benchmarks"})]})}const E=Object.freeze(Object.defineProperty({__proto__:null,default:y},Symbol.toStringTag,{value:"Module"}));function b(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Mobile Deployment: ONNX, Core ML, and Runtime Optimization"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Deploying LLMs on mobile and edge devices requires converting models to optimized formats (ONNX, Core ML, TensorFlow Lite) and leveraging hardware-specific acceleration. The goal is to achieve acceptable latency (under 50ms per token) within tight memory budgets (4-8 GB RAM shared with the OS and other apps)."}),e.jsx(a,{title:"ONNX Runtime",definition:"ONNX (Open Neural Network Exchange) is a hardware-agnostic model format that represents computation as a directed graph. ONNX Runtime (ORT) executes these graphs with optimizations including operator fusion, constant folding, and hardware-specific kernel selection. For LLMs, ORT provides quantized inference, KV cache management, and beam search as built-in operators.",id:"def-onnx"}),e.jsx(a,{title:"Core ML",definition:"Core ML is Apple's on-device ML framework optimizing for the Apple Neural Engine (ANE), GPU, and CPU. It supports INT4 weight-only quantization, attention caching, and dynamic shapes. A 3B model quantized to INT4 can run at 15-30 tokens/second on iPhone 15 Pro using the ANE's 35 TOPS of compute.",id:"def-coreml"}),e.jsx(i,{title:"Mobile Memory Budget",problem:"Determine if a 2.7B model (INT4) can run on a device with 6 GB RAM, 2 GB reserved for OS.",steps:[{formula:"\\text{Available RAM} = 6 - 2 = 4\\text{ GB}",explanation:"OS and background apps reserve roughly 2 GB."},{formula:"\\text{Model weights} = 2.7B \\times 0.5 = 1.35\\text{ GB}",explanation:"INT4 quantization: 0.5 bytes per parameter."},{formula:"\\text{KV cache (2K ctx)} \\approx 0.15\\text{ GB}",explanation:"KV cache for 2048 tokens with GQA."},{formula:"\\text{Runtime overhead} \\approx 0.3\\text{ GB}",explanation:"Activations, framework overhead, and intermediate buffers."},{formula:"\\text{Total} = 1.35 + 0.15 + 0.3 = 1.8\\text{ GB} < 4\\text{ GB} \\checkmark",explanation:"The model fits with 2.2 GB to spare."}],id:"example-mobile-memory"}),e.jsx(n,{title:"onnx_export_and_optimize.py",code:`import torch
import torch.nn as nn

# Step 1: Define a small model for demonstration
class TinyLM(nn.Module):
    def __init__(self, vocab=32000, d=512, layers=4, heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads, dim_feedforward=d*4, batch_first=True)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight  # Tie weights

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

model = TinyLM()
model.eval()
params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {params:,}")

# Step 2: Export to ONNX
dummy_input = torch.randint(0, 32000, (1, 128))

torch.onnx.export(
    model,
    dummy_input,
    "tiny_lm.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"},
                  "logits": {0: "batch", 1: "seq_len"}},
    opset_version=17,
)
print("Exported to ONNX")

# Step 3: Optimize with ONNX Runtime
# pip install onnxruntime
import onnxruntime as ort

# Create optimized session
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_filepath = "tiny_lm_optimized.onnx"

session = ort.InferenceSession("tiny_lm.onnx", sess_options)

# Benchmark inference
import time
input_np = dummy_input.numpy()
times = []
for _ in range(10):
    start = time.time()
    outputs = session.run(None, {"input_ids": input_np})
    times.append((time.time() - start) * 1000)

print(f"ONNX Runtime: {sum(times)/len(times):.1f}ms avg ({len(times)} runs)")
print(f"Output shape: {outputs[0].shape}")

# Step 4: Quantize ONNX model to INT8
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "tiny_lm.onnx",
    "tiny_lm_int8.onnx",
    weight_type=QuantType.QInt8,
)

import os
original_size = os.path.getsize("tiny_lm.onnx") / 1e6
quant_size = os.path.getsize("tiny_lm_int8.onnx") / 1e6
print(f"\\nOriginal: {original_size:.1f} MB")
print(f"INT8:     {quant_size:.1f} MB ({original_size/quant_size:.1f}x smaller)")`,id:"code-onnx-deploy"}),e.jsx(t,{type:"tip",title:"Deployment Framework Comparison",content:"ONNX Runtime: best cross-platform support, works on Windows/Linux/Android/iOS. Core ML: best on Apple devices (ANE acceleration). TensorFlow Lite: good for Android with GPU delegate. llama.cpp: excellent for LLMs specifically, supports many quantization formats, runs on CPU with SIMD acceleration. MLC LLM: compiles models for multiple backends with near-optimal performance.",id:"note-framework-comparison"}),e.jsx(t,{type:"note",title:"Hardware Acceleration on Mobile",content:"Modern phones have dedicated ML accelerators: Apple Neural Engine (35 TOPS on A17), Qualcomm Hexagon NPU (45 TOPS on Snapdragon 8 Gen 3), and Samsung Exynos NPU. These achieve 5-10x better performance per watt than running on the mobile GPU or CPU. Framework support for these accelerators varies.",id:"note-hardware-accel"}),e.jsx(r,{title:"Thermal Throttling",content:"Mobile devices throttle performance under sustained load to prevent overheating. An LLM generating tokens continuously can trigger throttling within 30-60 seconds, reducing speed by 30-50%. Design for burst inference (short responses) or implement adaptive batch sizing that backs off when thermal state is elevated.",id:"warning-thermal"})]})}const O=Object.freeze(Object.defineProperty({__proto__:null,default:b},Symbol.toStringTag,{value:"Module"}));function w(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Speculative Decoding: Draft and Verify"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Speculative decoding accelerates autoregressive generation by using a fast draft model to propose multiple tokens, then verifying them in parallel with the full target model. Since verification of K tokens costs roughly the same as generating one token (a single forward pass), accepted speculations provide near-free speedup."}),e.jsx(a,{title:"Speculative Decoding",definition:"Given a target model $M_p$ and a faster draft model $M_q$, speculative decoding: (1) generates $K$ draft tokens from $M_q$, (2) scores all $K$ tokens in a single forward pass of $M_p$, and (3) accepts token $i$ with probability $\\min\\left(1, \\frac{p(x_i)}{q(x_i)}\\right)$. Rejected tokens are resampled from the adjusted distribution $\\text{norm}(\\max(0, p - q))$.",notation:"Expected tokens per step: $\\frac{1 - \\alpha^{K+1}}{1 - \\alpha}$ where $\\alpha$ is the average acceptance rate. Speedup $\\approx \\frac{\\text{target model time}}{(\\text{draft time} \\times K + \\text{verify time})} \\times \\text{accepted tokens}$.",id:"def-speculative"}),e.jsx(o,{title:"Lossless Speculative Decoding",statement:"Speculative decoding with rejection sampling produces the exact same output distribution as standard autoregressive decoding from the target model $M_p$. It is a lossless acceleration technique.",proof:"At each position, the acceptance criterion $\\min(1, p(x)/q(x))$ combined with the rejection distribution $\\text{norm}(\\max(0, p - q))$ ensures the marginal probability of each token equals $p(x)$. This follows from the standard rejection sampling proof: the probability of accepting $x$ sampled from $q$ is $q(x) \\cdot \\min(1, p(x)/q(x))$, and the resampling corrects for rejected proposals.",id:"thm-speculative-lossless"}),e.jsx(i,{title:"Speculative Decoding Example",problem:"Draft model proposes 4 tokens: ['The', 'cat', 'sat', 'down']. Target model verifies each. Show the acceptance process.",steps:[{formula:"q = [0.3, 0.4, 0.2, 0.3], \\quad p = [0.4, 0.3, 0.25, 0.1]",explanation:"Draft (q) and target (p) probabilities for each proposed token."},{formula:"\\text{Token 1 (The): } \\min(1, 0.4/0.3) = 1.0 \\rightarrow \\text{ACCEPT}",explanation:"Target assigns higher probability, always accepted."},{formula:"\\text{Token 2 (cat): } \\min(1, 0.3/0.4) = 0.75 \\rightarrow r=0.5 < 0.75 \\rightarrow \\text{ACCEPT}",explanation:"Accepted with 75% probability (random draw was favorable)."},{formula:"\\text{Token 3 (sat): } \\min(1, 0.25/0.2) = 1.0 \\rightarrow \\text{ACCEPT}",explanation:"Target agrees with draft, accepted."},{formula:"\\text{Token 4 (down): } \\min(1, 0.1/0.3) = 0.33 \\rightarrow r=0.8 > 0.33 \\rightarrow \\text{REJECT}",explanation:"Target disagrees. Resample from norm(max(0, p - q)). Got 3 tokens for free!"}],id:"example-speculative"}),e.jsx(n,{title:"speculative_decoding.py",code:`import torch
import torch.nn.functional as F
import time

def speculative_decode(target_model, draft_model, prompt_ids,
                       max_tokens=50, K=4, temperature=1.0):
    """Speculative decoding with rejection sampling."""
    generated = prompt_ids.clone()

    tokens_generated = 0
    draft_calls = 0
    target_calls = 0

    while tokens_generated < max_tokens:
        # Step 1: Generate K draft tokens
        draft_ids = generated.clone()
        draft_probs = []
        for _ in range(K):
            with torch.no_grad():
                logits = draft_model(draft_ids)[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                token = torch.multinomial(probs, 1)
                draft_probs.append(probs)
                draft_ids = torch.cat([draft_ids, token], dim=-1)
            draft_calls += 1

        # Step 2: Verify all K tokens in ONE forward pass of target
        with torch.no_grad():
            # Target model processes prompt + all K draft tokens
            target_logits = target_model(draft_ids)
            target_calls += 1

        # Step 3: Accept/reject each draft token
        n_accepted = 0
        for i in range(K):
            pos = generated.shape[1] + i
            target_probs = F.softmax(target_logits[:, pos - 1, :] / temperature, dim=-1)
            draft_token = draft_ids[:, pos]

            p = target_probs[0, draft_token[0]]
            q = draft_probs[i][0, draft_token[0]]

            # Rejection sampling
            if torch.rand(1).item() < min(1.0, (p / q).item()):
                generated = torch.cat([generated, draft_token.unsqueeze(-1)], dim=-1)
                n_accepted += 1
            else:
                # Resample from adjusted distribution
                adjusted = torch.clamp(target_probs - draft_probs[i], min=0)
                adjusted = adjusted / adjusted.sum()
                new_token = torch.multinomial(adjusted, 1)
                generated = torch.cat([generated, new_token], dim=-1)
                n_accepted += 1
                break  # Stop verifying after first rejection

        tokens_generated += n_accepted

    acceptance_rate = tokens_generated / (draft_calls + 0.001)
    return generated, {
        "tokens": tokens_generated,
        "draft_calls": draft_calls,
        "target_calls": target_calls,
        "avg_accepted_per_step": acceptance_rate,
    }

# Simulate with mock models (same interface, different speeds)
class MockModel(torch.nn.Module):
    def __init__(self, vocab=1000, d=128, slow=False):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d)
        size = d * 4 if slow else d
        self.proj = torch.nn.Linear(d, size)
        self.head = torch.nn.Linear(size, vocab)

    def forward(self, x):
        h = self.embed(x).mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1)
        return self.head(F.relu(self.proj(h)))

target = MockModel(slow=True)
draft = MockModel(slow=False)

prompt = torch.randint(0, 1000, (1, 10))
output, stats = speculative_decode(target, draft, prompt, max_tokens=20, K=4)

print(f"Generated {stats['tokens']} tokens")
print(f"Draft model calls: {stats['draft_calls']}")
print(f"Target model calls: {stats['target_calls']}")
print(f"Avg accepted per verify: {stats['avg_accepted_per_step']:.2f}")
print(f"Theoretical speedup: {stats['tokens'] / stats['target_calls']:.1f}x")`,id:"code-speculative"}),e.jsx(t,{type:"tip",title:"Choosing a Draft Model",content:"The ideal draft model is 5-10x faster than the target with >70% acceptance rate. Options include: (1) a smaller model from the same family (e.g., LLaMA-7B drafting for LLaMA-70B), (2) a quantized version of the target, (3) a distilled student model, or (4) a pruned version with fewer layers. Self-speculative decoding skips layers in the target model itself as the draft.",id:"note-draft-choice"}),e.jsx(r,{title:"Diminishing Returns",content:"Increasing K (speculation length) has diminishing returns because acceptance probability drops exponentially with sequence length. If the acceptance rate is 70%, the probability of all K=8 tokens being accepted is 0.7^8 = 5.7%. K=4-5 is typically optimal. Also, speculative decoding helps most when the target model is memory-bound (large batch sizes reduce the benefit).",id:"warning-diminishing"})]})}const C=Object.freeze(Object.defineProperty({__proto__:null,default:w},Symbol.toStringTag,{value:"Module"}));function v(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Latency Optimization: Batching, Caching, and Scheduling"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Serving LLMs efficiently at scale requires optimizing beyond the model itself. Key-value caching avoids redundant computation, continuous batching maximizes GPU utilization, and memory management strategies like PagedAttention eliminate fragmentation. Together, these techniques can improve serving throughput by 10-20x."}),e.jsx(a,{title:"KV Cache",definition:"During autoregressive generation, the KV cache stores the key and value tensors from all previous tokens, avoiding recomputation. For each new token, only one new K and V vector per layer is computed and appended. Cache size: $2 \\times L \\times n_{kv} \\times d_h \\times T$ values, where $L$ is layers, $n_{kv}$ is KV heads, $d_h$ is head dimension, and $T$ is sequence length.",notation:"Without KV cache: generating $T$ tokens costs $O(T^2 \\cdot L \\cdot d)$ FLOPs. With KV cache: $O(T \\cdot L \\cdot d)$ — a $T\\times$ speedup.",id:"def-kv-cache"}),e.jsx(a,{title:"Continuous Batching",definition:"Continuous (or inflight) batching dynamically adds and removes requests from a running batch as individual sequences finish. Unlike static batching (which waits for the longest sequence), continuous batching immediately fills freed slots with new requests, achieving near-100% GPU utilization.",id:"def-continuous-batching"}),e.jsx(i,{title:"KV Cache Memory Calculation",problem:"Calculate KV cache memory for a 7B model serving 32 concurrent users with 4K context.",steps:[{formula:"\\text{LLaMA-7B: } L=32, n_{kv}=32, d_h=128",explanation:"Model configuration: 32 layers, 32 KV heads, head dim 128."},{formula:"\\text{Per-token KV} = 2 \\times 32 \\times 32 \\times 128 \\times 2 = 524\\text{ KB}",explanation:"2 (K+V) * layers * heads * dim * 2 bytes (FP16)."},{formula:"\\text{Per-user (4K ctx)} = 524\\text{ KB} \\times 4096 = 2.1\\text{ GB}",explanation:"Each user with full 4K context needs 2.1 GB of KV cache."},{formula:"\\text{32 users} = 2.1 \\times 32 = 67.1\\text{ GB}",explanation:"KV cache alone exceeds one 80GB GPU! This is why GQA and PagedAttention matter."}],id:"example-kv-cache-memory"}),e.jsx(n,{title:"serving_optimizations.py",code:`import torch
import time
from collections import deque

class KVCache:
    """Simple KV cache implementation."""
    def __init__(self, n_layers, n_heads, head_dim, max_seq_len, dtype=torch.float16):
        self.n_layers = n_layers
        self.cache_k = torch.zeros(n_layers, max_seq_len, n_heads, head_dim, dtype=dtype)
        self.cache_v = torch.zeros(n_layers, max_seq_len, n_heads, head_dim, dtype=dtype)
        self.seq_len = 0

    def update(self, layer_idx, new_k, new_v):
        """Append new K, V vectors for one token."""
        self.cache_k[layer_idx, self.seq_len] = new_k
        self.cache_v[layer_idx, self.seq_len] = new_v

    def get(self, layer_idx):
        """Get all cached K, V up to current sequence length."""
        return (self.cache_k[layer_idx, :self.seq_len + 1],
                self.cache_v[layer_idx, :self.seq_len + 1])

    def advance(self):
        self.seq_len += 1

    def memory_bytes(self):
        return self.cache_k.numel() * 2 + self.cache_v.numel() * 2  # FP16

# Memory analysis for different configurations
configs = {
    "LLaMA-7B": {"layers": 32, "kv_heads": 32, "head_dim": 128},
    "LLaMA-7B GQA-4": {"layers": 32, "kv_heads": 4, "head_dim": 128},
    "Mistral-7B": {"layers": 32, "kv_heads": 8, "head_dim": 128},
}

print("KV Cache Memory (per user, FP16):")
print(f"{'Model':<20} {'1K ctx':>8} {'4K ctx':>8} {'32K ctx':>8} {'128K ctx':>9}")
for name, cfg in configs.items():
    for ctx in [1024, 4096, 32768, 131072]:
        cache = KVCache(cfg["layers"], cfg["kv_heads"], cfg["head_dim"], ctx)
        mb = cache.memory_bytes() / 1e6
        if ctx == 1024:
            print(f"{name:<20}", end="")
        print(f" {mb:>7.0f}M", end="")
    print()

# Continuous batching simulator
class ContinuousBatcher:
    """Simulates continuous batching for LLM serving."""
    def __init__(self, max_batch_size=32):
        self.max_batch = max_batch_size
        self.active = {}       # slot_id -> remaining_tokens
        self.queue = deque()   # Waiting requests
        self.completed = 0
        self.total_tokens = 0
        self.idle_slots = 0

    def add_request(self, request_id, output_length):
        self.queue.append((request_id, output_length))

    def step(self):
        """Process one generation step for all active requests."""
        # Fill empty slots from queue
        while len(self.active) < self.max_batch and self.queue:
            req_id, length = self.queue.popleft()
            slot = len(self.active)
            self.active[slot] = (req_id, length)

        # Generate one token for each active request
        finished = []
        for slot, (req_id, remaining) in self.active.items():
            remaining -= 1
            if remaining <= 0:
                finished.append(slot)
                self.completed += 1
            else:
                self.active[slot] = (req_id, remaining)

        # Free finished slots immediately (continuous batching!)
        for slot in finished:
            del self.active[slot]

        active_count = len(self.active)
        self.idle_slots += self.max_batch - active_count
        self.total_tokens += active_count
        return active_count

# Simulate serving workload
batcher = ContinuousBatcher(max_batch_size=16)
import random
random.seed(42)
for i in range(100):
    batcher.add_request(f"req_{i}", random.randint(10, 200))

steps = 0
while batcher.active or batcher.queue:
    batcher.step()
    steps += 1

utilization = batcher.total_tokens / (steps * 16)
print(f"\\nContinuous Batching Simulation:")
print(f"  Requests served: {batcher.completed}")
print(f"  Total steps: {steps}")
print(f"  GPU utilization: {utilization:.1%}")
print(f"  Throughput: {batcher.total_tokens / steps:.1f} tokens/step")`,id:"code-serving"}),e.jsx(t,{type:"note",title:"PagedAttention (vLLM)",content:"PagedAttention (Kwon et al., 2023) manages KV cache like virtual memory pages. Instead of pre-allocating contiguous memory per sequence, it allocates fixed-size blocks on demand. This eliminates 60-80% of memory waste from internal fragmentation and enables serving 2-4x more concurrent users on the same hardware.",id:"note-paged-attention"}),e.jsx(t,{type:"tip",title:"Optimization Priority",content:"For single-user latency: (1) quantize weights to INT4, (2) enable KV cache, (3) use Flash Attention, (4) consider speculative decoding. For multi-user throughput: (1) continuous batching, (2) PagedAttention, (3) tensor parallelism across GPUs, (4) quantization for memory savings. Profile your specific workload to find the bottleneck.",id:"note-optimization-priority"}),e.jsx(r,{title:"Prefill vs. Decode Bottlenecks",content:"LLM serving has two distinct phases: prefill (processing the prompt, compute-bound) and decode (generating tokens one at a time, memory-bound). Optimizing for one can hurt the other. Chunked prefill and separate prefill/decode batching help manage this tension in production systems.",id:"warning-prefill-decode"})]})}const Q=Object.freeze(Object.defineProperty({__proto__:null,default:v},Symbol.toStringTag,{value:"Module"}));export{z as a,j as b,L as c,M as d,S as e,N as f,A as g,B as h,P as i,W as j,F as k,E as l,O as m,C as n,Q as o,q as s};
