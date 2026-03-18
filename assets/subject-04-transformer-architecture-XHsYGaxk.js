import{j as e}from"./vendor-DWbzdFaj.js";import{r as t}from"./vendor-katex-BYl39Yo6.js";import{D as a,N as n,E as o,P as i,W as s,T as r}from"./subject-01-text-fundamentals-DG6tAvii.js";function l(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Query, Key, and Value: The Intuition Behind Attention"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Self-attention allows every token in a sequence to attend to every other token. The mechanism revolves around three learned projections — Query, Key, and Value — that together determine how information flows between positions."}),e.jsx(a,{title:"Query, Key, Value Projections",definition:"Given an input matrix $X \\in \\mathbb{R}^{n \\times d}$, we compute $Q = XW^Q$, $K = XW^K$, $V = XW^V$ where $W^Q, W^K \\in \\mathbb{R}^{d \\times d_k}$ and $W^V \\in \\mathbb{R}^{d \\times d_v}$ are learned parameter matrices.",notation:"Q = query, K = key, V = value, n = sequence length, d = model dimension, d_k = key dimension",id:"def-qkv"}),e.jsx(n,{type:"intuition",title:"The Library Analogy",content:"Think of attention like a library search. The Query is your search question, the Keys are book titles on the shelf, and the Values are the actual book contents. You compare your query to each key, then retrieve a weighted combination of the values based on how well each key matched.",id:"note-library-analogy"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"How QKV Projections Work"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:`Each token's embedding is projected into three different vector spaces. The query vector asks "what am I looking for?", the key vector says "what do I contain?", and the value vector holds "what information do I provide if selected?"`}),e.jsx(t.BlockMath,{math:"Q = XW^Q, \\quad K = XW^K, \\quad V = XW^V"}),e.jsx(o,{title:"QKV Computation for a Short Sequence",problem:"Given a 3-token sequence with d=4 and d_k=d_v=2, show how Q, K, V are computed.",steps:[{formula:"X \\in \\mathbb{R}^{3 \\times 4}",explanation:"Input: 3 tokens, each with a 4-dimensional embedding."},{formula:"W^Q \\in \\mathbb{R}^{4 \\times 2}",explanation:"The query weight matrix projects from d=4 to d_k=2."},{formula:"Q = XW^Q \\in \\mathbb{R}^{3 \\times 2}",explanation:"Each token now has a 2-dimensional query vector."},{formula:"\\text{Same for } K = XW^K,\\; V = XW^V",explanation:"Key and value projections follow the identical pattern with their own weight matrices."}],id:"example-qkv-shapes"}),e.jsx(i,{title:"qkv_projections.py",code:`import torch
import torch.nn as nn

# Dimensions
batch_size, seq_len, d_model, d_k = 2, 5, 16, 8

# Input embeddings
X = torch.randn(batch_size, seq_len, d_model)

# Learned projection matrices
W_Q = nn.Linear(d_model, d_k, bias=False)
W_K = nn.Linear(d_model, d_k, bias=False)
W_V = nn.Linear(d_model, d_k, bias=False)

# Compute Q, K, V
Q = W_Q(X)  # (batch, seq_len, d_k)
K = W_K(X)  # (batch, seq_len, d_k)
V = W_V(X)  # (batch, seq_len, d_k)

print(f"Input shape:  {X.shape}")   # [2, 5, 16]
print(f"Query shape:  {Q.shape}")   # [2, 5, 8]
print(f"Key shape:    {K.shape}")   # [2, 5, 8]
print(f"Value shape:  {V.shape}")   # [2, 5, 8]

# Attention scores: how much each query attends to each key
scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_len, seq_len)
print(f"Score shape:  {scores.shape}")  # [2, 5, 5]`,id:"code-qkv"}),e.jsx(s,{title:"QKV Are Not Interchangeable",content:"Although Q, K, and V are all linear projections of the same input X, they serve fundamentally different roles. Swapping Q and K transposes the attention matrix (reversing who attends to whom). The value matrix V determines what information is actually retrieved — it is never used in the compatibility computation.",id:"warning-qkv-roles"}),e.jsx(n,{type:"note",title:"Parameter Count",content:"The QKV projections account for 3 * d_model * d_k parameters per attention head (ignoring biases). In GPT-3 with d_model=12288 and 96 heads (d_k=128), that is 3 * 12288 * 12288 ≈ 453M parameters just for QKV across all heads.",id:"note-param-count"})]})}const S=Object.freeze(Object.defineProperty({__proto__:null,default:l},Symbol.toStringTag,{value:"Module"}));function d(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Scaled Dot-Product Attention"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Scaled dot-product attention is the core computation inside every transformer. It computes a weighted sum of values where the weights come from the compatibility between queries and keys, scaled by the square root of the key dimension."}),e.jsx(a,{title:"Scaled Dot-Product Attention",definition:"$\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$ where $Q \\in \\mathbb{R}^{n \\times d_k}$, $K \\in \\mathbb{R}^{m \\times d_k}$, $V \\in \\mathbb{R}^{m \\times d_v}$, and $d_k$ is the key dimension.",notation:"n = query length, m = key/value length, d_k = key dim, d_v = value dim",id:"def-scaled-dot-product"}),e.jsxs("h2",{className:"text-2xl font-semibold",children:["Why Scale by ",e.jsx(t.InlineMath,{math:"\\sqrt{d_k}"}),"?"]}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["When ",e.jsx(t.InlineMath,{math:"d_k"})," is large, the dot products ",e.jsx(t.InlineMath,{math:"QK^T"})," tend to grow in magnitude, pushing the softmax into regions with extremely small gradients. Dividing by ",e.jsx(t.InlineMath,{math:"\\sqrt{d_k}"})," keeps the variance of the dot products at approximately 1, ensuring healthy gradient flow."]}),e.jsx(r,{title:"Variance of Dot Products",statement:"If the components of Q and K are independent random variables with mean 0 and variance 1, then $\\text{Var}(q \\cdot k) = d_k$. Scaling by $\\frac{1}{\\sqrt{d_k}}$ normalizes the variance back to 1.",proofSteps:["Let q_i, k_i be i.i.d. with mean 0 and variance 1.","q \\cdot k = \\sum_{i=1}^{d_k} q_i k_i","\\text{Var}(q_i k_i) = E[q_i^2 k_i^2] - (E[q_i k_i])^2 = 1 \\cdot 1 - 0 = 1","\\text{Var}(q \\cdot k) = \\sum_{i=1}^{d_k} \\text{Var}(q_i k_i) = d_k","\\text{Var}\\left(\\frac{q \\cdot k}{\\sqrt{d_k}}\\right) = \\frac{d_k}{d_k} = 1"],id:"thm-scaling"}),e.jsx(o,{title:"Softmax Saturation Without Scaling",problem:"Show how large dot products cause softmax to saturate for d_k = 512.",steps:[{formula:"\\text{Var}(q \\cdot k) = 512",explanation:"Without scaling, dot products have standard deviation ≈ 22.6."},{formula:"\\text{softmax}([22, -18, 20, -15]) \\approx [0.88, 0.0, 0.12, 0.0]",explanation:"Large magnitude differences make softmax nearly one-hot, killing gradients."},{formula:"\\text{After scaling: } [0.97, -0.80, 0.88, -0.66]",explanation:"Dividing by √512 ≈ 22.6 keeps values small, producing smoother attention weights."}],id:"example-saturation"}),e.jsx(i,{title:"scaled_dot_product_attention.py",code:`import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, n, d_k)
    K: (batch, m, d_k)
    V: (batch, m, d_v)
    Returns: (batch, n, d_v), attention_weights (batch, n, m)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights

# Demo
batch, n, d_k, d_v = 1, 4, 64, 64
Q = torch.randn(batch, n, d_k)
K = torch.randn(batch, n, d_k)
V = torch.randn(batch, n, d_v)

out, weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {out.shape}")        # [1, 4, 64]
print(f"Attention weights shape: {weights.shape}")  # [1, 4, 4]
print(f"Weights sum to 1: {weights.sum(dim=-1)}")   # All ones

# Show effect of scaling on weight entropy
scores_unscaled = torch.matmul(Q, K.transpose(-2, -1))
scores_scaled = scores_unscaled / math.sqrt(d_k)
w_unscaled = F.softmax(scores_unscaled, dim=-1)
w_scaled = F.softmax(scores_scaled, dim=-1)
entropy_un = -(w_unscaled * w_unscaled.log()).sum(-1).mean()
entropy_sc = -(w_scaled * w_scaled.log()).sum(-1).mean()
print(f"Entropy unscaled: {entropy_un:.3f}, scaled: {entropy_sc:.3f}")`,id:"code-sdpa"}),e.jsx(n,{type:"tip",title:"PyTorch Built-in",content:"Since PyTorch 2.0, use torch.nn.functional.scaled_dot_product_attention which automatically selects the most efficient backend (FlashAttention, Memory-Efficient Attention, or the math fallback) depending on hardware and input sizes.",id:"note-pytorch-sdpa"}),e.jsx(s,{title:"Numerical Stability",content:"In practice, softmax is computed as softmax(x - max(x)) to avoid overflow. When using masked attention, positions set to -inf are handled correctly because exp(-inf) = 0. However, if all positions in a row are masked, you get NaN — always ensure at least one position is unmasked per query.",id:"warning-numerical"})]})}const E=Object.freeze(Object.defineProperty({__proto__:null,default:d},Symbol.toStringTag,{value:"Module"}));function c(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Multi-Head Attention"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Rather than performing a single attention function, multi-head attention runs several attention operations in parallel, each with different learned projections. This allows the model to jointly attend to information from different representation subspaces at different positions."}),e.jsx(a,{title:"Multi-Head Attention",definition:"$\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\ldots, \\text{head}_h)W^O$ where $\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ and $W^O \\in \\mathbb{R}^{hd_v \\times d_{\\text{model}}}$.",notation:"h = number of heads, d_k = d_model / h (typically), W^O = output projection",id:"def-multihead"}),e.jsx(n,{type:"intuition",title:"Why Multiple Heads?",content:"A single attention head can only compute one set of attention weights per position. With 8 heads, one head might attend to syntactic structure (subject-verb agreement), another to semantic similarity, another to positional neighbors, etc. Multiple heads give the model a richer set of information-routing patterns.",id:"note-why-heads"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Head Splitting and Concatenation"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["In practice, instead of creating ",e.jsx(t.InlineMath,{math:"h"})," separate weight matrices, we project to the full ",e.jsx(t.InlineMath,{math:"d_{\\text{model}}"})," dimension and then reshape into ",e.jsx(t.InlineMath,{math:"h"})," heads of size ",e.jsx(t.InlineMath,{math:"d_k = d_{\\text{model}} / h"}),". After attention, heads are concatenated and projected back."]}),e.jsx(t.BlockMath,{math:"d_k = d_v = \\frac{d_{\\text{model}}}{h}"}),e.jsx(o,{title:"Multi-Head Dimensions in GPT-2 Small",problem:"GPT-2 Small has d_model=768 and h=12 heads. Trace the tensor shapes.",steps:[{formula:"X \\in \\mathbb{R}^{B \\times n \\times 768}",explanation:"Input: batch of sequences with 768-dim embeddings."},{formula:"QKV \\in \\mathbb{R}^{B \\times n \\times 768} \\text{ each}",explanation:"Full projection, then reshaped to (B, 12, n, 64)."},{formula:"\\text{head}_i \\in \\mathbb{R}^{B \\times n \\times 64}",explanation:"Each head operates on d_k=768/12=64 dimensional slices."},{formula:"\\text{Concat} \\in \\mathbb{R}^{B \\times n \\times 768}",explanation:"Concatenating 12 heads of dim 64 gives back 768."},{formula:"\\text{Output} = \\text{Concat} \\cdot W^O \\in \\mathbb{R}^{B \\times n \\times 768}",explanation:"Final linear projection mixes information across heads."}],id:"example-gpt2-shapes"}),e.jsx(i,{title:"multi_head_attention.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        B, n, _ = Q.shape

        # Project and reshape: (B, n, d_model) -> (B, h, n, d_k)
        Q = self.W_q(Q).view(B, n, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention per head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # (B, h, n, d_k)

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(B, n, self.d_model)
        return self.W_o(context)

# Usage
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)
out = mha(x, x, x)  # Self-attention: Q=K=V=x
print(f"Output: {out.shape}")  # [2, 10, 512]
print(f"Parameters: {sum(p.numel() for p in mha.parameters()):,}")`,id:"code-mha"}),e.jsx(s,{title:"Head Count Must Divide d_model",content:"If d_model is not evenly divisible by the number of heads, you cannot cleanly split the dimensions. This is why transformer model dimensions are almost always multiples of common head counts (64, 128). GPT-3's d_model=12288 with 96 heads gives d_k=128 exactly.",id:"warning-divisibility"}),e.jsx(n,{type:"note",title:"Computational Cost Is Unchanged",content:"Multi-head attention with h heads of dimension d_k = d_model/h has the same total computation as single-head attention with full d_model. The cost is O(n^2 * d_model) either way. The benefit is purely representational — multiple independent attention patterns at no extra FLOPs.",id:"note-cost"})]})}const F=Object.freeze(Object.defineProperty({__proto__:null,default:c},Symbol.toStringTag,{value:"Module"}));function m(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Attention Masking: Causal and Padding Masks"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Masking controls which positions each token can attend to. Causal masks prevent tokens from looking ahead (essential for autoregressive generation), while padding masks ensure that padding tokens do not participate in attention computations."}),e.jsx(a,{title:"Causal (Look-Ahead) Mask",definition:"A causal mask is a lower-triangular matrix $M \\in \\{0, -\\infty\\}^{n \\times n}$ where $M_{ij} = 0$ if $i \\geq j$ and $M_{ij} = -\\infty$ otherwise. Applied as $\\text{softmax}\\!\\left(\\frac{QK^T}{\\sqrt{d_k}} + M\\right)V$.",notation:"Position i can attend to positions 0 through i, but not to positions i+1 through n-1.",id:"def-causal-mask"}),e.jsx(a,{title:"Padding Mask",definition:"A padding mask is a binary vector $p \\in \\{0, 1\\}^n$ where $p_j = 0$ for padding positions. It is broadcast so that all queries ignore padding keys: $\\text{scores}_{ij} = -\\infty$ wherever $p_j = 0$.",id:"def-padding-mask"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Causal Masking for Autoregressive Models"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:'In decoder-only models like GPT, each token must only attend to itself and previous tokens. Without the causal mask, the model would "cheat" during training by reading future tokens.'}),e.jsx(o,{title:"Causal Mask for 4 Tokens",problem:"Construct and apply a causal mask for the sentence 'The cat sat down'.",steps:[{formula:"M = \\begin{pmatrix} 0 & -\\infty & -\\infty & -\\infty \\\\ 0 & 0 & -\\infty & -\\infty \\\\ 0 & 0 & 0 & -\\infty \\\\ 0 & 0 & 0 & 0 \\end{pmatrix}",explanation:"Lower-triangular: token i attends only to tokens 0..i."},{formula:"\\text{scores} + M",explanation:"Adding -inf makes those positions 0 after softmax."},{formula:"\\text{softmax row 2} = \\text{softmax}([s_{20}, s_{21}, s_{22}, -\\infty])",explanation:'"sat" attends to "The", "cat", and "sat" only.'}],id:"example-causal"}),e.jsx(i,{title:"attention_masks.py",code:`import torch
import torch.nn.functional as F
import math

def create_causal_mask(seq_len, device='cpu'):
    """Lower-triangular causal mask."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask  # 1 = attend, 0 = block

def create_padding_mask(lengths, max_len, device='cpu'):
    """Mask out padding positions given actual sequence lengths."""
    arange = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
    mask = arange < lengths.unsqueeze(1)  # (batch, max_len)
    return mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, max_len) for broadcasting

def masked_attention(Q, K, V, causal=True, padding_mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if causal:
        n = Q.size(-2)
        causal_mask = create_causal_mask(n, Q.device)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

    if padding_mask is not None:
        scores = scores.masked_fill(padding_mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights

# Demo: Causal mask
B, n, d = 1, 6, 32
Q = K = V = torch.randn(B, n, d)
out, w = masked_attention(Q, K, V, causal=True)
print("Causal attention weights (should be lower-triangular):")
print(w[0].round(decimals=2))

# Demo: Padding mask for variable-length sequences
lengths = torch.tensor([4, 6])  # First sequence has 4 real tokens, second has 6
pad_mask = create_padding_mask(lengths, max_len=6)
Q2 = K2 = V2 = torch.randn(2, n, d)
out2, w2 = masked_attention(Q2, K2, V2, causal=False, padding_mask=pad_mask)
print(f"\\nPadding mask shape: {pad_mask.shape}")
print(f"Seq 1 attends to positions: {pad_mask[0, 0, 0]}")`,id:"code-masks"}),e.jsx(n,{type:"tip",title:"Combining Masks",content:"In practice, causal and padding masks are combined with a logical AND (or equivalently, both are added to the scores as additive -inf terms). Decoder models need both: the causal mask for autoregressive ordering, and the padding mask for batched sequences of different lengths.",id:"note-combining"}),e.jsx(s,{title:"Prefix LM vs. Causal LM Masking",content:"Some architectures (e.g., T5, UL2) use a prefix LM mask where a prefix portion of the sequence has full bidirectional attention while the rest uses causal masking. This is different from a pure causal mask and allows bidirectional encoding of the input while generating the output autoregressively.",id:"warning-prefix-lm"}),e.jsx(n,{type:"historical",title:"Masked Self-Attention in the Original Paper",content:"Vaswani et al. (2017) introduced masked self-attention specifically for the decoder. The encoder uses full bidirectional attention (no causal mask). Modern decoder-only models (GPT series) use causal masking throughout, eliminating the encoder entirely.",id:"note-history"})]})}const A=Object.freeze(Object.defineProperty({__proto__:null,default:m},Symbol.toStringTag,{value:"Module"}));function h(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Positional Encoding: Sinusoidal Embeddings"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:'Self-attention is inherently permutation-invariant — it has no notion of token order. Positional encodings inject sequence position information into the model, enabling it to distinguish "the cat sat on the mat" from "the mat sat on the cat."'}),e.jsx(a,{title:"Sinusoidal Positional Encoding",definition:"For position $\\text{pos}$ and dimension $i$: $PE_{(\\text{pos}, 2i)} = \\sin\\!\\left(\\frac{\\text{pos}}{10000^{2i/d_{\\text{model}}}}\\right)$, $PE_{(\\text{pos}, 2i+1)} = \\cos\\!\\left(\\frac{\\text{pos}}{10000^{2i/d_{\\text{model}}}}\\right)$",notation:"pos = token position (0-indexed), i = dimension index, d_model = embedding dimension",id:"def-sinusoidal-pe"}),e.jsx(n,{type:"intuition",title:"Why Sinusoids?",content:"Each dimension oscillates at a different frequency, from high-frequency (small i) to low-frequency (large i). This creates a unique 'fingerprint' for each position. Additionally, the encoding for position pos+k can be expressed as a linear function of the encoding at pos, allowing the model to learn relative position attention.",id:"note-why-sinusoids"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The Encoding Formula"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The wavelengths form a geometric progression from ",e.jsx(t.InlineMath,{math:"2\\pi"})," to"," ",e.jsx(t.InlineMath,{math:"10000 \\cdot 2\\pi"}),". This means lower dimensions capture fine-grained position differences while higher dimensions capture broader positional patterns."]}),e.jsx(t.BlockMath,{math:"PE_{(\\text{pos}, 2i)} = \\sin\\!\\left(\\frac{\\text{pos}}{10000^{2i/d}}\\right), \\quad PE_{(\\text{pos}, 2i+1)} = \\cos\\!\\left(\\frac{\\text{pos}}{10000^{2i/d}}\\right)"}),e.jsx(o,{title:"Computing PE for Position 3 with d_model=8",problem:"Calculate the positional encoding vector for position 3 in an 8-dimensional model.",steps:[{formula:"PE_{(3,0)} = \\sin(3 / 10000^{0/8}) = \\sin(3) \\approx 0.141",explanation:"Dimension 0 (sin): frequency = 1, highest frequency."},{formula:"PE_{(3,1)} = \\cos(3 / 10000^{0/8}) = \\cos(3) \\approx -0.990",explanation:"Dimension 1 (cos): same frequency as dimension 0."},{formula:"PE_{(3,2)} = \\sin(3 / 10000^{2/8}) = \\sin(0.300) \\approx 0.296",explanation:"Dimension 2 (sin): lower frequency, wavelength ≈ 10."},{formula:"PE_{(3,6)} = \\sin(3 / 10000^{6/8}) \\approx \\sin(0.003) \\approx 0.003",explanation:"Dimension 6 (sin): very low frequency, nearly flat."}],id:"example-pe-calc"}),e.jsx(i,{title:"positional_encoding.py",code:`import torch
import math

def sinusoidal_positional_encoding(max_len, d_model):
    """Generate sinusoidal positional encodings."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (max_len, d_model)

# Generate encodings
pe = sinusoidal_positional_encoding(max_len=100, d_model=64)
print(f"PE shape: {pe.shape}")  # [100, 64]

# Key property: dot product between positions captures relative distance
dot_products = torch.matmul(pe, pe.T)
print(f"PE[0] · PE[1] = {dot_products[0, 1]:.3f}")
print(f"PE[0] · PE[50] = {dot_products[0, 50]:.3f}")
print(f"PE[0] · PE[99] = {dot_products[0, 99]:.3f}")

# In a transformer, PE is added to token embeddings
class TokenWithPosition(torch.nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, d_model)
        self.register_buffer('pe', sinusoidal_positional_encoding(max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        return self.tok_emb(x) + self.pe[:seq_len]

model = TokenWithPosition(vocab_size=1000, d_model=64)
tokens = torch.randint(0, 1000, (2, 10))
embedded = model(tokens)
print(f"Embedded shape: {embedded.shape}")  # [2, 10, 64]`,id:"code-pe"}),e.jsx(s,{title:"Learned vs. Sinusoidal Encodings",content:"Many modern models (GPT-2, BERT) use learned positional embeddings instead of sinusoidal ones. Learned embeddings are more flexible but cannot extrapolate to unseen positions beyond max_len. Sinusoidal encodings theoretically generalize but in practice also degrade for positions far beyond training lengths.",id:"warning-learned-vs-fixed"}),e.jsx(n,{type:"historical",title:"From Fixed to Learned to Relative",content:"The original transformer (2017) used sinusoidal encodings. GPT-2 and BERT switched to learned absolute positions. Modern models increasingly use relative position methods (RoPE, ALiBi) that encode position differences rather than absolute positions, enabling better length generalization.",id:"note-evolution"})]})}const K=Object.freeze(Object.defineProperty({__proto__:null,default:h},Symbol.toStringTag,{value:"Module"}));function p(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Feed-Forward Networks in Transformers"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:`Every transformer layer contains a position-wise feed-forward network (FFN) applied independently to each token. This two-layer MLP is where the majority of the model's parameters reside and where much of the "knowledge storage" happens.`}),e.jsx(a,{title:"Position-Wise Feed-Forward Network",definition:"$\\text{FFN}(x) = W_2 \\cdot \\sigma(W_1 x + b_1) + b_2$ where $W_1 \\in \\mathbb{R}^{d_{\\text{model}} \\times d_{ff}}$, $W_2 \\in \\mathbb{R}^{d_{ff} \\times d_{\\text{model}}}$, and $\\sigma$ is a nonlinear activation function.",notation:"d_ff is typically 4 × d_model. σ is ReLU in the original transformer, GELU in modern variants.",id:"def-ffn"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"ReLU vs. GELU Activation"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The original transformer uses ReLU: ",e.jsx(t.InlineMath,{math:"\\text{ReLU}(x) = \\max(0, x)"}),". Modern models (BERT, GPT-2+) prefer GELU, which provides a smooth approximation that allows small negative values to pass through."]}),e.jsx(t.BlockMath,{math:"\\text{GELU}(x) = x \\cdot \\Phi(x) \\approx 0.5x\\left(1 + \\tanh\\left[\\sqrt{2/\\pi}(x + 0.044715x^3)\\right]\\right)"}),e.jsx(o,{title:"FFN Parameter Count in GPT-3",problem:"Calculate the FFN parameters per layer in GPT-3 (d_model=12288, d_ff=49152).",steps:[{formula:"W_1: 12288 \\times 49152 = 603,979,776",explanation:"First linear layer expands from d_model to d_ff = 4 * d_model."},{formula:"W_2: 49152 \\times 12288 = 603,979,776",explanation:"Second linear layer contracts back to d_model."},{formula:"b_1 + b_2 = 49152 + 12288 = 61,440",explanation:"Bias terms (small relative to weights)."},{formula:"\\text{Total} \\approx 1.208\\text{B parameters per layer}",explanation:"The FFN has ~2/3 of each layer's total parameters."}],id:"example-ffn-params"}),e.jsx(n,{type:"intuition",title:"FFN as Key-Value Memory",content:"Research by Geva et al. (2021) showed that FFN layers function as key-value memories. The first layer's rows act as keys that pattern-match on input features, while the second layer's columns act as values that contribute to the output distribution. Individual neurons can encode specific facts.",id:"note-kv-memory"}),e.jsx(i,{title:"feed_forward_network.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerFFN(nn.Module):
    """Position-wise FFN with configurable activation."""
    def __init__(self, d_model, d_ff, activation='gelu', dropout=0.1):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        if self.activation == 'relu':
            hidden = F.relu(self.W1(x))
        elif self.activation == 'gelu':
            hidden = F.gelu(self.W1(x))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        return self.W2(self.dropout(hidden))

# SwiGLU variant used in LLaMA, PaLM, etc.
class SwiGLUFFN(nn.Module):
    """Gated FFN: SwiGLU activation (Shazeer 2020)."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.W_gate(x))      # SiLU = x * sigmoid(x)
        up = self.W_up(x)
        return self.W_down(self.dropout(gate * up))

# Compare parameter counts
d_model, d_ff = 768, 3072
ffn_standard = TransformerFFN(d_model, d_ff)
ffn_swiglu = SwiGLUFFN(d_model, d_ff)

params_std = sum(p.numel() for p in ffn_standard.parameters())
params_glu = sum(p.numel() for p in ffn_swiglu.parameters())
print(f"Standard FFN params: {params_std:,}")   # ~4.7M
print(f"SwiGLU FFN params:   {params_glu:,}")   # ~7.1M (3 matrices)

# Forward pass
x = torch.randn(2, 10, d_model)
print(f"Standard output: {ffn_standard(x).shape}")
print(f"SwiGLU output:   {ffn_swiglu(x).shape}")`,id:"code-ffn"}),e.jsx(s,{title:"SwiGLU Changes the Parameter Budget",content:"SwiGLU uses three weight matrices instead of two, increasing FFN parameters by 50%. To keep the total parameter count constant, LLaMA reduces d_ff from 4*d_model to roughly 2.67*d_model (specifically 8/3*d_model rounded to multiples of 256). Always account for this when comparing architectures.",id:"warning-swiglu-params"}),e.jsx(n,{type:"note",title:"Position-Wise Means Per-Token",content:"The FFN is applied identically and independently to every position in the sequence. There is no interaction between tokens in the FFN — all cross-position communication happens in the attention layer. This makes FFNs embarrassingly parallel across the sequence dimension.",id:"note-position-wise"})]})}const B=Object.freeze(Object.defineProperty({__proto__:null,default:p},Symbol.toStringTag,{value:"Module"}));function u(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Layer Normalization vs. Batch Normalization"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Normalization stabilizes training by controlling the scale of activations. Transformers universally use Layer Normalization rather than Batch Normalization because it normalizes across features independently per sample, making it compatible with variable-length sequences and small batch sizes."}),e.jsx(a,{title:"Layer Normalization",definition:"For an input vector $x \\in \\mathbb{R}^d$: $\\text{LayerNorm}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta$ where $\\mu = \\frac{1}{d}\\sum_{i=1}^d x_i$, $\\sigma^2 = \\frac{1}{d}\\sum_{i=1}^d (x_i - \\mu)^2$, and $\\gamma, \\beta \\in \\mathbb{R}^d$ are learned parameters.",notation:"μ = mean across features, σ² = variance across features, γ = scale, β = shift, ε ≈ 1e-5",id:"def-layernorm"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Layer Norm vs. Batch Norm"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Batch Normalization computes statistics across the batch dimension for each feature, while Layer Normalization computes statistics across the feature dimension for each sample. This distinction is critical for sequence models."}),e.jsx(o,{title:"Normalization Axis Comparison",problem:"For a tensor of shape (batch=4, seq_len=10, d_model=512), compare which dimensions BatchNorm and LayerNorm normalize over.",steps:[{formula:"\\text{BatchNorm: mean over } (4, 10) \\text{ for each of 512 features}",explanation:"Computes one mean and variance per feature across all samples and positions."},{formula:"\\text{LayerNorm: mean over } (512,) \\text{ for each position}",explanation:"Computes one mean and variance per (sample, position) pair across all 512 features."},{formula:"\\text{BatchNorm stats: 512 means, 512 variances}",explanation:"Statistics depend on batch — problematic at inference with batch size 1."},{formula:"\\text{LayerNorm stats: 4 \\times 10 = 40 means, 40 variances}",explanation:"Statistics are independent of batch size, computed per token."}],id:"example-norm-axes"}),e.jsx(i,{title:"layer_norm_comparison.py",code:`import torch
import torch.nn as nn

batch, seq_len, d_model = 4, 10, 512
x = torch.randn(batch, seq_len, d_model) * 5 + 3  # Shifted and scaled

# --- Layer Normalization (standard in transformers) ---
layer_norm = nn.LayerNorm(d_model)
ln_out = layer_norm(x)

# Check: each position has mean≈0, std≈1
print("LayerNorm per-position stats:")
print(f"  Mean: {ln_out[0, 0].mean():.4f}")  # ≈ 0
print(f"  Std:  {ln_out[0, 0].std():.4f}")   # ≈ 1

# --- RMSNorm (used in LLaMA, Gemma) ---
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019)."""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight

rms_norm = RMSNorm(d_model)
rms_out = rms_norm(x)
print(f"\\nRMSNorm output shape: {rms_out.shape}")
print(f"  RMS of output: {rms_out[0, 0].pow(2).mean().sqrt():.4f}")  # ≈ 1

# --- Pre-Norm vs Post-Norm placement ---
class PreNormBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, d_model)

    def forward(self, x):
        return x + self.ff(self.norm(x))  # Norm BEFORE sublayer

class PostNormBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.norm(x + self.ff(x))  # Norm AFTER sublayer

print(f"\\nPreNorm output:  {PreNormBlock(d_model)(x).shape}")
print(f"PostNorm output: {PostNormBlock(d_model)(x).shape}")`,id:"code-layernorm"}),e.jsx(n,{type:"note",title:"Pre-Norm vs. Post-Norm",content:"The original transformer uses Post-Norm (normalize after the residual addition). GPT-2 and most modern models switched to Pre-Norm (normalize before the sublayer), which makes training more stable and eliminates the need for learning rate warmup. Pre-Norm ensures the residual path has an unimpeded gradient highway.",id:"note-pre-post"}),e.jsx(s,{title:"RMSNorm Drops the Mean Centering",content:"RMSNorm (used in LLaMA, Gemma, Mistral) simplifies LayerNorm by removing the mean subtraction and learned bias. This saves compute and works well in practice, but the output is not zero-centered. When porting weights between architectures, do not interchange LayerNorm and RMSNorm without retraining.",id:"warning-rmsnorm"}),e.jsx(n,{type:"historical",title:"Evolution of Normalization",content:"BatchNorm (Ioffe & Szegedy, 2015) was designed for CNNs. LayerNorm (Ba et al., 2016) was proposed specifically for RNNs and later adopted by transformers. RMSNorm (Zhang & Sennrich, 2019) simplified LayerNorm with minimal quality loss. Modern LLMs almost universally use Pre-RMSNorm.",id:"note-history"})]})}const $=Object.freeze(Object.defineProperty({__proto__:null,default:u},Symbol.toStringTag,{value:"Module"}));function f(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Residual Connections and Gradient Flow"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Residual (skip) connections are essential for training deep transformers. By adding the input of each sublayer directly to its output, they create a gradient highway that prevents vanishing gradients and allows information to flow unchanged through many layers."}),e.jsx(a,{title:"Residual Connection",definition:"Given a sublayer function $F(x)$, a residual connection computes $\\text{output} = x + F(x)$. In a transformer layer with Pre-Norm: $\\text{output} = x + \\text{Attention}(\\text{LayerNorm}(x))$.",notation:"x = input, F(x) = sublayer transformation (attention or FFN)",id:"def-residual"}),e.jsx(r,{title:"Gradient Flow Through Residual Connections",statement:"For a network with L residual blocks, the gradient of the loss with respect to an early layer's output $x_l$ includes a direct identity term: $\\frac{\\partial \\mathcal{L}}{\\partial x_l} = \\frac{\\partial \\mathcal{L}}{\\partial x_L} \\left(1 + \\frac{\\partial}{\\partial x_l}\\sum_{i=l}^{L-1} F_i(x_i)\\right)$. The '1' term ensures gradients can flow directly from the loss to any layer.",id:"thm-gradient-flow"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Why Residuals Matter"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Without residual connections, gradients must pass through every layer's nonlinearity. In a 96-layer GPT-3, this would cause severe vanishing gradients. The skip connection provides a direct path, ensuring the gradient magnitude stays bounded."}),e.jsx(o,{title:"Gradient Magnitude With and Without Residuals",problem:"Compare gradient norms after 10 layers with and without residual connections.",steps:[{formula:"\\text{Without: } \\|\\nabla\\| \\approx \\prod_{i=1}^{10} \\|J_i\\|",explanation:"Gradient is a product of Jacobians — exponential decay if ||J_i|| < 1."},{formula:"\\text{If } \\|J_i\\| \\approx 0.9: \\; 0.9^{10} \\approx 0.35",explanation:"65% gradient signal lost after just 10 layers."},{formula:"\\text{With residuals: } \\nabla = I + \\text{higher-order terms}",explanation:"The identity component ensures gradient magnitude ≥ 1."},{formula:"\\text{At 96 layers: } 0.9^{96} \\approx 0.00003 \\text{ vs. } \\approx 1.0",explanation:"Residual connections prevent catastrophic gradient collapse."}],id:"example-gradient-decay"}),e.jsx(i,{title:"residual_connections.py",code:`import torch
import torch.nn as nn

class TransformerLayerPreNorm(nn.Module):
    """Single transformer layer with Pre-Norm residual connections."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Residual 1: x + Attention(LayerNorm(x))
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + self.dropout(attn_out)

        # Residual 2: x + FFN(LayerNorm(x))
        x = x + self.ffn(self.norm2(x))
        return x

# Build a deep model and verify gradient flow
d_model, num_heads, d_ff, num_layers = 256, 8, 1024, 12
layers = nn.ModuleList([
    TransformerLayerPreNorm(d_model, num_heads, d_ff)
    for _ in range(num_layers)
])
final_norm = nn.LayerNorm(d_model)

x = torch.randn(1, 20, d_model, requires_grad=True)
h = x
for layer in layers:
    h = layer(h)
h = final_norm(h)

loss = h.sum()
loss.backward()

print(f"Input gradient norm: {x.grad.norm():.4f}")
print(f"Input gradient mean: {x.grad.mean():.6f}")
print(f"Gradient is finite: {x.grad.isfinite().all()}")

# Compare: without residuals, gradient would vanish
class NoResidualLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
    def forward(self, x):
        return torch.tanh(self.linear(x))  # No skip connection

no_res = nn.Sequential(*[NoResidualLayer(d_model) for _ in range(num_layers)])
x2 = torch.randn(1, 20, d_model, requires_grad=True)
loss2 = no_res(x2).sum()
loss2.backward()
print(f"\\nNo-residual gradient norm: {x2.grad.norm():.6f}")  # Much smaller`,id:"code-residual"}),e.jsx(n,{type:"intuition",title:"Residuals as an Ensemble",content:"Veit et al. (2016) showed that residual networks can be viewed as an ensemble of many shallow networks. Each layer adds a small refinement to the representation. Deleting a single layer has minimal impact, unlike in a plain network where it would be catastrophic.",id:"note-ensemble"}),e.jsx(s,{title:"Residual Connection Scale",content:"In very deep transformers (100+ layers), even with residual connections, the activations can grow unboundedly because each layer adds to the residual stream. Techniques like DeepNorm (scaling the residual by a factor α > 1) or fixup initialization address this by carefully controlling the scale of each sublayer's contribution.",id:"warning-scale"})]})}const R=Object.freeze(Object.defineProperty({__proto__:null,default:f},Symbol.toStringTag,{value:"Module"}));function _(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"The Transformer Encoder Stack"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The encoder processes an input sequence with full bidirectional attention, producing contextualized representations where every token can attend to every other token. Encoder-only models like BERT are built by stacking N identical encoder layers."}),e.jsx(a,{title:"Encoder Layer",definition:"Each encoder layer consists of two sublayers: (1) multi-head self-attention, and (2) a position-wise feed-forward network. Each sublayer is wrapped with a residual connection and layer normalization: $\\text{output} = \\text{LayerNorm}(x + \\text{Sublayer}(x))$.",notation:"N = number of stacked layers (6 in original, 12 in BERT-base, 24 in BERT-large)",id:"def-encoder-layer"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Encoder Architecture"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The encoder takes token embeddings plus positional encodings as input and passes them through N identical layers. The output is a sequence of contextualized vectors, one per input token, that encode rich bidirectional context."}),e.jsx(t.BlockMath,{math:"h^{(l)} = \\text{EncoderLayer}(h^{(l-1)}), \\quad l = 1, \\ldots, N"}),e.jsx(o,{title:"Information Flow in a 6-Layer Encoder",problem:"Trace how the word 'bank' gets contextualized in 'The bank of the river was muddy'.",steps:[{formula:"h^{(0)}_{\\text{bank}} = e_{\\text{bank}} + PE_1",explanation:'Initial embedding is context-free — "bank" could mean financial or river.'},{formula:"h^{(1)}: \\text{bank attends to river, muddy}",explanation:"Layer 1 attention starts incorporating nearby context."},{formula:'h^{(3)}: \\text{captures "bank of the river" phrase}',explanation:"Middle layers build phrase-level representations."},{formula:"h^{(6)}: \\text{fully disambiguated to riverbank}",explanation:"Final representation encodes complete sentence context."}],id:"example-contextualization"}),e.jsx(i,{title:"transformer_encoder.py",code:`import torch
import torch.nn as nn
import math

class TransformerEncoder(nn.Module):
    """Full transformer encoder stack."""
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_emb(positions)
        x = self.dropout(x)

        # Convert padding mask: True = ignore
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.final_norm(x)

# Build a BERT-base-sized encoder
encoder = TransformerEncoder(
    vocab_size=30522, d_model=768, num_heads=12,
    d_ff=3072, num_layers=12, max_len=512
)

# Forward pass
input_ids = torch.randint(0, 30522, (2, 128))
mask = torch.ones(2, 128)
mask[0, 100:] = 0  # First sequence padded after position 100

output = encoder(input_ids, attention_mask=mask)
print(f"Encoder output: {output.shape}")  # [2, 128, 768]

# Count parameters
total = sum(p.numel() for p in encoder.parameters())
print(f"Total parameters: {total:,}")  # ~86M (BERT-base scale)`,id:"code-encoder"}),e.jsx(n,{type:"note",title:"Bidirectional Context",content:"The key property of the encoder is bidirectional attention. Every token sees every other token, including future tokens. This makes encoders ideal for understanding tasks (classification, NER, similarity) but unsuitable for text generation — they cannot be used autoregressively without modification.",id:"note-bidirectional"}),e.jsx(s,{title:"Encoder-Only Models Are Not Generative",content:"Although BERT-style encoders can be used for masked language modeling (predicting [MASK] tokens), they cannot generate text left-to-right. For generation, you need either a decoder (GPT) or an encoder-decoder (T5). Encoder-only models excel at classification, retrieval, and extraction tasks.",id:"warning-not-generative"}),e.jsx(n,{type:"historical",title:"Notable Encoder-Only Models",content:"BERT (Devlin et al., 2018) popularized encoder-only transformers. RoBERTa (2019) improved training. DeBERTa (2021) added disentangled attention. Modern embedding models like E5, GTE, and BGE are all encoder-based, showing encoders remain vital for retrieval and understanding tasks.",id:"note-history"})]})}const z=Object.freeze(Object.defineProperty({__proto__:null,default:_},Symbol.toStringTag,{value:"Module"}));function x(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"The Transformer Decoder and Cross-Attention"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The decoder generates output tokens autoregressively, one at a time. Each decoder layer contains three sublayers: masked self-attention (causal), cross-attention to encoder outputs (in encoder-decoder models), and a feed-forward network."}),e.jsx(a,{title:"Decoder Layer (Encoder-Decoder)",definition:"Each decoder layer applies: (1) causal self-attention on decoder tokens, (2) cross-attention where Q comes from the decoder and K, V come from the encoder output, and (3) a position-wise FFN. Each sublayer uses residual connections and layer normalization.",id:"def-decoder-layer"}),e.jsx(a,{title:"Cross-Attention",definition:"Cross-attention computes $\\text{Attention}(Q_{\\text{dec}}, K_{\\text{enc}}, V_{\\text{enc}})$ where queries are from the decoder layer and keys/values are from the encoder output. This allows the decoder to selectively read from the input representation.",notation:"Q = decoder hidden states, K = V = encoder output",id:"def-cross-attention"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Decoder-Only vs. Encoder-Decoder"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Decoder-only models (GPT series) drop the cross-attention sublayer entirely, using only causal self-attention and FFN. The input and output are concatenated into a single sequence with causal masking."}),e.jsx(o,{title:"Three Sublayers in a Full Decoder",problem:"Trace tensor flow through one decoder layer in a translation model.",steps:[{formula:"h^{\\text{dec}} \\in \\mathbb{R}^{B \\times m \\times d}",explanation:"Decoder input: m target tokens generated so far."},{formula:"\\text{CausalAttn}(h^{\\text{dec}})",explanation:"Masked self-attention — each target token attends only to previous target tokens."},{formula:"\\text{CrossAttn}(Q=h^{\\text{dec}}, K=h^{\\text{enc}}, V=h^{\\text{enc}})",explanation:"Cross-attention lets each target token read from all encoder (source) positions."},{formula:"\\text{FFN}(\\cdot)",explanation:"Feed-forward network processes each position independently."}],id:"example-decoder-flow"}),e.jsx(i,{title:"transformer_decoder.py",code:`import torch
import torch.nn as nn
import math

class CausalDecoderOnly(nn.Module):
    """GPT-style decoder-only transformer."""
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerEncoderLayer(  # No cross-attn needed
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_ff, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) * math.sqrt(self.d_model)
        x = self.dropout(x + self.pos_emb(positions))

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T,
            device=input_ids.device)

        x = self.decoder(x, mask=causal_mask, is_causal=True)
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

# Build a small GPT-like model
model = CausalDecoderOnly(
    vocab_size=50257, d_model=768, num_heads=12,
    d_ff=3072, num_layers=12
)

tokens = torch.randint(0, 50257, (2, 64))
logits = model(tokens)
print(f"Logits shape: {logits.shape}")  # [2, 64, 50257]
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Autoregressive generation (greedy)
@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=20):
    model.eval()
    ids = prompt_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(ids)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids

prompt = torch.randint(0, 50257, (1, 5))
generated = generate(model, prompt, max_new_tokens=10)
print(f"Generated sequence length: {generated.shape[1]}")`,id:"code-decoder"}),e.jsx(n,{type:"intuition",title:"Why Decoder-Only Models Dominate",content:"Decoder-only models are simpler (no cross-attention), naturally handle any task as text-to-text completion, and scale more predictably. They can replicate encoder-decoder behavior by processing the input and output as a single concatenated sequence with appropriate attention masking.",id:"note-decoder-dominance"}),e.jsx(s,{title:"KV Cache for Efficient Generation",content:"During autoregressive generation, recomputing attention over all previous tokens at each step is O(n²) total. The KV cache stores previously computed key and value vectors, reducing each step to O(n). Without caching, generation speed degrades dramatically for long sequences.",id:"warning-kv-cache"}),e.jsx(n,{type:"note",title:"Cross-Attention Lives On",content:"While pure LLMs are decoder-only, cross-attention remains crucial in multimodal models. Vision-language models like Flamingo use cross-attention to let the language decoder attend to visual features. Whisper uses cross-attention for the audio encoder-to-text decoder connection.",id:"note-cross-attn-lives"})]})}const Q=Object.freeze(Object.defineProperty({__proto__:null,default:x},Symbol.toStringTag,{value:"Module"}));function g(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"The Full Encoder-Decoder Architecture"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The original transformer is an encoder-decoder model designed for sequence-to-sequence tasks. The encoder processes the input bidirectionally, and the decoder generates the output autoregressively while attending to the encoder's representations via cross-attention."}),e.jsx(a,{title:"Encoder-Decoder Transformer",definition:"The full architecture maps an input sequence $(x_1, \\ldots, x_n)$ to an output sequence $(y_1, \\ldots, y_m)$ via: $h = \\text{Encoder}(x_1, \\ldots, x_n)$, then $P(y_t | y_{<t}, h) = \\text{Decoder}(y_{<t}, h)$ for each output step $t$.",notation:"h = encoder hidden states, y_{<t} = previously generated tokens",id:"def-enc-dec"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Information Flow"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The encoder runs once on the full input. The decoder then runs autoregressively, generating one token at a time. At each step, the decoder's cross-attention layers read from the encoder output, allowing it to selectively focus on relevant parts of the input."}),e.jsx(o,{title:"Machine Translation: English to French",problem:"Show the encoder-decoder flow for translating 'The cat sat' to 'Le chat assis'.",steps:[{formula:'\\text{Encoder}(\\text{"The", "cat", "sat"}) \\rightarrow h \\in \\mathbb{R}^{3 \\times d}',explanation:"Encoder produces contextualized representations for all 3 source tokens."},{formula:'\\text{Decoder step 1: } P(y_1 | \\text{<bos>}, h) \\rightarrow \\text{"Le"}',explanation:"Decoder generates first token, attending to all encoder positions."},{formula:'\\text{Decoder step 2: } P(y_2 | \\text{"Le"}, h) \\rightarrow \\text{"chat"}',explanation:'Cross-attention focuses on "cat" in the encoder output.'},{formula:'\\text{Decoder step 3: } P(y_3 | \\text{"Le", "chat"}, h) \\rightarrow \\text{"assis"}',explanation:'Cross-attention focuses on "sat" — note the word alignment is learned.'}],id:"example-translation"}),e.jsx(i,{title:"encoder_decoder_transformer.py",code:`import torch
import torch.nn as nn
import math

class EncoderDecoderTransformer(nn.Module):
    """Full encoder-decoder transformer (T5-style)."""
    def __init__(self, src_vocab, tgt_vocab, d_model=512, num_heads=8,
                 d_ff=2048, num_layers=6, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_ff, dropout=dropout,
            activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.output_proj = nn.Linear(d_model, tgt_vocab)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src_ids, src_mask=None):
        T = src_ids.size(1)
        pos = torch.arange(T, device=src_ids.device).unsqueeze(0)
        x = self.dropout(self.src_emb(src_ids) * math.sqrt(self.d_model)
                         + self.pos_emb(pos))
        padding_mask = (src_mask == 0) if src_mask is not None else None
        return self.transformer.encoder(x, src_key_padding_mask=padding_mask)

    def decode(self, tgt_ids, memory, tgt_mask=None, memory_mask=None):
        T = tgt_ids.size(1)
        pos = torch.arange(T, device=tgt_ids.device).unsqueeze(0)
        x = self.dropout(self.tgt_emb(tgt_ids) * math.sqrt(self.d_model)
                         + self.pos_emb(pos))
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        mem_pad = (memory_mask == 0) if memory_mask is not None else None
        out = self.transformer.decoder(x, memory, tgt_mask=causal,
                                       memory_key_padding_mask=mem_pad)
        return self.output_proj(out)

    def forward(self, src_ids, tgt_ids, src_mask=None):
        memory = self.encode(src_ids, src_mask)
        logits = self.decode(tgt_ids, memory, memory_mask=src_mask)
        return logits

# Build and test
model = EncoderDecoderTransformer(src_vocab=32000, tgt_vocab=32000)
src = torch.randint(0, 32000, (2, 20))
tgt = torch.randint(0, 32000, (2, 15))
logits = model(src, tgt)
print(f"Output logits: {logits.shape}")  # [2, 15, 32000]
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")`,id:"code-enc-dec"}),e.jsx(n,{type:"note",title:"Encoder-Decoder Use Cases",content:"Encoder-decoder models excel at tasks with distinct input/output: translation, summarization, and question answering. T5 frames all NLP tasks as text-to-text, using an encoder-decoder. BART uses a similar architecture for denoising pretraining. Whisper uses encoder-decoder for speech-to-text.",id:"note-use-cases"}),e.jsx(s,{title:"Encoder-Decoder vs. Decoder-Only Scaling",content:"Encoder-decoder models have roughly 2x the parameters of a decoder-only model with the same layer size, since both encoder and decoder have separate stacks. When comparing, a 6B encoder-decoder (like Flan-T5-XL) has comparable parameter count to a 6B decoder-only model but splits capacity between understanding and generation.",id:"warning-param-comparison"}),e.jsx(n,{type:"historical",title:"The Rise and Decline of Encoder-Decoder",content:"The original transformer (2017) was encoder-decoder. T5 (2019) and BART (2019) refined the approach. However, GPT-3 (2020) demonstrated that decoder-only models with sufficient scale could match or surpass encoder-decoder models on most tasks, leading to the current dominance of decoder-only architectures in large-scale LLMs.",id:"note-history"})]})}const V=Object.freeze(Object.defineProperty({__proto__:null,default:g},Symbol.toStringTag,{value:"Module"}));function y(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:'"Attention Is All You Need" — Paper Walkthrough'}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Published by Vaswani et al. at NeurIPS 2017, this paper introduced the transformer architecture, replacing recurrence entirely with self-attention. It remains one of the most influential papers in deep learning, with every modern LLM tracing its lineage to this work."}),e.jsx(n,{type:"historical",title:"Context: The State of NLP in 2017",content:"Before transformers, sequence-to-sequence models relied on LSTMs and GRUs with attention (Bahdanau 2014, Luong 2015). These models processed tokens sequentially, preventing parallelization and struggling with long-range dependencies. The transformer eliminated recurrence entirely, enabling massive parallelism during training.",id:"note-context"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Key Contributions"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The paper introduced several ideas that became standard: scaled dot-product attention, multi-head attention, positional encoding, and the specific combination of residual connections with layer normalization in an encoder-decoder structure."}),e.jsx(a,{title:"The Transformer (Vaswani et al., 2017)",definition:"A sequence-to-sequence model using stacked self-attention and feed-forward layers. The core attention formula: $\\text{Attention}(Q,K,V) = \\text{softmax}\\!\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$ with multi-head extension: $\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1, \\ldots, \\text{head}_h)W^O$.",notation:"Original config: d_model=512, h=8, d_ff=2048, N=6 layers (encoder & decoder)",id:"def-original-transformer"}),e.jsx(o,{title:"Original Paper Configuration",problem:"Calculate key numbers for the base and big transformer models from the paper.",steps:[{formula:"\\text{Base: } d=512, h=8, d_k=64, d_{ff}=2048, N=6",explanation:"Base model: 65M parameters, trained on 8 P100 GPUs for 12 hours."},{formula:"\\text{Big: } d=1024, h=16, d_k=64, d_{ff}=4096, N=6",explanation:"Big model: 213M parameters, trained on 8 P100 GPUs for 3.5 days."},{formula:"\\text{WMT14 EN-DE: 28.4 BLEU (big)}",explanation:"State-of-the-art machine translation, surpassing all existing models."},{formula:"\\text{Training cost: } \\$150{-}\\$1500 \\text{ (2017 prices)}",explanation:"Remarkably cheap by modern standards — GPT-4 cost ~$100M+."}],id:"example-paper-config"}),e.jsx(i,{title:"original_transformer_config.py",code:`import torch
import torch.nn as nn

def count_transformer_params(d_model, num_heads, d_ff, num_layers,
                              src_vocab, tgt_vocab):
    """Estimate parameter count for the original transformer."""
    d_k = d_model // num_heads

    # Per encoder layer
    attn_params = 4 * d_model * d_model  # Q, K, V, O projections
    ffn_params = 2 * d_model * d_ff + d_model + d_ff  # 2 linears + biases
    ln_params = 4 * d_model  # 2 layer norms * (gamma + beta)
    enc_layer = attn_params + ffn_params + ln_params

    # Per decoder layer (extra cross-attention)
    dec_layer = enc_layer + attn_params + 2 * d_model  # + cross-attn + LN

    # Embeddings
    src_emb = src_vocab * d_model
    tgt_emb = tgt_vocab * d_model

    total = (num_layers * enc_layer +
             num_layers * dec_layer +
             src_emb + tgt_emb)
    return total

# Base model
base = count_transformer_params(
    d_model=512, num_heads=8, d_ff=2048,
    num_layers=6, src_vocab=37000, tgt_vocab=37000
)
print(f"Base model: {base:,} params ({base/1e6:.0f}M)")

# Big model
big = count_transformer_params(
    d_model=1024, num_heads=16, d_ff=4096,
    num_layers=6, src_vocab=37000, tgt_vocab=37000
)
print(f"Big model:  {big:,} params ({big/1e6:.0f}M)")

# Build the actual base model using PyTorch
model = nn.Transformer(
    d_model=512, nhead=8, num_encoder_layers=6,
    num_decoder_layers=6, dim_feedforward=2048,
    dropout=0.1, batch_first=True
)
actual_params = sum(p.numel() for p in model.parameters())
print(f"\\nPyTorch base model: {actual_params:,} params")

# The paper's key insight: training speed
print("\\nTraining parallelism comparison:")
print("  LSTM: O(n) sequential steps — cannot parallelize across time")
print("  Transformer: O(1) sequential steps — full sequence in parallel")
print("  Self-attention: O(n^2) compute but all in parallel")`,id:"code-paper-config"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Paper Structure Summary"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Section 1 motivates replacing recurrence. Section 2 reviews prior work. Section 3 presents the architecture (the famous Figure 1). Section 4 explains why self-attention is preferable to recurrence and convolutions. Sections 5-6 show training details and results on WMT translation benchmarks."}),e.jsx(n,{type:"tip",title:"The Most Important Figure in Deep Learning",content:"Figure 1 of the paper (the architecture diagram) is arguably the most reproduced figure in AI research. When reading it, note three types of attention: encoder self-attention (bidirectional), decoder self-attention (causal), and encoder-decoder cross-attention. These three patterns cover nearly all attention variants used today.",id:"note-figure1"}),e.jsx(s,{title:"What the Paper Got Wrong (or Didn't Predict)",content:"The paper focused on machine translation and encoder-decoder models. It did not predict that (1) decoder-only models would dominate, (2) scale would matter more than architecture, (3) the same architecture would work for vision, audio, and code, or (4) emergent abilities would appear at scale. The architecture was far more general than the authors realized.",id:"warning-predictions"}),e.jsx(n,{type:"note",title:"Citation Impact",content:"'Attention Is All You Need' has over 130,000 citations, making it one of the most cited papers in computer science history. The 8 authors went on to found or join leading AI companies (Google Brain/DeepMind, Cohere, Character.AI, Adept, Essential AI, Near), demonstrating the paper's outsized impact on both research and industry.",id:"note-citations"})]})}const O=Object.freeze(Object.defineProperty({__proto__:null,default:y},Symbol.toStringTag,{value:"Module"}));function b(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"The O(n²) Attention Problem"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Standard self-attention computes pairwise interactions between all token pairs, resulting in quadratic time and memory complexity with respect to sequence length. This bottleneck limits the context length that transformers can practically handle."}),e.jsx(a,{title:"Attention Complexity",definition:"For a sequence of length $n$ with model dimension $d$, scaled dot-product attention requires: time $O(n^2 d)$ for the matrix multiplication $QK^T$, and memory $O(n^2)$ to store the attention matrix (per head).",notation:"n = sequence length, d = model/head dimension, h = number of heads",id:"def-complexity"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Where the Quadratic Cost Comes From"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The attention matrix ",e.jsx(t.InlineMath,{math:"A = \\text{softmax}(QK^T / \\sqrt{d_k})"})," has shape ",e.jsx(t.InlineMath,{math:"n \\times n"}),". Computing it requires ",e.jsx(t.InlineMath,{math:"n^2"})," ","dot products, and storing it requires ",e.jsx(t.InlineMath,{math:"n^2"})," floating-point numbers per head. For long sequences, memory is often the binding constraint."]}),e.jsx(t.BlockMath,{math:"\\text{Memory: } n^2 \\cdot h \\cdot \\text{sizeof(float)} \\quad \\text{FLOPs: } 2n^2 d \\text{ per QK}^T + 2n^2 d_v \\text{ per AV}"}),e.jsx(o,{title:"Memory Requirements at Scale",problem:"Calculate attention memory for different sequence lengths with h=32 heads in fp16.",steps:[{formula:"n=2048: \\; 2048^2 \\times 32 \\times 2\\text{B} = 256\\text{MB}",explanation:"Manageable — fits easily on a modern GPU."},{formula:"n=8192: \\; 8192^2 \\times 32 \\times 2\\text{B} = 4\\text{GB}",explanation:"Significant — consumes a large fraction of GPU memory."},{formula:"n=32768: \\; 32768^2 \\times 32 \\times 2\\text{B} = 64\\text{GB}",explanation:"Exceeds most single GPU memory — requires special techniques."},{formula:"n=131072: \\; 131072^2 \\times 32 \\times 2\\text{B} = 1\\text{TB}",explanation:"Impossible with naive attention — FlashAttention or sparse methods required."}],id:"example-memory"}),e.jsx(i,{title:"attention_complexity_benchmark.py",code:`import torch
import torch.nn.functional as F
import math
import time

def benchmark_attention(seq_lengths, d_model=64, num_heads=1, device='cpu'):
    """Benchmark attention time and memory for different sequence lengths."""
    results = []
    for n in seq_lengths:
        Q = torch.randn(1, num_heads, n, d_model, device=device)
        K = torch.randn(1, num_heads, n, d_model, device=device)
        V = torch.randn(1, num_heads, n, d_model, device=device)

        # Time the attention computation
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Memory of attention matrix
        attn_mem_mb = n * n * num_heads * 4 / (1024 ** 2)  # fp32

        results.append({
            'seq_len': n,
            'time_ms': elapsed * 1000,
            'attn_matrix_mb': attn_mem_mb,
        })
        del Q, K, V, scores, attn, out

    return results

# Run benchmark
lengths = [128, 256, 512, 1024, 2048, 4096]
results = benchmark_attention(lengths)

print(f"{'Seq Len':>8} | {'Time (ms)':>10} | {'Attn Mem (MB)':>14} | {'Ratio':>6}")
print("-" * 50)
for i, r in enumerate(results):
    ratio = r['time_ms'] / results[0]['time_ms'] if i > 0 else 1.0
    print(f"{r['seq_len']:>8} | {r['time_ms']:>10.2f} | "
          f"{r['attn_matrix_mb']:>14.2f} | {ratio:>6.1f}x")

# Theoretical scaling
print("\\nTheoretical O(n²) scaling:")
for n in lengths:
    relative = (n / lengths[0]) ** 2
    print(f"  n={n}: {relative:.1f}x vs n={lengths[0]}")`,id:"code-benchmark"}),e.jsx(n,{type:"note",title:"FLOPs vs. Memory Bottleneck",content:"Modern GPUs have enormous compute throughput but limited memory bandwidth. The attention matrix is often memory-bound rather than compute-bound. For n=8192 with 32 heads, the attention matrices alone consume 4GB — before accounting for gradients, activations, and the model itself. This is why memory-efficient methods like FlashAttention focus on IO rather than FLOPs.",id:"note-memory-bound"}),e.jsx(s,{title:"Quadratic Scaling Limits Context Length",content:"Doubling the sequence length quadruples both compute and memory for attention. This is why early transformers used max_len=512 (BERT) or 1024 (GPT-2). Reaching 100K+ context windows required architectural innovations (FlashAttention, sparse patterns, linear attention) and significant engineering.",id:"warning-scaling-limit"}),e.jsx(n,{type:"intuition",title:"Not All Attention Weights Matter",content:"Empirically, attention matrices are often sparse — most weights are near zero. This observation motivates sparse attention methods: if most entries are negligible, we can approximate the full attention by computing only the important entries, reducing O(n²) toward O(n log n) or O(n).",id:"note-sparsity"})]})}const C=Object.freeze(Object.defineProperty({__proto__:null,default:b},Symbol.toStringTag,{value:"Module"}));function k(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Sparse Attention: Local and Strided Patterns"}),e.jsxs("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:["Sparse attention restricts which token pairs can attend to each other, replacing the full ",e.jsx(t.InlineMath,{math:"n \\times n"})," attention matrix with a sparse pattern that computes only a subset of entries. This reduces complexity from"," ",e.jsx(t.InlineMath,{math:"O(n^2)"})," to ",e.jsx(t.InlineMath,{math:"O(n\\sqrt{n})"})," or"," ",e.jsx(t.InlineMath,{math:"O(n \\log n)"}),"."]}),e.jsx(a,{title:"Sparse Attention",definition:"Instead of computing attention over all $n^2$ pairs, each query attends to a fixed subset $S(i) \\subset \\{1, \\ldots, n\\}$ of keys. The attention output becomes $\\text{out}_i = \\sum_{j \\in S(i)} \\alpha_{ij} v_j$ where $|S(i)| \\ll n$.",notation:"S(i) = sparse connectivity set for query position i",id:"def-sparse-attention"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Common Sparsity Patterns"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The two most common patterns are local (sliding window) attention and strided (dilated) attention. Combining them gives each token both fine-grained local context and long-range global reach."}),e.jsx(o,{title:"Local + Strided Pattern (Sparse Transformer)",problem:"For a sequence of 16 tokens with window=4 and stride=4, show the connectivity.",steps:[{formula:"\\text{Local: token 8 attends to } \\{5, 6, 7, 8\\}",explanation:"Sliding window of size 4 — captures nearby context."},{formula:"\\text{Strided: token 8 attends to } \\{0, 4, 8, 12\\}",explanation:"Every 4th token — captures distant context with stride."},{formula:"\\text{Combined: } \\{0, 4, 5, 6, 7, 8, 12\\}",explanation:"Union of local and strided — 7 keys instead of 16."},{formula:"\\text{Complexity: } O(n\\sqrt{n}) \\text{ with } w = \\sqrt{n}",explanation:"Each token attends to O(√n) others, total O(n√n)."}],id:"example-sparse-patterns"}),e.jsx(i,{title:"sparse_attention_patterns.py",code:`import torch
import torch.nn.functional as F
import math

def create_local_mask(seq_len, window_size):
    """Sliding window attention mask."""
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = True
    return mask

def create_strided_mask(seq_len, stride):
    """Strided (dilated) attention mask."""
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        # Attend to positions that are multiples of stride, up to position i
        for j in range(0, i + 1, stride):
            mask[i, j] = True
        mask[i, i] = True  # Always attend to self
    return mask

def create_combined_mask(seq_len, window_size, stride):
    """Combine local and strided patterns."""
    local = create_local_mask(seq_len, window_size)
    strided = create_strided_mask(seq_len, stride)
    return local | strided  # Union

def sparse_attention(Q, K, V, mask):
    """Attention with arbitrary sparse mask."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))
    weights = F.softmax(scores, dim=-1)
    weights = weights.masked_fill(weights.isnan(), 0.0)
    return torch.matmul(weights, V)

seq_len = 16
local_mask = create_local_mask(seq_len, window_size=4)
strided_mask = create_strided_mask(seq_len, stride=4)
combined_mask = create_combined_mask(seq_len, window_size=4, stride=4)

# Count non-zero entries
full_entries = seq_len * seq_len
local_entries = local_mask.sum().item()
strided_entries = strided_mask.sum().item()
combined_entries = combined_mask.sum().item()

print(f"Full attention entries:     {full_entries}")
print(f"Local (w=4) entries:        {local_entries} ({local_entries/full_entries:.1%})")
print(f"Strided (s=4) entries:      {strided_entries} ({strided_entries/full_entries:.1%})")
print(f"Combined entries:           {combined_entries} ({combined_entries/full_entries:.1%})")

# Run sparse attention
Q = K = V = torch.randn(1, seq_len, 64)
out = sparse_attention(Q, K, V, combined_mask)
print(f"\\nSparse attention output: {out.shape}")`,id:"code-sparse"}),e.jsx(n,{type:"note",title:"Longformer's Sliding Window + Global Tokens",content:"Longformer (Beltagy et al., 2020) uses sliding window attention for most tokens but designates certain positions (e.g., [CLS]) as global tokens that attend to and are attended by all positions. This hybrid achieves O(n) complexity while maintaining the ability to aggregate global information.",id:"note-longformer"}),e.jsx(s,{title:"Sparse Patterns Require Custom Kernels",content:"Naive implementation of sparse attention using masks on a dense matrix still computes O(n²) entries — it just zeros some out. True efficiency requires custom CUDA kernels that only compute the non-zero entries. Libraries like xformers and Triton provide optimized sparse attention implementations.",id:"warning-custom-kernels"}),e.jsx(n,{type:"historical",title:"Evolution of Sparse Attention",content:"Sparse Transformer (Child et al., 2019) introduced factored sparse patterns. BigBird (2020) added random attention connections for theoretical completeness. Longformer (2020) popularized sliding window + global. Mistral (2023) showed that sliding window attention works well even for large-scale LLMs.",id:"note-history"})]})}const I=Object.freeze(Object.defineProperty({__proto__:null,default:k},Symbol.toStringTag,{value:"Module"}));function w(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Linear Attention via Kernel Methods"}),e.jsxs("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:["Linear attention replaces the softmax with a decomposable kernel function, enabling the attention computation to be rewritten with matrix multiplications in a different order — reducing complexity from ",e.jsx(t.InlineMath,{math:"O(n^2 d)"})," to"," ",e.jsx(t.InlineMath,{math:"O(n d^2)"}),", which is linear in sequence length."]}),e.jsx(a,{title:"Linear Attention",definition:"Replace $\\text{softmax}(QK^T)V$ with $\\phi(Q)(\\phi(K)^T V)$ where $\\phi$ is a feature map. By computing $\\phi(K)^T V \\in \\mathbb{R}^{d \\times d}$ first, we avoid materializing the $n \\times n$ attention matrix.",notation:"φ = kernel feature map, typically φ(x) = elu(x) + 1 or φ(x) = exp(x) (approximate softmax)",id:"def-linear-attention"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The Kernel Trick for Attention"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["Standard attention: compute ",e.jsx(t.InlineMath,{math:"(QK^T)V"})," left-to-right, creating the ",e.jsx(t.InlineMath,{math:"n \\times n"})," matrix first. Linear attention: compute"," ",e.jsx(t.InlineMath,{math:"Q(K^T V)"})," right-to-left, creating a ",e.jsx(t.InlineMath,{math:"d \\times d"})," ","matrix instead. When ",e.jsx(t.InlineMath,{math:"d \\ll n"}),", this is dramatically cheaper."]}),e.jsx(t.BlockMath,{math:"\\text{Standard: } \\underbrace{(QK^T)}_{n \\times n} V \\quad \\text{vs.} \\quad \\text{Linear: } \\phi(Q) \\underbrace{(\\phi(K)^T V)}_{d \\times d_v}"}),e.jsx(o,{title:"Complexity Comparison",problem:"Compare FLOPs for standard vs. linear attention with n=8192, d=128.",steps:[{formula:"\\text{Standard: } O(n^2 d) = O(8192^2 \\times 128) \\approx 8.6 \\times 10^9",explanation:"Quadratic in n: must compute all pairwise interactions."},{formula:"\\text{Linear: } O(n d^2) = O(8192 \\times 128^2) \\approx 1.3 \\times 10^8",explanation:"Linear in n: compute KV first, then multiply by Q."},{formula:"\\text{Speedup: } \\frac{n}{d} = \\frac{8192}{128} = 64\\times",explanation:"Linear attention is faster when n >> d (long sequences)."}],id:"example-complexity"}),e.jsx(i,{title:"linear_attention.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

def standard_attention(Q, K, V):
    """O(n^2 d) standard attention."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)

def linear_attention(Q, K, V, feature_map='elu'):
    """O(n d^2) linear attention using kernel feature maps."""
    if feature_map == 'elu':
        phi_Q = F.elu(Q) + 1    # Ensure non-negative
        phi_K = F.elu(K) + 1
    elif feature_map == 'relu':
        phi_Q = F.relu(Q)
        phi_K = F.relu(K)
    else:
        raise ValueError(f"Unknown feature map: {feature_map}")

    # Key insight: compute K^T V first (d x d), then multiply by Q
    # Standard: (n x n) @ (n x d) = O(n^2 d)
    # Linear:   (n x d) @ ((d x n) @ (n x d)) = (n x d) @ (d x d) = O(n d^2)
    KV = torch.matmul(phi_K.transpose(-2, -1), V)    # (d_k, d_v)
    Z = phi_K.transpose(-2, -1).sum(dim=-1, keepdim=True)  # normalizer
    out = torch.matmul(phi_Q, KV)                      # (n, d_v)
    normalizer = torch.matmul(phi_Q, Z) + 1e-6         # (n, 1)
    return out / normalizer

def causal_linear_attention(Q, K, V, feature_map='elu'):
    """Causal linear attention using cumulative sums (RNN-like)."""
    if feature_map == 'elu':
        phi_Q = F.elu(Q) + 1
        phi_K = F.elu(K) + 1
    else:
        phi_Q = F.relu(Q) + 1e-6
        phi_K = F.relu(K) + 1e-6

    B, n, d = phi_Q.shape
    d_v = V.size(-1)

    # Running state: accumulate K^T V incrementally
    S = torch.zeros(B, d, d_v, device=Q.device)  # Running KV state
    Z = torch.zeros(B, d, 1, device=Q.device)    # Running normalizer
    outputs = []

    for t in range(n):
        k_t = phi_K[:, t:t+1, :]     # (B, 1, d)
        v_t = V[:, t:t+1, :]          # (B, 1, d_v)
        q_t = phi_Q[:, t:t+1, :]      # (B, 1, d)

        S = S + torch.matmul(k_t.transpose(-2, -1), v_t)
        Z = Z + k_t.transpose(-2, -1)

        out_t = torch.matmul(q_t, S) / (torch.matmul(q_t, Z) + 1e-6)
        outputs.append(out_t)

    return torch.cat(outputs, dim=1)

# Compare outputs
B, n, d = 2, 64, 32
Q = torch.randn(B, n, d)
K = torch.randn(B, n, d)
V = torch.randn(B, n, d)

std_out = standard_attention(Q, K, V)
lin_out = linear_attention(Q, K, V)
print(f"Standard output: {std_out.shape}")
print(f"Linear output:   {lin_out.shape}")
print(f"Max difference:  {(std_out - lin_out).abs().max():.4f}")
print("(Outputs differ because kernel is approximate, not exact softmax)")`,id:"code-linear"}),e.jsx(n,{type:"intuition",title:"Linear Attention as an RNN",content:"Causal linear attention can be computed as a recurrence: maintain a running state S = Σ φ(k_t) v_t^T and normalizer Z = Σ φ(k_t). Each new output is q_t^T S / q_t^T Z. This makes it equivalent to a linear RNN, enabling O(1) per-token generation without a KV cache.",id:"note-rnn-connection"}),e.jsx(s,{title:"Quality Degradation",content:"Linear attention approximates softmax attention but does not replicate it exactly. The softmax's ability to create sharp, peaked distributions is lost with most kernel feature maps. Models trained with linear attention typically underperform standard attention on language modeling, which is why it has not replaced softmax in state-of-the-art LLMs.",id:"warning-quality"}),e.jsx(n,{type:"note",title:"Modern Linear-Time Models",content:"The idea of linear-time sequence modeling has evolved beyond kernel attention into state-space models (Mamba, S4) and linear RNNs (RWKV, RetNet). These achieve competitive quality with O(n) complexity by using structured recurrences rather than attention approximations.",id:"note-modern-linear"})]})}const W=Object.freeze(Object.defineProperty({__proto__:null,default:w},Symbol.toStringTag,{value:"Module"}));function v(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"FlashAttention: IO-Aware Exact Attention"}),e.jsxs("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:["FlashAttention computes exact softmax attention without materializing the full"," ",e.jsx(t.InlineMath,{math:"n \\times n"})," attention matrix in GPU HBM. By using tiling and recomputation, it reduces memory usage from ",e.jsx(t.InlineMath,{math:"O(n^2)"})," to"," ",e.jsx(t.InlineMath,{math:"O(n)"})," while also running 2-4x faster than standard attention by minimizing HBM reads/writes."]}),e.jsx(a,{title:"FlashAttention",definition:"An IO-aware attention algorithm that tiles the Q, K, V matrices into blocks that fit in GPU SRAM (shared memory), computes partial attention within tiles using the online softmax trick, and never writes the full $n \\times n$ attention matrix to HBM.",notation:"HBM = High Bandwidth Memory (GPU main memory, ~2TB/s), SRAM = on-chip memory (~19TB/s but only ~20MB)",id:"def-flash-attention"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The GPU Memory Hierarchy"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The key insight is that GPU compute (TFLOPs) has grown much faster than memory bandwidth. Standard attention is bottlenecked by reading/writing the ",e.jsx(t.InlineMath,{math:"n^2"})," ","attention matrix to HBM. FlashAttention keeps intermediate results in fast SRAM."]}),e.jsx(o,{title:"IO Cost: Standard vs. FlashAttention",problem:"Compare HBM reads/writes for n=4096, d=64 with block size B=256.",steps:[{formula:"\\text{Standard: read } Q,K,V \\in O(nd) + \\text{write } A \\in O(n^2)",explanation:"Must write the full 4096² attention matrix to HBM."},{formula:"\\text{Standard IO: } O(n^2) = 4096^2 \\approx 16.7\\text{M elements}",explanation:"Memory bandwidth becomes the bottleneck, not compute."},{formula:"\\text{Flash: read } Q,K,V \\text{ in blocks of } B=256",explanation:"Process 256-token tiles that fit in SRAM (~100KB each)."},{formula:"\\text{Flash IO: } O(n^2 d / M) \\text{ where } M = \\text{SRAM size}",explanation:"With sufficient SRAM, IO is subquadratic. Never writes full attention matrix."}],id:"example-io-cost"}),e.jsx(n,{type:"intuition",title:"The Online Softmax Trick",content:"The main challenge with tiling is that softmax requires knowing the maximum value across the entire row (for numerical stability) before computing exp(). The online softmax algorithm (Milakov & Gimelshein, 2018) maintains running max and sum statistics, allowing softmax to be computed incrementally across tiles without seeing the full row at once.",id:"note-online-softmax"}),e.jsx(i,{title:"flash_attention_concept.py",code:`import torch
import torch.nn.functional as F
import math

def flash_attention_simplified(Q, K, V, block_size=256):
    """
    Simplified FlashAttention in pure PyTorch (for understanding).
    Real FlashAttention uses custom CUDA kernels for SRAM tiling.
    """
    B, n, d = Q.shape
    output = torch.zeros_like(V)
    row_max = torch.full((B, n, 1), float('-inf'), device=Q.device)
    row_sum = torch.zeros(B, n, 1, device=Q.device)

    num_blocks = math.ceil(n / block_size)

    for j in range(num_blocks):
        # Load one block of K, V
        j_start = j * block_size
        j_end = min(j_start + block_size, n)
        K_block = K[:, j_start:j_end, :]
        V_block = V[:, j_start:j_end, :]

        # Compute scores for this block
        scores = torch.matmul(Q, K_block.transpose(-2, -1)) / math.sqrt(d)

        # Online softmax update
        block_max = scores.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(row_max, block_max)

        # Rescale previous accumulator
        exp_old = torch.exp(row_max - new_max)
        exp_new = torch.exp(scores - new_max)

        # Update output and statistics
        output = output * exp_old
        output = output + torch.matmul(exp_new, V_block)
        row_sum = row_sum * exp_old + exp_new.sum(dim=-1, keepdim=True)
        row_max = new_max

    # Final normalization
    output = output / row_sum
    return output

# Verify correctness against standard attention
B, n, d = 2, 1024, 64
Q = torch.randn(B, n, d)
K = torch.randn(B, n, d)
V = torch.randn(B, n, d)

standard_out = F.scaled_dot_product_attention(Q, K, V)
flash_out = flash_attention_simplified(Q, K, V, block_size=128)
max_diff = (standard_out - flash_out).abs().max().item()
print(f"Max difference: {max_diff:.8f}")  # Should be ~1e-6 (numerical)
print(f"Results match: {max_diff < 1e-4}")

# PyTorch 2.0+ uses FlashAttention automatically
print("\\nPyTorch SDPA backends:")
print(f"  FlashAttention available: {torch.backends.cuda.flash_sdp_enabled()}"
      if hasattr(torch.backends, 'cuda') else "  (CPU — no FlashAttention)")

# Memory comparison (conceptual)
n_vals = [1024, 4096, 16384]
for n in n_vals:
    standard_mem = n * n * 4 / 1e6   # fp32 attention matrix in MB
    flash_mem = 2 * n * d * 4 / 1e6  # Only Q, K, V blocks + O(n) stats
    print(f"n={n:>6}: standard={standard_mem:>8.1f}MB, flash={flash_mem:>6.1f}MB, "
          f"saving={standard_mem/flash_mem:.0f}x")`,id:"code-flash"}),e.jsx(s,{title:"FlashAttention Is Not an Approximation",content:"Unlike sparse or linear attention, FlashAttention computes the exact same result as standard softmax attention (up to floating-point precision). It is purely an implementation optimization — same math, different computation order. This is why it has become universally adopted: no quality tradeoff.",id:"warning-exact"}),e.jsx(n,{type:"note",title:"FlashAttention Versions",content:"FlashAttention v1 (Dao et al., 2022) introduced tiled attention with online softmax. FlashAttention v2 (2023) improved parallelism and work partitioning for 2x further speedup. FlashAttention v3 (2024) leveraged H100-specific features (TMA, FP8). The algorithm is now the default backend in PyTorch's scaled_dot_product_attention.",id:"note-versions"}),e.jsx(n,{type:"tip",title:"Using FlashAttention in Practice",content:"In PyTorch 2.0+, simply use torch.nn.functional.scaled_dot_product_attention(Q, K, V) — it automatically selects FlashAttention when available. For Hugging Face models, pass attn_implementation='flash_attention_2' to from_pretrained(). No code changes needed for the standard case.",id:"note-practical"})]})}const D=Object.freeze(Object.defineProperty({__proto__:null,default:v},Symbol.toStringTag,{value:"Module"}));function j(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Sinusoidal Positional Encoding: The Math"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The sinusoidal positional encoding from the original transformer encodes each position as a unique vector using sine and cosine functions at geometrically spaced frequencies. This design has elegant mathematical properties that enable the model to learn relative position attention."}),e.jsx(a,{title:"Sinusoidal Encoding Formulas",definition:"$PE_{(\\text{pos}, 2i)} = \\sin\\!\\left(\\frac{\\text{pos}}{10000^{2i/d}}\\right), \\quad PE_{(\\text{pos}, 2i+1)} = \\cos\\!\\left(\\frac{\\text{pos}}{10000^{2i/d}}\\right)$ for dimension pairs $i = 0, 1, \\ldots, d/2 - 1$.",notation:"Each dimension pair (2i, 2i+1) oscillates at wavelength λ_i = 2π · 10000^{2i/d}",id:"def-sinusoidal-formulas"}),e.jsx(r,{title:"Relative Position via Linear Transformation",statement:"For any fixed offset $k$, there exists a linear transformation $M_k$ (independent of position) such that $PE_{\\text{pos}+k} = M_k \\cdot PE_{\\text{pos}}$. Specifically, for each dimension pair, $M_k$ is a 2D rotation matrix with angle $k \\cdot \\omega_i$.",proofSteps:["\\text{Let } \\omega_i = 1/10000^{2i/d}. \\text{ Then for dimension pair } (2i, 2i+1):","PE_{\\text{pos}}^{(2i, 2i+1)} = (\\sin(\\omega_i \\cdot \\text{pos}), \\cos(\\omega_i \\cdot \\text{pos}))","PE_{\\text{pos}+k}^{(2i, 2i+1)} = (\\sin(\\omega_i(\\text{pos}+k)), \\cos(\\omega_i(\\text{pos}+k)))","\\text{By angle addition: } \\begin{pmatrix} \\sin(\\alpha+\\beta) \\\\ \\cos(\\alpha+\\beta) \\end{pmatrix} = \\begin{pmatrix} \\cos\\beta & \\sin\\beta \\\\ -\\sin\\beta & \\cos\\beta \\end{pmatrix} \\begin{pmatrix} \\sin\\alpha \\\\ \\cos\\alpha \\end{pmatrix}","\\text{So } M_k \\text{ is block-diagonal with 2x2 rotation matrices } R(k\\omega_i) \\text{ for each pair.}"],id:"thm-relative-position"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Frequency Spectrum"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The wavelengths span from ",e.jsx(t.InlineMath,{math:"2\\pi \\approx 6.28"})," (dimension 0) to"," ",e.jsx(t.InlineMath,{math:"2\\pi \\times 10000 \\approx 62{,}832"})," (dimension d-1). Lower dimensions change rapidly with position (fine-grained), while higher dimensions change slowly (coarse-grained), similar to a binary counter or Fourier features."]}),e.jsx(o,{title:"Wavelength Distribution for d_model=512",problem:"Calculate the wavelength for the first, middle, and last dimension pairs.",steps:[{formula:"\\lambda_0 = 2\\pi \\cdot 10000^{0/512} = 2\\pi \\approx 6.28",explanation:"First pair: completes a full cycle every ~6 positions."},{formula:"\\lambda_{128} = 2\\pi \\cdot 10000^{256/512} = 2\\pi \\cdot 100 \\approx 628",explanation:"Middle pair: one cycle per ~628 positions."},{formula:"\\lambda_{255} = 2\\pi \\cdot 10000^{510/512} \\approx 62{,}208",explanation:"Last pair: barely changes over typical sequence lengths."}],id:"example-wavelengths"}),e.jsx(i,{title:"sinusoidal_deep_dive.py",code:`import torch
import math

def sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

pe = sinusoidal_pe(1024, 512)

# Property 1: Each position has a unique encoding
norms = torch.cdist(pe[:100], pe[:100])
print("Min distance between different positions:",
      norms[norms > 0].min().item())  # > 0

# Property 2: Relative position dot product
# PE[pos] · PE[pos+k] depends mainly on k, not pos
def dot_product_at_offset(pe, offset, num_samples=50):
    dots = []
    for p in range(num_samples):
        if p + offset < pe.size(0):
            dots.append(torch.dot(pe[p], pe[p + offset]).item())
    return torch.tensor(dots)

for k in [1, 5, 10, 50]:
    dots = dot_product_at_offset(pe, k)
    print(f"Offset k={k:>2}: mean dot={dots.mean():.2f}, "
          f"std={dots.std():.4f}")

# Property 3: The linear transformation M_k
def rotation_matrix_2d(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, s], [-s, c]])

# Verify: PE[pos+k] = M_k @ PE[pos] for each dimension pair
pos, k = 7, 3
d_model = 512
for i in range(3):  # Check first 3 dimension pairs
    omega_i = 1.0 / (10000 ** (2 * i / d_model))
    M_k = rotation_matrix_2d(k * omega_i)
    pe_pos = pe[pos, 2*i:2*i+2]
    pe_pos_k = pe[pos + k, 2*i:2*i+2]
    predicted = M_k @ pe_pos
    error = (pe_pos_k - predicted).abs().max().item()
    print(f"Dim pair {i}: prediction error = {error:.2e}")  # ~1e-7`,id:"code-sinusoidal"}),e.jsx(n,{type:"tip",title:"Visualizing Positional Encodings",content:"Plot PE as a heatmap with position on the y-axis and dimension on the x-axis. You will see alternating vertical stripes (low dimensions change fast) transitioning to nearly solid bands (high dimensions change slowly). This pattern resembles a binary counter viewed in analog form.",id:"note-visualization"}),e.jsx(s,{title:"Sinusoidal Encodings Do Not Extrapolate Well",content:"Despite the elegant theory that M_k exists for any k, models trained with sinusoidal encodings at max_len=512 perform poorly at position 1000. The issue is not the encoding itself but that the model's attention weights were never trained to handle those position patterns. This limitation motivated learned and relative position methods.",id:"warning-extrapolation"})]})}const G=Object.freeze(Object.defineProperty({__proto__:null,default:j},Symbol.toStringTag,{value:"Module"}));function q(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"RoPE: Rotary Position Embeddings"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Rotary Position Embeddings (RoPE) encode position by rotating query and key vectors in 2D subspaces. This elegant method naturally encodes relative positions in the attention dot product and has become the dominant positional encoding in modern LLMs including LLaMA, Mistral, Qwen, and GPT-NeoX."}),e.jsx(a,{title:"Rotary Position Embedding (RoPE)",definition:"For position $m$, RoPE applies a rotation to each consecutive pair of dimensions: $f(x_m, m) = R_{\\Theta,m} x_m$ where $R_{\\Theta,m}$ is a block-diagonal matrix of 2D rotations. The key property: $\\langle f(q_m, m), f(k_n, n) \\rangle = g(q_m, k_n, m-n)$, making the dot product depend only on relative position $m - n$.",notation:"Θ = {θ_i = 10000^{-2i/d}} are the rotation frequencies (same as sinusoidal PE)",id:"def-rope"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The Rotation Mechanism"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["For each pair of dimensions ",e.jsx(t.InlineMath,{math:"(2i, 2i+1)"}),", RoPE rotates the query and key vectors by angle ",e.jsx(t.InlineMath,{math:"m\\theta_i"})," at position"," ",e.jsx(t.InlineMath,{math:"m"}),". When computing ",e.jsx(t.InlineMath,{math:"q_m^T k_n"}),", the absolute rotations cancel and only the relative rotation"," ",e.jsx(t.InlineMath,{math:"(m-n)\\theta_i"})," remains."]}),e.jsx(t.BlockMath,{math:"R_{\\Theta,m} = \\begin{pmatrix} \\cos m\\theta_1 & -\\sin m\\theta_1 & & \\\\ \\sin m\\theta_1 & \\cos m\\theta_1 & & \\\\ & & \\ddots & \\\\ & & & \\cos m\\theta_{d/2} & -\\sin m\\theta_{d/2} \\\\ & & & \\sin m\\theta_{d/2} & \\cos m\\theta_{d/2} \\end{pmatrix}"}),e.jsx(o,{title:"RoPE Relative Position Property",problem:"Show that q_m^T k_n depends only on m - n for a single dimension pair.",steps:[{formula:"q_m^{\\text{rot}} = (q_1 \\cos m\\theta - q_2 \\sin m\\theta,\\; q_1 \\sin m\\theta + q_2 \\cos m\\theta)",explanation:"Rotate the query 2D vector by angle mθ."},{formula:"k_n^{\\text{rot}} = (k_1 \\cos n\\theta - k_2 \\sin n\\theta,\\; k_1 \\sin n\\theta + k_2 \\cos n\\theta)",explanation:"Rotate the key 2D vector by angle nθ."},{formula:"q_m^{\\text{rot}} \\cdot k_n^{\\text{rot}} = (q_1 k_1 + q_2 k_2)\\cos(m-n)\\theta + (q_1 k_2 - q_2 k_1)\\sin(m-n)\\theta",explanation:"The dot product depends on (m-n)θ — only relative position matters."}],id:"example-rope-relative"}),e.jsx(i,{title:"rotary_position_embeddings.py",code:`import torch
import torch.nn as nn

def precompute_rope_freqs(dim, max_len, theta=10000.0):
    """Precompute RoPE rotation frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_len).float()
    # Outer product: (max_len, dim/2)
    angles = torch.outer(positions, freqs)
    # Stack cos and sin
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    """Apply rotary embeddings to input tensor.
    x: (batch, seq_len, num_heads, d_head) or (batch, num_heads, seq_len, d_head)
    """
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]

    # Reshape cos/sin for broadcasting
    cos = cos[:x1.shape[-2]].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, d/2)
    sin = sin[:x1.shape[-2]].unsqueeze(0).unsqueeze(0)

    # Apply rotation
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)

# Setup
d_head = 64
max_len = 2048
cos_cached, sin_cached = precompute_rope_freqs(d_head, max_len)

# Verify relative position property
B, n, h = 1, 32, 8
Q = torch.randn(B, h, n, d_head)
K = torch.randn(B, h, n, d_head)

Q_rot = apply_rope(Q, cos_cached, sin_cached)
K_rot = apply_rope(K, cos_cached, sin_cached)

# Dot product Q[pos_m] · K[pos_n] should depend on m-n
scores = torch.matmul(Q_rot, K_rot.transpose(-2, -1))
print(f"Attention scores shape: {scores.shape}")  # [1, 8, 32, 32]

# RoPE preserves vector norms (rotation doesn't change length)
norm_before = Q[0, 0, 0].norm().item()
norm_after = Q_rot[0, 0, 0].norm().item()
print(f"Q norm before RoPE: {norm_before:.4f}")
print(f"Q norm after RoPE:  {norm_after:.4f}")
print(f"Norm preserved: {abs(norm_before - norm_after) < 1e-5}")

# RoPE is only applied to Q and K, NOT to V
print("\\nRoPE applied to: Q (queries), K (keys)")
print("RoPE NOT applied to: V (values) — values carry content, not position")`,id:"code-rope"}),e.jsx(n,{type:"intuition",title:"Why RoPE Works So Well",content:"RoPE combines the best of absolute and relative position encoding. It injects absolute position into Q and K (each position gets a unique rotation), but the attention score naturally depends only on relative position (rotations compose). Unlike additive position encodings, RoPE modifies the dot product geometry directly, making position information inseparable from content.",id:"note-why-rope-works"}),e.jsx(s,{title:"RoPE Base Frequency and Context Length",content:"The base frequency θ=10000 determines the longest wavelength and thus the effective context window. For longer contexts, models increase θ (e.g., Code Llama uses θ=1M for 100K context). NTK-aware scaling and YaRN further modify the frequency schedule to extrapolate beyond training lengths without fine-tuning.",id:"warning-base-freq"}),e.jsx(n,{type:"note",title:"RoPE Adoption",content:"RoPE (Su et al., 2021) was first adopted by GPT-NeoX and PaLM. LLaMA's success cemented RoPE as the standard. Nearly all modern open-weight LLMs use RoPE: LLaMA 2/3, Mistral, Qwen, Gemma, DeepSeek, Yi, and Phi. The main exception is models using ALiBi (like BLOOM and MPT).",id:"note-adoption"})]})}const U=Object.freeze(Object.defineProperty({__proto__:null,default:q},Symbol.toStringTag,{value:"Module"}));function T(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"ALiBi: Attention with Linear Biases"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"ALiBi (Attention with Linear Biases) takes a radically simple approach to position encoding: instead of adding position embeddings to token representations, it adds a linear distance-based bias directly to the attention scores. This requires zero learned parameters and enables strong length extrapolation."}),e.jsx(a,{title:"ALiBi (Press et al., 2022)",definition:"ALiBi adds a static bias to attention scores: $\\text{score}_{ij} = q_i^T k_j - m \\cdot |i - j|$ where $m$ is a head-specific slope. No positional embeddings are added to the token representations.",notation:"m = head slope (geometric sequence), |i-j| = distance between positions",id:"def-alibi"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The Linear Bias"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["Each head uses a different slope ",e.jsx(t.InlineMath,{math:"m"})," from a geometric sequence. For ",e.jsx(t.InlineMath,{math:"h"})," heads, the slopes are"," ",e.jsx(t.InlineMath,{math:"2^{-8/h}, 2^{-16/h}, \\ldots, 2^{-8}"}),". Heads with small slopes attend broadly; heads with large slopes attend locally."]}),e.jsx(t.BlockMath,{math:"\\text{ALiBi bias}_{ij}^{(k)} = -m_k \\cdot |i - j|, \\quad m_k = 2^{-8k/h}, \\; k = 1, \\ldots, h"}),e.jsx(o,{title:"ALiBi Slopes for 8 Heads",problem:"Calculate the slopes and their effect on attention range.",steps:[{formula:"m_1 = 2^{-1} = 0.5",explanation:"Steepest slope — strongly penalizes distant tokens."},{formula:"m_4 = 2^{-4} = 0.0625",explanation:"Medium slope — moderate distance penalty."},{formula:"m_8 = 2^{-8/8 \\cdot 8} = 2^{-8} \\approx 0.0039",explanation:"Shallowest slope — nearly uniform attention across positions."},{formula:"\\text{At distance 100: bias}_1 = -50, \\text{ bias}_8 = -0.39",explanation:"Head 1 ignores distant tokens; head 8 can attend to the whole context."}],id:"example-slopes"}),e.jsx(i,{title:"alibi_attention.py",code:`import torch
import torch.nn.functional as F
import math

def get_alibi_slopes(num_heads):
    """Compute ALiBi slopes for each head."""
    # Geometric sequence: 2^(-8/h), 2^(-16/h), ..., 2^(-8)
    ratio = 2 ** (-8 / num_heads)
    slopes = torch.tensor([ratio ** (i + 1) for i in range(num_heads)])
    return slopes

def build_alibi_bias(seq_len, num_heads):
    """Build the full ALiBi bias matrix."""
    slopes = get_alibi_slopes(num_heads)  # (num_heads,)

    # Distance matrix |i - j| for causal attention (j <= i)
    positions = torch.arange(seq_len)
    # For causal: distance is i - j (always >= 0)
    distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq, seq)

    # For causal masking, set future positions to -inf later
    bias = -slopes.unsqueeze(1).unsqueeze(2) * distance.abs().unsqueeze(0)
    return bias  # (num_heads, seq_len, seq_len)

def alibi_attention(Q, K, V, alibi_bias, causal=True):
    """Attention with ALiBi bias (no positional embeddings on Q, K, V)."""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Add ALiBi bias
    scores = scores + alibi_bias[:, :scores.size(-2), :scores.size(-1)]

    if causal:
        mask = torch.triu(torch.ones_like(scores[0]), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights

# Demo
num_heads = 8
seq_len = 32
d_head = 64

slopes = get_alibi_slopes(num_heads)
print("ALiBi slopes:", [f"{s:.6f}" for s in slopes.tolist()])

alibi_bias = build_alibi_bias(seq_len, num_heads)
print(f"\\nALiBi bias shape: {alibi_bias.shape}")  # [8, 32, 32]

# No positional embeddings — just raw token embeddings
Q = torch.randn(num_heads, seq_len, d_head)
K = torch.randn(num_heads, seq_len, d_head)
V = torch.randn(num_heads, seq_len, d_head)

out, weights = alibi_attention(Q, K, V, alibi_bias)
print(f"Output shape: {out.shape}")

# Length extrapolation: bias extends naturally to longer sequences
long_bias = build_alibi_bias(seq_len * 4, num_heads)
print(f"\\nExtrapolated bias shape: {long_bias.shape}")  # [8, 128, 128]
print("No retraining needed — bias is a fixed function of distance")`,id:"code-alibi"}),e.jsx(n,{type:"intuition",title:"Why ALiBi Extrapolates",content:"ALiBi's bias is a simple linear function of distance with no learned parameters. At inference, extending to longer sequences just means computing the bias for larger distances — the penalty function is the same regardless of sequence length. The model learns to use content-based attention with a distance prior, and this prior generalizes naturally.",id:"note-extrapolation"}),e.jsx(s,{title:"ALiBi vs. RoPE Tradeoffs",content:"ALiBi is simpler and extrapolates better out-of-the-box, but RoPE generally achieves higher quality within the training context length. Most state-of-the-art LLMs (LLaMA, Mistral, Qwen) chose RoPE with frequency scaling rather than ALiBi. ALiBi is used in BLOOM and MPT but has fallen out of favor for the largest models.",id:"warning-vs-rope"}),e.jsx(n,{type:"note",title:"No Positional Embeddings Needed",content:"ALiBi's key simplification is that it removes positional embeddings entirely. Token representations are pure content vectors. Position information exists only in the attention bias. This reduces model parameters (no position embedding table) and simplifies the architecture.",id:"note-no-embeddings"}),e.jsx(n,{type:"historical",title:"ALiBi's Influence",content:"Press et al. (2022) showed that ALiBi trained on 1024 tokens could extrapolate to 2048+ at inference with minimal degradation. This was groundbreaking at the time. While RoPE ultimately won the adoption race, ALiBi's insight — that position need not be in the embeddings — influenced later work on position extrapolation.",id:"note-history"})]})}const H=Object.freeze(Object.defineProperty({__proto__:null,default:T},Symbol.toStringTag,{value:"Module"}));function N(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Extending Context Length Beyond Training"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"A model trained on sequences of length L often fails catastrophically at length 2L. Context extrapolation techniques modify positional encodings at inference time (or with minimal fine-tuning) to extend the effective context window, enabling models trained on 4K tokens to handle 128K or more."}),e.jsx(a,{title:"Context Length Extrapolation",definition:"The ability of a model to process sequences longer than its training context length $L_{\\text{train}}$ while maintaining quality. Key methods modify the positional encoding frequency or scale: position interpolation scales positions to fit within $[0, L_{\\text{train}}]$, while NTK-aware methods adjust the frequency basis.",id:"def-extrapolation"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Position Interpolation (PI)"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["Instead of extrapolating beyond trained positions, PI linearly downscales all positions to fit within the original range. For a model trained on L=4096 used at L'=16384, position",e.jsx(t.InlineMath,{math:"p"})," becomes ",e.jsx(t.InlineMath,{math:"p \\cdot L / L'"}),"."]}),e.jsx(t.BlockMath,{math:"\\text{PI: } \\theta'_i = \\theta_i \\cdot \\frac{L_{\\text{train}}}{L_{\\text{target}}}, \\quad \\text{NTK: } \\theta'_i = \\left(\\frac{L_{\\text{target}}}{L_{\\text{train}}}\\right)^{2i/(d-2)} \\cdot \\theta_i"}),e.jsx(o,{title:"Methods for Extending from 4K to 32K",problem:"Compare position interpolation, NTK scaling, and YaRN for 8x extension.",steps:[{formula:"\\text{PI: scale all } \\theta_i \\text{ by } 1/8",explanation:"All frequencies reduced uniformly. Needs ~1000 steps of fine-tuning."},{formula:"\\text{NTK-aware: scale } \\theta_i \\text{ non-uniformly}",explanation:"Low frequencies scaled more, high frequencies kept — preserves local resolution."},{formula:"\\text{YaRN: NTK + temperature + attention scaling}",explanation:"Combines NTK with attention logit scaling. Best zero-shot extrapolation."},{formula:"\\text{Dynamic NTK: adjust } \\theta \\text{ per sequence length}",explanation:"Scale only as much as needed — works without any fine-tuning."}],id:"example-methods"}),e.jsx(i,{title:"context_extrapolation.py",code:`import torch
import math

def rope_freqs(dim, base=10000.0):
    """Standard RoPE frequencies."""
    return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

def position_interpolation(dim, scale_factor, base=10000.0):
    """Position Interpolation: uniformly scale all frequencies."""
    freqs = rope_freqs(dim, base)
    return freqs / scale_factor

def ntk_aware_scaling(dim, scale_factor, base=10000.0):
    """NTK-aware scaling: increase base frequency."""
    new_base = base * (scale_factor ** (dim / (dim - 2)))
    return rope_freqs(dim, new_base)

def yarn_scaling(dim, scale_factor, base=10000.0, beta_fast=32, beta_slow=1):
    """YaRN: interpolate between PI and NTK based on frequency."""
    freqs = rope_freqs(dim, base)
    # Low frequencies get PI (more interpolation)
    # High frequencies stay unchanged (NTK-like)
    low_freq_factor = 1.0
    high_freq_factor = scale_factor

    wavelengths = 2 * math.pi / freqs
    ramp = (wavelengths / (beta_fast * 2 * math.pi) - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    ramp = ramp.clamp(0, 1)

    # Smooth interpolation between scaled and unscaled
    scaled_freqs = freqs / scale_factor
    yarn_freqs = (1 - ramp) * scaled_freqs + ramp * freqs
    return yarn_freqs

# Compare frequency distributions
dim = 64
scale = 8  # Extend 4K -> 32K

original = rope_freqs(dim)
pi = position_interpolation(dim, scale)
ntk = ntk_aware_scaling(dim, scale)
yarn = yarn_scaling(dim, scale)

print(f"{'Dim pair':>10} | {'Original':>12} | {'PI (÷8)':>12} | {'NTK':>12} | {'YaRN':>12}")
print("-" * 65)
for i in range(0, dim // 2, 4):
    print(f"{i:>10} | {original[i]:>12.6f} | {pi[i]:>12.6f} | "
          f"{ntk[i]:>12.6f} | {yarn[i]:>12.6f}")

# Key insight: high-frequency dimensions (small i) handle local patterns
# and should NOT be scaled much. Low-frequency dimensions handle global
# patterns and can be safely compressed.
print("\\nScaling ratio (method / original):")
print(f"  PI dim 0:   {pi[0]/original[0]:.4f} (high freq, LOCAL — over-compressed!)")
print(f"  PI dim 15:  {pi[15]/original[15]:.4f} (low freq, GLOBAL — correct)")
print(f"  NTK dim 0:  {ntk[0]/original[0]:.4f} (high freq preserved)")
print(f"  NTK dim 15: {ntk[15]/original[15]:.4f} (low freq compressed)")
print(f"  YaRN dim 0: {yarn[0]/original[0]:.4f} (high freq mostly preserved)")
print(f"  YaRN dim 15:{yarn[15]/original[15]:.4f} (low freq compressed)")`,id:"code-extrapolation"}),e.jsx(n,{type:"intuition",title:"Why Uniform Scaling Hurts Local Patterns",content:"Position Interpolation scales all frequencies equally, but high-frequency dimensions encode local patterns (nearby token relationships). Compressing these means the model can no longer distinguish between adjacent tokens as well. NTK-aware methods preserve high frequencies and only compress low frequencies, maintaining local resolution while extending global range.",id:"note-local-vs-global"}),e.jsx(s,{title:"Extrapolation Is Not Free",content:"While these methods enable longer contexts, quality still degrades compared to training natively at the target length. A model trained on 4K and extrapolated to 32K will underperform a model trained natively on 32K, especially for tasks requiring precise attention over the full context (like needle-in-a-haystack retrieval). Long-context fine-tuning remains valuable.",id:"warning-quality-gap"}),e.jsx(n,{type:"note",title:"The Context Length Arms Race",content:"GPT-3 (2020): 2K tokens. GPT-3.5: 4K/16K. GPT-4: 8K/128K. Claude: 100K/200K. Gemini 1.5: 1M/2M tokens. This rapid expansion was enabled by FlashAttention (reducing memory), RoPE scaling (extending positions), and training innovations (progressive length training, long-context data curation).",id:"note-arms-race"}),e.jsx(n,{type:"tip",title:"Practical Recommendations",content:"For extending an existing RoPE model: (1) try Dynamic NTK scaling first (no fine-tuning needed), (2) if quality is insufficient, use YaRN with 200-1000 steps of fine-tuning on long-context data, (3) for maximum quality, progressively train on increasing lengths. Always evaluate with needle-in-a-haystack tests across the full context window.",id:"note-practical"})]})}const X=Object.freeze(Object.defineProperty({__proto__:null,default:N},Symbol.toStringTag,{value:"Module"}));export{E as a,F as b,A as c,K as d,B as e,$ as f,R as g,z as h,Q as i,V as j,O as k,C as l,I as m,W as n,D as o,G as p,U as q,H as r,S as s,X as t};
