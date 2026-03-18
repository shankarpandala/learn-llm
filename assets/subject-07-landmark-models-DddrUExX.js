import{j as e}from"./vendor-DWbzdFaj.js";import{r}from"./vendor-katex-BYl39Yo6.js";import{D as a,N as t,E as n,P as i,W as o,T as s}from"./subject-01-text-fundamentals-DG6tAvii.js";function l(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"The GPT Evolution: GPT-1, GPT-2, and GPT-3"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"OpenAI's Generative Pre-trained Transformer series established the paradigm of large-scale autoregressive language modeling. Each generation scaled parameters, data, and compute by orders of magnitude, revealing emergent capabilities that smaller models lacked."}),e.jsx(a,{title:"Autoregressive Language Model",definition:"A model that generates text by predicting one token at a time, conditioning on all previous tokens. The probability of a sequence is $P(x_1, \\ldots, x_n) = \\prod_{i=1}^{n} P(x_i \\mid x_1, \\ldots, x_{i-1})$.",id:"def-autoregressive"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"GPT-1 (June 2018)"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"GPT-1 introduced the idea of unsupervised pre-training followed by supervised fine-tuning. With 117M parameters, 12 transformer layers, and a 768-dimensional hidden state, it was trained on the BooksCorpus (~5GB of text). It used a learned positional embedding and the standard decoder-only transformer with masked self-attention."}),e.jsx(t,{type:"historical",title:"The Pre-training Revolution",content:"Before GPT-1, NLP relied on task-specific architectures. Radford et al. (2018) showed that a single pre-trained model could be fine-tuned to achieve state-of-the-art on 9 of 12 NLP benchmarks, fundamentally shifting how NLP research was done.",id:"note-gpt1-impact"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"GPT-2 (February 2019)"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"GPT-2 scaled to 1.5B parameters (48 layers, 1600-dimensional hidden state) trained on WebText (40GB scraped from Reddit outbound links with 3+ karma). The key insight was that language models could perform downstream tasks as zero-shot learners without any fine-tuning, simply by framing tasks as text completion."}),e.jsx(n,{title:"GPT-2 Model Sizes",problem:"Compare the four released GPT-2 model variants.",steps:[{formula:"GPT\\text{-}2~\\text{Small}: 117\\text{M params}, L{=}12, d{=}768, h{=}12",explanation:"Matches GPT-1 in size but trained on significantly more data."},{formula:"GPT\\text{-}2~\\text{Medium}: 345\\text{M params}, L{=}24, d{=}1024, h{=}16",explanation:"Doubled depth and increased width over Small."},{formula:"GPT\\text{-}2~\\text{Large}: 762\\text{M params}, L{=}36, d{=}1280, h{=}20",explanation:"Further scaling of both depth and width."},{formula:"GPT\\text{-}2~\\text{XL}: 1.5\\text{B params}, L{=}48, d{=}1600, h{=}25",explanation:"The full model, 10x larger than GPT-1."}],id:"example-gpt2-sizes"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"GPT-3 (June 2020)"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"GPT-3 massively scaled to 175B parameters (96 layers, 12288-dimensional hidden state, 96 attention heads). Trained on a filtered Common Crawl plus curated datasets totaling ~570GB of text. It introduced few-shot in-context learning: providing examples directly in the prompt without any gradient updates."}),e.jsx(a,{title:"In-Context Learning",definition:"The ability of a language model to learn a task from a few demonstration examples provided in the prompt at inference time, without any parameter updates. Formally, $P(y \\mid x, \\{(x_i, y_i)\\}_{i=1}^{k})$ where the examples are simply concatenated as text.",id:"def-icl"}),e.jsx(i,{title:"loading_gpt2_huggingface.py",code:`from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 (124M version)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Inspect architecture
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Layers: {model.config.n_layer}")
print(f"Hidden size: {model.config.n_embd}")
print(f"Attention heads: {model.config.n_head}")
print(f"Vocab size: {model.config.vocab_size}")
# Parameters: 124,439,808
# Layers: 12, Hidden: 768, Heads: 12, Vocab: 50257

# Generate text
input_ids = tokenizer.encode("The future of AI is", return_tensors="pt")
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
    )
print(tokenizer.decode(output[0], skip_special_tokens=True))`,id:"code-gpt2-load"}),e.jsx(t,{type:"intuition",title:"Scaling Laws",content:"GPT-3 demonstrated that model performance follows power-law scaling with model size, dataset size, and compute budget. Kaplan et al. (2020) formalized this: loss scales as L ~ C^(-0.05) where C is compute in PetaFLOP-days. This means predictable gains from scaling.",id:"note-scaling-laws"}),e.jsx(o,{title:"Compute Requirements",content:"GPT-3 required ~3640 PetaFLOP-days to train, estimated at $4.6M on cloud GPUs in 2020 prices. The 175B parameters demand at least 350GB in float16, making it impractical to run locally. API access became the primary mode of interaction, establishing the LLM-as-a-service paradigm.",id:"warning-compute"})]})}const S=Object.freeze(Object.defineProperty({__proto__:null,default:l},Symbol.toStringTag,{value:"Module"}));function d(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"InstructGPT and RLHF"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"InstructGPT (Ouyang et al., 2022) bridged the gap between raw language model capability and human-aligned behavior. By combining supervised fine-tuning with reinforcement learning from human feedback (RLHF), OpenAI created a 1.3B parameter model that was preferred over the 175B GPT-3 by human evaluators."}),e.jsx(a,{title:"Reinforcement Learning from Human Feedback (RLHF)",definition:"A training paradigm that aligns language models with human preferences using three stages: (1) supervised fine-tuning on demonstrations, (2) training a reward model on human comparisons, and (3) optimizing the policy with PPO against the reward model while constraining divergence from the SFT model via a KL penalty: $\\max_\\pi \\mathbb{E}_{x \\sim D}[\\mathbb{E}_{y \\sim \\pi(\\cdot|x)}[R(x,y)] - \\beta \\, \\text{KL}(\\pi \\| \\pi_{\\text{SFT}})]$.",id:"def-rlhf"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Stage 1: Supervised Fine-Tuning (SFT)"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Human labelers wrote high-quality responses to a set of prompts. A pre-trained GPT-3 model was fine-tuned on these demonstrations using standard cross-entropy loss. This created the SFT model, which already showed improved instruction following over the base model."}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Stage 2: Reward Model Training"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Labelers ranked multiple model outputs for the same prompt. These rankings trained a reward model (RM) to predict human preferences. The RM takes a prompt-response pair and outputs a scalar reward score."}),e.jsx(s,{title:"Bradley-Terry Preference Model",statement:"Given two responses $y_w$ (preferred) and $y_l$ (dispreferred) to prompt $x$, the reward model is trained by maximizing the log-likelihood of the observed preference under the Bradley-Terry model.",proof:e.jsx(r.BlockMath,{math:"\\mathcal{L}_{\\text{RM}} = -\\mathbb{E}_{(x, y_w, y_l) \\sim D}\\left[\\log \\sigma\\left(R_\\theta(x, y_w) - R_\\theta(x, y_l)\\right)\\right]"}),id:"theorem-bradley-terry"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Stage 3: PPO Optimization"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The SFT model was further optimized using Proximal Policy Optimization (PPO) to maximize the reward model's scores while staying close to the SFT distribution. The KL divergence penalty prevents reward hacking, where the model exploits loopholes in the reward model."}),e.jsx(n,{title:"RLHF Training Pipeline",problem:"Outline the data requirements for each RLHF stage in InstructGPT.",steps:[{formula:"\\text{SFT}: \\sim 13{,}000 \\text{ demonstrations}",explanation:"Human labelers wrote ideal responses to sampled prompts from the OpenAI API."},{formula:"\\text{RM}: \\sim 33{,}000 \\text{ comparisons}",explanation:"Labelers ranked 4-9 outputs per prompt, yielding pairwise comparison data."},{formula:"\\text{PPO}: \\sim 31{,}000 \\text{ prompts}",explanation:"Additional prompts without labels, optimized against the frozen reward model."}],id:"example-rlhf-data"}),e.jsx(i,{title:"reward_model_concept.py",code:`import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """Simplified reward model for RLHF."""
    def __init__(self, base_model_name="gpt2"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        # Use the last token's hidden state as the sequence representation
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)

# Bradley-Terry preference loss
def preference_loss(reward_chosen, reward_rejected):
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()

# Example usage
rm = RewardModel("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

chosen = tokenizer("Explain gravity: Gravity is the force...", return_tensors="pt", padding=True)
rejected = tokenizer("Explain gravity: I don't know lol", return_tensors="pt", padding=True)

r_chosen = rm(chosen.input_ids, chosen.attention_mask)
r_rejected = rm(rejected.input_ids, rejected.attention_mask)

loss = preference_loss(r_chosen, r_rejected)
print(f"Reward chosen: {r_chosen.item():.4f}")
print(f"Reward rejected: {r_rejected.item():.4f}")
print(f"Preference loss: {loss.item():.4f}")`,id:"code-reward-model"}),e.jsx(t,{type:"note",title:"InstructGPT vs ChatGPT",content:"ChatGPT (November 2022) used the same RLHF methodology as InstructGPT but was built on top of a more capable base model (GPT-3.5). The conversational format and dialogue-specific training data made it the fastest-growing consumer application in history, reaching 100 million users in two months.",id:"note-chatgpt"}),e.jsx(o,{title:"Reward Hacking",content:"Without the KL penalty, the policy can find adversarial responses that achieve high reward model scores without actually being helpful. For example, the model might learn to produce verbose, confident-sounding but incorrect answers that the reward model rates highly. Careful calibration of the KL coefficient beta is essential.",id:"warning-reward-hacking"})]})}const G=Object.freeze(Object.defineProperty({__proto__:null,default:d},Symbol.toStringTag,{value:"Module"}));function m(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"GPT-4: Capabilities and Multimodal Reasoning"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"GPT-4 (March 2023) represented a qualitative leap in capability. While OpenAI did not disclose architectural details, GPT-4 demonstrated human-level performance on professional exams, robust multimodal understanding (text and images), and significantly improved reasoning, factuality, and steerability compared to GPT-3.5."}),e.jsx(a,{title:"Multimodal Language Model",definition:"A model that accepts and reasons over multiple input modalities (e.g., text and images). GPT-4V processes visual inputs through a vision encoder whose representations are projected into the language model's embedding space, enabling unified text-image reasoning.",id:"def-multimodal"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Benchmark Performance"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"GPT-4 scored in the 90th percentile on the bar exam (vs 10th for GPT-3.5), 99th percentile on the GRE Verbal, and achieved 86.4% on MMLU (5-shot), up from 70% for GPT-3.5. These gains came from better pre-training data, RLHF refinement, and likely architectural improvements including mixture-of-experts."}),e.jsx(n,{title:"GPT-4 Exam Performance",problem:"Compare GPT-3.5 and GPT-4 on key professional and academic benchmarks.",steps:[{formula:"\\text{Bar Exam}: \\text{GPT-3.5} \\approx 10\\text{th pctile} \\to \\text{GPT-4} \\approx 90\\text{th pctile}",explanation:"A massive jump, demonstrating strong legal reasoning capabilities."},{formula:"\\text{MMLU (5-shot)}: 70.0\\% \\to 86.4\\%",explanation:"Broad improvement across 57 subjects from STEM to humanities."},{formula:"\\text{HumanEval (code)}: 48.1\\% \\to 67.0\\%",explanation:"Significant gains in code generation and understanding."},{formula:"\\text{AP Calculus BC}: 43\\% \\to 76\\%",explanation:"Major improvement in complex mathematical reasoning."}],id:"example-gpt4-benchmarks"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Rumored Architecture"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"While not officially confirmed, credible reports suggest GPT-4 uses a Mixture-of-Experts (MoE) architecture with 8 experts of roughly 220B parameters each, for a total of ~1.76T parameters but only ~280B active per forward pass. This allows much greater total capacity while keeping inference cost manageable."}),e.jsx(t,{type:"intuition",title:"Why MoE for GPT-4?",content:"A dense 1.76T parameter model would be prohibitively expensive to run. MoE lets you store vast knowledge across experts but only activate a small subset per token. Think of it like a company with many specialists: for any given question, only the relevant experts are consulted, keeping response time fast.",id:"note-moe-intuition"}),e.jsx(i,{title:"gpt4_api_usage.py",code:`from openai import OpenAI

client = OpenAI()  # uses OPENAI_API_KEY env var

# Text completion with GPT-4
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the key architectural differences "
         "between GPT-3 and GPT-4 in 3 bullet points."},
    ],
    temperature=0.7,
    max_tokens=300,
)
print(response.choices[0].message.content)

# GPT-4 Vision (multimodal)
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Example: analyzing an architecture diagram
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this neural network architecture."},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{encode_image('arch.png')}"
            }},
        ]},
    ],
)
print(response.choices[0].message.content)

# Token usage tracking
usage = response.usage
print(f"Prompt tokens: {usage.prompt_tokens}")
print(f"Completion tokens: {usage.completion_tokens}")
print(f"Total tokens: {usage.total_tokens}")`,id:"code-gpt4-api"}),e.jsx(t,{type:"note",title:"Predictable Scaling",content:"The GPT-4 technical report revealed that OpenAI could predict GPT-4's final loss and benchmark performance from much smaller training runs. They trained models at 1/10,000th and 1/1,000th the compute and extrapolated, demonstrating that scaling laws enable reliable planning of training runs worth tens of millions of dollars.",id:"note-predictable-scaling"}),e.jsx(o,{title:"Closed-Source Limitations",content:"GPT-4's architecture, training data, and parameter count are not officially disclosed. This limits reproducibility and independent analysis. The research community has increasingly pushed for open models (LLaMA, Mistral) partly as a response to this opacity.",id:"warning-closed-source"})]})}const N=Object.freeze(Object.defineProperty({__proto__:null,default:m},Symbol.toStringTag,{value:"Module"}));function c(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Architectural Innovations Across GPT Versions"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Each GPT generation incorporated improvements to the base transformer decoder architecture. These innovations in normalization, positional encoding, attention patterns, and numerical precision became standard across the field."}),e.jsx(a,{title:"Pre-Layer Normalization",definition:"A reordering of the transformer block where layer normalization is applied before the attention and FFN sublayers rather than after (post-norm). The output becomes $x + \\text{Attn}(\\text{LN}(x))$ instead of $\\text{LN}(x + \\text{Attn}(x))$. This improves training stability and enables deeper networks without careful learning rate warmup.",id:"def-pre-ln"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Normalization Strategies"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"GPT-1 used post-layer normalization (original transformer style). GPT-2 switched to pre-layer normalization which became the standard for all subsequent large models. Modern models like LLaMA further adopted RMSNorm, which drops the mean-centering of LayerNorm for faster computation with comparable performance."}),e.jsx(n,{title:"Normalization Comparison",problem:"Compare LayerNorm vs RMSNorm computation for a hidden state vector h.",steps:[{formula:"\\text{LayerNorm}(h) = \\gamma \\cdot \\frac{h - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta",explanation:"Standard LayerNorm: centers by mean, scales by variance, applies learned affine transform."},{formula:"\\text{RMSNorm}(h) = \\gamma \\cdot \\frac{h}{\\sqrt{\\frac{1}{d}\\sum_{i=1}^{d} h_i^2 + \\epsilon}}",explanation:"RMSNorm: only normalizes by root mean square. No mean centering, no bias term. ~10-15% faster."}],id:"example-norm-comparison"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Positional Encoding Evolution"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"GPT-1/2/3 used learned absolute positional embeddings, which limited generalization beyond the training sequence length. Modern models adopted Rotary Position Embeddings (RoPE), which encode relative positions through rotation matrices applied to query and key vectors."}),e.jsx(a,{title:"Rotary Position Embeddings (RoPE)",definition:"A positional encoding that applies a rotation matrix $R_\\theta^{(m)}$ to query and key vectors at position $m$. The attention score between positions $m$ and $n$ depends only on the relative distance: $q_m^T k_n = (R_\\theta^{(m)} q)^T (R_\\theta^{(n)} k) = q^T R_\\theta^{(n-m)} k$. This naturally encodes relative position without explicit relative attention computation.",id:"def-rope"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Activation Functions"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"GPT-1/2 used GELU activation in the feed-forward layers. Modern architectures switched to SwiGLU (Shazeer, 2020), a gated variant that combines Swish activation with a gating mechanism, improving performance at similar parameter counts."}),e.jsx(i,{title:"architectural_components.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in LLaMA, Mistral)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class SwiGLU(nn.Module):
    """SwiGLU activation for feed-forward network (Shazeer 2020)."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # gate projection

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# Compare parameter counts
dim, hidden = 4096, 11008  # LLaMA-7B dimensions
swiglu = SwiGLU(dim, hidden)
standard_ffn = nn.Sequential(nn.Linear(dim, hidden * 2 // 3 * 4), nn.GELU(), nn.Linear(hidden * 2 // 3 * 4, dim))

print(f"SwiGLU params: {sum(p.numel() for p in swiglu.parameters()):,}")
print(f"Standard FFN params: {sum(p.numel() for p in standard_ffn.parameters()):,}")

# Test RMSNorm vs LayerNorm speed
import time
x = torch.randn(32, 2048, 4096)
rmsnorm = RMSNorm(4096)
layernorm = nn.LayerNorm(4096)

for name, norm in [("LayerNorm", layernorm), ("RMSNorm", rmsnorm)]:
    start = time.perf_counter()
    for _ in range(100):
        _ = norm(x)
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed:.3f}s for 100 iterations")`,id:"code-arch-components"}),e.jsx(t,{type:"tip",title:"Architecture Cheat Sheet",content:"GPT-1/2/3: Post/Pre-LN, learned positional, GELU, dense FFN. GPT-4 (rumored): Pre-LN, possibly RoPE, SwiGLU, MoE. LLaMA/Mistral: Pre-RMSNorm, RoPE, SwiGLU, GQA. The trend is clear: RMSNorm + RoPE + SwiGLU + GQA has become the default modern recipe.",id:"note-arch-cheatsheet"}),e.jsx(o,{title:"Hidden Dimension Scaling with SwiGLU",content:"SwiGLU uses three weight matrices instead of two, so for a fair parameter comparison the hidden dimension is typically set to 2/3 of what a standard FFN would use. If you see a model config with hidden_dim = 11008 for dim = 4096, this is the 2/3 adjustment (4 * 4096 * 2/3 rounded up).",id:"warning-swiglu-dims"})]})}const C=Object.freeze(Object.defineProperty({__proto__:null,default:c},Symbol.toStringTag,{value:"Module"}));function p(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"LLaMA 1 and 2: Meta's Open Approach"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Meta's LLaMA (Large Language Model Meta AI) series catalyzed the open-source LLM movement. LLaMA 1 (February 2023) showed that smaller, well-trained models could match or exceed much larger ones. LLaMA 2 (July 2023) added commercial licensing and RLHF-tuned chat variants."}),e.jsx(a,{title:"Chinchilla-Optimal Training",definition:"The Chinchilla scaling law (Hoffmann et al., 2022) states that for a given compute budget $C$, the optimal model size $N$ and dataset size $D$ scale equally: $N \\propto C^{0.5}$ and $D \\propto C^{0.5}$. Concretely, the optimal token count is approximately $20 \\times N$, meaning a 7B model should be trained on ~140B tokens.",id:"def-chinchilla"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"LLaMA 1 Architecture"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"LLaMA 1 combined proven techniques: pre-RMSNorm, RoPE positional embeddings, SwiGLU activations, and no bias terms. Crucially, it trained for much longer than Chinchilla prescribed -- the 7B model saw 1T tokens (7x the Chinchilla-optimal 140B), which improved inference-time performance at the cost of more training compute."}),e.jsx(n,{title:"LLaMA 1 Model Family",problem:"Compare the four LLaMA 1 model sizes and their training configurations.",steps:[{formula:"\\text{LLaMA-7B}: L{=}32, d{=}4096, h{=}32, \\text{FFN}{=}11008",explanation:"Trained on 1T tokens. Outperformed GPT-3 (175B) on most benchmarks."},{formula:"\\text{LLaMA-13B}: L{=}40, d{=}5120, h{=}40, \\text{FFN}{=}13824",explanation:"Trained on 1T tokens. Competitive with Chinchilla (70B) and PaLM (540B) on many tasks."},{formula:"\\text{LLaMA-33B}: L{=}60, d{=}6656, h{=}52, \\text{FFN}{=}17920",explanation:"Trained on 1.4T tokens. Strong performance across reasoning benchmarks."},{formula:"\\text{LLaMA-65B}: L{=}80, d{=}8192, h{=}64, \\text{FFN}{=}22016",explanation:"Trained on 1.4T tokens. Matched or exceeded PaLM-540B despite being 8x smaller."}],id:"example-llama1-sizes"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"LLaMA 2 Improvements"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"LLaMA 2 trained on 2T tokens (2x LLaMA 1), extended context from 2048 to 4096 tokens, and introduced Grouped-Query Attention (GQA) in the 34B and 70B variants. The Chat versions used extensive RLHF with over 1 million human annotations."}),e.jsx(a,{title:"Grouped-Query Attention (GQA)",definition:"A compromise between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). In GQA, $h$ query heads share $g$ key-value heads where $1 < g < h$. Each group of $h/g$ query heads shares one KV head. This reduces KV cache memory by a factor of $h/g$ while maintaining most of MHA's quality. LLaMA 2 70B uses $h{=}64$ query heads with $g{=}8$ KV heads.",id:"def-gqa"}),e.jsx(i,{title:"load_llama_model.py",code:`from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load LLaMA 2 7B (requires access approval on HuggingFace)
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Inspect architecture
config = model.config
print(f"Hidden size: {config.hidden_size}")          # 4096
print(f"Num layers: {config.num_hidden_layers}")     # 32
print(f"Num attention heads: {config.num_attention_heads}")      # 32
print(f"Num KV heads: {config.num_key_value_heads}")             # 32 (MHA for 7B)
print(f"Intermediate size: {config.intermediate_size}")          # 11008
print(f"Vocab size: {config.vocab_size}")             # 32000
print(f"Max position: {config.max_position_embeddings}")  # 4096
print(f"RoPE theta: {config.rope_theta}")             # 10000.0

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")  # ~6.7B

# Generate text
inputs = tokenizer("The key innovation of LLaMA is", return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100, temperature=0.7, top_p=0.9)
print(tokenizer.decode(output[0], skip_special_tokens=True))`,id:"code-llama-load"}),e.jsx(t,{type:"historical",title:"The LLaMA Leak",content:"LLaMA 1 weights were initially released under a research-only license, but within a week they were leaked publicly. This accidental open-sourcing sparked an explosion of community fine-tunes (Alpaca, Vicuna, WizardLM) and fundamentally shifted the LLM landscape toward openness. Meta embraced this with LLaMA 2's commercial license.",id:"note-llama-leak"}),e.jsx(o,{title:"Tokenizer Limitations",content:"LLaMA 1/2 use a SentencePiece BPE tokenizer with only 32,000 tokens. This is much smaller than GPT-4's ~100K vocabulary, which means LLaMA tokenizes non-English text and code less efficiently, requiring more tokens for the same content. LLaMA 3 addressed this with a 128K vocabulary.",id:"warning-tokenizer"})]})}const R=Object.freeze(Object.defineProperty({__proto__:null,default:p},Symbol.toStringTag,{value:"Module"}));function h(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Mistral 7B: Sliding Window Attention"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Mistral 7B (September 2023) from the Paris-based startup Mistral AI outperformed LLaMA 2 13B on all benchmarks and matched LLaMA 1 34B on many tasks, despite having only 7.3B parameters. It introduced sliding window attention (SWA) and a rolling KV cache to handle long sequences efficiently while maintaining strong performance."}),e.jsx(a,{title:"Sliding Window Attention (SWA)",definition:"An attention pattern where each token attends only to the previous $W$ tokens (the window size) rather than all preceding tokens. With $W{=}4096$ and $L$ layers, information can propagate across $W \\times L$ tokens through the network. For Mistral 7B with $W{=}4096$ and $L{=}32$, this gives a theoretical receptive field of $4096 \\times 32 = 131{,}072$ tokens.",id:"def-swa"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Architecture Details"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Mistral 7B uses 32 layers, a hidden dimension of 4096, 32 query heads with 8 KV heads (GQA), and an intermediate size of 14336. It combines SWA with GQA and a rolling KV cache that only stores the last W positions, reducing memory from O(n) to O(W) for the cache."}),e.jsx(n,{title:"KV Cache Memory Savings",problem:"Compare KV cache memory for a 16K token sequence between full attention and SWA with W=4096.",steps:[{formula:"\\text{Full KV cache} = 2 \\times L \\times n \\times h_{kv} \\times d_h \\times 2\\text{B}",explanation:"For Mistral 7B: 2 * 32 * 16384 * 8 * 128 * 2 = 2.1 GB in float16."},{formula:"\\text{SWA KV cache} = 2 \\times L \\times W \\times h_{kv} \\times d_h \\times 2\\text{B}",explanation:"With W=4096: 2 * 32 * 4096 * 8 * 128 * 2 = 0.5 GB in float16."},{formula:"\\text{Savings} = 1 - \\frac{W}{n} = 1 - \\frac{4096}{16384} = 75\\%",explanation:"The rolling cache gives 4x memory reduction for this sequence length."}],id:"example-kv-savings"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Rolling Buffer Cache"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Instead of growing the KV cache linearly with sequence length, Mistral uses a fixed-size circular buffer of size W. Position i is stored at index (i mod W). When the buffer is full, the oldest entries are overwritten. This ensures constant memory usage regardless of sequence length."}),e.jsx(i,{title:"mistral_sliding_window.py",code:`from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Mistral 7B
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Inspect sliding window config
config = model.config
print(f"Hidden size: {config.hidden_size}")             # 4096
print(f"Num layers: {config.num_hidden_layers}")        # 32
print(f"Num attention heads: {config.num_attention_heads}")  # 32
print(f"Num KV heads: {config.num_key_value_heads}")    # 8 (GQA)
print(f"Sliding window: {config.sliding_window}")       # 4096
print(f"Intermediate size: {config.intermediate_size}") # 14336
print(f"Vocab size: {config.vocab_size}")               # 32000

# Conceptual rolling buffer implementation
class RollingKVCache:
    def __init__(self, window_size, num_heads, head_dim, dtype=torch.float16):
        self.window_size = window_size
        self.keys = torch.zeros(window_size, num_heads, head_dim, dtype=dtype)
        self.values = torch.zeros(window_size, num_heads, head_dim, dtype=dtype)
        self.position = 0

    def update(self, key, value):
        idx = self.position % self.window_size
        self.keys[idx] = key
        self.values[idx] = value
        self.position += 1

    def get_cache(self):
        if self.position < self.window_size:
            return self.keys[:self.position], self.values[:self.position]
        return self.keys, self.values

# The cache never exceeds window_size entries
cache = RollingKVCache(window_size=4096, num_heads=8, head_dim=128)
print(f"Cache memory: {cache.keys.numel() * 2 * 2 / 1024**2:.1f} MB (fixed)")`,id:"code-mistral-swa"}),e.jsx(t,{type:"note",title:"Mistral vs LLaMA 2 Performance",content:"Mistral 7B outperforms LLaMA 2 13B on MMLU (60.1 vs 54.8), HellaSwag (81.3 vs 80.7), and all reasoning benchmarks despite having nearly half the parameters. This efficiency comes from better training data curation, GQA, and the SWA mechanism that enables longer effective context.",id:"note-mistral-perf"}),e.jsx(o,{title:"SWA Information Loss",content:"Sliding window attention means tokens beyond the window cannot be directly attended to. While information propagates through layers (giving a larger effective context), this is lossy -- the model cannot perfectly recall details from early in a very long document. For tasks requiring precise long-range retrieval, full attention or retrieval augmentation may be needed.",id:"warning-swa-loss"})]})}const F=Object.freeze(Object.defineProperty({__proto__:null,default:h},Symbol.toStringTag,{value:"Module"}));function u(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Falcon and MPT: Alternative Open Architectures"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Falcon (TII, Abu Dhabi) and MPT (MosaicML) were among the first fully open-source models to compete with LLaMA. They introduced important ideas around data quality, multi-query attention at scale, and ALiBi positional encoding, influencing subsequent model designs."}),e.jsx(a,{title:"Multi-Query Attention (MQA)",definition:"An attention variant where all query heads share a single key and value head. Given $h$ attention heads, MQA reduces KV cache memory by a factor of $h$ compared to standard multi-head attention: $\\text{MQA}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V$ where $K, V \\in \\mathbb{R}^{n \\times d_k}$ are shared across all heads.",id:"def-mqa"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Falcon Architecture"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Falcon came in 7B, 40B, and 180B sizes. The 40B and 180B models used multi-query attention to reduce inference memory. A key innovation was the RefinedWeb dataset: 5 trillion tokens of web data filtered through aggressive deduplication and quality filtering. Falcon demonstrated that web-only data, properly filtered, could match curated datasets."}),e.jsx(n,{title:"Falcon Model Configurations",problem:"Compare the Falcon model family architectures.",steps:[{formula:"\\text{Falcon-7B}: L{=}32, d{=}4544, h{=}71, \\text{MHA}",explanation:"Unusual dimensions (not power of 2). Uses standard multi-head attention. Trained on 1.5T tokens from RefinedWeb."},{formula:"\\text{Falcon-40B}: L{=}60, d{=}8192, h{=}128, \\text{MQA}",explanation:"Uses multi-query attention with 1 KV head. Trained on 1T tokens. Topped the Hugging Face Open LLM Leaderboard at launch."},{formula:"\\text{Falcon-180B}: L{=}80, d{=}14848, h{=}232, \\text{GQA}",explanation:"Uses GQA with 8 KV groups. Trained on 3.5T tokens. Approached GPT-4 on some benchmarks."}],id:"example-falcon-configs"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"MPT Architecture"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"MosaicML's MPT (MosaicML Pretrained Transformer) models used ALiBi (Attention with Linear Biases) instead of positional embeddings. ALiBi adds a linear bias to attention scores based on the distance between query and key positions, enabling length extrapolation without any learned positional parameters."}),e.jsx(a,{title:"ALiBi (Attention with Linear Biases)",definition:"A positional encoding method that adds a head-specific linear bias to attention scores: $\\text{softmax}(q_i^T k_j - m \\cdot |i - j|)$ where $m$ is a head-specific slope. Slopes are set geometrically: $m_i = 2^{-8i/h}$ for $h$ heads. This requires no learned parameters and naturally extrapolates to longer sequences than seen during training.",id:"def-alibi"}),e.jsx(i,{title:"falcon_and_mpt_usage.py",code:`from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load Falcon-7B
falcon_name = "tiiuae/falcon-7b"
falcon_tokenizer = AutoTokenizer.from_pretrained(falcon_name)
falcon_model = AutoModelForCausalLM.from_pretrained(
    falcon_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

config = falcon_model.config
print("=== Falcon-7B ===")
print(f"Hidden size: {config.hidden_size}")       # 4544
print(f"Num layers: {config.num_hidden_layers}")  # 32
print(f"Num heads: {config.num_attention_heads}") # 71
print(f"Vocab size: {config.vocab_size}")         # 65024

# Load MPT-7B
mpt_name = "mosaicml/mpt-7b"
mpt_tokenizer = AutoTokenizer.from_pretrained(mpt_name)
mpt_model = AutoModelForCausalLM.from_pretrained(
    mpt_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

mpt_config = mpt_model.config
print("\\n=== MPT-7B ===")
print(f"Hidden size: {mpt_config.d_model}")       # 4096
print(f"Num layers: {mpt_config.n_layers}")       # 32
print(f"Num heads: {mpt_config.n_heads}")         # 32

# ALiBi slopes computation
import math
def get_alibi_slopes(num_heads):
    closest_power = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power) - 3)))
    slopes = [base ** (i + 1) for i in range(closest_power)]
    return slopes

slopes = get_alibi_slopes(32)
print(f"\\nALiBi slopes (first 8): {[f'{s:.6f}' for s in slopes[:8]]}")`,id:"code-falcon-mpt"}),e.jsx(t,{type:"intuition",title:"RefinedWeb: Data Quality Over Quantity",content:"Falcon's RefinedWeb pipeline applies URL filtering, language identification, fuzzy deduplication (MinHash), and quality heuristics to Common Crawl. The key finding: web data cleaned to near-curated quality produces models as good as those trained on hand-picked datasets. This democratized training data, as anyone can apply similar filtering to publicly available crawls.",id:"note-refinedweb"}),e.jsx(o,{title:"Trust Remote Code",content:"Both Falcon and MPT require trust_remote_code=True when loading, meaning they execute custom Python code from the model repository. Always review the model card and source code before using this flag, as it could potentially execute arbitrary code on your machine.",id:"warning-trust-remote"})]})}const E=Object.freeze(Object.defineProperty({__proto__:null,default:u},Symbol.toStringTag,{value:"Module"}));function f(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Phi Models: Small Models, High-Quality Data"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Microsoft's Phi series challenged the assumption that larger models are always better. Phi-1 (1.3B), Phi-2 (2.7B), and Phi-3 (3.8B) demonstrated that carefully curated, high-quality training data can produce small models that punch far above their weight class, often matching models 10-25x their size."}),e.jsx(a,{title:"Data Quality Scaling",definition:"The principle that model performance depends not just on dataset size but critically on data quality. For small models, the quality-to-quantity ratio becomes paramount: $\\text{Performance} \\propto f(\\text{quality}) \\cdot g(N, D)$ where the quality function $f$ dominates at small model sizes $N$.",id:"def-data-quality"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Phi-1: Textbooks Are All You Need"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:'Phi-1 (June 2023) was a 1.3B parameter model trained specifically for code generation. It achieved 50.6% on HumanEval, outperforming models 10x larger. The secret was "textbook quality" data: a combination of filtered code from The Stack, GPT-3.5-generated textbook explanations, and GPT-3.5-generated exercises totaling only 7B tokens.'}),e.jsx(n,{title:"Phi Series Evolution",problem:"Trace the Phi family's progression in size, data strategy, and benchmark performance.",steps:[{formula:"\\text{Phi-1 (1.3B)}: \\text{HumanEval} = 50.6\\%",explanation:'Code-only model. Trained on 7B tokens of "textbook quality" code data. Matched StarCoder 15B.'},{formula:"\\text{Phi-1.5 (1.3B)}: \\text{Common sense reasoning competitive with 5x larger}",explanation:"Extended to natural language. Used 30B tokens of synthetic textbook + web data."},{formula:"\\text{Phi-2 (2.7B)}: \\text{MMLU} = 56.7\\%",explanation:"Matched Mistral 7B and LLaMA 2 70B on some benchmarks. Trained on 1.4T tokens of curated web + synthetic data."},{formula:"\\text{Phi-3-mini (3.8B)}: \\text{MMLU} = 69.0\\%",explanation:"Matched LLaMA 3 8B. Used heavily filtered web data + synthetic data. 3.3T tokens."}],id:"example-phi-evolution"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Synthetic Data Pipeline"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:'The Phi models pioneered the use of LLM-generated synthetic data for training. GPT-3.5/4 was prompted to generate textbook-style explanations, exercises with solutions, and step-by-step reasoning chains. This "data distillation" transfers knowledge from a larger model into training data for a smaller one.'}),e.jsx(i,{title:"phi_model_usage.py",code:`from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load Phi-3-mini (3.8B parameters)
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Check model size
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e9:.2f}B")
print(f"Memory footprint: {total_params * 2 / 1e9:.2f} GB (float16)")

# Phi-3 uses ChatML format
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function for binary search."},
]

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = pipe(messages, max_new_tokens=200, temperature=0.1, do_sample=True)
print(output[0]["generated_text"][-1]["content"])

# Compare model sizes for perspective
models = {
    "Phi-3-mini": 3.8e9,
    "Mistral-7B": 7.3e9,
    "LLaMA-2-13B": 13e9,
    "LLaMA-2-70B": 70e9,
}
print("\\nModel size comparison:")
for name, params in models.items():
    ratio = params / 3.8e9
    print(f"  {name}: {params/1e9:.1f}B ({ratio:.1f}x Phi-3-mini)")`,id:"code-phi-usage"}),e.jsx(t,{type:"intuition",title:"Why Synthetic Data Works",content:"Web text contains noise, irrelevant content, and poor explanations. Synthetic textbook data provides clear, structured explanations with consistent quality. Think of it as the difference between learning from random blog posts versus a well-written textbook -- the same amount of reading yields very different learning outcomes.",id:"note-synthetic-intuition"}),e.jsx(o,{title:"Benchmark Contamination Concerns",content:"A recurring criticism of Phi models is potential benchmark contamination: if GPT-4-generated training data inadvertently includes content similar to benchmark test sets, performance may be inflated. Microsoft has addressed this with contamination analysis, but the concern highlights the difficulty of evaluating models trained on synthetic data derived from models that may have seen the benchmarks.",id:"warning-contamination"}),e.jsx(t,{type:"tip",title:"Running Phi on Consumer Hardware",content:"Phi-3-mini at 3.8B parameters requires only ~7.6GB in float16 or ~2GB when quantized to 4-bit. This makes it runnable on most modern laptops and even some phones. Use the GGUF format with llama.cpp or MLX for best performance on consumer devices.",id:"note-phi-hardware"})]})}const q=Object.freeze(Object.defineProperty({__proto__:null,default:f},Symbol.toStringTag,{value:"Module"}));function g(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"LLaMA 3: Scaling Open Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"LLaMA 3 (April 2024) marked a significant leap for open-weight models. The 8B and 70B models matched or exceeded GPT-3.5 on most benchmarks, while the 405B variant approached GPT-4 and Claude 3.5 Sonnet. Key changes included a 128K vocabulary, 8K default context (128K extended), and training on 15T+ tokens."}),e.jsx(a,{title:"Over-Training",definition:"Training a model on significantly more tokens than Chinchilla-optimal. LLaMA 3 8B was trained on 15T tokens, approximately $100\\times$ the Chinchilla-optimal amount of ~150B tokens. This sacrifices training compute efficiency for improved inference efficiency: a smaller, over-trained model is cheaper to deploy than a larger, compute-optimally trained one.",id:"def-overtraining"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Architecture Improvements"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"LLaMA 3 maintains the same core recipe as LLaMA 2 (RMSNorm, RoPE, SwiGLU, GQA) but with important refinements: a 4x larger vocabulary (128,256 vs 32,000), GQA across all model sizes (not just 70B+), and training on 15T tokens of multilingual data."}),e.jsx(n,{title:"LLaMA 3 Model Family",problem:"Compare LLaMA 3 model configurations against LLaMA 2 equivalents.",steps:[{formula:"\\text{LLaMA 3 8B}: L{=}32, d{=}4096, h_q{=}32, h_{kv}{=}8",explanation:"GQA with 8 KV heads (vs MHA in LLaMA 2 7B). Vocab 128K. Trained on 15T tokens."},{formula:"\\text{LLaMA 3 70B}: L{=}80, d{=}8192, h_q{=}64, h_{kv}{=}8",explanation:"Same GQA ratio as LLaMA 2 70B. 15T training tokens (7.5x more than LLaMA 2)."},{formula:"\\text{LLaMA 3.1 405B}: L{=}126, d{=}16384, h_q{=}128, h_{kv}{=}8",explanation:"Largest open model at release. 128K context with RoPE scaling. Approaches GPT-4."},{formula:"\\text{Vocab}: 32{,}000 \\to 128{,}256",explanation:"4x larger vocabulary dramatically improves multilingual and code tokenization efficiency."}],id:"example-llama3-configs"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Training Data and Process"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"LLaMA 3 was trained on over 15 trillion tokens from a curated mix of web data, with improved filtering using Llama 2 itself as a quality classifier. The data mix was ~5% code, with significantly more multilingual content than LLaMA 2. Post-training involved both SFT and DPO (Direct Preference Optimization) rather than PPO-based RLHF."}),e.jsx(i,{title:"llama3_usage.py",code:`from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load LLaMA 3 8B Instruct
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

config = model.config
print(f"Vocab size: {config.vocab_size}")                 # 128256
print(f"Hidden size: {config.hidden_size}")               # 4096
print(f"Layers: {config.num_hidden_layers}")              # 32
print(f"Query heads: {config.num_attention_heads}")       # 32
print(f"KV heads: {config.num_key_value_heads}")          # 8
print(f"Intermediate: {config.intermediate_size}")        # 14336
print(f"RoPE theta: {config.rope_theta}")                 # 500000.0

# Compare tokenizer efficiency
llama2_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
llama3_tok = tokenizer

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "Bonjour le monde, comment allez-vous aujourd'hui?",
]
print("\\nTokenizer efficiency comparison:")
for text in texts:
    l2 = len(llama2_tok.encode(text))
    l3 = len(llama3_tok.encode(text))
    saving = (1 - l3 / l2) * 100
    print(f"  '{text[:40]}...' L2={l2} L3={l3} ({saving:.0f}% fewer tokens)")

# Chat generation with LLaMA 3 format
messages = [
    {"role": "system", "content": "You are a concise technical assistant."},
    {"role": "user", "content": "What are the key improvements in LLaMA 3?"},
]
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = pipe(messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])`,id:"code-llama3"}),e.jsx(t,{type:"note",title:"LLaMA 3.1 and Extended Context",content:"LLaMA 3.1 extended context to 128K tokens using a progressive RoPE frequency scaling approach. The base RoPE theta was increased from 500,000 to 500,000 with additional NTK-aware scaling. Training on long documents was done in stages: first 8K, then 32K, then 128K context lengths.",id:"note-llama31-context"}),e.jsx(o,{title:"Memory Requirements",content:"LLaMA 3 405B requires ~810GB in bfloat16, needing at least 8x A100 80GB GPUs. Even the 70B model needs ~140GB in bfloat16. For practical deployment, quantization (GPTQ, AWQ, or GGUF 4-bit) reduces the 70B model to ~35GB, fitting on a single A100 or two consumer GPUs.",id:"warning-llama3-memory"})]})}const $=Object.freeze(Object.defineProperty({__proto__:null,default:g},Symbol.toStringTag,{value:"Module"}));function x(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Google Gemma: Lightweight Open Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Gemma (February 2024) is Google DeepMind's family of open-weight models built on the same research and technology as the Gemini models. Available in 2B and 7B sizes, Gemma demonstrated that Google's internal training infrastructure and data pipeline could produce highly competitive small models for the open-source community."}),e.jsx(a,{title:"Gemma Architecture",definition:"Gemma uses a decoder-only transformer with Multi-Query Attention (2B) or Multi-Head Attention (7B), RoPE positional embeddings, GeGLU activation (a GELU-gated variant of GLU), and RMSNorm. A distinctive feature is embedding tying: the input and output embedding matrices are shared, and the embedding dimension is normalized by $\\sqrt{d}$ before the final projection.",id:"def-gemma-arch"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Architecture Details"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Gemma follows the modern transformer recipe but with a few Google-specific choices. It uses GeGLU (GELU-gated linear unit) instead of SwiGLU, and applies a learnable scaling factor to embeddings. Both models were trained on 6T tokens of primarily English web data, code, and mathematics."}),e.jsx(n,{title:"Gemma Model Configurations",problem:"Compare Gemma 1 and Gemma 2 model variants.",steps:[{formula:"\\text{Gemma 2B}: L{=}18, d{=}2048, h{=}8, d_{ff}{=}16384",explanation:"Uses MQA (1 KV head). 2T tokens. Surprisingly capable for its size."},{formula:"\\text{Gemma 7B}: L{=}28, d{=}3072, h{=}16, d_{ff}{=}24576",explanation:"Uses MHA (16 KV heads). 6T tokens. Outperforms LLaMA 2 7B and Mistral 7B."},{formula:"\\text{Gemma 2 9B}: L{=}42, d{=}3584, h{=}16, h_{kv}{=}8",explanation:"Uses GQA. Alternates local (4096) and global attention layers. Knowledge distillation from larger model."},{formula:"\\text{Gemma 2 27B}: L{=}46, d{=}4608, h{=}32, h_{kv}{=}16",explanation:"GQA with soft-capping on attention logits. Matches LLaMA 3 70B on several benchmarks."}],id:"example-gemma-configs"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Gemma 2 Innovations"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Gemma 2 introduced alternating local and global attention layers (reducing the quadratic cost of full attention), logit soft-capping to prevent attention weights from becoming too sharp, and knowledge distillation from a larger teacher model during pre-training."}),e.jsx(a,{title:"Logit Soft-Capping",definition:"A technique that prevents attention logits from growing unboundedly by applying $\\text{logits} = \\text{cap} \\cdot \\tanh\\left(\\frac{\\text{logits}}{\\text{cap}}\\right)$ where the cap value (e.g., 50.0) limits the maximum attention logit magnitude. This stabilizes training and prevents entropy collapse in attention distributions.",id:"def-soft-capping"}),e.jsx(i,{title:"gemma_usage.py",code:`from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load Gemma 2 9B
model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Inspect architecture
config = model.config
print(f"Model type: {config.model_type}")
print(f"Hidden size: {config.hidden_size}")           # 3584
print(f"Num layers: {config.num_hidden_layers}")      # 42
print(f"Query heads: {config.num_attention_heads}")   # 16
print(f"KV heads: {config.num_key_value_heads}")      # 8
print(f"Intermediate: {config.intermediate_size}")    # 14336
print(f"Vocab size: {config.vocab_size}")             # 256000

# Gemma uses a very large vocabulary (256K)
texts = ["Hello world", "Bonjour le monde", "import torch"]
for text in texts:
    tokens = tokenizer.encode(text)
    print(f"'{text}' -> {len(tokens)} tokens: {tokens}")

# Count parameters by component
embedding_params = model.model.embed_tokens.weight.numel()
total_params = sum(p.numel() for p in model.parameters())
print(f"\\nEmbedding params: {embedding_params / 1e9:.2f}B "
      f"({embedding_params / total_params * 100:.1f}% of total)")
print(f"Total params: {total_params / 1e9:.2f}B")

# Generate with Gemma chat format
messages = [{"role": "user", "content": "Explain attention soft-capping in 2 sentences."}]
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = pipe(messages, max_new_tokens=100)
print(output[0]["generated_text"][-1]["content"])`,id:"code-gemma"}),e.jsx(t,{type:"note",title:"Gemma's 256K Vocabulary",content:"Gemma uses a 256,000-token SentencePiece vocabulary, the largest among open models. This dramatically improves tokenization efficiency for non-English languages and code. However, the large embedding matrix (256K * d) means embedding parameters account for a larger fraction of the total model size, especially for the 2B variant.",id:"note-gemma-vocab"}),e.jsx(o,{title:"Attention Compatibility",content:"Gemma 2's alternating local/global attention and logit soft-capping require custom attention implementations. Standard FlashAttention does not support soft-capping natively, which initially limited inference optimization. Updated versions of frameworks like vLLM and TGI added support, but check compatibility before deploying.",id:"warning-gemma-compat"})]})}const O=Object.freeze(Object.defineProperty({__proto__:null,default:x},Symbol.toStringTag,{value:"Module"}));function y(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Qwen: Alibaba's Multilingual Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Qwen (Tongyi Qianwen) from Alibaba Cloud represents China's strongest contribution to open-weight language models. The Qwen 2 and Qwen 2.5 series span from 0.5B to 72B parameters, with particularly strong performance on multilingual tasks, mathematics, and code generation."}),e.jsx(a,{title:"Qwen Architecture",definition:"Qwen 2 uses a standard decoder-only transformer with GQA, SwiGLU activations, RMSNorm, RoPE positional embeddings, and a vocabulary of 151,646 tokens. It supports a native context length of 32,768 tokens extendable to 131,072 with YaRN (Yet another RoPE extensioN) scaling.",id:"def-qwen-arch"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Qwen 2.5 Model Family"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Qwen 2.5 (September 2024) expanded the family to include specialized variants for code (Qwen2.5-Coder) and mathematics (Qwen2.5-Math). The base models range from 0.5B to 72B, all trained on 18 trillion tokens, making them among the most overtrained open models."}),e.jsx(n,{title:"Qwen 2.5 Model Sizes",problem:"Compare the Qwen 2.5 model family configurations.",steps:[{formula:"\\text{Qwen2.5-0.5B}: L{=}24, d{=}896, h_q{=}14, h_{kv}{=}2",explanation:"Tiny model suitable for edge devices. GQA with 7:1 ratio."},{formula:"\\text{Qwen2.5-7B}: L{=}28, d{=}3584, h_q{=}28, h_{kv}{=}4",explanation:"Competitive with LLaMA 3 8B and Mistral 7B. 7:1 GQA ratio."},{formula:"\\text{Qwen2.5-32B}: L{=}64, d{=}5120, h_q{=}40, h_{kv}{=}8",explanation:"Strong mid-range model. 5:1 GQA ratio."},{formula:"\\text{Qwen2.5-72B}: L{=}80, d{=}8192, h_q{=}64, h_{kv}{=}8",explanation:"Flagship model. Competitive with LLaMA 3.1 70B. 8:1 GQA ratio."}],id:"example-qwen-sizes"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Training Data and Multilingual Focus"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Qwen models are trained on 18T tokens covering 29+ languages with strong emphasis on Chinese and English. The training mix includes web text, books, code, and curated multilingual data. Qwen's tokenizer uses a byte-level BPE with 151,646 tokens, designed for efficient encoding of CJK characters."}),e.jsx(i,{title:"qwen_usage.py",code:`from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load Qwen2.5-7B-Instruct
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Inspect architecture
config = model.config
print(f"Hidden size: {config.hidden_size}")             # 3584
print(f"Layers: {config.num_hidden_layers}")            # 28
print(f"Query heads: {config.num_attention_heads}")     # 28
print(f"KV heads: {config.num_key_value_heads}")        # 4
print(f"Intermediate: {config.intermediate_size}")      # 18944
print(f"Vocab size: {config.vocab_size}")               # 152064
print(f"RoPE theta: {config.rope_theta}")               # 1000000.0

# Multilingual tokenizer efficiency
texts = {
    "English": "The quick brown fox jumps over the lazy dog.",
    "Chinese": "Transformer 是一种强大的神经网络架构。",
    "Code": "def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
}
for lang, text in texts.items():
    tokens = tokenizer.encode(text)
    print(f"{lang}: {len(tokens)} tokens for {len(text)} chars (ratio: {len(tokens)/len(text):.2f})")

# Generate with Qwen chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain GQA in 3 bullet points."},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
print(tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))`,id:"code-qwen"}),e.jsx(t,{type:"note",title:"Qwen2.5-Coder",content:"The Qwen2.5-Coder series is specifically trained on 5.5T tokens of code data and achieves state-of-the-art performance among open models on code benchmarks. The 32B-Instruct variant matches GPT-4o on HumanEval and MBPP, making it one of the strongest open code models available.",id:"note-qwen-coder"}),e.jsx(o,{title:"License Considerations",content:"Qwen models use a custom Apache-2.0-compatible license, but some variants have usage restrictions for models above certain sizes or for specific commercial applications. Always check the specific model card on HuggingFace for the exact license terms before deploying in production.",id:"warning-qwen-license"})]})}const K=Object.freeze(Object.defineProperty({__proto__:null,default:y},Symbol.toStringTag,{value:"Module"}));function _(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"DeepSeek V2 and V3: Efficient Scale"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"DeepSeek, a Chinese AI lab backed by the quantitative trading firm High-Flyer, produced some of the most innovative open models. DeepSeek-V2 introduced Multi-head Latent Attention (MLA) and DeepSeekMoE, while DeepSeek-V3 scaled to 671B total parameters with only 37B active, matching frontier closed models at a fraction of the training cost."}),e.jsx(a,{title:"Multi-head Latent Attention (MLA)",definition:"An attention mechanism that compresses the KV cache by projecting keys and values into a low-rank latent space before storage. Instead of caching full-dimensional KV pairs, MLA stores a compressed latent vector $c_t = W_{DKV} [k_t; v_t]$ of dimension $d_c \\ll 2 \\times n_h \\times d_h$. At attention time, keys and values are reconstructed from the latent. This reduces KV cache by 93.3% compared to standard MHA.",id:"def-mla"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"DeepSeek-V2 Architecture"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"DeepSeek-V2 (236B total, 21B active) combined MLA with a fine-grained MoE architecture using 160 routed experts and 2 shared experts per layer, with top-6 routing. This achieved GPT-4-level performance while requiring only 42.5% of the training compute of DeepSeek-V1 67B."}),e.jsx(n,{title:"DeepSeek Architecture Comparison",problem:"Compare DeepSeek V2 and V3 architectures.",steps:[{formula:"\\text{V2}: 236\\text{B total}, 21\\text{B active}, L{=}60",explanation:"MLA + DeepSeekMoE (160 routed + 2 shared experts per layer, top-6). Context: 128K."},{formula:"\\text{V3}: 671\\text{B total}, 37\\text{B active}, L{=}61",explanation:"MLA + DeepSeekMoE (256 routed + 1 shared expert, top-8). Auxiliary-loss-free load balancing."},{formula:"\\text{V3 KV cache}: d_c = 512 \\text{ vs MHA } d_{kv} = 7680",explanation:"MLA compresses KV cache from 7680 dimensions to 512, a 93.3% reduction per layer."},{formula:"\\text{V3 training cost}: \\sim\\$5.5\\text{M} (2.788\\text{M H800 GPU-hours})",explanation:"Remarkably low cost for a frontier model. 14.8T training tokens with FP8 mixed precision."}],id:"example-deepseek-arch"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"FP8 Mixed-Precision Training"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"DeepSeek-V3 pioneered FP8 training for large MoE models, using 8-bit floating point for most matrix multiplications while maintaining FP32 master weights. This nearly doubled training throughput compared to BF16 on H800 GPUs, contributing significantly to the low training cost."}),e.jsx(i,{title:"deepseek_usage.py",code:`from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load DeepSeek-V2-Lite (a smaller variant for experimentation)
model_name = "deepseek-ai/DeepSeek-V2-Lite"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Inspect MoE configuration
config = model.config
print(f"Hidden size: {config.hidden_size}")
print(f"Num layers: {config.num_hidden_layers}")
print(f"Num experts: {getattr(config, 'n_routed_experts', 'N/A')}")
print(f"Top-K experts: {getattr(config, 'num_experts_per_tok', 'N/A')}")

# Count total vs active parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e9:.1f}B")

# MLA KV cache comparison
d_model = 5120   # DeepSeek-V2 hidden dim
n_heads = 128    # number of attention heads
d_head = 128     # head dimension
d_c = 512        # MLA compressed dimension

mha_kv_size = 2 * n_heads * d_head  # standard MHA KV per token
mla_kv_size = d_c                    # MLA compressed KV per token

print(f"\\nKV cache per token per layer:")
print(f"  Standard MHA: {mha_kv_size} dimensions ({mha_kv_size * 2} bytes in FP16)")
print(f"  MLA: {mla_kv_size} dimensions ({mla_kv_size * 2} bytes in FP16)")
print(f"  Compression ratio: {mla_kv_size / mha_kv_size:.1%}")

# For a 128K context, 60-layer model
seq_len = 131072
for name, kv_dim in [("MHA", mha_kv_size), ("MLA", mla_kv_size)]:
    cache_gb = 2 * 60 * seq_len * kv_dim * 2 / 1e9
    print(f"  {name} total KV cache (128K ctx): {cache_gb:.1f} GB")`,id:"code-deepseek"}),e.jsx(t,{type:"intuition",title:"Why MLA Is Transformative",content:"Traditional KV cache grows as O(n * L * h * d_h) and is the primary bottleneck for long-context inference. MLA compresses this to O(n * L * d_c) where d_c is much smaller than h * d_h. Think of it as storing a compressed summary of each token's key-value information rather than the full representation, with the ability to reconstruct the full KV when needed.",id:"note-mla-intuition"}),e.jsx(o,{title:"Custom CUDA Kernels Required",content:"DeepSeek-V2/V3's MLA and fine-grained MoE require custom CUDA kernels for efficient inference. Standard HuggingFace transformers inference may be significantly slower than the optimized implementation. For production deployment, use the official DeepSeek inference framework or vLLM with DeepSeek support.",id:"warning-deepseek-kernels"})]})}const I=Object.freeze(Object.defineProperty({__proto__:null,default:_},Symbol.toStringTag,{value:"Module"}));function b(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Cohere Command R: RAG-Optimized Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Cohere's Command R (March 2024) and Command R+ (April 2024) were purpose-built for retrieval-augmented generation (RAG) and enterprise tool use. With 35B and 104B parameters respectively, they introduced built-in grounded generation with inline citations, native tool calling, and strong multilingual performance across 10 languages."}),e.jsx(a,{title:"Grounded Generation",definition:"A generation paradigm where the model produces responses that are explicitly grounded in provided documents. The model generates inline citations (e.g., [doc1], [doc2]) pointing to specific source passages, enabling verifiable outputs. Formally, the model learns $P(y, c \\mid x, D)$ where $c$ is the citation set and $D$ is the document collection.",id:"def-grounded-gen"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Architecture and Training"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Command R uses a standard decoder-only transformer with GQA, RoPE, SwiGLU, and a 128K context window. What makes it distinctive is not the base architecture but the post-training: extensive fine-tuning on RAG tasks, tool use, structured output generation, and multi-turn conversations with document context."}),e.jsx(n,{title:"Command R Model Family",problem:"Compare Command R and Command R+ specifications.",steps:[{formula:"\\text{Command R (35B)}: L{=}40, d{=}8192, h_q{=}64, h_{kv}{=}8",explanation:"Optimized for RAG workloads. 128K context. 10 supported languages. Apache 2.0 licensed."},{formula:"\\text{Command R+ (104B)}: L{=}64, d{=}12288, h_q{=}96, h_{kv}{=}8",explanation:"Larger variant for complex enterprise tasks. Approaches GPT-4 on RAG benchmarks."},{formula:"\\text{Context}: 128\\text{K tokens with RAG-specific training}",explanation:"Trained specifically to handle long documents with accurate citation and attribution."}],id:"example-command-r-specs"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Built-in Tool Use"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Command R models have native support for tool/function calling. They can generate structured JSON tool invocations, process tool results, and synthesize multi-step tool-assisted answers. This is trained directly into the model rather than relying on prompt engineering."}),e.jsx(i,{title:"command_r_rag_example.py",code:`from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Command R (35B) - requires significant GPU memory
model_name = "CohereForAI/c4ai-command-r-v01"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# For demonstration, inspect the chat template with RAG
documents = [
    {"title": "LLM Scaling", "text": "Scaling laws show that model performance improves "
     "predictably with increased parameters, data, and compute."},
    {"title": "RAG Overview", "text": "Retrieval-Augmented Generation combines a retriever "
     "with a generator to produce grounded, factual responses."},
]

# Command R uses a special RAG-aware chat format
conversation = [
    {"role": "user", "content": "What is RAG and how does it relate to scaling?"},
]

# Apply Command R's grounded generation template
grounded_prompt = tokenizer.apply_grounded_generation_template(
    conversation,
    documents=documents,
    citation_mode="accurate",  # generates inline citations
    tokenize=False,
    add_generation_prompt=True,
)
print("=== Grounded Generation Prompt ===")
print(grounded_prompt[:500])

# Tool use example with Command R
tools = [
    {"name": "web_search", "description": "Search the web for information",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "The search query"}
     }, "required": ["query"]}},
    {"name": "calculator", "description": "Perform mathematical calculations",
     "parameters": {"type": "object", "properties": {
         "expression": {"type": "string", "description": "Math expression"}
     }, "required": ["expression"]}},
]

tool_prompt = tokenizer.apply_tool_use_template(
    [{"role": "user", "content": "What is 42 * 17 and who invented the transformer?"}],
    tools=tools,
    tokenize=False,
    add_generation_prompt=True,
)
print("\\n=== Tool Use Prompt ===")
print(tool_prompt[:500])`,id:"code-command-r"}),e.jsx(t,{type:"note",title:"Enterprise RAG Focus",content:"Command R's RAG training includes learning to correctly attribute claims to source documents, refuse to answer when documents don't contain relevant information, and distinguish between information from documents versus parametric knowledge. This makes it particularly suited for enterprise deployments where factual grounding and auditability are critical.",id:"note-enterprise-rag"}),e.jsx(t,{type:"tip",title:"Using Command R via API",content:"While the open-weight models are available on HuggingFace, Cohere's API provides the most optimized inference with built-in RAG pipeline support (including document ingestion, retrieval, and grounded generation) through the /chat endpoint with the 'documents' parameter.",id:"note-api-tip"}),e.jsx(o,{title:"Citation Accuracy",content:"While Command R is trained for grounded generation, citation accuracy is not perfect. The model may occasionally hallucinate citations or attribute information to the wrong document. Always implement verification logic that checks whether cited passages actually support the generated claims in production RAG systems.",id:"warning-citation-accuracy"})]})}const D=Object.freeze(Object.defineProperty({__proto__:null,default:b},Symbol.toStringTag,{value:"Module"}));function k(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Mixture of Experts: Fundamentals"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:'Mixture of Experts (MoE) is a technique that decouples the total number of model parameters from the computation required per input. By routing each token to only a subset of "expert" sub-networks, MoE models can have trillions of parameters while keeping per-token compute similar to a much smaller dense model.'}),e.jsx(a,{title:"Mixture of Experts Layer",definition:"An MoE layer replaces the standard feed-forward network (FFN) with $E$ parallel expert networks and a gating (router) network. For input $x$, the router produces weights $g(x) \\in \\mathbb{R}^E$ and the output is a weighted sum of expert outputs: $\\text{MoE}(x) = \\sum_{i=1}^{E} g_i(x) \\cdot \\text{FFN}_i(x)$. With top-$K$ routing, only $K \\ll E$ experts are activated per token.",id:"def-moe"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The Router Network"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The router (or gating network) is typically a simple linear layer followed by a softmax that produces a probability distribution over experts. Only the top-K experts with the highest probabilities are activated, and the output is the weighted sum of their outputs."}),e.jsx(s,{title:"Top-K Sparse Gating",statement:"For a token representation $x \\in \\mathbb{R}^d$ and router weights $W_g \\in \\mathbb{R}^{E \\times d}$, the sparse gating function selects the top-K experts and renormalizes their weights.",proof:e.jsx(r.BlockMath,{math:"g(x) = \\text{Softmax}(\\text{TopK}(W_g \\cdot x, K))"}),corollaries:["Only K out of E experts compute their output, giving a speedup of E/K.","The router itself adds minimal overhead: just a single matrix multiply plus top-K selection."],id:"theorem-topk-gating"}),e.jsx(n,{title:"MoE Compute Savings",problem:"Compare FLOPs for a dense model vs an MoE model with E=8 experts and K=2.",steps:[{formula:"\\text{Dense FFN}: 2 \\times d \\times d_{ff} \\times 2 = 4 \\cdot d \\cdot d_{ff}",explanation:"Two linear layers (up-project and down-project), each costing 2*d*d_ff FLOPs."},{formula:"\\text{MoE FFN}: K \\times 4 \\cdot d \\cdot d_{ff} = 2 \\times 4 \\cdot d \\cdot d_{ff}",explanation:"Only K=2 experts compute. Same per-expert cost but only 2 out of 8 are active."},{formula:"\\text{Speedup} = \\frac{E}{K} = \\frac{8}{2} = 4\\times",explanation:"MoE uses 4x fewer FLOPs than an equivalently-sized dense model (ignoring router cost)."},{formula:"\\text{Total params} = E \\times \\text{expert\\_params} = 8 \\times \\text{FFN}",explanation:"But the model stores 8x more parameters in memory, enabling more capacity."}],id:"example-moe-flops"}),e.jsx(i,{title:"moe_layer_implementation.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """A single expert: standard FFN with SwiGLU."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoELayer(nn.Module):
    """Mixture of Experts with top-K routing."""
    def __init__(self, dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(dim, hidden_dim) for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # (B*S, D)

        # Router: compute expert scores
        router_logits = self.gate(x_flat)  # (B*S, E)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (B*S, K)

        # Compute expert outputs (naive loop - real implementations use scatter/gather)
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  # (B*S,)
            weight = top_k_weights[:, i].unsqueeze(-1)  # (B*S, 1)
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += weight[mask] * expert_output

        return output.view(batch_size, seq_len, dim)

# Test
moe = MoELayer(dim=512, hidden_dim=1024, num_experts=8, top_k=2)
x = torch.randn(2, 16, 512)
out = moe(x)
print(f"Input shape: {x.shape}, Output shape: {out.shape}")
print(f"Total params: {sum(p.numel() for p in moe.parameters()):,}")
print(f"Active params per token: ~{sum(p.numel() for p in moe.experts[0].parameters()) * 2 + 512 * 8:,}")`,id:"code-moe-layer"}),e.jsx(t,{type:"intuition",title:"Why MoE Works",content:"Different tokens need different types of processing. A code token, a math token, and a natural language token may benefit from different learned representations. MoE lets the model specialize: some experts may focus on syntax, others on semantics, others on reasoning patterns. The router learns to dispatch each token to the most relevant experts.",id:"note-moe-intuition"}),e.jsx(o,{title:"Memory vs Compute Tradeoff",content:"MoE reduces FLOPs but not memory. All expert parameters must be stored in GPU memory even though only K are used per token. A model with 8 experts of 7B each requires 56B parameters in memory but only computes 14B FLOPs (top-2). This makes MoE models memory-bound rather than compute-bound.",id:"warning-moe-memory"})]})}const Q=Object.freeze(Object.defineProperty({__proto__:null,default:k},Symbol.toStringTag,{value:"Module"}));function w(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Switch Transformer: Simplified MoE"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The Switch Transformer (Fedus et al., 2022) simplified MoE by routing each token to exactly one expert (top-1 routing), rather than combining outputs from multiple experts. This simplified routing, reduced communication costs, and scaled to 1.6 trillion parameters while maintaining training stability through careful auxiliary loss design."}),e.jsx(a,{title:"Switch Routing",definition:"A top-1 routing mechanism where each token is sent to exactly one expert: $\\text{expert}(x) = \\arg\\max_i (W_g \\cdot x)_i$. The output is scaled by the router probability: $y = g_i(x) \\cdot \\text{FFN}_i(x)$ where $g_i(x) = \\text{softmax}(W_g \\cdot x)_i$ for the selected expert $i$. This is the simplest possible sparse routing strategy.",id:"def-switch-routing"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Key Design Decisions"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The Switch Transformer made several crucial simplifications over prior MoE work: top-1 routing (vs top-2 in GShard), expert capacity factor to handle imbalanced routing, and selective precision training where the router operates in float32 while experts use bfloat16."}),e.jsx(s,{title:"Load Balancing Loss",statement:"To prevent expert collapse (all tokens routed to a few experts), the Switch Transformer adds an auxiliary loss that encourages uniform routing across experts.",proof:e.jsx(r.BlockMath,{math:"\\mathcal{L}_{\\text{balance}} = \\alpha \\cdot E \\cdot \\sum_{i=1}^{E} f_i \\cdot p_i"}),corollaries:["f_i is the fraction of tokens routed to expert i (discrete, non-differentiable).","p_i is the average router probability assigned to expert i (differentiable).","The product f_i * p_i is minimized when both are uniform at 1/E, encouraging balanced routing.","Alpha is typically set to 0.01 to avoid overwhelming the primary language modeling loss."],id:"theorem-balance-loss"}),e.jsx(n,{title:"Expert Capacity",problem:"Calculate the expert capacity for a batch of 32 tokens with 8 experts and capacity factor 1.25.",steps:[{formula:"\\text{tokens\\_per\\_expert} = \\frac{T}{E} = \\frac{32}{8} = 4",explanation:"With uniform routing, each expert would get exactly 4 tokens."},{formula:"\\text{capacity} = \\lceil CF \\times \\frac{T}{E} \\rceil = \\lceil 1.25 \\times 4 \\rceil = 5",explanation:"The capacity factor (CF=1.25) adds a 25% buffer for routing imbalance."},{formula:"\\text{overflow tokens are dropped}",explanation:"If more than 5 tokens route to one expert, extras pass through unchanged (skip the FFN)."}],id:"example-capacity"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Scaling Results"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The Switch Transformer scaled from 32 to 2048 experts, reaching 1.6T parameters. Despite having 1000x more parameters than T5-Base, the Switch-Base model used the same compute per token and achieved significant speedups: 7x faster pre-training to reach the same loss as T5-Base."}),e.jsx(i,{title:"switch_transformer_concept.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class SwitchLayer(nn.Module):
    """Switch Transformer MoE layer with top-1 routing and load balancing."""
    def __init__(self, dim, hidden_dim, num_experts=8, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        T = x_flat.shape[0]

        # Router (use float32 for stability)
        router_logits = self.gate(x_flat.float())
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-1 routing
        expert_indices = router_probs.argmax(dim=-1)  # (T,)
        expert_weights = router_probs.gather(1, expert_indices.unsqueeze(-1)).squeeze(-1)

        # Compute load balancing loss
        f = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            f[i] = (expert_indices == i).float().mean()
        p = router_probs.mean(dim=0)
        balance_loss = self.num_experts * (f * p).sum()

        # Expert capacity
        capacity = int(self.capacity_factor * T / self.num_experts)

        # Dispatch to experts
        output = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask = (expert_indices == i)
            if mask.sum() == 0:
                continue
            # Apply capacity limit
            token_indices = mask.nonzero(as_tuple=True)[0][:capacity]
            expert_input = x_flat[token_indices]
            expert_output = self.experts[i](expert_input)
            output[token_indices] = expert_weights[token_indices].unsqueeze(-1) * expert_output

        return output.view(B, S, D), balance_loss

# Test
switch = SwitchLayer(dim=512, hidden_dim=2048, num_experts=8)
x = torch.randn(2, 32, 512)
out, loss = switch(x)
print(f"Output shape: {out.shape}")
print(f"Balance loss: {loss.item():.4f}")

# Check routing distribution
with torch.no_grad():
    logits = switch.gate(x.view(-1, 512))
    indices = logits.argmax(dim=-1)
    for i in range(8):
        count = (indices == i).sum().item()
        print(f"  Expert {i}: {count} tokens ({count/64*100:.0f}%)")`,id:"code-switch"}),e.jsx(t,{type:"historical",title:"From GShard to Switch",content:"GShard (Lepikhin et al., 2021) used top-2 routing and was the first MoE to scale to 600B parameters for machine translation. The Switch Transformer showed that top-1 routing was not only simpler but also more efficient, as it halved the communication cost in distributed settings while maintaining quality.",id:"note-gshard-history"}),e.jsx(o,{title:"Token Dropping",content:"When experts overflow their capacity, excess tokens skip the expert layer entirely. During training, this acts as implicit regularization. During inference with capacity_factor=1.0, up to ~10% of tokens can be dropped in poorly balanced models, degrading output quality. Always monitor the fraction of dropped tokens.",id:"warning-token-dropping"})]})}const V=Object.freeze(Object.defineProperty({__proto__:null,default:w},Symbol.toStringTag,{value:"Module"}));function v(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Mixtral: Sparse MoE in Practice"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Mixtral 8x7B (December 2023) from Mistral AI was the first widely-deployed open-source MoE language model. With 46.7B total parameters but only 12.9B active per token, it matched or exceeded LLaMA 2 70B on most benchmarks while being 6x faster at inference. It demonstrated that MoE was practical for real-world deployment, not just research."}),e.jsx(a,{title:"Mixtral Architecture",definition:"Mixtral replaces every feed-forward layer in a Mistral 7B-style transformer with a Sparse MoE layer containing 8 experts and top-2 routing. The attention layers remain shared (not replicated). Each expert is a standard SwiGLU FFN identical to Mistral 7B's FFN. Total params: 8 experts * 5.6B FFN params + 7.1B shared = 46.7B. Active per token: 2 experts * 5.6B + 7.1B = 12.9B.",id:"def-mixtral"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Architecture Breakdown"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Mixtral uses 32 transformer layers, each with a shared GQA attention block (32 query heads, 8 KV heads) and an MoE feed-forward block. The hidden dimension is 4096, each expert has an intermediate size of 14336 (SwiGLU), and top-2 routing is used with a softmax gate. Sliding window attention from Mistral 7B is retained."}),e.jsx(n,{title:"Mixtral Parameter Budget",problem:"Break down where the 46.7B parameters in Mixtral 8x7B are allocated.",steps:[{formula:"\\text{Attention (shared)}: 32 \\times (4 \\times d^2 / \\text{GQA}) \\approx 2.1\\text{B}",explanation:"Q, K, V, O projections with GQA (32 query, 8 KV heads). Shared across all tokens."},{formula:"\\text{Per expert FFN}: 3 \\times d \\times d_{ff} = 3 \\times 4096 \\times 14336 \\approx 176\\text{M}",explanation:"Three weight matrices for SwiGLU (w1, w2, w3). Per layer, per expert."},{formula:"\\text{Total FFN}: 32 \\times 8 \\times 176\\text{M} \\approx 45.1\\text{B}",explanation:"32 layers * 8 experts * 176M params. This dominates the parameter count."},{formula:"\\text{Active FFN}: 32 \\times 2 \\times 176\\text{M} \\approx 11.3\\text{B}",explanation:"Only 2 experts active per token, so effective compute is ~12.9B total."}],id:"example-mixtral-params"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Expert Specialization"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:`Analysis of Mixtral's routing patterns reveals that experts do not specialize by topic (e.g., "science expert" or "code expert"). Instead, specialization is more syntactic: experts tend to handle specific token types, positions in sentences, or linguistic patterns. The routing is not deterministic and varies with context.`}),e.jsx(i,{title:"mixtral_usage.py",code:`from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Mixtral 8x7B (needs ~90GB+ for float16, use quantized for less)
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,  # Quantize to fit in ~24GB
)

# Inspect MoE configuration
config = model.config
print(f"Hidden size: {config.hidden_size}")              # 4096
print(f"Num layers: {config.num_hidden_layers}")         # 32
print(f"Num experts: {config.num_local_experts}")        # 8
print(f"Top-K: {config.num_experts_per_tok}")            # 2
print(f"Intermediate size: {config.intermediate_size}")  # 14336
print(f"Sliding window: {config.sliding_window}")        # 4096

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"\\nTotal parameters: {total / 1e9:.1f}B")

# Estimate active parameters per token
attn_params = sum(
    p.numel() for name, p in model.named_parameters()
    if "self_attn" in name
)
expert_params_each = sum(
    p.numel() for name, p in model.named_parameters()
    if "experts.0" in name
)
active_params = attn_params + expert_params_each * 2  # top-2 routing
print(f"Active params per token: ~{active_params / 1e9:.1f}B")

# Generate
messages = [{"role": "user", "content": "Compare MoE and dense transformers in 3 points."}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
print(tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))`,id:"code-mixtral"}),e.jsx(t,{type:"note",title:"Mixtral 8x22B",content:"Mistral followed up with Mixtral 8x22B (April 2024), scaling to 176B total parameters with 39B active. It uses 8 experts with top-2 routing, a 64K vocabulary, and 64K context length. It significantly outperforms Mixtral 8x7B and competes with LLaMA 3 70B while being faster at inference.",id:"note-mixtral-8x22b"}),e.jsx(o,{title:"Memory Requirements for MoE",content:"Despite using only 12.9B active parameters, Mixtral 8x7B requires loading all 46.7B parameters into memory (~93GB in float16). This means it needs more GPUs than a dense 13B model despite similar inference speed. Quantization (4-bit GPTQ/AWQ) reduces this to ~24GB, making single-GPU inference feasible.",id:"warning-mixtral-memory"}),e.jsx(t,{type:"intuition",title:"MoE as Model Compression",content:"Think of MoE as a form of conditional computation: the model has the knowledge capacity of a 47B model but the inference cost of a 13B model. It's like having a 47B model that was 'compressed' at inference time by only activating the relevant parts. This is fundamentally different from weight pruning or quantization.",id:"note-moe-compression"})]})}const W=Object.freeze(Object.defineProperty({__proto__:null,default:v},Symbol.toStringTag,{value:"Module"}));function T(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Expert Load Balancing"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Load balancing is the central challenge in MoE training. Without proper balancing mechanisms, models suffer from expert collapse (most tokens routed to a few experts) or routing oscillation. Multiple strategies have been developed, from auxiliary losses to auxiliary-loss-free approaches."}),e.jsx(a,{title:"Expert Collapse",definition:"A failure mode in MoE training where the router converges to sending most or all tokens to a small subset of experts, leaving the rest untrained. This creates a positive feedback loop: popular experts improve, increasing their routing probability further. The result is wasted capacity equivalent to a much smaller dense model.",id:"def-expert-collapse"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Auxiliary Loss Approaches"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The most common approach adds a differentiable loss term that penalizes imbalanced routing. The Switch Transformer's balance loss computes the product of token fractions and router probabilities per expert, which is minimized when routing is uniform."}),e.jsx(s,{title:"Z-Loss for Router Stability",statement:"The Z-loss (Zoph et al., 2022, ST-MoE) penalizes large router logits to prevent the router from becoming too confident, which helps maintain exploration of all experts during training.",proof:e.jsx(r.BlockMath,{math:"\\mathcal{L}_z = \\frac{1}{T} \\sum_{t=1}^{T} \\left(\\log \\sum_{i=1}^{E} e^{z_i^{(t)}}\\right)^2"}),corollaries:["z_i^(t) are the raw router logits before softmax for token t.","The Z-loss is zero when all logits are equal (uniform routing) and grows as logits diverge.","Used in conjunction with the balance loss, not as a replacement."],id:"theorem-zloss"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Auxiliary-Loss-Free Balancing"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"DeepSeek-V3 introduced an auxiliary-loss-free approach where each expert maintains a bias term that is adjusted based on the expert's current load. Overloaded experts get their bias decreased (making them less likely to be selected), while underloaded experts get increased bias. This achieves better balancing without the gradient interference of auxiliary losses."}),e.jsx(n,{title:"Balancing Strategies Comparison",problem:"Compare three load balancing approaches used in major MoE models.",steps:[{formula:"\\mathcal{L}_{\\text{Switch}} = \\alpha E \\sum_i f_i \\cdot p_i",explanation:"Switch Transformer: product of token fraction and router probability. Alpha ~0.01. Simple but can interfere with language modeling loss."},{formula:"\\mathcal{L}_{\\text{ST-MoE}} = \\mathcal{L}_{\\text{Switch}} + \\beta \\mathcal{L}_z",explanation:"ST-MoE adds Z-loss for logit regularization. Better stability for very large models."},{formula:"b_i \\leftarrow b_i + \\gamma \\cdot (\\bar{f} - f_i)",explanation:"DeepSeek-V3: bias adjustment. No gradient interference. Expert bias updated proportionally to load deficit."}],id:"example-balancing"}),e.jsx(i,{title:"load_balancing_strategies.py",code:`import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedMoERouter(nn.Module):
    """MoE router with multiple load balancing strategies."""
    def __init__(self, dim, num_experts=8, top_k=2, balance_type="switch"):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_type = balance_type
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # For auxiliary-loss-free balancing (DeepSeek-V3 style)
        if balance_type == "bias":
            self.expert_bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)
            self.bias_update_rate = 0.001

    def forward(self, x):
        # x: (T, D)
        logits = self.gate(x.float())  # (T, E)

        # Apply expert bias if using bias-based balancing
        if self.balance_type == "bias":
            logits = logits + self.expert_bias.unsqueeze(0)

        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Renormalize top-k weights
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute auxiliary loss
        aux_loss = self._compute_balance_loss(logits, probs, top_k_indices)

        # Update bias (for bias-based method, during training)
        if self.balance_type == "bias" and self.training:
            self._update_bias(top_k_indices)

        return top_k_weights, top_k_indices, aux_loss

    def _compute_balance_loss(self, logits, probs, indices):
        T = probs.shape[0]
        E = self.num_experts

        if self.balance_type == "switch":
            # Token fraction per expert
            f = torch.zeros(E, device=probs.device)
            for k in range(self.top_k):
                for i in range(E):
                    f[i] += (indices[:, k] == i).float().sum() / T
            f = f / self.top_k
            p = probs.mean(dim=0)
            return E * (f * p).sum()

        elif self.balance_type == "zloss":
            # Z-loss: penalize large router logits
            z_loss = torch.logsumexp(logits, dim=-1).square().mean()
            # Plus standard balance loss
            f = torch.zeros(E, device=probs.device)
            for k in range(self.top_k):
                for i in range(E):
                    f[i] += (indices[:, k] == i).float().sum() / T
            f = f / self.top_k
            p = probs.mean(dim=0)
            return E * (f * p).sum() + 0.001 * z_loss

        return torch.tensor(0.0, device=probs.device)

    def _update_bias(self, indices):
        with torch.no_grad():
            counts = torch.zeros(self.num_experts, device=indices.device)
            for k in range(self.top_k):
                counts.scatter_add_(0, indices[:, k], torch.ones_like(indices[:, k], dtype=torch.float))
            avg = counts.mean()
            self.expert_bias += self.bias_update_rate * (avg - counts)

# Compare strategies
x = torch.randn(256, 512)
for strategy in ["switch", "zloss", "bias"]:
    router = BalancedMoERouter(512, num_experts=8, top_k=2, balance_type=strategy)
    router.train()
    weights, indices, loss = router(x)
    counts = torch.zeros(8)
    for k in range(2):
        for i in range(8):
            counts[i] += (indices[:, k] == i).sum()
    cv = counts.std() / counts.mean()  # coefficient of variation
    print(f"{strategy:8s} | balance_loss={loss.item():.4f} | CV={cv.item():.3f} | "
          f"counts={counts.int().tolist()}")`,id:"code-balancing"}),e.jsx(t,{type:"intuition",title:"Why Auxiliary Losses Are Problematic",content:"Auxiliary balance losses add gradients to the router that conflict with the language modeling objective. The router must simultaneously route tokens to the best expert AND maintain balance -- these goals often conflict. The DeepSeek-V3 bias approach elegantly sidesteps this by moving balancing out of the gradient computation entirely, letting the router optimize purely for routing quality.",id:"note-aux-loss-problem"}),e.jsx(o,{title:"Balance vs Specialization Tradeoff",content:"Perfect load balance means every expert processes the same number of tokens, but this may not be optimal. Some token types (e.g., code, math) may genuinely benefit from more expert capacity. Over-aggressive balancing can prevent useful specialization. Modern approaches aim for approximately balanced (within 10-20%) rather than perfectly balanced routing.",id:"warning-balance-tradeoff"})]})}const H=Object.freeze(Object.defineProperty({__proto__:null,default:T},Symbol.toStringTag,{value:"Module"}));function M(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Chain-of-Thought: Emergence of Reasoning"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:'Chain-of-Thought (CoT) prompting (Wei et al., 2022) revealed that large language models can perform complex multi-step reasoning when prompted to "think step by step." This emergent capability appears only above a critical model size threshold (~100B parameters) and fundamentally changed how we elicit reasoning from language models.'}),e.jsx(a,{title:"Chain-of-Thought Prompting",definition:"A prompting technique that includes intermediate reasoning steps in few-shot examples, encouraging the model to generate similar step-by-step reasoning before arriving at a final answer. Formally, instead of learning $P(a \\mid q)$ directly, the model generates a reasoning chain: $P(a \\mid q) = \\sum_r P(a \\mid q, r) \\cdot P(r \\mid q)$ where $r$ represents the chain of reasoning steps.",id:"def-cot"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Emergence and Scale"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"CoT reasoning is an emergent ability: it provides no benefit for models below ~10B parameters, marginal benefit at 10-60B, and dramatic improvements above 100B. With CoT, PaLM 540B solved 58% of GSM8K math problems (vs 18% without CoT), and GPT-3.5 improved from 35% to 57% on the same benchmark."}),e.jsx(n,{title:"CoT vs Direct Prompting",problem:"Show how CoT prompting improves math reasoning on a grade-school problem.",steps:[{formula:"\\text{Direct}: Q \\to A",explanation:'Without CoT: "Roger has 5 tennis balls. He buys 2 cans of 3. How many?" -> "11" (correct but fragile).'},{formula:"\\text{CoT}: Q \\to R_1 \\to R_2 \\to \\ldots \\to A",explanation:'With CoT: "Roger starts with 5 balls. He buys 2 cans * 3 balls = 6 balls. Total = 5 + 6 = 11."'},{formula:"\\text{GSM8K accuracy}: 18\\% \\to 58\\% \\text{ (PaLM 540B)}",explanation:"CoT more than triples accuracy on grade-school math, showing it enables genuine multi-step reasoning."}],id:"example-cot-comparison"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Zero-Shot CoT"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:`Kojima et al. (2022) showed that simply appending "Let's think step by step" to a prompt elicits CoT reasoning without any few-shot examples. This "zero-shot CoT" works across diverse reasoning tasks and suggests that the reasoning capability is already present in large models -- it just needs to be activated.`}),e.jsx(i,{title:"cot_prompting.py",code:`from openai import OpenAI

client = OpenAI()

def solve_with_cot(question, model="gpt-4"):
    """Compare direct vs CoT prompting on a math problem."""

    # Direct prompting
    direct_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"{question}\\nAnswer with just the number."},
        ],
        temperature=0,
    )

    # Chain-of-Thought prompting
    cot_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"{question}\\nLet's think step by step."},
        ],
        temperature=0,
    )

    return {
        "direct": direct_response.choices[0].message.content,
        "cot": cot_response.choices[0].message.content,
    }

# Test problems of increasing difficulty
problems = [
    "If a store sells 3 apples for $2 and you buy 12 apples, how much do you pay?",
    "A train travels at 60 mph for 2.5 hours, then 80 mph for 1.5 hours. What's the total distance?",
    "In a room, there are 5 people. Each person shakes hands with every other person exactly once. How many handshakes occur?",
]

for problem in problems:
    result = solve_with_cot(problem)
    print(f"Q: {problem}")
    print(f"Direct: {result['direct'][:80]}")
    print(f"CoT: {result['cot'][:200]}")
    print()

# Self-Consistency: sample multiple CoT paths and take majority vote
def self_consistency(question, model="gpt-4", n_samples=5):
    answers = []
    for _ in range(n_samples):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"{question}\\nThink step by step, then give your final answer as a number."}],
            temperature=0.7,
        )
        answers.append(response.choices[0].message.content)
    return answers

# Self-consistency improves accuracy by ~10-15% over single CoT
results = self_consistency("What is 17 * 23 + 45 - 12?")
print(f"Self-consistency results ({len(results)} samples):")
for i, a in enumerate(results):
    print(f"  Sample {i}: {a[-50:]}")`,id:"code-cot"}),e.jsx(t,{type:"intuition",title:"Why CoT Works",content:"A transformer has fixed depth, limiting the computational steps per token. Without CoT, the model must solve the entire problem in a single forward pass. CoT essentially gives the model 'scratch space': each generated token is an additional compute step. For a problem requiring k reasoning steps, CoT converts it from depth-k (impossible for a fixed-depth network) to length-k (feasible through autoregressive generation).",id:"note-cot-intuition"}),e.jsx(t,{type:"note",title:"Self-Consistency Decoding",content:"Wang et al. (2023) improved CoT with self-consistency: sample multiple reasoning chains at temperature > 0 and take the majority vote on the final answer. This is based on the insight that correct reasoning paths are more likely to converge on the same answer than incorrect ones. Self-consistency improves GSM8K accuracy by 10-15% over single-sample CoT.",id:"note-self-consistency"}),e.jsx(o,{title:"CoT Limitations",content:"CoT does not guarantee correct reasoning. Models can produce plausible-sounding but logically flawed chains, especially for problems requiring novel reasoning patterns not seen in training. CoT also increases output length (and thus cost and latency). For simple tasks, CoT can actually hurt performance by introducing unnecessary complexity.",id:"warning-cot-limitations"})]})}const U=Object.freeze(Object.defineProperty({__proto__:null,default:M},Symbol.toStringTag,{value:"Module"}));function L(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"OpenAI o1 and o3: Trained Reasoning"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:`OpenAI's o1 (September 2024) and o3 (December 2024) models represent a paradigm shift: instead of just prompting for chain-of-thought, these models are explicitly trained to reason through reinforcement learning. They generate internal "thinking" tokens before responding, achieving breakthroughs on mathematics, coding, and science benchmarks that seemed out of reach for standard LLMs.`}),e.jsx(a,{title:"Test-Time Compute Scaling",definition:"The principle that model performance can be improved by allocating more computation at inference time rather than during training. Instead of the traditional paradigm where performance scales with training compute, reasoning models scale with inference compute: $\\text{Performance} \\propto f(\\text{thinking tokens})$. More thinking time yields better answers.",id:"def-test-time-compute"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"How o1 Works"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The o1 model generates a hidden chain-of-thought (CoT) before producing its visible response. This internal reasoning process is trained using large-scale reinforcement learning, where the model learns to break problems into steps, verify intermediate results, backtrack when encountering errors, and try alternative approaches."}),e.jsx(n,{title:"o1/o3 Benchmark Results",problem:"Compare o1 and o3 performance against GPT-4 on key reasoning benchmarks.",steps:[{formula:"\\text{AIME 2024 (math)}: \\text{GPT-4} = 12\\% \\to \\text{o1} = 83\\% \\to \\text{o3} = 96\\%",explanation:"AIME is a competitive math exam. o3 solves nearly all problems, approaching human expert level."},{formula:"\\text{Codeforces (coding)}: \\text{GPT-4} = 11\\text{th pctile} \\to \\text{o1} = 89\\text{th pctile}",explanation:"o1 reaches competitive programmer level on Codeforces rating."},{formula:"\\text{GPQA Diamond (PhD science)}: \\text{GPT-4} = 56\\% \\to \\text{o1} = 78\\%",explanation:"PhD-level science questions where domain experts achieve ~70%. o1 surpasses experts."},{formula:"\\text{ARC-AGI}: \\text{GPT-4} = 5\\% \\to \\text{o3 (high)} = 88\\%",explanation:"Abstract reasoning benchmark designed to test generalization. o3 with high compute achieves 88%."}],id:"example-o1-benchmarks"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Training Methodology"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"While exact details are proprietary, o1/o3 likely use a process reward model (PRM) that evaluates each step of reasoning, combined with RL training (potentially MCTS-like search during training) to learn effective reasoning strategies. The model learns to generate productive thinking patterns: decomposition, verification, backtracking, and analogical reasoning."}),e.jsx(i,{title:"o1_api_usage.py",code:`from openai import OpenAI

client = OpenAI()

# o1 models use a different API pattern
# No system message, no temperature control (always greedy)
response = client.chat.completions.create(
    model="o1",
    messages=[
        {"role": "user", "content": """
Solve this step by step:
Find all integer solutions to the equation:
x^3 + y^3 + z^3 = 42
where |x|, |y|, |z| <= 10^16
"""},
    ],
    # o1 models don't support temperature, top_p, or system messages
    # They internally generate reasoning tokens before responding
)

print(response.choices[0].message.content)
print(f"\\nCompletion tokens: {response.usage.completion_tokens}")
print(f"Reasoning tokens: {response.usage.completion_tokens_details.reasoning_tokens}")
print(f"Visible tokens: {response.usage.completion_tokens - response.usage.completion_tokens_details.reasoning_tokens}")

# Compare reasoning effort levels (o3-mini)
for effort in ["low", "medium", "high"]:
    response = client.chat.completions.create(
        model="o3-mini",
        messages=[{"role": "user", "content": "What is the 100th prime number?"}],
        reasoning_effort=effort,
    )
    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
    print(f"\\n{effort:6s} effort: {reasoning_tokens} reasoning tokens, "
          f"answer: {response.choices[0].message.content[:50]}")

# Process reward model concept
# The model internally evaluates each reasoning step
def conceptual_prm_scoring(steps):
    """Illustrates how a process reward model scores reasoning steps."""
    scores = {
        "problem decomposition": 0.95,
        "correct formula application": 0.90,
        "arithmetic check": 0.85,
        "verification of answer": 0.92,
        "final answer": 0.88,
    }
    for step, score in scores.items():
        print(f"  Step: {step:30s} -> PRM score: {score:.2f}")
    return min(scores.values())

print("\\nConceptual PRM scoring:")
min_score = conceptual_prm_scoring(None)
print(f"  Minimum step score: {min_score:.2f} (bottleneck for chain reliability)")`,id:"code-o1-api"}),e.jsx(t,{type:"intuition",title:"Why RL for Reasoning?",content:"Standard next-token prediction (SFT) only teaches the model to imitate reasoning traces. RL lets the model discover novel reasoning strategies by rewarding correct final answers. The model can learn that backtracking (going back and trying a different approach) is valuable, even though backtracking rarely appears in human-written text. RL optimizes for outcomes, not imitation.",id:"note-rl-reasoning"}),e.jsx(t,{type:"note",title:"Hidden Reasoning Tokens",content:"o1's internal reasoning tokens are not shown to the user (only a summary is provided). This serves multiple purposes: it protects proprietary reasoning patterns, reduces the visible output length, and prevents users from becoming confused by the model's internal deliberation process. The reasoning tokens do count toward pricing.",id:"note-hidden-reasoning"}),e.jsx(o,{title:"Cost and Latency",content:"o1 generates thousands of reasoning tokens before responding, making it 10-50x more expensive and slower than GPT-4o for simple tasks. Use o1/o3 only for genuinely difficult reasoning problems (math, code, logic). For simple Q&A, summarization, or creative writing, standard models are more cost-effective and often faster.",id:"warning-o1-cost"})]})}const Z=Object.freeze(Object.defineProperty({__proto__:null,default:L},Symbol.toStringTag,{value:"Module"}));function A(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"DeepSeek R1: Open Reasoning Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"DeepSeek-R1 (January 2025) was the first open-weight reasoning model to match OpenAI o1 on major benchmarks. Built on DeepSeek-V3, it demonstrated that reasoning capabilities could be achieved through pure reinforcement learning without supervised fine-tuning on reasoning traces, and released full model weights including distilled smaller variants."}),e.jsx(a,{title:"DeepSeek R1 Training Pipeline",definition:"DeepSeek-R1 uses a multi-stage training approach: (1) R1-Zero: pure RL with GRPO on the base model, demonstrating emergent reasoning without SFT, (2) Cold start: SFT on curated long-CoT data to improve readability, (3) RL training with rule-based rewards for math/code (verifiable) and model-based rewards for other tasks, (4) Rejection sampling + SFT for the final model.",id:"def-r1-pipeline"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"R1-Zero: Reasoning from Pure RL"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The most remarkable finding was R1-Zero: applying Group Relative Policy Optimization (GRPO) directly to the base DeepSeek-V3 model with only correctness rewards produced emergent chain-of-thought reasoning, self-verification, and reflection behaviors without any supervised reasoning examples. The model discovered these strategies on its own."}),e.jsx(n,{title:"DeepSeek R1 Benchmark Performance",problem:"Compare DeepSeek R1 against OpenAI o1 on key reasoning benchmarks.",steps:[{formula:"\\text{AIME 2024}: \\text{R1} = 79.8\\% \\text{ vs o1} = 79.2\\%",explanation:"R1 matches o1 on competitive mathematics with pass@1 accuracy."},{formula:"\\text{MATH-500}: \\text{R1} = 97.3\\% \\text{ vs o1} = 96.4\\%",explanation:"On the MATH benchmark, R1 slightly exceeds o1."},{formula:"\\text{Codeforces}: \\text{R1} = 96.3\\text{th pctile vs o1} = 96.6\\text{th pctile}",explanation:"Near-identical competitive programming performance."},{formula:"\\text{GPQA Diamond}: \\text{R1} = 71.5\\% \\text{ vs o1} = 78.0\\%",explanation:"R1 trails slightly on PhD-level science questions."}],id:"example-r1-benchmarks"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Group Relative Policy Optimization (GRPO)"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"GRPO is a variant of PPO that eliminates the need for a separate critic/value model. For each prompt, it samples a group of outputs, computes rewards for each, and uses the group's mean and standard deviation to normalize advantages. This makes it more memory-efficient than PPO while maintaining stable training."}),e.jsx(a,{title:"GRPO Advantage Estimation",definition:"For a prompt $q$ and a group of $G$ sampled outputs $\\{o_i\\}_{i=1}^{G}$ with rewards $\\{r_i\\}$, the advantage for output $o_i$ is computed relative to the group: $A_i = \\frac{r_i - \\text{mean}(\\{r_j\\})}{\\text{std}(\\{r_j\\})}$. This eliminates the need for a learned value function, reducing memory by approximately 50% compared to PPO.",id:"def-grpo"}),e.jsx(i,{title:"deepseek_r1_usage.py",code:`from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# DeepSeek-R1 distilled variants are practical to run locally
# Available: 1.5B, 7B, 8B, 14B, 32B, 70B distilled models
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# R1 generates <think>...</think> blocks before the final answer
prompt = """Solve this step by step:
If the sum of two numbers is 15 and their product is 56, what are the numbers?"""

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.6,
        top_p=0.95,
    )

response = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print(response)

# Parse thinking vs answer
if "<think>" in response and "</think>" in response:
    think_start = response.index("<think>") + len("<think>")
    think_end = response.index("</think>")
    thinking = response[think_start:think_end].strip()
    answer = response[think_end + len("</think>"):].strip()
    print(f"\\n--- Thinking ({len(thinking.split())} words) ---")
    print(thinking[:300] + "...")
    print(f"\\n--- Answer ---")
    print(answer)

# GRPO concept implementation
def grpo_advantages(rewards):
    """Compute GRPO-style group-relative advantages."""
    rewards = torch.tensor(rewards, dtype=torch.float32)
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8
    advantages = (rewards - mean_r) / std_r
    return advantages

# Example: 8 sampled outputs with binary rewards (correct/incorrect)
rewards = [1, 0, 1, 1, 0, 1, 0, 1]  # 5 correct, 3 incorrect
advantages = grpo_advantages(rewards)
print(f"\\nGRPO advantages: {advantages.tolist()}")
print(f"Correct outputs get positive advantage: {advantages[0].item():.3f}")
print(f"Incorrect outputs get negative advantage: {advantages[1].item():.3f}")`,id:"code-r1"}),e.jsx(t,{type:"historical",title:"Emergent Reasoning in R1-Zero",content:"R1-Zero spontaneously developed several reasoning behaviors through RL alone: (1) Extended thinking chains that grow longer for harder problems, (2) Self-verification ('let me check this'), (3) Backtracking ('wait, that's wrong, let me try again'), (4) Breaking problems into sub-problems. These emerged without any CoT training data, suggesting reasoning is a natural optimization target for RL on correctness rewards.",id:"note-r1-zero-emergence"}),e.jsx(t,{type:"tip",title:"Distilled Models",content:"DeepSeek released distilled R1 variants by fine-tuning Qwen and LLaMA base models on R1's reasoning traces. The 32B distilled model outperforms o1-mini on most benchmarks, while the 7B model provides strong reasoning capability on consumer hardware. These distilled models use standard SFT, making them easier to further fine-tune.",id:"note-distilled"}),e.jsx(o,{title:"Language Mixing in Reasoning",content:"R1 occasionally mixes Chinese and English in its thinking tokens, even when the prompt is entirely in English. This reflects the multilingual training data and can produce confusing intermediate reasoning. The distilled models based on LLaMA/Qwen tend to have less language mixing but it can still occur.",id:"warning-language-mixing"})]})}const Y=Object.freeze(Object.defineProperty({__proto__:null,default:A},Symbol.toStringTag,{value:"Module"}));function j(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Extended Thinking and Test-Time Compute"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:'Extended thinking represents a fundamental shift in how we scale AI capabilities. Rather than relying solely on larger models or more training data, test-time compute scaling allows models to "think longer" on harder problems. This paradigm, explored by OpenAI o1, DeepSeek R1, and Anthropic Claude, suggests that inference-time computation may be as important as training-time computation.'}),e.jsx(a,{title:"Test-Time Compute",definition:"Additional computation performed at inference time beyond a single forward pass, used to improve output quality. This includes generating reasoning tokens, search over solution candidates, self-verification, and iterative refinement. The key insight is that performance scales log-linearly with test-time compute: $\\text{accuracy} \\approx a + b \\cdot \\log(\\text{thinking tokens})$.",id:"def-test-time-compute"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The Scaling Paradigm Shift"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Traditional scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022) focused on training compute. The test-time compute paradigm adds a new dimension: a smaller model with more inference compute can match a larger model with less. This has profound implications for deployment economics and capability elicitation."}),e.jsx(s,{title:"Compute-Optimal Inference",statement:"For a fixed total inference budget B, the optimal strategy allocates compute between model size and thinking tokens. A smaller model thinking for longer can outperform a larger model answering immediately.",proof:e.jsx(r.BlockMath,{math:"B = C_{\\text{model}} \\cdot (N_{\\text{prompt}} + N_{\\text{thinking}} + N_{\\text{answer}})"}),corollaries:["C_model scales with parameter count. N_thinking can be varied per query.","For easy queries: minimize thinking tokens (fast, cheap response).","For hard queries: maximize thinking tokens (slow, expensive, but more accurate).","Optimal policy: adapt thinking budget to problem difficulty."],id:"theorem-compute-optimal"}),e.jsx(n,{title:"Test-Time Compute Strategies",problem:"Compare different approaches to spending test-time compute.",steps:[{formula:"\\text{Chain-of-Thought}: O(k) \\text{ tokens for } k \\text{ steps}",explanation:"Linear chain of reasoning. Simple but no error recovery. Used by CoT prompting."},{formula:"\\text{Self-Consistency}: O(n \\cdot k) \\text{ for } n \\text{ chains}",explanation:"Sample n independent chains and majority-vote. Better accuracy but linear cost increase."},{formula:"\\text{Tree Search}: O(b^d) \\text{ for branching } b, \\text{ depth } d",explanation:"Explore a tree of reasoning paths with pruning. Used internally by o1/o3 (likely)."},{formula:"\\text{Iterative Refinement}: O(r \\cdot k) \\text{ for } r \\text{ rounds}",explanation:"Generate, critique, and refine. Each round improves quality. Diminishing returns after 3-5 rounds."}],id:"example-ttc-strategies"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Process Reward Models"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Process Reward Models (PRMs) evaluate individual reasoning steps rather than just the final answer. This enables more granular credit assignment during RL training and allows search algorithms to prune bad reasoning branches early. Lightman et al. (2023) showed that PRMs dramatically improve performance when combined with best-of-N sampling."}),e.jsx(i,{title:"test_time_compute_strategies.py",code:`import torch
import random
from typing import List, Tuple

# Strategy 1: Best-of-N with reward model scoring
def best_of_n(generate_fn, reward_fn, prompt: str, n: int = 8) -> str:
    """Generate N responses and return the highest-scored one."""
    candidates = [generate_fn(prompt) for _ in range(n)]
    scores = [reward_fn(prompt, c) for c in candidates]
    best_idx = max(range(n), key=lambda i: scores[i])
    return candidates[best_idx], scores[best_idx]

# Strategy 2: Iterative refinement
def iterative_refine(generate_fn, critique_fn, prompt: str, rounds: int = 3) -> str:
    """Generate, critique, and refine iteratively."""
    response = generate_fn(prompt)
    for r in range(rounds):
        critique = critique_fn(prompt, response)
        if "correct" in critique.lower():
            break
        refined_prompt = f"{prompt}\\n\\nPrevious attempt: {response}\\nCritique: {critique}\\nPlease fix the issues."
        response = generate_fn(refined_prompt)
    return response

# Strategy 3: Tree search with process reward model
class ReasoningTree:
    """Simplified tree search for reasoning (conceptual)."""
    def __init__(self, step_generator, step_evaluator, beam_width=3):
        self.generate = step_generator
        self.evaluate = step_evaluator
        self.beam_width = beam_width

    def search(self, problem: str, max_depth: int = 5) -> List[str]:
        beams = [{"steps": [], "score": 0.0, "state": problem}]

        for depth in range(max_depth):
            all_candidates = []
            for beam in beams:
                # Generate candidate next steps
                next_steps = self.generate(beam["state"], n=3)
                for step in next_steps:
                    score = self.evaluate(beam["steps"] + [step])
                    all_candidates.append({
                        "steps": beam["steps"] + [step],
                        "score": beam["score"] + score,
                        "state": beam["state"] + "\\n" + step,
                    })
            # Keep top-k beams
            beams = sorted(all_candidates, key=lambda x: x["score"], reverse=True)[:self.beam_width]

        return beams[0]["steps"]

# Compute budget analysis
def compute_budget_analysis():
    """Analyze how different strategies spend compute."""
    strategies = {
        "Single pass": {"thinking_tokens": 0, "total_passes": 1},
        "CoT": {"thinking_tokens": 500, "total_passes": 1},
        "Self-consistency (8x)": {"thinking_tokens": 500, "total_passes": 8},
        "Best-of-16 + PRM": {"thinking_tokens": 500, "total_passes": 16},
        "Tree search (b=3, d=5)": {"thinking_tokens": 100, "total_passes": 3**5},
    }

    base_cost = 100  # tokens for prompt + answer
    for name, config in strategies.items():
        total_tokens = config["total_passes"] * (base_cost + config["thinking_tokens"])
        relative_cost = total_tokens / base_cost
        print(f"{name:30s}: {total_tokens:>8,} tokens ({relative_cost:>6.1f}x base cost)")

compute_budget_analysis()
# Single pass:                       100 tokens (   1.0x base cost)
# CoT:                               600 tokens (   6.0x base cost)
# Self-consistency (8x):           4,800 tokens (  48.0x base cost)
# Best-of-16 + PRM:                9,600 tokens (  96.0x base cost)
# Tree search (b=3, d=5):         48,600 tokens ( 486.0x base cost)`,id:"code-ttc"}),e.jsx(t,{type:"intuition",title:"The Inference Scaling Hypothesis",content:"Training scaling hits diminishing returns: doubling training compute yields small, predictable improvements. But test-time compute scaling may have a different curve. For reasoning tasks, giving a model 10x more thinking time can yield qualitative breakthroughs (e.g., solving a problem it couldn't before). This suggests a future where smaller, efficient models paired with adaptive inference compute may be more practical than ever-larger trained models.",id:"note-inference-scaling"}),e.jsx(t,{type:"note",title:"Adaptive Compute Budget",content:"Not all queries need the same amount of thinking. 'What is the capital of France?' needs zero thinking tokens, while 'Prove that there are infinitely many primes' benefits from extensive reasoning. Optimal systems should estimate query difficulty and allocate thinking budget accordingly. o3-mini's low/medium/high reasoning effort settings are an early version of this.",id:"note-adaptive-compute"}),e.jsx(o,{title:"Diminishing Returns and Overthinking",content:"Test-time compute scaling is not unlimited. Beyond a certain point, additional thinking tokens provide diminishing returns and can even hurt performance ('overthinking'). The model may second-guess correct answers, introduce unnecessary complexity, or hallucinate issues that don't exist. Knowing when to stop thinking is as important as knowing how to think.",id:"warning-overthinking"})]})}const J=Object.freeze(Object.defineProperty({__proto__:null,default:j},Symbol.toStringTag,{value:"Module"}));export{G as a,N as b,C as c,R as d,F as e,E as f,q as g,$ as h,O as i,K as j,I as k,D as l,Q as m,V as n,W as o,H as p,U as q,Z as r,S as s,Y as t,J as u};
