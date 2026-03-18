import{j as e}from"./vendor-DWbzdFaj.js";import"./vendor-katex-BYl39Yo6.js";import{D as o,E as s,P as t,N as a,W as n}from"./subject-01-text-fundamentals-DG6tAvii.js";function i(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Greedy & Beam Search"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"When an LLM generates text, it produces a probability distribution over the vocabulary at each step. Decoding strategies determine how we select the next token from that distribution. Greedy and beam search are the two foundational deterministic approaches."}),e.jsx(o,{title:"Greedy Decoding",definition:"Greedy decoding selects the token with the highest probability at each step: $y_t = \\arg\\max_{w} P(w \\mid y_{<t}, x)$. It is fast but can miss globally optimal sequences.",id:"def-greedy"}),e.jsx(s,{title:"Greedy vs Optimal",problem:"Given P('The') = 0.6, P('A') = 0.4 at step 1, and P('cat'|'The') = 0.3, P('dog'|'A') = 0.9, compare greedy vs optimal.",steps:[{formula:'Greedy: "The" (0.6) -> "cat" (0.3) = 0.18',explanation:"Greedy picks the highest probability at each step independently."},{formula:'Optimal: "A" (0.4) -> "dog" (0.9) = 0.36',explanation:'The globally better sequence was missed because greedy committed to "The" early.'},{formula:"Beam search with k=2 would find both paths",explanation:"By keeping multiple candidates, beam search avoids this trap."}],id:"example-greedy-vs-optimal"}),e.jsx(o,{title:"Beam Search",definition:"Beam search maintains $k$ candidate sequences (beams) at each step, expanding each by all vocabulary tokens and keeping the top-$k$ by cumulative log-probability: $\\text{score}(y) = \\sum_{t=1}^{T} \\log P(y_t \\mid y_{<t}, x)$.",notation:"$k$ is the beam width. $k=1$ reduces to greedy search.",id:"def-beam-search"}),e.jsx(t,{title:"greedy_decoding.py",code:`import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def greedy_decode(prompt, max_tokens=30):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = input_ids.clone()

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]  # last token logits
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

print(greedy_decode("The future of AI is"))
# Greedy often produces repetitive, generic text`,id:"code-greedy"}),e.jsx(t,{title:"beam_search.py",code:`import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def beam_search(prompt, beam_width=5, max_tokens=30):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Each beam: (token_ids, cumulative_log_prob)
    beams = [(input_ids, 0.0)]

    for _ in range(max_tokens):
        all_candidates = []
        for seq, score in beams:
            with torch.no_grad():
                logits = model(seq).logits[:, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)

            top_k_lp, top_k_ids = torch.topk(log_probs, beam_width)
            for i in range(beam_width):
                new_seq = torch.cat([seq, top_k_ids[:, i:i+1]], dim=-1)
                new_score = score + top_k_lp[0, i].item()
                all_candidates.append((new_seq, new_score))

        # Keep top-k beams by score
        beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    best_seq = beams[0][0]
    return tokenizer.decode(best_seq[0], skip_special_tokens=True)

print(beam_search("The future of AI is", beam_width=5))`,id:"code-beam-search"}),e.jsx(a,{type:"intuition",title:"Length Normalization",content:"Beam search tends to favor shorter sequences because log-probabilities are negative and accumulate. Length normalization divides the score by sequence length raised to a power alpha: score / T^alpha. A typical alpha is 0.6-0.7.",id:"note-length-norm"}),e.jsx(n,{title:"Beam Search Is Not Always Better",content:"For open-ended generation (stories, conversations), beam search tends to produce dull, repetitive text. It works best for tasks with a 'correct' answer like machine translation or summarization. For creative generation, sampling methods (next section) are preferred.",id:"warning-beam-limitations"}),e.jsx(a,{type:"tip",title:"HuggingFace Generate API",content:"In practice, use model.generate() with num_beams=5 for beam search or do_sample=False for greedy. The library handles length penalties, early stopping, and n-gram repetition penalties automatically.",id:"note-hf-generate"})]})}const le=Object.freeze(Object.defineProperty({__proto__:null,default:i},Symbol.toStringTag,{value:"Module"}));function r(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Top-k, Top-p & Nucleus Sampling"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Sampling introduces randomness into text generation, producing more diverse and creative outputs. Rather than always picking the most likely token, we sample from the probability distribution, optionally restricting which tokens are eligible."}),e.jsx(o,{title:"Pure Random Sampling",definition:"Sample the next token directly from the full vocabulary distribution: $y_t \\sim P(w \\mid y_{<t}, x)$. This can produce incoherent text because low-probability tokens have a non-zero chance of being selected.",id:"def-random-sampling"}),e.jsx(o,{title:"Top-k Sampling",definition:"Restrict sampling to the $k$ most probable tokens, redistributing probability mass among them. Formally, let $V^{(k)}$ be the top-$k$ tokens, then $P'(w) = P(w) / \\sum_{w' \\in V^{(k)}} P(w')$ for $w \\in V^{(k)}$, and $0$ otherwise.",notation:"$k$ = number of tokens to keep. GPT-2 used $k = 40$ by default.",id:"def-top-k"}),e.jsx(o,{title:"Top-p (Nucleus) Sampling",definition:"Select the smallest set of tokens whose cumulative probability exceeds $p$: $V^{(p)} = \\min\\{V' \\subseteq V : \\sum_{w \\in V'} P(w) \\geq p\\}$. This adapts the number of candidates dynamically based on the distribution shape.",notation:"$p \\in (0, 1]$. Typical values: $p = 0.9$ or $p = 0.95$.",id:"def-top-p"}),e.jsx(s,{title:"Top-k vs Top-p Behavior",problem:"Compare top-k=3 and top-p=0.9 on a peaked vs flat distribution.",steps:[{formula:"Peaked: P = [0.7, 0.15, 0.08, 0.04, 0.03]",explanation:"Top-k=3 keeps [0.7, 0.15, 0.08]. Top-p=0.9 keeps [0.7, 0.15, 0.08] (sum=0.93). Similar result."},{formula:"Flat: P = [0.22, 0.20, 0.19, 0.18, 0.11, 0.10]",explanation:"Top-k=3 keeps only [0.22, 0.20, 0.19]. Top-p=0.9 keeps [0.22, 0.20, 0.19, 0.18, 0.11] (sum=0.90). More tokens."},{formula:"Top-p adapts to uncertainty",explanation:"When the model is uncertain (flat distribution), top-p allows more diversity. When confident, it restricts choices."}],id:"example-topk-topp"}),e.jsx(t,{title:"sampling_strategies.py",code:`import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def sample_next_token(logits, top_k=0, top_p=1.0, temperature=1.0):
    """Apply top-k and/or top-p filtering then sample."""
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k_vals, _ = torch.topk(logits, top_k)
        min_top_k = top_k_vals[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_top_k,
                             torch.full_like(logits, float('-inf')),
                             logits)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative prob above threshold
        remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[remove_mask] = float('-inf')
        # Scatter back to original ordering
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# Generate with different strategies
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

for name, kwargs in [
    ("Top-k=50", {"top_k": 50}),
    ("Top-p=0.9", {"top_p": 0.9}),
    ("Top-k=50 + Top-p=0.95", {"top_k": 50, "top_p": 0.95}),
]:
    gen = input_ids.clone()
    for _ in range(30):
        with torch.no_grad():
            logits = model(gen).logits[:, -1, :]
        next_id = sample_next_token(logits, **kwargs)
        gen = torch.cat([gen, next_id], dim=-1)
    print(f"{name}: {tokenizer.decode(gen[0])}")`,id:"code-sampling"}),e.jsx(t,{title:"hf_sampling.py",code:`from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

input_ids = tokenizer.encode("The best way to learn is", return_tensors="pt")

# HuggingFace generate with sampling parameters
output = model.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=True,
    top_k=50,
    top_p=0.92,
    temperature=0.8,
    no_repeat_ngram_size=3,  # prevent 3-gram repetition
)

print(tokenizer.decode(output[0], skip_special_tokens=True))`,id:"code-hf-sampling"}),e.jsx(a,{type:"tip",title:"Combining Top-k and Top-p",content:"In practice, top-k and top-p are often used together. Top-k provides a hard ceiling on candidates, while top-p further refines within that set. Most APIs (OpenAI, Anthropic) expose top_p as the primary sampling parameter.",id:"note-combining"}),e.jsx(n,{title:"Reproducibility with Sampling",content:"Sampling is stochastic -- the same prompt produces different outputs each run. Set a random seed (torch.manual_seed) for reproducible results during development. In production, embrace the variability or use temperature=0 for deterministic behavior.",id:"warning-reproducibility"})]})}const me=Object.freeze(Object.defineProperty({__proto__:null,default:r},Symbol.toStringTag,{value:"Module"}));function l(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Temperature & Repetition Penalties"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Temperature scaling and repetition penalties are two critical knobs for controlling LLM output quality. Temperature adjusts the sharpness of the probability distribution, while repetition penalties discourage the model from repeating itself."}),e.jsx(o,{title:"Temperature Scaling",definition:"Temperature divides logits before the softmax: $P(w_i) = \\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)}$. As $T \\to 0$, the distribution becomes a one-hot (greedy). As $T \\to \\infty$, it becomes uniform (random).",notation:"$T$ = temperature. $T=1$ is the default (unmodified distribution). $T<1$ sharpens. $T>1$ flattens.",id:"def-temperature"}),e.jsx(s,{title:"Temperature Effects on a Distribution",problem:"Given logits [2.0, 1.0, 0.5, 0.1], compute probabilities at T=0.5, T=1.0, and T=2.0.",steps:[{formula:"T=0.5: softmax([4.0, 2.0, 1.0, 0.2]) \\approx [0.84, 0.11, 0.04, 0.01]",explanation:"Low temperature concentrates mass on the top token."},{formula:"T=1.0: softmax([2.0, 1.0, 0.5, 0.1]) \\approx [0.47, 0.17, 0.10, 0.07]",explanation:"Default temperature preserves the original distribution shape."},{formula:"T=2.0: softmax([1.0, 0.5, 0.25, 0.05]) \\approx [0.33, 0.20, 0.16, 0.13]",explanation:"High temperature makes unlikely tokens more probable."}],id:"example-temperature"}),e.jsx(t,{title:"temperature_demo.py",code:`import torch
import torch.nn.functional as F
import numpy as np

logits = torch.tensor([[2.0, 1.0, 0.5, 0.1, -0.5]])
tokens = ["the", "a", "an", "one", "some"]

print("Temperature effects on probability distribution:")
print(f"{'Token':<8} {'T=0.3':<10} {'T=0.7':<10} {'T=1.0':<10} {'T=1.5':<10} {'T=2.0':<10}")
print("-" * 58)

for temp in [0.3, 0.7, 1.0, 1.5, 2.0]:
    probs = F.softmax(logits / temp, dim=-1)[0]
    row = f""
    for i, tok in enumerate(tokens):
        row = f"{tok:<8}" if i == 0 else row
    vals = "  ".join(f"{p:.4f}" for p in probs.tolist())
    print(f"         {vals}  (T={temp})")

# Entropy as a measure of randomness
for temp in [0.3, 0.7, 1.0, 1.5, 2.0]:
    probs = F.softmax(logits / temp, dim=-1)[0]
    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    print(f"T={temp}: entropy = {entropy:.3f}")`,id:"code-temperature"}),e.jsx(o,{title:"Repetition Penalty",definition:"Repetition penalty reduces the logit of previously generated tokens: if token $w$ appeared before, $z_w' = z_w / \\\\theta$ when $z_w > 0$, or $z_w' = z_w \\\\cdot \\\\theta$ when $z_w < 0$, where $\\\\theta > 1$ is the penalty factor.",notation:"$\\\\theta$ = repetition penalty. $\\\\theta=1.0$ means no penalty. Typical range: $1.1$ to $1.3$.",id:"def-repetition-penalty"}),e.jsx(t,{title:"repetition_penalty.py",code:`import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "The cat sat on the mat. The cat"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Without repetition penalty -- likely to repeat
output_no_penalty = model.generate(
    input_ids, max_new_tokens=40,
    do_sample=True, temperature=0.7, top_p=0.9,
    repetition_penalty=1.0,
)

# With repetition penalty
output_with_penalty = model.generate(
    input_ids, max_new_tokens=40,
    do_sample=True, temperature=0.7, top_p=0.9,
    repetition_penalty=1.2,
)

# With no_repeat_ngram_size (prevents exact n-gram repeats)
output_ngram = model.generate(
    input_ids, max_new_tokens=40,
    do_sample=True, temperature=0.7, top_p=0.9,
    no_repeat_ngram_size=3,
)

print("No penalty:", tokenizer.decode(output_no_penalty[0]))
print("Rep penalty 1.2:", tokenizer.decode(output_with_penalty[0]))
print("No 3-gram repeat:", tokenizer.decode(output_ngram[0]))`,id:"code-repetition"}),e.jsx(a,{type:"tip",title:"Practical Temperature Guidelines",content:"For factual Q&A and code: T=0.0-0.3. For general conversation: T=0.5-0.8. For creative writing and brainstorming: T=0.8-1.2. Never go above T=2.0 in production -- output becomes incoherent.",id:"note-temperature-guidelines"}),e.jsx(a,{type:"note",title:"Frequency vs Presence Penalty",content:"OpenAI's API offers two distinct penalties: frequency_penalty scales with how many times a token appeared, while presence_penalty applies a flat penalty if a token appeared at all. They serve different purposes -- frequency prevents overuse while presence encourages topic diversity.",id:"note-freq-presence"}),e.jsx(n,{title:"Temperature Zero Is Not Deterministic Everywhere",content:"While temperature=0 should be deterministic, floating-point rounding in GPU operations can cause slight non-determinism. For truly reproducible outputs, also set random seeds and use deterministic CUDA operations.",id:"warning-determinism"})]})}const pe=Object.freeze(Object.defineProperty({__proto__:null,default:l},Symbol.toStringTag,{value:"Module"}));function m(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Structured & Constrained Generation"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"LLMs generate free-form text, but many applications require structured outputs like JSON, SQL, or XML. Constrained generation techniques guarantee that model output conforms to a specified schema or grammar, eliminating parsing failures entirely."}),e.jsx(o,{title:"Constrained Decoding",definition:"Constrained decoding modifies the sampling distribution at each step to only allow tokens that are valid according to a grammar or schema. Invalid tokens receive $-\\infty$ logits (are masked out), ensuring 100% conformance.",id:"def-constrained"}),e.jsx(a,{type:"intuition",title:"How It Works",content:`Think of it like autocomplete with strict rules. At each token position, a finite state machine or parser tracks what tokens are valid next. If you have generated '{"name": "' so far, the only valid continuations are string characters -- not a closing brace or number. The model still chooses among valid tokens using its learned distribution.`,id:"note-how-it-works"}),e.jsx(t,{title:"outlines_json.py",code:`# Outlines: structured generation library
# pip install outlines
import outlines

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")

# Define a JSON schema
from pydantic import BaseModel
from typing import List

class MovieReview(BaseModel):
    title: str
    year: int
    rating: float  # 0-10
    genres: List[str]
    summary: str

# Create a generator constrained to this schema
generator = outlines.generate.json(model, MovieReview)

prompt = """Review the movie Inception and provide a structured review.
Output your review as JSON:"""

# Output is GUARANTEED to be valid JSON matching the schema
review = generator(prompt)
print(review)
# MovieReview(title='Inception', year=2010, rating=9.2,
#   genres=['Sci-Fi', 'Thriller'], summary='A mind-bending...')`,id:"code-outlines-json"}),e.jsx(t,{title:"outlines_regex.py",code:`import outlines

model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")

# Constrain to a regex pattern -- e.g., US phone number
phone_generator = outlines.generate.regex(
    model,
    r"\\(\\d{3}\\) \\d{3}-\\d{4}"
)

result = phone_generator("Generate a US phone number: ")
print(result)  # Always matches (XXX) XXX-XXXX format

# Constrain to specific choices
sentiment_generator = outlines.generate.choice(
    model,
    ["positive", "negative", "neutral"]
)

sentiment = sentiment_generator(
    "Classify the sentiment: 'This movie was absolutely terrible.' -> "
)
print(sentiment)  # Always one of the three choices

# Grammar-based generation (e.g., arithmetic expressions)
grammar = r"""
    start: expr
    expr: term (("+"|"-") term)*
    term: NUMBER
    NUMBER: /[0-9]+/
"""
math_gen = outlines.generate.cfg(model, grammar)
result = math_gen("Write a math expression: ")
print(result)  # e.g., "42+17-3"`,id:"code-outlines-regex"}),e.jsx(s,{title:"Guidance Template Example",problem:"Use Microsoft Guidance to create a structured character profile.",steps:[{formula:"Define template with {{gen}} blocks",explanation:"Guidance uses template syntax where {{gen}} marks places the LLM fills in."},{formula:"Add constraints: stop tokens, regex, choices",explanation:'Each gen block can have constraints like stop="\\n" or pattern="[0-9]+".'},{formula:"LLM fills in constrained slots",explanation:"The model generates text that must satisfy all constraints, producing a valid structured output."}],id:"example-guidance"}),e.jsx(t,{title:"instructor_openai.py",code:`# Instructor: structured outputs from any OpenAI-compatible API
# pip install instructor openai
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

# Patch OpenAI client with instructor
client = instructor.from_openai(OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama
    api_key="ollama",
))

class ExtractedEntity(BaseModel):
    name: str
    entity_type: str = Field(description="person, org, location, etc.")
    confidence: float = Field(ge=0.0, le=1.0)

class ExtractionResult(BaseModel):
    entities: List[ExtractedEntity]
    summary: str

# Instructor ensures the response matches the schema
result = client.chat.completions.create(
    model="llama3.2",
    response_model=ExtractionResult,
    messages=[{
        "role": "user",
        "content": "Extract entities: Apple CEO Tim Cook announced new AI features at WWDC in Cupertino."
    }],
)

for entity in result.entities:
    print(f"  {entity.name} ({entity.entity_type}): {entity.confidence:.0%}")`,id:"code-instructor"}),e.jsx(a,{type:"note",title:"Performance Impact",content:"Constrained generation adds minimal overhead. The grammar/schema check at each step is fast (typically microseconds) compared to the model forward pass (milliseconds). The main cost is that masking tokens can sometimes force the model into lower-quality continuations.",id:"note-performance"}),e.jsx(n,{title:"Structured Output Does Not Mean Correct Output",content:"Constrained generation guarantees syntactic validity (valid JSON, matching schema) but not semantic correctness. The model can still hallucinate values, provide wrong numbers, or fill fields with nonsensical content. Always validate the content, not just the structure.",id:"warning-correctness"})]})}const ce=Object.freeze(Object.defineProperty({__proto__:null,default:m},Symbol.toStringTag,{value:"Module"}));function p(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"KV-Cache & Paged Attention"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Autoregressive generation recomputes attention over all previous tokens at each step. The KV-cache stores previously computed key and value matrices, converting generation from quadratic to linear complexity per step. Paged Attention further optimizes memory allocation."}),e.jsx(o,{title:"KV-Cache",definition:"During autoregressive generation, the key ($K$) and value ($V$) projections for all previous tokens are cached in GPU memory. At step $t$, only the new token's $Q$, $K$, $V$ are computed, and attention is: $\\text{Attn}(q_t, K_{1:t}, V_{1:t}) = \\text{softmax}(q_t K_{1:t}^T / \\sqrt{d_k}) V_{1:t}$.",notation:"Without cache: $O(t^2 d)$ per step. With cache: $O(t \\cdot d)$ per step.",id:"def-kv-cache"}),e.jsx(s,{title:"KV-Cache Memory Calculation",problem:"Calculate KV-cache memory for LLaMA-3 8B with sequence length 4096.",steps:[{formula:"Layers = 32, heads = 32, d_{head} = 128",explanation:"LLaMA-3 8B architecture parameters."},{formula:"KV per token = 2 \\times 32 \\times 32 \\times 128 \\times 2 = 524\\text{KB}",explanation:"2 for K and V, 32 layers, 32 heads, 128 dims, 2 bytes (fp16)."},{formula:"Total for 4096 tokens: 524\\text{KB} \\times 4096 \\approx 2\\text{GB}",explanation:"Each sequence requires ~2GB of KV-cache in GPU memory."},{formula:"With GQA (8 KV heads): 2GB / 4 = 512\\text{MB}",explanation:"Grouped-Query Attention reduces KV-cache by the group factor."}],id:"example-kv-memory"}),e.jsx(t,{title:"kv_cache_demo.py",code:`import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()

prompt = "The key-value cache is important because"
input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

# WITHOUT KV-cache: recompute everything each step
start = time.time()
generated = input_ids.clone()
for _ in range(50):
    with torch.no_grad():
        outputs = model(generated)
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        generated = torch.cat([generated, next_token], dim=-1)
no_cache_time = time.time() - start

# WITH KV-cache: only compute the new token
start = time.time()
generated = input_ids.clone()
past_key_values = None
for _ in range(50):
    with torch.no_grad():
        outputs = model(
            generated[:, -1:] if past_key_values else generated,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        generated = torch.cat([generated, next_token], dim=-1)
cache_time = time.time() - start

print(f"Without KV-cache: {no_cache_time:.3f}s")
print(f"With KV-cache:    {cache_time:.3f}s")
print(f"Speedup:          {no_cache_time/cache_time:.1f}x")`,id:"code-kv-cache"}),e.jsx(o,{title:"Paged Attention",definition:"Paged Attention (vLLM) manages KV-cache like virtual memory pages. Instead of pre-allocating contiguous memory for the maximum sequence length, it allocates fixed-size blocks on demand and uses a block table to map logical positions to physical memory locations.",id:"def-paged-attention"}),e.jsx(a,{type:"intuition",title:"Why Paging Matters",content:"Without paging, serving a batch of 8 requests with max_length=4096 requires pre-allocating 8 x 2GB = 16GB even if most sequences are short. Paged Attention only allocates memory as tokens are generated, reducing waste by 60-80% and enabling 2-4x higher throughput.",id:"note-paging-intuition"}),e.jsx(t,{title:"Terminal",code:`# vLLM uses Paged Attention automatically
# Install and run vLLM to see the benefits
pip install vllm

# Start vLLM server with automatic KV-cache management
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --gpu-memory-utilization 0.9 \\
    --max-model-len 8192

# vLLM reports KV-cache usage in logs:
# INFO: GPU blocks: 2450, CPU blocks: 512
# Each block holds KV for 16 tokens
# Total KV-cache capacity: 2450 * 16 = 39,200 tokens`,id:"code-vllm-paged"}),e.jsx(n,{title:"KV-Cache Is the Memory Bottleneck",content:"For long-context models (128K+ tokens), KV-cache can exceed the model weights in memory usage. A 7B model with 128K context needs ~32GB just for KV-cache in fp16. Techniques like GQA, KV-cache quantization (fp8), and sliding window attention are essential for long contexts.",id:"warning-memory"}),e.jsx(a,{type:"note",title:"Prefix Caching",content:"When multiple requests share a common prefix (system prompt), the KV-cache for that prefix can be computed once and shared. vLLM supports automatic prefix caching, which is especially valuable for chat applications where every request includes the same system prompt.",id:"note-prefix-caching"})]})}const de=Object.freeze(Object.defineProperty({__proto__:null,default:p},Symbol.toStringTag,{value:"Module"}));function c(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Continuous Batching"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Static batching wastes GPU cycles waiting for the longest sequence in a batch to finish. Continuous (or dynamic) batching allows new requests to join and completed requests to leave the batch at every iteration step, maximizing GPU utilization."}),e.jsx(o,{title:"Static Batching",definition:"In static batching, a batch of requests starts together and the GPU waits for all sequences to reach their max length or EOS token. If one request generates 10 tokens and another generates 500, the GPU is idle for the short request during 490 steps.",id:"def-static-batching"}),e.jsx(o,{title:"Continuous Batching",definition:"Continuous batching (iteration-level scheduling) checks after every forward pass whether any sequence has finished. Completed sequences are evicted and new requests from the queue are inserted immediately. The batch size fluctuates dynamically.",id:"def-continuous-batching"}),e.jsx(s,{title:"Throughput Comparison",problem:"Compare static vs continuous batching with 4 requests: lengths 20, 50, 100, 200 tokens.",steps:[{formula:"Static: all wait for 200 steps = 800 total slot-steps",explanation:"Each of 4 slots runs for 200 steps regardless of actual output length."},{formula:"Useful work: 20+50+100+200 = 370 slot-steps",explanation:"Only 370/800 = 46% GPU utilization."},{formula:"Continuous: slot freed at step 20, new request enters",explanation:"As soon as the 20-token request finishes, a new request fills its slot."},{formula:"Continuous achieves 85-95% utilization",explanation:"The GPU is almost always doing useful work."}],id:"example-throughput"}),e.jsx(t,{title:"continuous_batching_sim.py",code:`import random
import time
from collections import deque

class Request:
    def __init__(self, id, prompt_len, output_len):
        self.id = id
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.tokens_generated = 0
        self.start_time = None

    @property
    def done(self):
        return self.tokens_generated >= self.output_len

def simulate_batching(requests, max_batch_size, mode="continuous"):
    queue = deque(requests)
    batch = []
    step = 0
    completed = []

    while queue or batch:
        # Fill batch from queue
        while len(batch) < max_batch_size and queue:
            req = queue.popleft()
            req.start_time = step
            batch.append(req)

        # One forward pass -- generate one token per request
        step += 1
        for req in batch:
            req.tokens_generated += 1

        if mode == "continuous":
            # Remove finished requests immediately
            finished = [r for r in batch if r.done]
            batch = [r for r in batch if not r.done]
            for r in finished:
                r.end_time = step
                completed.append(r)
        else:  # static
            if all(r.done for r in batch):
                for r in batch:
                    r.end_time = step
                    completed.extend(batch)
                batch = []

    return completed, step

# Generate 20 requests with varying output lengths
random.seed(42)
reqs_cont = [Request(i, 10, random.randint(10, 200)) for i in range(20)]
reqs_stat = [Request(i, 10, reqs_cont[i].output_len) for i in range(20)]

completed_c, steps_c = simulate_batching(reqs_cont, max_batch_size=4, mode="continuous")
completed_s, steps_s = simulate_batching(reqs_stat, max_batch_size=4, mode="static")

avg_latency_c = sum(r.end_time - r.start_time for r in completed_c) / len(completed_c)
avg_latency_s = sum(r.end_time - r.start_time for r in completed_s) / len(completed_s)

print(f"Static batching:     {steps_s} steps, avg latency {avg_latency_s:.0f}")
print(f"Continuous batching: {steps_c} steps, avg latency {avg_latency_c:.0f}")
print(f"Throughput improvement: {steps_s/steps_c:.2f}x")`,id:"code-simulation"}),e.jsx(t,{title:"Terminal",code:`# vLLM uses continuous batching by default
# Start with specific batch settings:
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --max-num-seqs 256 \\
    --max-num-batched-tokens 8192

# TGI also supports continuous batching:
docker run --gpus all -p 8080:80 \\
    ghcr.io/huggingface/text-generation-inference:latest \\
    --model-id meta-llama/Llama-3.1-8B-Instruct \\
    --max-batch-total-tokens 8192 \\
    --max-concurrent-requests 128`,id:"code-frameworks"}),e.jsx(a,{type:"note",title:"Prefill vs Decode Phases",content:"Each request has two phases: prefill (processing the full prompt in parallel) and decode (generating tokens one by one). Prefill is compute-bound while decode is memory-bound. Advanced schedulers like chunked prefill interleave these phases to balance GPU compute and memory bandwidth.",id:"note-prefill-decode"}),e.jsx(n,{title:"Batch Size vs Latency Tradeoff",content:"Larger batches improve throughput but increase per-request latency because each forward pass takes longer. Monitor time-to-first-token (TTFT) and inter-token latency (ITL) alongside throughput when tuning batch sizes.",id:"warning-latency"})]})}const ue=Object.freeze(Object.defineProperty({__proto__:null,default:c},Symbol.toStringTag,{value:"Module"}));function d(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Speculative Decoding"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Speculative decoding uses a small, fast draft model to propose multiple tokens, then verifies them in parallel with the large target model. When the draft model's predictions match, you get multiple tokens per forward pass of the large model, dramatically reducing latency."}),e.jsx(o,{title:"Speculative Decoding",definition:"A draft model $M_q$ generates $\\gamma$ candidate tokens autoregressively. The target model $M_p$ then scores all $\\gamma$ tokens in a single forward pass. Tokens are accepted from left to right using a rejection sampling scheme that preserves the target distribution exactly.",notation:"$\\gamma$ = speculation length (typically 3-5). $\\alpha$ = acceptance rate (fraction of draft tokens accepted).",id:"def-speculative"}),e.jsx(s,{title:"Speculative Decoding Walk-through",problem:"Draft model proposes 4 tokens. Target model verifies them.",steps:[{formula:'Draft proposes: ["The", "cat", "sat", "on"]',explanation:"Small model (e.g., 1B params) generates 4 tokens very quickly."},{formula:"Target scores all 4 in one pass",explanation:"Large model (e.g., 70B) processes all 4 tokens in parallel, as fast as processing 1."},{formula:'Accept: "The" \\checkmark, "cat" \\checkmark, "sat" \\checkmark, "on" \\times',explanation:'First 3 match target distribution (accepted). "on" rejected -- target preferred "down".'},{formula:"Result: 3 tokens + 1 corrected = 4 tokens from 1 large forward pass",explanation:"Instead of 4 large model calls, we used 1 large + 4 small calls. Net speedup ~2-3x."}],id:"example-speculative"}),e.jsx(t,{title:"speculative_decoding.py",code:`import torch
import torch.nn.functional as F

def speculative_decode(
    target_model, draft_model, tokenizer, prompt,
    gamma=4, max_tokens=100
):
    """Simplified speculative decoding loop."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    generated = input_ids.clone()
    total_tokens = 0
    accepted_tokens = 0

    while total_tokens < max_tokens:
        # Step 1: Draft model generates gamma candidates
        draft_ids = generated.clone()
        draft_probs_list = []
        for _ in range(gamma):
            with torch.no_grad():
                logits = draft_model(draft_ids).logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                draft_probs_list.append(probs)
                next_token = torch.multinomial(probs, 1)
                draft_ids = torch.cat([draft_ids, next_token], dim=-1)

        # Step 2: Target model scores all draft tokens in ONE pass
        with torch.no_grad():
            target_logits = target_model(draft_ids).logits
            # Get target probs for each draft position
            n = generated.shape[1]
            target_probs = [
                F.softmax(target_logits[:, n + i - 1, :], dim=-1)
                for i in range(gamma)
            ]

        # Step 3: Accept/reject using rejection sampling
        accepted = 0
        for i in range(gamma):
            draft_token = draft_ids[:, n + i]
            p = target_probs[i][:, draft_token].item()
            q = draft_probs_list[i][:, draft_token].item()

            # Accept with probability min(1, p/q)
            if torch.rand(1).item() < min(1.0, p / q):
                accepted += 1
            else:
                # Reject: sample from adjusted distribution
                adjusted = torch.clamp(target_probs[i] - draft_probs_list[i], min=0)
                adjusted = adjusted / adjusted.sum()
                correction = torch.multinomial(adjusted, 1)
                generated = torch.cat([
                    generated, draft_ids[:, n:n+accepted], correction
                ], dim=-1)
                break
        else:
            # All accepted -- also sample one bonus token
            bonus = torch.multinomial(
                F.softmax(target_logits[:, -1, :], dim=-1), 1
            )
            generated = torch.cat([
                generated, draft_ids[:, n:n+gamma], bonus
            ], dim=-1)
            accepted += 1

        total_tokens += accepted + 1
        accepted_tokens += accepted

    acceptance_rate = accepted_tokens / (total_tokens or 1)
    return tokenizer.decode(generated[0]), acceptance_rate`,id:"code-speculative"}),e.jsx(t,{title:"Terminal",code:`# vLLM supports speculative decoding natively
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-70B-Instruct \\
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \\
    --num-speculative-tokens 5 \\
    --gpu-memory-utilization 0.9

# Or use ngram-based speculation (no draft model needed)
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-70B-Instruct \\
    --speculative-model "[ngram]" \\
    --ngram-prompt-lookup-max 4 \\
    --num-speculative-tokens 5`,id:"code-vllm-spec"}),e.jsx(a,{type:"intuition",title:"Why It Preserves the Target Distribution",content:"The rejection sampling scheme is mathematically designed so that the final output distribution is identical to the target model. If the draft model is perfect (matches the target exactly), all tokens are accepted. If it is terrible, we fall back to sampling one token per large model call -- never worse than standard decoding.",id:"note-distribution"}),e.jsx(n,{title:"Draft Model Quality Matters",content:"The speedup depends on the acceptance rate, which depends on how well the draft model approximates the target. A poor draft model means most tokens are rejected and you get no benefit. Best results come from draft models in the same family (e.g., LLaMA-1B drafting for LLaMA-70B).",id:"warning-draft-quality"}),e.jsx(a,{type:"note",title:"Self-Speculative Decoding",content:"Some approaches skip the separate draft model entirely. Medusa adds extra prediction heads to the target model. EAGLE uses the target model's own hidden states. Layer-skipping uses a subset of the target model's layers as the draft. These avoid the complexity of maintaining two models.",id:"note-self-speculative"})]})}const he=Object.freeze(Object.defineProperty({__proto__:null,default:d},Symbol.toStringTag,{value:"Module"}));function u(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Tensor Parallelism for Inference"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"When a model is too large for a single GPU, tensor parallelism splits individual weight matrices across multiple GPUs so they compute portions of each layer simultaneously. Pipeline parallelism is the alternative, splitting by layers instead of within layers."}),e.jsx(o,{title:"Tensor Parallelism (TP)",definition:"Tensor parallelism partitions weight matrices across $N$ GPUs. For a linear layer $Y = XW$, the weight $W$ is split column-wise: $W = [W_1 | W_2 | \\ldots | W_N]$. Each GPU computes $Y_i = X W_i$, then results are combined via all-reduce or all-gather.",notation:"TP degree $N$ = number of GPUs sharing each layer. Communication cost: one all-reduce per layer.",id:"def-tensor-parallelism"}),e.jsx(o,{title:"Pipeline Parallelism (PP)",definition:"Pipeline parallelism assigns different layers to different GPUs. GPU 0 runs layers 0-15, GPU 1 runs layers 16-31. Data flows sequentially through GPUs. This requires less inter-GPU communication but creates pipeline bubbles.",id:"def-pipeline-parallelism"}),e.jsx(s,{title:"LLaMA-3.1 70B Across GPUs",problem:"How to serve a 70B parameter model that requires ~140GB in fp16?",steps:[{formula:"Single GPU (80GB A100): not enough for fp16",explanation:"Need at least 140GB for weights alone, plus KV-cache and activations."},{formula:"TP=2 on 2x A100: 70GB weights per GPU",explanation:"Each GPU holds half of every weight matrix. Fast all-reduce via NVLink."},{formula:"TP=4 on 4x A100: 35GB per GPU, room for large batches",explanation:"More headroom for KV-cache means higher throughput."},{formula:"PP=2 + TP=2 on 4x A100: hybrid approach",explanation:"Layers 0-39 on GPUs 0-1 (TP=2), layers 40-79 on GPUs 2-3 (TP=2)."}],id:"example-70b-serving"}),e.jsx(t,{title:"Terminal",code:`# vLLM with tensor parallelism
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-70B-Instruct \\
    --tensor-parallel-size 4 \\
    --gpu-memory-utilization 0.9 \\
    --max-model-len 8192

# TGI with tensor parallelism
docker run --gpus all -p 8080:80 \\
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \\
    ghcr.io/huggingface/text-generation-inference:latest \\
    --model-id meta-llama/Llama-3.1-70B-Instruct \\
    --num-shard 4

# Check GPU memory distribution
nvidia-smi --query-gpu=index,memory.used,memory.total \\
    --format=csv,noheader,nounits`,id:"code-tp-serving"}),e.jsx(t,{title:"tp_benchmark.py",code:`import requests
import time
import concurrent.futures

API_URL = "http://localhost:8000/v1/completions"

def send_request(prompt):
    start = time.time()
    resp = requests.post(API_URL, json={
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
    })
    latency = time.time() - start
    tokens = resp.json()["usage"]["completion_tokens"]
    return latency, tokens

# Benchmark: measure throughput with concurrent requests
prompts = [f"Explain concept {i} in machine learning:" for i in range(32)]

start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    results = list(pool.map(send_request, prompts))
total_time = time.time() - start

total_tokens = sum(t for _, t in results)
avg_latency = sum(l for l, _ in results) / len(results)

print(f"Total time: {total_time:.1f}s")
print(f"Throughput: {total_tokens/total_time:.0f} tokens/sec")
print(f"Avg latency: {avg_latency:.2f}s")
print(f"Requests completed: {len(results)}")`,id:"code-benchmark"}),e.jsx(a,{type:"tip",title:"Choosing TP vs PP",content:"Use tensor parallelism when GPUs are connected via fast interconnects (NVLink, 600 GB/s). Use pipeline parallelism when GPUs are on different nodes connected by slower networks (InfiniBand, 200 Gb/s). For inference, TP is almost always preferred because it reduces latency -- every GPU participates in every token.",id:"note-tp-vs-pp"}),e.jsx(n,{title:"Diminishing Returns with High TP",content:"Communication overhead grows with TP degree. TP=8 on 8 GPUs means 8 all-reduce operations per layer. If the all-reduce time exceeds the compute time, adding more GPUs actually hurts latency. For small models, TP=2 is often the sweet spot.",id:"warning-diminishing-returns"}),e.jsx(a,{type:"note",title:"Quantization as an Alternative",content:"Before reaching for multi-GPU serving, consider quantization. A 70B model in 4-bit quantization fits on a single 80GB GPU (~35GB). This avoids all communication overhead. Quality loss is minimal with modern quantization methods (GPTQ, AWQ).",id:"note-quantization-alt"})]})}const ge=Object.freeze(Object.defineProperty({__proto__:null,default:u},Symbol.toStringTag,{value:"Module"}));function h(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"What is Ollama & Why It Matters"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Ollama is an open-source tool that makes running large language models locally as simple as running a Docker container. It wraps llama.cpp with a user-friendly CLI, REST API, and model management system, making local LLM inference accessible to everyone."}),e.jsx(o,{title:"Ollama",definition:"Ollama is a local LLM runtime that downloads, manages, and serves quantized language models. It provides a Docker-like experience for LLMs: pull a model by name, run it with one command, and interact via CLI or REST API.",id:"def-ollama"}),e.jsx(a,{type:"intuition",title:"Docker for LLMs",content:"Just as Docker made it trivial to run complex server software with 'docker run nginx', Ollama makes it trivial to run LLMs with 'ollama run llama3.2'. It handles model downloads, quantization format selection, GPU detection, memory management, and API serving -- all automatically.",id:"note-docker-analogy"}),e.jsx(s,{title:"Ollama in 30 Seconds",problem:"Go from zero to chatting with an LLM locally.",steps:[{formula:"curl -fsSL https://ollama.com/install.sh | sh",explanation:"One-line install on macOS/Linux."},{formula:"ollama run llama3.2",explanation:"Downloads the model (~2GB for 3B) and starts an interactive chat."},{formula:`curl http://localhost:11434/api/generate -d '{"model":"llama3.2","prompt":"Hello"}'`,explanation:"Or use the REST API from any programming language."}],id:"example-quick-start"}),e.jsx(t,{title:"Terminal",code:`# Check if Ollama is running
ollama --version
# ollama version is 0.5.x

# List available local models
ollama list
# NAME              ID           SIZE    MODIFIED
# llama3.2:latest   a80c4f17acd5 2.0 GB  2 hours ago

# Quick test
ollama run llama3.2 "What is the capital of France?"
# The capital of France is Paris.

# Check system info
ollama ps
# NAME          ID           SIZE    PROCESSOR  UNTIL
# llama3.2      a80c4f17acd5 3.5 GB 100% GPU   4 minutes from now`,id:"code-quick-start"}),e.jsx(t,{title:"ollama_python.py",code:`# Using the official Ollama Python library
# pip install ollama
import ollama

# Simple generation
response = ollama.generate(
    model="llama3.2",
    prompt="Explain quantum computing in one paragraph."
)
print(response["response"])

# Chat with message history
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to reverse a string."},
]
response = ollama.chat(model="llama3.2", messages=messages)
print(response["message"]["content"])

# Streaming responses
for chunk in ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Tell me a short joke."}],
    stream=True,
):
    print(chunk["message"]["content"], end="", flush=True)
print()`,id:"code-python-lib"}),e.jsx(a,{type:"note",title:"Key Features",content:"Ollama supports: automatic GPU detection and offloading, GGUF model format, Modelfiles for customization, concurrent model serving, OpenAI-compatible API, vision models (LLaVA), embedding models, and cross-platform support (macOS, Linux, Windows).",id:"note-features"}),e.jsx(n,{title:"Not for Production at Scale",content:"Ollama is designed for local development and small-scale serving. For production workloads with many concurrent users, consider vLLM, TGI, or TensorRT-LLM which offer continuous batching, tensor parallelism, and higher throughput. Ollama excels at simplicity, not raw performance.",id:"warning-scale"})]})}const fe=Object.freeze(Object.defineProperty({__proto__:null,default:h},Symbol.toStringTag,{value:"Module"}));function g(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Installing Ollama"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Ollama runs on macOS, Linux, and Windows. Installation is straightforward on all platforms, with Docker as an additional option for containerized deployments."}),e.jsx(t,{title:"Terminal — macOS",code:`# Option 1: Download from ollama.com (recommended)
# Visit https://ollama.com/download and download the macOS app

# Option 2: Homebrew
brew install ollama

# Start the Ollama service
ollama serve
# Or the macOS app runs it as a menu bar service automatically

# Verify installation
ollama --version`,id:"code-macos"}),e.jsx(t,{title:"Terminal — Linux",code:`# One-line install script (recommended)
curl -fsSL https://ollama.com/install.sh | sh

# This installs Ollama and sets up a systemd service
# The service starts automatically

# Check the service status
sudo systemctl status ollama

# View logs
journalctl -u ollama -f

# Manual start if needed
ollama serve`,id:"code-linux"}),e.jsx(t,{title:"Terminal — Docker",code:`# CPU only
docker run -d -v ollama:/root/.ollama -p 11434:11434 \\
    --name ollama ollama/ollama

# With NVIDIA GPU support
docker run -d --gpus all -v ollama:/root/.ollama -p 11434:11434 \\
    --name ollama ollama/ollama

# With AMD GPU support (ROCm)
docker run -d --device /dev/kfd --device /dev/dri \\
    -v ollama:/root/.ollama -p 11434:11434 \\
    --name ollama ollama/ollama:rocm

# Run a model inside the container
docker exec -it ollama ollama run llama3.2

# Docker Compose example
cat > docker-compose.yml << 'EOF'
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
volumes:
  ollama_data:
EOF

docker compose up -d`,id:"code-docker"}),e.jsx(a,{type:"tip",title:"Windows Installation",content:"Download the installer from ollama.com/download. Ollama runs as a system tray application on Windows. It supports NVIDIA GPUs natively and AMD GPUs via ROCm. WSL2 is not required -- Ollama runs as a native Windows application.",id:"note-windows"}),e.jsx(t,{title:"Terminal — Post-Installation Verification",code:`# Verify Ollama is running
curl http://localhost:11434/
# Ollama is running

# Check available endpoints
curl http://localhost:11434/api/tags
# {"models": []}  (empty until you pull a model)

# Pull a small model to test
ollama pull llama3.2:1b

# Run a quick test
ollama run llama3.2:1b "Say hello in 3 words"

# Check GPU detection
ollama ps
# Shows which processor (CPU/GPU) is being used`,id:"code-verify"}),e.jsx(n,{title:"Firewall and Port Configuration",content:"By default, Ollama only listens on localhost (127.0.0.1:11434). To allow remote access, set OLLAMA_HOST=0.0.0.0 in the environment. Be careful exposing Ollama to the network -- it has no built-in authentication.",id:"warning-firewall"}),e.jsx(a,{type:"note",title:"Storage Location",content:"Models are stored in ~/.ollama/models on macOS/Linux and C:\\\\Users\\\\<user>\\\\.ollama\\\\models on Windows. A typical 7B model requires 4-8GB of disk space depending on quantization. Set OLLAMA_MODELS to change the storage directory.",id:"note-storage"})]})}const ye=Object.freeze(Object.defineProperty({__proto__:null,default:g},Symbol.toStringTag,{value:"Module"}));function f(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Pulling & Running Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Ollama uses a Docker-like pull mechanism to download models. Each model has a name and optional tag specifying the size and quantization variant. Once pulled, models can be run interactively or served via API."}),e.jsx(t,{title:"Terminal",code:`# Pull a model (downloads if not already present)
ollama pull llama3.2
# pulling manifest
# pulling dde5aa3fc5ff... 100% 2.0 GB
# pulling 966de95ca8a6... 100% 1.4 KB
# verifying sha256 digest
# writing manifest
# success

# Pull a specific size/quantization variant
ollama pull llama3.2:1b        # 1B parameter version
ollama pull llama3.2:3b        # 3B parameter version
ollama pull llama3.1:8b-q4_0   # 8B with Q4_0 quantization

# Pull other popular models
ollama pull mistral             # Mistral 7B
ollama pull phi3:mini           # Microsoft Phi-3 Mini
ollama pull gemma2:9b           # Google Gemma 2 9B
ollama pull qwen2.5:7b          # Alibaba Qwen 2.5 7B
ollama pull deepseek-r1:8b      # DeepSeek R1 distilled 8B`,id:"code-pull"}),e.jsx(t,{title:"Terminal",code:`# Run interactively (pulls automatically if not present)
ollama run llama3.2
# >>> Send a message (/? for help)
# Type your prompt and press Enter
# Use /bye to exit

# Run with a one-shot prompt (non-interactive)
ollama run llama3.2 "Explain Docker in one sentence"
# Docker is a platform for building and running applications
# in isolated containers.

# Pipe input from stdin
echo "Translate to French: Hello world" | ollama run llama3.2
# Bonjour le monde

# Use with files
cat code.py | ollama run llama3.2 "Review this Python code:"

# List downloaded models
ollama list
# NAME              ID           SIZE    MODIFIED
# llama3.2:latest   a80c4f17acd5 2.0 GB  5 min ago
# mistral:latest    f974a74358d6 4.1 GB  1 hour ago
# phi3:mini         4f2222927938 2.2 GB  2 hours ago`,id:"code-run"}),e.jsx(s,{title:"Understanding Model Tags",problem:"What does 'llama3.1:8b-instruct-q4_K_M' mean?",steps:[{formula:"llama3.1 = model family",explanation:"Meta LLaMA 3.1 model family."},{formula:"8b = 8 billion parameters",explanation:"The parameter count variant."},{formula:"instruct = instruction-tuned",explanation:"Fine-tuned to follow instructions (vs base/raw model)."},{formula:"q4_K_M = 4-bit quantization (K-quant, medium)",explanation:"Specific quantization format balancing quality and size."}],id:"example-tags"}),e.jsx(t,{title:"manage_models.py",code:`import ollama

# List all local models
models = ollama.list()
for model in models["models"]:
    size_gb = model["size"] / (1024**3)
    print(f"{model['name']:<30} {size_gb:.1f} GB")

# Get model details
info = ollama.show("llama3.2")
print(f"Format: {info.get('details', {}).get('format', 'N/A')}")
print(f"Family: {info.get('details', {}).get('family', 'N/A')}")
print(f"Parameters: {info.get('details', {}).get('parameter_size', 'N/A')}")
print(f"Quantization: {info.get('details', {}).get('quantization_level', 'N/A')}")

# Delete a model to free space
# ollama.delete("mistral:latest")`,id:"code-manage"}),e.jsx(a,{type:"tip",title:"Resumable Downloads",content:"If a download is interrupted, running 'ollama pull' again resumes from where it left off. Ollama uses content-addressable storage, so layers shared between models are only downloaded once.",id:"note-resume"}),e.jsx(n,{title:"Disk Space Requirements",content:"Models can be large. A 7B model at Q4_K_M is about 4GB, while a 70B model is 40GB+. Check available disk space before pulling large models. Use 'ollama rm <model>' to delete models you no longer need.",id:"warning-disk"})]})}const xe=Object.freeze(Object.defineProperty({__proto__:null,default:f},Symbol.toStringTag,{value:"Module"}));function y(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Ollama Model Library & Tags"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The Ollama model library at ollama.com/library hosts hundreds of pre-quantized models ready to run. Understanding the library structure helps you pick the right model for your hardware and use case."}),e.jsx(o,{title:"Ollama Model Library",definition:"A curated registry of GGUF-quantized models hosted by Ollama. Each model entry includes multiple tags for different sizes (1B, 3B, 7B, 70B) and quantization levels (q4_0, q4_K_M, q8_0, fp16). The default tag is typically the best quality-to-size ratio.",id:"def-library"}),e.jsx(t,{title:"Terminal",code:`# Browse the library from CLI
ollama list   # shows local models

# Popular model families and their sizes:
#
# General Purpose:
#   llama3.2:1b     (1.3GB)  - Small, fast, good for simple tasks
#   llama3.2:3b     (2.0GB)  - Good balance for edge devices
#   llama3.1:8b     (4.7GB)  - Strong general-purpose model
#   llama3.1:70b    (40GB)   - Near GPT-4 quality
#
# Code:
#   codellama:7b    (3.8GB)  - Code generation
#   deepseek-coder-v2:16b (8.9GB) - Strong code model
#   qwen2.5-coder:7b (4.7GB) - Alibaba code model
#
# Small & Fast:
#   phi3:mini       (2.2GB)  - Microsoft, punches above weight
#   gemma2:2b       (1.6GB)  - Google, very capable for size
#   qwen2.5:0.5b    (0.4GB)  - Tiny but functional
#
# Reasoning:
#   deepseek-r1:8b  (4.9GB)  - Chain-of-thought reasoning
#   qwq:32b         (20GB)   - Strong reasoning
#
# Vision:
#   llava:7b        (4.7GB)  - Image understanding
#   llama3.2-vision:11b (7.9GB) - LLaMA vision

# See all available tags for a model
ollama show llama3.2 --list`,id:"code-library"}),e.jsx(s,{title:"Choosing the Right Model",problem:"You have a laptop with 16GB RAM and integrated GPU. Which model should you run?",steps:[{formula:"Available RAM for model: ~10GB (OS uses 4-6GB)",explanation:"Leave headroom for the operating system and other applications."},{formula:"Best options: llama3.2:3b (2GB) or llama3.1:8b-q4_0 (4.7GB)",explanation:"Both fit comfortably. The 8B model is significantly smarter."},{formula:"For coding: qwen2.5-coder:7b (4.7GB)",explanation:"Specialized code models outperform general models on code tasks."},{formula:"Avoid: any 70B model (40GB+)",explanation:"Will not fit in memory and will be extremely slow with CPU-only inference."}],id:"example-choosing"}),e.jsx(t,{title:"compare_models.py",code:`import ollama
import time

models_to_test = ["llama3.2:1b", "llama3.2:3b", "phi3:mini"]
prompt = "Write a Python function to find the nth Fibonacci number."

for model_name in models_to_test:
    try:
        start = time.time()
        response = ollama.generate(model=model_name, prompt=prompt)
        elapsed = time.time() - start

        tokens = response.get("eval_count", 0)
        speed = tokens / elapsed if elapsed > 0 else 0

        print(f"\\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Time: {elapsed:.1f}s | Tokens: {tokens} | Speed: {speed:.0f} tok/s")
        print(f"Response preview: {response['response'][:200]}...")
    except Exception as e:
        print(f"{model_name}: {e} (not downloaded?)")`,id:"code-compare"}),e.jsx(a,{type:"tip",title:"Model Naming Convention",content:"The pattern is: family:size-variant-quantization. If you just specify 'ollama pull llama3.2', you get the default tag which is usually the instruction-tuned version with q4_K_M quantization -- the best balance of quality and size.",id:"note-naming"}),e.jsx(n,{title:"Model Licenses Vary",content:"Not all models on Ollama's library are fully open. LLaMA models have Meta's community license, Gemma has Google's terms, and some models restrict commercial use. Always check the model's license before deploying in production.",id:"warning-licenses"})]})}const be=Object.freeze(Object.defineProperty({__proto__:null,default:y},Symbol.toStringTag,{value:"Module"}));function x(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"The Modelfile"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"A Modelfile is Ollama's equivalent of a Dockerfile. It defines a custom model configuration including the base model, system prompt, parameters, and chat template. Modelfiles let you create purpose-built assistants without any fine-tuning."}),e.jsx(o,{title:"Modelfile",definition:"A text file with instructions for creating a custom Ollama model. It specifies the base model (FROM), system prompt (SYSTEM), generation parameters (PARAMETER), chat template (TEMPLATE), and optional adapter weights (ADAPTER).",id:"def-modelfile"}),e.jsx(t,{title:"Terminal",code:`# Create a basic Modelfile
cat > Modelfile << 'EOF'
# Base model
FROM llama3.2

# System prompt
SYSTEM """You are a senior Python developer. You write clean, well-documented
code following PEP 8 conventions. You always include type hints and docstrings.
When asked to write code, provide complete, runnable examples."""

# Generation parameters
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 1024
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
EOF

# Build the custom model
ollama create python-dev -f Modelfile
# transferring model data
# creating model layer
# writing manifest
# success

# Run it
ollama run python-dev "Write a function to merge two sorted lists"`,id:"code-basic-modelfile"}),e.jsx(s,{title:"Modelfile Instructions Reference",problem:"What instructions are available in a Modelfile?",steps:[{formula:"FROM <model> — base model (required)",explanation:"Specifies the parent model: FROM llama3.2, FROM ./model.gguf, etc."},{formula:"SYSTEM <text> — system prompt",explanation:"Sets the system message that defines the model personality."},{formula:"PARAMETER <key> <value> — generation settings",explanation:"temperature, top_p, top_k, num_predict, stop, repeat_penalty, etc."},{formula:"TEMPLATE <template> — chat format template",explanation:"Go template defining how messages are formatted for the model."},{formula:"ADAPTER <path> — LoRA adapter",explanation:"Path to a GGUF LoRA adapter file to apply on top of the base model."},{formula:"LICENSE <text> — license information",explanation:"Embeds license text in the model metadata."}],id:"example-instructions"}),e.jsx(t,{title:"Terminal",code:`# Advanced Modelfile with custom template
cat > Modelfile-analyst << 'MODELFILE'
FROM llama3.1:8b

SYSTEM """You are a data analyst. You communicate findings clearly using
bullet points and tables. Always cite your reasoning and acknowledge
uncertainty. Format numbers with appropriate precision."""

PARAMETER temperature 0.2
PARAMETER top_p 0.85
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 8192
PARAMETER num_predict 2048

# Custom template (Go template syntax)
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
MODELFILE

ollama create data-analyst -f Modelfile-analyst
ollama run data-analyst "Analyze the trend: Q1=$1.2M, Q2=$1.5M, Q3=$1.1M, Q4=$1.8M"`,id:"code-advanced-modelfile"}),e.jsx(t,{title:"create_models.py",code:`import ollama

# Create a model programmatically
modelfile_content = '''FROM llama3.2

SYSTEM """You are a friendly tutor who explains concepts step by step.
Use analogies and examples. Ask follow-up questions to check understanding."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
'''

# Create the model via Python API
ollama.create(model="tutor", modelfile=modelfile_content)

# Test it
response = ollama.chat(
    model="tutor",
    messages=[{"role": "user", "content": "Explain recursion to a beginner"}]
)
print(response["message"]["content"])`,id:"code-python-create"}),e.jsx(a,{type:"tip",title:"Iterate Quickly",content:"Creating a model from a Modelfile is instant (no training involved). Change the system prompt, rebuild with 'ollama create', and test immediately. This makes Modelfiles ideal for rapid prototyping of different assistant personalities.",id:"note-iterate"}),e.jsx(n,{title:"Template Compatibility",content:"Custom TEMPLATE instructions must match the base model's expected chat format. Using the wrong template causes garbled output. When in doubt, omit TEMPLATE and let Ollama use the model's default. Only customize it when you need to change how messages are structured.",id:"warning-template"})]})}const _e=Object.freeze(Object.defineProperty({__proto__:null,default:x},Symbol.toStringTag,{value:"Module"}));function b(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Creating Custom Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Beyond basic Modelfiles, Ollama lets you create specialized models for specific domains, chain models together, and manage a library of purpose-built assistants. This section covers practical patterns for building and organizing custom models."}),e.jsx(s,{title:"Common Custom Model Patterns",problem:"What kinds of custom models are most useful?",steps:[{formula:"Domain expert: system prompt + low temperature",explanation:"Medical, legal, financial assistants with domain-specific instructions."},{formula:"Output formatter: strict output format instructions",explanation:"JSON-only responses, markdown tables, specific report formats."},{formula:"Persona: character + style instructions",explanation:"Customer service agents, tutors, creative writing helpers."},{formula:"Chain-of-thought: reasoning instructions",explanation:"Step-by-step problem solving with explicit reasoning."}],id:"example-patterns"}),e.jsx(t,{title:"Terminal",code:`# Pattern 1: JSON-only responder
cat > Modelfile-json << 'EOF'
FROM llama3.2

SYSTEM """You are a JSON API. You ONLY respond with valid JSON.
Never include explanations, markdown, or any text outside the JSON object.
Always use double quotes for keys and string values.
If you cannot fulfill the request, respond with {"error": "description"}."""

PARAMETER temperature 0.1
PARAMETER top_p 0.8
EOF

ollama create json-api -f Modelfile-json
ollama run json-api "Extract entities from: Apple CEO Tim Cook visited Tokyo"
# {"entities": [{"name": "Apple", "type": "organization"},
#  {"name": "Tim Cook", "type": "person"}, {"name": "Tokyo", "type": "location"}]}

# Pattern 2: Code reviewer
cat > Modelfile-reviewer << 'EOF'
FROM llama3.1:8b

SYSTEM """You are a strict code reviewer. For each code snippet:
1. List bugs and issues (CRITICAL, WARNING, INFO)
2. Suggest specific fixes with corrected code
3. Rate overall quality (1-10)
Keep reviews concise and actionable."""

PARAMETER temperature 0.2
PARAMETER num_predict 2048
EOF

ollama create code-reviewer -f Modelfile-reviewer`,id:"code-patterns"}),e.jsx(t,{title:"model_manager.py",code:`import ollama
import json

# Define a library of custom models
MODEL_CONFIGS = {
    "sql-expert": {
        "base": "llama3.1:8b",
        "system": (
            "You are a SQL expert. Convert natural language to SQL queries. "
            "Always specify the assumed table schema before the query. "
            "Use standard SQL syntax compatible with PostgreSQL."
        ),
        "params": {"temperature": 0.1, "top_p": 0.85},
    },
    "eli5": {
        "base": "llama3.2",
        "system": (
            "Explain everything as if talking to a 5-year-old. Use simple words, "
            "fun analogies, and short sentences. Never use jargon or technical terms."
        ),
        "params": {"temperature": 0.8, "top_p": 0.95},
    },
    "summarizer": {
        "base": "llama3.2",
        "system": (
            "You are a summarization engine. Provide concise bullet-point summaries. "
            "Maximum 5 bullet points. Each bullet should be one sentence. "
            "Capture the key facts and insights only."
        ),
        "params": {"temperature": 0.2, "top_p": 0.8},
    },
}

def create_all_models():
    for name, config in MODEL_CONFIGS.items():
        params = "\\n".join(
            f"PARAMETER {k} {v}" for k, v in config["params"].items()
        )
        modelfile = f'''FROM {config["base"]}

SYSTEM """{config["system"]}"""

{params}
'''
        print(f"Creating {name}...")
        ollama.create(model=name, modelfile=modelfile)
        print(f"  Done!")

def test_model(name, prompt):
    response = ollama.generate(model=name, prompt=prompt)
    print(f"[{name}] {response['response'][:300]}")

create_all_models()
test_model("sql-expert", "Find all users who signed up last month")
test_model("eli5", "What is quantum entanglement?")
test_model("summarizer", "Summarize the concept of machine learning")`,id:"code-manager"}),e.jsx(a,{type:"tip",title:"Version Your Modelfiles",content:"Store Modelfiles in version control alongside your application code. This ensures reproducibility -- anyone can recreate your exact model configuration. Include a README documenting each model's purpose and expected behavior.",id:"note-version-control"}),e.jsx(n,{title:"System Prompts Are Not Security Boundaries",content:"Users can override or extract system prompts through prompt injection. Do not rely on system prompts for access control or to hide sensitive information. Treat them as behavioral guidelines, not security mechanisms.",id:"warning-security"}),e.jsx(a,{type:"note",title:"Model Inheritance",content:"Custom models built with FROM reference the base model's weights. Deleting the base model will break custom models that depend on it. Use 'ollama show <model>' to see the full dependency chain.",id:"note-inheritance"})]})}const ve=Object.freeze(Object.defineProperty({__proto__:null,default:b},Symbol.toStringTag,{value:"Module"}));function _(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Importing Custom GGUF Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"While Ollama's library covers popular models, you may want to run models from Hugging Face or your own quantized models. Ollama supports importing any GGUF file, the standard format for quantized models used by llama.cpp."}),e.jsx(o,{title:"GGUF Format",definition:"GGUF (GPT-Generated Unified Format) is a binary file format for storing quantized LLM weights and metadata. It is the successor to GGML and is the native format used by llama.cpp and Ollama. A single .gguf file contains the full model: weights, tokenizer, and configuration.",id:"def-gguf"}),e.jsx(t,{title:"Terminal",code:`# Step 1: Download a GGUF from Hugging Face
# Many users upload GGUF quantizations (e.g., TheBloke, bartowski)
pip install huggingface-hub

# Download a specific GGUF file
huggingface-cli download \\
    bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \\
    Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \\
    --local-dir ./models

# Step 2: Create a Modelfile pointing to the GGUF
cat > Modelfile << 'EOF'
FROM ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Set appropriate chat template for LLaMA 3
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
EOF

# Step 3: Import into Ollama
ollama create my-llama3 -f Modelfile
# transferring model data
# using existing layer sha256:abc123...
# creating model layer
# writing manifest
# success

# Step 4: Run it
ollama run my-llama3 "Hello, who are you?"`,id:"code-import-gguf"}),e.jsx(t,{title:"convert_to_gguf.py",code:`# Convert a Hugging Face model to GGUF using llama.cpp
# First, clone llama.cpp and install requirements
# git clone https://github.com/ggerganov/llama.cpp
# pip install -r llama.cpp/requirements.txt

import subprocess
import os

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "./converted"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Download model from Hugging Face
from huggingface_hub import snapshot_download
model_path = snapshot_download(MODEL_ID, local_dir=f"{OUTPUT_DIR}/hf-model")

# Step 2: Convert to GGUF (fp16)
subprocess.run([
    "python", "llama.cpp/convert_hf_to_gguf.py",
    f"{OUTPUT_DIR}/hf-model",
    "--outfile", f"{OUTPUT_DIR}/model-f16.gguf",
    "--outtype", "f16",
], check=True)

# Step 3: Quantize to Q4_K_M
subprocess.run([
    "./llama.cpp/build/bin/llama-quantize",
    f"{OUTPUT_DIR}/model-f16.gguf",
    f"{OUTPUT_DIR}/model-q4_K_M.gguf",
    "Q4_K_M",
], check=True)

print(f"Quantized model: {OUTPUT_DIR}/model-q4_K_M.gguf")
size_mb = os.path.getsize(f"{OUTPUT_DIR}/model-q4_K_M.gguf") / (1024**2)
print(f"Size: {size_mb:.0f} MB")`,id:"code-convert"}),e.jsx(s,{title:"Common GGUF Sources",problem:"Where to find pre-quantized GGUF files?",steps:[{formula:'Hugging Face: search for "GGUF" in model names',explanation:"Users like bartowski, TheBloke, and others upload quantized versions of popular models."},{formula:"Ollama library: all models are GGUF internally",explanation:"Ollama downloads are GGUF files stored in ~/.ollama/models/blobs/."},{formula:"Self-quantize: use llama.cpp convert scripts",explanation:"Convert any Hugging Face safetensors model to GGUF with custom quantization."}],id:"example-sources"}),e.jsx(a,{type:"tip",title:"Check Metadata Before Importing",content:"Use 'llama.cpp/build/bin/llama-gguf-info model.gguf' to inspect a GGUF file's metadata: architecture, quantization type, context length, tokenizer info. This helps you set the correct template and parameters in your Modelfile.",id:"note-metadata"}),e.jsx(n,{title:"Chat Template Must Match",content:"When importing a GGUF, Ollama cannot always auto-detect the correct chat template. If your imported model gives garbled output, the most likely cause is a wrong or missing TEMPLATE in the Modelfile. Check the original model's documentation for the expected format.",id:"warning-template"})]})}const we=Object.freeze(Object.defineProperty({__proto__:null,default:_},Symbol.toStringTag,{value:"Module"}));function v(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Importing Fine-tuned LoRA Weights"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"If you have fine-tuned a model using LoRA, you can import those adapter weights into Ollama. This lets you serve your custom fine-tuned models with the same ease as any Ollama model, complete with API access and model management."}),e.jsx(o,{title:"LoRA Adapter in Ollama",definition:"Ollama supports loading LoRA (Low-Rank Adaptation) weights on top of a base model. The adapter must be in GGUF format. At inference time, the adapter weights are merged with the base model, producing the fine-tuned behavior.",id:"def-lora-ollama"}),e.jsx(t,{title:"Terminal",code:`# Step 1: Convert LoRA adapter to GGUF format
# If your adapter is from Hugging Face / PEFT format:
python llama.cpp/convert_lora_to_gguf.py \\
    --base ./base-model-hf \\
    --lora ./my-lora-adapter \\
    --outfile ./my-adapter.gguf

# Step 2: Create a Modelfile with the ADAPTER instruction
cat > Modelfile-lora << 'EOF'
# Base model (must match what the LoRA was trained on)
FROM llama3.1:8b

# Apply the LoRA adapter
ADAPTER ./my-adapter.gguf

# Optional: customize system prompt for your fine-tune
SYSTEM """You are a customer support agent for Acme Corp.
Answer questions about our products and policies helpfully."""

PARAMETER temperature 0.3
EOF

# Step 3: Build the model
ollama create acme-support -f Modelfile-lora

# Step 4: Test it
ollama run acme-support "What is your return policy?"

# Verify the model shows the adapter info
ollama show acme-support`,id:"code-import-lora"}),e.jsx(s,{title:"LoRA Import Workflow",problem:"End-to-end: train a LoRA with Unsloth, deploy with Ollama.",steps:[{formula:"Train LoRA with Unsloth/PEFT on your dataset",explanation:"Fine-tune produces adapter weights (adapter_model.safetensors)."},{formula:"Convert adapter to GGUF: convert_lora_to_gguf.py",explanation:"Transforms the safetensors LoRA into Ollama-compatible format."},{formula:"Create Modelfile with FROM base + ADAPTER path",explanation:"The base model must match the model the LoRA was trained on."},{formula:"ollama create my-model -f Modelfile",explanation:"Ollama merges the adapter and creates a servable model."}],id:"example-workflow"}),e.jsx(t,{title:"full_lora_pipeline.py",code:`# Complete pipeline: export LoRA from PEFT and import to Ollama
import subprocess
import ollama

# Assume we have a PEFT LoRA adapter at ./lora-output/
LORA_PATH = "./lora-output"
BASE_MODEL_HF = "./base-model-hf"  # HF format base model
ADAPTER_GGUF = "./adapter.gguf"

# Step 1: Convert LoRA to GGUF
print("Converting LoRA adapter to GGUF...")
subprocess.run([
    "python", "llama.cpp/convert_lora_to_gguf.py",
    "--base", BASE_MODEL_HF,
    "--lora", LORA_PATH,
    "--outfile", ADAPTER_GGUF,
], check=True)

# Step 2: Create Ollama model with adapter
modelfile = f"""FROM llama3.1:8b
ADAPTER {ADAPTER_GGUF}

SYSTEM """You are an AI assistant fine-tuned for medical Q&A.
Provide accurate, evidence-based answers. Always recommend
consulting a healthcare professional for medical decisions."""

PARAMETER temperature 0.2
PARAMETER top_p 0.9
"""

print("Creating Ollama model...")
ollama.create(model="medical-qa", modelfile=modelfile)

# Step 3: Test the fine-tuned model
response = ollama.chat(
    model="medical-qa",
    messages=[{
        "role": "user",
        "content": "What are the common symptoms of type 2 diabetes?"
    }]
)
print(response["message"]["content"])`,id:"code-pipeline"}),e.jsx(a,{type:"tip",title:"Unsloth Direct Export",content:"Unsloth can export directly to Ollama-compatible GGUF format using model.save_pretrained_gguf() and model.push_to_hub_gguf(). This skips the manual conversion step entirely -- the recommended approach if you use Unsloth for training.",id:"note-unsloth"}),e.jsx(n,{title:"Base Model Must Match",content:"The LoRA adapter is only compatible with the exact base model it was trained on. A LoRA trained on LLaMA-3.1-8B cannot be applied to LLaMA-3.2-3B or Mistral-7B. If the base model does not match, you will get errors or nonsensical outputs.",id:"warning-base-match"}),e.jsx(a,{type:"note",title:"Merged vs Adapter Serving",content:"Ollama merges the LoRA weights into the base model at load time. This means inference speed is identical to the base model -- there is no overhead from having a separate adapter. The tradeoff is that you cannot hot-swap adapters at runtime.",id:"note-merged"})]})}const ke=Object.freeze(Object.defineProperty({__proto__:null,default:v},Symbol.toStringTag,{value:"Module"}));function w(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Quantization Levels in Ollama"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Quantization reduces model size by representing weights with fewer bits. Ollama models come in various quantization levels, each offering a different tradeoff between quality, speed, and memory usage. Understanding these helps you choose the right variant."}),e.jsx(o,{title:"GGUF Quantization Types",definition:"GGUF supports multiple quantization schemes. The naming convention is Q{bits}_{type}: Q4_0 (basic 4-bit), Q4_K_M (4-bit with k-quant medium), Q5_K_M (5-bit k-quant medium), Q8_0 (8-bit), and F16 (16-bit float). K-quants use importance-based mixed precision.",id:"def-quantization"}),e.jsx(s,{title:"Quantization Comparison for LLaMA-3.1-8B",problem:"Compare size, speed, and quality across quantization levels.",steps:[{formula:"Q4_0: 4.3 GB — basic 4-bit, fastest, lowest quality",explanation:"Simple round-to-nearest quantization. Noticeable quality loss."},{formula:"Q4_K_M: 4.9 GB — recommended default",explanation:"K-quant keeps important weights at higher precision. Best quality/size ratio."},{formula:"Q5_K_M: 5.7 GB — higher quality 5-bit",explanation:"Barely noticeable quality loss. Good if you have the extra RAM."},{formula:"Q8_0: 8.5 GB — near lossless 8-bit",explanation:"Almost indistinguishable from fp16 for most tasks."},{formula:"F16: 16 GB — full fp16 precision",explanation:"Maximum quality, but requires 2-4x more memory."}],id:"example-comparison"}),e.jsx(t,{title:"Terminal",code:`# Pull different quantization levels of the same model
ollama pull llama3.1:8b-instruct-q4_0
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull llama3.1:8b-instruct-q5_K_M
ollama pull llama3.1:8b-instruct-q8_0

# Compare sizes
ollama list | grep llama3.1
# llama3.1:8b-instruct-q4_0    4.3 GB
# llama3.1:8b-instruct-q4_K_M  4.9 GB
# llama3.1:8b-instruct-q5_K_M  5.7 GB
# llama3.1:8b-instruct-q8_0    8.5 GB

# Quantize your own GGUF with llama.cpp
# Available types: Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_1,
#   Q4_K_S, Q4_K_M, Q5_0, Q5_1, Q5_K_S, Q5_K_M, Q6_K, Q8_0, F16, F32
./llama-quantize model-f16.gguf model-q4_K_M.gguf Q4_K_M`,id:"code-quant-levels"}),e.jsx(t,{title:"benchmark_quants.py",code:`import ollama
import time

MODELS = [
    "llama3.1:8b-instruct-q4_0",
    "llama3.1:8b-instruct-q4_K_M",
    "llama3.1:8b-instruct-q8_0",
]

PROMPTS = [
    "What is the derivative of x^3 + 2x?",
    "Write a Python quicksort implementation.",
    "Explain the difference between TCP and UDP.",
]

for model in MODELS:
    total_tokens = 0
    total_time = 0

    for prompt in PROMPTS:
        start = time.time()
        resp = ollama.generate(model=model, prompt=prompt)
        elapsed = time.time() - start

        total_tokens += resp.get("eval_count", 0)
        total_time += elapsed

    speed = total_tokens / total_time if total_time > 0 else 0
    print(f"{model:<40} {speed:>6.1f} tok/s  ({total_tokens} tokens)")

# Typical results (Apple M2 Pro):
# q4_0:   ~45 tok/s  (fastest but lower quality)
# q4_K_M: ~40 tok/s  (best balance)
# q8_0:   ~25 tok/s  (highest quality, slower)`,id:"code-benchmark"}),e.jsx(a,{type:"tip",title:"The Sweet Spot: Q4_K_M",content:"For most users, Q4_K_M is the best default. It uses importance-based mixed precision (keeping attention and output layers at higher precision) and produces nearly the same quality as Q5_K_M at smaller size. This is why Ollama uses Q4_K_M as the default tag for most models.",id:"note-sweet-spot"}),e.jsx(n,{title:"Quality Degrades Below Q4",content:"Q3_K and Q2_K quantizations show significant quality degradation, especially for reasoning and math tasks. Avoid going below Q4 unless your use case only involves simple text generation or you are severely memory-constrained.",id:"warning-quality"}),e.jsx(a,{type:"note",title:"Bigger Model, Lower Quant vs Smaller Model, Higher Quant",content:"A 13B model at Q4_K_M often outperforms a 7B model at Q8_0 despite similar memory usage. Larger models are more robust to quantization. When memory is fixed, prefer a bigger model with more aggressive quantization over a smaller model at higher precision.",id:"note-bigger-better"})]})}const Ae=Object.freeze(Object.defineProperty({__proto__:null,default:w},Symbol.toStringTag,{value:"Module"}));function k(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"GPU Acceleration & Layer Offloading"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Ollama automatically detects GPUs and offloads model layers for acceleration. Understanding how GPU offloading works helps you optimize performance, especially when the model does not fully fit in GPU memory."}),e.jsx(o,{title:"Layer Offloading",definition:"A transformer model consists of many layers (e.g., 32 for a 7B model). Each layer can be placed on GPU or CPU independently. Offloading $N$ of $L$ total layers to GPU means those layers run at GPU speed while the rest run on CPU. More GPU layers = faster inference.",id:"def-offloading"}),e.jsx(t,{title:"Terminal",code:`# Check GPU detection
ollama ps
# NAME        ID           SIZE    PROCESSOR     UNTIL
# llama3.2    a80c4f17acd5 3.5 GB  100% GPU      4 min from now

# Environment variables for GPU control:

# Force number of GPU layers (0 = CPU only)
OLLAMA_GPU_LAYERS=20 ollama serve

# Specify which GPU to use (for multi-GPU systems)
CUDA_VISIBLE_DEVICES=0 ollama serve    # Use GPU 0 only
CUDA_VISIBLE_DEVICES=0,1 ollama serve  # Use GPUs 0 and 1

# Set GPU memory limit (leave room for KV cache)
OLLAMA_GPU_MEMORY=6g ollama serve      # Limit to 6GB GPU RAM

# Check NVIDIA GPU status
nvidia-smi
# Shows GPU utilization, memory usage, temperature

# Monitor GPU usage during generation
watch -n 0.5 nvidia-smi`,id:"code-gpu-config"}),e.jsx(s,{title:"Partial GPU Offloading",problem:"You have a 6GB GPU and want to run a model that needs 8GB.",steps:[{formula:"Model: 32 layers, ~250MB per layer = 8GB total",explanation:"The full model does not fit in 6GB of VRAM."},{formula:"Offload 22 layers to GPU: 22 \\times 250MB = 5.5GB",explanation:"Leave ~500MB headroom for KV-cache and CUDA context."},{formula:"Remaining 10 layers run on CPU",explanation:"CPU layers are slower but the GPU layers dominate throughput."},{formula:"Result: ~60% of GPU-only speed",explanation:"Partial offloading is much faster than pure CPU inference."}],id:"example-partial"}),e.jsx(t,{title:"gpu_benchmark.py",code:`import ollama
import time
import subprocess

def get_gpu_memory():
    """Get current GPU memory usage via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        used, total = result.stdout.strip().split(", ")
        return int(used), int(total)
    except Exception:
        return None, None

def benchmark_model(model_name, prompt, n_tokens=100):
    """Benchmark a model and report GPU usage."""
    used_before, total = get_gpu_memory()

    start = time.time()
    resp = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={"num_predict": n_tokens}
    )
    elapsed = time.time() - start

    used_after, _ = get_gpu_memory()
    tokens = resp.get("eval_count", 0)
    speed = tokens / elapsed if elapsed > 0 else 0

    print(f"Model: {model_name}")
    print(f"  Speed: {speed:.1f} tok/s")
    print(f"  GPU memory: {used_before}MB -> {used_after}MB (total: {total}MB)")
    print(f"  Processor: {resp.get('model', 'unknown')}")

prompt = "Write a detailed explanation of how neural networks learn."
benchmark_model("llama3.2", prompt)
benchmark_model("llama3.1:8b", prompt)`,id:"code-benchmark"}),e.jsx(a,{type:"tip",title:"Apple Silicon",content:"On Apple Silicon Macs (M1/M2/M3/M4), Ollama uses Metal for GPU acceleration with unified memory. The GPU and CPU share the same RAM, so there is no data transfer overhead. A Mac with 32GB unified memory can run models that would need a 24GB+ discrete GPU.",id:"note-apple-silicon"}),e.jsx(n,{title:"VRAM Is Not Just for Weights",content:"The model weights are only part of GPU memory usage. KV-cache grows with context length (hundreds of MB for long conversations), and CUDA/Metal context takes 300-500MB. Leave at least 1-2GB of headroom beyond the model size.",id:"warning-vram"}),e.jsx(a,{type:"note",title:"Flash Attention in Ollama",content:"Ollama benefits from flash attention implementations in llama.cpp, which reduce GPU memory usage and speed up attention computation. This is enabled automatically when your GPU supports it (CUDA compute capability 7.0+, all Apple Silicon).",id:"note-flash-attention"})]})}const Te=Object.freeze(Object.defineProperty({__proto__:null,default:k},Symbol.toStringTag,{value:"Module"}));function A(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Multi-Model Serving"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Ollama can load and serve multiple models concurrently. Understanding how Ollama manages model loading, unloading, and memory sharing is key to running multiple models efficiently on limited hardware."}),e.jsx(o,{title:"Concurrent Model Serving",definition:"Ollama keeps recently used models in memory and can serve requests to different models simultaneously. Models are loaded on first request and unloaded after an idle timeout (default 5 minutes). The OLLAMA_MAX_LOADED_MODELS environment variable controls how many models stay loaded.",id:"def-multi-model"}),e.jsx(t,{title:"Terminal",code:`# See currently loaded models
ollama ps
# NAME              ID           SIZE    PROCESSOR    UNTIL
# llama3.2:latest   a80c4f17acd5 3.5 GB  100% GPU     4 min
# mistral:latest    f974a74358d6 5.1 GB  100% GPU     2 min

# Configure concurrent models
export OLLAMA_MAX_LOADED_MODELS=3      # Keep up to 3 models in memory
export OLLAMA_KEEP_ALIVE="10m"         # Models stay loaded for 10 minutes
export OLLAMA_NUM_PARALLEL=4           # Handle 4 concurrent requests per model

# Start Ollama with these settings
ollama serve

# Or set keep_alive per request via API
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Hello",
  "keep_alive": "30m"
}'

# Unload a model immediately
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "keep_alive": 0
}'`,id:"code-multi-model"}),e.jsx(t,{title:"model_router.py",code:`import ollama

# Route requests to different models based on task type
ROUTING_TABLE = {
    "code": "qwen2.5-coder:7b",
    "general": "llama3.2",
    "creative": "llama3.1:8b",
    "math": "deepseek-r1:8b",
    "fast": "llama3.2:1b",
}

def classify_task(prompt):
    """Simple keyword-based task classification."""
    prompt_lower = prompt.lower()
    if any(w in prompt_lower for w in ["code", "function", "program", "debug"]):
        return "code"
    if any(w in prompt_lower for w in ["poem", "story", "creative", "imagine"]):
        return "creative"
    if any(w in prompt_lower for w in ["calculate", "math", "equation", "proof"]):
        return "math"
    return "general"

def smart_generate(prompt, task_type=None):
    """Route to the best model for the task."""
    if task_type is None:
        task_type = classify_task(prompt)
    model = ROUTING_TABLE.get(task_type, ROUTING_TABLE["general"])
    print(f"Routing to {model} (task: {task_type})")

    response = ollama.generate(model=model, prompt=prompt)
    return response["response"]

# Test with different prompts
print(smart_generate("Write a Python function to parse CSV files"))
print(smart_generate("Write a haiku about autumn"))
print(smart_generate("What is the integral of sin(x)?"))`,id:"code-router"}),e.jsx(s,{title:"Memory Planning for Multi-Model",problem:"You have 24GB GPU memory. How many models can you serve?",steps:[{formula:"CUDA context: ~500MB overhead",explanation:"Base GPU memory used by the CUDA runtime."},{formula:"3B model (Q4_K_M): ~2GB each",explanation:"Small models for fast responses."},{formula:"8B model (Q4_K_M): ~5GB each",explanation:"Medium models for quality responses."},{formula:"Example: 1x 8B + 2x 3B + KV-cache = 5+4+2 ≈ 11GB",explanation:"Leaves 13GB for KV-cache and concurrent requests."}],id:"example-memory-planning"}),e.jsx(a,{type:"tip",title:"Preload Models",content:"Send a request with an empty prompt to preload a model without generating output. This avoids cold-start latency when the first real request arrives. Useful for models you know will be needed soon.",id:"note-preload"}),e.jsx(n,{title:"OOM with Multiple Models",content:"If total model memory exceeds available GPU RAM, Ollama will fall back to CPU for some models or refuse to load. Monitor GPU memory with 'nvidia-smi' and set OLLAMA_MAX_LOADED_MODELS conservatively. It is better to have models swap in/out than to crash from OOM.",id:"warning-oom"})]})}const Oe=Object.freeze(Object.defineProperty({__proto__:null,default:A},Symbol.toStringTag,{value:"Module"}));function T(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Ollama REST API Deep Dive"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Ollama exposes a REST API on port 11434 that lets you integrate LLM inference into any application. The API supports generation, chat, embeddings, and model management, plus an OpenAI-compatible endpoint for drop-in compatibility."}),e.jsx(o,{title:"Ollama API Endpoints",definition:"The Ollama REST API provides: /api/generate (text completion), /api/chat (multi-turn conversation), /api/embeddings (vector embeddings), /api/tags (list models), /api/show (model info), /api/pull and /api/push (model management), and /v1/* (OpenAI-compatible).",id:"def-api"}),e.jsx(t,{title:"Terminal",code:`# Generate endpoint -- single-turn completion
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the sky blue?",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "top_p": 0.9,
    "num_predict": 200
  }
}'
# Returns: {"model":"llama3.2","response":"The sky appears blue...","done":true,...}

# Chat endpoint -- multi-turn conversation
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user", "content": "What are its main features?"}
  ],
  "stream": false
}'

# Embeddings endpoint
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "Ollama makes running LLMs easy"
}'

# List local models
curl http://localhost:11434/api/tags`,id:"code-curl"}),e.jsx(t,{title:"ollama_api_client.py",code:`import requests
import json

BASE_URL = "http://localhost:11434"

def generate(prompt, model="llama3.2", **options):
    """Call the generate endpoint."""
    resp = requests.post(f"{BASE_URL}/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    })
    return resp.json()

def chat(messages, model="llama3.2", **options):
    """Call the chat endpoint."""
    resp = requests.post(f"{BASE_URL}/api/chat", json={
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    })
    return resp.json()

def get_embeddings(text, model="nomic-embed-text"):
    """Get text embeddings."""
    resp = requests.post(f"{BASE_URL}/api/embeddings", json={
        "model": model,
        "prompt": text,
    })
    return resp.json()["embedding"]

# Usage examples
result = generate("Explain REST APIs in one sentence", temperature=0.3)
print(f"Generate: {result['response']}")

result = chat([
    {"role": "user", "content": "What is 2+2?"},
], temperature=0)
print(f"Chat: {result['message']['content']}")

embedding = get_embeddings("Hello world")
print(f"Embedding dim: {len(embedding)}")  # 768 for nomic-embed

# Using the OpenAI-compatible endpoint
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
resp = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(f"OpenAI compat: {resp.choices[0].message.content}")`,id:"code-python-api"}),e.jsx(s,{title:"API Response Fields",problem:"What information does the generate endpoint return?",steps:[{formula:"response: the generated text",explanation:"The main output string."},{formula:"eval_count: number of tokens generated",explanation:"Useful for tracking token usage."},{formula:"eval_duration: time spent generating (nanoseconds)",explanation:"Compute tokens/second as eval_count / (eval_duration / 1e9)."},{formula:"prompt_eval_count: tokens in the prompt",explanation:"Number of tokens the prompt was tokenized into."},{formula:"total_duration: total request time",explanation:"Includes model loading, prompt processing, and generation."}],id:"example-response"}),e.jsx(a,{type:"tip",title:"OpenAI Compatibility",content:"The /v1/chat/completions endpoint is compatible with the OpenAI Python SDK. Set base_url='http://localhost:11434/v1' and api_key='ollama'. This means you can swap between Ollama and OpenAI by changing just two lines of code.",id:"note-openai-compat"}),e.jsx(n,{title:"No Built-in Authentication",content:"The Ollama API has no authentication mechanism. Anyone who can reach port 11434 can use your models. If exposing to a network, put a reverse proxy (nginx, caddy) with authentication in front of Ollama.",id:"warning-auth"})]})}const Ie=Object.freeze(Object.defineProperty({__proto__:null,default:T},Symbol.toStringTag,{value:"Module"}));function O(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Streaming Responses & Context Windows"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Streaming lets users see tokens as they are generated rather than waiting for the full response. This dramatically improves perceived latency. Understanding context windows is equally important for managing conversation length."}),e.jsx(o,{title:"Streaming",definition:"In streaming mode, the server sends each generated token as a separate JSON object as soon as it is produced. The client receives a series of newline-delimited JSON (NDJSON) chunks, each containing a partial response. The final chunk has done: true.",id:"def-streaming"}),e.jsx(t,{title:"Terminal",code:`# Streaming with curl (default behavior)
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Count from 1 to 10"
}'
# {"model":"llama3.2","response":"1","done":false}
# {"model":"llama3.2","response":",","done":false}
# {"model":"llama3.2","response":" 2","done":false}
# ... one chunk per token ...
# {"model":"llama3.2","response":"","done":true,"eval_count":25,...}

# Non-streaming (wait for full response)
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Count from 1 to 10",
  "stream": false
}'
# Returns one JSON object with the complete response`,id:"code-streaming-curl"}),e.jsx(t,{title:"streaming_client.py",code:`import requests
import json
import sys

def stream_generate(prompt, model="llama3.2"):
    """Stream tokens from Ollama and print them in real-time."""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt},
        stream=True,  # Enable HTTP streaming
    )

    full_response = ""
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get("response", "")
            full_response += token
            print(token, end="", flush=True)

            if chunk.get("done"):
                stats = {
                    "tokens": chunk.get("eval_count", 0),
                    "speed": chunk.get("eval_count", 0) /
                             (chunk.get("eval_duration", 1) / 1e9),
                }
                print(f"\\n\\n--- {stats['tokens']} tokens at "
                      f"{stats['speed']:.1f} tok/s ---")
    return full_response

# Using the official Python library (streaming)
import ollama

def stream_chat(messages, model="llama3.2"):
    """Stream a chat response."""
    for chunk in ollama.chat(model=model, messages=messages, stream=True):
        print(chunk["message"]["content"], end="", flush=True)
    print()

stream_generate("Write a short poem about coding")
print("\\n" + "="*50)
stream_chat([{"role": "user", "content": "Tell me a joke"}])`,id:"code-streaming-python"}),e.jsx(o,{title:"Context Window",definition:"The context window is the maximum number of tokens a model can process in a single request (prompt + response combined). Common sizes: 4096 (GPT-2), 8192 (LLaMA 3), 128K (LLaMA 3.1). Exceeding the window causes the model to lose early context.",id:"def-context-window"}),e.jsx(t,{title:"Terminal",code:`# Set context window size in Ollama
# Via API options:
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Hello",
  "options": {
    "num_ctx": 8192
  }
}'

# Via Modelfile:
# PARAMETER num_ctx 16384

# Default is typically 2048 to save memory
# Increase for long conversations or documents
# Maximum depends on the model (check model card)

# Check a model's default context size
ollama show llama3.2 --modelfile | grep num_ctx`,id:"code-context"}),e.jsx(a,{type:"tip",title:"Context Window vs Memory",content:"Doubling the context window roughly doubles the KV-cache memory usage. A 3B model with num_ctx=2048 might use 3.5GB, but with num_ctx=32768 it could use 6GB+. Only increase context size when you actually need it.",id:"note-context-memory"}),e.jsx(n,{title:"Conversation History Accumulates",content:"In chat mode, every previous message is re-sent with each request. A long conversation can silently exceed the context window, causing the model to drop early messages. Monitor token counts and implement conversation summarization or sliding window truncation for long chats.",id:"warning-history"})]})}const Se=Object.freeze(Object.defineProperty({__proto__:null,default:O},Symbol.toStringTag,{value:"Module"}));function I(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Ollama with Vision Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Ollama supports multimodal vision-language models that can understand images alongside text. You can send images via the API and get descriptions, analysis, or answers to questions about visual content, all running locally."}),e.jsx(o,{title:"Vision-Language Models in Ollama",definition:"Ollama supports models like LLaVA, BakLLaVA, and LLaMA 3.2 Vision that process both text and images. Images are sent as base64-encoded data in the 'images' field of the API request. The model's vision encoder processes the image while the language model generates the response.",id:"def-vision"}),e.jsx(t,{title:"Terminal",code:`# Pull a vision model
ollama pull llava:7b                    # LLaVA 1.6 (Mistral-based)
ollama pull llama3.2-vision:11b         # LLaMA 3.2 Vision 11B
ollama pull moondream                   # Small but capable (1.8B)

# Use vision from CLI (drag and drop or path)
ollama run llava:7b "Describe this image: ./photo.jpg"

# Via API with base64 image
# (base64 encoding shown in Python examples below)
curl http://localhost:11434/api/generate -d '{
  "model": "llava:7b",
  "prompt": "What do you see in this image?",
  "images": ["<base64-encoded-image>"]
}'`,id:"code-vision-setup"}),e.jsx(t,{title:"vision_api.py",code:`import ollama
import base64
from pathlib import Path

def encode_image(image_path):
    """Read and base64-encode an image file."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Method 1: Using the Ollama Python library
response = ollama.chat(
    model="llava:7b",
    messages=[{
        "role": "user",
        "content": "Describe this image in detail.",
        "images": ["./screenshot.png"],  # File path works directly
    }]
)
print(response["message"]["content"])

# Method 2: Using requests with base64
import requests

image_b64 = encode_image("./chart.png")
resp = requests.post("http://localhost:11434/api/chat", json={
    "model": "llava:7b",
    "messages": [{
        "role": "user",
        "content": "What does this chart show? Summarize the key trends.",
        "images": [image_b64],
    }],
    "stream": False,
})
print(resp.json()["message"]["content"])

# Method 3: Multiple images
response = ollama.chat(
    model="llava:7b",
    messages=[{
        "role": "user",
        "content": "Compare these two images. What are the differences?",
        "images": ["./image1.png", "./image2.png"],
    }]
)
print(response["message"]["content"])`,id:"code-vision-python"}),e.jsx(s,{title:"Vision Model Use Cases",problem:"What can you do with local vision models?",steps:[{formula:"Image description and captioning",explanation:"Generate alt text, describe photos, catalog visual content."},{formula:"Document/screenshot understanding",explanation:"Extract text, understand layouts, read charts and diagrams."},{formula:"Visual question answering",explanation:"Ask specific questions about image content."},{formula:"Code from screenshots",explanation:"Convert UI mockups or screenshots into code."}],id:"example-use-cases"}),e.jsx(a,{type:"tip",title:"Model Selection for Vision",content:"LLaMA 3.2 Vision 11B offers the best quality but needs ~8GB. LLaVA 7B is a good balance. Moondream (1.8B) is remarkably capable for its size, running on just 2GB -- ideal for edge devices or when you need fast image understanding.",id:"note-model-selection"}),e.jsx(n,{title:"Image Size and Performance",content:"Large images are resized internally but still increase processing time. Pre-resize images to reasonable dimensions (e.g., 768x768) before sending. Very high-resolution images do not improve accuracy and waste compute.",id:"warning-image-size"})]})}const Pe=Object.freeze(Object.defineProperty({__proto__:null,default:I},Symbol.toStringTag,{value:"Module"}));function S(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Ollama for Embeddings"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Beyond text generation, Ollama can serve embedding models that convert text into dense vector representations. These embeddings power similarity search, RAG systems, clustering, and classification, all running locally with no API costs."}),e.jsx(o,{title:"Embedding Models in Ollama",definition:"Embedding models map text to fixed-dimensional vectors where semantically similar texts produce similar vectors. Ollama serves embedding models via the /api/embeddings endpoint. Popular choices include nomic-embed-text (768 dims) and mxbai-embed-large (1024 dims).",id:"def-embeddings"}),e.jsx(t,{title:"Terminal",code:`# Pull embedding models
ollama pull nomic-embed-text      # 274MB, 768 dimensions
ollama pull mxbai-embed-large     # 670MB, 1024 dimensions
ollama pull all-minilm            # 45MB, 384 dimensions (tiny but fast)

# Generate an embedding via API
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "Ollama makes embeddings easy"
}'
# Returns: {"embedding": [0.0123, -0.0456, ...]}  (768 floats)`,id:"code-setup"}),e.jsx(t,{title:"embeddings_demo.py",code:`import ollama
import numpy as np

def get_embedding(text, model="nomic-embed-text"):
    """Get embedding vector for a text string."""
    resp = ollama.embeddings(model=model, prompt=text)
    return np.array(resp["embedding"])

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Demonstrate semantic similarity
texts = [
    "The cat sat on the mat",
    "A feline rested on the rug",      # semantically similar
    "Quantum physics is fascinating",   # unrelated
    "The dog lay on the carpet",        # somewhat similar
]

embeddings = [get_embedding(t) for t in texts]

print("Pairwise cosine similarities:")
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"  {sim:.3f}: '{texts[i][:40]}' <-> '{texts[j][:40]}'")

# Simple semantic search
query = "A pet resting on furniture"
query_emb = get_embedding(query)

scores = [(cosine_similarity(query_emb, emb), text)
          for emb, text in zip(embeddings, texts)]
scores.sort(reverse=True)

print(f"\\nSearch: '{query}'")
for score, text in scores:
    print(f"  {score:.3f}: {text}")`,id:"code-demo"}),e.jsx(t,{title:"local_vector_store.py",code:`import ollama
import numpy as np
import json

class LocalVectorStore:
    """Minimal vector store using Ollama embeddings."""

    def __init__(self, model="nomic-embed-text"):
        self.model = model
        self.documents = []
        self.embeddings = []

    def add(self, text, metadata=None):
        resp = ollama.embeddings(model=self.model, prompt=text)
        self.documents.append({"text": text, "metadata": metadata or {}})
        self.embeddings.append(np.array(resp["embedding"]))

    def search(self, query, top_k=3):
        query_emb = np.array(
            ollama.embeddings(model=self.model, prompt=query)["embedding"]
        )
        scores = [
            np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            for emb in self.embeddings
        ]
        indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], scores[i]) for i in indices]

# Build a knowledge base
store = LocalVectorStore()
store.add("Python is a high-level programming language", {"topic": "python"})
store.add("JavaScript runs in web browsers", {"topic": "js"})
store.add("Docker containers package applications", {"topic": "devops"})
store.add("Neural networks are inspired by the brain", {"topic": "ml"})
store.add("SQL is used for database queries", {"topic": "databases"})

results = store.search("How do I build a web app?", top_k=2)
for doc, score in results:
    print(f"  [{score:.3f}] {doc['text']}")`,id:"code-vector-store"}),e.jsx(a,{type:"tip",title:"Embedding Speed",content:"Embedding models are much faster than generative models because they process input in a single forward pass with no autoregressive loop. nomic-embed-text can embed hundreds of documents per second on a modern GPU. Batch your embedding requests for maximum throughput.",id:"note-speed"}),e.jsx(n,{title:"Choose the Right Model for Your Use Case",content:"nomic-embed-text is optimized for retrieval (RAG). all-minilm is tiny but lower quality. For production RAG systems, test your embedding model on your specific domain -- a model that excels on general benchmarks may underperform on specialized text like medical or legal documents.",id:"warning-model-choice"})]})}const Le=Object.freeze(Object.defineProperty({__proto__:null,default:S},Symbol.toStringTag,{value:"Module"}));function P(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Running Ollama on Remote Servers"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"While Ollama excels at local development, you can also run it on a remote server with a powerful GPU and access it from your laptop. This section covers SSH tunneling, reverse proxy setup, and secure remote access patterns."}),e.jsx(t,{title:"Terminal — SSH Tunnel (simplest)",code:`# On your local machine, create an SSH tunnel to the remote Ollama
ssh -L 11434:localhost:11434 user@gpu-server.example.com

# Now Ollama on the remote server is accessible at localhost:11434
# All existing tools and scripts work without changes
curl http://localhost:11434/api/tags

# For a persistent tunnel (runs in background)
ssh -fNL 11434:localhost:11434 user@gpu-server.example.com

# Kill the tunnel when done
kill $(lsof -ti:11434 | head -1)`,id:"code-ssh-tunnel"}),e.jsx(t,{title:"Terminal — Remote Server Setup",code:`# On the remote GPU server:

# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Configure to listen on all interfaces
# Edit the systemd service
sudo systemctl edit ollama.service

# Add these lines:
# [Service]
# Environment="OLLAMA_HOST=0.0.0.0"

# Or set the environment variable directly
export OLLAMA_HOST=0.0.0.0
ollama serve

# 3. Pull models
ollama pull llama3.1:8b

# 4. Set up Nginx reverse proxy with basic auth
sudo apt install nginx apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd ollama_user

cat | sudo tee /etc/nginx/sites-available/ollama << 'NGINX'
server {
    listen 443 ssl;
    server_name ollama.example.com;

    ssl_certificate /etc/letsencrypt/live/ollama.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ollama.example.com/privkey.pem;

    auth_basic "Ollama API";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://127.0.0.1:11434;
        proxy_set_header Host $host;
        proxy_buffering off;           # Important for streaming
        proxy_read_timeout 300s;       # Long timeout for generation
    }
}
NGINX

sudo ln -s /etc/nginx/sites-available/ollama /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx`,id:"code-server-setup"}),e.jsx(t,{title:"remote_client.py",code:`import ollama
import os

# Option 1: Point Ollama client to remote server
# Set environment variable before importing ollama
os.environ["OLLAMA_HOST"] = "http://gpu-server:11434"

response = ollama.generate(model="llama3.1:8b", prompt="Hello from remote!")
print(response["response"])

# Option 2: Use requests with authentication (Nginx setup)
import requests

REMOTE_URL = "https://ollama.example.com"
AUTH = ("ollama_user", "your_password")

resp = requests.post(
    f"{REMOTE_URL}/api/generate",
    json={"model": "llama3.1:8b", "prompt": "Hello!", "stream": False},
    auth=AUTH,
)
print(resp.json()["response"])

# Option 3: Use OpenAI SDK with remote Ollama
from openai import OpenAI

client = OpenAI(
    base_url="https://ollama.example.com/v1",
    api_key="ollama",  # Not used but required by SDK
)

response = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "What GPU are you running on?"}],
)
print(response.choices[0].message.content)`,id:"code-remote-client"}),e.jsx(s,{title:"Remote Access Methods Compared",problem:"Which remote access method should you choose?",steps:[{formula:"SSH tunnel: simplest, most secure, no config needed",explanation:"Best for personal use. Requires SSH access to the server."},{formula:"Reverse proxy + HTTPS: production-grade, shareable",explanation:"Best for team access. Requires domain name and SSL certificate."},{formula:"Tailscale/WireGuard VPN: zero-config networking",explanation:"Best for accessing from multiple devices without exposing to internet."}],id:"example-methods"}),e.jsx(n,{title:"Never Expose Ollama Directly to the Internet",content:"Ollama has no authentication, rate limiting, or abuse prevention. Exposing port 11434 directly allows anyone to use your GPU, download models, and potentially access sensitive data in prompts. Always use a reverse proxy with authentication or a VPN.",id:"warning-security"}),e.jsx(a,{type:"tip",title:"Tailscale for Easy Remote Access",content:"Tailscale creates a private mesh VPN with zero configuration. Install it on both your laptop and GPU server, then access Ollama at the server's Tailscale IP. No port forwarding, no firewall changes, no certificates needed.",id:"note-tailscale"})]})}const je=Object.freeze(Object.defineProperty({__proto__:null,default:P},Symbol.toStringTag,{value:"Module"}));function L(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Ollama vs llama.cpp vs LM Studio"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Ollama is not the only way to run LLMs locally. llama.cpp provides the raw inference engine, while LM Studio offers a full desktop GUI. Understanding the tradeoffs helps you choose the right tool for your workflow."}),e.jsx(o,{title:"The Local LLM Stack",definition:"llama.cpp is the C++ inference engine that powers both Ollama and LM Studio. Ollama wraps it with model management and an API server. LM Studio wraps it with a graphical interface and model discovery. All three use the same GGUF format and produce identical outputs for the same model and parameters.",id:"def-stack"}),e.jsx(s,{title:"Feature Comparison",problem:"When should you use each tool?",steps:[{formula:"Ollama: API-first, CLI, Docker, automation",explanation:"Best for developers building applications. Simple CLI, REST API, scriptable."},{formula:"llama.cpp: maximum control, custom builds, research",explanation:"Best when you need specific compile flags, custom kernels, or cutting-edge features."},{formula:"LM Studio: GUI, visual model browser, no coding",explanation:"Best for non-developers or quick model exploration. Download and chat in clicks."}],id:"example-when-to-use"}),e.jsx(t,{title:"Terminal",code:`# Ollama: simplest workflow
ollama run llama3.2 "Hello world"

# llama.cpp: most control
# Build from source:
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build

# Run with explicit parameters:
./build/bin/llama-cli \\
    -m ./models/llama-3.2-3b-q4_K_M.gguf \\
    -p "Hello world" \\
    --n-gpu-layers 99 \\
    --ctx-size 4096 \\
    --temp 0.7 \\
    --top-p 0.9 \\
    --repeat-penalty 1.1

# llama.cpp server (OpenAI-compatible)
./build/bin/llama-server \\
    -m ./models/llama-3.2-3b-q4_K_M.gguf \\
    --host 0.0.0.0 --port 8080 \\
    --n-gpu-layers 99 \\
    --ctx-size 4096 \\
    --parallel 4

# LM Studio: download from https://lmstudio.ai
# No CLI - it is a desktop application with GUI`,id:"code-comparison"}),e.jsx(t,{title:"benchmark_comparison.py",code:`import requests
import time

# Benchmark Ollama vs llama.cpp server on the same model
ENDPOINTS = {
    "Ollama": "http://localhost:11434/v1/chat/completions",
    "llama.cpp": "http://localhost:8080/v1/chat/completions",
    # LM Studio also exposes an OpenAI-compatible endpoint
    "LM Studio": "http://localhost:1234/v1/chat/completions",
}

prompt = "Explain the difference between TCP and UDP in 3 sentences."

for name, url in ENDPOINTS.items():
    try:
        start = time.time()
        resp = requests.post(url, json={
            "model": "llama3.2",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.7,
        }, timeout=30)
        elapsed = time.time() - start

        if resp.status_code == 200:
            data = resp.json()
            tokens = data.get("usage", {}).get("completion_tokens", "?")
            print(f"{name:<12} {elapsed:.2f}s  {tokens} tokens")
        else:
            print(f"{name:<12} Error: {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"{name:<12} Not running")

# Note: Performance should be very similar since they all use llama.cpp
# Differences come from default settings, batching, and overhead`,id:"code-benchmark"}),e.jsx(a,{type:"note",title:"Detailed Comparison",content:"Ollama: auto GPU detection, model management, Docker support, no GUI. llama.cpp: compile-time optimizations, grammar constraints, LoRA hot-loading, batch API. LM Studio: model discovery UI, chat interface, parameter sliders, export conversations. All support the same GGUF models and OpenAI-compatible APIs.",id:"note-detailed"}),e.jsx(a,{type:"tip",title:"They Are Not Mutually Exclusive",content:"Many developers use LM Studio for interactive exploration, Ollama for local development APIs, and llama.cpp server for production edge deployments. You can even run them side-by-side on different ports since they all use the same model files.",id:"note-coexist"}),e.jsx(n,{title:"Ollama Abstracts Away Important Details",content:"Ollama's simplicity means less control. You cannot easily set specific compile flags, use grammar-constrained generation (natively), or fine-tune batch processing parameters. If you hit Ollama's limits, dropping down to llama.cpp directly gives you full control.",id:"warning-abstraction"})]})}const Me=Object.freeze(Object.defineProperty({__proto__:null,default:L},Symbol.toStringTag,{value:"Module"}));function j(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"What is Open WebUI"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Open WebUI is a self-hosted, feature-rich chat interface for LLMs. It provides a ChatGPT-like experience that works with local models (via Ollama) and cloud APIs, giving you full control over your data and infrastructure."}),e.jsx(o,{title:"Open WebUI",definition:"Open WebUI (formerly Ollama WebUI) is an open-source web application that provides a polished chat interface for interacting with LLMs. It supports multiple backends (Ollama, OpenAI, vLLM), multi-user access, RAG, web search, tool use, and extensive customization.",id:"def-open-webui"}),e.jsx(s,{title:"Key Features",problem:"What makes Open WebUI stand out among LLM interfaces?",steps:[{formula:"Multi-backend: Ollama + OpenAI + any compatible API",explanation:"Switch between local and cloud models in the same conversation."},{formula:"RAG: upload documents and chat with them",explanation:"Built-in document processing, embedding, and retrieval."},{formula:"Multi-user: authentication, roles, sharing",explanation:"Deploy for a team with access control and conversation sharing."},{formula:"Extensible: tools, pipelines, plugins",explanation:"Add web search, code execution, image generation, and custom functions."}],id:"example-features"}),e.jsx(t,{title:"Terminal",code:`# Quickest way to try Open WebUI
# (Assumes Ollama is already running on localhost:11434)

# Option 1: Docker (recommended)
docker run -d -p 3000:8080 \\
    --add-host=host.docker.internal:host-gateway \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    --restart always \\
    ghcr.io/open-webui/open-webui:main

# Open http://localhost:3000 in your browser
# Create an admin account on first visit

# Option 2: Bundled with Ollama (all-in-one)
docker run -d -p 3000:8080 \\
    --gpus all \\
    -v ollama:/root/.ollama \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:ollama

# Option 3: pip install (no Docker)
pip install open-webui
open-webui serve --port 3000`,id:"code-quickstart"}),e.jsx(t,{title:"verify_setup.py",code:`import requests

# Check if Open WebUI is running
try:
    resp = requests.get("http://localhost:3000/api/version")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Open WebUI version: {data.get('version', 'unknown')}")
        print("Status: Running!")
    else:
        print(f"Unexpected status: {resp.status_code}")
except requests.ConnectionError:
    print("Open WebUI is not running on port 3000")

# Check connected backends
try:
    resp = requests.get("http://localhost:3000/api/models")
    if resp.status_code == 200:
        models = resp.json()
        print(f"Available models: {len(models.get('data', []))}")
        for m in models.get("data", [])[:5]:
            print(f"  - {m.get('id', 'unknown')}")
except Exception as e:
    print(f"Could not fetch models: {e}")`,id:"code-verify"}),e.jsx(a,{type:"note",title:"Data Privacy",content:"All data stays on your server. Conversations, uploaded documents, and user information are stored locally in the open-webui Docker volume. Nothing is sent to external services unless you explicitly configure a cloud API backend.",id:"note-privacy"}),e.jsx(n,{title:"Resource Requirements",content:"Open WebUI itself is lightweight (uses ~200MB RAM), but it runs alongside your LLM backend. A system running both Ollama with a 7B model and Open WebUI needs at least 8GB RAM total. The web interface works best with a modern browser.",id:"warning-resources"})]})}const Ee=Object.freeze(Object.defineProperty({__proto__:null,default:j},Symbol.toStringTag,{value:"Module"}));function M(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Installation (Docker, pip, Source)"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Open WebUI can be installed via Docker (recommended), pip, or from source. Docker provides the most reliable setup with automatic updates, while pip is convenient for development environments."}),e.jsx(t,{title:"Terminal — Docker Installation",code:`# Standard Docker install (connects to existing Ollama)
docker run -d -p 3000:8080 \\
    --add-host=host.docker.internal:host-gateway \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    --restart always \\
    ghcr.io/open-webui/open-webui:main

# With GPU passthrough (for local embedding models)
docker run -d -p 3000:8080 \\
    --gpus all \\
    --add-host=host.docker.internal:host-gateway \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    --restart always \\
    ghcr.io/open-webui/open-webui:main

# Docker Compose with Ollama included
cat > docker-compose.yml << 'EOF'
services:
  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    volumes:
      - open_webui_data:/app/backend/data
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - ollama

volumes:
  ollama_data:
  open_webui_data:
EOF

docker compose up -d`,id:"code-docker"}),e.jsx(t,{title:"Terminal — pip Installation",code:`# Install via pip (Python 3.11+ recommended)
pip install open-webui

# Start the server
open-webui serve --port 3000

# Or with custom settings
OLLAMA_BASE_URL=http://localhost:11434 \\
DATA_DIR=./open-webui-data \\
open-webui serve --port 3000 --host 0.0.0.0`,id:"code-pip"}),e.jsx(t,{title:"Terminal — Update and Maintenance",code:`# Update Docker installation
docker pull ghcr.io/open-webui/open-webui:main
docker stop open-webui
docker rm open-webui
# Re-run the docker run command (data persists in the volume)

# Update pip installation
pip install --upgrade open-webui

# Backup data (Docker)
docker cp open-webui:/app/backend/data ./open-webui-backup

# View logs
docker logs -f open-webui

# Reset admin password
docker exec open-webui open-webui reset-admin-password`,id:"code-update"}),e.jsx(s,{title:"Post-Installation Checklist",problem:"What to do after installing Open WebUI?",steps:[{formula:"Navigate to http://localhost:3000",explanation:"Open the web interface in your browser."},{formula:"Create admin account",explanation:"The first user to sign up becomes the admin."},{formula:"Verify Ollama connection",explanation:"Go to Settings > Connections and confirm Ollama is connected."},{formula:"Pull a model from the UI or via Ollama CLI",explanation:"You need at least one model to start chatting."}],id:"example-checklist"}),e.jsx(a,{type:"tip",title:"Docker Volume Persistence",content:"The -v open-webui:/app/backend/data flag ensures your data (conversations, settings, uploaded files) persists across container restarts and updates. Never use --rm or forget this volume mount, or you will lose all data on restart.",id:"note-persistence"}),e.jsx(n,{title:"First User Is Admin",content:"The very first account created on a fresh Open WebUI installation automatically becomes the administrator. Set up the admin account immediately after deployment to prevent unauthorized users from claiming admin access.",id:"warning-admin"})]})}const Ue=Object.freeze(Object.defineProperty({__proto__:null,default:M},Symbol.toStringTag,{value:"Module"}));function E(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Connecting to Ollama Backend"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The most common setup pairs Open WebUI with a local Ollama instance. The connection is straightforward but requires understanding how Docker networking works to avoid common pitfalls."}),e.jsx(o,{title:"Ollama Backend Connection",definition:"Open WebUI communicates with Ollama via its REST API. The OLLAMA_BASE_URL environment variable tells Open WebUI where to find Ollama. From inside a Docker container, 'localhost' refers to the container itself, not the host machine.",id:"def-connection"}),e.jsx(t,{title:"Terminal",code:`# Scenario 1: Open WebUI in Docker, Ollama on host
# Use --add-host to make host.docker.internal resolve to the host
docker run -d -p 3000:8080 \\
    --add-host=host.docker.internal:host-gateway \\
    -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Scenario 2: Both in Docker Compose (recommended)
cat > docker-compose.yml << 'EOF'
services:
  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    volumes:
      - open_webui_data:/app/backend/data
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - ollama

volumes:
  ollama_data:
  open_webui_data:
EOF

docker compose up -d

# Scenario 3: pip install (both on same host)
# No special networking needed -- localhost works
OLLAMA_BASE_URL=http://localhost:11434 open-webui serve`,id:"code-connection"}),e.jsx(t,{title:"troubleshoot_connection.py",code:`import requests

OLLAMA_URLS = [
    "http://localhost:11434",
    "http://host.docker.internal:11434",
    "http://ollama:11434",
    "http://127.0.0.1:11434",
]

print("Testing Ollama connectivity:")
for url in OLLAMA_URLS:
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            print(f"  {url} -> OK ({resp.text.strip()})")
        else:
            print(f"  {url} -> HTTP {resp.status_code}")
    except requests.ConnectionError:
        print(f"  {url} -> Connection refused")
    except requests.Timeout:
        print(f"  {url} -> Timeout")

# Also verify models are available
working_url = "http://localhost:11434"
try:
    resp = requests.get(f"{working_url}/api/tags")
    models = resp.json().get("models", [])
    print(f"\\nModels available: {len(models)}")
    for m in models:
        print(f"  - {m['name']} ({m['size']/(1024**3):.1f} GB)")
except Exception as e:
    print(f"\\nCannot list models: {e}")`,id:"code-troubleshoot"}),e.jsx(s,{title:"Common Connection Issues",problem:"Open WebUI cannot see Ollama models. How to debug?",steps:[{formula:"Check Ollama is running: curl http://localhost:11434/",explanation:'Should return "Ollama is running". If not, start Ollama.'},{formula:"Docker networking: use host.docker.internal, not localhost",explanation:"Inside a container, localhost is the container itself."},{formula:"Firewall: Ollama binds to 127.0.0.1 by default",explanation:"Set OLLAMA_HOST=0.0.0.0 if Ollama needs to accept remote connections."},{formula:"Check Open WebUI Settings > Connections",explanation:"The URL can be changed from the admin UI after login."}],id:"example-troubleshoot"}),e.jsx(a,{type:"tip",title:"Pull Models from the UI",content:"Open WebUI can pull Ollama models directly from the interface. Go to the model selector dropdown and type a model name to pull it. This is convenient but can be disabled by admins who want to control which models are available.",id:"note-pull-from-ui"}),e.jsx(n,{title:"Ollama Must Be Running First",content:"If Ollama is not running when Open WebUI starts, the connection will fail and no models will appear. In Docker Compose, use depends_on to ensure Ollama starts first. For manual setups, start Ollama before Open WebUI.",id:"warning-order"})]})}const Ge=Object.freeze(Object.defineProperty({__proto__:null,default:E},Symbol.toStringTag,{value:"Module"}));function U(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Connecting to OpenAI-Compatible APIs"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Open WebUI is not limited to Ollama. It can connect to any OpenAI-compatible API, including OpenAI itself, Anthropic (via proxy), vLLM, TGI, LiteLLM, and other serving frameworks. This lets you use the same interface for local and cloud models."}),e.jsx(o,{title:"OpenAI-Compatible API",definition:"An API that implements the same endpoint structure as OpenAI's Chat Completions API (/v1/chat/completions, /v1/models). Most LLM serving frameworks implement this standard, enabling tool interoperability.",id:"def-openai-api"}),e.jsx(t,{title:"Terminal",code:`# Configure OpenAI API in Open WebUI via environment variables
docker run -d -p 3000:8080 \\
    --add-host=host.docker.internal:host-gateway \\
    -e OPENAI_API_BASE_URLS="https://api.openai.com/v1;http://localhost:8000/v1" \\
    -e OPENAI_API_KEYS="sk-your-openai-key;none" \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Or configure via the UI:
# 1. Go to Admin Panel > Settings > Connections
# 2. Add a new OpenAI API connection
# 3. Enter the base URL and API key

# Common OpenAI-compatible endpoints:
# OpenAI:     https://api.openai.com/v1
# vLLM:       http://localhost:8000/v1
# TGI:        http://localhost:8080/v1
# LiteLLM:    http://localhost:4000/v1
# Together:   https://api.together.xyz/v1
# Groq:       https://api.groq.com/openai/v1
# Ollama:     http://localhost:11434/v1`,id:"code-config"}),e.jsx(t,{title:"test_connections.py",code:`import requests

# Test various OpenAI-compatible endpoints
ENDPOINTS = {
    "OpenAI": {
        "url": "https://api.openai.com/v1/models",
        "headers": {"Authorization": "Bearer sk-your-key"},
    },
    "vLLM (local)": {
        "url": "http://localhost:8000/v1/models",
        "headers": {},
    },
    "Ollama (OpenAI compat)": {
        "url": "http://localhost:11434/v1/models",
        "headers": {},
    },
    "Groq": {
        "url": "https://api.groq.com/openai/v1/models",
        "headers": {"Authorization": "Bearer gsk_your-key"},
    },
}

for name, config in ENDPOINTS.items():
    try:
        resp = requests.get(
            config["url"],
            headers=config["headers"],
            timeout=5,
        )
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            model_names = [m["id"] for m in models[:3]]
            print(f"{name}: {len(models)} models - {model_names}")
        else:
            print(f"{name}: HTTP {resp.status_code}")
    except Exception as e:
        print(f"{name}: {type(e).__name__}")`,id:"code-test"}),e.jsx(s,{title:"Setting Up Cloud Providers",problem:"How to connect Open WebUI to various cloud providers?",steps:[{formula:"OpenAI: URL=https://api.openai.com/v1, key=sk-...",explanation:"Direct access to GPT-4, GPT-4o, etc."},{formula:"Groq: URL=https://api.groq.com/openai/v1, key=gsk_...",explanation:"Ultra-fast inference for LLaMA, Mixtral on Groq hardware."},{formula:"Together AI: URL=https://api.together.xyz/v1, key=...",explanation:"Access to 100+ open models with pay-per-token pricing."},{formula:"LiteLLM proxy: URL=http://localhost:4000/v1",explanation:"Unified proxy that routes to any provider, including Anthropic."}],id:"example-providers"}),e.jsx(a,{type:"tip",title:"Using Anthropic Models",content:"Anthropic's API is not directly OpenAI-compatible, but LiteLLM can translate between them. Run LiteLLM as a proxy and point Open WebUI to it. This gives you access to Claude models through the same interface.",id:"note-anthropic"}),e.jsx(n,{title:"API Key Security",content:"When adding cloud API keys to Open WebUI, they are stored in the application database. In multi-user setups, admin-configured keys are shared across all users. Be careful about who has admin access and consider using per-user API keys instead.",id:"warning-keys"})]})}const Re=Object.freeze(Object.defineProperty({__proto__:null,default:U},Symbol.toStringTag,{value:"Module"}));function G(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Multi-Backend Configuration"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Open WebUI can connect to multiple LLM backends simultaneously, letting users choose between local Ollama models, cloud APIs, and specialized serving frameworks from a single unified interface. This is ideal for teams that need flexibility."}),e.jsx(o,{title:"Multi-Backend Setup",definition:"Open WebUI supports connecting to one Ollama instance and multiple OpenAI-compatible API endpoints at the same time. All models from all backends appear in a single model selector, and users can switch between them mid-conversation.",id:"def-multi-backend"}),e.jsx(t,{title:"Terminal",code:`# Docker Compose with multiple backends
cat > docker-compose.yml << 'EOF'
services:
  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    volumes:
      - ./litellm-config.yaml:/app/config.yaml
    command: ["--config", "/app/config.yaml", "--port", "4000"]
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=\${ANTHROPIC_API_KEY}

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    volumes:
      - open_webui_data:/app/backend/data
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - OPENAI_API_BASE_URLS=http://litellm:4000/v1
      - OPENAI_API_KEYS=sk-litellm
    depends_on:
      - ollama
      - litellm

volumes:
  ollama_data:
  open_webui_data:
EOF

# LiteLLM config for multiple providers
cat > litellm-config.yaml << 'EOF'
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: llama-groq
    litellm_params:
      model: groq/llama-3.1-70b-versatile
      api_key: os.environ/GROQ_API_KEY
EOF

docker compose up -d`,id:"code-compose"}),e.jsx(t,{title:"test_multi_backend.py",code:`from openai import OpenAI

# Test that Open WebUI can proxy to all backends
owui = OpenAI(
    base_url="http://localhost:3000/api",  # Open WebUI API
    api_key="your-open-webui-api-key",     # Generated in settings
)

# List all available models across backends
models = owui.models.list()
print("Available models:")
for m in models.data:
    print(f"  {m.id}")

# Test different backends through Open WebUI
backends = {
    "llama3.2": "Ollama (local)",
    "gpt-4o": "OpenAI (cloud)",
    "claude-sonnet": "Anthropic via LiteLLM",
}

for model, backend in backends.items():
    try:
        resp = owui.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=10,
        )
        print(f"{backend}: {resp.choices[0].message.content}")
    except Exception as e:
        print(f"{backend}: {e}")`,id:"code-test"}),e.jsx(s,{title:"Use Cases for Multi-Backend",problem:"Why run multiple backends?",steps:[{formula:"Cost optimization: local for drafts, cloud for final",explanation:"Use cheap local models for iteration, expensive cloud models for quality."},{formula:"Capability routing: code model + general model",explanation:"Specialized models for specific tasks, general models for everything else."},{formula:"Fallback: if local GPU is busy, fall back to cloud",explanation:"Ensures availability when local resources are constrained."}],id:"example-use-cases"}),e.jsx(a,{type:"tip",title:"Model Naming",content:"When models from different backends have the same name, Open WebUI disambiguates them. You can also rename models in the admin settings to make them more user-friendly, e.g., renaming 'meta-llama/Meta-Llama-3.1-8B-Instruct' to 'LLaMA 3.1 8B'.",id:"note-naming"}),e.jsx(n,{title:"Latency Varies by Backend",content:"Users may not realize that switching from a local Ollama model to a cloud API changes latency characteristics significantly. Consider labeling models with their backend type so users can make informed choices about speed vs quality.",id:"warning-latency"})]})}const Ce=Object.freeze(Object.defineProperty({__proto__:null,default:G},Symbol.toStringTag,{value:"Module"}));function R(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"User Management & Authentication"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Open WebUI supports multi-user environments with role-based access control. Administrators can manage users, control model access, and configure authentication methods including OAuth and LDAP integration."}),e.jsx(o,{title:"User Roles",definition:"Open WebUI has three roles: Admin (full access, system configuration, user management), User (standard chat access, can create conversations), and Pending (registered but not yet approved). Admins can also restrict which models each role can access.",id:"def-roles"}),e.jsx(t,{title:"Terminal",code:`# Environment variables for authentication
docker run -d -p 3000:8080 \\
    -e WEBUI_AUTH=true \\
    -e ENABLE_SIGNUP=true \\
    -e DEFAULT_USER_ROLE=pending \\
    -e ENABLE_OAUTH_SIGNUP=true \\
    -e OAUTH_PROVIDER_NAME=google \\
    -e OAUTH_CLIENT_ID=your-client-id \\
    -e OAUTH_CLIENT_SECRET=your-client-secret \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Disable signup (admin creates all accounts)
# -e ENABLE_SIGNUP=false

# Auto-approve new users
# -e DEFAULT_USER_ROLE=user

# LDAP authentication
# -e ENABLE_LDAP=true
# -e LDAP_SERVER_HOST=ldap.example.com
# -e LDAP_SERVER_PORT=389
# -e LDAP_SEARCH_BASE=dc=example,dc=com`,id:"code-auth-config"}),e.jsx(t,{title:"manage_users.py",code:`import requests

BASE_URL = "http://localhost:3000/api/v1"
ADMIN_TOKEN = "your-admin-api-token"  # From Settings > Account > API Keys
HEADERS = {"Authorization": f"Bearer {ADMIN_TOKEN}"}

# List all users
resp = requests.get(f"{BASE_URL}/users", headers=HEADERS)
users = resp.json()
for user in users:
    print(f"  {user['name']} ({user['email']}) - role: {user['role']}")

# Approve a pending user
def approve_user(user_id):
    resp = requests.post(
        f"{BASE_URL}/users/{user_id}/role",
        headers=HEADERS,
        json={"role": "user"},
    )
    return resp.json()

# Update user role
def set_admin(user_id):
    resp = requests.post(
        f"{BASE_URL}/users/{user_id}/role",
        headers=HEADERS,
        json={"role": "admin"},
    )
    return resp.json()

# Delete a user
def delete_user(user_id):
    resp = requests.delete(
        f"{BASE_URL}/users/{user_id}",
        headers=HEADERS,
    )
    return resp.status_code == 200

# Example: approve all pending users
pending = [u for u in users if u["role"] == "pending"]
for user in pending:
    approve_user(user["id"])
    print(f"Approved: {user['name']}")`,id:"code-manage-users"}),e.jsx(s,{title:"Authentication Options",problem:"What authentication methods does Open WebUI support?",steps:[{formula:"Built-in: email/password with local accounts",explanation:"Default method. No external dependencies."},{formula:"OAuth 2.0: Google, GitHub, Microsoft, custom OIDC",explanation:"Single sign-on with existing identity providers."},{formula:"LDAP: Active Directory / OpenLDAP",explanation:"Enterprise directory integration."},{formula:"Trusted header: reverse proxy authentication",explanation:"Let nginx/Authelia handle auth and pass user info via headers."}],id:"example-auth-methods"}),e.jsx(a,{type:"tip",title:"Per-User Model Access",content:"Admins can restrict which models each user or role can access. This is useful for limiting expensive cloud model usage to specific users while giving everyone access to local models.",id:"note-model-access"}),e.jsx(n,{title:"Secure Your Admin Account",content:"The admin account has full access to all conversations, user data, and system settings. Use a strong password, enable two-factor authentication if available, and limit the number of admin accounts. Regularly audit admin access.",id:"warning-admin-security"})]})}const qe=Object.freeze(Object.defineProperty({__proto__:null,default:R},Symbol.toStringTag,{value:"Module"}));function C(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Chat Interface Features"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Open WebUI provides a rich chat experience that goes beyond basic text exchange. Features like conversation branching, message editing, regeneration, and file attachments make it a powerful tool for working with LLMs."}),e.jsx(s,{title:"Core Chat Features",problem:"What can you do in an Open WebUI conversation?",steps:[{formula:"Branching: create alternative responses at any point",explanation:"Click the branch icon on any message to explore different response paths."},{formula:"Regenerate: retry the last response with same or different model",explanation:"If the response is poor, regenerate without retyping the prompt."},{formula:"Edit & resubmit: modify any previous message",explanation:"Change a prompt mid-conversation and get a new response chain."},{formula:"File attachments: upload documents, images, code",explanation:"Attach files directly to messages for the model to process."},{formula:"Code highlighting: syntax-highlighted code blocks",explanation:"Code in responses is automatically highlighted with copy buttons."}],id:"example-features"}),e.jsx(t,{title:"Terminal",code:`# Open WebUI chat features are primarily used through the web UI
# But many can be accessed via the API as well

# Create a conversation via API
curl -X POST http://localhost:3000/api/v1/chats/new \\
    -H "Authorization: Bearer YOUR_TOKEN" \\
    -H "Content-Type: application/json" \\
    -d '{
        "chat": {
            "title": "API Test Chat",
            "messages": []
        }
    }'

# Send a message in a conversation
curl -X POST http://localhost:3000/api/chat/completions \\
    -H "Authorization: Bearer YOUR_TOKEN" \\
    -H "Content-Type: application/json" \\
    -d '{
        "model": "llama3.2",
        "messages": [
            {"role": "user", "content": "What is the meaning of life?"}
        ],
        "stream": true
    }'`,id:"code-api-chat"}),e.jsx(t,{title:"chat_export.py",code:`import requests
import json

BASE_URL = "http://localhost:3000/api/v1"
TOKEN = "your-api-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# List all conversations
resp = requests.get(f"{BASE_URL}/chats", headers=HEADERS)
chats = resp.json()

print(f"Total conversations: {len(chats)}")
for chat in chats[:5]:
    title = chat.get("title", "Untitled")
    msg_count = len(chat.get("chat", {}).get("messages", []))
    print(f"  [{chat['id'][:8]}] {title} ({msg_count} messages)")

# Export a specific conversation
if chats:
    chat_id = chats[0]["id"]
    resp = requests.get(f"{BASE_URL}/chats/{chat_id}", headers=HEADERS)
    chat_data = resp.json()

    # Save as JSON
    with open(f"chat_{chat_id[:8]}.json", "w") as f:
        json.dump(chat_data, f, indent=2)
    print(f"Exported chat to chat_{chat_id[:8]}.json")

    # Format as readable text
    for msg in chat_data.get("chat", {}).get("messages", []):
        role = msg["role"].upper()
        content = msg["content"][:100]
        print(f"  [{role}] {content}...")`,id:"code-export"}),e.jsx(a,{type:"tip",title:"Keyboard Shortcuts",content:"Open WebUI supports keyboard shortcuts: Enter to send (Shift+Enter for newline), Ctrl+Shift+C to copy last response, Ctrl+/ to toggle sidebar. These shortcuts make the chat experience much faster for power users.",id:"note-shortcuts"}),e.jsx(a,{type:"note",title:"Markdown Rendering",content:"Open WebUI renders full markdown in responses including tables, code blocks with syntax highlighting, LaTeX math, lists, and headings. This makes it excellent for technical conversations where structured formatting matters.",id:"note-markdown"}),e.jsx(n,{title:"Conversation History Size",content:"Each message in the conversation history is sent to the LLM with every new request. Very long conversations can exceed the model's context window, causing early messages to be truncated. Start new conversations for new topics to avoid this.",id:"warning-history"})]})}const Ne=Object.freeze(Object.defineProperty({__proto__:null,default:C},Symbol.toStringTag,{value:"Module"}));function q(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Model Selection & Parameter Tuning"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Open WebUI lets you select models and tune generation parameters directly from the chat interface. Understanding these parameters helps you get better results for different tasks without writing any code."}),e.jsx(o,{title:"Generation Parameters in Open WebUI",definition:"Open WebUI exposes key generation parameters through the chat settings panel: temperature (creativity), top-p (nucleus sampling threshold), top-k (vocabulary restriction), max tokens (response length), and frequency/presence penalties (repetition control).",id:"def-params"}),e.jsx(s,{title:"Parameter Presets by Task",problem:"What parameters work best for different tasks?",steps:[{formula:"Coding: T=0.1, top_p=0.9, max_tokens=2048",explanation:"Low temperature for precise, deterministic code generation."},{formula:"Creative writing: T=0.9, top_p=0.95, freq_penalty=0.5",explanation:"High temperature for diverse, creative output with reduced repetition."},{formula:"Factual Q&A: T=0.0, top_p=1.0, max_tokens=500",explanation:"Zero temperature for the most likely (factual) response."},{formula:"Brainstorming: T=1.2, top_p=0.98, presence_penalty=0.8",explanation:"High temperature and presence penalty for maximum diversity."}],id:"example-presets"}),e.jsx(t,{title:"Terminal",code:`# Access model parameters through the API
curl -X POST http://localhost:3000/api/chat/completions \\
    -H "Authorization: Bearer YOUR_TOKEN" \\
    -H "Content-Type: application/json" \\
    -d '{
        "model": "llama3.2",
        "messages": [
            {"role": "user", "content": "Write a haiku about programming"}
        ],
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": 100,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "stream": false
    }'

# In the Open WebUI interface:
# 1. Click the gear icon next to the model selector
# 2. Adjust sliders for temperature, top-p, etc.
# 3. Settings persist per conversation

# Or set defaults in the Modelfile for Ollama models:
# PARAMETER temperature 0.7
# PARAMETER top_p 0.9`,id:"code-params-api"}),e.jsx(t,{title:"param_comparison.py",code:`import requests
import json

API_URL = "http://localhost:3000/api/chat/completions"
HEADERS = {
    "Authorization": "Bearer YOUR_TOKEN",
    "Content-Type": "application/json",
}

def generate_with_params(prompt, model="llama3.2", **params):
    """Test different parameter settings."""
    resp = requests.post(API_URL, headers=HEADERS, json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "stream": False,
        **params,
    })
    return resp.json()["choices"][0]["message"]["content"]

prompt = "Describe a sunset."

# Compare different temperature settings
for temp in [0.0, 0.5, 1.0, 1.5]:
    result = generate_with_params(prompt, temperature=temp)
    print(f"T={temp}: {result[:80]}...")
    print()

# Compare top-p settings
for top_p in [0.5, 0.8, 0.95]:
    result = generate_with_params(prompt, temperature=0.8, top_p=top_p)
    print(f"top_p={top_p}: {result[:80]}...")
    print()`,id:"code-comparison"}),e.jsx(a,{type:"tip",title:"Save Parameter Presets",content:"Create model presets in Open WebUI for common task types. Each preset saves the model choice, system prompt, and all parameter settings. Switch between presets with one click instead of reconfiguring parameters each time.",id:"note-presets"}),e.jsx(n,{title:"Parameters Interact with Each Other",content:"Temperature, top-p, and top-k all affect the same sampling distribution. Setting temperature=2.0 with top_p=0.1 produces confusing results because temperature flattens the distribution while top_p aggressively filters it. Start with one parameter and adjust gradually.",id:"warning-interaction"})]})}const Be=Object.freeze(Object.defineProperty({__proto__:null,default:q},Symbol.toStringTag,{value:"Module"}));function N(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"System Prompts & Model Presets"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"System prompts define how a model behaves, its personality, capabilities, and constraints. Open WebUI lets you create reusable presets combining system prompts with model parameters, making it easy to switch between different assistant configurations."}),e.jsx(o,{title:"System Prompt",definition:"A system prompt is a special instruction given to the model before the user's message. It sets the context, personality, output format, and behavioral constraints. In the OpenAI message format, it uses role: 'system'.",id:"def-system-prompt"}),e.jsx(t,{title:"Terminal",code:`# Set a global default system prompt via environment
docker run -d -p 3000:8080 \\
    -e DEFAULT_SYSTEM_PROMPT="You are a helpful, concise assistant." \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# In the UI:
# 1. Click the model selector dropdown
# 2. Click "Set System Prompt" (pencil icon)
# 3. Enter your system prompt
# 4. It persists for that conversation

# Or create a Model Preset:
# Admin Panel > Workspace > Models > Add Model
# Configure system prompt + parameters as a reusable preset`,id:"code-system-prompt"}),e.jsx(t,{title:"preset_examples.py",code:`import requests

API_URL = "http://localhost:3000/api/chat/completions"
HEADERS = {
    "Authorization": "Bearer YOUR_TOKEN",
    "Content-Type": "application/json",
}

# Define reusable presets
PRESETS = {
    "code_reviewer": {
        "system": (
            "You are an expert code reviewer. Analyze code for bugs, "
            "performance issues, security vulnerabilities, and style. "
            "Rate severity as CRITICAL/WARNING/INFO. Be concise."
        ),
        "temperature": 0.2,
        "top_p": 0.9,
    },
    "socratic_tutor": {
        "system": (
            "You are a Socratic tutor. Never give direct answers. Instead, "
            "guide the student by asking leading questions that help them "
            "discover the answer themselves. Use encouragement."
        ),
        "temperature": 0.7,
        "top_p": 0.95,
    },
    "json_api": {
        "system": (
            "You are a JSON API. Respond ONLY with valid JSON. No markdown, "
            "no explanations, no text outside JSON. Use descriptive keys."
        ),
        "temperature": 0.1,
        "top_p": 0.8,
    },
    "eli5": {
        "system": (
            "Explain everything as if to a 5-year-old. Use simple words, "
            "short sentences, fun analogies. No jargon or technical terms."
        ),
        "temperature": 0.8,
        "top_p": 0.95,
    },
}

def chat_with_preset(preset_name, user_message, model="llama3.2"):
    preset = PRESETS[preset_name]
    resp = requests.post(API_URL, headers=HEADERS, json={
        "model": model,
        "messages": [
            {"role": "system", "content": preset["system"]},
            {"role": "user", "content": user_message},
        ],
        "temperature": preset["temperature"],
        "top_p": preset["top_p"],
        "stream": False,
    })
    return resp.json()["choices"][0]["message"]["content"]

# Test presets
code = "def add(a, b): return a + b"
print("Code review:", chat_with_preset("code_reviewer", f"Review: {code}"))
print("\\nELI5:", chat_with_preset("eli5", "What is gravity?"))`,id:"code-presets"}),e.jsx(s,{title:"Effective System Prompt Patterns",problem:"What makes a system prompt effective?",steps:[{formula:'Role definition: "You are a [specific role]"',explanation:"Give the model a clear identity and expertise area."},{formula:'Output format: "Respond in [format]"',explanation:"Specify JSON, markdown, bullet points, tables, etc."},{formula:'Constraints: "Never [behavior to avoid]"',explanation:"Explicitly state what the model should not do."},{formula:"Examples: show the desired output format",explanation:"One or two examples in the system prompt dramatically improve consistency."}],id:"example-patterns"}),e.jsx(a,{type:"tip",title:"Share Presets Across Users",content:"Admin-created model presets are available to all users. Create presets for common team use cases (customer support, code review, documentation) so everyone benefits from optimized configurations.",id:"note-sharing"}),e.jsx(n,{title:"System Prompts Use Context Window",content:"A long system prompt consumes tokens from the context window. A 500-token system prompt on a 4096-context model leaves only 3596 tokens for conversation. Keep system prompts concise -- aim for under 200 tokens.",id:"warning-context"})]})}const ze=Object.freeze(Object.defineProperty({__proto__:null,default:N},Symbol.toStringTag,{value:"Module"}));function B(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"RAG in Open WebUI"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Open WebUI has built-in Retrieval-Augmented Generation (RAG) capabilities. Upload documents, and the system automatically chunks, embeds, and retrieves relevant passages to augment the LLM's responses with your specific knowledge."}),e.jsx(o,{title:"RAG in Open WebUI",definition:"Open WebUI's RAG pipeline: (1) upload documents (PDF, TXT, DOCX, etc.), (2) automatic chunking and embedding using a local or remote embedding model, (3) storage in a built-in vector database (ChromaDB), (4) at query time, relevant chunks are retrieved and injected into the prompt.",id:"def-rag"}),e.jsx(s,{title:"RAG Workflow",problem:"How does document Q&A work in Open WebUI?",steps:[{formula:"Upload: drag PDF/DOCX into chat or use + button",explanation:"Documents are processed and stored automatically."},{formula:"Chunk: documents are split into ~500 token chunks",explanation:"Overlapping chunks ensure context is not lost at boundaries."},{formula:"Embed: chunks are converted to vectors via embedding model",explanation:"Uses Ollama embedding model or OpenAI embeddings."},{formula:"Retrieve: user query is embedded and matched against chunks",explanation:"Top-k most similar chunks are retrieved (default k=4)."},{formula:"Augment: retrieved chunks are prepended to the prompt",explanation:"The LLM sees relevant context and generates a grounded answer."}],id:"example-workflow"}),e.jsx(t,{title:"Terminal",code:`# Configure RAG settings via environment variables
docker run -d -p 3000:8080 \\
    -e RAG_EMBEDDING_MODEL=nomic-embed-text \\
    -e RAG_EMBEDDING_ENGINE=ollama \\
    -e CHUNK_SIZE=500 \\
    -e CHUNK_OVERLAP=50 \\
    -e RAG_TOP_K=4 \\
    -e RAG_RELEVANCE_THRESHOLD=0.3 \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Make sure the embedding model is available in Ollama
ollama pull nomic-embed-text

# In the UI:
# 1. Click the + button in chat to upload a document
# 2. Or go to Workspace > Documents to manage a knowledge base
# 3. Create collections to group related documents
# 4. Reference a collection with # in chat: "#my-collection"`,id:"code-config"}),e.jsx(t,{title:"rag_api.py",code:`import requests

BASE_URL = "http://localhost:3000/api/v1"
HEADERS = {"Authorization": "Bearer YOUR_TOKEN"}

# Upload a document for RAG
with open("report.pdf", "rb") as f:
    resp = requests.post(
        f"{BASE_URL}/files/",
        headers=HEADERS,
        files={"file": ("report.pdf", f, "application/pdf")},
    )
    file_data = resp.json()
    print(f"Uploaded: {file_data.get('id')}")

# Create a knowledge collection
resp = requests.post(
    f"{BASE_URL}/knowledge/create",
    headers=HEADERS,
    json={
        "name": "Q3 Reports",
        "description": "Quarterly financial reports",
    },
)
collection = resp.json()

# Chat with document context
resp = requests.post(
    f"{BASE_URL}/../api/chat/completions",
    headers={**HEADERS, "Content-Type": "application/json"},
    json={
        "model": "llama3.2",
        "messages": [
            {"role": "user", "content": "What were the Q3 revenue numbers?"}
        ],
        "files": [{"type": "file", "id": file_data["id"]}],
        "stream": False,
    },
)
print(resp.json()["choices"][0]["message"]["content"])`,id:"code-rag-api"}),e.jsx(a,{type:"tip",title:"Optimizing RAG Quality",content:"For better RAG results: (1) use domain-appropriate chunk sizes (smaller for Q&A, larger for summarization), (2) increase top_k if answers span multiple sections, (3) use a strong embedding model like nomic-embed-text, and (4) organize related documents into collections.",id:"note-quality"}),e.jsx(n,{title:"RAG Is Not Perfect",content:"RAG retrieval can miss relevant passages if the query uses different terminology than the document. It may also retrieve irrelevant chunks that confuse the model. Always verify important answers against the source document. Consider adjusting the relevance threshold if you get too many false positives.",id:"warning-limitations"})]})}const De=Object.freeze(Object.defineProperty({__proto__:null,default:B},Symbol.toStringTag,{value:"Module"}));function z(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Web Search Integration"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Open WebUI can augment LLM responses with real-time web search results. When enabled, the system searches the web for relevant information and includes it in the context, helping models provide up-to-date answers."}),e.jsx(o,{title:"Web Search in Open WebUI",definition:"Web search integration fetches live search results, scrapes relevant pages, and injects the content into the LLM prompt. This gives local models access to current information beyond their training cutoff date.",id:"def-web-search"}),e.jsx(t,{title:"Terminal",code:`# Configure web search via environment variables
# Supports multiple search engines:

# Option 1: SearXNG (self-hosted, privacy-focused, free)
docker run -d -p 8888:8080 \\
    --name searxng \\
    -v searxng-data:/etc/searxng \\
    searxng/searxng

docker run -d -p 3000:8080 \\
    -e ENABLE_RAG_WEB_SEARCH=true \\
    -e RAG_WEB_SEARCH_ENGINE=searxng \\
    -e SEARXNG_QUERY_URL=http://searxng:8080/search?q=<query>&format=json \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Option 2: Google PSE (requires API key)
# -e RAG_WEB_SEARCH_ENGINE=google_pse
# -e GOOGLE_PSE_API_KEY=your-key
# -e GOOGLE_PSE_ENGINE_ID=your-engine-id

# Option 3: Brave Search API
# -e RAG_WEB_SEARCH_ENGINE=brave
# -e BRAVE_SEARCH_API_KEY=your-key

# Option 4: DuckDuckGo (no API key needed)
# -e RAG_WEB_SEARCH_ENGINE=duckduckgo`,id:"code-config"}),e.jsx(t,{title:"docker-compose-search.py",code:`# Docker Compose with SearXNG for fully self-hosted web search
compose_config = """
services:
  searxng:
    image: searxng/searxng
    volumes:
      - searxng-data:/etc/searxng
    environment:
      - SEARXNG_BASE_URL=http://searxng:8080/

  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    volumes:
      - open_webui_data:/app/backend/data
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - ENABLE_RAG_WEB_SEARCH=true
      - RAG_WEB_SEARCH_ENGINE=searxng
      - SEARXNG_QUERY_URL=http://searxng:8080/search?q=<query>&format=json
      - RAG_WEB_SEARCH_RESULT_COUNT=5
      - RAG_WEB_SEARCH_CONCURRENT_REQUESTS=5
    depends_on:
      - ollama
      - searxng

volumes:
  ollama_data:
  open_webui_data:
  searxng-data:
"""

with open("docker-compose.yml", "w") as f:
    f.write(compose_config)

print("Created docker-compose.yml with web search support")
print("Run: docker compose up -d")`,id:"code-compose"}),e.jsx(s,{title:"How Web Search Augmentation Works",problem:"What happens when a user asks a question with web search enabled?",steps:[{formula:"Query reformulation: LLM converts user message to search query",explanation:"The model extracts key search terms from the conversational input."},{formula:"Search: query is sent to the configured search engine",explanation:"Returns top N URLs and snippets."},{formula:"Scraping: web pages are fetched and text is extracted",explanation:"HTML is cleaned to plain text, limited to a reasonable length."},{formula:"Injection: search results are added to the prompt context",explanation:"The LLM receives both the user question and relevant web content."}],id:"example-flow"}),e.jsx(a,{type:"tip",title:"Toggle Per-Message",content:"You can enable or disable web search per message using the toggle in the chat input area. This is useful when you only need current information for specific questions, not every message in a conversation.",id:"note-toggle"}),e.jsx(n,{title:"Web Search Adds Latency",content:"Each web search adds 2-5 seconds to response time (search + scraping + processing). The retrieved content also uses context window tokens. Keep RAG_WEB_SEARCH_RESULT_COUNT low (3-5) to balance freshness against speed.",id:"warning-latency"})]})}const We=Object.freeze(Object.defineProperty({__proto__:null,default:z},Symbol.toStringTag,{value:"Module"}));function D(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Function Calling & Tools"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Open WebUI supports tools and functions that extend the LLM's capabilities beyond text generation. Tools let the model execute code, query APIs, access databases, and interact with external systems, all within the chat interface."}),e.jsx(o,{title:"Tools in Open WebUI",definition:"Tools are Python functions that the LLM can invoke during a conversation. Open WebUI provides a framework for defining tools with descriptions, parameter schemas, and execution logic. The model decides when and how to use available tools based on the user's request.",id:"def-tools"}),e.jsx(s,{title:"Built-in vs Custom Tools",problem:"What tools come with Open WebUI and how do you add your own?",steps:[{formula:"Built-in: web search, code execution, image generation",explanation:"Available out of the box with minimal configuration."},{formula:"Community: browse and install from the tool marketplace",explanation:"Open WebUI has a community repository of shared tools."},{formula:"Custom: write Python functions with the Tools API",explanation:"Define your own tools for any custom integration."}],id:"example-tools"}),e.jsx(t,{title:"custom_tool.py",code:`# Example: Custom tool for Open WebUI
# This goes in the Open WebUI Tools editor (Admin > Workspace > Tools)

"""
title: Weather Tool
description: Get current weather for a location
author: your-name
version: 0.1.0
"""

import requests
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        api_key: str = Field(
            default="", description="OpenWeatherMap API key"
        )

    def __init__(self):
        self.valves = self.Valves()

    def get_weather(
        self,
        location: str,
    ) -> str:
        """
        Get the current weather for a given location.

        :param location: City name (e.g., 'London', 'New York')
        :return: Weather description with temperature
        """
        try:
            resp = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "q": location,
                    "appid": self.valves.api_key,
                    "units": "metric",
                },
                timeout=10,
            )
            data = resp.json()
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            return f"Weather in {location}: {desc}, {temp}°C"
        except Exception as e:
            return f"Could not get weather: {str(e)}"`,id:"code-custom-tool"}),e.jsx(t,{title:"calculator_tool.py",code:`# Another example: Calculator tool
# Paste into Tools editor in Open WebUI

"""
title: Calculator
description: Perform mathematical calculations
author: your-name
version: 0.1.0
"""

import math
from pydantic import BaseModel


class Tools:
    def calculate(self, expression: str) -> str:
        """
        Evaluate a mathematical expression safely.

        :param expression: Math expression to evaluate (e.g., 'sqrt(144) + 2**3')
        :return: The result of the calculation
        """
        # Safe math functions
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": math.sqrt,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "pi": math.pi,
            "e": math.e, "inf": float("inf"),
        }
        try:
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return f"Result: {expression} = {result}"
        except Exception as e:
            return f"Error evaluating '{expression}': {str(e)}"

    def unit_convert(self, value: float, from_unit: str, to_unit: str) -> str:
        """
        Convert between common units.

        :param value: Numeric value to convert
        :param from_unit: Source unit (e.g., 'km', 'miles', 'celsius')
        :param to_unit: Target unit (e.g., 'miles', 'km', 'fahrenheit')
        :return: Converted value with units
        """
        conversions = {
            ("km", "miles"): lambda v: v * 0.621371,
            ("miles", "km"): lambda v: v * 1.60934,
            ("celsius", "fahrenheit"): lambda v: v * 9/5 + 32,
            ("fahrenheit", "celsius"): lambda v: (v - 32) * 5/9,
            ("kg", "lbs"): lambda v: v * 2.20462,
            ("lbs", "kg"): lambda v: v * 0.453592,
        }
        key = (from_unit.lower(), to_unit.lower())
        if key in conversions:
            result = conversions[key](value)
            return f"{value} {from_unit} = {result:.4f} {to_unit}"
        return f"Unknown conversion: {from_unit} -> {to_unit}"`,id:"code-calculator"}),e.jsx(a,{type:"tip",title:"Tool Discovery",content:"Browse community tools at openwebui.com. Tools can be installed with one click and include integrations for Jira, Slack, databases, file systems, and more. Check the tool's source code before installing to understand what it does.",id:"note-discovery"}),e.jsx(n,{title:"Tool Security",content:"Custom tools execute arbitrary Python code on your server. Only install tools from trusted sources and review the code before enabling. Malicious tools could access your file system, network, or other resources. Admins should control which tools are available to users.",id:"warning-security"})]})}const Fe=Object.freeze(Object.defineProperty({__proto__:null,default:D},Symbol.toStringTag,{value:"Module"}));function W(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Custom Pipelines & Filters"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Pipelines in Open WebUI let you intercept and transform messages before they reach the model (inlet) or after the model responds (outlet). This enables content filtering, logging, custom routing, translation, and advanced processing workflows."}),e.jsx(o,{title:"Pipelines",definition:"A pipeline is a Python class with inlet (pre-processing) and/or outlet (post-processing) methods that modify the message flow. Inlet filters can modify, reject, or augment user messages. Outlet filters can modify, annotate, or log model responses.",id:"def-pipelines"}),e.jsx(s,{title:"Pipeline Types",problem:"What kinds of pipelines can you build?",steps:[{formula:"Filter pipeline: content moderation, PII removal",explanation:"Block or redact sensitive content before it reaches the model."},{formula:"Augmentation pipeline: add context, inject RAG results",explanation:"Enrich prompts with additional data before generation."},{formula:"Routing pipeline: choose model based on query",explanation:"Automatically route to specialized models based on task type."},{formula:"Logging pipeline: audit, analytics, cost tracking",explanation:"Log all interactions for compliance or optimization."}],id:"example-types"}),e.jsx(t,{title:"content_filter_pipeline.py",code:`# Content moderation pipeline
# Add via Admin > Workspace > Functions > Add Pipeline

"""
title: Content Filter
description: Filters inappropriate content and PII
author: your-name
version: 0.1.0
"""

import re
from pydantic import BaseModel, Field
from typing import Optional


class Pipeline:
    class Valves(BaseModel):
        block_keywords: str = Field(
            default="password,secret,ssn",
            description="Comma-separated list of blocked keywords"
        )
        redact_emails: bool = Field(
            default=True,
            description="Redact email addresses"
        )

    def __init__(self):
        self.name = "Content Filter"
        self.valves = self.Valves()

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Pre-process user message before sending to LLM."""
        messages = body.get("messages", [])
        blocked_words = [
            w.strip() for w in self.valves.block_keywords.split(",")
        ]

        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"]

                # Check for blocked keywords
                for word in blocked_words:
                    if word.lower() in content.lower():
                        raise Exception(
                            f"Message blocked: contains restricted term"
                        )

                # Redact email addresses
                if self.valves.redact_emails:
                    content = re.sub(
                        r'[\\w.-]+@[\\w.-]+\\.\\w+',
                        '[EMAIL REDACTED]',
                        content
                    )
                msg["content"] = content

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Post-process model response."""
        messages = body.get("messages", [])
        for msg in messages:
            if msg["role"] == "assistant":
                # Add disclaimer to responses
                if "medical" in msg["content"].lower():
                    msg["content"] += (
                        "\\n\\n*Disclaimer: This is not medical advice. "
                        "Consult a healthcare professional.*"
                    )
        return body`,id:"code-filter"}),e.jsx(t,{title:"logging_pipeline.py",code:`# Logging and analytics pipeline
"""
title: Usage Logger
description: Logs all conversations for analytics
author: your-name
version: 0.1.0
"""

import json
import time
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class Pipeline:
    class Valves(BaseModel):
        log_file: str = Field(
            default="/app/backend/data/usage_log.jsonl",
            description="Path to log file"
        )

    def __init__(self):
        self.name = "Usage Logger"
        self.valves = self.Valves()

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Log incoming requests."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "request",
            "user": user.get("name", "unknown") if user else "unknown",
            "model": body.get("model", "unknown"),
            "message_count": len(body.get("messages", [])),
            "last_message_length": len(
                body.get("messages", [{}])[-1].get("content", "")
            ),
        }

        with open(self.valves.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\\n")

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Log responses."""
        messages = body.get("messages", [])
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]

        if assistant_msgs:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "response",
                "user": user.get("name", "unknown") if user else "unknown",
                "response_length": len(assistant_msgs[-1].get("content", "")),
            }
            with open(self.valves.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\\n")

        return body`,id:"code-logging"}),e.jsx(a,{type:"tip",title:"Pipeline Chaining",content:"Multiple pipelines can be active simultaneously and they execute in order. For example: content filter (inlet) -> RAG augmentation (inlet) -> model generates response -> logging (outlet) -> response formatting (outlet). Order matters for correctness.",id:"note-chaining"}),e.jsx(n,{title:"Pipeline Errors Block Messages",content:"If a pipeline raises an exception in the inlet, the message is blocked and the user sees an error. Test pipelines thoroughly before enabling them in production. A buggy filter pipeline can make the entire chat interface unusable.",id:"warning-errors"})]})}const He=Object.freeze(Object.defineProperty({__proto__:null,default:W},Symbol.toStringTag,{value:"Module"}));function F(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Image Generation Integration"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Open WebUI can integrate with image generation backends like AUTOMATIC1111, ComfyUI, and OpenAI's DALL-E. Users can generate images directly in chat conversations, combining text and visual AI capabilities in one interface."}),e.jsx(o,{title:"Image Generation in Open WebUI",definition:"Open WebUI supports multiple image generation backends. When configured, users can request images in chat and the system routes the request to the appropriate backend, displays the generated image inline, and optionally stores it for later reference.",id:"def-image-gen"}),e.jsx(t,{title:"Terminal",code:`# Option 1: AUTOMATIC1111 (Stable Diffusion WebUI)
# Start A1111 with API enabled:
python launch.py --api --listen

# Configure in Open WebUI:
docker run -d -p 3000:8080 \\
    -e IMAGE_GENERATION_ENGINE=automatic1111 \\
    -e AUTOMATIC1111_BASE_URL=http://host.docker.internal:7860 \\
    -e ENABLE_IMAGE_GENERATION=true \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Option 2: ComfyUI
# Start ComfyUI server, then configure:
# -e IMAGE_GENERATION_ENGINE=comfyui
# -e COMFYUI_BASE_URL=http://host.docker.internal:8188

# Option 3: OpenAI DALL-E
# Uses the OpenAI API key already configured
# -e IMAGE_GENERATION_ENGINE=openai
# -e IMAGE_GENERATION_MODEL=dall-e-3

# In the chat, type a message like:
# "Generate an image of a sunset over mountains"
# Or use /image command if configured`,id:"code-config"}),e.jsx(t,{title:"image_gen_api.py",code:`import requests
import base64
from pathlib import Path

BASE_URL = "http://localhost:3000"
HEADERS = {"Authorization": "Bearer YOUR_TOKEN"}

# Generate an image through Open WebUI's API
resp = requests.post(
    f"{BASE_URL}/api/v1/images/generations",
    headers={**HEADERS, "Content-Type": "application/json"},
    json={
        "prompt": "A serene mountain landscape at sunset, photorealistic",
        "n": 1,
        "size": "512x512",
    },
)

if resp.status_code == 200:
    data = resp.json()
    for i, image in enumerate(data.get("data", [])):
        if "b64_json" in image:
            img_bytes = base64.b64decode(image["b64_json"])
            Path(f"generated_{i}.png").write_bytes(img_bytes)
            print(f"Saved generated_{i}.png")
        elif "url" in image:
            print(f"Image URL: {image['url']}")
else:
    print(f"Error: {resp.status_code} - {resp.text}")

# Using AUTOMATIC1111 directly for more control
A1111_URL = "http://localhost:7860"
resp = requests.post(f"{A1111_URL}/sdapi/v1/txt2img", json={
    "prompt": "a cute robot reading a book, digital art",
    "negative_prompt": "blurry, low quality",
    "steps": 30,
    "width": 512,
    "height": 512,
    "cfg_scale": 7.5,
    "sampler_name": "DPM++ 2M Karras",
})
if resp.status_code == 200:
    images = resp.json().get("images", [])
    for i, img_b64 in enumerate(images):
        Path(f"sd_image_{i}.png").write_bytes(base64.b64decode(img_b64))
        print(f"Saved sd_image_{i}.png")`,id:"code-api"}),e.jsx(s,{title:"Image Generation Backends",problem:"Compare the available image generation backends.",steps:[{formula:"AUTOMATIC1111: full Stable Diffusion control, many models",explanation:"Most flexible. Supports LoRAs, ControlNet, inpainting. Requires GPU."},{formula:"ComfyUI: node-based workflows, advanced pipelines",explanation:"Most powerful for complex generation pipelines. Steeper learning curve."},{formula:"DALL-E 3: highest quality, no local GPU needed",explanation:"OpenAI cloud API. Best quality but costs per image."}],id:"example-backends"}),e.jsx(a,{type:"tip",title:"LLM-Enhanced Prompts",content:"Open WebUI can use the LLM to enhance image generation prompts. A simple request like 'draw a cat' gets expanded into a detailed prompt with style, lighting, and composition details before being sent to the image generator.",id:"note-prompt-enhancement"}),e.jsx(n,{title:"GPU Memory Sharing",content:"Running both an LLM (via Ollama) and Stable Diffusion on the same GPU requires careful memory management. A 7B LLM uses ~5GB and SD 1.5 uses ~4GB -- tight for an 8GB GPU. Consider using smaller LLMs or offloading SD to CPU for generation.",id:"warning-gpu-memory"})]})}const Ke=Object.freeze(Object.defineProperty({__proto__:null,default:F},Symbol.toStringTag,{value:"Module"}));function H(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Voice Input/Output (STT & TTS)"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Open WebUI supports both speech-to-text (STT) for voice input and text-to-speech (TTS) for spoken responses. This enables hands-free interaction with LLMs, making the interface accessible on mobile devices and for voice-first workflows."}),e.jsx(o,{title:"STT & TTS in Open WebUI",definition:"Speech-to-text converts audio input to text before sending to the LLM. Text-to-speech converts the LLM's text response to audio. Open WebUI supports browser-native speech APIs, OpenAI Whisper/TTS, and self-hosted alternatives.",id:"def-voice"}),e.jsx(t,{title:"Terminal",code:`# Configure voice with OpenAI APIs
docker run -d -p 3000:8080 \\
    -e AUDIO_STT_ENGINE=openai \\
    -e AUDIO_STT_MODEL=whisper-1 \\
    -e AUDIO_TTS_ENGINE=openai \\
    -e AUDIO_TTS_MODEL=tts-1 \\
    -e AUDIO_TTS_VOICE=nova \\
    -e OPENAI_API_KEY=sk-your-key \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Self-hosted STT with local Whisper
# -e AUDIO_STT_ENGINE=openai
# -e AUDIO_STT_OPENAI_API_BASE_URL=http://localhost:8000/v1
# Uses faster-whisper or whisper.cpp server

# Browser-native speech (no server needed, limited quality)
# -e AUDIO_STT_ENGINE=web

# Available TTS voices (OpenAI): alloy, echo, fable, onyx, nova, shimmer`,id:"code-config"}),e.jsx(t,{title:"self_hosted_whisper.py",code:`# Set up a self-hosted Whisper server for STT
# Option 1: faster-whisper-server
# pip install faster-whisper
# Provides an OpenAI-compatible Whisper API

import subprocess
import requests
import tempfile
import wave
import struct
import math

# Start faster-whisper server (run separately)
# python -m faster_whisper_server --model large-v3 --port 8000

# Test STT by sending audio to the local Whisper server
WHISPER_URL = "http://localhost:8000/v1/audio/transcriptions"

# Generate a test audio file (sine wave)
def create_test_audio(filename, duration=2, freq=440):
    sample_rate = 16000
    n_samples = int(sample_rate * duration)
    with wave.open(filename, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(n_samples):
            value = int(32767 * math.sin(2 * math.pi * freq * i / sample_rate))
            wav.writeframes(struct.pack('<h', value))

# In practice, record real audio or use a file
# Here we just demonstrate the API call
audio_file = "test.wav"
create_test_audio(audio_file)

with open(audio_file, "rb") as f:
    resp = requests.post(
        WHISPER_URL,
        files={"file": ("audio.wav", f, "audio/wav")},
        data={"model": "large-v3"},
    )
    print(f"Transcription: {resp.json().get('text', 'N/A')}")

# For TTS, use Piper (fast, local, open-source)
# pip install piper-tts
# piper --model en_US-lessac-medium.onnx --output_file output.wav
print("Local STT + TTS configured for Open WebUI")`,id:"code-whisper"}),e.jsx(s,{title:"Voice Integration Options",problem:"Compare STT and TTS options for Open WebUI.",steps:[{formula:"Browser Web Speech API: free, no setup, basic quality",explanation:"Uses the browser built-in speech recognition. Works offline in Chrome."},{formula:"OpenAI Whisper + TTS: best quality, cloud API costs",explanation:"Whisper for STT, TTS-1 for speech. ~$0.006/min STT, $0.015/1K chars TTS."},{formula:"Self-hosted Whisper + Piper: free, runs locally",explanation:"faster-whisper for STT, Piper for TTS. Needs GPU for real-time speed."}],id:"example-options"}),e.jsx(a,{type:"tip",title:"Mobile Voice Chat",content:"Open WebUI's voice features work well on mobile browsers. Press and hold the microphone button to record, release to send. Combined with TTS, this creates a voice assistant experience entirely running on your own hardware.",id:"note-mobile"}),e.jsx(n,{title:"STT Accuracy",content:"Speech recognition accuracy depends heavily on the model and audio quality. Background noise, accents, and technical terminology can reduce accuracy. For critical inputs, always review the transcribed text before sending to the LLM.",id:"warning-accuracy"})]})}const $e=Object.freeze(Object.defineProperty({__proto__:null,default:H},Symbol.toStringTag,{value:"Module"}));function K(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Admin Panel"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The Admin Panel is the control center for managing your Open WebUI deployment. It covers user management, model access control, connection configuration, and system-wide settings that affect all users."}),e.jsx(o,{title:"Admin Panel",definition:"The Admin Panel is accessible to users with the admin role. It provides centralized management of connections (Ollama, OpenAI), user accounts and roles, model visibility and access control, default settings, and system configuration.",id:"def-admin"}),e.jsx(s,{title:"Admin Panel Sections",problem:"What can you configure in the admin panel?",steps:[{formula:"Connections: Ollama URL, OpenAI API keys, custom endpoints",explanation:"Configure all LLM backends from one place."},{formula:"Users: approve/deny signups, assign roles, set permissions",explanation:"Control who can access the system and what they can do."},{formula:"Models: show/hide models, set defaults, create presets",explanation:"Control which models are visible and set default parameters."},{formula:"Settings: auth, RAG config, web search, image gen",explanation:"System-wide configuration for all features."}],id:"example-sections"}),e.jsx(t,{title:"Terminal",code:`# Key environment variables for admin configuration
docker run -d -p 3000:8080 \\
    -e WEBUI_AUTH=true \\
    -e ENABLE_SIGNUP=true \\
    -e DEFAULT_USER_ROLE=pending \\
    -e ENABLE_ADMIN_EXPORT=true \\
    -e ENABLE_ADMIN_CHAT_ACCESS=false \\
    -e ENABLE_COMMUNITY_SHARING=false \\
    -e DEFAULT_MODELS="llama3.2" \\
    -e MODEL_FILTER_ENABLED=true \\
    -e MODEL_FILTER_LIST="llama3.2;mistral;phi3:mini" \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Restrict model access (only listed models are visible)
# MODEL_FILTER_ENABLED=true + MODEL_FILTER_LIST filters the model dropdown

# Disable admin access to user conversations
# ENABLE_ADMIN_CHAT_ACCESS=false (privacy-respecting default)

# Reset admin password if locked out
docker exec open-webui open-webui reset-admin-password
# Outputs a temporary password`,id:"code-admin-config"}),e.jsx(t,{title:"admin_api.py",code:`import requests

BASE_URL = "http://localhost:3000/api/v1"
ADMIN_TOKEN = "your-admin-token"
HEADERS = {"Authorization": f"Bearer {ADMIN_TOKEN}"}

# Get system configuration
resp = requests.get(f"{BASE_URL}/configs", headers=HEADERS)
if resp.status_code == 200:
    config = resp.json()
    print("Current configuration:")
    for key, value in config.items():
        if not key.startswith("_"):
            print(f"  {key}: {value}")

# List all users with their roles
resp = requests.get(f"{BASE_URL}/users", headers=HEADERS)
users = resp.json()
print(f"\\nTotal users: {len(users)}")
role_counts = {}
for user in users:
    role = user.get("role", "unknown")
    role_counts[role] = role_counts.get(role, 0) + 1
    print(f"  {user['name']:<20} {user['email']:<30} {role}")
print(f"\\nRole distribution: {role_counts}")

# Export all chats (admin function for backup)
resp = requests.get(
    f"{BASE_URL}/chats/all/export",
    headers=HEADERS,
)
if resp.status_code == 200:
    import json
    with open("all_chats_backup.json", "w") as f:
        json.dump(resp.json(), f, indent=2)
    print("\\nExported all chats to all_chats_backup.json")`,id:"code-admin-api"}),e.jsx(a,{type:"tip",title:"Regular Backups",content:"Back up the Open WebUI data volume regularly. It contains all conversations, user accounts, uploaded documents, and settings. Use 'docker cp open-webui:/app/backend/data ./backup' or mount the volume to a backed-up filesystem.",id:"note-backups"}),e.jsx(n,{title:"Admin Chat Access",content:"When ENABLE_ADMIN_CHAT_ACCESS is true, admins can view all user conversations. This may be required for compliance but raises privacy concerns. Communicate the policy clearly to users and disable it unless strictly necessary.",id:"warning-chat-access"})]})}const Ve=Object.freeze(Object.defineProperty({__proto__:null,default:K},Symbol.toStringTag,{value:"Module"}));function $(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Theming & Customization"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Open WebUI supports visual customization through themes, custom CSS, and branding options. You can match the interface to your organization's look and feel or simply personalize it for your own preferences."}),e.jsx(s,{title:"Customization Options",problem:"What can you visually customize in Open WebUI?",steps:[{formula:"Dark/Light mode: toggle in settings",explanation:"Built-in dark and light themes with automatic system detection."},{formula:"Custom CSS: inject arbitrary CSS styles",explanation:"Full control over colors, fonts, spacing, and layout."},{formula:"Branding: custom logo, name, description",explanation:"White-label the interface for your organization."},{formula:"Chat interface: message bubbles, code themes",explanation:"Customize how conversations are displayed."}],id:"example-options"}),e.jsx(t,{title:"Terminal",code:`# Set custom branding via environment variables
docker run -d -p 3000:8080 \\
    -e WEBUI_NAME="Acme AI Assistant" \\
    -e ENABLE_SIGNUP=true \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Custom CSS can be added through the Admin Panel:
# Settings > Interface > Custom CSS

# Or mount a custom CSS file:
# -v ./custom.css:/app/build/static/custom.css`,id:"code-branding"}),e.jsx(t,{title:"custom_themes.py",code:`# Example custom CSS themes for Open WebUI
# Apply via Admin Panel > Settings > Interface > Custom CSS

themes = {
    "corporate_blue": """
/* Corporate blue theme */
:root {
    --primary-color: #1e40af;
    --primary-hover: #1d4ed8;
    --background-color: #f8fafc;
    --sidebar-bg: #1e293b;
    --sidebar-text: #e2e8f0;
}

.dark {
    --background-color: #0f172a;
    --sidebar-bg: #1e293b;
}

/* Custom font */
body {
    font-family: 'Inter', -apple-system, sans-serif;
}

/* Rounded message bubbles */
.message-content {
    border-radius: 16px;
    padding: 12px 16px;
}
""",
    "minimal": """
/* Minimal, clean theme */
.sidebar {
    border-right: 1px solid #e5e7eb;
}

/* Hide the logo/branding */
.logo-container { display: none; }

/* Increase content width */
.max-w-3xl { max-width: 56rem; }

/* Subtle code blocks */
pre {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
}
""",
    "high_contrast": """
/* High contrast for accessibility */
body { font-size: 18px; }
.dark {
    --background-color: #000000;
    --text-color: #ffffff;
}
.message-content {
    line-height: 1.8;
    letter-spacing: 0.02em;
}
a { text-decoration: underline; }
""",
}

# Print theme for copy-pasting into Open WebUI
for name, css in themes.items():
    print(f"\\n{'='*40}")
    print(f"Theme: {name}")
    print(f"{'='*40}")
    print(css)`,id:"code-themes"}),e.jsx(a,{type:"tip",title:"Per-User Preferences",content:"Each user can set their own theme preference (dark/light) independently. Custom CSS set by admins applies to all users. For per-user CSS customization, users can use browser extensions like Stylus.",id:"note-per-user"}),e.jsx(a,{type:"note",title:"Custom Landing Page",content:"You can customize the landing page shown to new visitors before they log in. This is useful for displaying your organization's AI usage policy, instructions, or branding. Configure through the admin panel under Interface settings.",id:"note-landing"}),e.jsx(n,{title:"CSS Injection Safety",content:"Custom CSS is powerful but can break the interface if not carefully tested. Always test CSS changes in a development environment first. Keep a backup of working CSS before making changes. Invalid CSS can make the admin panel inaccessible.",id:"warning-css"})]})}const Qe=Object.freeze(Object.defineProperty({__proto__:null,default:$},Symbol.toStringTag,{value:"Module"}));function V(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"API Access to Open WebUI"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Open WebUI exposes an OpenAI-compatible API that routes through all configured backends. This means you can use Open WebUI as a unified API gateway, getting the benefits of its model management, RAG, and pipeline features from any client application."}),e.jsx(o,{title:"Open WebUI API",definition:"Open WebUI provides an API at /api/chat/completions that is compatible with the OpenAI SDK. Requests go through Open WebUI's pipeline system (filters, RAG, tools) before reaching the backend model. Authentication uses bearer tokens generated in the UI settings.",id:"def-api"}),e.jsx(t,{title:"Terminal",code:`# Generate an API key:
# 1. Open WebUI > Settings > Account
# 2. Click "Create API Key"
# 3. Copy the key (shown only once)

# Test the API with curl
curl http://localhost:3000/api/chat/completions \\
    -H "Authorization: Bearer sk-your-open-webui-key" \\
    -H "Content-Type: application/json" \\
    -d '{
        "model": "llama3.2",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "stream": false
    }'

# List available models
curl http://localhost:3000/api/models \\
    -H "Authorization: Bearer sk-your-open-webui-key"

# Streaming response
curl http://localhost:3000/api/chat/completions \\
    -H "Authorization: Bearer sk-your-open-webui-key" \\
    -H "Content-Type: application/json" \\
    -d '{
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "Count to 5"}],
        "stream": true
    }'`,id:"code-curl"}),e.jsx(t,{title:"openai_sdk_client.py",code:`from openai import OpenAI

# Point the standard OpenAI SDK at Open WebUI
client = OpenAI(
    base_url="http://localhost:3000/api",
    api_key="sk-your-open-webui-key",
)

# Chat completion (works with any configured backend)
response = client.chat.completions.create(
    model="llama3.2",  # Any model visible in Open WebUI
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain Docker in one sentence."},
    ],
    temperature=0.7,
    max_tokens=200,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Write a haiku about AI"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()

# List models from all backends
models = client.models.list()
for model in models.data:
    print(f"  {model.id}")

# This means any tool that supports OpenAI API works with Open WebUI:
# - LangChain
# - LlamaIndex
# - AutoGen
# - CrewAI
# - Custom applications`,id:"code-sdk"}),e.jsx(t,{title:"langchain_integration.py",code:`# Use Open WebUI as backend for LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(
    base_url="http://localhost:3000/api",
    api_key="sk-your-open-webui-key",
    model="llama3.2",
    temperature=0.7,
)

# Simple chat
response = llm.invoke([
    SystemMessage(content="You are a Python expert."),
    HumanMessage(content="What is a decorator?"),
])
print(response.content)

# Batch processing
prompts = [
    [HumanMessage(content=f"Explain {concept} briefly")]
    for concept in ["REST APIs", "GraphQL", "gRPC"]
]
results = llm.batch(prompts)
for prompt, result in zip(["REST", "GraphQL", "gRPC"], results):
    print(f"\\n{prompt}: {result.content[:100]}...")`,id:"code-langchain"}),e.jsx(s,{title:"API Gateway Benefits",problem:"Why use Open WebUI as an API gateway?",steps:[{formula:"Unified endpoint for all backends",explanation:"One API URL for Ollama, OpenAI, Anthropic, and custom models."},{formula:"Pipeline processing for all API requests",explanation:"Content filters, logging, and augmentation apply to API calls too."},{formula:"User-level API keys and access control",explanation:"Each user gets their own key with appropriate model access."},{formula:"Conversation logging and analytics",explanation:"All API interactions are tracked alongside web UI usage."}],id:"example-gateway"}),e.jsx(a,{type:"tip",title:"Replace OpenAI in Existing Apps",content:"Any application using the OpenAI SDK can switch to Open WebUI by changing two lines: base_url and api_key. No other code changes needed. This makes it trivial to move from cloud to local models.",id:"note-migration"}),e.jsx(n,{title:"API Rate Limits",content:"Open WebUI does not have built-in API rate limiting. High-volume API usage can overwhelm the backend, especially with local models. Implement rate limiting in a reverse proxy (nginx) if exposing the API to multiple clients or applications.",id:"warning-rate-limits"})]})}const Ye=Object.freeze(Object.defineProperty({__proto__:null,default:V},Symbol.toStringTag,{value:"Module"}));function Q(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"vLLM"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"vLLM is a high-throughput, memory-efficient inference engine for LLMs. Its key innovation is PagedAttention, which manages KV-cache memory like operating system virtual memory pages, enabling 2-4x higher throughput than naive serving implementations."}),e.jsx(o,{title:"vLLM",definition:"vLLM (Virtual LLM) is an open-source LLM serving library featuring PagedAttention for efficient KV-cache management, continuous batching, tensor parallelism, speculative decoding, and an OpenAI-compatible API server. It is the most popular production LLM serving framework.",id:"def-vllm"}),e.jsx(o,{title:"PagedAttention",definition:"PagedAttention stores KV-cache in non-contiguous memory blocks (pages) mapped via a block table. This eliminates memory fragmentation and waste from pre-allocation, achieving near-optimal memory utilization. Memory waste drops from 60-80% to under 4%.",id:"def-paged-attention"}),e.jsx(t,{title:"Terminal",code:`# Install vLLM
pip install vllm

# Start the OpenAI-compatible API server
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --gpu-memory-utilization 0.9 \\
    --max-model-len 8192 \\
    --dtype auto

# With tensor parallelism for large models
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-70B-Instruct \\
    --tensor-parallel-size 4 \\
    --gpu-memory-utilization 0.9

# With speculative decoding
python -m vllm.entrypoints.openai.api_server \\
    --model meta-llama/Llama-3.1-70B-Instruct \\
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \\
    --num-speculative-tokens 5

# Docker deployment
docker run --gpus all -p 8000:8000 \\
    -v ~/.cache/huggingface:/root/.cache/huggingface \\
    vllm/vllm-openai:latest \\
    --model meta-llama/Llama-3.1-8B-Instruct`,id:"code-server"}),e.jsx(t,{title:"vllm_client.py",code:`from openai import OpenAI
import time

# vLLM serves an OpenAI-compatible API
client = OpenAI(base_url="http://localhost:8000/v1", api_key="vllm")

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain PagedAttention in 3 sentences."},
    ],
    temperature=0.7,
    max_tokens=200,
)
print(response.choices[0].message.content)

# Batch processing for maximum throughput
import concurrent.futures

def send_request(prompt):
    start = time.time()
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    return time.time() - start, resp.usage.completion_tokens

prompts = [f"Explain concept #{i} in machine learning" for i in range(32)]
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
    results = list(pool.map(send_request, prompts))
total = time.time() - start

total_tokens = sum(t for _, t in results)
print(f"Throughput: {total_tokens/total:.0f} tok/s ({len(prompts)} requests in {total:.1f}s)")`,id:"code-client"}),e.jsx(s,{title:"vLLM Key Features",problem:"What makes vLLM the go-to production serving framework?",steps:[{formula:"PagedAttention: 2-4x throughput improvement",explanation:"Efficient KV-cache memory management eliminates waste."},{formula:"Continuous batching: near-100% GPU utilization",explanation:"New requests join the batch as old ones complete."},{formula:"Tensor parallelism: serve 70B+ models across GPUs",explanation:"Split model weights across multiple GPUs for large models."},{formula:"OpenAI-compatible API: drop-in replacement",explanation:"Works with any tool that supports the OpenAI API."}],id:"example-features"}),e.jsx(a,{type:"tip",title:"Prefix Caching",content:"Enable prefix caching with --enable-prefix-caching for workloads where many requests share the same system prompt. The KV-cache for the shared prefix is computed once and reused, reducing time-to-first-token by 50-90% for chat applications.",id:"note-prefix-caching"}),e.jsx(n,{title:"Memory Planning",content:"vLLM pre-allocates GPU memory based on --gpu-memory-utilization (default 0.9 = 90%). Set this lower if other processes share the GPU. The --max-model-len flag limits context length and thus KV-cache size. Start conservative and increase based on actual usage.",id:"warning-memory"})]})}const Xe=Object.freeze(Object.defineProperty({__proto__:null,default:Q},Symbol.toStringTag,{value:"Module"}));function Y(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Text Generation Inference (TGI)"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"TGI is HuggingFace's production-grade serving solution for large language models. Written in Rust for performance-critical paths, TGI provides continuous batching, tensor parallelism, quantization, and tight integration with the HuggingFace Hub. It powers HuggingFace's Inference Endpoints and is a battle-tested choice for deploying models at scale."}),e.jsx(o,{title:"Text Generation Inference (TGI)",definition:"TGI is an open-source toolkit from HuggingFace for deploying LLMs. It features a Rust-based token streaming server, Flash Attention, continuous batching, and native support for GPTQ/AWQ/EETQ quantization. TGI exposes both an OpenAI-compatible API and its own Messages API.",id:"def-tgi"}),e.jsx(t,{title:"Terminal",code:`# Launch TGI with Docker (recommended)
docker run --gpus all --shm-size 1g -p 8080:80 \\
    -v $HOME/.cache/huggingface:/data \\
    ghcr.io/huggingface/text-generation-inference:latest \\
    --model-id meta-llama/Llama-3.1-8B-Instruct \\
    --max-input-tokens 4096 \\
    --max-total-tokens 8192 \\
    --max-batch-prefill-tokens 4096

# Multi-GPU with tensor parallelism
docker run --gpus all --shm-size 1g -p 8080:80 \\
    -v $HOME/.cache/huggingface:/data \\
    ghcr.io/huggingface/text-generation-inference:latest \\
    --model-id meta-llama/Llama-3.1-70B-Instruct \\
    --num-shard 4 \\
    --quantize bitsandbytes-nf4

# Serve a GPTQ-quantized model
docker run --gpus all --shm-size 1g -p 8080:80 \\
    -v $HOME/.cache/huggingface:/data \\
    ghcr.io/huggingface/text-generation-inference:latest \\
    --model-id TheBloke/Llama-2-13B-chat-GPTQ \\
    --quantize gptq

# Test with curl (TGI native endpoint)
curl http://localhost:8080/generate \\
    -H "Content-Type: application/json" \\
    -d '{"inputs": "What is deep learning?", "parameters": {"max_new_tokens": 128}}'

# OpenAI-compatible endpoint
curl http://localhost:8080/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{
        "model": "tgi",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 128
    }'`,id:"code-tgi-launch"}),e.jsx(t,{title:"tgi_client.py",code:`import requests
from huggingface_hub import InferenceClient

# Option 1: HuggingFace InferenceClient (recommended)
client = InferenceClient("http://localhost:8080")

response = client.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain TGI in two sentences."},
    ],
    max_tokens=200,
    temperature=0.7,
)
print(response.choices[0].message.content)

# Streaming with InferenceClient
for token in client.chat_completion(
    messages=[{"role": "user", "content": "Count from 1 to 10"}],
    max_tokens=100,
    stream=True,
):
    delta = token.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()

# Option 2: OpenAI SDK (works with TGI's compatible endpoint)
from openai import OpenAI

openai_client = OpenAI(base_url="http://localhost:8080/v1", api_key="-")
resp = openai_client.chat.completions.create(
    model="tgi",
    messages=[{"role": "user", "content": "What is Flash Attention?"}],
    max_tokens=150,
)
print(resp.choices[0].message.content)

# Option 3: Direct HTTP for TGI-specific features
resp = requests.post("http://localhost:8080/generate", json={
    "inputs": "Translate to French: Hello, how are you?",
    "parameters": {
        "max_new_tokens": 64,
        "temperature": 0.3,
        "repetition_penalty": 1.1,
        "return_full_text": False,
    }
})
print(resp.json()["generated_text"])`,id:"code-tgi-client"}),e.jsx(s,{title:"TGI Configuration Options",problem:"What are the key TGI server parameters?",steps:[{formula:"--max-input-tokens 4096",explanation:"Maximum number of tokens in the input prompt."},{formula:"--max-total-tokens 8192",explanation:"Maximum total tokens (input + output) per request."},{formula:"--max-batch-prefill-tokens 4096",explanation:"Controls prefill batch size. Lower values reduce latency spikes."},{formula:"--num-shard 4",explanation:"Number of GPU shards for tensor parallelism."},{formula:"--quantize gptq | awq | bitsandbytes-nf4",explanation:"Quantization method to reduce memory usage."}],id:"example-config"}),e.jsx(a,{type:"tip",title:"TGI Health & Metrics",content:"TGI exposes a /health endpoint for load balancer health checks and a /metrics endpoint with Prometheus-compatible metrics including queue size, batch size, and inference latency. Use these to monitor and autoscale your deployment.",id:"note-metrics"}),e.jsx(n,{title:"Shared Memory Requirement",content:"TGI uses shared memory for inter-process communication. Always set --shm-size 1g (or higher) in Docker, otherwise the container will crash with 'Bus error' on multi-GPU setups. In Kubernetes, mount an emptyDir volume at /dev/shm with a sizeLimit.",id:"warning-shm"})]})}const Je=Object.freeze(Object.defineProperty({__proto__:null,default:Y},Symbol.toStringTag,{value:"Module"}));function X(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"TensorRT-LLM"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"TensorRT-LLM is NVIDIA's high-performance inference library that compiles LLM architectures into optimized TensorRT engines. It squeezes maximum throughput from NVIDIA GPUs through kernel fusion, quantization, in-flight batching, and custom CUDA kernels. When absolute performance on NVIDIA hardware matters, TensorRT-LLM is the gold standard."}),e.jsx(o,{title:"TensorRT-LLM",definition:"TensorRT-LLM is NVIDIA's open-source library for compiling and serving LLMs. It converts model weights into optimized TensorRT engines with fused kernels, FP8/INT4 quantization, paged KV-cache, in-flight batching, and multi-GPU tensor/pipeline parallelism. Models must be compiled before serving.",id:"def-tensorrt"}),e.jsx(t,{title:"Terminal",code:`# Pull the TensorRT-LLM Docker image
docker pull nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3

# Clone TensorRT-LLM for build scripts
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM/examples/llama

# Step 1: Convert HuggingFace checkpoint
python convert_checkpoint.py \\
    --model_dir meta-llama/Llama-3.1-8B-Instruct \\
    --output_dir ./checkpoint \\
    --dtype float16

# Step 2: Build the TensorRT engine
trtllm-build \\
    --checkpoint_dir ./checkpoint \\
    --output_dir ./engine \\
    --gemm_plugin float16 \\
    --max_input_len 4096 \\
    --max_seq_len 8192 \\
    --max_batch_size 64

# Step 3: Run with Triton Inference Server
docker run --gpus all -p 8000:8000 -p 8001:8001 \\
    -v $(pwd)/engine:/engines \\
    nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3 \\
    tritonserver --model-repository=/engines

# Quantize to FP8 for Hopper GPUs (H100)
python convert_checkpoint.py \\
    --model_dir meta-llama/Llama-3.1-8B-Instruct \\
    --output_dir ./checkpoint_fp8 \\
    --dtype float16 \\
    --qformat fp8 \\
    --calib_size 512`,id:"code-tensorrt-build"}),e.jsx(t,{title:"tensorrt_client.py",code:`import requests
import json

# Triton HTTP endpoint
TRITON_URL = "http://localhost:8000/v2/models/llama/generate"

def generate(prompt, max_tokens=128):
    """Send request to TensorRT-LLM via Triton."""
    payload = {
        "text_input": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": False,
    }
    resp = requests.post(TRITON_URL, json=payload)
    resp.raise_for_status()
    return resp.json()["text_output"]

result = generate("Explain TensorRT-LLM in two sentences.")
print(result)

# Streaming via Server-Sent Events
def stream_generate(prompt, max_tokens=128):
    payload = {
        "text_input": prompt,
        "max_tokens": max_tokens,
        "stream": True,
    }
    with requests.post(TRITON_URL, json=payload, stream=True) as resp:
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                print(data.get("text_output", ""), end="", flush=True)
    print()

stream_generate("List 5 benefits of model compilation.")`,id:"code-tensorrt-client"}),e.jsx(s,{title:"TensorRT-LLM Optimization Pipeline",problem:"What happens during TensorRT-LLM engine compilation?",steps:[{formula:"Weight conversion: HF -> TRT checkpoint",explanation:"Rearranges weights into the format TensorRT expects."},{formula:"Kernel fusion: combine multiple ops into one",explanation:"Fuses attention, LayerNorm, and activation into single GPU kernels."},{formula:"Quantization: FP16 -> FP8 / INT4-AWQ",explanation:"Reduces memory and increases compute throughput on supported hardware."},{formula:"Engine build: compile optimized execution plan",explanation:"Generates a GPU-specific binary optimized for the target hardware."}],id:"example-pipeline"}),e.jsx(a,{type:"note",title:"Performance Advantage",content:"TensorRT-LLM typically achieves 1.5-3x higher throughput than general-purpose frameworks on the same NVIDIA hardware. The tradeoff is a longer setup process: you must compile engines for each specific GPU architecture (e.g., A100 vs H100) and model configuration. Engines are not portable across GPU types.",id:"note-performance"}),e.jsx(n,{title:"Build Time & Complexity",content:"Engine compilation can take 10-60 minutes depending on model size. The build is tied to specific GPU architecture, max batch size, and sequence length. Changing any of these requires a rebuild. For rapid experimentation, start with vLLM or TGI and switch to TensorRT-LLM once your deployment configuration is stable.",id:"warning-complexity"})]})}const Ze=Object.freeze(Object.defineProperty({__proto__:null,default:X},Symbol.toStringTag,{value:"Module"}));function J(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"llama.cpp Server & koboldcpp"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"llama.cpp provides a lightweight, dependency-free HTTP server for serving GGUF models with an OpenAI-compatible API. It runs on CPUs, Apple Silicon, and NVIDIA/AMD GPUs without requiring heavy frameworks. koboldcpp extends this with a richer UI and API for creative writing and roleplay workloads."}),e.jsx(o,{title:"llama-server",definition:"llama-server (formerly llama.cpp server) is an HTTP server bundled with llama.cpp that serves GGUF-format models via an OpenAI-compatible REST API. It supports continuous batching, parallel requests, prompt caching, GPU offloading, grammar-constrained generation, and embedding extraction.",id:"def-llama-server"}),e.jsx(t,{title:"Terminal",code:`# Build llama.cpp with CUDA support
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release -j$(nproc)

# Start the server with a GGUF model
./bin/llama-server \\
    --model ~/models/Llama-3.1-8B-Instruct-Q4_K_M.gguf \\
    --host 0.0.0.0 \\
    --port 8080 \\
    --ctx-size 8192 \\
    --n-gpu-layers 99 \\
    --parallel 4 \\
    --cont-batching

# CPU-only (no GPU flags needed)
./bin/llama-server \\
    --model ~/models/Phi-3-mini-4k-instruct-Q4_K_M.gguf \\
    --host 0.0.0.0 \\
    --port 8080 \\
    --ctx-size 4096 \\
    --threads 8

# Test with curl (OpenAI-compatible)
curl http://localhost:8080/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 128,
        "temperature": 0.7
    }'

# Docker alternative
docker run -p 8080:8080 \\
    -v ~/models:/models \\
    ghcr.io/ggerganov/llama.cpp:server-cuda \\
    --model /models/Llama-3.1-8B-Instruct-Q4_K_M.gguf \\
    --n-gpu-layers 99 --ctx-size 8192 --parallel 4`,id:"code-llama-server"}),e.jsx(t,{title:"llamacpp_client.py",code:`from openai import OpenAI
import requests

# OpenAI SDK works directly with llama-server
client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")

response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain GGUF format briefly."},
    ],
    temperature=0.7,
    max_tokens=200,
)
print(response.choices[0].message.content)

# Grammar-constrained generation (JSON output)
resp = requests.post("http://localhost:8080/completion", json={
    "prompt": "List three colors as JSON:\\n",
    "n_predict": 128,
    "grammar": '''root ::= "{" ws "\\"colors\\"" ws ":" ws arr "}"
arr ::= "[" ws string ("," ws string)* ws "]"
string ::= "\\"" [a-zA-Z]+ "\\""
ws ::= " "?''',
})
print("Grammar-constrained:", resp.json()["content"])

# Generate embeddings
resp = requests.post("http://localhost:8080/v1/embeddings", json={
    "input": "Hello world",
    "model": "local-model",
})
embedding = resp.json()["data"][0]["embedding"]
print(f"Embedding dims: {len(embedding)}")

# Server health and slots info
health = requests.get("http://localhost:8080/health").json()
print(f"Server status: {health['status']}")`,id:"code-llamacpp-client"}),e.jsx(s,{title:"koboldcpp",problem:"How does koboldcpp extend llama.cpp?",steps:[{formula:"Built-in web UI for chat and story writing",explanation:"No separate frontend needed; includes KoboldAI Lite interface."},{formula:"KoboldAI API + OpenAI API compatibility",explanation:"Supports both API standards for broad client compatibility."},{formula:"One-click Windows/Linux/macOS launchers",explanation:"Pre-built binaries with GUI for model selection and GPU config."},{formula:"Multimodal support (LLaVA models)",explanation:"Serve vision-language models with image input support."}],id:"example-koboldcpp"}),e.jsx(a,{type:"tip",title:"Prompt Caching",content:"llama-server automatically caches prompt evaluations. When multiple requests share the same prefix (e.g., a system prompt), subsequent requests skip re-evaluating that prefix. This significantly reduces time-to-first-token for chat applications.",id:"note-prompt-cache"}),e.jsx(n,{title:"Parallel Slots",content:"The --parallel flag sets the number of concurrent requests. Each slot reserves context memory, so --parallel 4 --ctx-size 8192 allocates 4 x 8192 tokens of KV cache. This can quickly exceed GPU VRAM with large contexts. Monitor memory with --verbose and reduce parallel slots if you see OOM errors.",id:"warning-parallel"})]})}const et=Object.freeze(Object.defineProperty({__proto__:null,default:J},Symbol.toStringTag,{value:"Module"}));function Z(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"LiteLLM: Unified LLM Proxy"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"LiteLLM provides a unified OpenAI-compatible interface to over 100 LLM providers. It acts as a proxy layer that translates between the OpenAI API format and each provider's native API. This means you write your code once and can switch between OpenAI, Anthropic, local models, and dozens of other providers by changing a single model string."}),e.jsx(o,{title:"LiteLLM",definition:"LiteLLM is an open-source proxy server and Python SDK that provides a unified OpenAI-format API for 100+ LLM providers including OpenAI, Anthropic, Cohere, vLLM, Ollama, Azure, Bedrock, and Vertex AI. It handles authentication, retries, fallbacks, load balancing, spend tracking, and rate limiting.",id:"def-litellm"}),e.jsx(t,{title:"Terminal",code:`# Install LiteLLM
pip install 'litellm[proxy]'

# Start the proxy server with a config file
cat > litellm_config.yaml << 'EOF'
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-your-openai-key

  - model_name: claude
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: sk-ant-your-key

  - model_name: local-llama
    litellm_params:
      model: openai/meta-llama/Llama-3.1-8B-Instruct
      api_base: http://localhost:8000/v1
      api_key: none

  - model_name: ollama-model
    litellm_params:
      model: ollama/llama3.2
      api_base: http://localhost:11434

  # Load balancing: multiple deployments for one model name
  - model_name: fast-model
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: sk-key1
  - model_name: fast-model
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: sk-ant-key2

litellm_settings:
  drop_params: true
  set_verbose: false

general_settings:
  master_key: sk-litellm-master-key
EOF

litellm --config litellm_config.yaml --port 4000

# Docker deployment
docker run -p 4000:4000 \\
    -v $(pwd)/litellm_config.yaml:/app/config.yaml \\
    ghcr.io/berriai/litellm:main-latest \\
    --config /app/config.yaml`,id:"code-litellm-setup"}),e.jsx(t,{title:"litellm_client.py",code:`from openai import OpenAI
import litellm

# Option 1: Use as a proxy (any OpenAI SDK client works)
client = OpenAI(
    base_url="http://localhost:4000",
    api_key="sk-litellm-master-key",
)

# Route to different backends by model name
for model in ["gpt-4", "claude", "local-llama"]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10,
    )
    print(f"{model}: {resp.choices[0].message.content}")

# Option 2: Use as a Python SDK (no proxy needed)
response = litellm.completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "What is LiteLLM?"}],
    max_tokens=100,
)
print(response.choices[0].message.content)

# Fallback chain: try providers in order
response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    fallbacks=["anthropic/claude-sonnet-4-20250514", "ollama/llama3.2"],
    max_tokens=50,
)
print(response.choices[0].message.content)

# Spend tracking
from litellm import budget_manager
cost = litellm.completion_cost(completion_response=response)
print(f"Request cost: USD {cost:.6f}")`,id:"code-litellm-client"}),e.jsx(s,{title:"LiteLLM Key Features",problem:"What problems does LiteLLM solve?",steps:[{formula:"Unified API: one interface for 100+ providers",explanation:"Write code once, switch providers by changing the model string."},{formula:"Load balancing: distribute across deployments",explanation:"Multiple backends for the same model name; LiteLLM round-robins."},{formula:"Fallbacks: automatic failover on errors",explanation:"If one provider fails, transparently retry with another."},{formula:"Spend tracking: per-key and per-user budgets",explanation:"Track costs across all providers in one dashboard."},{formula:"Virtual keys: issue API keys with limits",explanation:"Create team/user keys with rate limits and budget caps."}],id:"example-features"}),e.jsx(a,{type:"tip",title:"Virtual Keys for Teams",content:"LiteLLM can issue virtual API keys with per-key rate limits, model access controls, and budget limits. This lets you give each team or application its own key while routing through your centralized proxy. Generate keys via the /key/generate API endpoint.",id:"note-virtual-keys"}),e.jsx(n,{title:"Parameter Translation",content:"Not all parameters translate perfectly between providers. For example, tool calling syntax differs between OpenAI and Anthropic. Enable drop_params: true in config to silently drop unsupported parameters rather than erroring. Test critical features with each backend before relying on automatic translation.",id:"warning-params"})]})}const tt=Object.freeze(Object.defineProperty({__proto__:null,default:Z},Symbol.toStringTag,{value:"Module"}));function ee(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Comparing Serving Frameworks"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Choosing the right serving framework depends on your hardware, model, latency requirements, and operational constraints. This section compares the key frameworks across the dimensions that matter most for production deployments."}),e.jsx(o,{title:"Serving Framework Selection Criteria",definition:"The key dimensions for comparing LLM serving frameworks are: throughput (tokens/second), latency (time-to-first-token and inter-token latency), hardware support (NVIDIA, AMD, CPU, Apple Silicon), model format support (HuggingFace, GGUF, TensorRT engines), ease of deployment, and API compatibility.",id:"def-criteria"}),e.jsx(s,{title:"Framework Comparison Matrix",problem:"How do the major frameworks compare?",steps:[{formula:"vLLM: Best general-purpose GPU serving",explanation:"Highest throughput for most models on NVIDIA GPUs. Easy setup. PagedAttention + continuous batching. OpenAI API."},{formula:"TGI: Best HuggingFace integration",explanation:"Tight Hub integration. Rust performance. Production-proven at HuggingFace scale. Good quantization support."},{formula:"TensorRT-LLM: Maximum NVIDIA performance",explanation:"1.5-3x faster than vLLM on same hardware. Requires compilation step. Best for stable, high-volume deployments."},{formula:"llama.cpp: Best for CPU / edge / Mac",explanation:"Runs anywhere. GGUF quantized models. Minimal dependencies. Great for local and low-resource deployments."},{formula:"LiteLLM: Best proxy / multi-provider gateway",explanation:"Not a serving engine itself. Unified API across 100+ providers. Load balancing and cost tracking."}],id:"example-matrix"}),e.jsx(t,{title:"benchmark_frameworks.py",code:`import requests
import time
import concurrent.futures
import json

def benchmark_endpoint(url, model, num_requests=50, max_tokens=128):
    """Benchmark an OpenAI-compatible endpoint."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Write a short paragraph about AI."}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    headers = {"Content-Type": "application/json"}

    latencies = []
    total_tokens = 0

    def send_one(_):
        start = time.time()
        resp = requests.post(url, json=payload, headers=headers)
        elapsed = time.time() - start
        data = resp.json()
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return elapsed, tokens

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(send_one, range(num_requests)))
    wall_time = time.time() - start

    for latency, tokens in results:
        latencies.append(latency)
        total_tokens += tokens

    latencies.sort()
    return {
        "total_time": round(wall_time, 2),
        "throughput_tok_s": round(total_tokens / wall_time, 1),
        "avg_latency": round(sum(latencies) / len(latencies), 3),
        "p50_latency": round(latencies[len(latencies)//2], 3),
        "p99_latency": round(latencies[int(len(latencies)*0.99)], 3),
        "total_tokens": total_tokens,
    }

# Compare frameworks (adjust URLs and models)
frameworks = {
    "vLLM":      ("http://localhost:8000/v1/chat/completions", "llama-3.1-8b"),
    "TGI":       ("http://localhost:8080/v1/chat/completions", "tgi"),
    "llama.cpp": ("http://localhost:8081/v1/chat/completions", "local"),
}

for name, (url, model) in frameworks.items():
    try:
        result = benchmark_endpoint(url, model, num_requests=50)
        print(f"\\n{name}:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"\\n{name}: unavailable ({e})")`,id:"code-benchmark"}),e.jsx(t,{title:"Terminal",code:`# Quick benchmarks with curl timing
# Time-to-first-token (TTFT) measurement
echo "--- vLLM TTFT ---"
curl -w "TTFT: %{time_starttransfer}s Total: %{time_total}s\\n" -s -o /dev/null \\
    http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"model":"llama","messages":[{"role":"user","content":"Hi"}],"max_tokens":1}'

echo "--- TGI TTFT ---"
curl -w "TTFT: %{time_starttransfer}s Total: %{time_total}s\\n" -s -o /dev/null \\
    http://localhost:8080/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"model":"tgi","messages":[{"role":"user","content":"Hi"}],"max_tokens":1}'

# Use wrk or hey for load testing
# hey -n 200 -c 32 -m POST \\
#   -H "Content-Type: application/json" \\
#   -d '{"model":"llama","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}' \\
#   http://localhost:8000/v1/chat/completions`,id:"code-curl-bench"}),e.jsx(a,{type:"intuition",title:"Decision Flowchart",content:"Start with this: (1) Need to run on CPU or Mac? Use llama.cpp. (2) Need multi-provider routing? Use LiteLLM as a proxy in front of any engine. (3) Deploying on NVIDIA GPUs with stable config? Try TensorRT-LLM for max performance. (4) Want the easiest GPU deployment? Use vLLM. (5) Already in the HuggingFace ecosystem? TGI integrates seamlessly.",id:"note-flowchart"}),e.jsx(n,{title:"Benchmarks Depend on Context",content:"Published benchmarks often use specific batch sizes, sequence lengths, and hardware that may not match your workload. Always benchmark with your actual model, expected request patterns, and hardware. A framework that wins on throughput may lose on latency, and vice versa.",id:"warning-benchmarks"})]})}const at=Object.freeze(Object.defineProperty({__proto__:null,default:ee},Symbol.toStringTag,{value:"Module"}));function te(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"OpenAI-Compatible APIs"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The OpenAI API format has become the de facto standard for LLM APIs. Nearly every serving framework (vLLM, TGI, llama.cpp, Ollama, LiteLLM) exposes endpoints that accept the same JSON schema as OpenAI. This means code written for the OpenAI SDK works with local models by changing just the base URL."}),e.jsx(o,{title:"OpenAI-Compatible API",definition:"An OpenAI-compatible API implements the same HTTP endpoints and JSON schema as OpenAI's Chat Completions API. The key endpoints are POST /v1/chat/completions for chat, POST /v1/completions for text completion, POST /v1/embeddings for embeddings, and GET /v1/models for listing available models.",id:"def-openai-compat"}),e.jsx(t,{title:"openai_compatible.py",code:`from openai import OpenAI

# The same code works with ANY OpenAI-compatible backend
# Just change base_url and api_key

backends = {
    "openai":    {"base_url": "https://api.openai.com/v1",   "api_key": "sk-..."},
    "vllm":      {"base_url": "http://localhost:8000/v1",     "api_key": "EMPTY"},
    "tgi":       {"base_url": "http://localhost:8080/v1",     "api_key": "EMPTY"},
    "ollama":    {"base_url": "http://localhost:11434/v1",    "api_key": "ollama"},
    "llamacpp":  {"base_url": "http://localhost:8081/v1",     "api_key": "EMPTY"},
    "litellm":   {"base_url": "http://localhost:4000",        "api_key": "sk-master"},
}

def query_backend(name, config, model="default"):
    client = OpenAI(**config)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What are you?"},
            ],
            max_tokens=50,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Query each available backend
models = {
    "openai": "gpt-4o-mini", "vllm": "llama-3.1-8b",
    "tgi": "tgi", "ollama": "llama3.2",
    "llamacpp": "local", "litellm": "gpt-4",
}
for name, config in backends.items():
    result = query_backend(name, config, models.get(name, "default"))
    print(f"{name:>10}: {result[:80]}")`,id:"code-backends"}),e.jsx(t,{title:"Terminal",code:`# The Chat Completions request format (works everywhere)
curl http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -H "Authorization: Bearer EMPTY" \\
    -d '{
        "model": "llama-3.1-8b",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ],
        "temperature": 0.7,
        "max_tokens": 128,
        "top_p": 0.95,
        "stream": false
    }'

# List available models
curl http://localhost:8000/v1/models \\
    -H "Authorization: Bearer EMPTY"

# Embeddings endpoint
curl http://localhost:8000/v1/embeddings \\
    -H "Content-Type: application/json" \\
    -d '{"model": "llama-3.1-8b", "input": "Hello world"}'`,id:"code-curl-api"}),e.jsx(s,{title:"Standard API Endpoints",problem:"What endpoints does an OpenAI-compatible server expose?",steps:[{formula:"POST /v1/chat/completions",explanation:"Chat-style completion with messages array. The most commonly used endpoint."},{formula:"POST /v1/completions",explanation:"Legacy text completion with a single prompt string."},{formula:"POST /v1/embeddings",explanation:"Generate vector embeddings for text. Not all servers support this."},{formula:"GET /v1/models",explanation:"List available models. Returns model IDs you can use in requests."}],id:"example-endpoints"}),e.jsx(a,{type:"note",title:"Compatibility Gaps",content:"While the core chat completions format is well-standardized, advanced features like function/tool calling, JSON mode, logprobs, and vision inputs vary across implementations. vLLM and LiteLLM have the broadest feature coverage. Always test specific features you rely on against your chosen backend.",id:"note-gaps"}),e.jsx(n,{title:"Model Names Are Not Standardized",content:"Each backend uses different model name conventions. OpenAI uses 'gpt-4o', vLLM uses the HuggingFace path, Ollama uses short names like 'llama3.2', and TGI uses 'tgi'. When switching backends, you must also update model names. LiteLLM solves this by letting you define custom model aliases.",id:"warning-model-names"})]})}const ot=Object.freeze(Object.defineProperty({__proto__:null,default:te},Symbol.toStringTag,{value:"Module"}));function ae(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Streaming & Server-Sent Events"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"LLMs generate tokens one at a time, and streaming delivers each token to the client as soon as it is produced rather than waiting for the full response. This dramatically reduces perceived latency: users see the first token in milliseconds instead of waiting seconds for the complete answer. Streaming uses the Server-Sent Events (SSE) protocol over HTTP."}),e.jsx(o,{title:"Server-Sent Events (SSE)",definition:"Server-Sent Events is an HTTP protocol where the server sends a stream of text events over a single long-lived connection. Each event is prefixed with 'data: ' and separated by double newlines. The stream ends with a 'data: [DONE]' sentinel. LLM APIs use SSE to stream individual token chunks as they are generated.",id:"def-sse"}),e.jsx(t,{title:"streaming_client.py",code:`from openai import OpenAI
import time

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# Streaming with the OpenAI SDK
print("=== Streaming Response ===")
start = time.time()
first_token_time = None

stream = client.chat.completions.create(
    model="llama-3.1-8b",
    messages=[{"role": "user", "content": "Explain streaming in 3 sentences."}],
    max_tokens=200,
    stream=True,
)

full_response = ""
for chunk in stream:
    if chunk.choices[0].delta.content:
        token = chunk.choices[0].delta.content
        if first_token_time is None:
            first_token_time = time.time() - start
        full_response += token
        print(token, end="", flush=True)

total_time = time.time() - start
print(f"\\n\\nTime to first token: {first_token_time*1000:.0f}ms")
print(f"Total time: {total_time*1000:.0f}ms")
print(f"Tokens: ~{len(full_response.split())}")

# Non-streaming comparison
start = time.time()
response = client.chat.completions.create(
    model="llama-3.1-8b",
    messages=[{"role": "user", "content": "Explain streaming in 3 sentences."}],
    max_tokens=200,
    stream=False,
)
print(f"\\nNon-streaming total: {(time.time()-start)*1000:.0f}ms")
print("(User sees nothing until this moment)")`,id:"code-streaming-client"}),e.jsx(t,{title:"raw_sse_client.py",code:`import requests
import json

# Raw SSE parsing (useful for understanding the protocol)
url = "http://localhost:8000/v1/chat/completions"
payload = {
    "model": "llama-3.1-8b",
    "messages": [{"role": "user", "content": "Count to 5."}],
    "max_tokens": 100,
    "stream": True,
}

response = requests.post(url, json=payload, stream=True,
                          headers={"Content-Type": "application/json"})

print("=== Raw SSE Events ===")
for line in response.iter_lines(decode_unicode=True):
    if not line:
        continue
    if line.startswith("data: "):
        data = line[6:]  # Strip "data: " prefix
        if data == "[DONE]":
            print("\\n[Stream complete]")
            break
        chunk = json.loads(data)
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            print(delta["content"], end="", flush=True)

# FastAPI server with SSE (building your own streaming proxy)
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import httpx, asyncio

app = FastAPI()

@app.post("/proxy/chat")
async def proxy_chat(request: dict):
    async def stream_response():
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", "http://localhost:8000/v1/chat/completions",
                                      json={**request, "stream": True}) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        yield f"{line}\\n\\n"
    return StreamingResponse(stream_response(), media_type="text/event-stream")
"""`,id:"code-raw-sse"}),e.jsx(s,{title:"Streaming vs Non-Streaming",problem:"When should you use streaming?",steps:[{formula:"Interactive chat: always stream",explanation:"Users perceive the response as faster when tokens appear immediately."},{formula:"Batch processing: do not stream",explanation:"Non-streaming is simpler to parse and has slightly less overhead."},{formula:"Tool/function calling: stream with care",explanation:"Function call arguments arrive incrementally; accumulate before parsing JSON."},{formula:"Proxy/middleware: forward the stream",explanation:"Pass SSE through without buffering to preserve low latency."}],id:"example-when-stream"}),e.jsx(a,{type:"intuition",title:"Why Streaming Feels Faster",content:"A 200-token response at 50 tok/s takes 4 seconds. Without streaming, the user stares at a blank screen for 4 seconds. With streaming, they see the first token after ~100ms (prefill time) and can start reading immediately. The total wall-clock time is the same, but perceived latency drops from 4 seconds to 100ms.",id:"note-perception"}),e.jsx(n,{title:"Buffering Breaks Streaming",content:"Reverse proxies (nginx, Cloudflare) and load balancers may buffer SSE responses, defeating the purpose of streaming. Configure proxy_buffering off in nginx, and disable response buffering in your CDN. Also set appropriate headers: Cache-Control: no-cache, X-Accel-Buffering: no.",id:"warning-buffering"})]})}const nt=Object.freeze(Object.defineProperty({__proto__:null,default:ae},Symbol.toStringTag,{value:"Module"}));function oe(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Rate Limiting & Load Balancing"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Production LLM deployments must handle variable traffic without letting any single user overwhelm the system. Rate limiting controls how many requests each client can make, while load balancing distributes requests across multiple model replicas to maximize throughput and availability."}),e.jsx(o,{title:"Rate Limiting",definition:"Rate limiting restricts the number of API requests a client can make within a time window. For LLM APIs, limits are typically expressed as requests per minute (RPM) and tokens per minute (TPM). The token-based limit is critical because a single request generating 4096 tokens consumes far more GPU time than one generating 10 tokens.",id:"def-rate-limiting"}),e.jsx(t,{title:"Terminal",code:`# Nginx rate limiting for an LLM API
cat > /etc/nginx/conf.d/llm-proxy.conf << 'EOF'
# Define rate limit zones
limit_req_zone $binary_remote_addr zone=llm_per_ip:10m rate=10r/s;
limit_req_zone $http_authorization zone=llm_per_key:10m rate=30r/s;

upstream llm_backends {
    least_conn;                    # Route to least-busy server
    server 10.0.0.1:8000 weight=3; # 3x A100 GPU
    server 10.0.0.2:8000 weight=1; # 1x A100 GPU
    server 10.0.0.3:8000 weight=1 backup; # Failover only
}

server {
    listen 443 ssl;
    server_name api.example.com;

    # Rate limit: 10 req/s per IP, burst up to 20
    location /v1/ {
        limit_req zone=llm_per_ip burst=20 nodelay;
        limit_req zone=llm_per_key burst=50 nodelay;

        # Disable buffering for streaming
        proxy_buffering off;
        proxy_set_header X-Accel-Buffering no;

        proxy_pass http://llm_backends;
        proxy_read_timeout 300s;  # LLM responses can be slow
    }
}
EOF

# Reload nginx
nginx -t && nginx -s reload`,id:"code-nginx"}),e.jsx(t,{title:"rate_limiter.py",code:`import time
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class TokenBucket:
    """Token bucket rate limiter for LLM APIs."""
    rpm: int = 60          # Requests per minute
    tpm: int = 100_000     # Tokens per minute
    _requests: list = field(default_factory=list)
    _tokens: list = field(default_factory=list)

    def _cleanup(self, bucket, window=60):
        now = time.time()
        while bucket and bucket[0] < now - window:
            bucket.pop(0)

    def check(self, estimated_tokens=500):
        """Check if request is allowed. Returns (allowed, retry_after_s)."""
        now = time.time()
        self._cleanup(self._requests)
        self._cleanup(self._tokens)
        if len(self._requests) >= self.rpm:
            retry = self._requests[0] + 60 - now
            return False, retry
        token_sum = sum(t for _, t in self._tokens) if self._tokens else 0
        if token_sum + estimated_tokens > self.tpm:
            return False, 5.0
        return True, 0

    def record(self, tokens_used):
        now = time.time()
        self._requests.append(now)
        self._tokens.append((now, tokens_used))

# Per-API-key rate limiting
limiters = defaultdict(lambda: TokenBucket(rpm=30, tpm=50_000))

def handle_request(api_key, estimated_tokens=500):
    limiter = limiters[api_key]
    allowed, retry_after = limiter.check(estimated_tokens)
    if not allowed:
        print(f"Rate limited: retry after {retry_after:.1f}s")
        return 429, {"retry_after": retry_after}
    # ... process request ...
    tokens_used = 150  # actual tokens from response
    limiter.record(tokens_used)
    return 200, {"tokens": tokens_used}

# Simulate traffic
for i in range(40):
    status, data = handle_request("key-user-1")
    print(f"Request {i+1}: {status} {data}")`,id:"code-rate-limiter"}),e.jsx(s,{title:"Load Balancing Strategies",problem:"How should you distribute requests across LLM replicas?",steps:[{formula:"Least-connections: route to least-busy server",explanation:"Best for LLMs since request durations vary widely based on output length."},{formula:"Weighted round-robin: allocate by GPU capacity",explanation:"Give more traffic to servers with more/faster GPUs."},{formula:"Session affinity: same user to same server",explanation:"Helps with prefix caching since the KV cache stays on one server."},{formula:"Health-check based: skip unhealthy servers",explanation:"Use /health endpoints to detect and route around GPU OOM or crashes."}],id:"example-lb"}),e.jsx(a,{type:"tip",title:"Client-Side Rate Limit Handling",content:"Well-behaved clients should respect 429 responses and Retry-After headers. The OpenAI Python SDK has built-in retry logic with exponential backoff. When building clients for rate-limited APIs, implement backoff: wait, double the wait on each retry, add jitter, and cap at a maximum delay.",id:"note-client-retry"}),e.jsx(n,{title:"Token Counting Before Generation",content:"Rate limiting by tokens is challenging because you do not know the output token count before generation starts. Use the max_tokens request parameter as an upper bound estimate, or implement post-hoc accounting that deducts actual tokens used after the response completes. Over-estimating is safer than under-estimating.",id:"warning-token-counting"})]})}const st=Object.freeze(Object.defineProperty({__proto__:null,default:oe},Symbol.toStringTag,{value:"Module"}));function ne(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Cost Optimization Strategies"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"LLM inference costs can escalate quickly, whether you are paying per-token to a cloud API or operating your own GPU fleet. Effective cost optimization combines prompt engineering, caching, model selection, and infrastructure tuning to reduce spend while maintaining output quality."}),e.jsx(o,{title:"Cost Per Token",definition:"LLM API costs are measured in dollars per million tokens, with input (prompt) tokens typically 3-10x cheaper than output (completion) tokens. For self-hosted models, cost is GPU-hours divided by tokens served. A well-optimized self-hosted deployment can cost $0.01-0.05 per million tokens compared to $0.50-15.00 for cloud APIs.",id:"def-cost-per-token"}),e.jsx(s,{title:"Cost Optimization Strategies",problem:"How can you reduce LLM inference costs?",steps:[{formula:"Prompt caching: reuse common prefixes",explanation:"Cached prompt tokens cost 50-90% less. System prompts and few-shot examples are prime candidates."},{formula:"Model routing: use smaller models for easy tasks",explanation:"Route simple queries to a 7B model and only use 70B+ for complex reasoning."},{formula:"Output length control: set tight max_tokens",explanation:"Every unnecessary output token costs money. Instruct the model to be concise."},{formula:"Semantic caching: cache similar query results",explanation:"If a nearly identical question was recently answered, return the cached response."},{formula:"Batch processing: use offline batch APIs",explanation:"OpenAI Batch API and similar offer 50% discounts for non-real-time workloads."}],id:"example-strategies"}),e.jsx(t,{title:"cost_tracker.py",code:`import time
from dataclasses import dataclass, field
from collections import defaultdict

# Pricing per million tokens (example rates)
PRICING = {
    "gpt-4o":       {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":  {"input": 0.15, "output": 0.60},
    "claude-sonnet": {"input": 3.00, "output": 15.00},
    "llama-3.1-8b": {"input": 0.05, "output": 0.05},  # Self-hosted estimate
    "llama-3.1-70b":{"input": 0.20, "output": 0.20},  # Self-hosted estimate
}

@dataclass
class CostTracker:
    costs: list = field(default_factory=list)
    by_model: dict = field(default_factory=lambda: defaultdict(float))

    def record(self, model, input_tokens, output_tokens):
        pricing = PRICING.get(model, {"input": 1.0, "output": 3.0})
        cost = (input_tokens * pricing["input"] +
                output_tokens * pricing["output"]) / 1_000_000
        self.costs.append({"model": model, "cost": cost, "time": time.time()})
        self.by_model[model] += cost
        return cost

    def total(self):
        return sum(c["cost"] for c in self.costs)

    def report(self):
        print(f"Total cost: USD {self.total():.4f}")
        print(f"Requests: {len(self.costs)}")
        for model, cost in sorted(self.by_model.items(), key=lambda x: -x[1]):
            print(f"  {model}: USD {cost:.4f}")

tracker = CostTracker()

# Simulate a day of usage
import random
for _ in range(1000):
    model = random.choice(["gpt-4o", "gpt-4o-mini", "llama-3.1-8b"])
    tracker.record(model, random.randint(100, 2000), random.randint(50, 500))

tracker.report()`,id:"code-cost-tracker"}),e.jsx(t,{title:"model_router.py",code:`from openai import OpenAI

# Route queries to the cheapest capable model
class ModelRouter:
    def __init__(self, clients):
        self.clients = clients  # model_name -> OpenAI client

    def classify_complexity(self, messages):
        """Simple heuristic: short queries -> small model."""
        last_msg = messages[-1]["content"]
        word_count = len(last_msg.split())
        # Complex indicators: code, math, multi-step reasoning
        complex_words = ["analyze", "compare", "implement", "debug",
                         "explain why", "step by step", "code"]
        is_complex = any(w in last_msg.lower() for w in complex_words)
        if word_count > 200 or is_complex:
            return "complex"
        return "simple"

    def route(self, messages, **kwargs):
        complexity = self.classify_complexity(messages)
        if complexity == "simple":
            model, client = "llama-3.1-8b", self.clients["small"]
        else:
            model, client = "gpt-4o", self.clients["large"]

        print(f"Routing to {model} (complexity={complexity})")
        return client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

# Setup
router = ModelRouter({
    "small": OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY"),
    "large": OpenAI(api_key="sk-your-key"),
})

# Simple query -> routed to cheap local model
resp = router.route(
    [{"role": "user", "content": "What is 2+2?"}],
    max_tokens=50,
)
print(resp.choices[0].message.content)

# Complex query -> routed to powerful cloud model
resp = router.route(
    [{"role": "user", "content": "Analyze this code and explain the bug step by step: ..."}],
    max_tokens=500,
)
print(resp.choices[0].message.content)`,id:"code-model-router"}),e.jsx(a,{type:"tip",title:"Semantic Caching",content:"Tools like GPTCache or Redis with vector search can cache LLM responses keyed by semantic similarity rather than exact match. When a new query is sufficiently similar to a cached one (cosine similarity > 0.95), return the cached response instantly at zero token cost. This works exceptionally well for FAQ-style workloads.",id:"note-semantic-cache"}),e.jsx(n,{title:"Quality vs Cost Tradeoff",content:"Aggressive cost optimization can degrade output quality. Always measure quality metrics (accuracy, helpfulness ratings, task success rate) alongside cost. A model that costs 10x less but fails 30% of the time may actually cost more when you factor in retries, user frustration, and downstream errors.",id:"warning-quality"})]})}const it=Object.freeze(Object.defineProperty({__proto__:null,default:ne},Symbol.toStringTag,{value:"Module"}));export{Ue as A,Ge as B,Re as C,Ce as D,qe as E,Ne as F,Be as G,ze as H,De as I,We as J,Fe as K,He as L,Ke as M,$e as N,Ve as O,Qe as P,Ye as Q,Xe as R,Je as S,Ze as T,et as U,tt as V,at as W,ot as X,nt as Y,st as Z,it as _,me as a,pe as b,ce as c,de as d,ue as e,he as f,ge as g,fe as h,ye as i,xe as j,be as k,_e as l,ve as m,we as n,ke as o,Ae as p,Te as q,Oe as r,le as s,Ie as t,Se as u,Pe as v,Le as w,je as x,Me as y,Ee as z};
