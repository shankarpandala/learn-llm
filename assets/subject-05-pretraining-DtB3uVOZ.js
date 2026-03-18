import{j as e}from"./vendor-DWbzdFaj.js";import{D as t,N as i,E as n,P as a,T as s,W as o}from"./subject-01-text-fundamentals-DG6tAvii.js";import"./vendor-katex-BYl39Yo6.js";function r(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"BERT: Bidirectional Encoder Representations from Transformers"}),e.jsx("p",{className:"text-lg text-gray-300",children:"BERT (Devlin et al., 2018) introduced deep bidirectional pretraining for language understanding. Unlike left-to-right models, BERT conditions on both left and right context simultaneously, enabling richer representations for downstream tasks."}),e.jsx(t,{title:"BERT Architecture",definition:"BERT is a multi-layer bidirectional Transformer encoder. BERT-Base has $L=12$ layers, $H=768$ hidden dimensions, and $A=12$ attention heads (110M parameters). BERT-Large uses $L=24$, $H=1024$, $A=16$ (340M parameters).",notation:"$\\text{BERT}(x) = \\text{TransformerEncoder}(E(x) + P(x) + S(x))$ where $E$ is token embedding, $P$ is position embedding, $S$ is segment embedding.",id:"bert-arch-def"}),e.jsx(i,{type:"historical",title:"Why Bidirectional Matters",content:"Before BERT, models like ELMo used separately trained left-to-right and right-to-left LSTMs. GPT used unidirectional (left-to-right) Transformers. BERT showed that jointly conditioning on both directions in every layer leads to substantially better representations.",id:"bidirectional-note"}),e.jsx(n,{title:"BERT Input Representation",problem:"How does BERT represent the input '[CLS] The cat sat [SEP] It was tired [SEP]'?",steps:[{formula:"E_{\\text{input}} = E_{\\text{token}} + E_{\\text{segment}} + E_{\\text{position}}",explanation:"Each input token gets three embeddings summed together."},{formula:"E_{\\text{segment}} \\in \\{E_A, E_B\\}",explanation:"Segment A for first sentence, Segment B for second sentence."},{formula:"E_{\\text{position}} \\in \\mathbb{R}^{512 \\times H}",explanation:"Learned positional embeddings support sequences up to 512 tokens."}],id:"bert-input-example"}),e.jsx(a,{title:"bert_architecture.py",code:`from transformers import BertModel, BertTokenizer, BertConfig
import torch

# Load pretrained BERT-Base
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Inspect architecture
config = model.config
print(f"Layers: {config.num_hidden_layers}")        # 12
print(f"Hidden size: {config.hidden_size}")           # 768
print(f"Attention heads: {config.num_attention_heads}")  # 12
print(f"Vocab size: {config.vocab_size}")             # 30522
print(f"Max position: {config.max_position_embeddings}")  # 512
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Encode a sentence pair
inputs = tokenizer(
    "The cat sat on the mat.",
    "It was very comfortable.",
    return_tensors="pt",
    padding=True
)
print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Token type IDs: {inputs['token_type_ids']}")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state: [batch, seq_len, hidden]
# outputs.pooler_output: [batch, hidden] (CLS representation)
print(f"Last hidden state: {outputs.last_hidden_state.shape}")
print(f"Pooler output: {outputs.pooler_output.shape}")

# Access individual layer outputs
outputs_all = model(**inputs, output_hidden_states=True)
print(f"Number of hidden states: {len(outputs_all.hidden_states)}")  # 13 (embed + 12 layers)`,id:"bert-code"}),e.jsx(s,{title:"BERT Pretraining Objectives",statement:"BERT is pretrained with two objectives: (1) Masked Language Modeling (MLM) and (2) Next Sentence Prediction (NSP). The total loss is $\\mathcal{L} = \\mathcal{L}_{\\text{MLM}} + \\mathcal{L}_{\\text{NSP}}$.",proof:"MLM enables bidirectional context by masking 15% of tokens and predicting them. NSP trains binary classification on whether sentence B follows sentence A. Together they produce representations useful for both token-level and sentence-level tasks.",id:"bert-objectives-thm"}),e.jsx(o,{title:"BERT Is Not a Generative Model",content:"BERT cannot generate text autoregressively. Its bidirectional attention means every token sees every other token during encoding. For generation tasks, use decoder-only (GPT) or encoder-decoder (T5) architectures.",id:"bert-not-generative"})]})}const S=Object.freeze(Object.defineProperty({__proto__:null,default:r},Symbol.toStringTag,{value:"Module"}));function l(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Masked Language Modeling (MLM)"}),e.jsx("p",{className:"text-lg text-gray-300",children:"Masked Language Modeling is BERT's primary pretraining objective. By randomly masking tokens and training the model to predict them from bidirectional context, MLM learns deep contextual representations without the constraint of unidirectional attention."}),e.jsx(t,{title:"MLM Objective",definition:"Given a sequence $x = (x_1, \\ldots, x_n)$, randomly select 15% of positions $\\mathcal{M}$. The MLM loss is $\\mathcal{L}_{\\text{MLM}} = -\\sum_{i \\in \\mathcal{M}} \\log P(x_i \\mid x_{\\backslash \\mathcal{M}})$, where $x_{\\backslash \\mathcal{M}}$ denotes the corrupted input.",notation:"The masking follows: 80% replaced with [MASK], 10% replaced with a random token, 10% kept unchanged.",id:"mlm-def"}),e.jsx(s,{title:"MLM Loss as Cross-Entropy",statement:"For each masked position $i \\in \\mathcal{M}$, the model outputs logits $z_i \\in \\mathbb{R}^{|V|}$ over the vocabulary. The per-token MLM loss is the standard cross-entropy: $\\mathcal{L}_i = -\\log \\frac{\\exp(z_i[x_i])}{\\sum_{v \\in V} \\exp(z_i[v])}$.",proof:"The total MLM loss averages over all masked positions: $\\mathcal{L}_{\\text{MLM}} = \\frac{1}{|\\mathcal{M}|} \\sum_{i \\in \\mathcal{M}} \\mathcal{L}_i$. This is equivalent to categorical cross-entropy between the predicted distribution and the one-hot target.",id:"mlm-loss-theorem"}),e.jsx(n,{title:"Masking Procedure",problem:"Given 'The cat sat on the mat', apply 15% masking with the 80/10/10 rule.",steps:[{formula:'\\text{Selected position: } i = 2 \\text{ ("cat")}',explanation:"15% of 6 tokens rounds to ~1 token selected for masking."},{formula:"P(\\text{[MASK]}) = 0.8, \\; P(\\text{random}) = 0.1, \\; P(\\text{keep}) = 0.1",explanation:'With 80% probability: "The [MASK] sat on the mat". With 10%: "The dog sat on the mat". With 10%: unchanged.'},{formula:'\\text{Target: predict } x_2 = \\text{"cat"}',explanation:"Regardless of the corruption strategy, the model must recover the original token."}],id:"masking-example"}),e.jsx(i,{type:"intuition",title:"Why the 80/10/10 Split?",content:"If we always used [MASK], the model would never see [MASK] during fine-tuning, creating a pretrain-finetune mismatch. Keeping 10% unchanged and 10% random forces the model to maintain good representations for all positions, not just masked ones.",id:"masking-split-note"}),e.jsx(a,{title:"mlm_training.py",code:`from transformers import (
    BertTokenizer, BertForMaskedLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from datasets import load_dataset
import torch

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Manual MLM example
text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"].clone()

# Mask "cat" (token index 2)
masked_ids = input_ids.clone()
masked_ids[0, 2] = tokenizer.mask_token_id  # [MASK] = 103

# Forward pass
with torch.no_grad():
    outputs = model(input_ids=masked_ids)
    logits = outputs.logits  # [1, seq_len, vocab_size]

# Prediction at masked position
predicted_id = logits[0, 2].argmax(dim=-1)
print(f"Original: {tokenizer.decode(input_ids[0, 2])}")
print(f"Predicted: {tokenizer.decode(predicted_id)}")

# Compute MLM loss manually
loss_fn = torch.nn.CrossEntropyLoss()
labels = torch.full_like(input_ids, -100)  # -100 = ignore
labels[0, 2] = input_ids[0, 2]  # Only compute loss at masked pos
loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
print(f"MLM loss at masked position: {loss.item():.4f}")

# Use DataCollator for automatic masking
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Example batch
batch = data_collator([inputs])
print(f"Masked input: {tokenizer.decode(batch['input_ids'][0])}")
print(f"Labels (non -100): {(batch['labels'][0] != -100).sum().item()} tokens masked")`,id:"mlm-code"}),e.jsx(o,{title:"MLM Creates Pretrain-Finetune Discrepancy",content:"During pretraining, 15% of tokens are corrupted. During fine-tuning, no tokens are masked. This distribution mismatch can slightly hurt performance. Models like XLNet address this with permutation language modeling, and ELECTRA uses replaced token detection instead.",id:"mlm-discrepancy-warning"})]})}const j=Object.freeze(Object.defineProperty({__proto__:null,default:l},Symbol.toStringTag,{value:"Module"}));function d(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Next Sentence Prediction (NSP)"}),e.jsx("p",{className:"text-lg text-gray-300",children:"Next Sentence Prediction is BERT's secondary pretraining objective. It trains a binary classifier on the [CLS] token representation to determine whether two sentences appear consecutively in the original corpus, helping the model understand inter-sentence relationships."}),e.jsx(t,{title:"NSP Objective",definition:"Given sentence pair $(A, B)$, NSP predicts a binary label $y \\in \\{\\\\text{IsNext}, \\\\text{NotNext}\\}$. During training, 50% of pairs are actual consecutive sentences ($y = \\\\text{IsNext}$) and 50% are randomly sampled ($y = \\\\text{NotNext}$).",notation:"$\\mathcal{L}_{\\\\text{NSP}} = -[y \\log \\hat{y} + (1-y)\\log(1-\\hat{y})]$ where $\\hat{y} = \\sigma(W_{\\\\text{NSP}} \\cdot h_{\\\\text{[CLS]}})$.",id:"nsp-def"}),e.jsx(n,{title:"NSP Training Pair Construction",problem:"Given corpus with sentences S1='The cat sat on the mat.' S2='It purred softly.' S3='Markets rose 2% today.', construct NSP pairs.",steps:[{formula:"\\text{Positive: } (S_1, S_2) \\rightarrow \\text{IsNext}",explanation:"S2 actually follows S1 in the corpus, so label is IsNext."},{formula:"\\text{Negative: } (S_1, S_3) \\rightarrow \\text{NotNext}",explanation:"S3 is randomly sampled from the corpus and does not follow S1."},{formula:"\\text{Input format: [CLS] } S_A \\text{ [SEP] } S_B \\text{ [SEP]}",explanation:"Both pairs are formatted with special tokens and segment embeddings."}],id:"nsp-pairs-example"}),e.jsx(i,{type:"intuition",title:"Purpose of NSP",content:"Many downstream tasks like question answering and natural language inference require understanding relationships between sentence pairs. NSP was designed to teach the model sentence-level coherence. The [CLS] token's representation captures this pair relationship.",id:"nsp-purpose"}),e.jsx(a,{title:"nsp_example.py",code:`from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

# Positive pair (consecutive sentences)
sentence_a = "The weather is beautiful today."
sentence_b = "Let's go for a walk in the park."
inputs_pos = tokenizer(sentence_a, sentence_b, return_tensors="pt")

# Negative pair (random sentences)
sentence_c = "Quantum computing uses qubits."
inputs_neg = tokenizer(sentence_a, sentence_c, return_tensors="pt")

with torch.no_grad():
    # Positive pair
    out_pos = model(**inputs_pos)
    logits_pos = out_pos.logits  # [batch, 2]
    prob_pos = torch.softmax(logits_pos, dim=-1)
    print(f"Positive pair - IsNext prob: {prob_pos[0, 0]:.4f}")

    # Negative pair
    out_neg = model(**inputs_neg)
    logits_neg = out_neg.logits
    prob_neg = torch.softmax(logits_neg, dim=-1)
    print(f"Negative pair - IsNext prob: {prob_neg[0, 0]:.4f}")

# NSP loss computation
labels_pos = torch.tensor([0])  # 0 = IsNext
labels_neg = torch.tensor([1])  # 1 = NotNext

out_with_loss = model(**inputs_pos, labels=labels_pos)
print(f"NSP loss (positive pair): {out_with_loss.loss.item():.4f}")

out_with_loss = model(**inputs_neg, labels=labels_neg)
print(f"NSP loss (negative pair): {out_with_loss.loss.item():.4f}")

# Full BERT pretraining combines both objectives
# Total loss = MLM_loss + NSP_loss
print("\\nNote: BERT pretraining loss = L_MLM + L_NSP")`,id:"nsp-code"}),e.jsx(o,{title:"NSP Was Later Found to Be Less Useful",content:"RoBERTa (Liu et al., 2019) showed that removing NSP and training with full-length sequences actually improves performance. The topic mismatch from random sentence sampling made NSP too easy -- the model could rely on topic signals rather than coherence. ALBERT replaced NSP with Sentence Order Prediction (SOP), which uses consecutive sentences in both orders.",id:"nsp-criticism"}),e.jsx(i,{type:"historical",title:"Evolution Beyond NSP",content:"RoBERTa dropped NSP entirely and used dynamic masking. ALBERT introduced SOP where both sentences come from the same document but may be swapped. SpanBERT also dropped NSP and found improvements. The consensus is that NSP provides marginal or negative benefit compared to longer training with better objectives.",id:"nsp-evolution"})]})}const N=Object.freeze(Object.defineProperty({__proto__:null,default:d},Symbol.toStringTag,{value:"Module"}));function c(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Whole Word Masking and SpanBERT"}),e.jsx("p",{className:"text-lg text-gray-300",children:"Standard BERT masks individual WordPiece tokens, which can leak information when partial words remain visible. Whole Word Masking (WWM) and SpanBERT address this by masking entire words or contiguous spans, forcing the model to learn deeper contextual understanding."}),e.jsx(t,{title:"Whole Word Masking",definition:"When a WordPiece subtoken is selected for masking, all subtokens belonging to the same word are also masked. If 'playing' is tokenized as ['play', '##ing'] and '##ing' is selected, both 'play' and '##ing' are masked together.",notation:"For word $w$ with subtokens $(t_1, t_2, \\ldots, t_k)$: if any $t_i \\in \\mathcal{M}$, then $\\forall j: t_j \\in \\mathcal{M}$.",id:"wwm-def"}),e.jsx(n,{title:"Standard vs Whole Word Masking",problem:"Compare masking for 'unbelievable' tokenized as ['un', '##bel', '##iev', '##able'].",steps:[{formula:"\\text{Standard: } [\\text{un}, \\text{[MASK]}, \\text{##iev}, \\text{##able}]",explanation:"Only ##bel is masked. The model can easily guess it from remaining subtokens."},{formula:"\\text{WWM: } [\\text{[MASK]}, \\text{[MASK]}, \\text{[MASK]}, \\text{[MASK]}]",explanation:'All subtokens of "unbelievable" are masked, forcing real prediction from context.'},{formula:"\\mathcal{L}_{\\text{WWM}} = -\\sum_{j=1}^{4} \\log P(t_j \\mid x_{\\backslash w})",explanation:"The loss requires predicting all subtokens from context alone."}],id:"wwm-example"}),e.jsx(t,{title:"SpanBERT",definition:"SpanBERT (Joshi et al., 2020) masks contiguous random spans of tokens. Span lengths are sampled from a geometric distribution $\\\\ell \\\\sim \\\\text{Geo}(p=0.2)$, clipped to $[1, 10]$, with mean span length of 3.8 tokens.",notation:"SpanBERT adds a Span Boundary Objective (SBO): $P(x_i \\\\mid x_{s-1}, x_{e+1}, p_{i}) $ where $s, e$ are span boundaries and $p_i$ is the relative position within the span.",id:"spanbert-def"}),e.jsx(i,{type:"intuition",title:"Why Span Masking Works Better",content:"Masking contiguous spans forces the model to reason about multi-token concepts rather than individual subwords. This is especially beneficial for tasks like coreference resolution, relation extraction, and question answering, where understanding spans of text is critical.",id:"span-masking-intuition"}),e.jsx(a,{title:"whole_word_masking.py",code:`from transformers import BertTokenizer, BertForMaskedLM
import torch
import random

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

def whole_word_masking(text, mlm_prob=0.15):
    """Apply whole-word masking to tokenized text."""
    tokens = tokenizer.tokenize(text)

    # Group subtokens into words
    word_groups = []
    current_word = []
    for i, token in enumerate(tokens):
        if token.startswith("##"):
            current_word.append(i)
        else:
            if current_word:
                word_groups.append(current_word)
            current_word = [i]
    if current_word:
        word_groups.append(current_word)

    # Select ~15% of words to mask
    num_to_mask = max(1, int(len(word_groups) * mlm_prob))
    selected = random.sample(word_groups, num_to_mask)
    mask_indices = set(idx for group in selected for idx in group)

    # Apply 80/10/10 rule
    masked_tokens = tokens.copy()
    labels = [-100] * len(tokens)
    for idx in mask_indices:
        labels[idx] = tokenizer.convert_tokens_to_ids(tokens[idx])
        r = random.random()
        if r < 0.8:
            masked_tokens[idx] = "[MASK]"
        elif r < 0.9:
            masked_tokens[idx] = tokenizer.convert_ids_to_tokens(
                random.randint(0, tokenizer.vocab_size - 1)
            )
    return masked_tokens, labels

# Example
text = "The unbelievable performance surprised everyone"
masked, labels = whole_word_masking(text, mlm_prob=0.3)
print(f"Original tokens: {tokenizer.tokenize(text)}")
print(f"Masked tokens:   {masked}")
print(f"Labels (masked):  {[l for l in labels if l != -100]}")

# SpanBERT-style span masking
import numpy as np

def span_masking(tokens, mlm_prob=0.15, max_span=10, p=0.2):
    """Sample geometric-distribution spans for masking."""
    n = len(tokens)
    budget = int(n * mlm_prob)
    mask_indices = set()

    while len(mask_indices) < budget:
        span_len = min(np.random.geometric(p), max_span)
        start = random.randint(0, n - 1)
        for i in range(start, min(start + span_len, n)):
            mask_indices.add(i)
    return mask_indices

tokens = tokenizer.tokenize("The quick brown fox jumps over the lazy dog")
spans = span_masking(tokens, mlm_prob=0.30)
print(f"\\nSpan masking indices: {sorted(spans)}")
result = [("[MASK]" if i in spans else t) for i, t in enumerate(tokens)]
print(f"Span masked: {result}")`,id:"wwm-code"}),e.jsx(o,{title:"Token Budget with WWM",content:"With whole-word masking, the actual percentage of masked tokens may exceed 15% because masking one subtoken forces all sibling subtokens to be masked. Implementations typically adjust the word selection probability to keep the total masked token count near the target budget.",id:"wwm-budget-warning"})]})}const B=Object.freeze(Object.defineProperty({__proto__:null,default:c},Symbol.toStringTag,{value:"Module"}));function p(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Autoregressive Language Modeling"}),e.jsx("p",{className:"text-lg text-gray-300",children:"Autoregressive (AR) language models generate text one token at a time, conditioning each prediction on all previously generated tokens. This left-to-right factorization is the foundation of GPT-family models and modern large language models."}),e.jsx(t,{title:"Autoregressive Factorization",definition:"An autoregressive language model decomposes the joint probability of a sequence into a product of conditional probabilities: $P(x_1, x_2, \\ldots, x_T) = \\prod_{t=1}^{T} P(x_t \\mid x_1, \\ldots, x_{t-1})$.",notation:"$P(x_t \\mid x_{<t})$ denotes the probability of token $x_t$ given all preceding tokens $x_{<t} = (x_1, \\ldots, x_{t-1})$.",id:"ar-def"}),e.jsx(s,{title:"Chain Rule of Probability",statement:"The autoregressive factorization is exact by the chain rule -- it introduces no approximation. Any joint distribution $P(x_{1:T})$ can be decomposed as $\\prod_{t=1}^T P(x_t \\mid x_{<t})$.",proof:"By the definition of conditional probability: $P(A, B) = P(A)P(B|A)$. Applied recursively: $P(x_1, x_2, x_3) = P(x_1) \\cdot P(x_2|x_1) \\cdot P(x_3|x_1, x_2)$. This generalizes to $T$ variables by induction.",id:"chain-rule-thm"}),e.jsx(n,{title:"Autoregressive Generation Step-by-Step",problem:"Generate text starting from 'The' using an autoregressive model.",steps:[{formula:'P(x_1) = P(\\text{"The"})',explanation:"Start with the prompt token."},{formula:'P(x_2 \\mid x_1) \\rightarrow \\text{"cat"} \\; (\\text{sampled})',explanation:"Model outputs distribution over vocabulary; sample or pick argmax."},{formula:'P(x_3 \\mid \\text{"The cat"}) \\rightarrow \\text{"sat"}',explanation:"Condition on full prefix. Causal mask ensures position 3 only sees positions 1-2."},{formula:"\\text{Continue until } x_t = \\text{[EOS] or max length}",explanation:"Generation terminates on end-of-sequence token or length limit."}],id:"ar-generation-example"}),e.jsx(i,{type:"intuition",title:"Causal Masking Enables AR",content:"In a Transformer decoder, the causal (triangular) attention mask ensures that position t can only attend to positions 1 through t. This prevents information leakage from future tokens and makes the model truly autoregressive during both training and inference.",id:"causal-mask-note"}),e.jsx(a,{title:"autoregressive_generation.py",code:`from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def autoregressive_generate(prompt, max_new_tokens=20, temperature=1.0):
    """Generate text one token at a time (greedy or sampling)."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = input_ids.clone()

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated)
            # logits at the last position
            next_logits = outputs.logits[:, -1, :] / temperature

        # Sample from distribution
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        generated = torch.cat([generated, next_token], dim=-1)

        # Stop at end of text
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Generate with different temperatures
print("=== Temperature 0.7 ===")
print(autoregressive_generate("The future of AI is", temperature=0.7))
print("\\n=== Temperature 1.0 ===")
print(autoregressive_generate("The future of AI is", temperature=1.0))
print("\\n=== Greedy (temperature -> 0) ===")
print(autoregressive_generate("The future of AI is", temperature=0.1))

# Visualize the causal mask
seq_len = 5
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
print(f"\\nCausal attention mask ({seq_len}x{seq_len}):")
print(causal_mask)`,id:"ar-code"}),e.jsx(o,{title:"Sequential Generation Is Slow",content:"Autoregressive generation is inherently sequential: each token depends on all previous tokens. This means generating T tokens requires T forward passes. KV-caching helps (caching key/value tensors for previous positions), but generation remains fundamentally O(T) in serial steps, unlike bidirectional models that process all positions in parallel.",id:"ar-slow-warning"})]})}const C=Object.freeze(Object.defineProperty({__proto__:null,default:p},Symbol.toStringTag,{value:"Module"}));function m(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"GPT Training Objective"}),e.jsx("p",{className:"text-lg text-gray-300",children:"The GPT family of models uses a simple but powerful training objective: maximize the likelihood of the next token given all preceding tokens. This causal language modeling objective scales remarkably well and forms the basis of modern LLMs."}),e.jsx(t,{title:"Causal Language Modeling Loss",definition:"Given a corpus of token sequences, the GPT training objective maximizes: $\\mathcal{L} = -\\frac{1}{T}\\sum_{t=1}^{T} \\log P_\\theta(x_t \\mid x_1, \\ldots, x_{t-1})$. This is the average negative log-likelihood per token, equivalent to cross-entropy with the data distribution.",notation:"Perplexity is $\\\\text{PPL} = \\\\exp(\\mathcal{L})$. Lower perplexity means better next-token prediction.",id:"gpt-loss-def"}),e.jsx(s,{title:"Relationship Between Cross-Entropy and Perplexity",statement:"For cross-entropy loss $\\mathcal{L}$ in nats, perplexity is $\\text{PPL} = e^{\\mathcal{L}}$. For cross-entropy in bits, $\\text{PPL} = 2^{\\mathcal{L}_{\\text{bits}}}$. A model with PPL of $k$ is as uncertain as a uniform distribution over $k$ tokens.",proof:"$\\text{PPL} = \\exp\\left(-\\frac{1}{T}\\sum_{t=1}^T \\log P(x_t \\mid x_{<t})\\right) = \\exp(\\mathcal{L})$. If $P$ is uniform over $k$ tokens, $\\mathcal{L} = \\log k$, so $\\text{PPL} = k$.",id:"ppl-theorem"}),e.jsx(n,{title:"Computing GPT Loss",problem:"Compute the loss for predicting 'The cat sat' with vocabulary size |V|=50257.",steps:[{formula:"P(\\text{cat} \\mid \\text{The}) = 0.02",explanation:'Model assigns 2% probability to "cat" following "The".'},{formula:"P(\\text{sat} \\mid \\text{The cat}) = 0.05",explanation:'Model assigns 5% probability to "sat" following "The cat".'},{formula:"\\mathcal{L} = -\\frac{1}{2}[\\log(0.02) + \\log(0.05)] = \\frac{1}{2}[3.91 + 3.00] = 3.46",explanation:"Average negative log-likelihood over the two predictions."},{formula:"\\text{PPL} = e^{3.46} \\approx 31.8",explanation:"Model is as confused as choosing uniformly from ~32 tokens."}],id:"gpt-loss-example"}),e.jsx(i,{type:"historical",title:"GPT Evolution",content:"GPT-1 (2018, 117M params) showed pretrain+finetune works for NLP. GPT-2 (2019, 1.5B) demonstrated zero-shot abilities with scale. GPT-3 (2020, 175B) introduced in-context learning without fine-tuning. Each generation kept the same causal LM objective -- only scale and data changed.",id:"gpt-evolution-note"}),e.jsx(a,{title:"gpt_training_objective.py",code:`from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Compute causal LM loss for a sequence
text = "The cat sat on the mat and purred softly"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]

with torch.no_grad():
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss         # average cross-entropy
    logits = outputs.logits     # [batch, seq_len, vocab_size]

print(f"Text: {text}")
print(f"Cross-entropy loss: {loss.item():.4f}")
print(f"Perplexity: {torch.exp(loss).item():.2f}")

# Manual loss computation to understand the objective
# Shift: predict token t from tokens 0..t-1
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = input_ids[:, 1:].contiguous()

# Per-token loss
loss_per_token = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    reduction="none"
)

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print("\\nPer-token losses:")
for i, (tok, l) in enumerate(zip(tokens[1:], loss_per_token)):
    prob = torch.exp(-l).item()
    print(f"  {tok:>12s}  loss={l.item():.3f}  P={prob:.4f}")

# Compare models of different sizes
for model_name in ["gpt2", "gpt2-medium"]:
    m = GPT2LMHeadModel.from_pretrained(model_name)
    m.eval()
    with torch.no_grad():
        out = m(input_ids=input_ids, labels=input_ids)
    params = sum(p.numel() for p in m.parameters())
    print(f"\\n{model_name}: {params/1e6:.0f}M params, "
          f"loss={out.loss.item():.3f}, PPL={torch.exp(out.loss).item():.1f}")`,id:"gpt-objective-code"}),e.jsx(o,{title:"Loss on First Token Is Undefined",content:"The first token has no preceding context, so GPT cannot predict it. In practice, the labels are shifted: we predict token t+1 from position t. A sequence of length T yields T-1 loss terms. This is why HuggingFace uses labels=input_ids and internally shifts them.",id:"first-token-warning"})]})}const A=Object.freeze(Object.defineProperty({__proto__:null,default:m},Symbol.toStringTag,{value:"Module"}));function u(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Next-Token Prediction Mechanics"}),e.jsx("p",{className:"text-lg text-gray-300",children:"The mechanics of next-token prediction involve converting hidden states into probability distributions over the vocabulary, then selecting the next token through various decoding strategies. Understanding this pipeline is crucial for both training and inference."}),e.jsx(t,{title:"Logit Computation",definition:"At each position $t$, the final hidden state $h_t \\in \\mathbb{R}^d$ is projected to vocabulary logits via the language model head: $z_t = W_{\\text{LM}} h_t + b$, where $W_{\\text{LM}} \\in \\mathbb{R}^{|V| \\times d}$. Often $W_{\\text{LM}}$ is tied to the input embedding matrix.",notation:"$P(x_{t+1} = v \\mid x_{\\leq t}) = \\text{softmax}(z_t)_v = \\frac{\\exp(z_t[v])}{\\sum_{v'} \\exp(z_t[v'])}$",id:"logit-def"}),e.jsx(n,{title:"From Hidden State to Token",problem:"Given hidden state h with d=768 and vocab size |V|=50257, trace the prediction pipeline.",steps:[{formula:"z = W_{\\text{LM}} h \\in \\mathbb{R}^{50257}",explanation:"Project 768-dim hidden state to 50257-dim logits (one per vocabulary token)."},{formula:"z' = z / \\tau \\quad (\\text{temperature scaling})",explanation:"Temperature tau < 1 sharpens distribution, tau > 1 flattens it."},{formula:"P = \\text{softmax}(z') \\in \\Delta^{|V|-1}",explanation:"Convert logits to probability simplex. Sum to 1."},{formula:"x_{t+1} \\sim \\text{Top-}k(P) \\text{ or } \\text{Top-}p(P)",explanation:"Sample from filtered distribution using top-k or nucleus (top-p) sampling."}],id:"prediction-pipeline"}),e.jsx(i,{type:"tip",title:"Weight Tying",content:"Most modern LLMs tie the output projection W_LM to the transpose of the input embedding matrix E. This means the logit for token v is the dot product of the hidden state with v's embedding: z[v] = e_v^T h. Weight tying reduces parameters and often improves quality by forcing input and output representations to share the same space.",id:"weight-tying-note"}),e.jsx(a,{title:"next_token_mechanics.py",code:`from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)

# Step 1: Get last hidden state at final position
last_hidden = outputs.hidden_states[-1][:, -1, :]  # [1, 768]
print(f"Hidden state shape: {last_hidden.shape}")

# Step 2: Apply LM head (linear projection)
logits = outputs.logits[:, -1, :]  # [1, 50257]
print(f"Logits shape: {logits.shape}")
print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

# Step 3: Check weight tying
embed_weight = model.transformer.wte.weight  # [50257, 768]
lm_head_weight = model.lm_head.weight        # [50257, 768]
print(f"Weights tied: {torch.equal(embed_weight, lm_head_weight)}")

# Step 4: Decoding strategies
def decode_strategies(logits, k=10, p=0.9, temp=1.0):
    """Compare greedy, top-k, and top-p decoding."""
    scaled = logits / temp
    probs = F.softmax(scaled, dim=-1)

    # Greedy
    greedy_id = probs.argmax(dim=-1)
    print(f"\\nGreedy: {tokenizer.decode(greedy_id)} (p={probs[0, greedy_id].item():.4f})")

    # Top-k: keep only top k tokens
    topk_vals, topk_idx = probs.topk(k, dim=-1)
    print(f"\\nTop-{k} candidates:")
    for i in range(k):
        print(f"  {tokenizer.decode(topk_idx[0, i]):>10s}  p={topk_vals[0, i]:.4f}")

    # Top-p (nucleus): keep smallest set with cumulative prob >= p
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    cutoff = (cumsum <= p).sum().item() + 1
    print(f"\\nTop-p ({p}): {cutoff} tokens in nucleus")

decode_strategies(logits, k=5, p=0.9, temp=0.8)

# Step 5: Entropy of the distribution
probs_full = F.softmax(logits, dim=-1)
entropy = -(probs_full * probs_full.log()).sum(dim=-1)
print(f"\\nEntropy: {entropy.item():.2f} nats")
print(f"Effective choices: {torch.exp(entropy).item():.0f} tokens")`,id:"next-token-code"}),e.jsx(o,{title:"Repetition and Degenerate Text",content:"Greedy and beam search decoding often produce repetitive, degenerate text. This happens because the model assigns high probability to recently seen tokens (a positive feedback loop). Sampling with temperature, top-k, or top-p helps, but too much randomness produces incoherent text. Finding the right balance is an active area of research.",id:"degenerate-warning"}),e.jsx(s,{title:"Softmax Temperature",statement:"For logits $z$ and temperature $\\tau > 0$: as $\\tau \\to 0$, $\\text{softmax}(z/\\tau)$ converges to a one-hot distribution on $\\arg\\max(z)$. As $\\tau \\to \\infty$, it converges to the uniform distribution $1/|V|$.",proof:"For $\\tau \\to 0$: the largest logit dominates the exponential, so $\\exp(z_{\\max}/\\tau) \\gg \\exp(z_i/\\tau)$ for $z_i < z_{\\max}$. For $\\tau \\to \\infty$: all $z_i/\\tau \\to 0$, so $\\exp(z_i/\\tau) \\to 1$ for all $i$, giving uniform $1/|V|$.",id:"temperature-theorem"})]})}const E=Object.freeze(Object.defineProperty({__proto__:null,default:u},Symbol.toStringTag,{value:"Module"}));function h(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Curriculum Learning and Data Ordering"}),e.jsx("p",{className:"text-lg text-gray-300",children:"Curriculum learning applies the principle that training benefits from a structured ordering of examples -- starting with easier or simpler data and progressively introducing harder material. In LLM pretraining, the order, mixture, and scheduling of data sources significantly impacts the final model quality."}),e.jsx(t,{title:"Curriculum Learning",definition:"Curriculum learning trains a model by presenting examples in a meaningful order rather than random shuffling. A curriculum function $C(t)$ maps training step $t$ to a data distribution $\\mathcal{D}_t$, typically progressing from simpler to more complex data.",notation:"$\\mathcal{D}_t = C(t)$ where $\\mathcal{D}_0$ is the easiest distribution and $\\mathcal{D}_T$ approaches the full data distribution.",id:"curriculum-def"}),e.jsx(n,{title:"Data Mixing Schedule",problem:"Design a data mixing schedule for pretraining a 7B parameter LLM.",steps:[{formula:"\\text{Phase 1 (0-50\\%): } \\{\\text{Web: 70\\%, Books: 15\\%, Code: 10\\%, Wiki: 5\\%}\\}",explanation:"Start with a broad web-heavy mixture for general language understanding."},{formula:"\\text{Phase 2 (50-80\\%): } \\{\\text{Web: 50\\%, Books: 20\\%, Code: 20\\%, Wiki: 10\\%}\\}",explanation:"Increase code and high-quality sources to improve reasoning."},{formula:"\\text{Phase 3 (80-100\\%): } \\{\\text{Web: 30\\%, Books: 25\\%, Code: 25\\%, Wiki: 15\\%, Math: 5\\%}\\}",explanation:"Final phase emphasizes quality, reasoning, and factual data."}],id:"data-mixing-example"}),e.jsx(i,{type:"intuition",title:"Why Data Order Matters",content:"Random shuffling is the default but not necessarily optimal. Research shows that presenting high-quality data later in training (when the model can better leverage it) can improve final performance. This is analogous to how students learn fundamentals before advanced topics. The LLaMA and Qwen teams have confirmed that carefully designed data schedules improved their models.",id:"data-order-intuition"}),e.jsx(s,{title:"Data Mixing Law",statement:"For a mixture of $k$ data sources with weights $w_i$ ($\\sum w_i = 1$), the overall loss approximately follows: $\\mathcal{L}(w_1, \\ldots, w_k) \\approx \\sum_{i=1}^k w_i \\cdot \\mathcal{L}_i(N \\cdot w_i)$, where $\\mathcal{L}_i(n)$ is the loss on source $i$ after training on $n$ tokens from that source.",proof:"Each source has its own scaling law $\\mathcal{L}_i(n) = A_i / n^{\\alpha_i} + \\mathcal{L}_{\\infty,i}$. The effective tokens from source $i$ is $n_i = N \\cdot w_i$ where $N$ is total tokens. The weighted loss can be optimized over weights $w_i$ using Lagrange multipliers.",id:"mixing-law-thm"}),e.jsx(a,{title:"curriculum_training.py",code:`import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np

class CurriculumScheduler:
    """Schedule data source mixing weights over training."""

    def __init__(self, sources, initial_weights, final_weights, warmup_steps):
        self.sources = sources
        self.initial = np.array(initial_weights, dtype=np.float64)
        self.final = np.array(final_weights, dtype=np.float64)
        self.warmup_steps = warmup_steps

    def get_weights(self, step):
        """Linearly interpolate from initial to final weights."""
        progress = min(step / self.warmup_steps, 1.0)
        weights = self.initial + progress * (self.final - self.initial)
        return weights / weights.sum()  # Normalize

# Define curriculum
scheduler = CurriculumScheduler(
    sources=["web", "books", "code", "wiki", "math"],
    initial_weights=[0.70, 0.15, 0.10, 0.04, 0.01],
    final_weights=  [0.30, 0.25, 0.25, 0.15, 0.05],
    warmup_steps=100_000
)

# Show schedule at different points
for step in [0, 25_000, 50_000, 75_000, 100_000]:
    w = scheduler.get_weights(step)
    parts = ", ".join(f"{s}={v:.1%}" for s, v in zip(scheduler.sources, w))
    print(f"Step {step:>7d}: {parts}")

# Data quality scoring for curriculum
def compute_difficulty(text, tokenizer=None):
    """Heuristic difficulty score based on text properties."""
    scores = {
        "length": min(len(text.split()) / 500, 1.0),
        "avg_word_len": min(np.mean([len(w) for w in text.split()]) / 10, 1.0),
        "unique_ratio": len(set(text.split())) / max(len(text.split()), 1),
    }
    return np.mean(list(scores.values()))

# Example: sort by difficulty
texts = [
    "The cat sat on the mat.",
    "Quantum entanglement describes correlations between particles.",
    "The Riemann hypothesis concerns the distribution of primes.",
]
scored = [(compute_difficulty(t), t) for t in texts]
for score, text in sorted(scored):
    print(f"  difficulty={score:.3f}: {text[:60]}")

# DoReMi-style domain reweighting
def doremi_update(domain_losses, domain_weights, ref_losses, step_size=0.01):
    """Update domain weights based on excess loss over reference."""
    excess = np.maximum(domain_losses - ref_losses, 0)
    log_weights = np.log(domain_weights) + step_size * excess
    new_weights = np.exp(log_weights)
    return new_weights / new_weights.sum()

weights = np.array([0.5, 0.3, 0.2])
losses = np.array([2.5, 3.1, 4.0])
ref = np.array([2.0, 2.5, 3.0])
new_w = doremi_update(losses, weights, ref)
print(f"\\nDoReMi reweighting: {weights} -> {new_w.round(3)}")`,id:"curriculum-code"}),e.jsx(o,{title:"Curriculum Design Is Largely Empirical",content:"There is no proven optimal curriculum for LLM pretraining. Most insights come from expensive ablation studies on smaller models and may not transfer perfectly to larger scales. Techniques like DoReMi attempt to automate mixing weight selection, but the search space is enormous. Always validate curriculum choices with held-out evaluations.",id:"curriculum-warning"})]})}const D=Object.freeze(Object.defineProperty({__proto__:null,default:h},Symbol.toStringTag,{value:"Module"}));function f(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Pretraining Data Curation"}),e.jsx("p",{className:"text-lg text-gray-300",children:"The quality and composition of pretraining data is one of the most critical factors in LLM performance. Modern LLMs are trained on trillions of tokens from diverse sources including web crawls, books, code, and curated datasets."}),e.jsx(t,{title:"Common Crawl",definition:"Common Crawl is a nonprofit that maintains an open repository of web crawl data, producing petabytes of raw HTML monthly. It forms the foundation of most pretraining datasets but requires extensive cleaning: raw Common Crawl is estimated to be only 1-5% high-quality text.",notation:"After filtering, Common Crawl typically yields datasets of 1-5 trillion tokens.",id:"common-crawl-def"}),e.jsx(n,{title:"Major Pretraining Datasets",problem:"Compare the composition of The Pile, RedPajama, and FineWeb.",steps:[{formula:"\\text{The Pile (2021): 825 GB, 22 diverse sources}",explanation:"Created by EleutherAI. Includes Pile-CC, PubMed, ArXiv, GitHub, StackExchange, Wikipedia, Books3, and more. First large-scale curated open dataset."},{formula:"\\text{RedPajama v2 (2023): 30T tokens, 5 languages}",explanation:"Created by Together AI. Web data with quality signals. Includes CommonCrawl, C4, GitHub, Books, ArXiv, Wikipedia, StackExchange."},{formula:"\\text{FineWeb (2024): 15T tokens, deduplicated CC}",explanation:"Created by HuggingFace. Aggressive filtering and deduplication of Common Crawl with transparent methodology. FineWeb-Edu subset has educational content scoring."}],id:"datasets-comparison"}),e.jsx(i,{type:"historical",title:"The Data Scaling Journey",content:"GPT-1 used BookCorpus (~800M tokens). GPT-2 used WebText (~8B tokens). GPT-3 used a 300B token mix. LLaMA used 1.4T tokens. LLaMA-2 used 2T tokens. Modern models like LLaMA-3 train on 15T+ tokens. Each generation dramatically increased data scale and quality requirements.",id:"data-scaling-history"}),e.jsx(a,{title:"data_curation_pipeline.py",code:`from datasets import load_dataset
import hashlib
import re

# Load a sample from FineWeb
# dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

# Simulated data curation pipeline
class DataCurationPipeline:
    """Pipeline for processing raw web text into pretraining data."""

    def __init__(self):
        self.stats = {"total": 0, "passed": 0, "reasons": {}}

    def language_filter(self, text, min_confidence=0.8):
        """Filter non-English text (simplified)."""
        ascii_ratio = sum(c.isascii() for c in text) / max(len(text), 1)
        return ascii_ratio > 0.8

    def quality_filter(self, text):
        """Heuristic quality filters inspired by C4/FineWeb."""
        words = text.split()

        # Minimum length
        if len(words) < 50:
            return False, "too_short"

        # Maximum fraction of lines ending with ellipsis
        lines = text.split("\\n")
        ellipsis_lines = sum(1 for l in lines if l.strip().endswith("..."))
        if lines and ellipsis_lines / len(lines) > 0.3:
            return False, "too_many_ellipsis"

        # Check for boilerplate
        boilerplate = ["cookie policy", "javascript", "subscribe now",
                       "click here", "terms of service"]
        lower = text.lower()
        if sum(1 for b in boilerplate if b in lower) >= 3:
            return False, "boilerplate"

        # Word length distribution
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 3 or avg_word_len > 12:
            return False, "unusual_word_length"

        # Repetition filter
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.1:
            return False, "too_repetitive"

        return True, "passed"

    def process(self, documents):
        """Run full pipeline on documents."""
        results = []
        for doc in documents:
            self.stats["total"] += 1
            if not self.language_filter(doc):
                self.stats["reasons"]["non_english"] =                     self.stats["reasons"].get("non_english", 0) + 1
                continue
            passed, reason = self.quality_filter(doc)
            if not passed:
                self.stats["reasons"][reason] =                     self.stats["reasons"].get(reason, 0) + 1
                continue
            self.stats["passed"] += 1
            results.append(doc)
        return results

# Run pipeline
pipeline = DataCurationPipeline()
sample_docs = [
    "The quick brown fox " * 100,  # repetitive
    "Click here to subscribe. Cookie policy. Terms of service. " * 20,
    "Short text.",
    " ".join(f"word{i}" for i in range(200)),  # normal length
    "Natural language processing has revolutionized how computers "
    "understand human language. " * 5 + "This comprehensive review "
    "covers tokenization, embeddings, and transformer architectures "
    "that power modern systems. " * 3,
]
clean = pipeline.process(sample_docs)
print(f"Input: {pipeline.stats['total']} docs")
print(f"Passed: {pipeline.stats['passed']} docs")
print(f"Filter reasons: {pipeline.stats['reasons']}")

# Data source mixing weights (typical for a 7B model)
sources = {
    "CommonCrawl": {"tokens": "4.5T", "weight": 0.67},
    "GitHub":      {"tokens": "0.5T", "weight": 0.045},
    "Wikipedia":   {"tokens": "0.1T", "weight": 0.045},
    "Books":       {"tokens": "0.3T", "weight": 0.045},
    "ArXiv":       {"tokens": "0.1T", "weight": 0.025},
    "StackExchange":{"tokens": "0.1T","weight": 0.02},
}
print("\\nTypical data mix:")
for name, info in sources.items():
    print(f"  {name:>16s}: {info['tokens']:>5s} tokens, weight={info['weight']:.1%}")`,id:"curation-code"}),e.jsx(o,{title:"Data Contamination",content:"Pretraining data may contain benchmark test sets (e.g., MMLU questions scraped from the web). This data contamination inflates evaluation scores. Careful decontamination via n-gram matching against benchmarks is essential but imperfect. Always report contamination analysis alongside benchmark results.",id:"contamination-warning"})]})}const F=Object.freeze(Object.defineProperty({__proto__:null,default:f},Symbol.toStringTag,{value:"Module"}));function g(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Deduplication and Data Filtering"}),e.jsx("p",{className:"text-lg text-gray-300",children:"Web-scale datasets contain massive amounts of duplicate and near-duplicate content. Deduplication is critical: training on duplicated data wastes compute, degrades model quality, and increases memorization risk. Filtering removes toxic, low-quality, or personally identifiable content."}),e.jsx(t,{title:"Exact Deduplication",definition:"Exact deduplication removes documents with identical content. This is typically implemented using hash-based methods: compute a hash $h(d)$ (e.g., SHA-256) for each document $d$, then remove all but one document per hash value.",notation:"Two documents are exact duplicates iff $h(d_1) = h(d_2)$. Exact dedup typically removes 10-30% of raw web data.",id:"exact-dedup-def"}),e.jsx(t,{title:"Fuzzy (Near) Deduplication",definition:"Near-deduplication identifies documents that are highly similar but not identical. MinHash with Locality-Sensitive Hashing (LSH) approximates Jaccard similarity between document n-gram sets. Documents with $J(d_1, d_2) > \\\\theta$ (typically $\\\\theta = 0.8$) are considered near-duplicates.",notation:"Jaccard similarity: $J(A, B) = \\\\frac{|A \\\\cap B|}{|A \\\\cup B|}$ where $A, B$ are n-gram sets.",id:"fuzzy-dedup-def"}),e.jsx(n,{title:"MinHash Deduplication",problem:"Use MinHash-LSH to find near-duplicate documents in a corpus.",steps:[{formula:"\\text{Step 1: Extract n-grams (e.g., 5-grams) from each document}",explanation:'Convert "the cat sat on the" to {(the,cat,sat,on,the), (cat,sat,on,the,...)}.'},{formula:"\\text{Step 2: Compute } k \\text{ MinHash signatures per document}",explanation:"Apply k random hash functions to the n-gram set. Each signature is the minimum hash value. Typical k=128."},{formula:"P(\\text{MinHash}(A) = \\text{MinHash}(B)) = J(A, B)",explanation:"MinHash probability equals Jaccard similarity, enabling fast approximation."},{formula:"\\text{Step 3: LSH bands to find candidate pairs efficiently}",explanation:"Split signatures into b bands of r rows. Documents matching in any band are candidates. Tune b,r to control precision/recall."}],id:"minhash-example"}),e.jsx(i,{type:"tip",title:"Deduplication at Scale",content:"For trillion-token datasets, exact dedup uses Bloom filters (space-efficient probabilistic set). MinHash-LSH for near-dedup is parallelizable with MapReduce. The FineWeb dataset used 5-gram MinHash with 128 hash functions and found ~40% near-duplicates in Common Crawl. Removing these significantly improved downstream benchmarks.",id:"dedup-scale-note"}),e.jsx(a,{title:"deduplication.py",code:`import hashlib
from collections import defaultdict
import random

# 1. Exact deduplication with hashing
def exact_dedup(documents):
    """Remove exact duplicate documents using SHA-256."""
    seen = set()
    unique = []
    for doc in documents:
        doc_hash = hashlib.sha256(doc.encode()).hexdigest()
        if doc_hash not in seen:
            seen.add(doc_hash)
            unique.append(doc)
    removed = len(documents) - len(unique)
    print(f"Exact dedup: {len(documents)} -> {len(unique)} ({removed} removed)")
    return unique

# 2. MinHash for near-deduplication
class MinHashDedup:
    def __init__(self, num_perm=128, ngram_size=5, threshold=0.8):
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.threshold = threshold
        self.max_hash = (1 << 32) - 1
        # Random hash parameters
        self.a = [random.randint(1, self.max_hash) for _ in range(num_perm)]
        self.b = [random.randint(0, self.max_hash) for _ in range(num_perm)]
        self.prime = (1 << 61) - 1

    def get_ngrams(self, text):
        words = text.lower().split()
        return set(tuple(words[i:i+self.ngram_size])
                   for i in range(len(words) - self.ngram_size + 1))

    def minhash_signature(self, text):
        ngrams = self.get_ngrams(text)
        sig = []
        for i in range(self.num_perm):
            min_val = float('inf')
            for ng in ngrams:
                h = hash(ng)
                val = (self.a[i] * h + self.b[i]) % self.prime
                min_val = min(min_val, val)
            sig.append(min_val)
        return sig

    def jaccard_estimate(self, sig1, sig2):
        return sum(a == b for a, b in zip(sig1, sig2)) / len(sig1)

    def deduplicate(self, documents):
        sigs = [self.minhash_signature(doc) for doc in documents]
        to_remove = set()
        for i in range(len(documents)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(documents)):
                if j in to_remove:
                    continue
                sim = self.jaccard_estimate(sigs[i], sigs[j])
                if sim >= self.threshold:
                    to_remove.add(j)
        result = [d for i, d in enumerate(documents) if i not in to_remove]
        print(f"MinHash dedup: {len(documents)} -> {len(result)}")
        return result

# Demo
docs = [
    "The cat sat on the mat and looked around the room quietly",
    "The cat sat on the mat and looked around the room quietly",  # exact
    "The cat sat on the mat and looked around the room softly",   # near-dup
    "Quantum computing uses qubits for parallel computation",
    "Machine learning models require large training datasets",
]
clean = exact_dedup(docs)
deduper = MinHashDedup(num_perm=64, ngram_size=3, threshold=0.7)
clean = deduper.deduplicate(clean)
for d in clean:
    print(f"  {d[:60]}...")`,id:"dedup-code"}),e.jsx(o,{title:"Over-Deduplication Can Hurt",content:"Aggressive deduplication can remove legitimate repeated content like common phrases, templates, or code patterns that the model should learn. Some studies show that moderate duplication (2-3x) of high-quality data can be beneficial. The goal is removing noise and spam duplication, not eliminating all repetition.",id:"over-dedup-warning"})]})}const G=Object.freeze(Object.defineProperty({__proto__:null,default:g},Symbol.toStringTag,{value:"Module"}));function _(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"BPE Training and Vocabulary Design"}),e.jsx("p",{className:"text-lg text-gray-300",children:"The tokenizer is the first component of any LLM and its design critically impacts model performance. Byte-Pair Encoding (BPE) is the dominant tokenization algorithm, learned from the pretraining corpus to create an efficient vocabulary of subword tokens."}),e.jsx(t,{title:"Byte-Pair Encoding (BPE)",definition:"BPE starts with a base vocabulary of individual bytes (or characters) and iteratively merges the most frequent adjacent pair of tokens into a new token. After $k$ merge operations, the vocabulary has $|V_{\\text{base}}| + k$ tokens. The merge order defines the tokenization rules.",notation:"At each step: find $(a, b) = \\arg\\max_{(x,y)} \\text{freq}(xy)$, add $ab$ to vocabulary, replace all occurrences of $a \\, b$ with $ab$.",id:"bpe-def"}),e.jsx(n,{title:"BPE Merge Process",problem:"Apply BPE merges to the corpus: 'low lower lowest low lower'.",steps:[{formula:"\\text{Initial tokens: l o w, l o w e r, l o w e s t, l o w, l o w e r}",explanation:"Start with character-level tokens. Count all adjacent pairs."},{formula:'\\text{Most frequent pair: (l, o) = 5. Merge: "lo"}',explanation:'Create new token "lo". Corpus becomes: lo w, lo w e r, lo w e s t, lo w, lo w e r.'},{formula:'\\text{Next pair: (lo, w) = 5. Merge: "low"}',explanation:"Corpus becomes: low, low e r, low e s t, low, low e r."},{formula:'\\text{Next pair: (e, r) = 2. Merge: "er"}',explanation:"Corpus becomes: low, low er, low e s t, low, low er. Continue until vocab size reached."}],id:"bpe-merge-example"}),e.jsx(i,{type:"intuition",title:"Why BPE Works Well",content:"BPE naturally discovers meaningful subword units: common words become single tokens, rare words split into known subwords. This balances vocabulary size (typically 32K-128K) against sequence length. Larger vocabularies mean shorter sequences but more parameters in the embedding layer. The sweet spot depends on the data and languages covered.",id:"bpe-intuition"}),e.jsx(s,{title:"Vocabulary Size Trade-off",statement:"For a fixed compute budget $C$, vocabulary size $|V|$ creates a trade-off: increasing $|V|$ decreases average sequence length by factor $\\rho(|V|)$ (compression ratio) but increases embedding parameters by $|V| \\times d$. The optimal vocabulary size satisfies $\\frac{\\partial}{\\partial |V|}[\\text{training cost}(|V|)] = 0$.",proof:"Total compute is proportional to $C \\propto 6ND$ where $N$ is parameters and $D$ is tokens. Larger $|V|$ means fewer tokens (shorter sequences) but more parameters in embeddings. Recent work suggests optimal $|V|$ scales as roughly $|V|^* \\propto N^{0.5}$ for model size $N$.",id:"vocab-tradeoff-thm"}),e.jsx(a,{title:"tokenizer_training.py",code:`from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers import normalizers, decoders
from transformers import AutoTokenizer

# Train a BPE tokenizer from scratch
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|padding|>", "<|unknown|>"],
    show_progress=True,
)

# Train on sample text (in practice, train on billions of tokens)
sample_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models learn representations from data.",
    "Natural language processing enables computers to understand text.",
    "Transformers use self-attention to process sequences in parallel.",
    "Large language models are pretrained on vast amounts of text data.",
] * 100  # Repeat for more frequent pairs

tokenizer.train_from_iterator(sample_texts, trainer=trainer)
print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

# Test tokenization
test = "Transformers revolutionized natural language processing"
encoded = tokenizer.encode(test)
print(f"Text: {test}")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Token count: {len(encoded.ids)}")

# Compare real tokenizers
print("\\n--- Comparing Real Tokenizers ---")
for name in ["bert-base-uncased", "gpt2", "meta-llama/Llama-2-7b-hf"]:
    try:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        ids = tok.encode(test)
        tokens = tok.tokenize(test)
        print(f"\\n{name}:")
        print(f"  Vocab size: {tok.vocab_size:,}")
        print(f"  Tokens ({len(tokens)}): {tokens[:10]}")
    except Exception as e:
        print(f"\\n{name}: (requires auth) vocab ~32K-128K")

# Compute compression ratio
import math
sample = "The Transformer architecture has become the backbone of modern NLP."
for name, tok in [("GPT-2", AutoTokenizer.from_pretrained("gpt2"))]:
    ids = tok.encode(sample)
    chars = len(sample)
    ratio = chars / len(ids)
    bits_per_char = math.log2(tok.vocab_size) / ratio
    print(f"\\n{name} compression:")
    print(f"  Characters: {chars}, Tokens: {len(ids)}")
    print(f"  Chars/token: {ratio:.1f}")
    print(f"  Bits/char (vocab): {bits_per_char:.1f}")`,id:"tokenizer-code"}),e.jsx(o,{title:"Tokenizer Fertility Across Languages",content:"BPE tokenizers trained primarily on English text produce far more tokens for non-English text (sometimes 3-10x more). This means the model sees fewer words per context window in other languages, effectively reducing its capacity for multilingual understanding. Modern tokenizers like those in LLaMA-3 use larger vocabularies (128K) and train on more balanced multilingual data to reduce this disparity.",id:"fertility-warning"})]})}const q=Object.freeze(Object.defineProperty({__proto__:null,default:_},Symbol.toStringTag,{value:"Module"}));function x(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Scaling Laws and Compute-Optimal Training"}),e.jsx("p",{className:"text-lg text-gray-300",children:"Scaling laws describe how model performance (loss) improves predictably with increases in model size, dataset size, and compute budget. These power-law relationships guide fundamental decisions about how to allocate resources during pretraining."}),e.jsx(t,{title:"Neural Scaling Laws (Kaplan et al., 2020)",definition:"For Transformer language models, the cross-entropy loss follows power laws: $L(N) = (N_c / N)^{\\alpha_N}$, $L(D) = (D_c / D)^{\\alpha_D}$, $L(C) = (C_c / C)^{\\alpha_C}$, where $N$ is parameters, $D$ is data tokens, $C$ is compute (FLOPs). Kaplan found $\\alpha_N \\approx 0.076$, $\\alpha_D \\approx 0.095$, $\\alpha_C \\approx 0.050$.",notation:"The combined scaling law is $L(N, D) = \\left[(N_c/N)^{\\alpha_N / \\alpha_D} + D_c/D\\right]^{\\alpha_D}$ capturing diminishing returns when either $N$ or $D$ is held fixed.",id:"scaling-laws-def"}),e.jsx(s,{title:"Chinchilla Scaling Law (Hoffmann et al., 2022)",statement:"For a compute budget $C \\approx 6ND$ FLOPs, the compute-optimal allocation is $N^* \\propto C^{0.5}$ and $D^* \\propto C^{0.5}$. Equivalently, the optimal token-to-parameter ratio is approximately $D^*/N^* \\approx 20$. A model with $N$ parameters should be trained on roughly $20N$ tokens.",proof:"Minimize $L(N, D)$ subject to $C = 6ND$. Using the parametric loss $L(N, D) = A/N^\\alpha + B/D^\\beta + L_\\infty$ with fitted $\\alpha \\approx 0.34$, $\\beta \\approx 0.28$, the Lagrangian yields $N^* \\propto C^{a/(a+b)}$ and $D^* \\propto C^{b/(a+b)}$ where $a = \\alpha/(\\alpha+1)$, $b = \\beta/(\\beta+1)$.",corollaries:["GPT-3 (175B params, 300B tokens) was undertrained by Chinchilla standards. Optimal: ~3.5T tokens.","Chinchilla (70B params, 1.4T tokens) matched GPT-3 performance with 4x less compute at inference.","LLaMA-1 (65B params, 1.4T tokens) and LLaMA-2 (70B, 2T tokens) followed this principle."],id:"chinchilla-thm"}),e.jsx(n,{title:"Compute-Optimal Model Sizing",problem:"Given a compute budget of 10^22 FLOPs, what are the optimal model and data sizes?",steps:[{formula:"C = 6ND \\implies 10^{22} = 6 \\cdot N \\cdot D",explanation:"The approximate compute formula for a forward+backward pass."},{formula:"D^* \\approx 20 N^* \\implies 10^{22} = 6 \\cdot N^* \\cdot 20 N^* = 120 (N^*)^2",explanation:"Apply the Chinchilla ratio D/N = 20."},{formula:"N^* = \\sqrt{10^{22}/120} \\approx 2.9 \\times 10^9 \\approx 2.9\\text{B params}",explanation:"Optimal model has roughly 3 billion parameters."},{formula:"D^* = 20 \\times 2.9 \\times 10^9 \\approx 58\\text{B tokens}",explanation:"Train on approximately 58 billion tokens for compute optimality."}],id:"compute-optimal-example"}),e.jsx(i,{type:"note",title:"Beyond Chinchilla: Inference-Optimal Scaling",content:"Chinchilla optimizes for training compute only. In practice, a smaller model trained for longer (on more data) has lower inference cost per query. LLaMA-3 8B was trained on 15T tokens (1875x parameters) -- far beyond Chinchilla-optimal -- because inference efficiency matters more in deployment. The optimal trade-off depends on total lifetime inference compute vs. one-time training cost.",id:"inference-optimal-note"}),e.jsx(a,{title:"scaling_laws.py",code:`import numpy as np

# Chinchilla scaling law parameters (Hoffmann et al., 2022)
A = 406.4       # Parameter scaling coefficient
B = 410.7       # Data scaling coefficient
alpha = 0.34    # Parameter scaling exponent
beta = 0.28     # Data scaling exponent
L_inf = 1.69    # Irreducible loss

def chinchilla_loss(N, D):
    """Compute expected loss given N parameters and D tokens."""
    return A / N**alpha + B / D**beta + L_inf

def compute_flops(N, D):
    """Approximate training FLOPs: C ≈ 6ND."""
    return 6 * N * D

def optimal_allocation(C):
    """Find compute-optimal N, D for budget C FLOPs."""
    # Chinchilla ratio: D ≈ 20N, so C = 6*N*20N = 120*N^2
    N_opt = np.sqrt(C / 120)
    D_opt = 20 * N_opt
    return N_opt, D_opt

# Table of compute-optimal models
print("Compute-Optimal Model Configurations:")
print(f"{'Budget (FLOPs)':>18s} {'Params':>12s} {'Tokens':>12s} {'Loss':>8s}")
print("-" * 54)

for exp in range(19, 26):
    C = 10 ** exp
    N, D = optimal_allocation(C)
    L = chinchilla_loss(N, D)
    def fmt(x):
        if x >= 1e12: return f"{x/1e12:.1f}T"
        if x >= 1e9:  return f"{x/1e9:.1f}B"
        if x >= 1e6:  return f"{x/1e6:.0f}M"
        return f"{x:.0f}"
    print(f"{C:>18.0e} {fmt(N):>12s} {fmt(D):>12s} {L:>8.3f}")

# Compare: training a model with different token ratios
N_fixed = 7e9  # 7B parameter model
print(f"\\n7B model at different token budgets:")
print(f"{'Tokens':>12s} {'Ratio D/N':>10s} {'Loss':>8s} {'FLOPs':>14s}")
for ratio in [5, 10, 20, 50, 100, 200]:
    D = ratio * N_fixed
    L = chinchilla_loss(N_fixed, D)
    C = compute_flops(N_fixed, D)
    print(f"{D/1e9:>10.0f}B {ratio:>10d} {L:>8.3f} {C:>14.2e}")

# Kaplan vs Chinchilla comparison
print("\\nKaplan vs Chinchilla optimal allocation for C=10^23:")
C = 1e23
# Kaplan: N scales faster (N ∝ C^0.73)
N_kaplan = 1.3e10 * (C / 1e23) ** 0.73
D_kaplan = C / (6 * N_kaplan)
# Chinchilla
N_chin, D_chin = optimal_allocation(C)
print(f"  Kaplan:     N={N_kaplan/1e9:.1f}B, D={D_kaplan/1e9:.0f}B (ratio={D_kaplan/N_kaplan:.0f})")
print(f"  Chinchilla: N={N_chin/1e9:.1f}B, D={D_chin/1e9:.0f}B (ratio={D_chin/N_chin:.0f})")`,id:"scaling-laws-code"}),e.jsx(o,{title:"Scaling Laws Have Limitations",content:"Scaling laws are empirical fits that assume: (1) the architecture stays the same, (2) data quality is constant, (3) hyperparameters are well-tuned. They may not hold for architectural innovations, mixture-of-experts models, or when data quality changes. They also do not predict emergent capabilities -- abilities that appear suddenly at certain scales and are not captured by smooth power laws.",id:"scaling-limitations-warning"})]})}const R=Object.freeze(Object.defineProperty({__proto__:null,default:x},Symbol.toStringTag,{value:"Module"}));function k(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Distributed Training: Data, Tensor, and Pipeline Parallelism"}),e.jsx("p",{className:"text-lg text-gray-300",children:"Training large language models requires distributing computation across many GPUs. Three fundamental parallelism strategies -- data, tensor, and pipeline parallelism -- can be combined to train models that exceed the memory and compute capacity of any single device."}),e.jsx(t,{title:"Data Parallelism (DP)",definition:"Each GPU holds a complete copy of the model. The training batch is split across $P$ GPUs, each processing $B/P$ samples. Gradients are synchronized via all-reduce before the optimizer step. Communication cost per step: $O(|\\\\theta|)$ where $|\\\\theta|$ is the number of parameters.",notation:"Effective batch size = $B_{\\text{per\\\\_gpu}} \\times P$. With ZeRO optimization, model states (optimizer, gradients, parameters) can be sharded across GPUs.",id:"data-parallel-def"}),e.jsx(t,{title:"Tensor Parallelism (TP)",definition:"Individual layers are split across GPUs. For a linear layer $Y = XW$, the weight matrix $W \\in \\mathbb{R}^{d \\times d}$ is partitioned column-wise: $W = [W_1, W_2, \\ldots, W_P]$ across $P$ GPUs. Each GPU computes $Y_i = XW_i$ and results are combined via all-reduce.",notation:"Communication per layer: two all-reduce ops of size $O(B \\cdot d)$. Typically used within a single node (high-bandwidth NVLink).",id:"tensor-parallel-def"}),e.jsx(t,{title:"Pipeline Parallelism (PP)",definition:"Model layers are partitioned into $S$ stages across GPUs. Stage $s$ holds layers $[l_s, l_{s+1})$. Micro-batches flow through stages sequentially. GPipe and PipeDream schedule micro-batches to minimize bubble overhead (idle time): bubble ratio $\\approx (S-1)/(S-1+M)$ where $M$ is the number of micro-batches.",notation:"With $M$ micro-batches and $S$ stages, pipeline bubble fraction $\\approx (S-1)/M$.",id:"pipeline-parallel-def"}),e.jsx(n,{title:"3D Parallelism for a 70B Model",problem:"Design a parallelism strategy for training a 70B parameter model on 128 A100 GPUs (80GB each).",steps:[{formula:"\\text{Memory per GPU (FP16): } 70\\text{B} \\times 2 = 140\\text{GB (model only)}",explanation:"Model weights alone exceed single GPU memory. Need model parallelism."},{formula:"\\text{Tensor Parallel} = 8 \\text{ (within each node of 8 GPUs)}",explanation:"TP=8 splits each layer across 8 GPUs. Each holds 70B/8 = 8.75B params/GPU for weights."},{formula:"\\text{Pipeline Parallel} = 4 \\text{ (across 4 nodes)}",explanation:"PP=4 splits 80 layers into 4 stages of 20 layers each. Reduces per-GPU memory."},{formula:"\\text{Data Parallel} = 128 / (8 \\times 4) = 4",explanation:"Remaining GPUs do data parallelism. Total: TP=8 x PP=4 x DP=4 = 128 GPUs."}],id:"3d-parallel-example"}),e.jsx(i,{type:"tip",title:"ZeRO: Memory-Efficient Data Parallelism",content:"DeepSpeed ZeRO (Zero Redundancy Optimizer) eliminates memory redundancy in data parallelism. ZeRO-1 shards optimizer states (4x savings). ZeRO-2 also shards gradients (8x savings). ZeRO-3 shards parameters too, enabling training models that don't fit on any single GPU using only data parallelism. FSDP (Fully Sharded Data Parallel) in PyTorch implements similar concepts.",id:"zero-note"}),e.jsx(a,{title:"distributed_training.py",code:`import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Data Parallel training setup
def setup_ddp(rank, world_size):
    """Initialize distributed training."""
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

# Example: DDP wrapper
# model = MyModel().to(rank)
# model = DDP(model, device_ids=[rank])

# Using HuggingFace Accelerate for easy multi-GPU
from accelerate import Accelerator

def train_with_accelerate():
    accelerator = Accelerator()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Accelerate handles device placement and parallelism
    model, optimizer = accelerator.prepare(model, optimizer)

    print(f"Device: {accelerator.device}")
    print(f"Num processes: {accelerator.num_processes}")
    print(f"Mixed precision: {accelerator.mixed_precision}")

# DeepSpeed ZeRO configuration
deepspeed_config = {
    "train_batch_size": 256,
    "gradient_accumulation_steps": 8,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,             # ZeRO Stage 3: shard everything
        "offload_optimizer": {
            "device": "cpu"     # Offload optimizer states to CPU
        },
        "offload_param": {
            "device": "cpu"     # Offload params to CPU when not in use
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
    },
}
print("DeepSpeed ZeRO-3 config ready")

# Memory estimation
def estimate_memory(N_params, precision="fp16", zero_stage=0, num_gpus=1):
    """Estimate per-GPU memory for training."""
    bytes_per_param = 2 if precision == "fp16" else 4
    # Model weights
    model_mem = N_params * bytes_per_param
    # Gradients
    grad_mem = N_params * bytes_per_param
    # Optimizer states (AdamW: 2 states in FP32)
    opt_mem = N_params * 4 * 2  # momentum + variance

    total = model_mem + grad_mem + opt_mem
    if zero_stage >= 1:
        opt_mem /= num_gpus
    if zero_stage >= 2:
        grad_mem /= num_gpus
    if zero_stage >= 3:
        model_mem /= num_gpus

    per_gpu = (model_mem + grad_mem + opt_mem) / 1e9
    print(f"N={N_params/1e9:.0f}B, {precision}, ZeRO-{zero_stage}, "
          f"{num_gpus} GPUs -> {per_gpu:.1f} GB/GPU")
    return per_gpu

estimate_memory(7e9, "fp16", zero_stage=0, num_gpus=1)
estimate_memory(7e9, "fp16", zero_stage=3, num_gpus=8)
estimate_memory(70e9, "fp16", zero_stage=3, num_gpus=64)`,id:"distributed-code"}),e.jsx(o,{title:"Communication Overhead",content:"Distributed training adds communication overhead between GPUs. Tensor parallelism requires low-latency, high-bandwidth connections (NVLink within a node). Pipeline parallelism introduces bubble overhead. Data parallelism requires gradient all-reduce. Poor network topology or bandwidth can make scaling efficiency drop well below the ideal linear speedup.",id:"comm-overhead-warning"})]})}const O=Object.freeze(Object.defineProperty({__proto__:null,default:k},Symbol.toStringTag,{value:"Module"}));function b(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Mixed Precision Training: FP16 and BF16"}),e.jsx("p",{className:"text-lg text-gray-300",children:"Mixed precision training uses lower-precision floating-point formats (FP16 or BF16) for most operations while maintaining FP32 for critical accumulations. This halves memory usage and dramatically increases throughput on modern GPUs with tensor cores."}),e.jsx(t,{title:"Floating Point Formats",definition:"FP32 uses 1 sign, 8 exponent, 23 mantissa bits (range $\\pm 3.4 \\times 10^{38}$). FP16 uses 1 sign, 5 exponent, 10 mantissa bits (range $\\pm 65504$). BF16 uses 1 sign, 8 exponent, 7 mantissa bits (range $\\pm 3.4 \\times 10^{38}$, same as FP32).",notation:"BF16 has the same exponent range as FP32 but less precision. FP16 has more precision but much smaller range. BF16 is preferred for LLM training due to reduced overflow risk.",id:"fp-formats-def"}),e.jsx(n,{title:"Memory Savings with Mixed Precision",problem:"Calculate memory savings for a 7B parameter model with mixed precision.",steps:[{formula:"\\text{FP32 model: } 7\\text{B} \\times 4 \\text{ bytes} = 28 \\text{ GB}",explanation:"Full precision model weights alone take 28 GB."},{formula:"\\text{FP16/BF16 model: } 7\\text{B} \\times 2 \\text{ bytes} = 14 \\text{ GB}",explanation:"Half precision cuts model weight memory in half."},{formula:"\\text{Master weights (FP32): } 28 \\text{ GB (kept for optimizer step)}",explanation:"FP32 master copy maintained for accurate parameter updates."},{formula:"\\text{Total: } 14 \\text{ (forward/backward)} + 28 \\text{ (master)} = 42 \\text{ GB vs } 28 \\text{ GB (FP32 only)}",explanation:"But activations and gradients are also in FP16, giving net savings plus 2x faster matmuls on tensor cores."}],id:"memory-savings-example"}),e.jsx(s,{title:"Loss Scaling for FP16",statement:"FP16 has minimum subnormal value $\\approx 5.96 \\times 10^{-8}$. Gradients smaller than this underflow to zero. Loss scaling multiplies the loss by a scale factor $S$ before backpropagation: $\\tilde{g} = S \\cdot \\nabla_\\theta \\mathcal{L}$. After backprop, gradients are divided by $S$ before the optimizer step.",proof:"The chain rule preserves the scale factor: $\\frac{\\partial (S \\cdot \\mathcal{L})}{\\partial \\theta} = S \\cdot \\frac{\\partial \\mathcal{L}}{\\partial \\theta}$. By scaling up, small gradients move into FP16 representable range. Dynamic loss scaling starts with large $S$ and halves it when overflow (NaN/Inf) is detected, doubles it after $N$ successful steps.",id:"loss-scaling-thm"}),e.jsx(i,{type:"tip",title:"BF16 Simplifies Training",content:"BF16 (Brain Float 16) shares FP32's exponent range, so loss scaling is typically unnecessary. This eliminates the complexity of dynamic loss scaling and the risk of NaN gradients from overflow. BF16 is the default for modern LLM training on A100, H100, and newer GPUs. The trade-off is slightly less mantissa precision than FP16 (7 vs 10 bits).",id:"bf16-note"}),e.jsx(a,{title:"mixed_precision.py",code:`import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# Demonstrate precision formats
def show_precision():
    x = torch.tensor(3.141592653589793)
    print(f"FP32: {x.float().item():.10f}")
    print(f"FP16: {x.half().float().item():.10f}")
    print(f"BF16: {x.bfloat16().float().item():.10f}")

    # Overflow example
    big = torch.tensor(70000.0)
    print(f"\\nFP16 max: {torch.finfo(torch.float16).max}")
    print(f"70000 in FP16: {big.half()}")  # inf!
    print(f"70000 in BF16: {big.bfloat16()}")  # fine

    # Small gradient example
    tiny = torch.tensor(1e-8)
    print(f"\\n1e-8 in FP16: {tiny.half()}")  # 0!
    print(f"1e-8 in BF16: {tiny.bfloat16()}")
    scaled = tiny * 1024  # Loss scaling
    print(f"1e-8 * 1024 in FP16: {scaled.half()}")

show_precision()

# Mixed precision training with PyTorch AMP
class SimpleModel(nn.Module):
    def __init__(self, d=1024):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d),
            nn.LayerNorm(d),
        )

    def forward(self, x):
        return self.layers(x)

def train_mixed_precision():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()  # Dynamic loss scaling for FP16

    x = torch.randn(32, 1024, device=device)
    target = torch.randn(32, 1024, device=device)

    for step in range(5):
        optimizer.zero_grad()

        # Forward pass in FP16/BF16
        with autocast(device_type=device, dtype=torch.float16):
            output = model(x)
            loss = nn.functional.mse_loss(output, target)

        # Backward with loss scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # Unscale before grad clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        print(f"Step {step}: loss={loss.item():.4f}, scale={scaler.get_scale()}")

# HuggingFace Trainer with mixed precision
training_args_example = {
    "fp16": False,           # Set True for FP16
    "bf16": True,            # Preferred on A100/H100
    "bf16_full_eval": True,
    "tf32": True,            # TF32 for FP32 ops on Ampere+
    "half_precision_backend": "auto",
}
print(f"\\nRecommended HF training args: {training_args_example}")

if torch.cuda.is_available():
    train_mixed_precision()
else:
    print("\\n(CUDA required for mixed precision demo)")`,id:"mixed-precision-code"}),e.jsx(o,{title:"Numerical Instability with FP16",content:"FP16 training can suffer from overflow in attention logits (softmax of large values), underflow in gradients, and loss of precision in layer norm statistics. Always use loss scaling with FP16. Consider BF16 if your hardware supports it. For very large models, some operations (softmax, layer norm, loss computation) should remain in FP32 even in mixed precision mode.",id:"fp16-instability-warning"})]})}const W=Object.freeze(Object.defineProperty({__proto__:null,default:b},Symbol.toStringTag,{value:"Module"}));function y(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Gradient Checkpointing and Model Saving"}),e.jsx("p",{className:"text-lg text-gray-300",children:"Training large models requires careful memory management and fault tolerance. Gradient checkpointing trades compute for memory by recomputing activations during the backward pass. Model checkpointing saves training state periodically to enable recovery from hardware failures during multi-week training runs."}),e.jsx(t,{title:"Gradient Checkpointing (Activation Checkpointing)",definition:"Instead of storing all intermediate activations for backpropagation, gradient checkpointing stores activations only at selected checkpoint boundaries. During the backward pass, activations between checkpoints are recomputed from the nearest checkpoint. This reduces memory from $O(L)$ to $O(\\sqrt{L})$ at the cost of ~33% more compute.",notation:"For $L$ layers with checkpoints every $\\sqrt{L}$ layers: memory = $O(\\sqrt{L})$ activations, compute = one extra forward pass per segment.",id:"grad-checkpoint-def"}),e.jsx(s,{title:"Optimal Checkpoint Placement",statement:"For a sequential model with $L$ layers, placing checkpoints every $k$ layers gives memory $O(L/k + k)$ for activations. The optimal spacing is $k^* = \\sqrt{L}$, yielding minimum memory $O(\\sqrt{L})$ with compute overhead factor of at most 2x (one additional forward pass per segment).",proof:"Memory consists of: $L/k$ stored checkpoints plus $k$ recomputed activations in the longest segment. Total: $f(k) = L/k + k$. Setting $f'(k) = -L/k^2 + 1 = 0$ gives $k = \\sqrt{L}$, so $f(\\sqrt{L}) = 2\\sqrt{L}$.",id:"checkpoint-placement-thm"}),e.jsx(n,{title:"Memory Savings with Gradient Checkpointing",problem:"Estimate memory savings for a 7B model with 32 layers, batch size 4, sequence length 4096.",steps:[{formula:"\\text{Activation per layer} \\approx B \\times S \\times d \\times 2 = 4 \\times 4096 \\times 4096 \\times 2 \\approx 128 \\text{ MB}",explanation:"Each layer stores activations in FP16 (2 bytes). Hidden dim d=4096."},{formula:"\\text{Without checkpointing: } 32 \\times 128 = 4096 \\text{ MB} \\approx 4 \\text{ GB}",explanation:"All 32 layers store activations. This is just activations, not weights."},{formula:"\\text{With checkpointing (every } \\sqrt{32} \\approx 6 \\text{ layers): } 2 \\times \\sqrt{32} \\times 128 \\approx 1.4 \\text{ GB}",explanation:"Only checkpoint layers plus one segment of recomputed activations in memory."}],id:"checkpoint-savings-example"}),e.jsx(i,{type:"tip",title:"Selective Checkpointing",content:"Not all layers use equal memory for activations. Attention layers store QKV projections and attention weights (O(S^2) per head), while MLP layers store intermediate activations. Modern implementations checkpoint attention layers selectively, since they are the primary memory bottleneck, while keeping MLP activations that are cheaper to store.",id:"selective-checkpoint-note"}),e.jsx(a,{title:"checkpointing.py",code:`import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Gradient checkpointing with PyTorch
class TransformerBlock(nn.Module):
    def __init__(self, d_model=1024):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x

class CheckpointedModel(nn.Module):
    def __init__(self, n_layers=32, d_model=1024, use_checkpoint=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model) for _ in range(n_layers)]
        )
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # Recompute activations in backward pass
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

# Compare memory usage
def measure_memory(use_checkpoint):
    if not torch.cuda.is_available():
        return 0
    torch.cuda.reset_peak_memory_stats()
    model = CheckpointedModel(
        n_layers=16, d_model=512, use_checkpoint=use_checkpoint
    ).cuda()
    x = torch.randn(2, 256, 512).cuda()
    out = model(x)
    out.sum().backward()
    peak = torch.cuda.max_memory_allocated() / 1e9
    return peak

if torch.cuda.is_available():
    mem_no_ckpt = measure_memory(False)
    mem_ckpt = measure_memory(True)
    print(f"Without checkpointing: {mem_no_ckpt:.2f} GB peak")
    print(f"With checkpointing:    {mem_ckpt:.2f} GB peak")
    print(f"Savings: {(1 - mem_ckpt/mem_no_ckpt)*100:.1f}%")
else:
    print("Memory comparison requires CUDA")

# HuggingFace gradient checkpointing
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.gradient_checkpointing_enable()
print(f"\\nGradient checkpointing enabled: {model.is_gradient_checkpointing}")

# Model saving and loading checkpoints
def save_training_checkpoint(model, optimizer, scheduler, step, path):
    """Save full training state for recovery."""
    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "rng_state": torch.random.get_rng_state(),
    }
    torch.save(checkpoint_dict, path)
    print(f"Saved checkpoint at step {step} to {path}")

# Checkpoint every N steps during training
print("\\nTypical checkpointing schedule:")
print("  Every 1000 steps: save full checkpoint (~recovery)")
print("  Every 100 steps: save lightweight metrics/logs")
print("  Keep last 3-5 checkpoints to save disk space")`,id:"checkpointing-code"}),e.jsx(o,{title:"Checkpoint Storage Can Be Enormous",content:"A 70B model checkpoint with optimizer states takes ~800GB (model 140GB + optimizer 560GB in FP32 + gradients 140GB). Saving every 1000 steps over a 100K step run produces 100 checkpoints = 80TB. Use sharded saving (save each GPU's shard separately), keep only recent checkpoints, and use asynchronous I/O to avoid stalling training.",id:"checkpoint-storage-warning"})]})}const I=Object.freeze(Object.defineProperty({__proto__:null,default:y},Symbol.toStringTag,{value:"Module"}));function w(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Training Stability: Loss Spikes and Debugging"}),e.jsx("p",{className:"text-lg text-gray-300",children:"Large-scale pretraining runs are plagued by instabilities: loss spikes, divergence, and NaN gradients. Understanding the causes and having mitigation strategies is essential for successful multi-week training runs costing millions of dollars."}),e.jsx(t,{title:"Loss Spike",definition:"A loss spike is a sudden, large increase in training loss that disrupts the learning trajectory. Common causes include: bad data batches (corrupted or extremely out-of-distribution), learning rate too high, numerical overflow in FP16, or gradient explosion. Loss spikes can be recoverable or lead to permanent divergence.",id:"loss-spike-def"}),e.jsx(n,{title:"Common Stability Issues and Fixes",problem:"Diagnose and fix common training instabilities.",steps:[{formula:"\\text{Issue: Gradient explosion} \\rightarrow ||g|| > 10^3",explanation:"Fix: Gradient clipping (typically max_norm=1.0). Also check for learning rate warmup."},{formula:"\\text{Issue: Loss spike from bad batch} \\rightarrow \\mathcal{L}_t > 3 \\cdot \\text{EMA}(\\mathcal{L})",explanation:"Fix: Skip parameter update for outlier batches. Log the batch for investigation."},{formula:"\\text{Issue: NaN in attention} \\rightarrow \\text{softmax overflow}",explanation:"Fix: Use BF16 instead of FP16, or apply attention score capping (e.g., cap at 30.0 before softmax)."},{formula:"\\text{Issue: Slow divergence} \\rightarrow \\text{loss plateau then increase}",explanation:"Fix: Reduce learning rate, increase warmup, check for data leakage or repetition."}],id:"stability-issues-example"}),e.jsx(i,{type:"historical",title:"Lessons from Large-Scale Training",content:"The PaLM paper (2022) reported loss spikes during training and resolved them by rewinding to a checkpoint ~100 steps before the spike and skipping ~200-500 data batches. The OPT-175B logbook documented numerous instabilities including hardware failures, loss spikes, and divergence. Meta's LLaMA team reported that the 65B model training required manual intervention for loss spikes.",id:"stability-history"}),e.jsx(s,{title:"Gradient Clipping",statement:"Gradient clipping by global norm scales the gradient vector when its norm exceeds a threshold: $\\hat{g} = g \\cdot \\min\\left(1, \\frac{c}{||g||_2}\\right)$ where $c$ is the clipping threshold. This bounds the step size without changing the gradient direction.",proof:"When $||g||_2 \\leq c$: $\\hat{g} = g$ (no change). When $||g||_2 > c$: $||\\hat{g}||_2 = ||g|| \\cdot c/||g|| = c$, so the gradient is rescaled to exactly norm $c$. The direction $\\hat{g}/||\\hat{g}|| = g/||g||$ is preserved.",id:"grad-clip-thm"}),e.jsx(a,{title:"training_stability.py",code:`import torch
import torch.nn as nn
import numpy as np

class TrainingMonitor:
    """Monitor and detect training instabilities."""

    def __init__(self, window_size=100, spike_threshold=3.0):
        self.losses = []
        self.grad_norms = []
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self.spikes = []

    def log(self, step, loss, grad_norm):
        self.losses.append(loss)
        self.grad_norms.append(grad_norm)

        # Detect loss spike
        if len(self.losses) > self.window_size:
            recent = self.losses[-self.window_size:-1]
            ema = np.mean(recent)
            if loss > self.spike_threshold * ema:
                self.spikes.append(step)
                return "SPIKE_DETECTED"

        # Detect NaN
        if np.isnan(loss) or np.isinf(loss):
            return "NAN_DETECTED"

        # Detect gradient explosion
        if grad_norm > 100.0:
            return "GRAD_EXPLOSION"

        return "OK"

    def summary(self):
        print(f"Total steps: {len(self.losses)}")
        print(f"Loss spikes: {len(self.spikes)} at steps {self.spikes}")
        print(f"Max grad norm: {max(self.grad_norms):.2f}")
        print(f"Final loss: {self.losses[-1]:.4f}")

# Simulate training with instabilities
monitor = TrainingMonitor()
np.random.seed(42)

for step in range(500):
    # Normal loss decrease with noise
    base_loss = 4.0 * np.exp(-step / 200) + 2.0
    noise = np.random.normal(0, 0.05)
    loss = base_loss + noise

    # Inject a loss spike at step 200
    if step == 200:
        loss *= 5.0
    # Inject gradient explosion at step 350
    grad_norm = np.random.exponential(1.0)
    if step == 350:
        grad_norm = 500.0

    status = monitor.log(step, loss, grad_norm)
    if status != "OK":
        print(f"Step {step}: {status} (loss={loss:.3f}, grad_norm={grad_norm:.1f})")

monitor.summary()

# Practical stability techniques
class StableTrainer:
    """Training loop with stability measures."""

    def __init__(self, model, optimizer, max_grad_norm=1.0, skip_threshold=5.0):
        self.model = model
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.skip_threshold = skip_threshold
        self.loss_ema = None
        self.ema_decay = 0.99

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(**batch)
        loss = outputs.loss

        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN/Inf loss detected! Skipping batch.")
            return None

        # Check for loss spike
        if self.loss_ema is not None:
            if loss.item() > self.skip_threshold * self.loss_ema:
                print(f"Loss spike: {loss.item():.2f} vs EMA {self.loss_ema:.2f}")
                return None

        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )

        # Check gradient norm
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print("NaN gradient norm! Skipping update.")
            return None

        self.optimizer.step()

        # Update EMA
        l = loss.item()
        if self.loss_ema is None:
            self.loss_ema = l
        else:
            self.loss_ema = self.ema_decay * self.loss_ema + (1 - self.ema_decay) * l

        return {"loss": l, "grad_norm": grad_norm.item()}

print("\\nKey stability settings:")
print("  grad_clip=1.0, warmup=2000, wd=0.1, beta2=0.95, bf16=True")`,id:"stability-code"}),e.jsx(o,{title:"Recovery from Divergence",content:"When training diverges (loss explodes or goes to NaN), the standard recovery procedure is: (1) rewind to the last good checkpoint, (2) skip the problematic data batches, (3) optionally reduce the learning rate by 10-50%, (4) resume training. For persistent instability, consider reducing model size, increasing batch size, or switching to BF16. Each recovery attempt costs hours of compute.",id:"divergence-recovery-warning"}),e.jsx(i,{type:"tip",title:"Pre-LN vs Post-LN",content:"Pre-Layer Normalization (applying LayerNorm before attention/MLP) is significantly more stable than Post-LN. Most modern LLMs use Pre-LN with RMSNorm. QK-norm further stabilizes attention at large scale.",id:"preln-note"})]})}const H=Object.freeze(Object.defineProperty({__proto__:null,default:w},Symbol.toStringTag,{value:"Module"}));function v(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Encoder-Only Models: The BERT Family"}),e.jsx("p",{className:"text-lg text-gray-300",children:"Encoder-only architectures use bidirectional self-attention where every token attends to every other token. They excel at understanding tasks (classification, NER, question answering) but cannot generate text autoregressively."}),e.jsx(t,{title:"Encoder-Only Architecture",definition:"An encoder-only Transformer processes an input sequence $x = (x_1, \\ldots, x_n)$ through $L$ layers of bidirectional self-attention and feed-forward networks. The attention mask is fully visible: $M_{ij} = 1$ for all $i, j$. Output is a contextualized representation $h_i \\in \\mathbb{R}^d$ for each position.",notation:"$H = \\text{TransformerEncoder}(X + P)$ where $H \\in \\mathbb{R}^{n \\times d}$ and attention is unrestricted.",id:"encoder-only-def"}),e.jsx(n,{title:"BERT Family Evolution",problem:"Trace the evolution of encoder-only models from BERT to modern variants.",steps:[{formula:"\\text{BERT (2018): 110M/340M, MLM+NSP, WordPiece}",explanation:"Original bidirectional pretraining. Revolutionized NLP benchmarks."},{formula:"\\text{RoBERTa (2019): Same arch, no NSP, dynamic masking, 10x data}",explanation:"Showed BERT was undertrained. Longer training, bigger batches, more data."},{formula:"\\text{ALBERT (2019): Factorized embeddings, cross-layer sharing}",explanation:"Reduced parameters 18x. Replaced NSP with Sentence Order Prediction."},{formula:"\\text{DeBERTa (2020): Disentangled attention, enhanced decoding}",explanation:"Separate content and position embeddings. SOTA on SuperGLUE. DeBERTa-v3 uses ELECTRA-style training."},{formula:"\\text{ELECTRA (2020): Replaced token detection, not MLM}",explanation:"Generator creates corrupted tokens, discriminator detects them. Trains on ALL tokens, not just 15%."}],id:"bert-family-example"}),e.jsx(i,{type:"intuition",title:"Why Bidirectional Attention Helps Understanding",content:"Consider disambiguating 'bank' in 'I went to the bank to deposit money' vs 'I sat by the river bank'. A left-to-right model processing 'bank' hasn't seen 'deposit' or 'river' yet. A bidirectional model sees the full context simultaneously, making disambiguation trivial. This is why encoder-only models dominate classification and extraction tasks.",id:"bidirectional-intuition"}),e.jsx(a,{title:"encoder_only_models.py",code:`from transformers import (
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, pipeline
)
import torch

# Compare BERT family models
models_info = {
    "bert-base-uncased":          {"params": "110M", "vocab": 30522},
    "roberta-base":               {"params": "125M", "vocab": 50265},
    "microsoft/deberta-v3-base":  {"params": "184M", "vocab": 128100},
}

text = "The quick brown fox jumps over the lazy dog."
for name, info in models_info.items():
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        inputs = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
        hidden = out.last_hidden_state
        print(f"{name}:")
        print(f"  Params: {info['params']}, Vocab: {info['vocab']}")
        print(f"  Hidden shape: {hidden.shape}")
        print(f"  CLS repr norm: {hidden[0, 0].norm():.2f}")
    except Exception as e:
        print(f"{name}: {e}")

# Task: Sentiment classification with BERT
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
results = classifier([
    "This movie was absolutely fantastic!",
    "I was disappointed by the poor acting.",
    "The weather is nice today.",
])
for r in results:
    print(f"  {r['label']}: {r['score']:.4f}")

# Task: Named Entity Recognition
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
entities = ner("Elon Musk founded SpaceX in Hawthorne, California.")
for e in entities:
    print(f"  {e['entity_group']}: {e['word']} ({e['score']:.3f})")

# ELECTRA: Replaced Token Detection
from transformers import ElectraForPreTraining, ElectraTokenizer

tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
model = ElectraForPreTraining.from_pretrained("google/electra-small-discriminator")

sentence = "The chef cooked a delicious meal"
fake = "The chef cooked a electric meal"  # 'electric' is a replaced token
inputs = tokenizer(fake, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predictions = (logits.squeeze() > 0).int()
tokens = tokenizer.tokenize(fake)
print("\\nELECTRA replaced token detection:")
for tok, pred in zip(tokens, predictions[1:-1]):
    label = "REPLACED" if pred == 1 else "original"
    print(f"  {tok:>12s}: {label}")`,id:"encoder-only-code"}),e.jsx(o,{title:"Encoder-Only Models Are Declining in Popularity",content:"Since 2022, decoder-only models have dominated both understanding and generation tasks. Models like GPT-4, LLaMA, and Claude use decoder-only architectures yet match or exceed encoder-only models on classification and NLU benchmarks. Encoder-only models remain valuable for efficient embedding, retrieval, and token-level tasks where bidirectional context and smaller model size are advantages.",id:"encoder-decline-warning"})]})}const U=Object.freeze(Object.defineProperty({__proto__:null,default:v},Symbol.toStringTag,{value:"Module"}));function T(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Decoder-Only Models: The GPT Family"}),e.jsx("p",{className:"text-lg text-gray-300",children:"Decoder-only architectures use causal (left-to-right) self-attention where each token can only attend to preceding tokens. This enables autoregressive generation and has become the dominant architecture for large language models."}),e.jsx(t,{title:"Decoder-Only Architecture",definition:"A decoder-only Transformer applies causal self-attention: $M_{ij} = \\mathbb{1}[j \\leq i]$. Position $i$ attends only to positions $1, \\ldots, i$. This enables autoregressive generation where $P(x_t \\mid x_{<t})$ is computed from the hidden state at position $t$. The same architecture serves both training (teacher forcing on full sequences) and inference (token-by-token generation).",notation:"$h_t = \\text{CausalTransformer}(x_1, \\ldots, x_t)_t$, $P(x_{t+1}) = \\text{softmax}(W_{\\text{LM}} h_t)$.",id:"decoder-only-def"}),e.jsx(n,{title:"GPT Family Evolution",problem:"Trace the evolution of decoder-only models from GPT-1 to modern LLMs.",steps:[{formula:"\\text{GPT-1 (2018): 117M, 12 layers, BookCorpus}",explanation:"Showed unsupervised pretraining + supervised fine-tuning works across NLP tasks."},{formula:"\\text{GPT-2 (2019): 1.5B, 48 layers, WebText}",explanation:"Demonstrated zero-shot transfer. Generated coherent paragraphs. Not initially released."},{formula:"\\text{GPT-3 (2020): 175B, 96 layers, 300B tokens}",explanation:"In-context learning via prompting. Few-shot performance rivaled fine-tuned models."},{formula:"\\text{LLaMA (2023): 7-65B, RMSNorm, SwiGLU, RoPE}",explanation:"Open weights. Showed smaller models trained on more data can match larger ones."},{formula:"\\text{LLaMA-3 (2024): 8-405B, 128K vocab, 15T tokens, GQA}",explanation:"Trained far beyond Chinchilla-optimal. 405B rivaled GPT-4 on benchmarks."}],id:"gpt-family-example"}),e.jsx(i,{type:"note",title:"Modern Decoder-Only Architectural Innovations",content:"Modern decoder-only models incorporate several innovations beyond the original GPT: RMSNorm instead of LayerNorm, SwiGLU or GeGLU activation in MLP, Rotary Position Embeddings (RoPE) instead of learned positions, Grouped-Query Attention (GQA) for efficient inference, and Pre-LN (normalization before sublayers). These changes improve training stability, inference speed, and long-context performance.",id:"modern-decoder-note"}),e.jsx(s,{title:"Training Efficiency of Causal LM",statement:"A causal LM training step on a sequence of length $T$ produces $T-1$ loss terms (one per position except the first), giving $T-1$ gradient signals per sequence. This is more data-efficient per token than MLM which only computes loss on ~15% of tokens ($0.15T$ signals per sequence).",proof:"Causal LM: $\\mathcal{L} = -\\frac{1}{T-1}\\sum_{t=2}^T \\log P(x_t|x_{<t})$, yielding $T-1$ cross-entropy terms. MLM: $\\mathcal{L} = -\\frac{1}{|\\mathcal{M}|}\\sum_{i \\in \\mathcal{M}} \\log P(x_i|x_{\\backslash\\mathcal{M}})$ with $|\\mathcal{M}| \\approx 0.15T$. Ratio: $(T-1)/(0.15T) \\approx 6.7\\times$ more training signal per sequence for causal LM.",id:"causal-efficiency-thm"}),e.jsx(a,{title:"decoder_only_models.py",code:`from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

# Compare decoder-only architectures
configs = {
    "GPT-2": "gpt2",
    "GPT-2 Medium": "gpt2-medium",
}

text = "The future of artificial intelligence"
for name, model_id in configs.items():
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    n_params = sum(p.numel() for p in model.parameters())

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    print(f"{name} ({model_id}):")
    print(f"  Layers: {config.n_layer}, Heads: {config.n_head}, "
          f"Hidden: {config.n_embd}")
    print(f"  Parameters: {n_params/1e6:.0f}M")
    print(f"  Loss: {outputs.loss.item():.3f}, PPL: {torch.exp(outputs.loss).item():.1f}")

# Modern architecture features (LLaMA-style)
print("\\n--- Modern Decoder Architecture (LLaMA-style) ---")
llama_config = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,       # GQA: 4 query groups per KV head
    "intermediate_size": 11008,      # SwiGLU MLP
    "rms_norm_eps": 1e-5,           # RMSNorm
    "rope_theta": 10000.0,          # RoPE base frequency
    "vocab_size": 32000,
    "max_position_embeddings": 4096,
}
for k, v in llama_config.items():
    print(f"  {k}: {v}")

# Generation with GPT-2
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

input_ids = tokenizer.encode("In a world where AI", return_tensors="pt")
with torch.no_grad():
    generated = model.generate(
        input_ids,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
    )
print(f"\\nGenerated: {tokenizer.decode(generated[0], skip_special_tokens=True)}")

# KV-Cache demonstration
with torch.no_grad():
    # First pass: compute all KV pairs
    out1 = model(input_ids, use_cache=True)
    past_kv = out1.past_key_values

    # Next token: only process the new token, reuse cached KV
    next_token = out1.logits[:, -1:, :].argmax(dim=-1)
    out2 = model(next_token, past_key_values=past_kv, use_cache=True)

    print(f"\\nKV-Cache:")
    print(f"  Layers cached: {len(past_kv)}")
    print(f"  KV shape per layer: {past_kv[0][0].shape}")  # [batch, heads, seq, d_head]
    print(f"  Cache memory: {sum(k.nelement() + v.nelement() for k, v in past_kv) * 4 / 1e6:.1f} MB")`,id:"decoder-only-code"}),e.jsx(o,{title:"Decoder-Only Models Are Inefficient Encoders",content:"Because causal attention prevents tokens from seeing future context, decoder-only models produce weaker representations for understanding tasks compared to bidirectional models of equal size. Workarounds include instruction tuning (asking the model to classify via generation) or using the last token's representation, but a dedicated encoder model will be more compute-efficient for embedding and classification tasks.",id:"decoder-encoding-warning"})]})}const K=Object.freeze(Object.defineProperty({__proto__:null,default:T},Symbol.toStringTag,{value:"Module"}));function $(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Encoder-Decoder Models: T5 and BART"}),e.jsx("p",{className:"text-lg text-gray-300",children:"Encoder-decoder architectures combine a bidirectional encoder with an autoregressive decoder connected via cross-attention. This design naturally handles sequence-to-sequence tasks: the encoder processes the input and the decoder generates the output."}),e.jsx(t,{title:"Encoder-Decoder Architecture",definition:"An encoder-decoder Transformer consists of: (1) an encoder with bidirectional self-attention producing representations $H_{\\text{enc}} = \\text{Encoder}(X)$, and (2) a decoder with causal self-attention plus cross-attention to $H_{\\text{enc}}$. The decoder generates output: $P(y_t \\mid y_{<t}, X) = \\text{Decoder}(y_{<t}, H_{\\text{enc}})_t$.",notation:"Cross-attention: $Q$ from decoder, $K, V$ from encoder. $\\text{CrossAttn}(Q, K_{\\text{enc}}, V_{\\text{enc}}) = \\text{softmax}(QK_{\\text{enc}}^T / \\sqrt{d})V_{\\text{enc}}$.",id:"enc-dec-def"}),e.jsx(n,{title:"T5: Text-to-Text Transfer Transformer",problem:"Understand how T5 frames all NLP tasks as text-to-text problems.",steps:[{formula:'\\text{Classification: } \\text{"sentiment: This movie is great"} \\rightarrow \\text{"positive"}',explanation:"T5 generates the label as text. Unified framework for all tasks."},{formula:'\\text{Translation: } \\text{"translate English to French: Hello"} \\rightarrow \\text{"Bonjour"}',explanation:"Task prefix tells the model what operation to perform."},{formula:'\\text{Summarization: } \\text{"summarize: [long article]"} \\rightarrow \\text{"[summary]"}',explanation:"Natural fit for encoder-decoder: encode long input, decode short output."},{formula:"\\text{Pretraining: span corruption with sentinel tokens}",explanation:"T5 masks random spans and replaces with <extra_id_N>. Model generates the missing spans."}],id:"t5-tasks-example"}),e.jsx(i,{type:"historical",title:"BART: Denoising Sequence-to-Sequence",content:"BART (Lewis et al., 2019) uses a different pretraining strategy: it corrupts text with various noise functions (token masking, deletion, infilling, permutation, rotation) and trains the decoder to reconstruct the original. This denoising objective is more flexible than T5's span corruption and gives BART strong performance on both generation and comprehension tasks.",id:"bart-note"}),e.jsx(s,{title:"Span Corruption Objective (T5)",statement:"Given input $x$, sample spans to corrupt with total masked tokens $\\approx 15\\%$. Replace each span with a unique sentinel token $\\langle s_i \\rangle$. The target is the concatenation of sentinel tokens followed by the masked tokens: $y = \\langle s_1 \\rangle t_{1,1} \\ldots t_{1,k_1} \\langle s_2 \\rangle t_{2,1} \\ldots$. The loss is $\\mathcal{L} = -\\sum_t \\log P(y_t \\mid y_{<t}, \\tilde{x})$.",proof:"Span corruption is more compute-efficient than token-level MLM because the target sequence is much shorter than the input (only the corrupted spans). T5 found mean span length 3 with 15% corruption rate optimal. The encoder processes the corrupted input bidirectionally, and the decoder generates only the missing spans.",id:"span-corruption-thm"}),e.jsx(a,{title:"encoder_decoder_models.py",code:`from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    AutoModelForSeq2SeqLM, AutoTokenizer
)
import torch

# T5: Text-to-Text model
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# T5 for multiple tasks using text prefixes
tasks = [
    "translate English to German: The house is wonderful.",
    "summarize: State authorities dispatched emergency crews Tuesday to "
    "fight wildfires that have burned across the state.",
    "stsb sentence1: The cat sat on the mat. sentence2: A cat is sitting on a mat.",
]

print("=== T5 Multi-Task ===")
for task_input in tasks:
    input_ids = t5_tokenizer(task_input, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = t5_model.generate(input_ids, max_new_tokens=50)
    result = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input:  {task_input[:60]}...")
    print(f"Output: {result}\\n")

# T5 Span Corruption (pretraining objective)
def simulate_span_corruption(text, tokenizer, mask_ratio=0.15, mean_span=3):
    """Simulate T5's span corruption pretraining objective."""
    tokens = tokenizer.tokenize(text)
    n = len(tokens)
    n_mask = max(1, int(n * mask_ratio))

    # Select spans
    masked = set()
    sentinel_id = 0
    spans = []
    while len(masked) < n_mask and sentinel_id < 100:
        import random
        start = random.randint(0, n - 1)
        length = min(random.randint(1, mean_span * 2), n - start)
        span_tokens = []
        for i in range(start, start + length):
            if i not in masked:
                masked.add(i)
                span_tokens.append(tokens[i])
        if span_tokens:
            spans.append((sentinel_id, span_tokens))
            sentinel_id += 1

    # Build corrupted input and target
    corrupted = []
    in_span = False
    current_sentinel = 0
    for i, tok in enumerate(tokens):
        if i in masked:
            if not in_span:
                corrupted.append(f"<extra_id_{current_sentinel}>")
                current_sentinel += 1
                in_span = True
        else:
            in_span = False
            corrupted.append(tok)

    target = []
    for sid, stoks in spans:
        target.append(f"<extra_id_{sid}>")
        target.extend(stoks)

    return " ".join(corrupted), " ".join(target)

text = "The quick brown fox jumps over the lazy dog in the park"
corrupted, target = simulate_span_corruption(text, t5_tokenizer)
print("=== Span Corruption ===")
print(f"Original:  {text}")
print(f"Corrupted: {corrupted}")
print(f"Target:    {target}")

# BART for summarization
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
print(f"\\n=== Architecture Comparison ===")
print(f"T5-small:   enc={t5_model.config.num_layers}L, "
      f"dec={t5_model.config.num_decoder_layers}L, "
      f"d={t5_model.config.d_model}, "
      f"params={sum(p.numel() for p in t5_model.parameters())/1e6:.0f}M")
print(f"BART-base:  enc={bart_model.config.encoder_layers}L, "
      f"dec={bart_model.config.decoder_layers}L, "
      f"d={bart_model.config.d_model}, "
      f"params={sum(p.numel() for p in bart_model.parameters())/1e6:.0f}M")`,id:"enc-dec-code"}),e.jsx(o,{title:"Encoder-Decoder Has Higher Inference Cost for Long Inputs",content:"The encoder must process the full input before any decoding begins. For interactive use cases (chatbots), this creates higher time-to-first-token latency. Additionally, encoder-decoder models have roughly 2x the parameters of a decoder-only model with equivalent decoder capacity. This is why modern LLMs have largely shifted to decoder-only architectures despite the theoretical elegance of encoder-decoder designs.",id:"enc-dec-cost-warning"})]})}const V=Object.freeze(Object.defineProperty({__proto__:null,default:$},Symbol.toStringTag,{value:"Module"}));function P(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Choosing the Right Architecture"}),e.jsx("p",{className:"text-lg text-gray-300",children:"Selecting between encoder-only, decoder-only, and encoder-decoder architectures depends on your task, latency requirements, model scale, and deployment constraints. Each architecture has distinct strengths that make it optimal for specific use cases."}),e.jsx(t,{title:"Architecture Selection Criteria",definition:"The three key dimensions for architecture selection are: (1) Task type: understanding (classification, retrieval) vs. generation (text, code), (2) Input-output structure: fixed-length classification vs. variable-length generation vs. sequence-to-sequence transformation, (3) Scale and efficiency: parameter count, inference latency, throughput requirements.",id:"selection-criteria-def"}),e.jsx(n,{title:"Task-Architecture Mapping",problem:"Match common NLP tasks to their optimal architecture.",steps:[{formula:"\\text{Text Classification, NER, Retrieval} \\rightarrow \\text{Encoder-Only (BERT, DeBERTa)}",explanation:"Bidirectional context gives best representations for understanding. Small models (110-340M) suffice. Fast inference."},{formula:"\\text{Open-ended Generation, Chat, Code} \\rightarrow \\text{Decoder-Only (GPT, LLaMA)}",explanation:"Autoregressive generation is natural. Scales well with parameters. In-context learning eliminates task-specific fine-tuning."},{formula:"\\text{Translation, Summarization} \\rightarrow \\text{Encoder-Decoder (T5, mBART)}",explanation:"Natural fit: encode source, decode target. Cross-attention aligns input-output. Best for structured transformations."},{formula:"\\text{General-purpose / Unknown tasks} \\rightarrow \\text{Decoder-Only (large LLM)}",explanation:"Large decoder-only models handle nearly all tasks via prompting. Default choice when task diversity is high."}],id:"task-mapping-example"}),e.jsx(i,{type:"tip",title:"Decision Framework",content:"Ask these questions in order: (1) Do you need to generate text? If no, use encoder-only for efficiency. (2) Is the task a structured input-to-output transformation (translation, summarization)? If yes, consider encoder-decoder. (3) Do you need flexibility across many tasks or conversational interaction? Use decoder-only. (4) Are you constrained on model size (<500M params)? Encoder-only models offer the best quality-per-parameter for understanding tasks.",id:"decision-framework-note"}),e.jsx(s,{title:"Compute Efficiency Comparison",statement:"For a fixed parameter budget $N$ and sequence length $S$: Encoder-only processes the full sequence in one pass: $C_{\\text{enc}} = 2NS$ FLOPs. Decoder-only with KV-cache generates $T$ tokens: $C_{\\text{dec}} = 2NS + 2NT$ FLOPs (prefill + generation). Encoder-decoder with input $S$ and output $T$: $C_{\\text{enc-dec}} = 2N_{\\text{enc}}S + 2N_{\\text{dec}}T + C_{\\text{cross}}$.",proof:"Each Transformer forward pass costs approximately $2N$ FLOPs per token (each parameter participates in one multiply-add). For encoding $S$ tokens: $2NS$. For generating $T$ tokens with KV-cache: each new token costs $2N$ FLOPs, so total generation is $2NT$. Encoder-decoder has separate parameter budgets for encoder and decoder, plus cross-attention overhead.",id:"compute-comparison-thm"}),e.jsx(a,{title:"architecture_comparison.py",code:`from transformers import (
    AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    AutoTokenizer, pipeline
)
import torch
import time

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def benchmark_inference(model_name, model_class, task_input, generate=False):
    """Benchmark model inference."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(task_input, return_tensors="pt", truncation=True, max_length=512)
    params = count_params(model)

    start = time.perf_counter()
    with torch.no_grad():
        if generate:
            out = model.generate(inputs["input_ids"], max_new_tokens=20)
            result = tokenizer.decode(out[0], skip_special_tokens=True)
        else:
            out = model(**inputs)
            result = f"hidden shape: {out.last_hidden_state.shape}"
    elapsed = time.perf_counter() - start

    return {
        "model": model_name,
        "params": f"{params/1e6:.0f}M",
        "time_ms": f"{elapsed*1000:.0f}",
        "result": result[:80],
    }

# Compare architectures
text = "Natural language processing has been transformed by deep learning."
comparisons = [
    ("bert-base-uncased", AutoModel, text, False),
    ("gpt2", AutoModelForCausalLM, text, True),
    ("t5-small", AutoModelForSeq2SeqLM, "summarize: " + text, True),
]

print("=== Architecture Comparison ===")
print(f"{'Model':>25s} {'Params':>8s} {'Time':>8s}  Result")
print("-" * 90)
for model_name, model_cls, inp, gen in comparisons:
    try:
        result = benchmark_inference(model_name, model_cls, inp, gen)
        print(f"{result['model']:>25s} {result['params']:>8s} "
              f"{result['time_ms']:>6s}ms  {result['result']}")
    except Exception as e:
        print(f"{model_name:>25s}: {str(e)[:60]}")

# Architecture recommendation engine
def recommend_architecture(task_type, scale, latency_ms, generates_text):
    """Recommend architecture based on requirements."""
    recommendations = []

    if not generates_text:
        if scale == "small":
            recommendations.append(("Encoder-Only (BERT/DeBERTa)", 0.95))
        else:
            recommendations.append(("Encoder-Only (large)", 0.80))
            recommendations.append(("Decoder-Only (prompted)", 0.70))
    elif task_type in ["translation", "summarization"]:
        recommendations.append(("Encoder-Decoder (T5/BART)", 0.90))
        recommendations.append(("Decoder-Only (prompted)", 0.75))
    elif task_type in ["chat", "code", "general"]:
        recommendations.append(("Decoder-Only (LLaMA/GPT)", 0.95))
    else:
        recommendations.append(("Decoder-Only (general)", 0.80))
        recommendations.append(("Encoder-Decoder", 0.60))

    return sorted(recommendations, key=lambda x: -x[1])

# Example recommendations
scenarios = [
    {"task_type": "classification", "scale": "small", "latency_ms": 10, "generates_text": False},
    {"task_type": "chat", "scale": "large", "latency_ms": 500, "generates_text": True},
    {"task_type": "translation", "scale": "medium", "latency_ms": 200, "generates_text": True},
    {"task_type": "retrieval", "scale": "small", "latency_ms": 5, "generates_text": False},
]

print("\\n=== Architecture Recommendations ===")
for s in scenarios:
    recs = recommend_architecture(**s)
    print(f"\\nTask: {s['task_type']}, Scale: {s['scale']}, Generates: {s['generates_text']}")
    for arch, score in recs:
        print(f"  {score:.0%} -> {arch}")

print("\\n=== Summary ===")
print("Encoder-only:    Best for embeddings, classification, retrieval")
print("Decoder-only:    Best for generation, chat, general-purpose LLMs")
print("Encoder-decoder: Best for structured seq2seq (translation, summarization)")`,id:"comparison-code"}),e.jsx(o,{title:"The Convergence Trend",content:"The distinction between architectures is blurring. Large decoder-only models can perform classification via generation ('Is this positive or negative?'). Encoder-decoder models can do open-ended generation. The practical choice increasingly comes down to: use a large decoder-only model for versatility, or a small encoder-only model for efficient specialization. Pure encoder-decoder models are becoming rarer in new development.",id:"convergence-warning"}),e.jsx(i,{type:"note",title:"Hybrid Approaches",content:"Some architectures blur the lines: PrefixLM (like UL2) uses bidirectional attention on a prefix and causal attention on the rest. Mixture-of-Denoisers (MoD) trains with multiple objectives simultaneously. These approaches attempt to combine the strengths of different architectures in a single model.",id:"hybrid-note"})]})}const Q=Object.freeze(Object.defineProperty({__proto__:null,default:P},Symbol.toStringTag,{value:"Module"}));export{j as a,N as b,B as c,C as d,A as e,E as f,D as g,F as h,G as i,q as j,R as k,O as l,W as m,I as n,H as o,U as p,K as q,V as r,S as s,Q as t};
