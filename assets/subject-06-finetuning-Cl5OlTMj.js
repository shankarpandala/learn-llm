import{j as e}from"./vendor-DWbzdFaj.js";import{r as s}from"./vendor-katex-BYl39Yo6.js";import{D as i,E as n,N as t,P as a,W as r,T as o}from"./subject-01-text-fundamentals-DG6tAvii.js";function l(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Transfer Learning: The Pretrain-Finetune Paradigm"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Transfer learning is the foundational idea behind modern LLM development. A model is first pretrained on a massive unsupervised corpus to learn general language representations, then finetuned on a smaller task-specific dataset. This two-stage paradigm dramatically reduces the data and compute required for downstream tasks."}),e.jsx(i,{title:"Transfer Learning",definition:"Transfer learning is the technique of reusing a model trained on one task (the source task) as the starting point for a model on a different task (the target task). In NLP, the source task is typically language modeling on a large corpus, and the target task is a specific downstream application.",id:"def-transfer-learning"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The Two-Stage Pipeline"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Pretraining captures syntactic structure, world knowledge, and reasoning patterns from billions of tokens. Finetuning adapts these general capabilities to a narrow task using orders of magnitude less data. The pretrained weights serve as an informed initialization rather than random weights."}),e.jsx(n,{title:"Pretraining vs. Finetuning Costs",problem:"Compare the resources needed for pretraining versus finetuning a 7B parameter model.",steps:[{formula:"\\text{Pretraining: } \\sim 1\\text{T tokens}, \\sim 100\\text{K GPU-hours}",explanation:"Pretraining requires massive compute, data, and weeks of training on large clusters."},{formula:"\\text{Finetuning: } \\sim 10\\text{K-100K examples}, \\sim 10\\text{-100 GPU-hours}",explanation:"Finetuning uses a tiny fraction of the pretraining budget and can often run on a single node."},{formula:"\\text{Ratio} \\approx 1000\\times \\text{ less compute}",explanation:"This massive efficiency gain is why transfer learning transformed NLP."}],id:"example-cost-comparison"}),e.jsx(t,{type:"historical",title:"History of Transfer Learning in NLP",content:"Transfer learning in NLP gained momentum with ULMFiT (Howard & Ruder, 2018), which introduced discriminative fine-tuning and gradual unfreezing. BERT (Devlin et al., 2018) and GPT (Radford et al., 2018) cemented the pretrain-finetune paradigm. GPT-2 and GPT-3 later showed that sufficiently large pretrained models could perform tasks zero-shot.",id:"note-history"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Finetuning Strategies"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"There are several strategies for finetuning: updating all parameters (full finetuning), freezing lower layers and only training upper layers, or using learning rate schedules that treat different layers differently (discriminative finetuning)."}),e.jsx(a,{title:"full_finetuning_hf.py",code:`from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

# Load pretrained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

# Load downstream dataset
dataset = load_dataset("imdb")

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized = dataset.map(tokenize_fn, batched=True)

# Finetuning configuration
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,          # Much smaller LR than pretraining
    weight_decay=0.01,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

# Finetune: all parameters are updated
trainer.train()`,id:"code-full-finetuning"}),e.jsx(r,{title:"Learning Rate Sensitivity",content:"Finetuning requires a much smaller learning rate than pretraining (typically 1e-5 to 5e-5). Using a pretraining-scale learning rate (e.g., 1e-3) will destroy the pretrained representations in just a few steps, a phenomenon known as catastrophic forgetting.",id:"warning-lr"}),e.jsx(t,{type:"intuition",title:"Why Transfer Learning Works",content:"Lower layers of neural networks learn general features (syntax, word relationships) while upper layers learn task-specific patterns. By preserving the lower layers' knowledge, finetuning lets the model build on a rich foundation rather than learning from scratch. This is analogous to how a medical student builds on general biology knowledge.",id:"note-intuition"})]})}const P=Object.freeze(Object.defineProperty({__proto__:null,default:l},Symbol.toStringTag,{value:"Module"}));function d(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Task Heads: Classification and Token Classification"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"A task head is a lightweight neural network layer added on top of a pretrained backbone. Different downstream tasks require different head architectures. The most common are sequence classification heads (sentiment, topic) and token classification heads (NER, POS tagging). The backbone provides rich contextualized representations while the head maps them to task-specific outputs."}),e.jsx(i,{title:"Task Head",definition:"A task head is a small neural network (typically one or two linear layers) appended to a pretrained encoder or decoder. It projects the hidden representations $h \\in \\mathbb{R}^d$ to the label space $\\mathbb{R}^C$ where $C$ is the number of classes. For sequence classification, the [CLS] token representation is used; for token classification, every token's representation is projected independently.",id:"def-task-head"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Sequence Classification"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Sequence classification assigns a single label to an entire input sequence. The model typically uses a pooled representation (the [CLS] token in BERT-style models) and feeds it through a linear classifier."}),e.jsx(n,{title:"Classification Head Architecture",problem:"Describe the forward pass of a sequence classification head on top of BERT.",steps:[{formula:"h_{\\text{CLS}} = \\text{BERT}(x)[0][:,0,:]",explanation:"Extract the [CLS] token hidden state from the last layer, shape (batch, hidden_dim)."},{formula:"h_{\\text{drop}} = \\text{Dropout}(h_{\\text{CLS}}, p=0.1)",explanation:"Apply dropout for regularization during finetuning."},{formula:"\\text{logits} = W h_{\\text{drop}} + b, \\quad W \\in \\mathbb{R}^{C \\times d}",explanation:"Project to the number of classes C via a linear layer."},{formula:"\\mathcal{L} = \\text{CrossEntropy}(\\text{logits}, y)",explanation:"Compute cross-entropy loss against ground-truth labels."}],id:"example-cls-head"}),e.jsx(a,{title:"sequence_classification.py",code:`from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2-class sentiment classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

# Inspect the classification head
print(model.classifier)
# Linear(in_features=768, out_features=2, bias=True)

# Forward pass
inputs = tokenizer("This movie was fantastic!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    print(f"Logits: {logits}")
    print(f"Probabilities: {probs}")
    print(f"Predicted class: {torch.argmax(probs, dim=-1).item()}")`,id:"code-seq-cls"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Token Classification"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Token classification assigns a label to each token independently. This is used for Named Entity Recognition (NER), Part-of-Speech tagging, and extractive question answering. Every token's hidden state is projected through the classification head."}),e.jsx(a,{title:"token_classification_ner.py",code:`from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline

model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# The token classification head: projects each token to label space
print(model.classifier)
# Linear(in_features=1024, out_features=9, bias=True)
# 9 labels: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC

# Use pipeline for easy inference
ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
results = ner("Hugging Face is based in New York City.")
for entity in results:
    print(f"  {entity['word']:15s} | {entity['entity_group']:5s} | score={entity['score']:.3f}")`,id:"code-token-cls"}),e.jsx(t,{type:"tip",title:"Subword Alignment for Token Classification",content:"When using subword tokenizers, a single word may be split into multiple tokens. For token classification, labels are typically assigned only to the first subword of each word. The remaining subwords are ignored in loss computation using a special label index (commonly -100 in PyTorch).",id:"note-subword-alignment"}),e.jsx(r,{title:"Head Initialization Matters",content:"The task head is initialized randomly while the backbone is pretrained. This mismatch means the randomly-initialized head produces large, noisy gradients early in training that can disrupt the pretrained backbone. Using a warmup schedule and small learning rate for the backbone helps mitigate this.",id:"warning-head-init"})]})}const A=Object.freeze(Object.defineProperty({__proto__:null,default:d},Symbol.toStringTag,{value:"Module"}));function p(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Catastrophic Forgetting and Elastic Weight Consolidation"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"When a neural network is finetuned on a new task, it tends to forget what it learned during pretraining or on previous tasks. This phenomenon, known as catastrophic forgetting, is a fundamental challenge in continual learning. Elastic Weight Consolidation (EWC) is a principled approach that uses the Fisher information matrix to protect important weights."}),e.jsx(i,{title:"Catastrophic Forgetting",definition:"Catastrophic forgetting occurs when a neural network, upon learning new information, abruptly loses previously learned knowledge. Formally, after training on task $A$ then task $B$, performance on task $A$ degrades significantly because gradient updates for $B$ overwrite parameters critical to $A$.",id:"def-catastrophic-forgetting"}),e.jsx(r,{title:"Real-World Impact",content:"Catastrophic forgetting is not just a theoretical concern. When finetuning an LLM for code generation, the model may lose its ability to hold a conversation. When finetuning for medical Q&A, it may forget basic reasoning. This makes naive sequential finetuning on multiple tasks unreliable.",id:"warning-real-impact"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Elastic Weight Consolidation (EWC)"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"EWC adds a regularization term that penalizes changes to parameters that were important for previous tasks. Importance is measured by the diagonal of the Fisher information matrix, which approximates the curvature of the loss landscape."}),e.jsx(o,{title:"EWC Objective",statement:`The EWC loss for task $B$, given a model previously trained on task $A$ with optimal parameters $\\theta^*_A$, is:
$$\\mathcal{L}_{EWC}(\\theta) = \\mathcal{L}_B(\\theta) + \\frac{\\lambda}{2} \\sum_i F_i (\\theta_i - \\theta^*_{A,i})^2$$
where $F_i$ is the diagonal element of the Fisher information matrix for parameter $i$, measuring its importance to task $A$.`,proof:"Starting from a Bayesian perspective, we want $\\log p(\\theta | \\mathcal{D}_A, \\mathcal{D}_B)$. By Bayes' rule, $\\log p(\\theta | \\mathcal{D}_A, \\mathcal{D}_B) = \\log p(\\mathcal{D}_B | \\theta) + \\log p(\\theta | \\mathcal{D}_A) - \\log p(\\mathcal{D}_B)$. Approximating $\\log p(\\theta | \\mathcal{D}_A)$ with a Laplace approximation (second-order Taylor around $\\theta^*_A$) gives the Fisher-weighted quadratic penalty.",id:"thm-ewc"}),e.jsx(n,{title:"Fisher Information Diagonal",problem:"Compute the Fisher information for a simple model parameter.",steps:[{formula:"F_i = \\mathbb{E}\\left[\\left(\\frac{\\partial \\log p(y|x,\\theta)}{\\partial \\theta_i}\\right)^2\\right]",explanation:"The Fisher information measures how sensitive the log-likelihood is to changes in parameter i."},{formula:"F_i \\approx \\frac{1}{N} \\sum_{n=1}^{N} \\left(\\frac{\\partial \\log p(y_n|x_n,\\theta^*_A)}{\\partial \\theta_i}\\right)^2",explanation:"In practice, approximate with the empirical Fisher over N data points from task A."},{formula:"\\text{High } F_i \\Rightarrow \\text{parameter } i \\text{ is important for task A}",explanation:"Parameters with high Fisher values are strongly penalized when they deviate from their task-A values."}],id:"example-fisher"}),e.jsx(a,{title:"ewc_implementation.py",code:`import torch
import torch.nn.functional as F
from copy import deepcopy

class EWC:
    """Elastic Weight Consolidation for continual learning."""

    def __init__(self, model, dataloader, device='cuda', num_samples=200):
        self.model = model
        self.device = device
        # Store optimal parameters from task A
        self.params_a = {n: p.clone().detach()
                         for n, p in model.named_parameters() if p.requires_grad}
        # Compute Fisher information diagonal
        self.fisher = self._compute_fisher(dataloader, num_samples)

    def _compute_fisher(self, dataloader, num_samples):
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**inputs)
            loss = outputs.loss
            self.model.zero_grad()
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            count += 1
        # Average over samples
        for n in fisher:
            fisher[n] /= count
        return fisher

    def penalty(self, model, lam=1000):
        """Compute EWC penalty: (lambda/2) * sum_i F_i * (theta_i - theta_A_i)^2"""
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params_a[n]) ** 2).sum()
        return (lam / 2) * loss

# Usage in training loop:
# total_loss = task_b_loss + ewc.penalty(model, lam=5000)`,id:"code-ewc"}),e.jsx(t,{type:"tip",title:"Alternatives to EWC",content:"Other approaches to mitigating catastrophic forgetting include: (1) Replay-based methods that mix old task data into new training, (2) Progressive Networks that add new capacity for each task, (3) Knowledge Distillation where the finetuned model is regularized to match the original model's outputs, and (4) Parameter-efficient methods like LoRA that keep the original weights frozen entirely.",id:"note-alternatives"}),e.jsx(t,{type:"intuition",title:"The Geometry of Forgetting",content:"Think of the loss landscape as a mountainous terrain. The pretrained model sits in a valley that works well for general language. Finetuning moves the model to a new valley for the specific task. If these valleys are far apart in parameter space, the model loses its general capabilities. EWC keeps the model close to the original valley along dimensions that matter most.",id:"note-geometry"})]})}const z=Object.freeze(Object.defineProperty({__proto__:null,default:p},Symbol.toStringTag,{value:"Module"}));function m(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Multi-Task Learning for Language Models"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Multi-task learning (MTL) trains a single model on multiple tasks simultaneously, allowing tasks to share representations and benefit from each other's training signal. Models like T5 and FLAN demonstrated that casting diverse NLP tasks into a unified text-to-text format enables powerful multi-task generalization."}),e.jsx(i,{title:"Multi-Task Learning",definition:"Multi-task learning is a training paradigm where a model is optimized on multiple objectives simultaneously. The combined loss is $\\mathcal{L} = \\sum_{t=1}^{T} \\alpha_t \\mathcal{L}_t$ where $\\alpha_t$ are task weights and $\\mathcal{L}_t$ is the loss for task $t$. Shared parameters learn representations that generalize across tasks.",id:"def-mtl"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The Text-to-Text Framework"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"T5 (Raffel et al., 2020) unified all NLP tasks into a text-to-text format. Classification becomes generating a label word, translation becomes generating the translated text, and summarization becomes generating a summary. This eliminates the need for task-specific heads."}),e.jsx(n,{title:"Text-to-Text Task Formatting",problem:"Show how different NLP tasks are formatted as text-to-text problems.",steps:[{formula:'\\text{Sentiment: } \\texttt{"classify: I love this!"} \\rightarrow \\texttt{"positive"}',explanation:"Classification is cast as generating the label token."},{formula:'\\text{Translation: } \\texttt{"translate en-fr: Hello"} \\rightarrow \\texttt{"Bonjour"}',explanation:"Translation uses a language-pair prefix."},{formula:'\\text{Summarize: } \\texttt{"summarize: [article]"} \\rightarrow \\texttt{"[summary]"}',explanation:"Summarization generates a condensed version."},{formula:'\\text{NLI: } \\texttt{"nli premise: ... hypothesis: ..."} \\rightarrow \\texttt{"entailment"}',explanation:"Natural Language Inference outputs the relationship label."}],id:"example-t2t"}),e.jsx(t,{type:"historical",title:"From T5 to FLAN",content:"T5 (2020) showed the power of the text-to-text framework. FLAN (2022) scaled this to 1,836 tasks and demonstrated that instruction-tuned models generalize to unseen tasks far better than models trained on individual tasks. FLAN-T5 and FLAN-PaLM became strong baselines showing that multi-task instruction tuning is a critical step in building capable LLMs.",id:"note-t5-flan"}),e.jsx(a,{title:"multitask_training.py",code:`from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, concatenate_datasets

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load multiple tasks
sst2 = load_dataset("glue", "sst2", split="train[:2000]")
mnli = load_dataset("glue", "mnli", split="train[:2000]")

# Format as text-to-text
def format_sst2(example):
    label = "positive" if example["label"] == 1 else "negative"
    return {"input_text": f"classify sentiment: {example['sentence']}",
            "target_text": label}

def format_mnli(example):
    labels = ["entailment", "neutral", "contradiction"]
    return {"input_text": f"nli premise: {example['premise']} hypothesis: {example['hypothesis']}",
            "target_text": labels[example["label"]]}

sst2_fmt = sst2.map(format_sst2, remove_columns=sst2.column_names)
mnli_fmt = mnli.map(format_mnli, remove_columns=mnli.column_names)

# Combine datasets (interleave for balanced training)
combined = concatenate_datasets([sst2_fmt, mnli_fmt]).shuffle(seed=42)

def tokenize_fn(examples):
    model_inputs = tokenizer(examples["input_text"], max_length=256, truncation=True)
    labels = tokenizer(examples["target_text"], max_length=32, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = combined.map(tokenize_fn, batched=True)

training_args = TrainingArguments(
    output_dir="./multitask-flan-t5",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=3e-5,
    warmup_steps=100,
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
trainer.train()`,id:"code-multitask"}),e.jsx(r,{title:"Task Balancing is Critical",content:"When tasks have very different dataset sizes, the model tends to overfit on smaller tasks and underfit on larger ones. Strategies include temperature-based sampling (sampling each task with probability proportional to its size raised to a temperature), equal mixing ratios, or dynamic task weighting based on validation performance.",id:"warning-task-balance"}),e.jsx(t,{type:"intuition",title:"Why Multi-Task Helps",content:"Multi-task learning acts as a regularizer: each task provides a different training signal that prevents the model from overfitting to any single task's idiosyncrasies. Tasks that require similar skills (e.g., NLI and reading comprehension both need reasoning) provide complementary supervision that improves shared representations.",id:"note-why-mtl"})]})}const F=Object.freeze(Object.defineProperty({__proto__:null,default:m},Symbol.toStringTag,{value:"Module"}));function c(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"LoRA: Low-Rank Adaptation"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Low-Rank Adaptation (LoRA) is a parameter-efficient finetuning method that freezes the pretrained model weights and injects trainable low-rank decomposition matrices into each layer. This dramatically reduces the number of trainable parameters while achieving performance comparable to full finetuning."}),e.jsx(i,{title:"Low-Rank Adaptation (LoRA)",definition:"LoRA modifies the forward pass of a pretrained weight matrix $W_0 \\in \\mathbb{R}^{d \\times k}$ by adding a low-rank update: $W = W_0 + BA$ where $B \\in \\mathbb{R}^{d \\times r}$ and $A \\in \\mathbb{R}^{r \\times k}$ with rank $r \\ll \\min(d, k)$. Only $A$ and $B$ are trained; $W_0$ remains frozen. The forward pass becomes $h = W_0 x + BAx$, and $A$ is initialized with a random Gaussian while $B$ is initialized to zero so that $BA = 0$ at the start of training.",id:"def-lora"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The Rank Decomposition"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The key insight behind LoRA is that weight updates during finetuning have low intrinsic rank. Instead of updating the full ",e.jsx(s.InlineMath,{math:"d \\times k"})," matrix, we parameterize the update as the product of two much smaller matrices."]}),e.jsx(n,{title:"Parameter Savings with LoRA",problem:"Compare trainable parameters for full finetuning vs. LoRA on a weight matrix of size 4096 x 4096 with rank r = 16.",steps:[{formula:"\\text{Full: } d \\times k = 4096 \\times 4096 = 16{,}777{,}216",explanation:"Full finetuning updates all 16.7M parameters in this single matrix."},{formula:"\\text{LoRA: } d \\times r + r \\times k = 4096 \\times 16 + 16 \\times 4096 = 131{,}072",explanation:"LoRA only trains the two low-rank matrices, totaling 131K parameters."},{formula:"\\text{Reduction: } \\frac{131{,}072}{16{,}777{,}216} = 0.78\\%",explanation:"LoRA uses less than 1% of the parameters while capturing the essential adaptation."}],id:"example-param-savings"}),e.jsx(o,{title:"LoRA Forward Pass",statement:`For an input $x$, the LoRA-adapted linear layer computes:
$$h = W_0 x + \\frac{\\alpha}{r} BAx$$
where $\\alpha$ is a scaling hyperparameter that controls the magnitude of the low-rank update relative to the pretrained weights. The ratio $\\frac{\\alpha}{r}$ normalizes the update so that changing $r$ does not require retuning the learning rate.`,id:"thm-lora-forward"}),e.jsx(a,{title:"lora_with_peft.py",code:`from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                        # Rank of the decomposition
    lora_alpha=32,               # Scaling factor (alpha/r applied)
    lora_dropout=0.05,           # Dropout on LoRA layers
    target_modules=[             # Which modules to apply LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",                 # Don't train bias terms
)

# Wrap model with LoRA adapters
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# trainable params: 13,631,488 || all params: 6,751,334,400 || trainable%: 0.2019

# The base weights are frozen, only LoRA matrices are trained
for name, param in model.named_parameters():
    if "lora" in name:
        print(f"TRAINABLE: {name}, shape={param.shape}")
    # All other params have requires_grad=False`,id:"code-lora-peft"}),e.jsx(t,{type:"intuition",title:"Why Low Rank Works",content:"Research by Aghajanyan et al. (2021) showed that pretrained models have a low intrinsic dimensionality: the effective parameter space needed for adaptation is much smaller than the full parameter count. LoRA exploits this by constraining updates to a low-rank subspace. Rank r = 4 to 64 typically suffices; increasing r beyond this shows diminishing returns.",id:"note-low-rank-intuition"}),e.jsx(r,{title:"LoRA Rank Selection",content:"Choosing rank r too low limits the model's ability to adapt (underfitting the task). Choosing r too high increases memory and compute without proportional gains. A good practice is to start with r = 16 and tune from there. Also, applying LoRA to all linear layers (not just attention) often improves results at modest additional cost.",id:"warning-rank-selection"}),e.jsx(t,{type:"tip",title:"Merging LoRA Weights",content:"After training, LoRA weights can be merged back into the base model: W_merged = W_0 + BA. This produces a standard model with zero inference overhead. Multiple LoRA adapters for different tasks can be swapped at serving time without reloading the base model, enabling efficient multi-tenant deployments.",id:"note-merging"})]})}const $=Object.freeze(Object.defineProperty({__proto__:null,default:c},Symbol.toStringTag,{value:"Module"}));function h(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Adapter Layers: Bottleneck Architecture"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Adapter modules are small bottleneck layers inserted within each transformer block. During finetuning, only these adapter parameters are trained while the original model weights remain frozen. This was one of the first parameter-efficient finetuning methods, introduced by Houlsby et al. (2019)."}),e.jsx(i,{title:"Adapter Layer",definition:"An adapter is a bottleneck module inserted into a transformer layer. It consists of a down-projection $W_{\\text{down}} \\in \\mathbb{R}^{d \\times m}$, a nonlinearity $\\sigma$, and an up-projection $W_{\\text{up}} \\in \\mathbb{R}^{m \\times d}$, plus a residual connection: $\\text{Adapter}(h) = h + W_{\\text{up}} \\, \\sigma(W_{\\text{down}} h)$ where $m \\ll d$ is the bottleneck dimension.",id:"def-adapter"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Bottleneck Architecture"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The bottleneck compresses the hidden representation to a lower dimension, applies a nonlinearity, then projects back up. The residual connection ensures the adapter can learn the identity function (doing nothing) by initializing near zero."}),e.jsx(n,{title:"Adapter Parameter Count",problem:"Calculate the number of trainable parameters for adapters in a 12-layer transformer with hidden size 768 and bottleneck size 64.",steps:[{formula:"\\text{Per adapter: } 2 \\times d \\times m = 2 \\times 768 \\times 64 = 98{,}304",explanation:"Each adapter has a down-projection and up-projection (ignoring biases)."},{formula:"\\text{Adapters per layer: } 2 \\text{ (after attention + after FFN)}",explanation:"The standard placement inserts adapters after both the self-attention and feed-forward sublayers."},{formula:"\\text{Total: } 12 \\times 2 \\times 98{,}304 = 2{,}359{,}296",explanation:"About 2.4M trainable parameters vs. ~110M total for BERT-base, roughly 2.1%."}],id:"example-adapter-params"}),e.jsx(a,{title:"adapter_implementation.py",code:`import torch
import torch.nn as nn

class AdapterLayer(nn.Module):
    """Bottleneck adapter module with residual connection."""

    def __init__(self, hidden_size, bottleneck_size=64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        # Initialize near zero so adapter starts as identity
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, hidden_states):
        residual = hidden_states
        x = self.down_proj(hidden_states)
        x = self.activation(x)
        x = self.up_proj(x)
        return residual + x

# Example: inject adapter into a transformer layer
class TransformerLayerWithAdapter(nn.Module):
    def __init__(self, original_layer, hidden_size, bottleneck_size=64):
        super().__init__()
        self.original_layer = original_layer
        self.adapter_attn = AdapterLayer(hidden_size, bottleneck_size)
        self.adapter_ffn = AdapterLayer(hidden_size, bottleneck_size)
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, hidden_states, **kwargs):
        # Original self-attention
        attn_output = self.original_layer.attention(hidden_states, **kwargs)
        attn_output = self.adapter_attn(attn_output)  # Adapter after attention
        # Original feed-forward
        ffn_output = self.original_layer.feed_forward(attn_output)
        ffn_output = self.adapter_ffn(ffn_output)     # Adapter after FFN
        return ffn_output

# Using HuggingFace adapters library
from adapters import AutoAdapterModel
model = AutoAdapterModel.from_pretrained("bert-base-uncased")
model.add_adapter("sentiment", config="pfeiffer")  # Pfeiffer adapter config
model.train_adapter("sentiment")                    # Freeze all except adapter`,id:"code-adapter"}),e.jsx(t,{type:"historical",title:"Adapter Variants",content:"Houlsby et al. (2019) proposed inserting two adapters per layer (after attention and FFN). Pfeiffer et al. (2021) showed that a single adapter after the FFN sublayer works nearly as well while halving the compute overhead. AdapterFusion (Pfeiffer et al., 2021) introduced a mechanism to combine multiple task adapters via attention.",id:"note-variants"}),e.jsx(r,{title:"Inference Latency Overhead",content:"Unlike LoRA, adapter layers cannot be merged into the base model. They add sequential computation at every layer during inference, introducing a small but measurable latency overhead (typically 5-10%). For latency-sensitive applications, LoRA is often preferred since its weights can be folded into the base model at zero inference cost.",id:"warning-latency"}),e.jsx(t,{type:"tip",title:"Choosing Bottleneck Size",content:"The bottleneck dimension m controls the capacity-efficiency tradeoff. For simple tasks like sentiment classification, m = 16 to 32 often suffices. For complex tasks like question answering or generation, m = 64 to 256 may be needed. Monitor validation performance to find the sweet spot.",id:"note-bottleneck-size"})]})}const R=Object.freeze(Object.defineProperty({__proto__:null,default:h},Symbol.toStringTag,{value:"Module"}));function u(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Prefix Tuning and Soft Prompts"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Prefix tuning and prompt tuning are parameter-efficient methods that prepend trainable continuous vectors (soft prompts) to the input or hidden states, steering the frozen model's behavior without modifying any of its original parameters. Instead of engineering discrete text prompts, the model learns optimal prompt embeddings through gradient descent."}),e.jsx(i,{title:"Prefix Tuning",definition:"Prefix tuning (Li & Liang, 2021) prepends trainable prefix vectors $P_k, P_v \\in \\mathbb{R}^{l \\times d}$ to the key and value matrices at every transformer layer, where $l$ is the prefix length and $d$ is the hidden dimension. The attention computation becomes $\\text{Attn}(Q, [P_k; K], [P_v; V])$, allowing the prefix to influence all subsequent token representations without modifying model weights.",id:"def-prefix-tuning"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Prompt Tuning vs. Prefix Tuning"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Prompt tuning (Lester et al., 2021) is a simplified variant that only prepends trainable embeddings to the input layer, rather than all layers. It uses fewer parameters but requires larger models to match prefix tuning's performance. At the scale of 10B+ parameters, prompt tuning matches full finetuning performance."}),e.jsx(n,{title:"Parameter Comparison: Prefix vs. Prompt Tuning",problem:"Compare trainable parameters for a 12-layer model with hidden size 768 and prefix length 20.",steps:[{formula:"\\text{Prompt tuning: } l \\times d = 20 \\times 768 = 15{,}360",explanation:"Only input-layer embeddings are trained, giving very few parameters."},{formula:"\\text{Prefix tuning: } 2 \\times L \\times l \\times d = 2 \\times 12 \\times 20 \\times 768 = 368{,}640",explanation:"Key and value prefixes at all 12 layers (2x for K and V)."},{formula:"\\text{Prefix is } 24\\times \\text{ more params, but still } < 0.4\\% \\text{ of BERT-base}",explanation:"Both methods are extremely parameter-efficient compared to the full model."}],id:"example-prefix-params"}),e.jsx(a,{title:"prefix_tuning_peft.py",code:`from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PrefixTuningConfig, get_peft_model, TaskType

model_name = "gpt2-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prefix tuning configuration
prefix_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,        # Length of the prefix
    encoder_hidden_size=1024,     # Hidden size of the reparameterization MLP
    prefix_projection=True,       # Use MLP to reparameterize prefix
)

model = get_peft_model(model, prefix_config)
model.print_trainable_parameters()
# trainable params: 983,040 || all params: 355,958,784 || trainable%: 0.28

# Prompt tuning (simpler variant)
from peft import PromptTuningConfig, PromptTuningInit

prompt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,
    prompt_tuning_init=PromptTuningInit.TEXT,       # Initialize from text
    prompt_tuning_init_text="Classify the sentiment of this text: ",
    tokenizer_name_or_path=model_name,
)

# The soft prompt is initialized from the embeddings of the init text
# then optimized via backpropagation during training`,id:"code-prefix-tuning"}),e.jsx(t,{type:"intuition",title:"What Do Soft Prompts Learn?",content:"Soft prompts live in the continuous embedding space and are not constrained to correspond to any real tokens. Analysis shows they often encode task-specific instructions in a form that is more expressive than any discrete text prompt could be. They can represent directions in embedding space that have no single-token equivalent, exploiting the model's representation geometry directly.",id:"note-soft-prompt-meaning"}),e.jsx(r,{title:"Prefix Length vs. Context Length",content:"Prefix tokens consume positions in the model's context window. A prefix of length 20 reduces your available context by 20 tokens. For tasks requiring long inputs (e.g., document summarization), keep the prefix short. Additionally, very long prefixes (100+) can lead to optimization difficulties without proportional performance gains.",id:"warning-prefix-length"}),e.jsx(t,{type:"tip",title:"Reparameterization Trick",content:"Directly optimizing prefix vectors can be unstable due to their high dimensionality relative to the small number of parameters. Prefix tuning uses a reparameterization MLP during training: the prefix is generated by a small feedforward network. After training, the MLP is discarded and only the generated prefix vectors are kept for inference.",id:"note-reparam"})]})}const S=Object.freeze(Object.defineProperty({__proto__:null,default:u},Symbol.toStringTag,{value:"Module"}));function f(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Comparing PEFT Methods"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Parameter-Efficient Fine-Tuning (PEFT) encompasses a family of methods that adapt large models by training only a small subset of parameters. Each method makes different tradeoffs between parameter count, inference overhead, implementation complexity, and task performance. Understanding these tradeoffs is essential for choosing the right approach."}),e.jsx(i,{title:"Parameter-Efficient Fine-Tuning (PEFT)",definition:"PEFT methods adapt a pretrained model by modifying or adding a small number of parameters $\\Delta\\theta$ while keeping the vast majority of original parameters $\\theta_0$ frozen. The effective model is $\\theta = \\theta_0 + \\Delta\\theta$ where $|\\Delta\\theta| \\ll |\\theta_0|$. Major families include additive (adapters, prefix tuning), reparameterization (LoRA), and selective (BitFit, sparse finetuning).",id:"def-peft"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Method Comparison"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The table below summarizes key properties of popular PEFT methods. The right choice depends on your constraints: if inference latency matters, choose LoRA (mergeable). If you need to switch tasks rapidly, adapters or LoRA adapters can be hot-swapped. If you want minimal parameters, prompt tuning uses the fewest."}),e.jsx(n,{title:"PEFT Methods at a Glance",problem:"Compare LoRA, Adapters, Prefix Tuning, and Prompt Tuning on a 7B parameter model.",steps:[{formula:"\\text{LoRA (r=16): } \\sim 0.2\\% \\text{ params, 0 inference overhead (merged)}",explanation:"LoRA matrices can be absorbed into the base weights, making inference identical to the original model."},{formula:"\\text{Adapters (m=64): } \\sim 0.5\\% \\text{ params, 5-10\\% latency overhead}",explanation:"Adapter layers add sequential computation that cannot be removed at inference time."},{formula:"\\text{Prefix Tuning (l=20): } \\sim 0.1\\% \\text{ params, minor KV cache overhead}",explanation:"Prefix vectors consume context window positions but add minimal compute."},{formula:"\\text{Prompt Tuning (l=20): } \\sim 0.002\\% \\text{ params, minimal overhead}",explanation:"Fewest parameters but requires very large models (10B+) to match other methods."}],id:"example-comparison"}),e.jsx(a,{title:"peft_methods_benchmark.py",code:`from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig, PrefixTuningConfig, PromptTuningConfig,
    get_peft_model, TaskType
)

model_name = "meta-llama/Llama-2-7b-hf"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

# Count base model params
total_params = sum(p.numel() for p in base_model.parameters())
print(f"Base model: {total_params:,} parameters")

# Method 1: LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

# Method 2: Prefix Tuning
prefix_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20,
)

# Method 3: Prompt Tuning
prompt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20,
)

# Compare trainable parameters
configs = {
    "LoRA (r=16)": lora_config,
    "Prefix Tuning (l=20)": prefix_config,
    "Prompt Tuning (l=20)": prompt_config,
}

for name, config in configs.items():
    model = get_peft_model(base_model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / total_params
    print(f"{name:30s}: {trainable:>12,} trainable ({pct:.4f}%)")
    del model  # Free memory

# Typical results for Llama-2-7B:
# LoRA (r=16)                  :   13,631,488 trainable (0.2019%)
# Prefix Tuning (l=20)         :    3,276,800 trainable (0.0485%)
# Prompt Tuning (l=20)         :       81,920 trainable (0.0012%)`,id:"code-peft-benchmark"}),e.jsx(t,{type:"tip",title:"QLoRA: Quantized LoRA",content:"QLoRA (Dettmers et al., 2023) combines 4-bit quantization of the base model with LoRA adapters trained in full precision. This enables finetuning a 65B parameter model on a single 48GB GPU. The base model is loaded in 4-bit NormalFloat format while LoRA gradients flow in BFloat16, giving near-lossless quality at a fraction of the memory cost.",id:"note-qlora"}),e.jsx(r,{title:"No Free Lunch",content:"PEFT methods trade capacity for efficiency. On complex tasks requiring significant behavioral changes (e.g., teaching a model a new language), full finetuning may still outperform PEFT methods. Always benchmark on your specific task. Additionally, PEFT methods can be combined: LoRA on attention layers plus adapters on FFN layers sometimes outperforms either alone.",id:"warning-no-free-lunch"}),e.jsx(t,{type:"note",title:"Practical Recommendation",content:"For most practitioners in 2024-2025, LoRA (or QLoRA) is the default recommendation. It offers the best balance of simplicity, performance, and flexibility. Use rank r = 16-64, apply to all linear layers, and set alpha = 2r. If memory is extremely constrained, QLoRA with 4-bit quantization lets you finetune models much larger than your GPU would normally support.",id:"note-recommendation"})]})}const C=Object.freeze(Object.defineProperty({__proto__:null,default:f},Symbol.toStringTag,{value:"Module"}));function g(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Instruction Datasets: Alpaca, Dolly, and OpenAssistant"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Instruction tuning transforms a base language model into one that follows user instructions. The quality of this transformation depends critically on the training data. Several landmark datasets have shaped the instruction-tuning landscape, each with different approaches to data collection: synthetic generation, human annotation, and community crowdsourcing."}),e.jsx(i,{title:"Instruction Dataset",definition:"An instruction dataset is a collection of (instruction, input, output) triples used to train a model to follow natural language commands. Each example specifies what the model should do (instruction), optional context (input), and the desired response (output). The training objective is standard language modeling: $\\mathcal{L} = -\\sum_t \\log p(y_t | y_{<t}, \\text{instruction}, \\text{input})$ where the loss is computed only over the output tokens.",id:"def-instruction-dataset"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Key Datasets"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The instruction-tuning data landscape evolved rapidly from early synthetic datasets to large-scale human-annotated collections."}),e.jsx(n,{title:"Major Instruction Datasets",problem:"Compare the properties of Alpaca, Dolly, and OpenAssistant datasets.",steps:[{formula:"\\text{Alpaca (Stanford, 2023): 52K synthetic examples}",explanation:"Generated by prompting GPT-3.5 with 175 seed tasks. Low cost but inherits GPT biases and has quality issues."},{formula:"\\text{Dolly (Databricks, 2023): 15K human-written examples}",explanation:"Written by Databricks employees. Fully open-source with a commercial license. Higher quality but smaller."},{formula:"\\text{OpenAssistant (LAION, 2023): 161K messages in 35 languages}",explanation:"Community-crowdsourced multi-turn conversations with human preference rankings. Rich but noisier."},{formula:"\\text{FLAN Collection (Google, 2023): 1,836 tasks, millions of examples}",explanation:"Aggregation of academic NLP tasks formatted as instructions. Broadest coverage but less conversational."}],id:"example-datasets"}),e.jsx(a,{title:"loading_instruction_datasets.py",code:`from datasets import load_dataset

# Stanford Alpaca - synthetic instruction data
alpaca = load_dataset("tatsu-lab/alpaca", split="train")
print(f"Alpaca: {len(alpaca)} examples")
print(alpaca[0])
# {'instruction': 'Give three tips for staying healthy.',
#  'input': '',
#  'output': '1. Eat a balanced diet... 2. Exercise regularly... 3. Get enough sleep...'}

# Databricks Dolly - human-written, commercially licensed
dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
print(f"\\nDolly: {len(dolly)} examples")
print(f"Categories: {set(dolly['category'])}")
# {'brainstorming', 'classification', 'closed_qa', 'creative_writing',
#  'general_qa', 'information_extraction', 'open_qa', 'summarization'}

# OpenAssistant Conversations - multi-turn with rankings
oasst = load_dataset("OpenAssistant/oasst1", split="train")
print(f"\\nOpenAssistant: {len(oasst)} messages")
# Filter for top-ranked assistant responses
top_responses = oasst.filter(lambda x: x["role"] == "assistant" and x["rank"] == 0)
print(f"Top-ranked responses: {len(top_responses)}")

# Format Alpaca data for training
def format_alpaca(example):
    if example["input"]:
        text = f"### Instruction:\\n{example['instruction']}\\n\\n### Input:\\n{example['input']}\\n\\n### Response:\\n{example['output']}"
    else:
        text = f"### Instruction:\\n{example['instruction']}\\n\\n### Response:\\n{example['output']}"
    return {"text": text}

formatted = alpaca.map(format_alpaca)
print(f"\\nFormatted example:\\n{formatted[0]['text'][:200]}")`,id:"code-instruction-datasets"}),e.jsx(t,{type:"historical",title:"The Self-Instruct Pipeline",content:"Alpaca used the Self-Instruct method (Wang et al., 2023) where a strong model generates instruction-following data. Starting from 175 human-written seed tasks, GPT-3.5 was prompted to generate new instructions, inputs, and outputs. This bootstrapping approach costs under $500 but raises questions about model collapse and license contamination when using proprietary model outputs.",id:"note-self-instruct"}),e.jsx(r,{title:"Synthetic Data Risks",content:"Training on synthetic data generated by a proprietary model (like GPT-4) may violate terms of service and creates legal uncertainty. The resulting model may also inherit biases, hallucination patterns, and stylistic quirks from the teacher model. Empirically, models trained on synthetic data can appear fluent but lack true reasoning depth, a phenomenon called 'imitation learning collapse'.",id:"warning-synthetic"}),e.jsx(t,{type:"tip",title:"Building Your Own Dataset",content:"For domain-specific applications, curating 1,000-5,000 high-quality examples often outperforms using 50,000 noisy ones. Start with a small seed set, use the model to generate candidates, then have domain experts filter and correct them. This human-in-the-loop approach balances cost and quality effectively.",id:"note-build-own"})]})}const O=Object.freeze(Object.defineProperty({__proto__:null,default:g},Symbol.toStringTag,{value:"Module"}));function x(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Chat Templates: ChatML, System/User/Assistant"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Chat templates define the structured format for multi-turn conversations with language models. They separate messages by role (system, user, assistant) using special tokens or delimiters. Consistent formatting during training and inference is crucial: a mismatch between the template used during finetuning and during deployment leads to degraded performance."}),e.jsx(i,{title:"Chat Template",definition:"A chat template is a formatting specification that converts a list of role-tagged messages into a single string for model input. It defines how to encode message boundaries, role indicators, and special tokens. Common formats include ChatML (OpenAI), Llama-style, and Alpaca-style templates. The tokenizer's apply_chat_template method handles this conversion.",id:"def-chat-template"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Common Chat Formats"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Different model families use different chat formats. Using the wrong format at inference time is one of the most common causes of poor model performance in production."}),e.jsx(n,{title:"Chat Format Comparison",problem:"Show how the same conversation is formatted in ChatML and Llama-2 templates.",steps:[{formula:"\\text{ChatML: } \\texttt{<|im\\_start|>system} \\ldots \\texttt{<|im\\_end|>}",explanation:"ChatML uses <|im_start|>role and <|im_end|> delimiters. Used by many open models."},{formula:"\\text{Llama-2: } \\texttt{[INST] <<SYS>>} \\ldots \\texttt{<</SYS>>} \\ldots \\texttt{[/INST]}",explanation:"Llama-2 nests the system prompt inside the first [INST] block with <<SYS>> tags."},{formula:"\\text{Llama-3: } \\texttt{<|begin\\_of\\_text|><|start\\_header\\_id|>system<|end\\_header\\_id|>}",explanation:"Llama-3 switched to a cleaner header-based format with explicit role markers."}],id:"example-format-comparison"}),e.jsx(a,{title:"chat_templates.py",code:`from transformers import AutoTokenizer

# Define a multi-turn conversation
messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "How do I reverse a list in Python?"},
    {"role": "assistant", "content": "You can use list[::-1] or list.reverse()."},
    {"role": "user", "content": "What's the difference between the two?"},
]

# ChatML format (used by many models including Qwen, Mistral variants)
chatml_template = """{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}<|im_start|>assistant
"""

# Apply templates with different tokenizers
model_names = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"=== {model_name.split('/')[-1]} ===")
    print(formatted[:300])
    print()

# Custom template for training
# Mask system and user tokens; only compute loss on assistant responses
def create_training_labels(tokenizer, messages):
    """Create input_ids and labels with masked non-assistant tokens."""
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    input_ids = tokenizer.encode(full_text)
    labels = [-100] * len(input_ids)  # Mask everything initially

    # Unmask only assistant response tokens
    # (Implementation depends on the specific template format)
    return input_ids, labels`,id:"code-chat-templates"}),e.jsx(t,{type:"tip",title:"The Jinja2 Template System",content:"HuggingFace tokenizers use Jinja2 templates stored in tokenizer_config.json. You can inspect any model's template with tokenizer.chat_template. When finetuning, always use the same template the model was pretrained/instruction-tuned with. If training from a base model, pick a standard template (ChatML is a good default) and use it consistently.",id:"note-jinja"}),e.jsx(r,{title:"Template Mismatch",content:"Using the wrong chat template at inference is equivalent to feeding the model garbled input. A model trained with ChatML will not respond properly if prompted with Llama-2 format. Always verify the template matches the model. Common symptoms of mismatch: the model repeats the prompt, generates the wrong role's response, or produces incoherent output.",id:"warning-mismatch"}),e.jsx(t,{type:"note",title:"Loss Masking for Chat Training",content:"During supervised finetuning on chat data, the loss should only be computed on assistant response tokens. System prompts and user messages are part of the input context but should not contribute to the training loss. This is achieved by setting labels to -100 (the PyTorch ignore index) for all non-assistant tokens.",id:"note-loss-masking"})]})}const M=Object.freeze(Object.defineProperty({__proto__:null,default:x},Symbol.toStringTag,{value:"Module"}));function _(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Supervised Fine-Tuning (SFT)"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Supervised Fine-Tuning is the process of training a pretrained language model on curated instruction-response pairs using standard cross-entropy loss. SFT is the first step after pretraining in the modern LLM alignment pipeline, transforming a base model that merely predicts next tokens into one that helpfully follows instructions."}),e.jsx(i,{title:"Supervised Fine-Tuning (SFT)",definition:"SFT optimizes a pretrained model $\\pi_\\theta$ on a dataset of demonstrations $\\mathcal{D} = \\{(x_i, y_i)\\}$ where $x_i$ is an instruction and $y_i$ is the desired response. The objective is to minimize the negative log-likelihood: $\\mathcal{L}_{\\text{SFT}}(\\theta) = -\\mathbb{E}_{(x,y) \\sim \\mathcal{D}} \\left[ \\sum_{t=1}^{|y|} \\log \\pi_\\theta(y_t | x, y_{<t}) \\right]$ where the loss is computed only over response tokens $y$.",id:"def-sft"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The SFT Training Pipeline"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"A complete SFT pipeline involves data preparation, formatting with chat templates, tokenization with proper label masking, and training with carefully chosen hyperparameters. The TRL library from HuggingFace provides a streamlined SFTTrainer for this workflow."}),e.jsx(n,{title:"SFT Hyperparameter Guidelines",problem:"What are typical hyperparameters for SFT of a 7B model?",steps:[{formula:"\\text{Learning rate: } 1 \\times 10^{-5} \\text{ to } 2 \\times 10^{-5}",explanation:"Lower than pretraining LR to preserve pretrained knowledge."},{formula:"\\text{Epochs: } 1 \\text{ to } 3",explanation:"SFT datasets are small; more epochs risk overfitting and losing generalization."},{formula:"\\text{Batch size: effective } 128 \\text{ via gradient accumulation}",explanation:"Large effective batches stabilize training on diverse instruction types."},{formula:"\\text{Warmup: } 3\\text{-}10\\% \\text{ of total steps}",explanation:"Gradual warmup prevents early instability from the randomly-initialized output shift."}],id:"example-sft-hparams"}),e.jsx(a,{title:"sft_with_trl.py",code:`from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load instruction dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Format into chat-style prompts
def format_instruction(example):
    if example["input"]:
        text = (f"<|im_start|>user\\n{example['instruction']}\\n"
                f"Input: {example['input']}<|im_end|>\\n"
                f"<|im_start|>assistant\\n{example['output']}<|im_end|>")
    else:
        text = (f"<|im_start|>user\\n{example['instruction']}<|im_end|>\\n"
                f"<|im_start|>assistant\\n{example['output']}<|im_end|>")
    return {"text": text}

dataset = dataset.map(format_instruction)

# Optional: use LoRA for parameter efficiency
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# Configure SFT training
sft_config = SFTConfig(
    output_dir="./sft-llama2",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=32,    # Effective batch = 128
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_seq_length=2048,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    bf16=True,
    gradient_checkpointing=True,       # Save memory
    dataset_text_field="text",
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,           # Pass LoRA config for PEFT training
)

# Train
trainer.train()

# Save the model
trainer.save_model("./sft-llama2-final")`,id:"code-sft-trl"}),e.jsx(t,{type:"intuition",title:"SFT as Behavior Cloning",content:"SFT is essentially behavior cloning: the model learns to imitate the demonstrations in the dataset. This means the model can only be as good as the data. If the demonstrations contain errors, the model learns to reproduce those errors. This is why data quality is paramount and why SFT alone is insufficient: the model needs further alignment (RLHF/DPO) to go beyond mimicking demonstrations.",id:"note-behavior-cloning"}),e.jsx(r,{title:"Overfitting on Small Datasets",content:"SFT datasets are typically small (1K-100K examples) compared to pretraining data (trillions of tokens). Training for too many epochs causes the model to memorize responses verbatim rather than learning to generalize. Monitor the validation loss carefully; it often starts increasing after 1-2 epochs. Use dropout, weight decay, and early stopping.",id:"warning-overfitting"}),e.jsx(t,{type:"tip",title:"Packing for Efficiency",content:"Short examples waste compute due to padding. Packing concatenates multiple examples into a single sequence (separated by EOS tokens) to fill the context window. TRL's SFTTrainer supports this with the packing=True option. This can speed up training by 2-5x on datasets with variable-length examples.",id:"note-packing"})]})}const q=Object.freeze(Object.defineProperty({__proto__:null,default:_},Symbol.toStringTag,{value:"Module"}));function y(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Data Quality Over Quantity"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Research has repeatedly shown that a small, high-quality instruction dataset can outperform a much larger noisy one. The LIMA paper (Zhou et al., 2023) demonstrated that just 1,000 carefully curated examples can produce a strong instruction-following model. This section explores what makes instruction data high-quality and how to curate effective datasets."}),e.jsx(i,{title:"Data Quality for Instruction Tuning",definition:"High-quality instruction data exhibits: (1) diversity of tasks and instruction styles, (2) correctness and factual accuracy of responses, (3) appropriate response length and detail level, (4) consistent formatting, and (5) alignment with desired behavior. The LIMA hypothesis states that most of a model's knowledge comes from pretraining, and instruction tuning only needs to teach the format of interaction.",id:"def-data-quality"}),e.jsx(n,{title:"Quality vs. Quantity Evidence",problem:"Compare model performance with different dataset sizes and quality levels.",steps:[{formula:"\\text{LIMA: 1K curated} \\approx \\text{Alpaca: 52K synthetic}",explanation:"LIMA with 1,000 hand-picked examples matched or exceeded Alpaca with 52K synthetic examples."},{formula:"\\text{Quality filtering 52K} \\rightarrow \\text{9K high-quality} \\uparrow \\text{performance}",explanation:"Filtering Alpaca data with quality heuristics improved performance while reducing dataset size by 80%."},{formula:"\\text{Deita (2024): score-based selection from 300K} \\rightarrow \\text{6K best}",explanation:"Automatic quality scoring and diversity selection produced a tiny but highly effective subset."}],id:"example-quality-evidence"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"Quality Filtering Strategies"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Several automated and semi-automated approaches can identify high-quality examples from a larger pool, reducing human annotation effort while maintaining quality."}),e.jsx(a,{title:"data_quality_filtering.py",code:`from datasets import load_dataset
import numpy as np

dataset = load_dataset("tatsu-lab/alpaca", split="train")

def quality_score(example):
    """Heuristic quality scoring for instruction data."""
    score = 0.0
    instruction = example["instruction"]
    output = example["output"]

    # Length heuristics
    if 10 < len(instruction.split()) < 100:
        score += 1.0  # Reasonable instruction length
    if 20 < len(output.split()) < 500:
        score += 1.0  # Not too short, not too long

    # Formatting quality
    if not output.startswith(instruction[:20]):
        score += 0.5  # Response doesn't just repeat the instruction
    if any(c in output for c in ["1.", "2.", "- ", "* "]):
        score += 0.5  # Structured response with lists

    # Diversity signals
    if example["input"]:
        score += 0.5  # Has additional context (more complex task)

    # Penalize low-effort responses
    if len(output.split()) < 5:
        score -= 2.0  # Very short responses are likely low quality
    if output.count("\\n") == 0 and len(output.split()) > 100:
        score -= 0.5  # Long wall of text without formatting

    return {"quality_score": score}

# Score and filter
scored = dataset.map(quality_score)
scores = np.array(scored["quality_score"])
print(f"Score distribution: mean={scores.mean():.2f}, std={scores.std():.2f}")
print(f"Total examples: {len(scored)}")

# Keep top 20% by quality
threshold = np.percentile(scores, 80)
high_quality = scored.filter(lambda x: x["quality_score"] >= threshold)
print(f"High-quality subset: {len(high_quality)} examples")

# LLM-as-judge scoring (more sophisticated)
def llm_judge_prompt(instruction, output):
    return f"""Rate this instruction-response pair on a scale of 1-5:
Instruction: {instruction}
Response: {output}

Criteria: accuracy, helpfulness, clarity, completeness.
Score (1-5):"""

# Use a strong model to score each example
# Then select top-K by LLM judge score for training`,id:"code-quality-filtering"}),e.jsx(t,{type:"note",title:"The LIMA Hypothesis",content:"The LIMA paper (Less Is More for Alignment) argued that alignment is primarily about learning style and format, not knowledge. The base model already has vast knowledge from pretraining. SFT just teaches it when and how to deploy that knowledge in response to instructions. This explains why 1,000 well-chosen examples suffice: they cover the space of interaction patterns, not the space of all knowledge.",id:"note-lima"}),e.jsx(r,{title:"Deduplication is Essential",content:"Many instruction datasets contain near-duplicate examples that waste training budget and can cause the model to overfit to specific phrasings. Always deduplicate before training using techniques like MinHash or embedding-based similarity. Even removing exact string duplicates can improve a dataset significantly.",id:"warning-dedup"}),e.jsx(t,{type:"tip",title:"Diversity-Aware Selection",content:"Beyond individual quality, the diversity of the selected subset matters. Use embedding-based clustering (e.g., k-means on sentence embeddings) to ensure coverage across task types, topics, and difficulty levels. Select examples that maximize coverage of the embedding space rather than greedily picking the highest-scored ones, which may all be similar.",id:"note-diversity"})]})}const N=Object.freeze(Object.defineProperty({__proto__:null,default:y},Symbol.toStringTag,{value:"Module"}));function b(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"RLHF: Reinforcement Learning from Human Feedback"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"RLHF is a training methodology that aligns language models with human preferences by using reinforcement learning. After supervised fine-tuning, RLHF trains the model to generate responses that humans prefer, going beyond what can be achieved with static demonstration data alone. This is the technique that transformed GPT-3 into ChatGPT."}),e.jsx(i,{title:"RLHF Pipeline",definition:"RLHF consists of three stages: (1) Supervised Fine-Tuning (SFT) on demonstrations to create a policy $\\pi_{\\text{SFT}}$, (2) Training a reward model $r_\\phi(x, y)$ on human preference comparisons, and (3) Optimizing the policy via RL (typically PPO) to maximize the reward while staying close to $\\pi_{\\text{SFT}}$: $\\max_\\theta \\mathbb{E}_{x \\sim \\mathcal{D}, y \\sim \\pi_\\theta(\\cdot|x)} \\left[ r_\\phi(x, y) - \\beta \\, \\text{KL}(\\pi_\\theta \\| \\pi_{\\text{SFT}}) \\right]$",id:"def-rlhf"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"The Three Stages"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"Each stage of RLHF serves a distinct purpose and requires different data, objectives, and computational resources."}),e.jsx(n,{title:"RLHF Pipeline Stages",problem:"Describe the data requirements and objectives for each RLHF stage.",steps:[{formula:"\\text{Stage 1 - SFT: } \\mathcal{D}_{\\text{demo}} = \\{(x_i, y_i^*)\\}",explanation:"Collect expert demonstrations. Train with cross-entropy loss on 10K-100K examples."},{formula:"\\text{Stage 2 - RM: } \\mathcal{D}_{\\text{pref}} = \\{(x_i, y_i^w, y_i^l)\\}",explanation:"Collect human preference pairs (chosen vs. rejected). Train reward model on 50K-500K comparisons."},{formula:"\\text{Stage 3 - RL: } \\max_\\theta \\mathbb{E}[r_\\phi(x,y)] - \\beta \\text{KL}(\\pi_\\theta \\| \\pi_{\\text{ref}})",explanation:"Optimize policy with PPO to maximize reward while constraining divergence from the SFT model."}],id:"example-rlhf-stages"}),e.jsx(a,{title:"rlhf_pipeline_overview.py",code:`from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
import torch

# Stage 1: SFT (done separately, see s3-sft.jsx)
sft_model_path = "./sft-llama2-final"

# Stage 2: Reward Model (done separately, see s2-reward-modeling.jsx)
reward_model_path = "./reward-model-final"

# Stage 3: PPO Training
# Load the SFT model as the starting policy
model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model_path)
tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
tokenizer.pad_token = tokenizer.eos_token

# Reference model (frozen copy of SFT model for KL penalty)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model_path)

# Load reward model for scoring
reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path)

# PPO configuration
ppo_config = PPOConfig(
    batch_size=64,
    mini_batch_size=16,
    learning_rate=1.41e-5,
    log_with="wandb",
    kl_penalty="kl",           # KL divergence type
    init_kl_coeff=0.2,         # Initial beta for KL penalty
    target=6.0,                # Target KL divergence
    adap_kl_ctrl=True,         # Adaptive KL coefficient
)

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# Training loop sketch
prompts = load_dataset("your/prompt-dataset", split="train")
for batch in prompts:
    # Generate responses from current policy
    query_tensors = tokenizer(batch["prompt"], return_tensors="pt").input_ids
    response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=256)

    # Score with reward model
    rewards = compute_rewards(reward_model, query_tensors, response_tensors)

    # PPO update step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    print(f"Reward mean: {stats['ppo/mean_scores']:.3f}")`,id:"code-rlhf-pipeline"}),e.jsx(t,{type:"historical",title:"Origins of RLHF",content:"RLHF was first applied to language models by Stiennon et al. (2020) for summarization. Ouyang et al. (2022) scaled it to InstructGPT, showing RLHF made a 1.3B model preferred over the 175B GPT-3. Anthropic's Constitutional AI (2022) extended RLHF with AI-generated feedback. The technique became widely known when ChatGPT launched in November 2022.",id:"note-history"}),e.jsx(r,{title:"RLHF is Complex and Unstable",content:"RLHF involves training three separate models (SFT, reward, policy) with complex interactions. PPO is notoriously sensitive to hyperparameters, and reward hacking (the model exploiting reward model weaknesses) is a persistent problem. Small errors in the reward model get amplified during RL optimization. This complexity motivated simpler alternatives like DPO.",id:"warning-complexity"}),e.jsx(t,{type:"intuition",title:"Why RL Over More Supervised Learning?",content:"SFT can only teach the model to imitate demonstrations. But human preferences encode information that is hard to demonstrate: 'response A is better than response B because it is more nuanced.' RL allows the model to explore the response space and learn from comparative feedback, discovering response strategies that might not appear in any demonstration dataset.",id:"note-why-rl"})]})}const D=Object.freeze(Object.defineProperty({__proto__:null,default:b},Symbol.toStringTag,{value:"Module"}));function w(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Reward Modeling and the Bradley-Terry Model"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"The reward model is the component of RLHF that translates human preferences into a scalar signal the RL algorithm can optimize. It is trained on pairs of responses where humans have indicated which they prefer. The Bradley-Terry model provides the statistical foundation for converting pairwise comparisons into a consistent reward function."}),e.jsx(i,{title:"Reward Model",definition:"A reward model $r_\\phi(x, y) \\in \\mathbb{R}$ assigns a scalar score to a prompt-response pair $(x, y)$. It is typically initialized from the SFT model with the language modeling head replaced by a scalar projection head. The model is trained on human preference data $\\mathcal{D} = \\{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\\}$ where $y_w$ is the preferred (chosen) response and $y_l$ is the dispreferred (rejected) response.",id:"def-reward-model"}),e.jsx(o,{title:"Bradley-Terry Preference Model",statement:`The Bradley-Terry model assumes the probability that response $y_w$ is preferred over $y_l$ given prompt $x$ follows a logistic model:
$$p(y_w \\succ y_l | x) = \\sigma(r_\\phi(x, y_w) - r_\\phi(x, y_l))$$
where $\\sigma$ is the sigmoid function. The reward model is trained to maximize the log-likelihood:
$$\\mathcal{L}_{\\text{RM}}(\\phi) = -\\mathbb{E}_{(x, y_w, y_l) \\sim \\mathcal{D}} \\left[ \\log \\sigma(r_\\phi(x, y_w) - r_\\phi(x, y_l)) \\right]$$
This is equivalent to binary cross-entropy where the 'label' is always that $y_w$ should score higher.`,proof:"The Bradley-Terry model originates from paired comparison theory. Given latent quality scores $r_w$ and $r_l$, the probability of preferring $w$ is $\\frac{e^{r_w}}{e^{r_w} + e^{r_l}} = \\sigma(r_w - r_l)$. Taking the negative log-likelihood and averaging over the preference dataset yields the training objective.",id:"thm-bradley-terry"}),e.jsx(n,{title:"Reward Model Training Example",problem:"Given two responses to 'Explain gravity simply', compute the loss.",steps:[{formula:'y_w = \\text{"Gravity pulls objects toward each other..."}',explanation:"The human-preferred (chosen) response: clear and accurate."},{formula:'y_l = \\text{"Gravity is a quantum phenomenon..."}',explanation:"The rejected response: overly complex and potentially inaccurate for the audience."},{formula:"r_\\phi(x, y_w) = 2.3, \\quad r_\\phi(x, y_l) = 1.1",explanation:"The reward model assigns scores to each response."},{formula:"\\mathcal{L} = -\\log \\sigma(2.3 - 1.1) = -\\log \\sigma(1.2) = -\\log(0.769) = 0.263",explanation:"The model correctly ranks the preferred response higher; loss is low."}],id:"example-rm-training"}),e.jsx(a,{title:"reward_model_training.py",code:`from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer
from datasets import load_dataset

# Load base model (typically same architecture as SFT model)
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=1, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load preference dataset (chosen/rejected pairs)
# Format: each example has 'chosen' and 'rejected' fields
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:10000]")

def preprocess(example):
    """Format preference pairs for reward model training."""
    return {
        "input_ids_chosen": tokenizer(
            example["chosen"], truncation=True, max_length=512
        )["input_ids"],
        "input_ids_rejected": tokenizer(
            example["rejected"], truncation=True, max_length=512
        )["input_ids"],
    }

dataset = dataset.map(preprocess)

# Training configuration
reward_config = RewardConfig(
    output_dir="./reward-model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    bf16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=200,
    max_length=512,
)

# Train reward model using TRL's RewardTrainer
# Internally computes: loss = -log(sigmoid(r(chosen) - r(rejected)))
trainer = RewardTrainer(
    model=model,
    args=reward_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./reward-model-final")

# Inference: score a new response
prompt = "What is machine learning?"
response = "Machine learning is a subset of AI where models learn from data."
inputs = tokenizer(prompt + response, return_tensors="pt").to(model.device)
with torch.no_grad():
    score = model(**inputs).logits.item()
    print(f"Reward score: {score:.3f}")`,id:"code-reward-model"}),e.jsx(r,{title:"Reward Model Overoptimization",content:"When the RL policy is optimized too aggressively against the reward model, it finds adversarial inputs that score highly but are not actually preferred by humans. This is called reward hacking or Goodhart's Law: the reward model is a proxy for human preferences, not the real thing. Mitigation strategies include KL penalties, reward model ensembles, and periodic reward model retraining.",id:"warning-overoptimization"}),e.jsx(t,{type:"tip",title:"Reward Model Quality Metrics",content:"Track reward model accuracy on a held-out preference test set. Good reward models achieve 70-75% accuracy on human preference pairs (human inter-annotator agreement is typically around 75-80%). If accuracy is below 65%, the model may not provide a useful training signal for RL. Also monitor the reward distribution during RL training: a collapsing or bimodal distribution suggests problems.",id:"note-rm-quality"})]})}const I=Object.freeze(Object.defineProperty({__proto__:null,default:w},Symbol.toStringTag,{value:"Module"}));function k(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"PPO for RLHF"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Proximal Policy Optimization (PPO) is the reinforcement learning algorithm most commonly used in RLHF. It optimizes the language model policy to maximize rewards from the reward model while constraining updates to be small, preventing the policy from diverging too far from the reference model. PPO balances exploration (finding better responses) with stability (not destroying existing capabilities)."}),e.jsx(i,{title:"PPO for Language Models",definition:"In the RLHF context, PPO treats text generation as a sequential decision process. The policy $\\pi_\\theta$ generates tokens autoregressively. The reward is the reward model score of the full response minus a KL penalty: $R(x,y) = r_\\phi(x,y) - \\beta \\log \\frac{\\pi_\\theta(y|x)}{\\pi_{\\text{ref}}(y|x)}$. PPO updates the policy using clipped surrogate objectives to ensure stable optimization.",id:"def-ppo-rlhf"}),e.jsx(o,{title:"PPO Clipped Objective",statement:`The PPO clipped surrogate objective is:
$$\\mathcal{L}^{\\text{CLIP}}(\\theta) = \\mathbb{E}_t \\left[ \\min\\left( \\rho_t(\\theta) \\hat{A}_t, \\; \\text{clip}(\\rho_t(\\theta), 1-\\epsilon, 1+\\epsilon) \\hat{A}_t \\right) \\right]$$
where $\\rho_t(\\theta) = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{\\text{old}}}(a_t|s_t)}$ is the probability ratio and $\\hat{A}_t$ is the estimated advantage. The clipping with $\\epsilon$ (typically 0.2) prevents large policy updates.`,id:"thm-ppo-clip"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"PPO in the RLHF Loop"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The PPO training loop for RLHF generates responses, scores them with the reward model, computes advantages using a value head, and updates the policy with the clipped objective. A KL penalty against the reference (SFT) model prevents mode collapse."}),e.jsx(n,{title:"PPO Training Step",problem:"Describe one iteration of the PPO training loop for RLHF.",steps:[{formula:"y \\sim \\pi_{\\theta_{\\text{old}}}(\\cdot | x)",explanation:"Sample responses from the current policy for a batch of prompts."},{formula:"r = r_\\phi(x, y) - \\beta \\, \\text{KL}(\\pi_\\theta \\| \\pi_{\\text{ref}})",explanation:"Compute reward minus KL penalty to form the total reward signal."},{formula:"\\hat{A}_t = \\text{GAE}(r, V_{\\psi})",explanation:"Estimate per-token advantages using Generalized Advantage Estimation with the value head."},{formula:"\\theta \\leftarrow \\theta + \\alpha \\nabla_\\theta \\mathcal{L}^{\\text{CLIP}}(\\theta)",explanation:"Update policy parameters using the clipped surrogate objective over multiple mini-batches."}],id:"example-ppo-step"}),e.jsx(a,{title:"ppo_rlhf_training.py",code:`from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load SFT model with value head for PPO
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "./sft-model", torch_dtype=torch.bfloat16, device_map="auto"
)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "./sft-model", torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./sft-model")
tokenizer.pad_token = tokenizer.eos_token

# Load reward model
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "./reward-model", torch_dtype=torch.bfloat16, device_map="auto"
)
reward_tokenizer = AutoTokenizer.from_pretrained("./reward-model")

# PPO configuration
ppo_config = PPOConfig(
    batch_size=128,
    mini_batch_size=32,
    gradient_accumulation_steps=4,
    learning_rate=1.41e-5,
    init_kl_coeff=0.2,         # Initial KL penalty coefficient beta
    target=6.0,                # Target KL divergence
    adap_kl_ctrl=True,         # Adaptively adjust beta
    cliprange=0.2,             # PPO clip epsilon
    cliprange_value=0.2,       # Value function clip range
    vf_coef=0.1,               # Value loss coefficient
    ppo_epochs=4,              # PPO epochs per batch
    max_grad_norm=0.5,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

def get_reward(prompt_ids, response_ids):
    """Score prompt-response pairs with reward model."""
    full_ids = torch.cat([prompt_ids, response_ids], dim=-1)
    with torch.no_grad():
        rewards = reward_model(full_ids).logits.squeeze(-1)
    return rewards

# Training loop
for epoch in range(3):
    for batch in prompt_dataloader:
        query_tensors = [tokenizer.encode(q, return_tensors="pt").squeeze()
                        for q in batch["prompt"]]

        # Generate responses
        response_tensors = ppo_trainer.generate(
            query_tensors, max_new_tokens=256,
            temperature=0.7, top_p=0.9,
        )

        # Compute rewards
        rewards = [get_reward(q, r) for q, r in zip(query_tensors, response_tensors)]

        # PPO step: computes advantages, clips ratios, updates policy + value head
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        print(f"reward/mean: {stats['ppo/mean_scores']:.3f}, "
              f"kl: {stats['objective/kl']:.3f}")`,id:"code-ppo-rlhf"}),e.jsx(r,{title:"PPO Instability",content:"PPO for RLHF is notoriously difficult to stabilize. Common failure modes include: reward hacking (exploiting reward model weaknesses), mode collapse (generating repetitive responses), KL divergence explosion, and training instability from the value head. Careful monitoring of reward, KL, entropy, and response quality metrics is essential.",id:"warning-instability"}),e.jsx(t,{type:"tip",title:"Practical PPO Tips",content:"Key stabilization techniques: (1) Use adaptive KL penalty that increases beta when KL exceeds the target, (2) Clip both policy and value function, (3) Use multiple PPO epochs (2-4) per batch for sample efficiency, (4) Normalize advantages within each mini-batch, (5) Use gradient clipping (max_grad_norm = 0.5-1.0), (6) Start with a small learning rate and warm up.",id:"note-ppo-tips"})]})}const E=Object.freeze(Object.defineProperty({__proto__:null,default:k},Symbol.toStringTag,{value:"Module"}));function T(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"DPO and ORPO: RL-Free Alignment"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Direct Preference Optimization (DPO) and Odds Ratio Preference Optimization (ORPO) bypass the complexity of training a separate reward model and running RL. They directly optimize the language model on preference data using modified supervised learning objectives. DPO has become the dominant alignment method due to its simplicity and stability."}),e.jsx(i,{title:"Direct Preference Optimization (DPO)",definition:"DPO (Rafailov et al., 2023) derives a closed-form solution to the RLHF objective by reparameterizing the reward function in terms of the optimal policy. Instead of training a reward model and running PPO, DPO directly optimizes the policy on preference pairs using a classification-like loss that implicitly defines the reward.",id:"def-dpo"}),e.jsx(o,{title:"DPO Loss Function",statement:`The DPO loss is derived by substituting the optimal policy from the KL-constrained reward maximization into the Bradley-Terry preference model:
$$\\mathcal{L}_{\\text{DPO}}(\\theta) = -\\mathbb{E}_{(x, y_w, y_l) \\sim \\mathcal{D}} \\left[ \\log \\sigma \\left( \\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)} \\right) \\right]$$
where $\\pi_{\\text{ref}}$ is the reference (SFT) model and $\\beta$ controls the deviation from the reference. The implicit reward is $r(x,y) = \\beta \\log \\frac{\\pi_\\theta(y|x)}{\\pi_{\\text{ref}}(y|x)} + \\beta \\log Z(x)$.`,proof:"Starting from the RLHF objective $\\max_\\pi \\mathbb{E}[r(x,y)] - \\beta \\text{KL}(\\pi \\| \\pi_{\\text{ref}})$, the optimal policy is $\\pi^*(y|x) = \\frac{1}{Z(x)} \\pi_{\\text{ref}}(y|x) \\exp(r(x,y)/\\beta)$. Rearranging gives $r(x,y) = \\beta \\log \\frac{\\pi^*(y|x)}{\\pi_{\\text{ref}}(y|x)} + \\beta \\log Z(x)$. Substituting into the Bradley-Terry model and canceling the partition function $Z(x)$ (which appears in both chosen and rejected) yields the DPO loss.",id:"thm-dpo-loss"}),e.jsx(n,{title:"DPO vs. RLHF Comparison",problem:"Compare the computational requirements of DPO and RLHF.",steps:[{formula:"\\text{RLHF: 3 models (policy + value + reward) in memory}",explanation:"PPO requires the policy, value head, reference model, and reward model simultaneously."},{formula:"\\text{DPO: 2 models (policy + reference) in memory}",explanation:"DPO only needs the policy being trained and a frozen reference model."},{formula:"\\text{RLHF: online generation + reward scoring + PPO}",explanation:"Each PPO step requires generating responses, scoring them, and running multiple optimization epochs."},{formula:"\\text{DPO: single forward-backward pass on preference pairs}",explanation:"DPO is a standard supervised learning loop, making it much simpler to implement and debug."}],id:"example-dpo-vs-rlhf"}),e.jsx(a,{title:"dpo_training.py",code:`from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset
from peft import LoraConfig

# Load SFT model as starting point
model_name = "./sft-model"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Reference model (frozen copy of SFT model)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

# Load preference dataset
# Each example needs: 'prompt', 'chosen', 'rejected'
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:5000]")

def format_preference(example):
    return {
        "prompt": example["chosen"].split("\\n\\nAssistant:")[0],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

dataset = dataset.map(format_preference)

# Optional LoRA for memory efficiency
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# DPO training configuration
dpo_config = DPOConfig(
    output_dir="./dpo-model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,           # Very low LR for DPO
    beta=0.1,                     # KL penalty strength
    max_length=512,
    max_prompt_length=256,
    bf16=True,
    logging_steps=10,
    gradient_checkpointing=True,
    loss_type="sigmoid",          # Standard DPO loss
)

# Train with DPO
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,          # None if using LoRA (implicit reference)
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
)

trainer.train()
trainer.save_model("./dpo-model-final")`,id:"code-dpo-training"}),e.jsx("h2",{className:"text-2xl font-semibold",children:"ORPO: Odds Ratio Preference Optimization"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"ORPO (Hong et al., 2024) eliminates the need for a reference model entirely by combining SFT and preference optimization into a single loss. It uses the odds ratio of generating chosen vs. rejected responses as the preference signal."}),e.jsx(t,{type:"note",title:"ORPO Loss",content:"The ORPO objective combines an SFT loss on the chosen response with an odds-ratio penalty: L_ORPO = L_SFT(y_w) + lambda * L_OR where L_OR = -log sigma(log odds(y_w) - log odds(y_l)) and odds(y) = p(y|x) / (1 - p(y|x)). This means ORPO does not require a reference model and can start from a base model rather than an SFT model.",id:"note-orpo"}),e.jsx(r,{title:"DPO Hyperparameter Sensitivity",content:"DPO is sensitive to the beta parameter and learning rate. Beta too high makes the model stay too close to the reference (underfitting preferences). Beta too low allows the model to deviate too far (reward hacking without an explicit reward model). Start with beta = 0.1 and LR = 5e-7, and adjust based on the chosen/rejected reward margin during training.",id:"warning-dpo-hparams"}),e.jsx(t,{type:"tip",title:"When to Use DPO vs. RLHF",content:"DPO is recommended as the default for most practitioners: it is simpler, more stable, and produces comparable results. Use RLHF (PPO) when you need online learning (generating and scoring new responses during training), when the preference landscape is complex, or when you have a strong reward model you want to leverage. ORPO is best when you want to skip the SFT stage entirely and train directly from a base model.",id:"note-when-to-use"})]})}const B=Object.freeze(Object.defineProperty({__proto__:null,default:T},Symbol.toStringTag,{value:"Module"}));export{A as a,z as b,F as c,$ as d,R as e,S as f,C as g,O as h,M as i,q as j,N as k,D as l,I as m,E as n,B as o,P as s};
