import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function GptObjective() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">GPT Training Objective</h1>
      <p className="text-lg text-gray-300">
        The GPT family of models uses a simple but powerful training objective: maximize the
        likelihood of the next token given all preceding tokens. This causal language modeling
        objective scales remarkably well and forms the basis of modern LLMs.
      </p>

      <DefinitionBlock
        title="Causal Language Modeling Loss"
        definition="Given a corpus of token sequences, the GPT training objective maximizes: $\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})$. This is the average negative log-likelihood per token, equivalent to cross-entropy with the data distribution."
        notation="Perplexity is $\\text{PPL} = \\exp(\mathcal{L})$. Lower perplexity means better next-token prediction."
        id="gpt-loss-def"
      />

      <TheoremBlock
        title="Relationship Between Cross-Entropy and Perplexity"
        statement="For cross-entropy loss $\mathcal{L}$ in nats, perplexity is $\text{PPL} = e^{\mathcal{L}}$. For cross-entropy in bits, $\text{PPL} = 2^{\mathcal{L}_{\text{bits}}}$. A model with PPL of $k$ is as uncertain as a uniform distribution over $k$ tokens."
        proof="$\text{PPL} = \exp\left(-\frac{1}{T}\sum_{t=1}^T \log P(x_t \mid x_{<t})\right) = \exp(\mathcal{L})$. If $P$ is uniform over $k$ tokens, $\mathcal{L} = \log k$, so $\text{PPL} = k$."
        id="ppl-theorem"
      />

      <ExampleBlock
        title="Computing GPT Loss"
        problem="Compute the loss for predicting 'The cat sat' with vocabulary size |V|=50257."
        steps={[
          {
            formula: 'P(\\text{cat} \\mid \\text{The}) = 0.02',
            explanation: 'Model assigns 2% probability to "cat" following "The".'
          },
          {
            formula: 'P(\\text{sat} \\mid \\text{The cat}) = 0.05',
            explanation: 'Model assigns 5% probability to "sat" following "The cat".'
          },
          {
            formula: '\\mathcal{L} = -\\frac{1}{2}[\\log(0.02) + \\log(0.05)] = \\frac{1}{2}[3.91 + 3.00] = 3.46',
            explanation: 'Average negative log-likelihood over the two predictions.'
          },
          {
            formula: '\\text{PPL} = e^{3.46} \\approx 31.8',
            explanation: 'Model is as confused as choosing uniformly from ~32 tokens.'
          }
        ]}
        id="gpt-loss-example"
      />

      <NoteBlock
        type="historical"
        title="GPT Evolution"
        content="GPT-1 (2018, 117M params) showed pretrain+finetune works for NLP. GPT-2 (2019, 1.5B) demonstrated zero-shot abilities with scale. GPT-3 (2020, 175B) introduced in-context learning without fine-tuning. Each generation kept the same causal LM objective -- only scale and data changed."
        id="gpt-evolution-note"
      />

      <PythonCode
        title="gpt_training_objective.py"
        code={`from transformers import GPT2LMHeadModel, GPT2Tokenizer
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
          f"loss={out.loss.item():.3f}, PPL={torch.exp(out.loss).item():.1f}")`}
        id="gpt-objective-code"
      />

      <WarningBlock
        title="Loss on First Token Is Undefined"
        content="The first token has no preceding context, so GPT cannot predict it. In practice, the labels are shifted: we predict token t+1 from position t. A sequence of length T yields T-1 loss terms. This is why HuggingFace uses labels=input_ids and internally shifts them."
        id="first-token-warning"
      />
    </div>
  )
}
