import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function NextTokenPrediction() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Next-Token Prediction Mechanics</h1>
      <p className="text-lg text-gray-300">
        The mechanics of next-token prediction involve converting hidden states into probability
        distributions over the vocabulary, then selecting the next token through various decoding
        strategies. Understanding this pipeline is crucial for both training and inference.
      </p>

      <DefinitionBlock
        title="Logit Computation"
        definition="At each position $t$, the final hidden state $h_t \in \mathbb{R}^d$ is projected to vocabulary logits via the language model head: $z_t = W_{\text{LM}} h_t + b$, where $W_{\text{LM}} \in \mathbb{R}^{|V| \times d}$. Often $W_{\text{LM}}$ is tied to the input embedding matrix."
        notation="$P(x_{t+1} = v \mid x_{\leq t}) = \text{softmax}(z_t)_v = \frac{\exp(z_t[v])}{\sum_{v'} \exp(z_t[v'])}$"
        id="logit-def"
      />

      <ExampleBlock
        title="From Hidden State to Token"
        problem="Given hidden state h with d=768 and vocab size |V|=50257, trace the prediction pipeline."
        steps={[
          {
            formula: 'z = W_{\\text{LM}} h \\in \\mathbb{R}^{50257}',
            explanation: 'Project 768-dim hidden state to 50257-dim logits (one per vocabulary token).'
          },
          {
            formula: 'z\' = z / \\tau \\quad (\\text{temperature scaling})',
            explanation: 'Temperature tau < 1 sharpens distribution, tau > 1 flattens it.'
          },
          {
            formula: 'P = \\text{softmax}(z\') \\in \\Delta^{|V|-1}',
            explanation: 'Convert logits to probability simplex. Sum to 1.'
          },
          {
            formula: 'x_{t+1} \\sim \\text{Top-}k(P) \\text{ or } \\text{Top-}p(P)',
            explanation: 'Sample from filtered distribution using top-k or nucleus (top-p) sampling.'
          }
        ]}
        id="prediction-pipeline"
      />

      <NoteBlock
        type="tip"
        title="Weight Tying"
        content="Most modern LLMs tie the output projection W_LM to the transpose of the input embedding matrix E. This means the logit for token v is the dot product of the hidden state with v's embedding: z[v] = e_v^T h. Weight tying reduces parameters and often improves quality by forcing input and output representations to share the same space."
        id="weight-tying-note"
      />

      <PythonCode
        title="next_token_mechanics.py"
        code={`from transformers import GPT2LMHeadModel, GPT2Tokenizer
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
print(f"Effective choices: {torch.exp(entropy).item():.0f} tokens")`}
        id="next-token-code"
      />

      <WarningBlock
        title="Repetition and Degenerate Text"
        content="Greedy and beam search decoding often produce repetitive, degenerate text. This happens because the model assigns high probability to recently seen tokens (a positive feedback loop). Sampling with temperature, top-k, or top-p helps, but too much randomness produces incoherent text. Finding the right balance is an active area of research."
        id="degenerate-warning"
      />

      <TheoremBlock
        title="Softmax Temperature"
        statement="For logits $z$ and temperature $\tau > 0$: as $\tau \to 0$, $\text{softmax}(z/\tau)$ converges to a one-hot distribution on $\arg\max(z)$. As $\tau \to \infty$, it converges to the uniform distribution $1/|V|$."
        proof="For $\tau \to 0$: the largest logit dominates the exponential, so $\exp(z_{\max}/\tau) \gg \exp(z_i/\tau)$ for $z_i < z_{\max}$. For $\tau \to \infty$: all $z_i/\tau \to 0$, so $\exp(z_i/\tau) \to 1$ for all $i$, giving uniform $1/|V|$."
        id="temperature-theorem"
      />
    </div>
  )
}
