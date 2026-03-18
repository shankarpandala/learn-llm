import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function AutoregressiveGeneration() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Autoregressive Language Modeling</h1>
      <p className="text-lg text-gray-300">
        Autoregressive (AR) language models generate text one token at a time, conditioning each
        prediction on all previously generated tokens. This left-to-right factorization is the
        foundation of GPT-family models and modern large language models.
      </p>

      <DefinitionBlock
        title="Autoregressive Factorization"
        definition="An autoregressive language model decomposes the joint probability of a sequence into a product of conditional probabilities: $P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})$."
        notation="$P(x_t \mid x_{<t})$ denotes the probability of token $x_t$ given all preceding tokens $x_{<t} = (x_1, \ldots, x_{t-1})$."
        id="ar-def"
      />

      <TheoremBlock
        title="Chain Rule of Probability"
        statement="The autoregressive factorization is exact by the chain rule -- it introduces no approximation. Any joint distribution $P(x_{1:T})$ can be decomposed as $\prod_{t=1}^T P(x_t \mid x_{<t})$."
        proof="By the definition of conditional probability: $P(A, B) = P(A)P(B|A)$. Applied recursively: $P(x_1, x_2, x_3) = P(x_1) \cdot P(x_2|x_1) \cdot P(x_3|x_1, x_2)$. This generalizes to $T$ variables by induction."
        id="chain-rule-thm"
      />

      <ExampleBlock
        title="Autoregressive Generation Step-by-Step"
        problem="Generate text starting from 'The' using an autoregressive model."
        steps={[
          {
            formula: 'P(x_1) = P(\\text{\"The\"})',
            explanation: 'Start with the prompt token.'
          },
          {
            formula: 'P(x_2 \\mid x_1) \\rightarrow \\text{\"cat\"} \\; (\\text{sampled})',
            explanation: 'Model outputs distribution over vocabulary; sample or pick argmax.'
          },
          {
            formula: 'P(x_3 \\mid \\text{\"The cat\"}) \\rightarrow \\text{\"sat\"}',
            explanation: 'Condition on full prefix. Causal mask ensures position 3 only sees positions 1-2.'
          },
          {
            formula: '\\text{Continue until } x_t = \\text{[EOS] or max length}',
            explanation: 'Generation terminates on end-of-sequence token or length limit.'
          }
        ]}
        id="ar-generation-example"
      />

      <NoteBlock
        type="intuition"
        title="Causal Masking Enables AR"
        content="In a Transformer decoder, the causal (triangular) attention mask ensures that position t can only attend to positions 1 through t. This prevents information leakage from future tokens and makes the model truly autoregressive during both training and inference."
        id="causal-mask-note"
      />

      <PythonCode
        title="autoregressive_generation.py"
        code={`from transformers import GPT2LMHeadModel, GPT2Tokenizer
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
print(causal_mask)`}
        id="ar-code"
      />

      <WarningBlock
        title="Sequential Generation Is Slow"
        content="Autoregressive generation is inherently sequential: each token depends on all previous tokens. This means generating T tokens requires T forward passes. KV-caching helps (caching key/value tensors for previous positions), but generation remains fundamentally O(T) in serial steps, unlike bidirectional models that process all positions in parallel."
        id="ar-slow-warning"
      />
    </div>
  )
}
