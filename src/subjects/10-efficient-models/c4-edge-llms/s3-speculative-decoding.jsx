import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function SpeculativeDecoding() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Speculative Decoding: Draft and Verify</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Speculative decoding accelerates autoregressive generation by using a fast draft
        model to propose multiple tokens, then verifying them in parallel with the full
        target model. Since verification of K tokens costs roughly the same as generating
        one token (a single forward pass), accepted speculations provide near-free speedup.
      </p>

      <DefinitionBlock
        title="Speculative Decoding"
        definition="Given a target model $M_p$ and a faster draft model $M_q$, speculative decoding: (1) generates $K$ draft tokens from $M_q$, (2) scores all $K$ tokens in a single forward pass of $M_p$, and (3) accepts token $i$ with probability $\min\left(1, \frac{p(x_i)}{q(x_i)}\right)$. Rejected tokens are resampled from the adjusted distribution $\text{norm}(\max(0, p - q))$."
        notation="Expected tokens per step: $\frac{1 - \alpha^{K+1}}{1 - \alpha}$ where $\alpha$ is the average acceptance rate. Speedup $\approx \frac{\text{target model time}}{(\text{draft time} \times K + \text{verify time})} \times \text{accepted tokens}$."
        id="def-speculative"
      />

      <TheoremBlock
        title="Lossless Speculative Decoding"
        statement="Speculative decoding with rejection sampling produces the exact same output distribution as standard autoregressive decoding from the target model $M_p$. It is a lossless acceleration technique."
        proof="At each position, the acceptance criterion $\min(1, p(x)/q(x))$ combined with the rejection distribution $\text{norm}(\max(0, p - q))$ ensures the marginal probability of each token equals $p(x)$. This follows from the standard rejection sampling proof: the probability of accepting $x$ sampled from $q$ is $q(x) \cdot \min(1, p(x)/q(x))$, and the resampling corrects for rejected proposals."
        id="thm-speculative-lossless"
      />

      <ExampleBlock
        title="Speculative Decoding Example"
        problem="Draft model proposes 4 tokens: ['The', 'cat', 'sat', 'down']. Target model verifies each. Show the acceptance process."
        steps={[
          {
            formula: 'q = [0.3, 0.4, 0.2, 0.3], \\quad p = [0.4, 0.3, 0.25, 0.1]',
            explanation: 'Draft (q) and target (p) probabilities for each proposed token.'
          },
          {
            formula: '\\text{Token 1 (The): } \\min(1, 0.4/0.3) = 1.0 \\rightarrow \\text{ACCEPT}',
            explanation: 'Target assigns higher probability, always accepted.'
          },
          {
            formula: '\\text{Token 2 (cat): } \\min(1, 0.3/0.4) = 0.75 \\rightarrow r=0.5 < 0.75 \\rightarrow \\text{ACCEPT}',
            explanation: 'Accepted with 75% probability (random draw was favorable).'
          },
          {
            formula: '\\text{Token 3 (sat): } \\min(1, 0.25/0.2) = 1.0 \\rightarrow \\text{ACCEPT}',
            explanation: 'Target agrees with draft, accepted.'
          },
          {
            formula: '\\text{Token 4 (down): } \\min(1, 0.1/0.3) = 0.33 \\rightarrow r=0.8 > 0.33 \\rightarrow \\text{REJECT}',
            explanation: 'Target disagrees. Resample from norm(max(0, p - q)). Got 3 tokens for free!'
          }
        ]}
        id="example-speculative"
      />

      <PythonCode
        title="speculative_decoding.py"
        code={`import torch
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
print(f"Theoretical speedup: {stats['tokens'] / stats['target_calls']:.1f}x")`}
        id="code-speculative"
      />

      <NoteBlock
        type="tip"
        title="Choosing a Draft Model"
        content="The ideal draft model is 5-10x faster than the target with >70% acceptance rate. Options include: (1) a smaller model from the same family (e.g., LLaMA-7B drafting for LLaMA-70B), (2) a quantized version of the target, (3) a distilled student model, or (4) a pruned version with fewer layers. Self-speculative decoding skips layers in the target model itself as the draft."
        id="note-draft-choice"
      />

      <WarningBlock
        title="Diminishing Returns"
        content="Increasing K (speculation length) has diminishing returns because acceptance probability drops exponentially with sequence length. If the acceptance rate is 70%, the probability of all K=8 tokens being accepted is 0.7^8 = 5.7%. K=4-5 is typically optimal. Also, speculative decoding helps most when the target model is memory-bound (large batch sizes reduce the benefit)."
        id="warning-diminishing"
      />
    </div>
  )
}
