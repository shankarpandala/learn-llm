import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ConstrainedDecoding() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Constrained Decoding Algorithms</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Constrained decoding modifies the token selection process during LLM generation to
        enforce structural, lexical, or semantic constraints. Rather than relying on post-hoc
        validation, constraints are applied at each decoding step, guaranteeing that the final
        output satisfies all requirements.
      </p>

      <DefinitionBlock
        title="Constrained Decoding"
        definition="Constrained decoding modifies the standard autoregressive generation process by applying a token mask $M_t \in \{0, 1\}^{|V|}$ at each step $t$. The effective distribution becomes $P'(x_t \mid x_{<t}) = \frac{P(x_t \mid x_{<t}) \cdot M_t(x_t)}{\sum_{x' \in V} P(x' \mid x_{<t}) \cdot M_t(x')}$ where $M_t$ is determined by the constraint specification and previously generated tokens."
        notation="V = vocabulary, M_t = mask at step t, P = base model, P' = constrained distribution"
        id="def-constrained"
      />

      <h2 className="text-2xl font-semibold">Token Masking Approach</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The simplest constrained decoding technique masks invalid tokens by setting their
        logits to negative infinity before the softmax. This effectively removes them from
        consideration without changing the relative probabilities of valid tokens.
      </p>

      <PythonCode
        title="token_masking.py"
        code={`import torch
import torch.nn.functional as F

# Simulate constrained decoding with token masking
vocab_size = 50000
logits = torch.randn(vocab_size)  # raw model logits

# Example: force output to be a valid integer
# Only allow digit tokens and end-of-sequence
digit_token_ids = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57]  # '0'-'9'
eos_token_id = 2

# Create mask: 1 for valid tokens, 0 for invalid
mask = torch.zeros(vocab_size)
mask[digit_token_ids] = 1
mask[eos_token_id] = 1

# Apply mask by setting invalid logits to -inf
masked_logits = logits.clone()
masked_logits[mask == 0] = float('-inf')

# Compare distributions
probs_unconstrained = F.softmax(logits, dim=0)
probs_constrained = F.softmax(masked_logits, dim=0)

print(f"Unconstrained: top token prob = {probs_unconstrained.max():.4f}")
print(f"Constrained:   top token prob = {probs_constrained.max():.4f}")
print(f"Constrained:   valid tokens get all probability mass")
print(f"Sum of constrained probs: {probs_constrained.sum():.6f}")  # 1.0

# The relative ranking among valid tokens is preserved
valid_probs = probs_constrained[digit_token_ids + [eos_token_id]]
print(f"Valid token probs: {valid_probs.numpy().round(4)}")`}
        id="code-masking"
      />

      <ExampleBlock
        title="Finite State Machine for Constrained Decoding"
        problem="Design an FSM that constrains output to valid email addresses."
        steps={[
          { formula: '\\text{State 0: local part} \\to [a-zA-Z0-9.]+', explanation: 'Accept alphanumeric characters and dots for the local part of the email.' },
          { formula: '\\text{State 0} \\xrightarrow{@} \\text{State 1: domain}', explanation: 'Transition to domain state when @ is generated.' },
          { formula: '\\text{State 1: domain} \\to [a-zA-Z0-9-]+ \\xrightarrow{.} \\text{State 2: TLD}', explanation: 'Accept domain characters, transition on dot to TLD state.' },
          { formula: '\\text{State 2: TLD} \\to [a-zA-Z]\\{2,\\} \\to \\text{Accept}', explanation: 'Accept 2+ alpha characters for TLD, then transition to accept state.' },
        ]}
        id="example-fsm"
      />

      <h2 className="text-2xl font-semibold">Outlines: Token-Level Constrained Generation</h2>

      <PythonCode
        title="outlines_constrained.py"
        code={`import outlines

# Load a model through Outlines
model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")

# Regex-constrained generation: valid US phone number
phone_generator = outlines.generate.regex(
    model,
    r"\\(\\d{3}\\) \\d{3}-\\d{4}"
)
phone = phone_generator("Generate a phone number: ")
print(f"Phone: {phone}")  # e.g., (555) 123-4567

# JSON Schema-constrained generation
from pydantic import BaseModel
from typing import List

class MovieReview(BaseModel):
    title: str
    year: int
    rating: float
    genres: List[str]
    recommend: bool

review_generator = outlines.generate.json(model, MovieReview)
review = review_generator(
    "Write a review for a sci-fi movie as JSON:\\n"
)
print(f"Title: {review.title}")
print(f"Rating: {review.rating}")
print(f"Recommend: {review.recommend}")

# Choice-constrained generation: pick from allowed values
sentiment = outlines.generate.choice(
    model,
    ["positive", "negative", "neutral"]
)
result = sentiment("The movie was okay but not great. Sentiment: ")
print(f"Sentiment: {result}")  # "neutral"`}
        id="code-outlines"
      />

      <PythonCode
        title="constrained_beam_search.py"
        code={`import torch

def constrained_beam_search(model, tokenizer, prompt, constraint_fn,
                            beam_width=4, max_length=50):
    """Beam search with constraint function.

    constraint_fn(generated_ids) -> list of valid next token ids
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Initialize beams: (score, token_ids)
    beams = [(0.0, input_ids[0].tolist())]

    for step in range(max_length):
        candidates = []
        for score, ids in beams:
            # Get model logits for this beam
            with torch.no_grad():
                outputs = model(torch.tensor([ids]))
                logits = outputs.logits[0, -1, :]

            # Apply constraints
            valid_tokens = constraint_fn(ids[len(input_ids[0]):])
            mask = torch.full_like(logits, float('-inf'))
            mask[valid_tokens] = 0
            constrained_logits = logits + mask

            # Get top-k valid continuations
            log_probs = torch.log_softmax(constrained_logits, dim=0)
            top_k = torch.topk(log_probs, beam_width)

            for log_p, token_id in zip(top_k.values, top_k.indices):
                new_score = score + log_p.item()
                new_ids = ids + [token_id.item()]
                candidates.append((new_score, new_ids))

        # Keep top beams
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

        # Check for completion
        if all(b[1][-1] == tokenizer.eos_token_id for b in beams):
            break

    return tokenizer.decode(beams[0][1])

# Example constraint: output must be valid Python variable name
# def python_var_constraint(generated_ids):
#     if len(generated_ids) == 0:
#         return letter_tokens + underscore_token
#     return letter_tokens + digit_tokens + underscore_token + [eos]`}
        id="code-beam"
      />

      <NoteBlock
        type="intuition"
        title="Why Constrained Decoding Preserves Quality"
        content="Constrained decoding does not change the model's learned distribution -- it only renormalizes it over valid tokens. If the model already assigns high probability to valid outputs (as well-trained models do), the constraint primarily eliminates low-probability invalid tokens. The quality impact is minimal because the model was already 'trying' to produce valid output; the constraint just ensures it succeeds."
        id="note-quality"
      />

      <WarningBlock
        title="Constraint Satisfaction vs. Semantic Quality"
        content="Constrained decoding guarantees syntactic validity but not semantic correctness. A grammar-constrained JSON generator will always produce valid JSON, but the content may be nonsensical. A SQL grammar constraint ensures valid syntax but not correct semantics. Always combine structural constraints with proper prompting for content quality."
        id="warning-semantic"
      />

    </div>
  )
}
