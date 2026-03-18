import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function GreedyBeamSearch() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Greedy & Beam Search</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        When an LLM generates text, it produces a probability distribution over the vocabulary at
        each step. Decoding strategies determine how we select the next token from that distribution.
        Greedy and beam search are the two foundational deterministic approaches.
      </p>

      <DefinitionBlock
        title="Greedy Decoding"
        definition="Greedy decoding selects the token with the highest probability at each step: $y_t = \arg\max_{w} P(w \mid y_{<t}, x)$. It is fast but can miss globally optimal sequences."
        id="def-greedy"
      />

      <ExampleBlock
        title="Greedy vs Optimal"
        problem="Given P('The') = 0.6, P('A') = 0.4 at step 1, and P('cat'|'The') = 0.3, P('dog'|'A') = 0.9, compare greedy vs optimal."
        steps={[
          { formula: 'Greedy: \"The\" (0.6) -> \"cat\" (0.3) = 0.18', explanation: 'Greedy picks the highest probability at each step independently.' },
          { formula: 'Optimal: \"A\" (0.4) -> \"dog\" (0.9) = 0.36', explanation: 'The globally better sequence was missed because greedy committed to \"The\" early.' },
          { formula: 'Beam search with k=2 would find both paths', explanation: 'By keeping multiple candidates, beam search avoids this trap.' },
        ]}
        id="example-greedy-vs-optimal"
      />

      <DefinitionBlock
        title="Beam Search"
        definition="Beam search maintains $k$ candidate sequences (beams) at each step, expanding each by all vocabulary tokens and keeping the top-$k$ by cumulative log-probability: $\text{score}(y) = \sum_{t=1}^{T} \log P(y_t \mid y_{<t}, x)$."
        notation="$k$ is the beam width. $k=1$ reduces to greedy search."
        id="def-beam-search"
      />

      <PythonCode
        title="greedy_decoding.py"
        code={`import torch
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
# Greedy often produces repetitive, generic text`}
        id="code-greedy"
      />

      <PythonCode
        title="beam_search.py"
        code={`import torch
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

print(beam_search("The future of AI is", beam_width=5))`}
        id="code-beam-search"
      />

      <NoteBlock
        type="intuition"
        title="Length Normalization"
        content="Beam search tends to favor shorter sequences because log-probabilities are negative and accumulate. Length normalization divides the score by sequence length raised to a power alpha: score / T^alpha. A typical alpha is 0.6-0.7."
        id="note-length-norm"
      />

      <WarningBlock
        title="Beam Search Is Not Always Better"
        content="For open-ended generation (stories, conversations), beam search tends to produce dull, repetitive text. It works best for tasks with a 'correct' answer like machine translation or summarization. For creative generation, sampling methods (next section) are preferred."
        id="warning-beam-limitations"
      />

      <NoteBlock
        type="tip"
        title="HuggingFace Generate API"
        content="In practice, use model.generate() with num_beams=5 for beam search or do_sample=False for greedy. The library handles length penalties, early stopping, and n-gram repetition penalties automatically."
        id="note-hf-generate"
      />
    </div>
  )
}
