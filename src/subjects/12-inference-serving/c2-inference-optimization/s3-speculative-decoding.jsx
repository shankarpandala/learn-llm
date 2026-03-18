import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function SpeculativeDecoding() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Speculative Decoding</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Speculative decoding uses a small, fast draft model to propose multiple tokens, then verifies
        them in parallel with the large target model. When the draft model's predictions match, you
        get multiple tokens per forward pass of the large model, dramatically reducing latency.
      </p>

      <DefinitionBlock
        title="Speculative Decoding"
        definition="A draft model $M_q$ generates $\gamma$ candidate tokens autoregressively. The target model $M_p$ then scores all $\gamma$ tokens in a single forward pass. Tokens are accepted from left to right using a rejection sampling scheme that preserves the target distribution exactly."
        notation="$\gamma$ = speculation length (typically 3-5). $\alpha$ = acceptance rate (fraction of draft tokens accepted)."
        id="def-speculative"
      />

      <ExampleBlock
        title="Speculative Decoding Walk-through"
        problem="Draft model proposes 4 tokens. Target model verifies them."
        steps={[
          { formula: 'Draft proposes: [\"The\", \"cat\", \"sat\", \"on\"]', explanation: 'Small model (e.g., 1B params) generates 4 tokens very quickly.' },
          { formula: 'Target scores all 4 in one pass', explanation: 'Large model (e.g., 70B) processes all 4 tokens in parallel, as fast as processing 1.' },
          { formula: 'Accept: \"The\" \\checkmark, \"cat\" \\checkmark, \"sat\" \\checkmark, \"on\" \\times', explanation: 'First 3 match target distribution (accepted). \"on\" rejected -- target preferred \"down\".' },
          { formula: 'Result: 3 tokens + 1 corrected = 4 tokens from 1 large forward pass', explanation: 'Instead of 4 large model calls, we used 1 large + 4 small calls. Net speedup ~2-3x.' },
        ]}
        id="example-speculative"
      />

      <PythonCode
        title="speculative_decoding.py"
        code={`import torch
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
    return tokenizer.decode(generated[0]), acceptance_rate`}
        id="code-speculative"
      />

      <PythonCode
        title="Terminal"
        code={`# vLLM supports speculative decoding natively
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
    --num-speculative-tokens 5`}
        id="code-vllm-spec"
      />

      <NoteBlock
        type="intuition"
        title="Why It Preserves the Target Distribution"
        content="The rejection sampling scheme is mathematically designed so that the final output distribution is identical to the target model. If the draft model is perfect (matches the target exactly), all tokens are accepted. If it is terrible, we fall back to sampling one token per large model call -- never worse than standard decoding."
        id="note-distribution"
      />

      <WarningBlock
        title="Draft Model Quality Matters"
        content="The speedup depends on the acceptance rate, which depends on how well the draft model approximates the target. A poor draft model means most tokens are rejected and you get no benefit. Best results come from draft models in the same family (e.g., LLaMA-1B drafting for LLaMA-70B)."
        id="warning-draft-quality"
      />

      <NoteBlock
        type="note"
        title="Self-Speculative Decoding"
        content="Some approaches skip the separate draft model entirely. Medusa adds extra prediction heads to the target model. EAGLE uses the target model's own hidden states. Layer-skipping uses a subset of the target model's layers as the draft. These avoid the complexity of maintaining two models."
        id="note-self-speculative"
      />
    </div>
  )
}
