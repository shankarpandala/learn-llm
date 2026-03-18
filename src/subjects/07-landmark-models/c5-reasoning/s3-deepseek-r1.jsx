import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function DeepSeekR1() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">DeepSeek R1: Open Reasoning Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        DeepSeek-R1 (January 2025) was the first open-weight reasoning model to match OpenAI o1
        on major benchmarks. Built on DeepSeek-V3, it demonstrated that reasoning capabilities
        could be achieved through pure reinforcement learning without supervised fine-tuning on
        reasoning traces, and released full model weights including distilled smaller variants.
      </p>

      <DefinitionBlock
        title="DeepSeek R1 Training Pipeline"
        definition="DeepSeek-R1 uses a multi-stage training approach: (1) R1-Zero: pure RL with GRPO on the base model, demonstrating emergent reasoning without SFT, (2) Cold start: SFT on curated long-CoT data to improve readability, (3) RL training with rule-based rewards for math/code (verifiable) and model-based rewards for other tasks, (4) Rejection sampling + SFT for the final model."
        id="def-r1-pipeline"
      />

      <h2 className="text-2xl font-semibold">R1-Zero: Reasoning from Pure RL</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The most remarkable finding was R1-Zero: applying Group Relative Policy Optimization (GRPO)
        directly to the base DeepSeek-V3 model with only correctness rewards produced emergent
        chain-of-thought reasoning, self-verification, and reflection behaviors without any
        supervised reasoning examples. The model discovered these strategies on its own.
      </p>

      <ExampleBlock
        title="DeepSeek R1 Benchmark Performance"
        problem="Compare DeepSeek R1 against OpenAI o1 on key reasoning benchmarks."
        steps={[
          { formula: '\\text{AIME 2024}: \\text{R1} = 79.8\\% \\text{ vs o1} = 79.2\\%', explanation: 'R1 matches o1 on competitive mathematics with pass@1 accuracy.' },
          { formula: '\\text{MATH-500}: \\text{R1} = 97.3\\% \\text{ vs o1} = 96.4\\%', explanation: 'On the MATH benchmark, R1 slightly exceeds o1.' },
          { formula: '\\text{Codeforces}: \\text{R1} = 96.3\\text{th pctile vs o1} = 96.6\\text{th pctile}', explanation: 'Near-identical competitive programming performance.' },
          { formula: '\\text{GPQA Diamond}: \\text{R1} = 71.5\\% \\text{ vs o1} = 78.0\\%', explanation: 'R1 trails slightly on PhD-level science questions.' },
        ]}
        id="example-r1-benchmarks"
      />

      <h2 className="text-2xl font-semibold">Group Relative Policy Optimization (GRPO)</h2>
      <p className="text-gray-700 dark:text-gray-300">
        GRPO is a variant of PPO that eliminates the need for a separate critic/value model. For
        each prompt, it samples a group of outputs, computes rewards for each, and uses the group's
        mean and standard deviation to normalize advantages. This makes it more memory-efficient
        than PPO while maintaining stable training.
      </p>

      <DefinitionBlock
        title="GRPO Advantage Estimation"
        definition="For a prompt $q$ and a group of $G$ sampled outputs $\{o_i\}_{i=1}^{G}$ with rewards $\{r_i\}$, the advantage for output $o_i$ is computed relative to the group: $A_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$. This eliminates the need for a learned value function, reducing memory by approximately 50% compared to PPO."
        id="def-grpo"
      />

      <PythonCode
        title="deepseek_r1_usage.py"
        code={`from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# DeepSeek-R1 distilled variants are practical to run locally
# Available: 1.5B, 7B, 8B, 14B, 32B, 70B distilled models
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# R1 generates <think>...</think> blocks before the final answer
prompt = """Solve this step by step:
If the sum of two numbers is 15 and their product is 56, what are the numbers?"""

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.6,
        top_p=0.95,
    )

response = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print(response)

# Parse thinking vs answer
if "<think>" in response and "</think>" in response:
    think_start = response.index("<think>") + len("<think>")
    think_end = response.index("</think>")
    thinking = response[think_start:think_end].strip()
    answer = response[think_end + len("</think>"):].strip()
    print(f"\\n--- Thinking ({len(thinking.split())} words) ---")
    print(thinking[:300] + "...")
    print(f"\\n--- Answer ---")
    print(answer)

# GRPO concept implementation
def grpo_advantages(rewards):
    """Compute GRPO-style group-relative advantages."""
    rewards = torch.tensor(rewards, dtype=torch.float32)
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8
    advantages = (rewards - mean_r) / std_r
    return advantages

# Example: 8 sampled outputs with binary rewards (correct/incorrect)
rewards = [1, 0, 1, 1, 0, 1, 0, 1]  # 5 correct, 3 incorrect
advantages = grpo_advantages(rewards)
print(f"\\nGRPO advantages: {advantages.tolist()}")
print(f"Correct outputs get positive advantage: {advantages[0].item():.3f}")
print(f"Incorrect outputs get negative advantage: {advantages[1].item():.3f}")`}
        id="code-r1"
      />

      <NoteBlock
        type="historical"
        title="Emergent Reasoning in R1-Zero"
        content="R1-Zero spontaneously developed several reasoning behaviors through RL alone: (1) Extended thinking chains that grow longer for harder problems, (2) Self-verification ('let me check this'), (3) Backtracking ('wait, that's wrong, let me try again'), (4) Breaking problems into sub-problems. These emerged without any CoT training data, suggesting reasoning is a natural optimization target for RL on correctness rewards."
        id="note-r1-zero-emergence"
      />

      <NoteBlock
        type="tip"
        title="Distilled Models"
        content="DeepSeek released distilled R1 variants by fine-tuning Qwen and LLaMA base models on R1's reasoning traces. The 32B distilled model outperforms o1-mini on most benchmarks, while the 7B model provides strong reasoning capability on consumer hardware. These distilled models use standard SFT, making them easier to further fine-tune."
        id="note-distilled"
      />

      <WarningBlock
        title="Language Mixing in Reasoning"
        content="R1 occasionally mixes Chinese and English in its thinking tokens, even when the prompt is entirely in English. This reflects the multilingual training data and can produce confusing intermediate reasoning. The distilled models based on LLaMA/Qwen tend to have less language mixing but it can still occur."
        id="warning-language-mixing"
      />
    </div>
  )
}
