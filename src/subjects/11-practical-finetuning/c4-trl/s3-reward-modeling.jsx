import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function RewardModeling() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Reward Model Training</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        A reward model learns to score responses based on human preferences. It is the foundation
        of RLHF (Reinforcement Learning from Human Feedback) and can also be used for data
        filtering, best-of-N sampling, and response ranking.
      </p>

      <DefinitionBlock
        title="Reward Model"
        definition="A reward model $R_\\phi(x, y) \\rightarrow \\mathbb{R}$ maps a prompt $x$ and response $y$ to a scalar score. It is trained on preference pairs using the Bradley-Terry loss: $L = -\\log \\sigma(R_\\phi(x, y_w) - R_\\phi(x, y_l))$ where $y_w$ is preferred over $y_l$."
        notation="L = -\log \sigma(R(x, y_w) - R(x, y_l))"
        id="def-reward-model"
      />

      <h2 className="text-2xl font-semibold">Training a Reward Model</h2>

      <PythonCode
        title="train_reward_model.py"
        code={`from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import torch

# Load base model for reward modeling
# Use a smaller model than the policy for efficiency
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,              # Single scalar output
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Load preference dataset
dataset = load_dataset(
    "argilla/ultrafeedback-binarized-preferences-cleaned",
    split="train[:5000]"
)

# Format: each example needs 'chosen' and 'rejected' fields
# These should be full conversation strings
print(f"Dataset size: {len(dataset)}")
print(f"Columns: {dataset.column_names}")

# Configure reward training
reward_config = RewardConfig(
    output_dir="./reward-model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-5,
    bf16=True,
    logging_steps=10,
    max_length=1024,
    gradient_checkpointing=True,
)

trainer = RewardTrainer(
    model=model,
    args=reward_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()`}
        id="code-train-reward"
      />

      <h2 className="text-2xl font-semibold">Using the Reward Model</h2>

      <PythonCode
        title="use_reward_model.py"
        code={`import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load trained reward model
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "./reward-model", torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./reward-model")

def score_response(prompt, response):
    """Score a single response given a prompt."""
    text = f"User: {prompt}\\nAssistant: {response}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=1024).to(reward_model.device)
    with torch.no_grad():
        score = reward_model(**inputs).logits[0].item()
    return score

# Best-of-N sampling: generate N responses, pick highest scored
def best_of_n(prompt, responses):
    scored = [(r, score_response(prompt, r)) for r in responses]
    scored.sort(key=lambda x: x[1], reverse=True)
    for resp, score in scored:
        print(f"Score: {score:.3f} | {resp[:80]}...")
    return scored[0][0]

# Example usage
prompt = "What are the benefits of exercise?"
responses = [
    "Exercise is good for you. You should do it.",
    "Regular exercise improves cardiovascular health, boosts mood through endorphin release, strengthens muscles and bones, and helps maintain a healthy weight. Even 30 minutes of moderate activity daily can significantly reduce the risk of chronic diseases.",
    "idk maybe google it lol",
]
best = best_of_n(prompt, responses)
print(f"\\nBest response: {best[:100]}...")`}
        id="code-use-reward"
      />

      <ExampleBlock
        title="Reward Model Architecture"
        problem="How is a language model adapted into a reward model?"
        steps={[
          { formula: '\\text{Base LLM} \\rightarrow \\text{Remove LM head}', explanation: 'The next-token prediction head is removed.' },
          { formula: '\\text{Add linear head: } \\mathbb{R}^d \\rightarrow \\mathbb{R}^1', explanation: 'A single linear layer maps the last hidden state to a scalar score.' },
          { formula: '\\text{Pool over last token}', explanation: 'The reward is the score at the last token position (like [CLS] in BERT).' },
          { formula: '\\text{Train with Bradley-Terry loss}', explanation: 'The model learns that chosen responses should score higher than rejected ones.' },
        ]}
        id="example-rm-architecture"
      />

      <NoteBlock
        type="note"
        title="Reward Model Size"
        content="The reward model does not need to be as large as the policy model. A 3B reward model can effectively guide an 8B policy model. This saves compute during RLHF training where the reward model is called every step."
        id="note-rm-size"
      />

      <WarningBlock
        title="Reward Hacking"
        content="When using a reward model with RL optimization, the policy may learn to exploit weaknesses in the reward model rather than genuinely improving. This is called reward hacking. Mitigations include: KL penalties, ensembles of reward models, and periodic human evaluation of high-scoring outputs."
        id="warning-reward-hacking"
      />
    </div>
  )
}
