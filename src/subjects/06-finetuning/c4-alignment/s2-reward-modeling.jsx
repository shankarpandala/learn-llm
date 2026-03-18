import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function RewardModeling() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Reward Modeling and the Bradley-Terry Model</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The reward model is the component of RLHF that translates human preferences into a
        scalar signal the RL algorithm can optimize. It is trained on pairs of responses where
        humans have indicated which they prefer. The Bradley-Terry model provides the statistical
        foundation for converting pairwise comparisons into a consistent reward function.
      </p>

      <DefinitionBlock
        title="Reward Model"
        definition="A reward model $r_\phi(x, y) \in \mathbb{R}$ assigns a scalar score to a prompt-response pair $(x, y)$. It is typically initialized from the SFT model with the language modeling head replaced by a scalar projection head. The model is trained on human preference data $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}$ where $y_w$ is the preferred (chosen) response and $y_l$ is the dispreferred (rejected) response."
        id="def-reward-model"
      />

      <TheoremBlock
        title="Bradley-Terry Preference Model"
        statement="The Bradley-Terry model assumes the probability that response $y_w$ is preferred over $y_l$ given prompt $x$ follows a logistic model:
$$p(y_w \succ y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$
where $\sigma$ is the sigmoid function. The reward model is trained to maximize the log-likelihood:
$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$
This is equivalent to binary cross-entropy where the 'label' is always that $y_w$ should score higher."
        proof="The Bradley-Terry model originates from paired comparison theory. Given latent quality scores $r_w$ and $r_l$, the probability of preferring $w$ is $\frac{e^{r_w}}{e^{r_w} + e^{r_l}} = \sigma(r_w - r_l)$. Taking the negative log-likelihood and averaging over the preference dataset yields the training objective."
        id="thm-bradley-terry"
      />

      <ExampleBlock
        title="Reward Model Training Example"
        problem="Given two responses to 'Explain gravity simply', compute the loss."
        steps={[
          { formula: 'y_w = \\text{"Gravity pulls objects toward each other..."}', explanation: 'The human-preferred (chosen) response: clear and accurate.' },
          { formula: 'y_l = \\text{"Gravity is a quantum phenomenon..."}', explanation: 'The rejected response: overly complex and potentially inaccurate for the audience.' },
          { formula: 'r_\\phi(x, y_w) = 2.3, \\quad r_\\phi(x, y_l) = 1.1', explanation: 'The reward model assigns scores to each response.' },
          { formula: '\\mathcal{L} = -\\log \\sigma(2.3 - 1.1) = -\\log \\sigma(1.2) = -\\log(0.769) = 0.263', explanation: 'The model correctly ranks the preferred response higher; loss is low.' },
        ]}
        id="example-rm-training"
      />

      <PythonCode
        title="reward_model_training.py"
        code={`from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
    print(f"Reward score: {score:.3f}")`}
        id="code-reward-model"
      />

      <WarningBlock
        title="Reward Model Overoptimization"
        content="When the RL policy is optimized too aggressively against the reward model, it finds adversarial inputs that score highly but are not actually preferred by humans. This is called reward hacking or Goodhart's Law: the reward model is a proxy for human preferences, not the real thing. Mitigation strategies include KL penalties, reward model ensembles, and periodic reward model retraining."
        id="warning-overoptimization"
      />

      <NoteBlock
        type="tip"
        title="Reward Model Quality Metrics"
        content="Track reward model accuracy on a held-out preference test set. Good reward models achieve 70-75% accuracy on human preference pairs (human inter-annotator agreement is typically around 75-80%). If accuracy is below 65%, the model may not provide a useful training signal for RL. Also monitor the reward distribution during RL training: a collapsing or bimodal distribution suggests problems."
        id="note-rm-quality"
      />
    </div>
  )
}
