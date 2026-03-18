import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function LoRA() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LoRA: Low-Rank Adaptation</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Low-Rank Adaptation (LoRA) is a parameter-efficient finetuning method that freezes
        the pretrained model weights and injects trainable low-rank decomposition matrices
        into each layer. This dramatically reduces the number of trainable parameters while
        achieving performance comparable to full finetuning.
      </p>

      <DefinitionBlock
        title="Low-Rank Adaptation (LoRA)"
        definition="LoRA modifies the forward pass of a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ by adding a low-rank update: $W = W_0 + BA$ where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$. Only $A$ and $B$ are trained; $W_0$ remains frozen. The forward pass becomes $h = W_0 x + BAx$, and $A$ is initialized with a random Gaussian while $B$ is initialized to zero so that $BA = 0$ at the start of training."
        id="def-lora"
      />

      <h2 className="text-2xl font-semibold">The Rank Decomposition</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The key insight behind LoRA is that weight updates during finetuning have low intrinsic
        rank. Instead of updating the full <InlineMath math="d \times k" /> matrix, we
        parameterize the update as the product of two much smaller matrices.
      </p>

      <ExampleBlock
        title="Parameter Savings with LoRA"
        problem="Compare trainable parameters for full finetuning vs. LoRA on a weight matrix of size 4096 x 4096 with rank r = 16."
        steps={[
          { formula: '\\text{Full: } d \\times k = 4096 \\times 4096 = 16{,}777{,}216', explanation: 'Full finetuning updates all 16.7M parameters in this single matrix.' },
          { formula: '\\text{LoRA: } d \\times r + r \\times k = 4096 \\times 16 + 16 \\times 4096 = 131{,}072', explanation: 'LoRA only trains the two low-rank matrices, totaling 131K parameters.' },
          { formula: '\\text{Reduction: } \\frac{131{,}072}{16{,}777{,}216} = 0.78\\%', explanation: 'LoRA uses less than 1% of the parameters while capturing the essential adaptation.' },
        ]}
        id="example-param-savings"
      />

      <TheoremBlock
        title="LoRA Forward Pass"
        statement="For an input $x$, the LoRA-adapted linear layer computes:
$$h = W_0 x + \frac{\alpha}{r} BAx$$
where $\alpha$ is a scaling hyperparameter that controls the magnitude of the low-rank update relative to the pretrained weights. The ratio $\frac{\alpha}{r}$ normalizes the update so that changing $r$ does not require retuning the learning rate."
        id="thm-lora-forward"
      />

      <PythonCode
        title="lora_with_peft.py"
        code={`from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Load base model
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                        # Rank of the decomposition
    lora_alpha=32,               # Scaling factor (alpha/r applied)
    lora_dropout=0.05,           # Dropout on LoRA layers
    target_modules=[             # Which modules to apply LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",                 # Don't train bias terms
)

# Wrap model with LoRA adapters
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# trainable params: 13,631,488 || all params: 6,751,334,400 || trainable%: 0.2019

# The base weights are frozen, only LoRA matrices are trained
for name, param in model.named_parameters():
    if "lora" in name:
        print(f"TRAINABLE: {name}, shape={param.shape}")
    # All other params have requires_grad=False`}
        id="code-lora-peft"
      />

      <NoteBlock
        type="intuition"
        title="Why Low Rank Works"
        content="Research by Aghajanyan et al. (2021) showed that pretrained models have a low intrinsic dimensionality: the effective parameter space needed for adaptation is much smaller than the full parameter count. LoRA exploits this by constraining updates to a low-rank subspace. Rank r = 4 to 64 typically suffices; increasing r beyond this shows diminishing returns."
        id="note-low-rank-intuition"
      />

      <WarningBlock
        title="LoRA Rank Selection"
        content="Choosing rank r too low limits the model's ability to adapt (underfitting the task). Choosing r too high increases memory and compute without proportional gains. A good practice is to start with r = 16 and tune from there. Also, applying LoRA to all linear layers (not just attention) often improves results at modest additional cost."
        id="warning-rank-selection"
      />

      <NoteBlock
        type="tip"
        title="Merging LoRA Weights"
        content="After training, LoRA weights can be merged back into the base model: W_merged = W_0 + BA. This produces a standard model with zero inference overhead. Multiple LoRA adapters for different tasks can be swapped at serving time without reloading the base model, enabling efficient multi-tenant deployments."
        id="note-merging"
      />
    </div>
  )
}
