import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function PrefixTuning() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Prefix Tuning and Soft Prompts</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Prefix tuning and prompt tuning are parameter-efficient methods that prepend trainable
        continuous vectors (soft prompts) to the input or hidden states, steering the frozen model's
        behavior without modifying any of its original parameters. Instead of engineering discrete
        text prompts, the model learns optimal prompt embeddings through gradient descent.
      </p>

      <DefinitionBlock
        title="Prefix Tuning"
        definition="Prefix tuning (Li & Liang, 2021) prepends trainable prefix vectors $P_k, P_v \in \mathbb{R}^{l \times d}$ to the key and value matrices at every transformer layer, where $l$ is the prefix length and $d$ is the hidden dimension. The attention computation becomes $\text{Attn}(Q, [P_k; K], [P_v; V])$, allowing the prefix to influence all subsequent token representations without modifying model weights."
        id="def-prefix-tuning"
      />

      <h2 className="text-2xl font-semibold">Prompt Tuning vs. Prefix Tuning</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Prompt tuning (Lester et al., 2021) is a simplified variant that only prepends trainable
        embeddings to the input layer, rather than all layers. It uses fewer parameters but
        requires larger models to match prefix tuning's performance. At the scale of 10B+
        parameters, prompt tuning matches full finetuning performance.
      </p>

      <ExampleBlock
        title="Parameter Comparison: Prefix vs. Prompt Tuning"
        problem="Compare trainable parameters for a 12-layer model with hidden size 768 and prefix length 20."
        steps={[
          { formula: '\\text{Prompt tuning: } l \\times d = 20 \\times 768 = 15{,}360', explanation: 'Only input-layer embeddings are trained, giving very few parameters.' },
          { formula: '\\text{Prefix tuning: } 2 \\times L \\times l \\times d = 2 \\times 12 \\times 20 \\times 768 = 368{,}640', explanation: 'Key and value prefixes at all 12 layers (2x for K and V).' },
          { formula: '\\text{Prefix is } 24\\times \\text{ more params, but still } < 0.4\\% \\text{ of BERT-base}', explanation: 'Both methods are extremely parameter-efficient compared to the full model.' },
        ]}
        id="example-prefix-params"
      />

      <PythonCode
        title="prefix_tuning_peft.py"
        code={`from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PrefixTuningConfig, get_peft_model, TaskType

model_name = "gpt2-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prefix tuning configuration
prefix_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,        # Length of the prefix
    encoder_hidden_size=1024,     # Hidden size of the reparameterization MLP
    prefix_projection=True,       # Use MLP to reparameterize prefix
)

model = get_peft_model(model, prefix_config)
model.print_trainable_parameters()
# trainable params: 983,040 || all params: 355,958,784 || trainable%: 0.28

# Prompt tuning (simpler variant)
from peft import PromptTuningConfig, PromptTuningInit

prompt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,
    prompt_tuning_init=PromptTuningInit.TEXT,       # Initialize from text
    prompt_tuning_init_text="Classify the sentiment of this text: ",
    tokenizer_name_or_path=model_name,
)

# The soft prompt is initialized from the embeddings of the init text
# then optimized via backpropagation during training`}
        id="code-prefix-tuning"
      />

      <NoteBlock
        type="intuition"
        title="What Do Soft Prompts Learn?"
        content="Soft prompts live in the continuous embedding space and are not constrained to correspond to any real tokens. Analysis shows they often encode task-specific instructions in a form that is more expressive than any discrete text prompt could be. They can represent directions in embedding space that have no single-token equivalent, exploiting the model's representation geometry directly."
        id="note-soft-prompt-meaning"
      />

      <WarningBlock
        title="Prefix Length vs. Context Length"
        content="Prefix tokens consume positions in the model's context window. A prefix of length 20 reduces your available context by 20 tokens. For tasks requiring long inputs (e.g., document summarization), keep the prefix short. Additionally, very long prefixes (100+) can lead to optimization difficulties without proportional performance gains."
        id="warning-prefix-length"
      />

      <NoteBlock
        type="tip"
        title="Reparameterization Trick"
        content="Directly optimizing prefix vectors can be unstable due to their high dimensionality relative to the small number of parameters. Prefix tuning uses a reparameterization MLP during training: the prefix is generated by a small feedforward network. After training, the MLP is discarded and only the generated prefix vectors are kept for inference."
        id="note-reparam"
      />
    </div>
  )
}
