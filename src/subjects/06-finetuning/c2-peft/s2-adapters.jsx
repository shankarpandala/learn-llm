import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Adapters() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Adapter Layers: Bottleneck Architecture</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Adapter modules are small bottleneck layers inserted within each transformer block.
        During finetuning, only these adapter parameters are trained while the original model
        weights remain frozen. This was one of the first parameter-efficient finetuning methods,
        introduced by Houlsby et al. (2019).
      </p>

      <DefinitionBlock
        title="Adapter Layer"
        definition="An adapter is a bottleneck module inserted into a transformer layer. It consists of a down-projection $W_{\text{down}} \in \mathbb{R}^{d \times m}$, a nonlinearity $\sigma$, and an up-projection $W_{\text{up}} \in \mathbb{R}^{m \times d}$, plus a residual connection: $\text{Adapter}(h) = h + W_{\text{up}} \, \sigma(W_{\text{down}} h)$ where $m \ll d$ is the bottleneck dimension."
        id="def-adapter"
      />

      <h2 className="text-2xl font-semibold">Bottleneck Architecture</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The bottleneck compresses the hidden representation to a lower dimension, applies a
        nonlinearity, then projects back up. The residual connection ensures the adapter can
        learn the identity function (doing nothing) by initializing near zero.
      </p>

      <ExampleBlock
        title="Adapter Parameter Count"
        problem="Calculate the number of trainable parameters for adapters in a 12-layer transformer with hidden size 768 and bottleneck size 64."
        steps={[
          { formula: '\\text{Per adapter: } 2 \\times d \\times m = 2 \\times 768 \\times 64 = 98{,}304', explanation: 'Each adapter has a down-projection and up-projection (ignoring biases).' },
          { formula: '\\text{Adapters per layer: } 2 \\text{ (after attention + after FFN)}', explanation: 'The standard placement inserts adapters after both the self-attention and feed-forward sublayers.' },
          { formula: '\\text{Total: } 12 \\times 2 \\times 98{,}304 = 2{,}359{,}296', explanation: 'About 2.4M trainable parameters vs. ~110M total for BERT-base, roughly 2.1%.' },
        ]}
        id="example-adapter-params"
      />

      <PythonCode
        title="adapter_implementation.py"
        code={`import torch
import torch.nn as nn

class AdapterLayer(nn.Module):
    """Bottleneck adapter module with residual connection."""

    def __init__(self, hidden_size, bottleneck_size=64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        # Initialize near zero so adapter starts as identity
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, hidden_states):
        residual = hidden_states
        x = self.down_proj(hidden_states)
        x = self.activation(x)
        x = self.up_proj(x)
        return residual + x

# Example: inject adapter into a transformer layer
class TransformerLayerWithAdapter(nn.Module):
    def __init__(self, original_layer, hidden_size, bottleneck_size=64):
        super().__init__()
        self.original_layer = original_layer
        self.adapter_attn = AdapterLayer(hidden_size, bottleneck_size)
        self.adapter_ffn = AdapterLayer(hidden_size, bottleneck_size)
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, hidden_states, **kwargs):
        # Original self-attention
        attn_output = self.original_layer.attention(hidden_states, **kwargs)
        attn_output = self.adapter_attn(attn_output)  # Adapter after attention
        # Original feed-forward
        ffn_output = self.original_layer.feed_forward(attn_output)
        ffn_output = self.adapter_ffn(ffn_output)     # Adapter after FFN
        return ffn_output

# Using HuggingFace adapters library
from adapters import AutoAdapterModel
model = AutoAdapterModel.from_pretrained("bert-base-uncased")
model.add_adapter("sentiment", config="pfeiffer")  # Pfeiffer adapter config
model.train_adapter("sentiment")                    # Freeze all except adapter`}
        id="code-adapter"
      />

      <NoteBlock
        type="historical"
        title="Adapter Variants"
        content="Houlsby et al. (2019) proposed inserting two adapters per layer (after attention and FFN). Pfeiffer et al. (2021) showed that a single adapter after the FFN sublayer works nearly as well while halving the compute overhead. AdapterFusion (Pfeiffer et al., 2021) introduced a mechanism to combine multiple task adapters via attention."
        id="note-variants"
      />

      <WarningBlock
        title="Inference Latency Overhead"
        content="Unlike LoRA, adapter layers cannot be merged into the base model. They add sequential computation at every layer during inference, introducing a small but measurable latency overhead (typically 5-10%). For latency-sensitive applications, LoRA is often preferred since its weights can be folded into the base model at zero inference cost."
        id="warning-latency"
      />

      <NoteBlock
        type="tip"
        title="Choosing Bottleneck Size"
        content="The bottleneck dimension m controls the capacity-efficiency tradeoff. For simple tasks like sentiment classification, m = 16 to 32 often suffices. For complex tasks like question answering or generation, m = 64 to 256 may be needed. Monitor validation performance to find the sweet spot."
        id="note-bottleneck-size"
      />
    </div>
  )
}
