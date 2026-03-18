import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function DilatedConv() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Dilated Convolutions</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Dilated (atrous) convolutions expand the receptive field exponentially without
        increasing the number of parameters or losing resolution. By inserting gaps between
        filter elements, a dilated convolution with dilation rate <InlineMath math="d" /> and
        kernel size <InlineMath math="k" /> has an effective receptive field
        of <InlineMath math="k + (k-1)(d-1)" /> tokens. Stacking layers with
        exponentially growing dilation rates allows a network to model very long-range
        dependencies efficiently.
      </p>

      <DefinitionBlock
        title="Dilated Convolution"
        definition="A dilated convolution with dilation rate $d$ applies the filter at every $d$-th position: $(x *_d w)_t = \sum_{j=0}^{k-1} w_j \cdot x_{t - d \cdot j}$. Standard convolution is the special case $d=1$."
        notation="$d$ = dilation rate, $k$ = kernel size. Effective receptive field = $k + (k-1)(d-1)$."
        id="def-dilated-conv"
      />

      <h2 className="text-2xl font-semibold">Exponential Receptive Field Growth</h2>
      <p className="text-gray-700 dark:text-gray-300">
        By stacking layers with dilation rates <InlineMath math="1, 2, 4, 8, \ldots" />,
        the receptive field grows exponentially while the number of parameters grows
        only linearly:
      </p>
      <BlockMath math="\text{Receptive field after } L \text{ layers} = 1 + k \cdot \sum_{l=0}^{L-1}(d_l - 1) + (k-1) \cdot L" />
      <p className="text-gray-700 dark:text-gray-300">
        With kernel size 3 and dilation rates [1, 2, 4, 8], four layers cover a receptive
        field of 31 tokens, compared to just 9 for standard convolutions.
      </p>

      <ExampleBlock
        title="Receptive Field Comparison"
        problem="Compare receptive fields for 4 stacked conv layers (kernel=3) with vs. without dilation."
        steps={[
          { formula: '\\text{Standard (d=1): } 3 + 2 + 2 + 2 = 9 \\text{ tokens}', explanation: 'Each standard conv layer adds (k-1)=2 to the receptive field.' },
          { formula: '\\text{Dilated (d=1,2,4,8): } 3 + 4 + 8 + 16 = 31 \\text{ tokens}', explanation: 'Dilation doubles the receptive field contribution of each layer.' },
          { formula: '\\text{Parameters: identical } (4 \\times 3 \\times d_{\\text{model}}^2)', explanation: 'Both use the same kernel size and number of layers -- same parameter count!' },
          { formula: '\\text{Ratio: } 31 / 9 \\approx 3.4\\times', explanation: 'Dilated convolutions cover 3.4x more context with zero extra parameters.' },
        ]}
        id="example-receptive-field"
      />

      <PythonCode
        title="dilated_conv_demo.py"
        code={`import torch
import torch.nn as nn

# Compare standard vs. dilated convolution
embed_dim = 128
x = torch.randn(4, embed_dim, 100)  # (batch, channels, seq_len)

# Standard convolution stack
conv_standard = nn.Sequential(
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, dilation=1),
    nn.ReLU(),
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, dilation=1),
    nn.ReLU(),
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, dilation=1),
    nn.ReLU(),
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, dilation=1),
)

# Dilated convolution stack (note: padding = dilation for 'same' output)
conv_dilated = nn.Sequential(
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, dilation=1),
    nn.ReLU(),
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=2, dilation=2),
    nn.ReLU(),
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=4, dilation=4),
    nn.ReLU(),
    nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=8, dilation=8),
)

out_std = conv_standard(x)
out_dil = conv_dilated(x)
print(f"Standard output: {out_std.shape}")  # (4, 128, 100)
print(f"Dilated output:  {out_dil.shape}")  # (4, 128, 100)

# Same parameters, but dilated has much larger receptive field
params_std = sum(p.numel() for p in conv_standard.parameters())
params_dil = sum(p.numel() for p in conv_dilated.parameters())
print(f"Standard params: {params_std:,}")  # 262,656
print(f"Dilated params:  {params_dil:,}")  # 262,656 -- identical!`}
        id="code-dilated-demo"
      />

      <PythonCode
        title="dilated_causal_conv.py"
        code={`import torch
import torch.nn as nn

class CausalDilatedConv(nn.Module):
    """Causal dilated convolution block (WaveNet-style)."""
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        # Causal padding: pad only on the left side
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              dilation=dilation)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        # Left-pad for causal convolution
        x_padded = nn.functional.pad(x, (self.padding, 0))
        out = self.conv(x_padded)
        # Residual connection
        out = out + x
        # Layer norm (transpose for channel-last)
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return torch.relu(out)

class DilatedConvStack(nn.Module):
    """Stack of causal dilated convolutions with residual connections."""
    def __init__(self, vocab_size, embed_dim, num_layers=6,
                 num_classes=5, kernel_size=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            CausalDilatedConv(embed_dim, kernel_size, dilation=2**i)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)  # (batch, embed, seq)
        h = emb
        for layer in self.layers:
            h = layer(h)
        # Global average pooling
        pooled = h.mean(dim=2)  # (batch, embed)
        return self.fc(pooled)

model = DilatedConvStack(10000, 128, num_layers=6, num_classes=5)
x = torch.randint(1, 10000, (8, 200))
print(f"Output: {model(x).shape}")  # (8, 5)
# Receptive field: 1 + 2*(1+2+4+8+16+32) = 127 tokens with just 6 layers!
print(f"Receptive field: {1 + 2*(1+2+4+8+16+32)} tokens")`}
        id="code-causal-dilated"
      />

      <NoteBlock
        type="intuition"
        title="Why Dilation Works"
        content="Imagine reading a document by looking at every word (standard), versus sampling every 2nd, 4th, and 8th word in successive passes. Each pass captures patterns at a different scale. Dilation lets each layer specialize: lower layers detect local patterns (bigrams, trigrams), while higher layers with large dilation rates detect document-level themes."
        id="note-dilation-intuition"
      />

      <WarningBlock
        title="Gridding Artifacts"
        content="When dilation rates are too large or poorly chosen, dilated convolutions can develop 'gridding artifacts' where some input positions are never covered by any filter. A common mitigation is to repeat the dilation pattern (e.g., [1,2,4,1,2,4]) or use hybrid architectures that combine dilated and standard convolutions."
        id="warning-gridding"
      />

      <NoteBlock
        type="historical"
        title="From WaveNet to Text"
        content="Dilated causal convolutions were popularized by WaveNet (van den Oord et al., 2016) for audio generation. The ByteNet architecture (Kalchbrenner et al., 2017) adapted this for NLP, achieving competitive machine translation results. These architectures inspired the idea that non-recurrent models could handle sequences effectively, paving the way for transformers."
        id="note-wavenet-history"
      />
    </div>
  )
}
