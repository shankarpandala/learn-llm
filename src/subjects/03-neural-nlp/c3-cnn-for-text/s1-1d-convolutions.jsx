import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function OneDConvolutions() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">1D Convolutions for Text</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Convolutional neural networks, originally designed for images, can be adapted for text
        by treating a sequence of word embeddings as a 1D signal. A 1D convolutional filter
        slides across the sequence to detect local n-gram patterns. Unlike RNNs, CNNs process
        all positions in parallel, making them significantly faster to train.
      </p>

      <DefinitionBlock
        title="1D Convolution for Text"
        definition="Given an input sequence of embeddings $X \in \mathbb{R}^{L \times d}$ and a filter $w \in \mathbb{R}^{k \times d}$ of width $k$, the 1D convolution produces feature map $c_i = f(w \cdot X_{i:i+k-1} + b)$ where $X_{i:i+k-1}$ is a window of $k$ consecutive embeddings."
        notation="$L$ = sequence length, $d$ = embedding dimension, $k$ = filter width (kernel size), $f$ = nonlinear activation."
        id="def-1d-conv"
      />

      <h2 className="text-2xl font-semibold">How 1D Conv Detects N-grams</h2>
      <p className="text-gray-700 dark:text-gray-300">
        A filter of width <InlineMath math="k" /> looks at <InlineMath math="k" /> consecutive
        tokens at a time, functioning as a learnable n-gram detector. A filter of width 2
        detects bigrams, width 3 detects trigrams, and so on. Multiple filters with different
        widths capture patterns at different scales.
      </p>
      <BlockMath math="c_i = \text{ReLU}\left(\sum_{j=0}^{k-1} w_j \cdot x_{i+j} + b\right)" />
      <BlockMath math="\text{Output length} = L - k + 1 \quad \text{(no padding)}" />
      <BlockMath math="\text{Output length} = L \quad \text{(with padding } \lfloor k/2 \rfloor \text{)}" />

      <ExampleBlock
        title="Trigram Detection"
        problem="A filter of width 3 slides over the sentence 'I love this movie very much'. What n-grams does it capture?"
        steps={[
          { formula: '\\text{Position 0: } [\\text{I, love, this}]', explanation: 'The filter sees the first trigram.' },
          { formula: '\\text{Position 1: } [\\text{love, this, movie}]', explanation: 'Slides one position right.' },
          { formula: '\\text{Position 2: } [\\text{this, movie, very}]', explanation: 'Continues sliding...' },
          { formula: '\\text{Position 3: } [\\text{movie, very, much}]', explanation: 'Last valid window. With 6 tokens and filter width 3, we get 4 output positions.' },
        ]}
        id="example-trigram"
      />

      <PythonCode
        title="text_1d_conv.py"
        code={`import torch
import torch.nn as nn

# Demonstrate 1D convolution on text embeddings
vocab_size = 10000
embed_dim = 128
seq_len = 50
batch_size = 8

# Simulate embedded text
embedding = nn.Embedding(vocab_size, embed_dim)
tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
x = embedding(tokens)  # (8, 50, 128)

# For Conv1d, input shape is (batch, channels, length)
# Channels = embed_dim, length = seq_len
x_conv = x.transpose(1, 2)  # (8, 128, 50)

# Single filter of width 3 (trigram detector)
conv = nn.Conv1d(
    in_channels=embed_dim,   # input channels = embedding dim
    out_channels=64,         # number of filters
    kernel_size=3,           # filter width (trigram)
    padding=1,               # same padding to preserve length
)
output = conv(x_conv)  # (8, 64, 50)
print(f"Input: {x_conv.shape}")   # (8, 128, 50)
print(f"Output: {output.shape}")  # (8, 64, 50)

# Multiple filter sizes for multi-scale detection
for k in [2, 3, 4, 5]:
    conv_k = nn.Conv1d(embed_dim, 32, kernel_size=k)
    out_k = conv_k(x_conv)
    print(f"  kernel={k}: output length = {out_k.shape[2]}")
# kernel=2: output length = 49
# kernel=3: output length = 48
# kernel=4: output length = 47
# kernel=5: output length = 46`}
        id="code-1d-conv"
      />

      <PythonCode
        title="conv_pooling_classifier.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvClassifier(nn.Module):
    """Basic 1D CNN text classifier with max-over-time pooling."""
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_size,
                 num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) token IDs
        emb = self.embedding(x).transpose(1, 2)  # (batch, embed, seq)
        conv_out = F.relu(self.conv(emb))         # (batch, filters, L')
        # Max-over-time pooling: capture strongest activation
        pooled = conv_out.max(dim=2).values       # (batch, filters)
        return self.fc(self.dropout(pooled))

model = SimpleConvClassifier(
    vocab_size=10000, embed_dim=128,
    num_filters=100, kernel_size=3, num_classes=5
)
x = torch.randint(1, 10000, (16, 60))
print(f"Output: {model(x).shape}")  # (16, 5)`}
        id="code-conv-classifier"
      />

      <NoteBlock
        type="intuition"
        title="Max-Over-Time Pooling"
        content="After convolution, we apply max pooling across the entire sequence to get one value per filter. This captures the single strongest activation of each filter, regardless of where it occurred. If a filter detects 'not good', max pooling will fire whenever that pattern appears anywhere in the review, giving translation-invariant feature detection."
        id="note-max-pooling"
      />

      <WarningBlock
        title="CNNs Have a Fixed Receptive Field"
        content="A single convolutional layer with kernel size k can only see k consecutive tokens. To capture longer-range dependencies, you need either (1) deeper networks where multiple layers expand the receptive field, (2) larger kernels (expensive), or (3) dilated convolutions. This is a fundamental limitation compared to RNNs that can theoretically propagate information across the entire sequence."
        id="warning-receptive-field"
      />

      <NoteBlock
        type="tip"
        title="Padding Strategy Matters"
        content="Use 'same' padding (padding = kernel_size // 2) when you need the output to have the same length as input (for sequence labeling). Use no padding when you want to reduce length and will apply pooling afterward (for classification)."
        id="note-padding"
      />
    </div>
  )
}
