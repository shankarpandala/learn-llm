import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function TextCNN() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">TextCNN (Kim 2014)</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        TextCNN, proposed by Yoon Kim in "Convolutional Neural Networks for Sentence
        Classification" (2014), demonstrated that a simple CNN architecture with multiple
        filter sizes and pre-trained word embeddings achieves excellent results on text
        classification. Despite its simplicity, TextCNN remains a strong baseline and is
        widely used in production due to its speed and effectiveness.
      </p>

      <DefinitionBlock
        title="TextCNN Architecture"
        definition="TextCNN applies multiple convolutional filters of different widths (e.g., 3, 4, 5) to an embedding matrix, applies ReLU and max-over-time pooling to each filter's output, concatenates the pooled features, and classifies through a fully connected layer with dropout."
        id="def-textcnn"
      />

      <h2 className="text-2xl font-semibold">Architecture Overview</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The model has four stages:
      </p>
      <BlockMath math="\text{Embed} \to \text{Conv}_{k_1, k_2, k_3} \to \text{ReLU + MaxPool} \to \text{Concat + FC}" />
      <p className="text-gray-700 dark:text-gray-300">
        For each kernel size <InlineMath math="k_i" /> with <InlineMath math="n_f" /> filters,
        the convolution produces an output of
        shape <InlineMath math="(n_f, L - k_i + 1)" />. Max-over-time pooling reduces each
        filter to a single scalar, giving <InlineMath math="n_f" /> features per kernel size.
        With 3 kernel sizes and 100 filters each, the final representation is 300-dimensional.
      </p>

      <ExampleBlock
        title="TextCNN Feature Dimensions"
        problem="Calculate the feature dimensions for TextCNN with kernel_sizes=[3,4,5], 100 filters each, on a 60-token input."
        steps={[
          { formula: '\\text{Conv}_{k=3}: (100, 58) \\xrightarrow{\\text{maxpool}} (100,)', explanation: 'Kernel 3: 60-3+1=58 positions, max-pooled to 100 features.' },
          { formula: '\\text{Conv}_{k=4}: (100, 57) \\xrightarrow{\\text{maxpool}} (100,)', explanation: 'Kernel 4: 60-4+1=57 positions, max-pooled to 100 features.' },
          { formula: '\\text{Conv}_{k=5}: (100, 56) \\xrightarrow{\\text{maxpool}} (100,)', explanation: 'Kernel 5: 60-5+1=56 positions, max-pooled to 100 features.' },
          { formula: '\\text{Concatenate: } 100 + 100 + 100 = 300', explanation: 'Final feature vector is 300-dimensional, fed to a dropout + linear layer.' },
        ]}
        id="example-textcnn-dims"
      />

      <PythonCode
        title="textcnn.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    """Kim (2014) CNN for sentence classification."""
    def __init__(self, vocab_size, embed_dim, num_classes,
                 num_filters=100, kernel_sizes=(3, 4, 5),
                 dropout=0.5, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        # One Conv1d per kernel size
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len) of token IDs
        emb = self.embedding(x).transpose(1, 2)  # (batch, embed, seq)

        # Apply each conv + ReLU + max-pool
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))            # (batch, num_filters, L')
            p = c.max(dim=2).values          # (batch, num_filters)
            pooled.append(p)

        # Concatenate all filter outputs
        cat = torch.cat(pooled, dim=1)       # (batch, num_filters * 3)
        return self.fc(self.dropout(cat))

# Instantiate and test
model = TextCNN(
    vocab_size=25000, embed_dim=300, num_classes=5,
    num_filters=100, kernel_sizes=(3, 4, 5)
)
x = torch.randint(1, 25000, (32, 60))
logits = model(x)
print(f"Output: {logits.shape}")  # (32, 5)

# Count parameters
total = sum(p.numel() for p in model.parameters())
embed = model.embedding.weight.numel()
print(f"Total params: {total:,}")
print(f"Embedding params: {embed:,}")
print(f"CNN params: {total - embed:,}")`}
        id="code-textcnn"
      />

      <PythonCode
        title="textcnn_multichannel.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNNMultichannel(nn.Module):
    """Kim (2014) multichannel variant: static + non-static embeddings."""
    def __init__(self, vocab_size, embed_dim, num_classes,
                 num_filters=100, kernel_sizes=(3, 4, 5)):
        super().__init__()
        # Channel 1: static (frozen pretrained)
        self.embed_static = nn.Embedding(vocab_size, embed_dim)
        self.embed_static.weight.requires_grad = False

        # Channel 2: non-static (fine-tuned pretrained)
        self.embed_nonstatic = nn.Embedding(vocab_size, embed_dim)

        # Convolutions operate on 2-channel input
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim * 2, num_filters, k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        e1 = self.embed_static(x).transpose(1, 2)
        e2 = self.embed_nonstatic(x).transpose(1, 2)
        emb = torch.cat([e1, e2], dim=1)  # (batch, 2*embed, seq)

        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))
            pooled.append(c.max(dim=2).values)
        cat = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(cat))

model = TextCNNMultichannel(25000, 300, 5)
x = torch.randint(1, 25000, (16, 50))
print(f"Multichannel output: {model(x).shape}")  # (16, 5)`}
        id="code-multichannel"
      />

      <NoteBlock
        type="note"
        title="Kim's Key Findings"
        content="Kim (2014) showed that: (1) Pre-trained word2vec embeddings are crucial -- random initialization loses ~2% accuracy. (2) The multichannel model (one static, one fine-tuned embedding) helps on small datasets. (3) A single-layer CNN with dropout 0.5 is sufficient -- deeper is not always better for this architecture. (4) TextCNN matches or beats more complex models on 4 out of 7 benchmarks."
        id="note-kim-findings"
      />

      <WarningBlock
        title="TextCNN Limitations"
        content="TextCNN treats max-pooled features as a bag of detected patterns with no positional information. It cannot model word order beyond the kernel size. For tasks requiring long-range reasoning (e.g., document-level inference, coreference resolution), RNNs or transformers are needed."
        id="warning-textcnn-limits"
      />

      <NoteBlock
        type="historical"
        title="TextCNN's Lasting Impact"
        content="Despite being published in 2014, TextCNN remains surprisingly competitive. It is still used as a baseline in NLP papers and deployed in production for tasks where speed matters (content moderation, spam detection). The key lesson: for classification tasks with short texts, local n-gram features captured by CNNs are often sufficient."
        id="note-textcnn-impact"
      />
    </div>
  )
}
