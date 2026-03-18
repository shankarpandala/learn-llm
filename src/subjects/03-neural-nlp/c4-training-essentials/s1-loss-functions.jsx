import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LossFunctions() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Loss Functions for Language Tasks</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The choice of loss function determines what the model optimizes for. In NLP,
        cross-entropy loss dominates classification and language modeling tasks, while
        specialized losses like CTC handle sequence alignment problems. Understanding
        these losses is critical for training effective models and diagnosing training issues.
      </p>

      <DefinitionBlock
        title="Cross-Entropy Loss"
        definition="For a classification problem with $C$ classes, the cross-entropy loss between the predicted distribution $\hat{y}$ and true label $y$ is $\mathcal{L} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$. For hard labels (one-hot), this simplifies to $\mathcal{L} = -\log(\hat{y}_{y^*})$ where $y^*$ is the true class."
        notation="$\hat{y}_c = \text{softmax}(z)_c = \frac{e^{z_c}}{\sum_j e^{z_j}}$, where $z$ are the logits."
        id="def-cross-entropy"
      />

      <h2 className="text-2xl font-semibold">Cross-Entropy for Language Modeling</h2>
      <p className="text-gray-700 dark:text-gray-300">
        In language modeling, cross-entropy is computed at each position in the sequence.
        The model predicts a distribution over the vocabulary
        <InlineMath math="V" /> at each step, and the loss is the negative log probability
        of the correct next token:
      </p>
      <BlockMath math="\mathcal{L}_{\text{LM}} = -\frac{1}{T}\sum_{t=1}^{T} \log P(w_t | w_1, \ldots, w_{t-1})" />
      <BlockMath math="\text{Perplexity} = \exp(\mathcal{L}_{\text{LM}}) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(w_t | w_{<t})\right)" />

      <ExampleBlock
        title="Cross-Entropy Calculation"
        problem="A model predicts P('cat')=0.7, P('dog')=0.2, P('fish')=0.1 and the true word is 'cat'. What is the loss?"
        steps={[
          { formula: '\\mathcal{L} = -\\log(0.7) = 0.357', explanation: 'Loss is the negative log probability of the correct class.' },
          { formula: '\\text{If true word were \"fish\": } \\mathcal{L} = -\\log(0.1) = 2.303', explanation: 'Lower probability assignments yield higher loss.' },
          { formula: '\\text{Perfect prediction: } \\mathcal{L} = -\\log(1.0) = 0', explanation: 'Zero loss when the model assigns probability 1 to the correct class.' },
          { formula: '\\text{Perplexity} = e^{0.357} = 1.43', explanation: 'Perplexity of 1.43 means the model is as confused as choosing uniformly among 1.43 options.' },
        ]}
        id="example-ce-calc"
      />

      <PythonCode
        title="cross_entropy_pytorch.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch cross-entropy combines log_softmax + NLLLoss
criterion = nn.CrossEntropyLoss()

# Classification example
logits = torch.tensor([[2.0, 1.0, 0.1]])  # raw model outputs
target = torch.tensor([0])                  # true class = 0
loss = criterion(logits, target)
print(f"Classification loss: {loss.item():.4f}")  # ~0.4170

# Manual verification
probs = F.softmax(logits, dim=-1)
manual_loss = -torch.log(probs[0, target[0]])
print(f"Manual loss: {manual_loss.item():.4f}")   # same

# Language modeling: loss over entire sequence
vocab_size = 10000
seq_len = 50
batch_size = 8

# Model outputs: (batch, seq_len, vocab_size)
logits_lm = torch.randn(batch_size, seq_len, vocab_size)
targets_lm = torch.randint(0, vocab_size, (batch_size, seq_len))

# Reshape for CrossEntropyLoss: (N, C) and (N,)
loss_lm = criterion(
    logits_lm.view(-1, vocab_size),
    targets_lm.view(-1)
)
perplexity = torch.exp(loss_lm)
print(f"LM loss: {loss_lm.item():.2f}")
print(f"Perplexity: {perplexity.item():.1f}")

# With padding: ignore padding tokens (index 0)
criterion_pad = nn.CrossEntropyLoss(ignore_index=0)
targets_padded = targets_lm.clone()
targets_padded[:, 40:] = 0  # simulate padding
loss_padded = criterion_pad(
    logits_lm.view(-1, vocab_size),
    targets_padded.view(-1)
)
print(f"Loss ignoring padding: {loss_padded.item():.2f}")`}
        id="code-cross-entropy"
      />

      <h2 className="text-2xl font-semibold">CTC Loss</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Connectionist Temporal Classification (CTC) loss handles sequence-to-sequence alignment
        when the input and output lengths differ and the alignment is unknown. It marginalizes
        over all possible alignments, making it ideal for speech recognition and OCR.
      </p>
      <BlockMath math="P(y | x) = \sum_{\pi \in \mathcal{B}^{-1}(y)} P(\pi | x)" />
      <p className="text-gray-700 dark:text-gray-300">
        where <InlineMath math="\mathcal{B}^{-1}(y)" /> is the set of all valid alignments
        that collapse to the target <InlineMath math="y" /> after removing blanks and
        repeated characters.
      </p>

      <PythonCode
        title="ctc_and_label_smoothing.py"
        code={`import torch
import torch.nn as nn

# CTC Loss for sequence alignment (e.g., speech recognition)
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

T = 50        # input sequence length
C = 28        # number of classes (alphabet + blank)
batch_size = 4
target_lengths = torch.tensor([5, 7, 4, 6])
input_lengths = torch.full((batch_size,), T, dtype=torch.long)

# Log probabilities: (T, batch, C) -- note: time-first!
log_probs = torch.randn(T, batch_size, C).log_softmax(dim=2)
targets = torch.randint(1, C, (sum(target_lengths),))  # skip blank=0

loss_ctc = ctc_loss(log_probs, targets, input_lengths, target_lengths)
print(f"CTC Loss: {loss_ctc.item():.2f}")

# Label Smoothing: prevents overconfident predictions
# Distributes (1 - smoothing) to true label, smoothing / (C-1) to others
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, logits, target):
        log_probs = torch.log_softmax(logits, dim=-1)
        # True class gets (1 - smoothing)
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1))
        nll = nll.squeeze(-1)
        # Uniform gets smoothing
        smooth = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean()

criterion_ls = LabelSmoothingLoss(classes=1000, smoothing=0.1)
logits = torch.randn(32, 1000)
targets = torch.randint(0, 1000, (32,))
print(f"Label smoothing loss: {criterion_ls(logits, targets).item():.2f}")`}
        id="code-ctc-smoothing"
      />

      <NoteBlock
        type="tip"
        title="Label Smoothing in Practice"
        content="Label smoothing with epsilon=0.1 is standard for transformer training. Instead of training toward a one-hot target, it distributes 10% of the probability mass uniformly. This prevents the model from becoming overconfident, improves calibration, and acts as a regularizer. Vaswani et al. (2017) used it in the original transformer paper."
        id="note-label-smoothing"
      />

      <WarningBlock
        title="Numerical Stability"
        content="Never compute cross-entropy as -log(softmax(x)). Instead, use log_softmax or PyTorch's CrossEntropyLoss, which applies the log-sum-exp trick for numerical stability. Direct computation of softmax followed by log can produce NaN or Inf for large logits due to floating-point overflow."
        id="warning-numerical"
      />
    </div>
  )
}
