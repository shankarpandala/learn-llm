import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function GradientClipping() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Gradient Clipping and Regularization</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Gradient clipping prevents training instability by rescaling gradients when they
        exceed a threshold. Combined with regularization techniques like dropout and weight
        decay, these methods form the foundation of stable neural network training. Every
        modern LLM training run relies on gradient clipping to prevent occasional large
        gradients from derailing optimization.
      </p>

      <DefinitionBlock
        title="Gradient Clipping by Norm"
        definition="Clip-by-norm rescales the entire gradient vector when its L2 norm exceeds a threshold $\tau$: $g \leftarrow g \cdot \frac{\tau}{\|g\|}$ if $\|g\| > \tau$. This preserves the gradient direction while bounding its magnitude."
        notation="$g = \nabla_\theta \mathcal{L}$, $\tau$ = max norm threshold, $\|g\| = \sqrt{\sum_i g_i^2}$."
        id="def-clip-norm"
      />

      <DefinitionBlock
        title="Gradient Clipping by Value"
        definition="Clip-by-value independently clamps each gradient component to $[-\tau, \tau]$: $g_i \leftarrow \max(-\tau, \min(\tau, g_i))$. This changes the gradient direction but is simpler to implement."
        id="def-clip-value"
      />

      <h2 className="text-2xl font-semibold">Clip by Norm vs. Clip by Value</h2>
      <BlockMath math="\text{By norm: } g \leftarrow \begin{cases} g & \text{if } \|g\|_2 \leq \tau \\ \tau \cdot \frac{g}{\|g\|_2} & \text{if } \|g\|_2 > \tau \end{cases}" />
      <BlockMath math="\text{By value: } g_i \leftarrow \text{clip}(g_i, -\tau, \tau) \quad \forall i" />

      <ExampleBlock
        title="Clipping in Practice"
        problem="A gradient vector g = [3.0, 4.0] has norm 5.0. Apply clip-by-norm with tau=2.0 and clip-by-value with tau=2.0."
        steps={[
          { formula: '\\|g\\| = \\sqrt{9 + 16} = 5.0 > 2.0', explanation: 'Norm exceeds threshold, clipping will be applied.' },
          { formula: '\\text{By norm: } g \\leftarrow [3, 4] \\times 2/5 = [1.2, 1.6]', explanation: 'Both components scaled equally. Direction preserved: still points the same way.' },
          { formula: '\\text{By value: } g \\leftarrow [\\min(3, 2), \\min(4, 2)] = [2.0, 2.0]', explanation: 'Components clamped independently. Direction changes: original angle was ~53deg, now 45deg.' },
          { formula: '\\text{By norm } \\|g\\| = 2.0, \\text{ by value } \\|g\\| = 2.83', explanation: 'Clip-by-norm gives exact norm control; clip-by-value does not.' },
        ]}
        id="example-clipping"
      />

      <PythonCode
        title="gradient_clipping.py"
        code={`import torch
import torch.nn as nn

model = nn.LSTM(128, 256, num_layers=3, batch_first=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Simulate a training step
x = torch.randn(8, 50, 128)
output, _ = model(x)
target = torch.randint(0, 256, (8, 50))
loss = criterion(output.reshape(-1, 256), target.reshape(-1))
loss.backward()

# Method 1: Clip by global norm (RECOMMENDED)
# Returns the total norm before clipping
total_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0      # typical values: 0.5, 1.0, 5.0
)
print(f"Gradient norm before clip: {total_norm:.4f}")

optimizer.step()
optimizer.zero_grad()

# Method 2: Clip by value
loss.backward()  # recompute gradients
torch.nn.utils.clip_grad_value_(
    model.parameters(),
    clip_value=0.5    # clamp each element to [-0.5, 0.5]
)

# Method 3: Manual clipping (for monitoring)
loss.backward()
grad_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        grad_norm += p.grad.data.norm(2).item() ** 2
grad_norm = grad_norm ** 0.5
print(f"Manual grad norm: {grad_norm:.4f}")

# Clip
max_norm = 1.0
if grad_norm > max_norm:
    scale = max_norm / grad_norm
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.mul_(scale)`}
        id="code-clipping"
      />

      <h2 className="text-2xl font-semibold">Regularization Techniques</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Beyond gradient clipping, regularization prevents overfitting and improves
        generalization. The key techniques for NLP are dropout, weight decay, and
        layer normalization.
      </p>

      <PythonCode
        title="regularization_techniques.py"
        code={`import torch
import torch.nn as nn

class RegularizedLSTM(nn.Module):
    """LSTM with multiple regularization strategies."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Embedding dropout: randomly zero entire word embeddings
        self.embed_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,    # dropout between LSTM layers
        )
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # Output dropout
        self.output_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embed_dropout(self.embedding(x))
        output, (h_n, _) = self.lstm(emb)
        h = self.layer_norm(h_n[-1])
        return self.fc(self.output_dropout(h))

# Full training setup with all regularization
model = RegularizedLSTM(10000, 128, 256, 5, dropout=0.3)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01  # L2 regularization (decoupled)
)

# Training loop with gradient monitoring
for epoch in range(5):
    model.train()
    x = torch.randint(1, 10000, (32, 100))
    y = torch.randint(0, 5, (32,))

    optimizer.zero_grad()
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    # Monitor and clip
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0
    )
    optimizer.step()

    print(f"Epoch {epoch}: loss={loss.item():.4f}, "
          f"grad_norm={grad_norm:.4f}")`}
        id="code-regularization"
      />

      <NoteBlock
        type="tip"
        title="Standard Clipping Values"
        content="For RNNs/LSTMs, clip-by-norm with max_norm=5.0 is common (Pascanu et al., 2013). For transformers, max_norm=1.0 is standard (used in GPT-2, GPT-3, BERT). For fine-tuning pre-trained models, max_norm=1.0 with a lower learning rate (2e-5 to 5e-5) works well. Always log gradient norms during training -- sudden spikes indicate instability."
        id="note-clipping-values"
      />

      <WarningBlock
        title="Clipping Hides Problems"
        content="If gradient norms are consistently being clipped (e.g., > 50% of steps), the learning rate is likely too high, the model architecture has issues, or the data contains anomalous samples. Clipping is a safety net, not a solution. Investigate persistent clipping by examining loss spikes and adjusting hyperparameters."
        id="warning-hiding-problems"
      />

      <NoteBlock
        type="note"
        title="Gradient Norm as a Training Diagnostic"
        content="Logging gradient norms over training reveals important patterns: (1) Norms should generally decrease as training converges. (2) Sudden spikes may indicate bad data batches or learning rate issues. (3) Very small norms suggest vanishing gradients or a loss plateau. Tools like Weights & Biases and TensorBoard make this monitoring trivial."
        id="note-monitoring"
      />
    </div>
  )
}
