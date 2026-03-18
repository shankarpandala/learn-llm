import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Optimizers() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Optimizers: Adam and AdamW</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Optimizers determine how model parameters are updated based on computed gradients.
        Adam (Adaptive Moment Estimation) became the default optimizer for deep learning
        due to its adaptive learning rates and momentum. AdamW fixes a subtle but important
        issue with weight decay in Adam and is now the standard for training transformers.
      </p>

      <DefinitionBlock
        title="Adam Optimizer"
        definition="Adam maintains per-parameter exponential moving averages of the gradient (first moment $m_t$) and squared gradient (second moment $v_t$), with bias correction. The update rule is $\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$."
        notation="$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$, $\hat{m}_t = m_t/(1-\beta_1^t)$, $\hat{v}_t = v_t/(1-\beta_2^t)$."
        id="def-adam"
      />

      <h2 className="text-2xl font-semibold">Adam Update Equations</h2>
      <BlockMath math="m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment)}" />
      <BlockMath math="v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment)}" />
      <BlockMath math="\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(bias correction)}" />
      <BlockMath math="\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t" />

      <ExampleBlock
        title="Why Bias Correction Matters"
        problem="Show why m_t is biased toward zero at the start of training (beta_1=0.9)."
        steps={[
          { formula: 'm_1 = 0.9 \\times 0 + 0.1 \\times g_1 = 0.1 g_1', explanation: 'After step 1, the moving average is only 10% of the true gradient due to initialization at 0.' },
          { formula: '\\hat{m}_1 = m_1 / (1 - 0.9^1) = 0.1g_1 / 0.1 = g_1', explanation: 'Bias correction divides by (1 - beta^t), restoring the true scale.' },
          { formula: 'm_{10} \\approx 0.65 g_{\\text{avg}}', explanation: 'After 10 steps, the bias is smaller but still present.' },
          { formula: '\\hat{m}_{10} = m_{10} / (1 - 0.9^{10}) = m_{10} / 0.65 \\approx g_{\\text{avg}}', explanation: 'Correction factor (1 - 0.9^10) = 0.65 exactly compensates the bias.' },
        ]}
        id="example-bias-correction"
      />

      <h2 className="text-2xl font-semibold">AdamW: Decoupled Weight Decay</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Loshchilov and Hutter (2019) showed that L2 regularization and weight decay are
        <em> not equivalent</em> in Adam. Standard Adam with L2 regularization adds the
        penalty to the gradient, which gets scaled by the adaptive learning rate. AdamW
        instead applies weight decay directly to the parameters:
      </p>
      <BlockMath math="\text{Adam + L2: } g_t = \nabla \mathcal{L}(\theta_t) + \lambda \theta_t \quad \text{(coupled)}" />
      <BlockMath math="\text{AdamW: } \theta_{t+1} = \theta_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right) \quad \text{(decoupled)}" />

      <PythonCode
        title="adam_vs_adamw.py"
        code={`import torch
import torch.nn as nn

# Standard Adam with L2 regularization (NOT recommended)
model_l2 = nn.Linear(100, 10)
optimizer_l2 = torch.optim.Adam(
    model_l2.parameters(),
    lr=1e-3,
    weight_decay=0.01  # This is L2 in Adam, NOT true weight decay!
)

# AdamW with decoupled weight decay (RECOMMENDED)
model_wd = nn.Linear(100, 10)
optimizer_wd = torch.optim.AdamW(
    model_wd.parameters(),
    lr=1e-3,
    weight_decay=0.01  # True decoupled weight decay
)

# Typical transformer training configuration
model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),   # default momentum parameters
    eps=1e-8,              # numerical stability
    weight_decay=0.01      # standard for transformers
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Optimizer states per param: 2 (m_t and v_t)")
print(f"Optimizer memory: ~{sum(p.numel() for p in model.parameters()) * 2 * 4 / 1e6:.1f} MB")`}
        id="code-adam-adamw"
      />

      <PythonCode
        title="optimizer_param_groups.py"
        code={`import torch
import torch.nn as nn

class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(30000, 512)
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=6
        )
        self.ln = nn.LayerNorm(512)
        self.head = nn.Linear(512, 30000)

    def forward(self, x):
        return self.head(self.ln(self.layers(self.embedding(x))))

model = TransformerLM()

# Separate parameter groups: no weight decay for biases and LayerNorm
no_decay = ['bias', 'ln', 'LayerNorm']
param_groups = [
    {
        'params': [p for n, p in model.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01,
    },
    {
        'params': [p for n, p in model.named_parameters()
                   if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0,  # no decay for biases and norms
    },
]

optimizer = torch.optim.AdamW(param_groups, lr=3e-4, betas=(0.9, 0.98))

# Count parameters in each group
for i, group in enumerate(param_groups):
    n_params = sum(p.numel() for p in group['params'])
    print(f"Group {i}: {n_params:,} params, wd={group['weight_decay']}")`}
        id="code-param-groups"
      />

      <NoteBlock
        type="intuition"
        title="Why Adaptive Learning Rates Help NLP"
        content="Word embeddings for rare words receive gradients very infrequently. With SGD, these embeddings are barely updated. Adam's per-parameter adaptive rate means rare-word embeddings get larger effective learning rates (due to small v_t), while frequent-word embeddings get smaller rates. This is why Adam converges much faster than SGD on NLP tasks."
        id="note-adaptive-nlp"
      />

      <WarningBlock
        title="Adam's Memory Overhead"
        content="Adam stores two additional tensors (m_t and v_t) per parameter, tripling memory compared to SGD. For a 7B parameter model in fp32, parameters take 28GB, and Adam states add 56GB more, totaling 84GB. This is why mixed-precision training and optimizer offloading are essential for large models."
        id="warning-adam-memory"
      />

      <NoteBlock
        type="tip"
        title="Hyperparameter Defaults"
        content="For most NLP tasks, these defaults work well: lr=1e-4 to 3e-4, beta1=0.9, beta2=0.999 (or 0.98 for transformers), eps=1e-8, weight_decay=0.01 to 0.1. The learning rate is the most important hyperparameter to tune. Always use a learning rate schedule (warmup + decay) rather than a constant rate."
        id="note-defaults"
      />
    </div>
  )
}
