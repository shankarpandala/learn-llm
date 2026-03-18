import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function TeacherForcing() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Teacher Forcing</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Teacher forcing is a training strategy for autoregressive models where the ground-truth
        previous token is fed as input to the decoder at each step, rather than the model's own
        prediction. It dramatically speeds up training but introduces a train-test mismatch
        known as exposure bias.
      </p>

      <DefinitionBlock
        title="Teacher Forcing"
        definition="During training, at each decoder time step $t$, use the ground-truth token $y_{t-1}^*$ as input instead of the model's predicted token $\hat{y}_{t-1}$. The training objective becomes $\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t^* | y_1^*, \ldots, y_{t-1}^*, c)$."
        notation="With teacher forcing: input is $y_{t-1}^*$. Without: input is $\hat{y}_{t-1} = \arg\max P(y | h_{t-1})$."
        id="def-teacher-forcing"
      />

      <h2 className="text-2xl font-semibold">Why Teacher Forcing Works</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Without teacher forcing, early in training the decoder makes many mistakes. These errors
        compound: a wrong token at step 3 pushes the decoder into states it never saw during
        training, leading to garbage outputs for the rest of the sequence. Teacher forcing
        prevents this error cascade by always providing correct context.
      </p>
      <BlockMath math="\text{Teacher forcing: } h_t^{\text{dec}} = f(y_{t-1}^*, h_{t-1}^{\text{dec}}, c)" />
      <BlockMath math="\text{Free running: } h_t^{\text{dec}} = f(\hat{y}_{t-1}, h_{t-1}^{\text{dec}}, c)" />

      <DefinitionBlock
        title="Exposure Bias"
        definition="Exposure bias is the discrepancy between training (where the decoder sees ground-truth tokens) and inference (where it sees its own predictions). The model is never 'exposed' to its own errors during training, so it cannot learn to recover from mistakes."
        id="def-exposure-bias"
      />

      <ExampleBlock
        title="Exposure Bias in Translation"
        problem="Show how a single error cascades during free-running decoding."
        steps={[
          { formula: '\\text{Target: \"The cat sat on the mat\"}', explanation: 'Ground truth sequence the model should produce.' },
          { formula: '\\hat{y}_1 = \\text{\"The\"} \\checkmark', explanation: 'First token is correct.' },
          { formula: '\\hat{y}_2 = \\text{\"dog\"} \\text{ (error!)}', explanation: 'Model makes an error at step 2, predicting "dog" instead of "cat".' },
          { formula: '\\hat{y}_3 = \\text{\"ran\"} \\text{ (cascading)}', explanation: 'Conditioned on "The dog", the model generates a plausible continuation but completely diverges from the target.' },
          { formula: '\\hat{y}_4 = \\text{\"away\"} \\to \\text{total divergence}', explanation: 'Each step compounds the error. The decoder has never seen this trajectory during training.' },
        ]}
        id="example-exposure-bias"
      />

      <h2 className="text-2xl font-semibold">Scheduled Sampling</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Scheduled sampling (Bengio et al., 2015) bridges the gap between teacher forcing and
        free-running by gradually decreasing the probability of using ground-truth tokens during
        training. The curriculum starts with mostly teacher forcing and transitions to mostly
        free running.
      </p>
      <BlockMath math="\epsilon_i = \max(\epsilon_{\min}, k^i) \quad \text{(exponential decay)}" />
      <p className="text-gray-700 dark:text-gray-300">
        where <InlineMath math="\epsilon_i" /> is the probability of using ground truth at
        epoch <InlineMath math="i" />, and <InlineMath math="k < 1" /> controls the decay rate.
      </p>

      <PythonCode
        title="teacher_forcing_training.py"
        code={`import torch
import torch.nn as nn
import random

def train_step(model, src, tgt, optimizer, criterion,
               teacher_forcing_ratio=0.5):
    """Single training step with teacher forcing."""
    model.train()
    optimizer.zero_grad()

    batch_size, tgt_len = tgt.shape
    vocab_size = model.decoder.fc_out.out_features
    outputs = torch.zeros(batch_size, tgt_len, vocab_size,
                          device=src.device)

    hidden, cell = model.encoder(src)
    input_tok = tgt[:, 0:1]  # <SOS>

    for t in range(1, tgt_len):
        pred, hidden, cell = model.decoder(input_tok, hidden, cell)
        outputs[:, t] = pred

        # Decide: teacher forcing or autoregressive
        use_teacher = random.random() < teacher_forcing_ratio
        if use_teacher:
            input_tok = tgt[:, t:t+1]       # ground truth
        else:
            input_tok = pred.argmax(-1, keepdim=True)  # model prediction

    # Reshape for cross-entropy: (batch * tgt_len, vocab)
    loss = criterion(
        outputs[:, 1:].reshape(-1, vocab_size),
        tgt[:, 1:].reshape(-1)
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()`}
        id="code-teacher-forcing"
      />

      <PythonCode
        title="scheduled_sampling.py"
        code={`import math

def get_teacher_forcing_ratio(epoch, strategy='exponential',
                              k=0.95, min_ratio=0.1):
    """Compute scheduled sampling ratio."""
    if strategy == 'exponential':
        # Exponential decay: starts at 1.0, decays by k each epoch
        return max(min_ratio, k ** epoch)
    elif strategy == 'linear':
        # Linear decay over 50 epochs
        return max(min_ratio, 1.0 - epoch / 50)
    elif strategy == 'inverse_sigmoid':
        # Inverse sigmoid decay (smoother transition)
        return max(min_ratio, k / (k + math.exp(epoch / k)))
    else:
        return 0.5  # constant

# Training loop with scheduled sampling
num_epochs = 100
for epoch in range(num_epochs):
    tf_ratio = get_teacher_forcing_ratio(epoch, strategy='exponential')
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: teacher_forcing_ratio = {tf_ratio:.3f}")
    # loss = train_step(model, src, tgt, optimizer, criterion, tf_ratio)

# Output:
# Epoch 0: teacher_forcing_ratio = 1.000
# Epoch 10: teacher_forcing_ratio = 0.599
# Epoch 20: teacher_forcing_ratio = 0.358
# Epoch 30: teacher_forcing_ratio = 0.215
# Epoch 40: teacher_forcing_ratio = 0.129
# Epoch 50: teacher_forcing_ratio = 0.100 (clamped)`}
        id="code-scheduled-sampling"
      />

      <WarningBlock
        title="Scheduled Sampling Has Limitations"
        content="While scheduled sampling reduces exposure bias, it is not a principled solution. Mixing ground-truth and predicted tokens creates an inconsistent training signal -- the decoder sees a trajectory that is neither fully correct nor fully self-generated. More principled approaches include sequence-level training with REINFORCE or minimum risk training."
        id="warning-scheduled-limits"
      />

      <NoteBlock
        type="tip"
        title="Modern Approaches to Exposure Bias"
        content="Transformers largely sidestep exposure bias during training because teacher forcing is exact parallel computation (no sequential dependence). However, autoregressive decoding at inference time still suffers from error accumulation. Techniques like nucleus sampling, beam search with length penalties, and speculative decoding help manage this."
        id="note-modern-approaches"
      />
    </div>
  )
}
