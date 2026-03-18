import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function KnowledgeDistillation() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Knowledge Distillation: Teacher-Student Training</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Knowledge distillation compresses a large teacher model into a smaller student model
        by training the student to mimic the teacher's soft probability distribution. This
        transfers the teacher's "dark knowledge" — the relative probabilities it assigns to
        incorrect classes — which contains far more information than hard labels alone.
      </p>

      <DefinitionBlock
        title="Knowledge Distillation"
        definition="Knowledge distillation trains a compact student network to reproduce the behavior of a larger teacher network. The student learns from the teacher's softened output distribution at temperature $T$: $q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$ where $z_i$ are the teacher's logits."
        notation="$L = \alpha \, L_{CE}(y, p_s) + (1 - \alpha) \, T^2 \, L_{KD}(q_t, q_s)$ where $L_{CE}$ is cross-entropy with hard labels, $L_{KD}$ is KL divergence between teacher and student soft outputs, and $\alpha$ balances the two terms."
        id="def-distillation"
      />

      <NoteBlock
        type="intuition"
        title="Why Soft Labels Help"
        content="When a teacher predicts 'cat' with 0.7, 'tiger' with 0.2, and 'dog' with 0.1, the student learns that cats look somewhat like tigers — structural knowledge absent from the hard label [1, 0, 0]. Temperature T > 1 further softens this distribution, amplifying these inter-class relationships."
        id="note-soft-labels"
      />

      <ExampleBlock
        title="Distillation Loss Computation"
        problem="Compute the distillation loss for a 3-class problem with temperature T=4, teacher logits [5, 3, 1], student logits [4, 2, 0.5], and true label class 0."
        steps={[
          {
            formula: 'q_t = \\text{softmax}([5/4, 3/4, 1/4]) = [0.506, 0.312, 0.182]',
            explanation: 'Soften teacher logits by dividing by temperature T=4.'
          },
          {
            formula: 'q_s = \\text{softmax}([4/4, 2/4, 0.5/4]) = [0.509, 0.297, 0.194]',
            explanation: 'Soften student logits similarly.'
          },
          {
            formula: 'L_{KD} = \\sum_i q_t^{(i)} \\log \\frac{q_t^{(i)}}{q_s^{(i)}} = 0.000457',
            explanation: 'KL divergence between teacher and student soft distributions.'
          },
          {
            formula: 'L = 0.5 \\cdot L_{CE} + 0.5 \\cdot 16 \\cdot 0.000457',
            explanation: 'Combine with hard-label loss. Multiply KD loss by T^2 = 16 to balance gradient magnitudes.'
          }
        ]}
        id="example-distillation-loss"
      />

      <PythonCode
        title="knowledge_distillation.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation."""
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        # Hard-label cross-entropy loss
        ce_loss = F.cross_entropy(student_logits, labels)

        # Soft-label KL divergence loss
        student_soft = F.log_softmax(student_logits / self.T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.T, dim=-1)
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')

        # Combine: multiply KD loss by T^2 to match gradient scale
        total = self.alpha * ce_loss + (1 - self.alpha) * (self.T ** 2) * kd_loss
        return total

# Example: distill a 6-layer teacher into a 2-layer student
teacher_logits = torch.tensor([[5.0, 3.0, 1.0], [2.0, 6.0, 0.5]])
student_logits = torch.tensor([[4.0, 2.0, 0.5], [1.5, 5.0, 0.3]])
labels = torch.tensor([0, 1])

criterion = DistillationLoss(temperature=4.0, alpha=0.5)
loss = criterion(student_logits, teacher_logits, labels)
print(f"Distillation loss: {loss.item():.4f}")

# Model size comparison
teacher_params = 110_000_000   # BERT-Base: 110M
student_params = 22_000_000    # DistilBERT: 22M (6 layers -> 3 layers, ~60% fewer params)
compression = teacher_params / student_params
print(f"Compression ratio: {compression:.1f}x")
print(f"Student retains ~97% of teacher performance on GLUE")`}
        id="code-distillation"
      />

      <TheoremBlock
        title="Gradient Scaling with Temperature"
        statement="The gradients of the KL divergence loss with respect to student logits scale as $1/T^2$ when using temperature $T$. Therefore, the distillation loss must be multiplied by $T^2$ to keep gradient magnitudes comparable to the hard-label loss."
        proof="For softmax with temperature, $\partial q_i / \partial z_j = (q_i(\delta_{ij} - q_j))/T$. Since KL divergence involves $\partial / \partial z$, the chain rule introduces two factors of $1/T$, yielding an overall $1/T^2$ scaling."
        id="thm-temperature-scaling"
      />

      <NoteBlock
        type="historical"
        title="Distillation in Practice"
        content="Hinton et al. (2015) formalized knowledge distillation. DistilBERT (Sanh et al., 2019) applied it to BERT, removing every other layer and achieving 97% of BERT's performance with 40% fewer parameters and 60% faster inference. TinyBERT further distills attention matrices and hidden states, not just output logits."
        id="note-distillation-history"
      />

      <WarningBlock
        title="Teacher Quality Matters"
        content="Distillation can only compress knowledge the teacher actually has. A poorly trained teacher produces noisy soft labels that may hurt the student. Always validate teacher performance before distillation. Also, very large temperature values can over-smooth distributions, losing discriminative information."
        id="warning-teacher-quality"
      />
    </div>
  )
}
