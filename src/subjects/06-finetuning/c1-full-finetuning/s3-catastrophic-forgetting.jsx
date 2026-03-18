import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function CatastrophicForgetting() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Catastrophic Forgetting and Elastic Weight Consolidation</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        When a neural network is finetuned on a new task, it tends to forget what it learned
        during pretraining or on previous tasks. This phenomenon, known as catastrophic forgetting,
        is a fundamental challenge in continual learning. Elastic Weight Consolidation (EWC) is
        a principled approach that uses the Fisher information matrix to protect important weights.
      </p>

      <DefinitionBlock
        title="Catastrophic Forgetting"
        definition="Catastrophic forgetting occurs when a neural network, upon learning new information, abruptly loses previously learned knowledge. Formally, after training on task $A$ then task $B$, performance on task $A$ degrades significantly because gradient updates for $B$ overwrite parameters critical to $A$."
        id="def-catastrophic-forgetting"
      />

      <WarningBlock
        title="Real-World Impact"
        content="Catastrophic forgetting is not just a theoretical concern. When finetuning an LLM for code generation, the model may lose its ability to hold a conversation. When finetuning for medical Q&A, it may forget basic reasoning. This makes naive sequential finetuning on multiple tasks unreliable."
        id="warning-real-impact"
      />

      <h2 className="text-2xl font-semibold">Elastic Weight Consolidation (EWC)</h2>
      <p className="text-gray-700 dark:text-gray-300">
        EWC adds a regularization term that penalizes changes to parameters that were important
        for previous tasks. Importance is measured by the diagonal of the Fisher information
        matrix, which approximates the curvature of the loss landscape.
      </p>

      <TheoremBlock
        title="EWC Objective"
        statement="The EWC loss for task $B$, given a model previously trained on task $A$ with optimal parameters $\theta^*_A$, is:
$$\mathcal{L}_{EWC}(\theta) = \mathcal{L}_B(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta^*_{A,i})^2$$
where $F_i$ is the diagonal element of the Fisher information matrix for parameter $i$, measuring its importance to task $A$."
        proof="Starting from a Bayesian perspective, we want $\log p(\theta | \mathcal{D}_A, \mathcal{D}_B)$. By Bayes' rule, $\log p(\theta | \mathcal{D}_A, \mathcal{D}_B) = \log p(\mathcal{D}_B | \theta) + \log p(\theta | \mathcal{D}_A) - \log p(\mathcal{D}_B)$. Approximating $\log p(\theta | \mathcal{D}_A)$ with a Laplace approximation (second-order Taylor around $\theta^*_A$) gives the Fisher-weighted quadratic penalty."
        id="thm-ewc"
      />

      <ExampleBlock
        title="Fisher Information Diagonal"
        problem="Compute the Fisher information for a simple model parameter."
        steps={[
          { formula: 'F_i = \\mathbb{E}\\left[\\left(\\frac{\\partial \\log p(y|x,\\theta)}{\\partial \\theta_i}\\right)^2\\right]', explanation: 'The Fisher information measures how sensitive the log-likelihood is to changes in parameter i.' },
          { formula: 'F_i \\approx \\frac{1}{N} \\sum_{n=1}^{N} \\left(\\frac{\\partial \\log p(y_n|x_n,\\theta^*_A)}{\\partial \\theta_i}\\right)^2', explanation: 'In practice, approximate with the empirical Fisher over N data points from task A.' },
          { formula: '\\text{High } F_i \\Rightarrow \\text{parameter } i \\text{ is important for task A}', explanation: 'Parameters with high Fisher values are strongly penalized when they deviate from their task-A values.' },
        ]}
        id="example-fisher"
      />

      <PythonCode
        title="ewc_implementation.py"
        code={`import torch
import torch.nn.functional as F
from copy import deepcopy

class EWC:
    """Elastic Weight Consolidation for continual learning."""

    def __init__(self, model, dataloader, device='cuda', num_samples=200):
        self.model = model
        self.device = device
        # Store optimal parameters from task A
        self.params_a = {n: p.clone().detach()
                         for n, p in model.named_parameters() if p.requires_grad}
        # Compute Fisher information diagonal
        self.fisher = self._compute_fisher(dataloader, num_samples)

    def _compute_fisher(self, dataloader, num_samples):
        fisher = {n: torch.zeros_like(p)
                  for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**inputs)
            loss = outputs.loss
            self.model.zero_grad()
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            count += 1
        # Average over samples
        for n in fisher:
            fisher[n] /= count
        return fisher

    def penalty(self, model, lam=1000):
        """Compute EWC penalty: (lambda/2) * sum_i F_i * (theta_i - theta_A_i)^2"""
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params_a[n]) ** 2).sum()
        return (lam / 2) * loss

# Usage in training loop:
# total_loss = task_b_loss + ewc.penalty(model, lam=5000)`}
        id="code-ewc"
      />

      <NoteBlock
        type="tip"
        title="Alternatives to EWC"
        content="Other approaches to mitigating catastrophic forgetting include: (1) Replay-based methods that mix old task data into new training, (2) Progressive Networks that add new capacity for each task, (3) Knowledge Distillation where the finetuned model is regularized to match the original model's outputs, and (4) Parameter-efficient methods like LoRA that keep the original weights frozen entirely."
        id="note-alternatives"
      />

      <NoteBlock
        type="intuition"
        title="The Geometry of Forgetting"
        content="Think of the loss landscape as a mountainous terrain. The pretrained model sits in a valley that works well for general language. Finetuning moves the model to a new valley for the specific task. If these valleys are far apart in parameter space, the model loses its general capabilities. EWC keeps the model close to the original valley along dimensions that matter most."
        id="note-geometry"
      />
    </div>
  )
}
