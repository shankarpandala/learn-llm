import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function CurriculumLearning() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Curriculum Learning and Data Ordering</h1>
      <p className="text-lg text-gray-300">
        Curriculum learning applies the principle that training benefits from a structured ordering
        of examples -- starting with easier or simpler data and progressively introducing harder
        material. In LLM pretraining, the order, mixture, and scheduling of data sources
        significantly impacts the final model quality.
      </p>

      <DefinitionBlock
        title="Curriculum Learning"
        definition="Curriculum learning trains a model by presenting examples in a meaningful order rather than random shuffling. A curriculum function $C(t)$ maps training step $t$ to a data distribution $\mathcal{D}_t$, typically progressing from simpler to more complex data."
        notation="$\mathcal{D}_t = C(t)$ where $\mathcal{D}_0$ is the easiest distribution and $\mathcal{D}_T$ approaches the full data distribution."
        id="curriculum-def"
      />

      <ExampleBlock
        title="Data Mixing Schedule"
        problem="Design a data mixing schedule for pretraining a 7B parameter LLM."
        steps={[
          {
            formula: '\\text{Phase 1 (0-50\\%): } \\{\\text{Web: 70\\%, Books: 15\\%, Code: 10\\%, Wiki: 5\\%}\\}',
            explanation: 'Start with a broad web-heavy mixture for general language understanding.'
          },
          {
            formula: '\\text{Phase 2 (50-80\\%): } \\{\\text{Web: 50\\%, Books: 20\\%, Code: 20\\%, Wiki: 10\\%}\\}',
            explanation: 'Increase code and high-quality sources to improve reasoning.'
          },
          {
            formula: '\\text{Phase 3 (80-100\\%): } \\{\\text{Web: 30\\%, Books: 25\\%, Code: 25\\%, Wiki: 15\\%, Math: 5\\%}\\}',
            explanation: 'Final phase emphasizes quality, reasoning, and factual data.'
          }
        ]}
        id="data-mixing-example"
      />

      <NoteBlock
        type="intuition"
        title="Why Data Order Matters"
        content="Random shuffling is the default but not necessarily optimal. Research shows that presenting high-quality data later in training (when the model can better leverage it) can improve final performance. This is analogous to how students learn fundamentals before advanced topics. The LLaMA and Qwen teams have confirmed that carefully designed data schedules improved their models."
        id="data-order-intuition"
      />

      <TheoremBlock
        title="Data Mixing Law"
        statement="For a mixture of $k$ data sources with weights $w_i$ ($\sum w_i = 1$), the overall loss approximately follows: $\mathcal{L}(w_1, \ldots, w_k) \approx \sum_{i=1}^k w_i \cdot \mathcal{L}_i(N \cdot w_i)$, where $\mathcal{L}_i(n)$ is the loss on source $i$ after training on $n$ tokens from that source."
        proof="Each source has its own scaling law $\mathcal{L}_i(n) = A_i / n^{\alpha_i} + \mathcal{L}_{\infty,i}$. The effective tokens from source $i$ is $n_i = N \cdot w_i$ where $N$ is total tokens. The weighted loss can be optimized over weights $w_i$ using Lagrange multipliers."
        id="mixing-law-thm"
      />

      <PythonCode
        title="curriculum_training.py"
        code={`import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np

class CurriculumScheduler:
    """Schedule data source mixing weights over training."""

    def __init__(self, sources, initial_weights, final_weights, warmup_steps):
        self.sources = sources
        self.initial = np.array(initial_weights, dtype=np.float64)
        self.final = np.array(final_weights, dtype=np.float64)
        self.warmup_steps = warmup_steps

    def get_weights(self, step):
        """Linearly interpolate from initial to final weights."""
        progress = min(step / self.warmup_steps, 1.0)
        weights = self.initial + progress * (self.final - self.initial)
        return weights / weights.sum()  # Normalize

# Define curriculum
scheduler = CurriculumScheduler(
    sources=["web", "books", "code", "wiki", "math"],
    initial_weights=[0.70, 0.15, 0.10, 0.04, 0.01],
    final_weights=  [0.30, 0.25, 0.25, 0.15, 0.05],
    warmup_steps=100_000
)

# Show schedule at different points
for step in [0, 25_000, 50_000, 75_000, 100_000]:
    w = scheduler.get_weights(step)
    parts = ", ".join(f"{s}={v:.1%}" for s, v in zip(scheduler.sources, w))
    print(f"Step {step:>7d}: {parts}")

# Data quality scoring for curriculum
def compute_difficulty(text, tokenizer=None):
    """Heuristic difficulty score based on text properties."""
    scores = {
        "length": min(len(text.split()) / 500, 1.0),
        "avg_word_len": min(np.mean([len(w) for w in text.split()]) / 10, 1.0),
        "unique_ratio": len(set(text.split())) / max(len(text.split()), 1),
    }
    return np.mean(list(scores.values()))

# Example: sort by difficulty
texts = [
    "The cat sat on the mat.",
    "Quantum entanglement describes correlations between particles.",
    "The Riemann hypothesis concerns the distribution of primes.",
]
scored = [(compute_difficulty(t), t) for t in texts]
for score, text in sorted(scored):
    print(f"  difficulty={score:.3f}: {text[:60]}")

# DoReMi-style domain reweighting
def doremi_update(domain_losses, domain_weights, ref_losses, step_size=0.01):
    """Update domain weights based on excess loss over reference."""
    excess = np.maximum(domain_losses - ref_losses, 0)
    log_weights = np.log(domain_weights) + step_size * excess
    new_weights = np.exp(log_weights)
    return new_weights / new_weights.sum()

weights = np.array([0.5, 0.3, 0.2])
losses = np.array([2.5, 3.1, 4.0])
ref = np.array([2.0, 2.5, 3.0])
new_w = doremi_update(losses, weights, ref)
print(f"\\nDoReMi reweighting: {weights} -> {new_w.round(3)}")`}
        id="curriculum-code"
      />

      <WarningBlock
        title="Curriculum Design Is Largely Empirical"
        content="There is no proven optimal curriculum for LLM pretraining. Most insights come from expensive ablation studies on smaller models and may not transfer perfectly to larger scales. Techniques like DoReMi attempt to automate mixing weight selection, but the search space is enormous. Always validate curriculum choices with held-out evaluations."
        id="curriculum-warning"
      />
    </div>
  )
}
