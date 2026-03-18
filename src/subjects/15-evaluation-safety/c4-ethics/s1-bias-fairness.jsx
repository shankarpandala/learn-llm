import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function BiasFairness() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Bias in LLMs and Fairness Metrics</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        LLMs trained on internet text inherit and can amplify societal biases related to
        race, gender, religion, and other protected attributes. Measuring and mitigating
        these biases is both a technical and ethical imperative.
      </p>

      <DefinitionBlock
        title="Algorithmic Bias"
        definition="Systematic and unfair differences in model behavior or outputs across demographic groups. In LLMs, bias manifests as stereotypical associations, differential performance across languages or dialects, and unequal representation in generated content."
        id="def-bias"
      />

      <h2 className="text-2xl font-semibold">Fairness Metrics</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Several formal fairness metrics have been proposed. Demographic parity requires equal
        positive prediction rates across groups:
      </p>
      <BlockMath math="\text{Demographic Parity: } P(\hat{Y}=1 | A=a) = P(\hat{Y}=1 | A=b) \quad \forall a, b" />
      <p className="text-gray-700 dark:text-gray-300">
        Equalized odds requires equal true positive and false positive rates:
      </p>
      <BlockMath math="\text{Equalized Odds: } P(\hat{Y}=1 | Y=y, A=a) = P(\hat{Y}=1 | Y=y, A=b) \quad \forall y, a, b" />
      <p className="text-gray-700 dark:text-gray-300">
        The disparate impact ratio measures group-level fairness:
      </p>
      <BlockMath math="\text{Disparate Impact} = \frac{P(\hat{Y}=1 | A=\text{minority})}{P(\hat{Y}=1 | A=\text{majority})}" />
      <p className="text-gray-700 dark:text-gray-300">
        A disparate impact ratio below 0.8 is generally considered evidence of adverse impact
        (the "four-fifths rule").
      </p>

      <TheoremBlock
        title="Impossibility of Fairness"
        statement="It is generally impossible to simultaneously satisfy demographic parity, equalized odds, and predictive parity (equal positive predictive values across groups) when base rates differ between groups (Chouldechova, 2017; Kleinberg et al., 2016)."
        corollaries={[
          'Practitioners must choose which fairness criteria to prioritize based on context.',
          'Trade-offs between fairness metrics should be explicitly documented and justified.',
        ]}
        id="thm-impossibility"
      />

      <ExampleBlock
        title="Measuring Bias in LLM Outputs"
        problem="Test whether a resume screening LLM shows gender bias."
        steps={[
          { formula: '\\text{Create matched pairs: identical resumes with different names}', explanation: 'Use names associated with different genders/ethnicities while keeping all qualifications identical.' },
          { formula: '\\text{Compute: } \\Delta = P(\\text{positive}|\\text{group A}) - P(\\text{positive}|\\text{group B})', explanation: 'Measure the difference in positive outcomes between groups.' },
          { formula: '\\text{Statistical test: } p\\text{-value for } H_0: \\Delta = 0', explanation: 'Use a chi-squared or permutation test to determine if differences are statistically significant.' },
        ]}
        id="example-bias-test"
      />

      <PythonCode
        title="bias_measurement.py"
        code={`import numpy as np
from collections import defaultdict

# Bias measurement for LLM-based classification
class BiasMeasurer:
    """Measure fairness metrics across demographic groups."""

    def __init__(self):
        self.predictions = defaultdict(list)  # group -> [(y_true, y_pred)]

    def add(self, group: str, y_true: int, y_pred: int):
        self.predictions[group].append((y_true, y_pred))

    def demographic_parity(self) -> dict:
        """Positive prediction rate per group."""
        rates = {}
        for group, preds in self.predictions.items():
            positives = sum(1 for _, yp in preds if yp == 1)
            rates[group] = positives / len(preds)
        return rates

    def equalized_odds(self) -> dict:
        """TPR and FPR per group."""
        result = {}
        for group, preds in self.predictions.items():
            tp = sum(1 for yt, yp in preds if yt == 1 and yp == 1)
            fn = sum(1 for yt, yp in preds if yt == 1 and yp == 0)
            fp = sum(1 for yt, yp in preds if yt == 0 and yp == 1)
            tn = sum(1 for yt, yp in preds if yt == 0 and yp == 0)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            result[group] = {"tpr": tpr, "fpr": fpr}
        return result

    def disparate_impact(self, minority: str, majority: str) -> float:
        """Disparate impact ratio."""
        rates = self.demographic_parity()
        if rates[majority] == 0:
            return float('inf')
        return rates[minority] / rates[majority]

# Simulate a resume screening scenario
np.random.seed(42)
measurer = BiasMeasurer()

# Simulate predictions with slight bias
for _ in range(500):
    qualified = np.random.random() > 0.4  # 60% are qualified
    # Group A: fair predictions
    pred_a = int(qualified) if np.random.random() > 0.1 else 1 - int(qualified)
    measurer.add("group_a", int(qualified), pred_a)
    # Group B: slightly biased (5% lower acceptance for same qualification)
    pred_b = int(qualified) if np.random.random() > 0.15 else 1 - int(qualified)
    measurer.add("group_b", int(qualified), pred_b)

print("Demographic Parity:", measurer.demographic_parity())
print("Equalized Odds:", measurer.equalized_odds())
di = measurer.disparate_impact("group_b", "group_a")
print(f"Disparate Impact: {di:.3f} {'(FAIR)' if di >= 0.8 else '(BIASED)'}")`}
        id="code-bias"
      />

      <WarningBlock
        title="Bias Is Multi-Dimensional"
        content="Bias in LLMs extends beyond binary protected attributes. It includes intersectional bias (e.g., Black women facing different bias than Black men or white women), linguistic bias (lower quality for non-standard dialects), and representation bias (whose perspectives are centered in generated text). Single-axis metrics miss much of the picture."
        id="warning-multi-dimensional"
      />

      <NoteBlock
        type="note"
        title="Debiasing Approaches"
        content="Common debiasing strategies include: balanced training data curation, contrastive learning to reduce stereotypical associations, RLHF with fairness-aware reward models, and post-processing output filters. No single technique eliminates all bias; continuous monitoring is essential."
        id="note-debiasing"
      />
    </div>
  )
}
