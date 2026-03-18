import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ABTesting() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">A/B Testing Fine-tuned Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        A/B testing compares two model variants under real-world conditions by routing
        traffic between them and measuring user-facing metrics. It is the definitive way
        to determine whether a fine-tuned model improves production outcomes compared to
        the baseline.
      </p>

      <DefinitionBlock
        title="A/B Testing for Models"
        definition="A/B testing (split testing) randomly assigns users or requests to one of two model variants and compares outcome metrics. Statistical significance is determined using hypothesis testing: given $n_A$ and $n_B$ samples with success rates $p_A$ and $p_B$, we compute $z = \frac{p_B - p_A}{\sqrt{p(1-p)(1/n_A + 1/n_B)}}$ where $p$ is the pooled rate."
        id="def-ab-testing"
      />

      <ExampleBlock
        title="Designing a Model A/B Test"
        problem="What metrics should you track when A/B testing a fine-tuned model?"
        steps={[
          { formula: '\\text{Primary: task completion rate}', explanation: 'Did the user accomplish their goal? This is the most important signal.' },
          { formula: '\\text{Latency: } t_{\\text{p50}}, t_{\\text{p95}}, t_{\\text{p99}}', explanation: 'Response time percentiles -- a better model is useless if too slow.' },
          { formula: '\\text{User satisfaction: thumbs up/down ratio}', explanation: 'Direct feedback signal from users interacting with the model.' },
          { formula: '\\text{Engagement: follow-up rate, session length}', explanation: 'Indirect signals of quality -- users engage more with better models.' },
        ]}
        id="example-ab-metrics"
      />

      <PythonCode
        title="ab_test_router.py"
        code={`import random
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class ABTestConfig:
    experiment_name: str
    model_a: str  # control (base/current model)
    model_b: str  # treatment (fine-tuned model)
    traffic_split: float = 0.5  # fraction going to model_b
    min_samples: int = 1000

class ABTestRouter:
    def __init__(self, config: ABTestConfig):
        self.config = config
        self.results = {"model_a": [], "model_b": []}

    def route_request(self, request_id: str) -> str:
        """Deterministically route request to model A or B."""
        bucket = hash(request_id) % 100
        if bucket < self.config.traffic_split * 100:
            return self.config.model_b
        return self.config.model_a

    def log_result(self, request_id: str, model: str, latency: float,
                   success: bool, user_rating: Optional[int] = None):
        key = "model_b" if model == self.config.model_b else "model_a"
        self.results[key].append({
            "request_id": request_id,
            "latency": latency,
            "success": success,
            "user_rating": user_rating,
            "timestamp": time.time(),
        })

    def compute_significance(self):
        """Compute statistical significance of A/B test results."""
        import numpy as np
        from scipy import stats

        a_success = [r["success"] for r in self.results["model_a"]]
        b_success = [r["success"] for r in self.results["model_b"]]

        n_a, n_b = len(a_success), len(b_success)
        p_a = sum(a_success) / n_a
        p_b = sum(b_success) / n_b
        p_pool = (sum(a_success) + sum(b_success)) / (n_a + n_b)

        se = (p_pool * (1 - p_pool) * (1/n_a + 1/n_b)) ** 0.5
        z = (p_b - p_a) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return {
            "model_a_rate": p_a, "model_b_rate": p_b,
            "lift": (p_b - p_a) / p_a if p_a > 0 else 0,
            "z_score": z, "p_value": p_value,
            "significant": p_value < 0.05,
            "n_a": n_a, "n_b": n_b,
        }

config = ABTestConfig(
    experiment_name="llama3-finetune-v2",
    model_a="./base-model",
    model_b="./finetuned-model",
    traffic_split=0.2,
)
router = ABTestRouter(config)`}
        id="code-ab-router"
      />

      <PythonCode
        title="ab_test_analysis.py"
        code={`import numpy as np

def analyze_ab_test(results_a, results_b, metric="success"):
    """Analyze A/B test results and print summary."""
    vals_a = [r[metric] for r in results_a if metric in r]
    vals_b = [r[metric] for r in results_b if metric in r]

    print(f"{'Metric':<20} {'Model A':>10} {'Model B':>10}")
    print("-" * 42)
    print(f"{'N samples':<20} {len(vals_a):>10} {len(vals_b):>10}")
    print(f"{'Mean':<20} {np.mean(vals_a):>10.4f} {np.mean(vals_b):>10.4f}")
    print(f"{'Std':<20} {np.std(vals_a):>10.4f} {np.std(vals_b):>10.4f}")

    lat_a = [r["latency"] for r in results_a]
    lat_b = [r["latency"] for r in results_b]
    print(f"{'Latency p50':<20} {np.percentile(lat_a, 50):>10.3f}s"
          f" {np.percentile(lat_b, 50):>10.3f}s")
    print(f"{'Latency p95':<20} {np.percentile(lat_a, 95):>10.3f}s"
          f" {np.percentile(lat_b, 95):>10.3f}s")

def required_sample_size(baseline_rate, mde, alpha=0.05, power=0.8):
    """Minimum samples per group to detect minimum detectable effect."""
    from scipy.stats import norm
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    p1, p2 = baseline_rate, baseline_rate + mde
    n = ((z_a + z_b) ** 2 * (p1*(1-p1) + p2*(1-p2))) / mde**2
    return int(np.ceil(n))

print(f"Samples needed for 2% MDE: {required_sample_size(0.75, 0.02)}")`}
        id="code-ab-analysis"
      />

      <NoteBlock
        type="tip"
        title="Start with Low Traffic"
        content="Begin with 5-10% traffic on the new model and ramp up gradually. This limits blast radius if the fine-tuned model has unexpected failure modes. Only go to 50/50 after initial metrics look stable."
        id="note-low-traffic"
      />

      <WarningBlock
        title="Do Not Peek at Results"
        content="Checking results repeatedly and stopping the test early when you see a positive signal inflates false positive rates (the peeking problem). Pre-register your sample size and significance threshold before starting. Use sequential testing methods if you must monitor continuously."
        id="warning-peeking"
      />
    </div>
  )
}
