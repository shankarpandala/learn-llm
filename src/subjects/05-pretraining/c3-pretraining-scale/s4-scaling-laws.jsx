import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function ScalingLaws() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Scaling Laws and Compute-Optimal Training</h1>
      <p className="text-lg text-gray-300">
        Scaling laws describe how model performance (loss) improves predictably with increases
        in model size, dataset size, and compute budget. These power-law relationships guide
        fundamental decisions about how to allocate resources during pretraining.
      </p>

      <DefinitionBlock
        title="Neural Scaling Laws (Kaplan et al., 2020)"
        definition="For Transformer language models, the cross-entropy loss follows power laws: $L(N) = (N_c / N)^{\alpha_N}$, $L(D) = (D_c / D)^{\alpha_D}$, $L(C) = (C_c / C)^{\alpha_C}$, where $N$ is parameters, $D$ is data tokens, $C$ is compute (FLOPs). Kaplan found $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, $\alpha_C \approx 0.050$."
        notation="The combined scaling law is $L(N, D) = \left[(N_c/N)^{\alpha_N / \alpha_D} + D_c/D\right]^{\alpha_D}$ capturing diminishing returns when either $N$ or $D$ is held fixed."
        id="scaling-laws-def"
      />

      <TheoremBlock
        title="Chinchilla Scaling Law (Hoffmann et al., 2022)"
        statement="For a compute budget $C \approx 6ND$ FLOPs, the compute-optimal allocation is $N^* \propto C^{0.5}$ and $D^* \propto C^{0.5}$. Equivalently, the optimal token-to-parameter ratio is approximately $D^*/N^* \approx 20$. A model with $N$ parameters should be trained on roughly $20N$ tokens."
        proof="Minimize $L(N, D)$ subject to $C = 6ND$. Using the parametric loss $L(N, D) = A/N^\alpha + B/D^\beta + L_\infty$ with fitted $\alpha \approx 0.34$, $\beta \approx 0.28$, the Lagrangian yields $N^* \propto C^{a/(a+b)}$ and $D^* \propto C^{b/(a+b)}$ where $a = \alpha/(\alpha+1)$, $b = \beta/(\beta+1)$."
        corollaries={[
          'GPT-3 (175B params, 300B tokens) was undertrained by Chinchilla standards. Optimal: ~3.5T tokens.',
          'Chinchilla (70B params, 1.4T tokens) matched GPT-3 performance with 4x less compute at inference.',
          'LLaMA-1 (65B params, 1.4T tokens) and LLaMA-2 (70B, 2T tokens) followed this principle.'
        ]}
        id="chinchilla-thm"
      />

      <ExampleBlock
        title="Compute-Optimal Model Sizing"
        problem="Given a compute budget of 10^22 FLOPs, what are the optimal model and data sizes?"
        steps={[
          {
            formula: 'C = 6ND \\implies 10^{22} = 6 \\cdot N \\cdot D',
            explanation: 'The approximate compute formula for a forward+backward pass.'
          },
          {
            formula: 'D^* \\approx 20 N^* \\implies 10^{22} = 6 \\cdot N^* \\cdot 20 N^* = 120 (N^*)^2',
            explanation: 'Apply the Chinchilla ratio D/N = 20.'
          },
          {
            formula: 'N^* = \\sqrt{10^{22}/120} \\approx 2.9 \\times 10^9 \\approx 2.9\\text{B params}',
            explanation: 'Optimal model has roughly 3 billion parameters.'
          },
          {
            formula: 'D^* = 20 \\times 2.9 \\times 10^9 \\approx 58\\text{B tokens}',
            explanation: 'Train on approximately 58 billion tokens for compute optimality.'
          }
        ]}
        id="compute-optimal-example"
      />

      <NoteBlock
        type="note"
        title="Beyond Chinchilla: Inference-Optimal Scaling"
        content="Chinchilla optimizes for training compute only. In practice, a smaller model trained for longer (on more data) has lower inference cost per query. LLaMA-3 8B was trained on 15T tokens (1875x parameters) -- far beyond Chinchilla-optimal -- because inference efficiency matters more in deployment. The optimal trade-off depends on total lifetime inference compute vs. one-time training cost."
        id="inference-optimal-note"
      />

      <PythonCode
        title="scaling_laws.py"
        code={`import numpy as np

# Chinchilla scaling law parameters (Hoffmann et al., 2022)
A = 406.4       # Parameter scaling coefficient
B = 410.7       # Data scaling coefficient
alpha = 0.34    # Parameter scaling exponent
beta = 0.28     # Data scaling exponent
L_inf = 1.69    # Irreducible loss

def chinchilla_loss(N, D):
    """Compute expected loss given N parameters and D tokens."""
    return A / N**alpha + B / D**beta + L_inf

def compute_flops(N, D):
    """Approximate training FLOPs: C ≈ 6ND."""
    return 6 * N * D

def optimal_allocation(C):
    """Find compute-optimal N, D for budget C FLOPs."""
    # Chinchilla ratio: D ≈ 20N, so C = 6*N*20N = 120*N^2
    N_opt = np.sqrt(C / 120)
    D_opt = 20 * N_opt
    return N_opt, D_opt

# Table of compute-optimal models
print("Compute-Optimal Model Configurations:")
print(f"{'Budget (FLOPs)':>18s} {'Params':>12s} {'Tokens':>12s} {'Loss':>8s}")
print("-" * 54)

for exp in range(19, 26):
    C = 10 ** exp
    N, D = optimal_allocation(C)
    L = chinchilla_loss(N, D)
    def fmt(x):
        if x >= 1e12: return f"{x/1e12:.1f}T"
        if x >= 1e9:  return f"{x/1e9:.1f}B"
        if x >= 1e6:  return f"{x/1e6:.0f}M"
        return f"{x:.0f}"
    print(f"{C:>18.0e} {fmt(N):>12s} {fmt(D):>12s} {L:>8.3f}")

# Compare: training a model with different token ratios
N_fixed = 7e9  # 7B parameter model
print(f"\\n7B model at different token budgets:")
print(f"{'Tokens':>12s} {'Ratio D/N':>10s} {'Loss':>8s} {'FLOPs':>14s}")
for ratio in [5, 10, 20, 50, 100, 200]:
    D = ratio * N_fixed
    L = chinchilla_loss(N_fixed, D)
    C = compute_flops(N_fixed, D)
    print(f"{D/1e9:>10.0f}B {ratio:>10d} {L:>8.3f} {C:>14.2e}")

# Kaplan vs Chinchilla comparison
print("\\nKaplan vs Chinchilla optimal allocation for C=10^23:")
C = 1e23
# Kaplan: N scales faster (N ∝ C^0.73)
N_kaplan = 1.3e10 * (C / 1e23) ** 0.73
D_kaplan = C / (6 * N_kaplan)
# Chinchilla
N_chin, D_chin = optimal_allocation(C)
print(f"  Kaplan:     N={N_kaplan/1e9:.1f}B, D={D_kaplan/1e9:.0f}B (ratio={D_kaplan/N_kaplan:.0f})")
print(f"  Chinchilla: N={N_chin/1e9:.1f}B, D={D_chin/1e9:.0f}B (ratio={D_chin/N_chin:.0f})")`}
        id="scaling-laws-code"
      />

      <WarningBlock
        title="Scaling Laws Have Limitations"
        content="Scaling laws are empirical fits that assume: (1) the architecture stays the same, (2) data quality is constant, (3) hyperparameters are well-tuned. They may not hold for architectural innovations, mixture-of-experts models, or when data quality changes. They also do not predict emergent capabilities -- abilities that appear suddenly at certain scales and are not captured by smooth power laws."
        id="scaling-limitations-warning"
      />
    </div>
  )
}
