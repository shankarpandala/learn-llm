import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function VanishingGradients() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Vanishing and Exploding Gradients</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The vanishing and exploding gradient problem is the central obstacle to training deep
        recurrent networks. Understanding why gradients decay or blow up through time is
        essential for appreciating why LSTM and GRU architectures were invented, and why
        gradient clipping became standard practice.
      </p>

      <DefinitionBlock
        title="Vanishing Gradient Problem"
        definition="When training an RNN via backpropagation through time (BPTT), the gradient of the loss with respect to early hidden states involves a product of Jacobians $\prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}$. If the spectral radius of $W_{hh}$ is less than 1, this product shrinks exponentially, making it impossible to learn long-range dependencies."
        notation="$\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{k=t+1}^{T} \text{diag}(\sigma'(z_k)) W_{hh}$"
        id="def-vanishing-gradient"
      />

      <h2 className="text-2xl font-semibold">Gradient Flow Through Time</h2>
      <p className="text-gray-700 dark:text-gray-300">
        For a vanilla RNN with <InlineMath math="h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)" />,
        the Jacobian of <InlineMath math="h_t" /> with respect to <InlineMath math="h_{t-1}" /> is:
      </p>
      <BlockMath math="\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1 - h_t^2) \cdot W_{hh}" />
      <p className="text-gray-700 dark:text-gray-300">
        Since <InlineMath math="\tanh'(x) = 1 - \tanh^2(x) \leq 1" />, and the diagonal entries
        are often much less than 1 for saturated units, repeated multiplication causes
        the gradient to shrink exponentially with the number of time steps.
      </p>

      <TheoremBlock
        title="Gradient Bound for Vanilla RNNs"
        statement="For a vanilla RNN, if $\|W_{hh}\| < 1 / \gamma$ where $\gamma = \max_t \|\text{diag}(1 - h_t^2)\|$, then $\left\|\frac{\partial h_T}{\partial h_t}\right\| \leq (\gamma \|W_{hh}\|)^{T-t} \to 0$ as $T - t \to \infty$."
        corollaries={['Long-range gradients vanish exponentially, making it impossible to credit-assign across many time steps.', 'Conversely, if $\\gamma \\|W_{hh}\\| > 1$, gradients explode exponentially.']}
        id="thm-gradient-bound"
      />

      <ExampleBlock
        title="Sigmoid Saturation"
        problem="Show why sigmoid and tanh activations cause gradient vanishing."
        steps={[
          { formula: "\\sigma'(x) = \\sigma(x)(1 - \\sigma(x)) \\leq 0.25", explanation: 'The sigmoid derivative peaks at 0.25 when x=0 and approaches 0 at the extremes.' },
          { formula: "\\tanh'(x) = 1 - \\tanh^2(x) \\leq 1", explanation: 'The tanh derivative equals 1 only at x=0 and decays rapidly for |x| > 2.' },
          { formula: '\\text{After } k \\text{ steps: } (0.25)^k \\text{ for sigmoid}', explanation: 'With sigmoid, after just 10 steps the gradient is attenuated by a factor of ~10^{-6}.' },
          { formula: '\\text{After 20 steps: } \\approx 10^{-12}', explanation: 'Gradients become numerically indistinguishable from zero in float32.' },
        ]}
        id="example-saturation"
      />

      <PythonCode
        title="visualize_gradient_flow.py"
        code={`import torch
import torch.nn as nn

def measure_gradient_flow(seq_len=50, hidden_size=128):
    """Measure how gradients decay through an RNN."""
    rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)

    # Random input sequence
    x = torch.randn(1, seq_len, hidden_size, requires_grad=True)
    h0 = torch.zeros(1, 1, hidden_size)

    output, _ = rnn(x, h0)

    # Compute gradient of final output w.r.t. input at each step
    loss = output[0, -1].sum()  # scalar from last time step
    loss.backward()

    # Gradient magnitude at each time step
    grad_norms = x.grad[0].norm(dim=1).detach()
    print(f"Gradient at step 0 (earliest): {grad_norms[0]:.6f}")
    print(f"Gradient at step {seq_len//2}: {grad_norms[seq_len//2]:.6f}")
    print(f"Gradient at step {seq_len-1} (latest): {grad_norms[-1]:.6f}")
    print(f"Ratio (first/last): {grad_norms[0]/grad_norms[-1]:.6f}")

measure_gradient_flow(seq_len=50)
# Typical output: gradient at step 0 is orders of magnitude
# smaller than at step 49, confirming vanishing gradients.`}
        id="code-grad-flow"
      />

      <PythonCode
        title="exploding_gradient_demo.py"
        code={`import torch
import torch.nn as nn

# Demonstrate exploding gradients with large weights
rnn = nn.RNN(64, 64, batch_first=True)
# Artificially scale recurrent weights to cause explosion
with torch.no_grad():
    rnn.weight_hh_l0.mul_(2.0)

x = torch.randn(1, 100, 64, requires_grad=True)
output, _ = rnn(x)
loss = output[0, -1].sum()

try:
    loss.backward()
    grad_norm = x.grad.norm().item()
    print(f"Gradient norm: {grad_norm}")
    if grad_norm > 1e6:
        print("Gradients exploded!")
except RuntimeError as e:
    print(f"Numerical error: {e}")

# Solution: gradient clipping
torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
print("After clipping, training can proceed stably.")`}
        id="code-exploding"
      />

      <NoteBlock
        type="intuition"
        title="Why This Matters for Language"
        content="Consider the sentence: 'The cat, which sat on the mat and watched the birds outside the window for hours, was hungry.' The RNN needs to connect 'cat' to 'was hungry' across ~15 tokens. Vanishing gradients mean the model cannot learn that 'cat' determines the verb form. This is why vanilla RNNs fail at subject-verb agreement over long distances."
        id="note-language-impact"
      />

      <WarningBlock
        title="Gradient Clipping Is Not a Full Solution"
        content="While gradient clipping prevents exploding gradients, it does nothing for vanishing gradients. Clipping rescales large gradients but cannot amplify small ones. Architectural changes like LSTM gates are needed to create gradient highways that allow information to flow unimpeded across many time steps."
        id="warning-clipping-limits"
      />
    </div>
  )
}
