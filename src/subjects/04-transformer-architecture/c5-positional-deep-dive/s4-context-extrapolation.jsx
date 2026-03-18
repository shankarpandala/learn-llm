import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ContextExtrapolation() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Extending Context Length Beyond Training</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        A model trained on sequences of length L often fails catastrophically at length 2L.
        Context extrapolation techniques modify positional encodings at inference time (or with
        minimal fine-tuning) to extend the effective context window, enabling models trained on
        4K tokens to handle 128K or more.
      </p>

      <DefinitionBlock
        title="Context Length Extrapolation"
        definition="The ability of a model to process sequences longer than its training context length $L_{\text{train}}$ while maintaining quality. Key methods modify the positional encoding frequency or scale: position interpolation scales positions to fit within $[0, L_{\text{train}}]$, while NTK-aware methods adjust the frequency basis."
        id="def-extrapolation"
      />

      <h2 className="text-2xl font-semibold">Position Interpolation (PI)</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Instead of extrapolating beyond trained positions, PI linearly downscales all positions
        to fit within the original range. For a model trained on L=4096 used at L'=16384, position
        <InlineMath math="p" /> becomes <InlineMath math="p \cdot L / L'" />.
      </p>
      <BlockMath math="\text{PI: } \theta'_i = \theta_i \cdot \frac{L_{\text{train}}}{L_{\text{target}}}, \quad \text{NTK: } \theta'_i = \left(\frac{L_{\text{target}}}{L_{\text{train}}}\right)^{2i/(d-2)} \cdot \theta_i" />

      <ExampleBlock
        title="Methods for Extending from 4K to 32K"
        problem="Compare position interpolation, NTK scaling, and YaRN for 8x extension."
        steps={[
          { formula: '\\text{PI: scale all } \\theta_i \\text{ by } 1/8', explanation: 'All frequencies reduced uniformly. Needs ~1000 steps of fine-tuning.' },
          { formula: '\\text{NTK-aware: scale } \\theta_i \\text{ non-uniformly}', explanation: 'Low frequencies scaled more, high frequencies kept — preserves local resolution.' },
          { formula: '\\text{YaRN: NTK + temperature + attention scaling}', explanation: 'Combines NTK with attention logit scaling. Best zero-shot extrapolation.' },
          { formula: '\\text{Dynamic NTK: adjust } \\theta \\text{ per sequence length}', explanation: 'Scale only as much as needed — works without any fine-tuning.' },
        ]}
        id="example-methods"
      />

      <PythonCode
        title="context_extrapolation.py"
        code={`import torch
import math

def rope_freqs(dim, base=10000.0):
    """Standard RoPE frequencies."""
    return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

def position_interpolation(dim, scale_factor, base=10000.0):
    """Position Interpolation: uniformly scale all frequencies."""
    freqs = rope_freqs(dim, base)
    return freqs / scale_factor

def ntk_aware_scaling(dim, scale_factor, base=10000.0):
    """NTK-aware scaling: increase base frequency."""
    new_base = base * (scale_factor ** (dim / (dim - 2)))
    return rope_freqs(dim, new_base)

def yarn_scaling(dim, scale_factor, base=10000.0, beta_fast=32, beta_slow=1):
    """YaRN: interpolate between PI and NTK based on frequency."""
    freqs = rope_freqs(dim, base)
    # Low frequencies get PI (more interpolation)
    # High frequencies stay unchanged (NTK-like)
    low_freq_factor = 1.0
    high_freq_factor = scale_factor

    wavelengths = 2 * math.pi / freqs
    ramp = (wavelengths / (beta_fast * 2 * math.pi) - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    ramp = ramp.clamp(0, 1)

    # Smooth interpolation between scaled and unscaled
    scaled_freqs = freqs / scale_factor
    yarn_freqs = (1 - ramp) * scaled_freqs + ramp * freqs
    return yarn_freqs

# Compare frequency distributions
dim = 64
scale = 8  # Extend 4K -> 32K

original = rope_freqs(dim)
pi = position_interpolation(dim, scale)
ntk = ntk_aware_scaling(dim, scale)
yarn = yarn_scaling(dim, scale)

print(f"{'Dim pair':>10} | {'Original':>12} | {'PI (÷8)':>12} | {'NTK':>12} | {'YaRN':>12}")
print("-" * 65)
for i in range(0, dim // 2, 4):
    print(f"{i:>10} | {original[i]:>12.6f} | {pi[i]:>12.6f} | "
          f"{ntk[i]:>12.6f} | {yarn[i]:>12.6f}")

# Key insight: high-frequency dimensions (small i) handle local patterns
# and should NOT be scaled much. Low-frequency dimensions handle global
# patterns and can be safely compressed.
print("\\nScaling ratio (method / original):")
print(f"  PI dim 0:   {pi[0]/original[0]:.4f} (high freq, LOCAL — over-compressed!)")
print(f"  PI dim 15:  {pi[15]/original[15]:.4f} (low freq, GLOBAL — correct)")
print(f"  NTK dim 0:  {ntk[0]/original[0]:.4f} (high freq preserved)")
print(f"  NTK dim 15: {ntk[15]/original[15]:.4f} (low freq compressed)")
print(f"  YaRN dim 0: {yarn[0]/original[0]:.4f} (high freq mostly preserved)")
print(f"  YaRN dim 15:{yarn[15]/original[15]:.4f} (low freq compressed)")`}
        id="code-extrapolation"
      />

      <NoteBlock
        type="intuition"
        title="Why Uniform Scaling Hurts Local Patterns"
        content="Position Interpolation scales all frequencies equally, but high-frequency dimensions encode local patterns (nearby token relationships). Compressing these means the model can no longer distinguish between adjacent tokens as well. NTK-aware methods preserve high frequencies and only compress low frequencies, maintaining local resolution while extending global range."
        id="note-local-vs-global"
      />

      <WarningBlock
        title="Extrapolation Is Not Free"
        content="While these methods enable longer contexts, quality still degrades compared to training natively at the target length. A model trained on 4K and extrapolated to 32K will underperform a model trained natively on 32K, especially for tasks requiring precise attention over the full context (like needle-in-a-haystack retrieval). Long-context fine-tuning remains valuable."
        id="warning-quality-gap"
      />

      <NoteBlock
        type="note"
        title="The Context Length Arms Race"
        content="GPT-3 (2020): 2K tokens. GPT-3.5: 4K/16K. GPT-4: 8K/128K. Claude: 100K/200K. Gemini 1.5: 1M/2M tokens. This rapid expansion was enabled by FlashAttention (reducing memory), RoPE scaling (extending positions), and training innovations (progressive length training, long-context data curation)."
        id="note-arms-race"
      />

      <NoteBlock
        type="tip"
        title="Practical Recommendations"
        content="For extending an existing RoPE model: (1) try Dynamic NTK scaling first (no fine-tuning needed), (2) if quality is insufficient, use YaRN with 200-1000 steps of fine-tuning on long-context data, (3) for maximum quality, progressively train on increasing lengths. Always evaluate with needle-in-a-haystack tests across the full context window."
        id="note-practical"
      />
    </div>
  )
}
