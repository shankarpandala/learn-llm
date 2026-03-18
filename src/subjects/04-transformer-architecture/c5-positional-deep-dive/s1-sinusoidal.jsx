import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

export default function SinusoidalEncoding() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Sinusoidal Positional Encoding: The Math</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The sinusoidal positional encoding from the original transformer encodes each position
        as a unique vector using sine and cosine functions at geometrically spaced frequencies.
        This design has elegant mathematical properties that enable the model to learn relative
        position attention.
      </p>

      <DefinitionBlock
        title="Sinusoidal Encoding Formulas"
        definition="$PE_{(\text{pos}, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right), \quad PE_{(\text{pos}, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)$ for dimension pairs $i = 0, 1, \ldots, d/2 - 1$."
        notation="Each dimension pair (2i, 2i+1) oscillates at wavelength λ_i = 2π · 10000^{2i/d}"
        id="def-sinusoidal-formulas"
      />

      <TheoremBlock
        title="Relative Position via Linear Transformation"
        statement="For any fixed offset $k$, there exists a linear transformation $M_k$ (independent of position) such that $PE_{\text{pos}+k} = M_k \cdot PE_{\text{pos}}$. Specifically, for each dimension pair, $M_k$ is a 2D rotation matrix with angle $k \cdot \omega_i$."
        proofSteps={[
          '\\text{Let } \\omega_i = 1/10000^{2i/d}. \\text{ Then for dimension pair } (2i, 2i+1):',
          'PE_{\\text{pos}}^{(2i, 2i+1)} = (\\sin(\\omega_i \\cdot \\text{pos}), \\cos(\\omega_i \\cdot \\text{pos}))',
          'PE_{\\text{pos}+k}^{(2i, 2i+1)} = (\\sin(\\omega_i(\\text{pos}+k)), \\cos(\\omega_i(\\text{pos}+k)))',
          '\\text{By angle addition: } \\begin{pmatrix} \\sin(\\alpha+\\beta) \\\\ \\cos(\\alpha+\\beta) \\end{pmatrix} = \\begin{pmatrix} \\cos\\beta & \\sin\\beta \\\\ -\\sin\\beta & \\cos\\beta \\end{pmatrix} \\begin{pmatrix} \\sin\\alpha \\\\ \\cos\\alpha \\end{pmatrix}',
          '\\text{So } M_k \\text{ is block-diagonal with 2x2 rotation matrices } R(k\\omega_i) \\text{ for each pair.}',
        ]}
        id="thm-relative-position"
      />

      <h2 className="text-2xl font-semibold">Frequency Spectrum</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The wavelengths span from <InlineMath math="2\pi \approx 6.28" /> (dimension 0) to{' '}
        <InlineMath math="2\pi \times 10000 \approx 62{,}832" /> (dimension d-1). Lower
        dimensions change rapidly with position (fine-grained), while higher dimensions change
        slowly (coarse-grained), similar to a binary counter or Fourier features.
      </p>

      <ExampleBlock
        title="Wavelength Distribution for d_model=512"
        problem="Calculate the wavelength for the first, middle, and last dimension pairs."
        steps={[
          { formula: '\\lambda_0 = 2\\pi \\cdot 10000^{0/512} = 2\\pi \\approx 6.28', explanation: 'First pair: completes a full cycle every ~6 positions.' },
          { formula: '\\lambda_{128} = 2\\pi \\cdot 10000^{256/512} = 2\\pi \\cdot 100 \\approx 628', explanation: 'Middle pair: one cycle per ~628 positions.' },
          { formula: '\\lambda_{255} = 2\\pi \\cdot 10000^{510/512} \\approx 62{,}208', explanation: 'Last pair: barely changes over typical sequence lengths.' },
        ]}
        id="example-wavelengths"
      />

      <PythonCode
        title="sinusoidal_deep_dive.py"
        code={`import torch
import math

def sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

pe = sinusoidal_pe(1024, 512)

# Property 1: Each position has a unique encoding
norms = torch.cdist(pe[:100], pe[:100])
print("Min distance between different positions:",
      norms[norms > 0].min().item())  # > 0

# Property 2: Relative position dot product
# PE[pos] · PE[pos+k] depends mainly on k, not pos
def dot_product_at_offset(pe, offset, num_samples=50):
    dots = []
    for p in range(num_samples):
        if p + offset < pe.size(0):
            dots.append(torch.dot(pe[p], pe[p + offset]).item())
    return torch.tensor(dots)

for k in [1, 5, 10, 50]:
    dots = dot_product_at_offset(pe, k)
    print(f"Offset k={k:>2}: mean dot={dots.mean():.2f}, "
          f"std={dots.std():.4f}")

# Property 3: The linear transformation M_k
def rotation_matrix_2d(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, s], [-s, c]])

# Verify: PE[pos+k] = M_k @ PE[pos] for each dimension pair
pos, k = 7, 3
d_model = 512
for i in range(3):  # Check first 3 dimension pairs
    omega_i = 1.0 / (10000 ** (2 * i / d_model))
    M_k = rotation_matrix_2d(k * omega_i)
    pe_pos = pe[pos, 2*i:2*i+2]
    pe_pos_k = pe[pos + k, 2*i:2*i+2]
    predicted = M_k @ pe_pos
    error = (pe_pos_k - predicted).abs().max().item()
    print(f"Dim pair {i}: prediction error = {error:.2e}")  # ~1e-7`}
        id="code-sinusoidal"
      />

      <NoteBlock
        type="tip"
        title="Visualizing Positional Encodings"
        content="Plot PE as a heatmap with position on the y-axis and dimension on the x-axis. You will see alternating vertical stripes (low dimensions change fast) transitioning to nearly solid bands (high dimensions change slowly). This pattern resembles a binary counter viewed in analog form."
        id="note-visualization"
      />

      <WarningBlock
        title="Sinusoidal Encodings Do Not Extrapolate Well"
        content="Despite the elegant theory that M_k exists for any k, models trained with sinusoidal encodings at max_len=512 perform poorly at position 1000. The issue is not the encoding itself but that the model's attention weights were never trained to handle those position patterns. This limitation motivated learned and relative position methods."
        id="warning-extrapolation"
      />
    </div>
  )
}
