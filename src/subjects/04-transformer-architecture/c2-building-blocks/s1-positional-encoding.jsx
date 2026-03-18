import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function PositionalEncoding() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Positional Encoding: Sinusoidal Embeddings</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Self-attention is inherently permutation-invariant — it has no notion of token order.
        Positional encodings inject sequence position information into the model, enabling it to
        distinguish "the cat sat on the mat" from "the mat sat on the cat."
      </p>

      <DefinitionBlock
        title="Sinusoidal Positional Encoding"
        definition="For position $\text{pos}$ and dimension $i$: $PE_{(\text{pos}, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$, $PE_{(\text{pos}, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$"
        notation="pos = token position (0-indexed), i = dimension index, d_model = embedding dimension"
        id="def-sinusoidal-pe"
      />

      <NoteBlock
        type="intuition"
        title="Why Sinusoids?"
        content="Each dimension oscillates at a different frequency, from high-frequency (small i) to low-frequency (large i). This creates a unique 'fingerprint' for each position. Additionally, the encoding for position pos+k can be expressed as a linear function of the encoding at pos, allowing the model to learn relative position attention."
        id="note-why-sinusoids"
      />

      <h2 className="text-2xl font-semibold">The Encoding Formula</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The wavelengths form a geometric progression from <InlineMath math="2\pi" /> to{' '}
        <InlineMath math="10000 \cdot 2\pi" />. This means lower dimensions capture fine-grained
        position differences while higher dimensions capture broader positional patterns.
      </p>
      <BlockMath math="PE_{(\text{pos}, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right), \quad PE_{(\text{pos}, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)" />

      <ExampleBlock
        title="Computing PE for Position 3 with d_model=8"
        problem="Calculate the positional encoding vector for position 3 in an 8-dimensional model."
        steps={[
          { formula: 'PE_{(3,0)} = \\sin(3 / 10000^{0/8}) = \\sin(3) \\approx 0.141', explanation: 'Dimension 0 (sin): frequency = 1, highest frequency.' },
          { formula: 'PE_{(3,1)} = \\cos(3 / 10000^{0/8}) = \\cos(3) \\approx -0.990', explanation: 'Dimension 1 (cos): same frequency as dimension 0.' },
          { formula: 'PE_{(3,2)} = \\sin(3 / 10000^{2/8}) = \\sin(0.300) \\approx 0.296', explanation: 'Dimension 2 (sin): lower frequency, wavelength ≈ 10.' },
          { formula: 'PE_{(3,6)} = \\sin(3 / 10000^{6/8}) \\approx \\sin(0.003) \\approx 0.003', explanation: 'Dimension 6 (sin): very low frequency, nearly flat.' },
        ]}
        id="example-pe-calc"
      />

      <PythonCode
        title="positional_encoding.py"
        code={`import torch
import math

def sinusoidal_positional_encoding(max_len, d_model):
    """Generate sinusoidal positional encodings."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (max_len, d_model)

# Generate encodings
pe = sinusoidal_positional_encoding(max_len=100, d_model=64)
print(f"PE shape: {pe.shape}")  # [100, 64]

# Key property: dot product between positions captures relative distance
dot_products = torch.matmul(pe, pe.T)
print(f"PE[0] · PE[1] = {dot_products[0, 1]:.3f}")
print(f"PE[0] · PE[50] = {dot_products[0, 50]:.3f}")
print(f"PE[0] · PE[99] = {dot_products[0, 99]:.3f}")

# In a transformer, PE is added to token embeddings
class TokenWithPosition(torch.nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, d_model)
        self.register_buffer('pe', sinusoidal_positional_encoding(max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        return self.tok_emb(x) + self.pe[:seq_len]

model = TokenWithPosition(vocab_size=1000, d_model=64)
tokens = torch.randint(0, 1000, (2, 10))
embedded = model(tokens)
print(f"Embedded shape: {embedded.shape}")  # [2, 10, 64]`}
        id="code-pe"
      />

      <WarningBlock
        title="Learned vs. Sinusoidal Encodings"
        content="Many modern models (GPT-2, BERT) use learned positional embeddings instead of sinusoidal ones. Learned embeddings are more flexible but cannot extrapolate to unseen positions beyond max_len. Sinusoidal encodings theoretically generalize but in practice also degrade for positions far beyond training lengths."
        id="warning-learned-vs-fixed"
      />

      <NoteBlock
        type="historical"
        title="From Fixed to Learned to Relative"
        content="The original transformer (2017) used sinusoidal encodings. GPT-2 and BERT switched to learned absolute positions. Modern models increasingly use relative position methods (RoPE, ALiBi) that encode position differences rather than absolute positions, enabling better length generalization."
        id="note-evolution"
      />
    </div>
  )
}
