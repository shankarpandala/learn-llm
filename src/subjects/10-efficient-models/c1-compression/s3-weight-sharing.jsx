import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function WeightSharing() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Weight Sharing and Weight Tying</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Weight sharing reduces model size by reusing the same parameters across different
        parts of the network. The most common form in LLMs is embedding-output weight tying,
        where the input embedding matrix and the output projection share the same parameters,
        saving hundreds of millions of parameters in large-vocabulary models.
      </p>

      <DefinitionBlock
        title="Weight Tying"
        definition="Weight tying constrains two or more parameter matrices to be identical. In language models, the input embedding $E \in \mathbb{R}^{V \times d}$ and the output projection $W_o \in \mathbb{R}^{d \times V}$ are tied: $W_o = E^\top$. The output logits become $z = h \cdot E^\top$ where $h$ is the final hidden state."
        notation="Memory saved $= V \times d$ parameters. For $V = 50{,}000$ and $d = 4{,}096$: savings = $204{,}800{,}000$ parameters ($\approx 390$ MB at FP16)."
        id="def-weight-tying"
      />

      <ExampleBlock
        title="Embedding-Output Tying Savings"
        problem="Calculate parameter savings from weight tying in a model with vocabulary V=32,000 and hidden dimension d=4,096."
        steps={[
          {
            formula: '\\text{Embedding params} = V \\times d = 32{,}000 \\times 4{,}096 = 131{,}072{,}000',
            explanation: 'Input embedding matrix size.'
          },
          {
            formula: '\\text{Without tying: } 2 \\times 131{,}072{,}000 = 262{,}144{,}000',
            explanation: 'Separate embedding and output projection would use 262M parameters.'
          },
          {
            formula: '\\text{With tying: } 131{,}072{,}000 \\text{ (shared)}',
            explanation: 'Weight tying halves this to 131M parameters.'
          },
          {
            formula: '\\text{Savings} = 131M \\text{ params} = 250\\text{ MB at FP16}',
            explanation: 'For a 7B model, this is roughly 1.9% of total parameters — small but free.'
          }
        ]}
        id="example-weight-tying"
      />

      <PythonCode
        title="weight_sharing.py"
        code={`import torch
import torch.nn as nn

class TiedEmbeddingLM(nn.Module):
    """Language model with tied input/output embeddings."""
    def __init__(self, vocab_size, d_model, n_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        # Output projection — weight tied with embedding
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight  # Tie weights!

    def forward(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        h = self.ln(h)
        return self.output_proj(h)  # Uses embedding weights transposed

# Compare parameter counts
V, d = 32_000, 4_096

model_tied = TiedEmbeddingLM(V, d, n_layers=2)
model_untied = TiedEmbeddingLM(V, d, n_layers=2)
# Break the tie for comparison
model_untied.output_proj.weight = nn.Parameter(torch.randn(V, d))

tied_params = sum(p.numel() for p in model_tied.parameters())
untied_params = sum(p.numel() for p in model_untied.parameters())

print(f"Tied model:   {tied_params:>12,} params")
print(f"Untied model: {untied_params:>12,} params")
print(f"Savings:      {untied_params - tied_params:>12,} params")
print(f"Savings:      {(untied_params - tied_params) * 2 / 1e6:.0f} MB (FP16)")

# Cross-layer weight sharing (ALBERT-style)
class ALBERTStyleEncoder(nn.Module):
    """All transformer layers share the same parameters."""
    def __init__(self, d_model, n_virtual_layers=12):
        super().__init__()
        # Only ONE physical layer, applied n times
        self.shared_layer = nn.TransformerEncoderLayer(
            d_model, nhead=8, batch_first=True
        )
        self.n_layers = n_virtual_layers

    def forward(self, x):
        for _ in range(self.n_layers):
            x = self.shared_layer(x)
        return x

shared = ALBERTStyleEncoder(768, n_virtual_layers=12)
unshared_count = 12 * sum(p.numel() for p in shared.shared_layer.parameters())
shared_count = sum(p.numel() for p in shared.parameters())
print(f"\\nALBERT-style sharing: {shared_count:,} vs {unshared_count:,}")
print(f"Reduction: {unshared_count / shared_count:.0f}x fewer params")`}
        id="code-weight-sharing"
      />

      <NoteBlock
        type="historical"
        title="Weight Tying in Major Models"
        content="Press & Wolf (2017) showed embedding-output tying improves perplexity while reducing parameters. GPT-2, T5, and LLaMA all use this technique. ALBERT (Lan et al., 2019) went further with cross-layer parameter sharing, achieving 18x fewer parameters than BERT-Large with competitive accuracy."
        id="note-weight-tying-history"
      />

      <NoteBlock
        type="note"
        title="Beyond Simple Sharing"
        content="Modern approaches include factorized embeddings (ALBERT decomposes V x d into V x e and e x d with e << d), grouped weight sharing across attention heads, and Universal Transformers that share weights across depth while using adaptive computation time."
        id="note-advanced-sharing"
      />

      <WarningBlock
        title="Cross-Layer Sharing Limitations"
        content="While ALBERT-style full cross-layer sharing dramatically reduces parameters, it does not reduce computation (FLOPs) since every layer still runs a full forward pass. It also tends to underperform independent layers when the parameter budget is not the bottleneck."
        id="warning-cross-layer"
      />
    </div>
  )
}
