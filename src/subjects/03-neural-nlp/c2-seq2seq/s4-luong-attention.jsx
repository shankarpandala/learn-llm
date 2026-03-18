import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LuongAttention() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Luong Attention Variants</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Luong et al. (2015) proposed simplified attention mechanisms that compute alignment
        scores using the current decoder hidden state (rather than the previous one as in
        Bahdanau). They introduced three scoring functions -- dot, general, and concat -- and
        compared global vs. local attention. The dot-product scoring became the basis for
        transformer attention.
      </p>

      <DefinitionBlock
        title="Luong Attention Scoring Functions"
        definition="Given encoder hidden state $h_j$ and decoder hidden state $s_t$, Luong defines three scoring functions: (1) Dot: $\text{score}(s_t, h_j) = s_t^T h_j$, (2) General: $\text{score}(s_t, h_j) = s_t^T W_a h_j$, (3) Concat: $\text{score}(s_t, h_j) = v_a^T \tanh(W_a [s_t; h_j])$."
        notation="In all cases, $\alpha_{t,j} = \text{softmax}_j(\text{score}(s_t, h_j))$ and $c_t = \sum_j \alpha_{t,j} h_j$."
        id="def-luong-attention"
      />

      <h2 className="text-2xl font-semibold">Scoring Functions Compared</h2>
      <BlockMath math="\text{Dot: } e_{t,j} = s_t^T h_j" />
      <BlockMath math="\text{General: } e_{t,j} = s_t^T W_a h_j" />
      <BlockMath math="\text{Concat: } e_{t,j} = v_a^T \tanh(W_a [s_t; h_j])" />
      <p className="text-gray-700 dark:text-gray-300">
        The dot product is the simplest and fastest, requiring no learnable parameters in the
        scoring function itself. However, it requires encoder and decoder to have the same
        hidden dimension. The general scoring adds a learnable matrix
        <InlineMath math="W_a \in \mathbb{R}^{d \times d}" /> that can handle different dimensions
        and learn a task-specific similarity metric.
      </p>

      <ExampleBlock
        title="Computational Comparison"
        problem="Compare FLOPs for the three scoring functions with hidden_dim=512 and src_len=30."
        steps={[
          { formula: '\\text{Dot: } 30 \\times 512 = 15{,}360 \\text{ multiplies}', explanation: 'One dot product per source position. No parameters to learn.' },
          { formula: '\\text{General: } 512^2 + 30 \\times 512 = 277{,}504', explanation: 'Matrix-vector product W_a * h_j (can be precomputed) plus dot products.' },
          { formula: '\\text{Concat: } 30 \\times (1024 \\times d_a + d_a)', explanation: 'Most expensive: concatenation, linear projection, tanh, and dot with v for each position.' },
          { formula: '\\text{Dot} \\ll \\text{General} < \\text{Concat}', explanation: 'Dot product is orders of magnitude faster, which is why transformers use scaled dot-product attention.' },
        ]}
        id="example-scoring-flops"
      />

      <h2 className="text-2xl font-semibold">Global vs. Local Attention</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Global attention attends to all source positions (like Bahdanau). Local attention
        predicts an alignment position <InlineMath math="p_t" /> and only attends to a
        window <InlineMath math="[p_t - D, p_t + D]" /> around it. This reduces computation
        from <InlineMath math="O(S)" /> to <InlineMath math="O(D)" /> per decoder step.
      </p>
      <BlockMath math="p_t = S \cdot \sigma(v_p^T \tanh(W_p s_t))" />
      <BlockMath math="\alpha_{t,j} = \text{align}(t, j) \cdot \exp\left(-\frac{(j - p_t)^2}{2\sigma^2}\right)" />

      <PythonCode
        title="luong_attention_variants.py"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongAttention(nn.Module):
    """Luong attention with dot, general, and concat scoring."""
    def __init__(self, enc_dim, dec_dim, method='dot'):
        super().__init__()
        self.method = method
        if method == 'general':
            self.W = nn.Linear(enc_dim, dec_dim, bias=False)
        elif method == 'concat':
            self.W = nn.Linear(enc_dim + dec_dim, dec_dim, bias=False)
            self.v = nn.Linear(dec_dim, 1, bias=False)

    def score(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, dec_dim)
        # encoder_outputs: (batch, src_len, enc_dim)
        if self.method == 'dot':
            # (batch, src_len)
            return torch.bmm(
                encoder_outputs,
                decoder_hidden.unsqueeze(2)
            ).squeeze(2)

        elif self.method == 'general':
            # W transforms encoder outputs, then dot with decoder
            energy = self.W(encoder_outputs)  # (batch, src_len, dec_dim)
            return torch.bmm(
                energy, decoder_hidden.unsqueeze(2)
            ).squeeze(2)

        elif self.method == 'concat':
            src_len = encoder_outputs.size(1)
            dec_expanded = decoder_hidden.unsqueeze(1).expand(
                -1, src_len, -1
            )
            concat = torch.cat([dec_expanded, encoder_outputs], dim=2)
            energy = torch.tanh(self.W(concat))
            return self.v(energy).squeeze(2)

    def forward(self, decoder_hidden, encoder_outputs):
        scores = self.score(decoder_hidden, encoder_outputs)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        return context.squeeze(1), weights

# Compare all three variants
for method in ['dot', 'general', 'concat']:
    attn = LuongAttention(512, 512, method=method)
    enc_out = torch.randn(4, 30, 512)
    dec_h = torch.randn(4, 512)
    ctx, w = attn(dec_h, enc_out)
    n_params = sum(p.numel() for p in attn.parameters())
    print(f"{method:>8}: context={ctx.shape}, params={n_params:,}")
# Output:
#      dot: context=torch.Size([4, 512]), params=0
#  general: context=torch.Size([4, 512]), params=262,144
#   concat: context=torch.Size([4, 512]), params=525,312`}
        id="code-luong-variants"
      />

      <PythonCode
        title="luong_decoder_integration.py"
        code={`import torch
import torch.nn as nn

class LuongDecoder(nn.Module):
    """Decoder using Luong attention (compute attention AFTER RNN step)."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, method='general'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.attention = LuongAttention(hidden_dim, hidden_dim, method)
        # Attentional hidden state: concat hidden + context
        self.W_c = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_token, decoder_hidden, encoder_outputs):
        embedded = self.embedding(tgt_token).unsqueeze(1)
        # Step 1: RNN step (key difference from Bahdanau)
        rnn_output, hidden = self.rnn(embedded, decoder_hidden.unsqueeze(0))
        hidden = hidden.squeeze(0)

        # Step 2: Compute attention using CURRENT hidden state
        context, attn_weights = self.attention(hidden, encoder_outputs)

        # Step 3: Attentional vector
        attn_hidden = torch.tanh(
            self.W_c(torch.cat([hidden, context], dim=-1))
        )
        prediction = self.fc_out(attn_hidden)
        return prediction, hidden, attn_weights

decoder = LuongDecoder(8000, 256, 512, method='general')
enc_out = torch.randn(4, 25, 512)
h = torch.randn(4, 512)
tok = torch.randint(0, 8000, (4,))
pred, h_new, w = decoder(tok, h, enc_out)
print(f"Prediction: {pred.shape}")  # (4, 8000)`}
        id="code-luong-decoder"
      />

      <NoteBlock
        type="intuition"
        title="From Luong Dot-Product to Transformer Attention"
        content="The dot-product scoring function in Luong attention is essentially the same as transformer attention without the scaling factor. Transformers generalize this by (1) adding the 1/sqrt(d) scaling, (2) using separate Q, K, V projections, and (3) applying multiple attention heads in parallel. Understanding Luong attention makes the transformer mechanism feel like a natural evolution."
        id="note-to-transformers"
      />

      <WarningBlock
        title="Bahdanau vs. Luong: Timing Matters"
        content="A subtle but important difference: Bahdanau computes attention BEFORE the RNN step (using s_{t-1}), while Luong computes it AFTER (using s_t). Luong's approach is simpler and empirically performs slightly better. In practice, the Luong-style 'attend after decode' pattern became standard."
        id="warning-timing"
      />

      <NoteBlock
        type="historical"
        title="Luong's Contribution"
        content="'Effective Approaches to Attention-based Neural Machine Translation' (Luong, Pham, Manning, 2015) systematically compared attention variants and established that simple dot-product attention is highly effective. This paper, combined with Bahdanau's, formed the foundation that Vaswani et al. built upon when creating the transformer in 2017."
        id="note-luong-history"
      />
    </div>
  )
}
