import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function CNNvsRNN() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Comparing CNNs vs RNNs for NLP</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        CNNs and RNNs represent two fundamentally different approaches to processing text.
        RNNs model sequences through recurrence and maintain an evolving hidden state, while
        CNNs detect local patterns through convolution and build global understanding through
        hierarchical stacking. Understanding their tradeoffs explains why both were eventually
        superseded by transformers, which combine the strengths of each.
      </p>

      <DefinitionBlock
        title="Computational Complexity Comparison"
        definition="For a sequence of length $L$, hidden dimension $d$, and kernel size $k$: RNN has $O(L \cdot d^2)$ sequential operations with path length $O(L)$. CNN has $O(k \cdot L \cdot d^2)$ parallelizable operations with path length $O(\log_k L)$ for dilated convolutions."
        id="def-complexity"
      />

      <h2 className="text-2xl font-semibold">Key Tradeoffs</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The fundamental difference is <strong>sequential vs. parallel</strong> computation.
        RNNs must process tokens one at a time (O(L) sequential steps), while CNNs process
        all positions simultaneously (O(1) sequential steps per layer, O(L/k) layers for full
        coverage).
      </p>

      <ExampleBlock
        title="Training Speed Comparison"
        problem="Compare wall-clock time to process a batch of 32 sequences of length 200 with hidden_dim=256."
        steps={[
          { formula: '\\text{LSTM: } 200 \\text{ sequential steps} \\times O(d^2)', explanation: 'Each step depends on the previous -- cannot parallelize across time steps.' },
          { formula: '\\text{CNN (k=3): } O(1) \\text{ steps} \\times O(k \\cdot d^2)', explanation: 'All positions computed simultaneously within each layer.' },
          { formula: '\\text{Empirical: CNN } \\approx 3\\text{-}10\\times \\text{ faster}', explanation: 'On GPU, CNNs exploit parallelism far better. LSTM achieves ~20K tokens/sec vs CNN ~100K tokens/sec.' },
          { formula: '\\text{But: CNN needs more layers for global context}', explanation: 'A single CNN layer only sees k tokens. Need O(log L) dilated layers for full coverage.' },
        ]}
        id="example-speed"
      />

      <PythonCode
        title="benchmark_cnn_vs_rnn.py"
        code={`import torch
import torch.nn as nn
import time

def benchmark(model, x, name, n_runs=100):
    """Benchmark forward pass speed."""
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(x)
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    print(f"{name}: {elapsed:.2f} ms/batch")

# Define comparable models
class LSTMModel(nn.Module):
    def __init__(self, vocab, embed, hidden, classes):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, classes)
    def forward(self, x):
        out, (h, _) = self.lstm(self.emb(x))
        return self.fc(h[-1])

class CNNModel(nn.Module):
    def __init__(self, vocab, embed, hidden, classes):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed, hidden // 3, k) for k in [3, 4, 5]
        ])
        self.fc = nn.Linear((hidden // 3) * 3, classes)
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        pooled = [torch.relu(c(e)).max(2).values for c in self.convs]
        return self.fc(torch.cat(pooled, 1))

V, E, H, C = 10000, 128, 256, 5
lstm = LSTMModel(V, E, H, C)
cnn = CNNModel(V, E, H, C)
x = torch.randint(0, V, (32, 200))

benchmark(lstm, x, "LSTM")
benchmark(cnn, x, "CNN")
# Typical CPU results:
#   LSTM: ~45 ms/batch
#   CNN:  ~8 ms/batch (5-6x faster)`}
        id="code-benchmark"
      />

      <h2 className="text-2xl font-semibold">When to Use Which</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The choice between CNN and RNN depends on the task characteristics:
      </p>

      <PythonCode
        title="task_comparison.py"
        code={`# Decision guide for CNN vs RNN (pre-transformer era)

task_recommendations = {
    "Sentiment analysis (short text)": {
        "winner": "CNN",
        "reason": "Local patterns (n-grams) are sufficient; speed matters"
    },
    "Named entity recognition": {
        "winner": "BiLSTM",
        "reason": "Needs full sequence context for each token prediction"
    },
    "Machine translation": {
        "winner": "LSTM + Attention",
        "reason": "Variable-length output, long-range dependencies"
    },
    "Document classification": {
        "winner": "CNN or hierarchical",
        "reason": "Key phrases can appear anywhere; max-pooling captures this"
    },
    "Language modeling": {
        "winner": "LSTM",
        "reason": "Autoregressive generation requires sequential processing"
    },
    "Spam detection": {
        "winner": "CNN",
        "reason": "Pattern matching (keywords, phrases); speed is critical"
    },
}

for task, info in task_recommendations.items():
    print(f"{task}")
    print(f"  Recommended: {info['winner']}")
    print(f"  Reason: {info['reason']}\\n")

# Note: In the transformer era (post-2018), transformers dominate
# nearly all of these tasks. But understanding CNN/RNN tradeoffs
# illuminates WHY transformers work: they combine parallel computation
# (like CNNs) with global attention (like RNNs).`}
        id="code-comparison"
      />

      <NoteBlock
        type="intuition"
        title="Why Transformers Won"
        content="RNNs have O(L) sequential path length (slow training) but constant-time long-range connections. CNNs have O(1) parallel computation but O(log L) path length for long-range connections. Transformers achieve BOTH: O(1) parallel computation across all positions AND O(1) path length between any two tokens via direct attention. They combine the parallelism of CNNs with the global context of RNNs."
        id="note-why-transformers"
      />

      <WarningBlock
        title="Benchmarks Can Be Misleading"
        content="Raw speed comparisons between CNNs and RNNs depend heavily on sequence length, batch size, hidden dimensions, and hardware (CPU vs GPU). CNNs are particularly advantaged on GPUs due to parallelism. On very short sequences (< 20 tokens), the overhead of multiple CNN layers can make LSTMs competitive."
        id="warning-benchmarks"
      />

      <NoteBlock
        type="note"
        title="The Hybrid Approach"
        content="Some architectures combine CNNs and RNNs: use CNN layers to extract local features and reduce sequence length, then feed the output to an RNN for global reasoning. The RCNN (Lai et al., 2015) and CNN-LSTM models were popular hybrids. Modern architectures like Mamba and RWKV revisit the RNN idea with better parallelism."
        id="note-hybrid"
      />
    </div>
  )
}
