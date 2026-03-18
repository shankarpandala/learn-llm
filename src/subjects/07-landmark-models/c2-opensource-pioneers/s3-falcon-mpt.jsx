import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function FalconMPT() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Falcon and MPT: Alternative Open Architectures</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Falcon (TII, Abu Dhabi) and MPT (MosaicML) were among the first fully open-source models
        to compete with LLaMA. They introduced important ideas around data quality, multi-query
        attention at scale, and ALiBi positional encoding, influencing subsequent model designs.
      </p>

      <DefinitionBlock
        title="Multi-Query Attention (MQA)"
        definition="An attention variant where all query heads share a single key and value head. Given $h$ attention heads, MQA reduces KV cache memory by a factor of $h$ compared to standard multi-head attention: $\text{MQA}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$ where $K, V \in \mathbb{R}^{n \times d_k}$ are shared across all heads."
        id="def-mqa"
      />

      <h2 className="text-2xl font-semibold">Falcon Architecture</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Falcon came in 7B, 40B, and 180B sizes. The 40B and 180B models used multi-query attention
        to reduce inference memory. A key innovation was the RefinedWeb dataset: 5 trillion tokens
        of web data filtered through aggressive deduplication and quality filtering. Falcon demonstrated
        that web-only data, properly filtered, could match curated datasets.
      </p>

      <ExampleBlock
        title="Falcon Model Configurations"
        problem="Compare the Falcon model family architectures."
        steps={[
          { formula: '\\text{Falcon-7B}: L{=}32, d{=}4544, h{=}71, \\text{MHA}', explanation: 'Unusual dimensions (not power of 2). Uses standard multi-head attention. Trained on 1.5T tokens from RefinedWeb.' },
          { formula: '\\text{Falcon-40B}: L{=}60, d{=}8192, h{=}128, \\text{MQA}', explanation: 'Uses multi-query attention with 1 KV head. Trained on 1T tokens. Topped the Hugging Face Open LLM Leaderboard at launch.' },
          { formula: '\\text{Falcon-180B}: L{=}80, d{=}14848, h{=}232, \\text{GQA}', explanation: 'Uses GQA with 8 KV groups. Trained on 3.5T tokens. Approached GPT-4 on some benchmarks.' },
        ]}
        id="example-falcon-configs"
      />

      <h2 className="text-2xl font-semibold">MPT Architecture</h2>
      <p className="text-gray-700 dark:text-gray-300">
        MosaicML's MPT (MosaicML Pretrained Transformer) models used ALiBi (Attention with Linear
        Biases) instead of positional embeddings. ALiBi adds a linear bias to attention scores based
        on the distance between query and key positions, enabling length extrapolation without any
        learned positional parameters.
      </p>

      <DefinitionBlock
        title="ALiBi (Attention with Linear Biases)"
        definition="A positional encoding method that adds a head-specific linear bias to attention scores: $\text{softmax}(q_i^T k_j - m \cdot |i - j|)$ where $m$ is a head-specific slope. Slopes are set geometrically: $m_i = 2^{-8i/h}$ for $h$ heads. This requires no learned parameters and naturally extrapolates to longer sequences than seen during training."
        id="def-alibi"
      />

      <PythonCode
        title="falcon_and_mpt_usage.py"
        code={`from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load Falcon-7B
falcon_name = "tiiuae/falcon-7b"
falcon_tokenizer = AutoTokenizer.from_pretrained(falcon_name)
falcon_model = AutoModelForCausalLM.from_pretrained(
    falcon_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

config = falcon_model.config
print("=== Falcon-7B ===")
print(f"Hidden size: {config.hidden_size}")       # 4544
print(f"Num layers: {config.num_hidden_layers}")  # 32
print(f"Num heads: {config.num_attention_heads}") # 71
print(f"Vocab size: {config.vocab_size}")         # 65024

# Load MPT-7B
mpt_name = "mosaicml/mpt-7b"
mpt_tokenizer = AutoTokenizer.from_pretrained(mpt_name)
mpt_model = AutoModelForCausalLM.from_pretrained(
    mpt_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

mpt_config = mpt_model.config
print("\\n=== MPT-7B ===")
print(f"Hidden size: {mpt_config.d_model}")       # 4096
print(f"Num layers: {mpt_config.n_layers}")       # 32
print(f"Num heads: {mpt_config.n_heads}")         # 32

# ALiBi slopes computation
import math
def get_alibi_slopes(num_heads):
    closest_power = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power) - 3)))
    slopes = [base ** (i + 1) for i in range(closest_power)]
    return slopes

slopes = get_alibi_slopes(32)
print(f"\\nALiBi slopes (first 8): {[f'{s:.6f}' for s in slopes[:8]]}")`}
        id="code-falcon-mpt"
      />

      <NoteBlock
        type="intuition"
        title="RefinedWeb: Data Quality Over Quantity"
        content="Falcon's RefinedWeb pipeline applies URL filtering, language identification, fuzzy deduplication (MinHash), and quality heuristics to Common Crawl. The key finding: web data cleaned to near-curated quality produces models as good as those trained on hand-picked datasets. This democratized training data, as anyone can apply similar filtering to publicly available crawls."
        id="note-refinedweb"
      />

      <WarningBlock
        title="Trust Remote Code"
        content="Both Falcon and MPT require trust_remote_code=True when loading, meaning they execute custom Python code from the model repository. Always review the model card and source code before using this flag, as it could potentially execute arbitrary code on your machine."
        id="warning-trust-remote"
      />
    </div>
  )
}
