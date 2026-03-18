import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function GPT123() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">The GPT Evolution: GPT-1, GPT-2, and GPT-3</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        OpenAI's Generative Pre-trained Transformer series established the paradigm of large-scale
        autoregressive language modeling. Each generation scaled parameters, data, and compute by
        orders of magnitude, revealing emergent capabilities that smaller models lacked.
      </p>

      <DefinitionBlock
        title="Autoregressive Language Model"
        definition="A model that generates text by predicting one token at a time, conditioning on all previous tokens. The probability of a sequence is $P(x_1, \ldots, x_n) = \prod_{i=1}^{n} P(x_i \mid x_1, \ldots, x_{i-1})$."
        id="def-autoregressive"
      />

      <h2 className="text-2xl font-semibold">GPT-1 (June 2018)</h2>
      <p className="text-gray-700 dark:text-gray-300">
        GPT-1 introduced the idea of unsupervised pre-training followed by supervised fine-tuning.
        With 117M parameters, 12 transformer layers, and a 768-dimensional hidden state, it was
        trained on the BooksCorpus (~5GB of text). It used a learned positional embedding and the
        standard decoder-only transformer with masked self-attention.
      </p>

      <NoteBlock
        type="historical"
        title="The Pre-training Revolution"
        content="Before GPT-1, NLP relied on task-specific architectures. Radford et al. (2018) showed that a single pre-trained model could be fine-tuned to achieve state-of-the-art on 9 of 12 NLP benchmarks, fundamentally shifting how NLP research was done."
        id="note-gpt1-impact"
      />

      <h2 className="text-2xl font-semibold">GPT-2 (February 2019)</h2>
      <p className="text-gray-700 dark:text-gray-300">
        GPT-2 scaled to 1.5B parameters (48 layers, 1600-dimensional hidden state) trained on
        WebText (40GB scraped from Reddit outbound links with 3+ karma). The key insight was that
        language models could perform downstream tasks as zero-shot learners without any fine-tuning,
        simply by framing tasks as text completion.
      </p>

      <ExampleBlock
        title="GPT-2 Model Sizes"
        problem="Compare the four released GPT-2 model variants."
        steps={[
          { formula: 'GPT\\text{-}2~\\text{Small}: 117\\text{M params}, L{=}12, d{=}768, h{=}12', explanation: 'Matches GPT-1 in size but trained on significantly more data.' },
          { formula: 'GPT\\text{-}2~\\text{Medium}: 345\\text{M params}, L{=}24, d{=}1024, h{=}16', explanation: 'Doubled depth and increased width over Small.' },
          { formula: 'GPT\\text{-}2~\\text{Large}: 762\\text{M params}, L{=}36, d{=}1280, h{=}20', explanation: 'Further scaling of both depth and width.' },
          { formula: 'GPT\\text{-}2~\\text{XL}: 1.5\\text{B params}, L{=}48, d{=}1600, h{=}25', explanation: 'The full model, 10x larger than GPT-1.' },
        ]}
        id="example-gpt2-sizes"
      />

      <h2 className="text-2xl font-semibold">GPT-3 (June 2020)</h2>
      <p className="text-gray-700 dark:text-gray-300">
        GPT-3 massively scaled to 175B parameters (96 layers, 12288-dimensional hidden state, 96
        attention heads). Trained on a filtered Common Crawl plus curated datasets totaling ~570GB
        of text. It introduced few-shot in-context learning: providing examples directly in the prompt
        without any gradient updates.
      </p>

      <DefinitionBlock
        title="In-Context Learning"
        definition="The ability of a language model to learn a task from a few demonstration examples provided in the prompt at inference time, without any parameter updates. Formally, $P(y \mid x, \{(x_i, y_i)\}_{i=1}^{k})$ where the examples are simply concatenated as text."
        id="def-icl"
      />

      <PythonCode
        title="loading_gpt2_huggingface.py"
        code={`from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 (124M version)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Inspect architecture
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Layers: {model.config.n_layer}")
print(f"Hidden size: {model.config.n_embd}")
print(f"Attention heads: {model.config.n_head}")
print(f"Vocab size: {model.config.vocab_size}")
# Parameters: 124,439,808
# Layers: 12, Hidden: 768, Heads: 12, Vocab: 50257

# Generate text
input_ids = tokenizer.encode("The future of AI is", return_tensors="pt")
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
    )
print(tokenizer.decode(output[0], skip_special_tokens=True))`}
        id="code-gpt2-load"
      />

      <NoteBlock
        type="intuition"
        title="Scaling Laws"
        content="GPT-3 demonstrated that model performance follows power-law scaling with model size, dataset size, and compute budget. Kaplan et al. (2020) formalized this: loss scales as L ~ C^(-0.05) where C is compute in PetaFLOP-days. This means predictable gains from scaling."
        id="note-scaling-laws"
      />

      <WarningBlock
        title="Compute Requirements"
        content="GPT-3 required ~3640 PetaFLOP-days to train, estimated at $4.6M on cloud GPUs in 2020 prices. The 175B parameters demand at least 350GB in float16, making it impractical to run locally. API access became the primary mode of interaction, establishing the LLM-as-a-service paradigm."
        id="warning-compute"
      />
    </div>
  )
}
