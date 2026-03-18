import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function BertOverview() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">BERT: Bidirectional Encoder Representations from Transformers</h1>
      <p className="text-lg text-gray-300">
        BERT (Devlin et al., 2018) introduced deep bidirectional pretraining for language understanding.
        Unlike left-to-right models, BERT conditions on both left and right context simultaneously,
        enabling richer representations for downstream tasks.
      </p>

      <DefinitionBlock
        title="BERT Architecture"
        definition="BERT is a multi-layer bidirectional Transformer encoder. BERT-Base has $L=12$ layers, $H=768$ hidden dimensions, and $A=12$ attention heads (110M parameters). BERT-Large uses $L=24$, $H=1024$, $A=16$ (340M parameters)."
        notation="$\text{BERT}(x) = \text{TransformerEncoder}(E(x) + P(x) + S(x))$ where $E$ is token embedding, $P$ is position embedding, $S$ is segment embedding."
        id="bert-arch-def"
      />

      <NoteBlock
        type="historical"
        title="Why Bidirectional Matters"
        content="Before BERT, models like ELMo used separately trained left-to-right and right-to-left LSTMs. GPT used unidirectional (left-to-right) Transformers. BERT showed that jointly conditioning on both directions in every layer leads to substantially better representations."
        id="bidirectional-note"
      />

      <ExampleBlock
        title="BERT Input Representation"
        problem="How does BERT represent the input '[CLS] The cat sat [SEP] It was tired [SEP]'?"
        steps={[
          {
            formula: 'E_{\\text{input}} = E_{\\text{token}} + E_{\\text{segment}} + E_{\\text{position}}',
            explanation: 'Each input token gets three embeddings summed together.'
          },
          {
            formula: 'E_{\\text{segment}} \\in \\{E_A, E_B\\}',
            explanation: 'Segment A for first sentence, Segment B for second sentence.'
          },
          {
            formula: 'E_{\\text{position}} \\in \\mathbb{R}^{512 \\times H}',
            explanation: 'Learned positional embeddings support sequences up to 512 tokens.'
          }
        ]}
        id="bert-input-example"
      />

      <PythonCode
        title="bert_architecture.py"
        code={`from transformers import BertModel, BertTokenizer, BertConfig
import torch

# Load pretrained BERT-Base
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Inspect architecture
config = model.config
print(f"Layers: {config.num_hidden_layers}")        # 12
print(f"Hidden size: {config.hidden_size}")           # 768
print(f"Attention heads: {config.num_attention_heads}")  # 12
print(f"Vocab size: {config.vocab_size}")             # 30522
print(f"Max position: {config.max_position_embeddings}")  # 512
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Encode a sentence pair
inputs = tokenizer(
    "The cat sat on the mat.",
    "It was very comfortable.",
    return_tensors="pt",
    padding=True
)
print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Token type IDs: {inputs['token_type_ids']}")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state: [batch, seq_len, hidden]
# outputs.pooler_output: [batch, hidden] (CLS representation)
print(f"Last hidden state: {outputs.last_hidden_state.shape}")
print(f"Pooler output: {outputs.pooler_output.shape}")

# Access individual layer outputs
outputs_all = model(**inputs, output_hidden_states=True)
print(f"Number of hidden states: {len(outputs_all.hidden_states)}")  # 13 (embed + 12 layers)`}
        id="bert-code"
      />

      <TheoremBlock
        title="BERT Pretraining Objectives"
        statement="BERT is pretrained with two objectives: (1) Masked Language Modeling (MLM) and (2) Next Sentence Prediction (NSP). The total loss is $\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$."
        proof="MLM enables bidirectional context by masking 15% of tokens and predicting them. NSP trains binary classification on whether sentence B follows sentence A. Together they produce representations useful for both token-level and sentence-level tasks."
        id="bert-objectives-thm"
      />

      <WarningBlock
        title="BERT Is Not a Generative Model"
        content="BERT cannot generate text autoregressively. Its bidirectional attention means every token sees every other token during encoding. For generation tasks, use decoder-only (GPT) or encoder-decoder (T5) architectures."
        id="bert-not-generative"
      />
    </div>
  )
}
