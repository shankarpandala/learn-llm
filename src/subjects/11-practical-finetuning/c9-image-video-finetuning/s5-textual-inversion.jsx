import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function TextualInversion() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Textual Inversion & Embeddings</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Textual Inversion learns a new text embedding to represent a concept without
        modifying any model weights. It is the lightest fine-tuning method, producing
        tiny embedding files (a few KB) that can be dropped into any compatible model.
        The trade-off is lower fidelity compared to LoRA or DreamBooth.
      </p>

      <DefinitionBlock
        title="Textual Inversion"
        definition="Textual Inversion optimizes a new token embedding $v^*$ in the text encoder's embedding space to represent a target concept. Only the embedding vector is trained while all model weights remain frozen: $v^* = \arg\min_{v} \mathbb{E}_{x, \epsilon, t}[\|\epsilon - \epsilon_\theta(x_t, t, c_\theta(v))\|^2]$. The resulting embedding is a single vector of dimension $d = 768$ (SD 1.5) or $d = 1280$ (SDXL)."
        id="def-textual-inversion"
      />

      <ExampleBlock
        title="When to Use Textual Inversion vs LoRA"
        problem="How do Textual Inversion and LoRA compare?"
        steps={[
          { formula: '\\text{TI: 4-100 KB file size}', explanation: 'Extremely small. Can share hundreds of concepts without storage concerns.' },
          { formula: '\\text{LoRA: 10-400 MB file size}', explanation: 'Larger but captures much more detail about the concept.' },
          { formula: '\\text{TI: no model weight changes}', explanation: 'Works across any checkpoint using the same text encoder. Maximum compatibility.' },
          { formula: '\\text{LoRA: modifies UNet attention}', explanation: 'Tied to the specific model architecture. Better fidelity for complex concepts.' },
          { formula: '\\text{TI: good for styles and simple concepts}', explanation: 'Works well for art styles, textures, and simple objects. Struggles with faces.' },
        ]}
        id="example-ti-vs-lora"
      />

      <PythonCode
        title="train_textual_inversion.py"
        code={`# Textual Inversion training with diffusers
# Install: pip install diffusers[training] accelerate

TRAIN_CMD = """
accelerate launch diffusers/examples/textual_inversion/textual_inversion.py \\
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \\
    --train_data_dir="./data/my_concept" \\
    --learnable_property="style" \\
    --placeholder_token="<my-style>" \\
    --initializer_token="painting" \\
    --resolution=512 \\
    --train_batch_size=1 \\
    --gradient_accumulation_steps=4 \\
    --max_train_steps=3000 \\
    --learning_rate=5e-4 \\
    --lr_scheduler="constant" \\
    --lr_warmup_steps=0 \\
    --output_dir="./textual-inversion-output" \\
    --save_steps=500 \\
    --mixed_precision="bf16"
"""

# Python API for initialization
from diffusers import StableDiffusionPipeline
import torch

def setup_textual_inversion(model_name, placeholder_token, init_token):
    """Initialize a new token for textual inversion."""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    num_added = tokenizer.add_tokens(placeholder_token)
    print(f"Added {num_added} token(s): '{placeholder_token}'")

    text_encoder.resize_token_embeddings(len(tokenizer))

    token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    init_id = tokenizer.convert_tokens_to_ids(init_token)

    embeds = text_encoder.get_input_embeddings().weight.data
    embeds[token_id] = embeds[init_id].clone()
    print(f"Initialized '{placeholder_token}' from '{init_token}'")

    return pipe, token_id

pipe, token_id = setup_textual_inversion(
    "stabilityai/stable-diffusion-xl-base-1.0",
    "<watercolor-sketch>",
    "watercolor"
)
print(TRAIN_CMD)`}
        id="code-train-ti"
      />

      <PythonCode
        title="use_textual_inversion.py"
        code={`import torch
from diffusers import StableDiffusionXLPipeline

def load_textual_inversion(model_name, embedding_path, token_name):
    """Load a textual inversion embedding and generate images."""
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to("cuda")

    pipe.load_textual_inversion(embedding_path, token=token_name)

    prompts = [
        f"a landscape in {token_name}",
        f"a portrait of a cat in {token_name}",
        f"a cityscape at sunset in {token_name}",
    ]

    images = []
    for prompt in prompts:
        image = pipe(
            prompt, num_inference_steps=30, guidance_scale=7.5,
        ).images[0]
        images.append(image)
        safe_name = prompt[:30].replace(" ", "_")
        image.save(f"ti_{safe_name}.png")

    return images

# Combine multiple embeddings
def combine_embeddings(model_name, embeddings):
    """Load multiple TI embeddings into one pipeline."""
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to("cuda")

    for path, token in embeddings:
        pipe.load_textual_inversion(path, token=token)

    image = pipe(
        "a <my-style> portrait with <my-lighting> effects",
        num_inference_steps=30,
    ).images[0]
    return image

load_textual_inversion(
    "stabilityai/stable-diffusion-xl-base-1.0",
    "./textual-inversion-output/learned_embeds.safetensors",
    "<my-style>"
)`}
        id="code-use-ti"
      />

      <NoteBlock
        type="tip"
        title="Initialize from a Related Token"
        content="Always initialize the new embedding from a semantically related token. For a style, use 'painting' or 'art'. For a dog breed, use 'dog'. This gives the optimization a much better starting point and typically converges in fewer steps."
        id="note-init-token"
      />

      <WarningBlock
        title="Textual Inversion Limitations"
        content="Textual Inversion can only capture what is expressible in the text embedding space. It cannot learn new visual patterns that the model has never seen. Complex subjects like specific faces or intricate patterns will not be captured well. Use LoRA or DreamBooth for high-fidelity subject preservation."
        id="warning-ti-limits"
      />

      <NoteBlock
        type="note"
        title="Multi-Vector Embeddings"
        content="Some implementations allow learning multiple embedding vectors per concept (e.g., 4-8 vectors instead of 1). This increases capacity at the cost of using more of the prompt token budget. Specify --num_vectors=4 in the training script to enable this."
        id="note-multi-vector"
      />
    </div>
  )
}
