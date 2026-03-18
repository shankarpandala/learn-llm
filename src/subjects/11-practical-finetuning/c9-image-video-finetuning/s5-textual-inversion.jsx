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
      <h1 className="text-3xl font-bold">Textual Inversion Training</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Textual inversion learns a new embedding vector for a concept while keeping the entire
        model frozen. It produces the smallest possible adaptation (a single embedding file of
        a few KB) and requires very little VRAM, making it accessible on any GPU.
      </p>

      <DefinitionBlock
        title="Textual Inversion"
        definition="Textual inversion optimizes a new token embedding $v^* \\in \\mathbb{R}^d$ to represent a visual concept. Given a set of images, it minimizes the diffusion loss while only updating $v^*$, keeping the UNet and text encoder frozen. The concept can then be invoked by using the associated token in prompts."
        notation="v^* = \arg\min_v \mathbb{E}_{z,\epsilon,t} \|\ \epsilon - \epsilon_\theta(z_t, t, c(v))\ \|^2"
        id="def-textual-inversion"
      />

      <PythonCode
        title="train_textual_inversion.sh"
        code={`# Train textual inversion with diffusers
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATA_DIR="./concept-images"
export OUTPUT_DIR="./textual-inversion-output"

accelerate launch textual_inversion_sdxl.py \\
  --pretrained_model_name_or_path=$MODEL_NAME \\
  --train_data_dir=$DATA_DIR \\
  --learnable_property="object" \\
  --placeholder_token="<my-concept>" \\
  --initializer_token="dog" \\
  --resolution=1024 \\
  --train_batch_size=1 \\
  --gradient_accumulation_steps=4 \\
  --max_train_steps=3000 \\
  --learning_rate=5e-4 \\
  --lr_scheduler="constant" \\
  --lr_warmup_steps=0 \\
  --output_dir=$OUTPUT_DIR \\
  --mixed_precision="bf16" \\
  --validation_prompt="a <my-concept> in a garden" \\
  --validation_steps=500 \\
  --seed=42`}
        id="code-train-ti"
      />

      <PythonCode
        title="use_textual_inversion.py"
        code={`from diffusers import StableDiffusionXLPipeline
import torch

# Load pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

# Load textual inversion embedding
pipe.load_textual_inversion(
    "./textual-inversion-output",
    token="<my-concept>",
)

# Generate images using the learned concept
prompts = [
    "a <my-concept> sitting on a beach at sunset",
    "a painting of <my-concept> in the style of Van Gogh",
    "a <my-concept> wearing a top hat, studio photography",
]

for i, prompt in enumerate(prompts):
    image = pipe(
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]
    image.save(f"ti_output_{i}.png")
    print(f"Generated: {prompt}")

# Combine with LoRA
# pipe.load_lora_weights("./style-lora")
# image = pipe("a <my-concept> in anime style").images[0]`}
        id="code-use-ti"
      />

      <ExampleBlock
        title="Textual Inversion vs LoRA vs DreamBooth"
        problem="How does textual inversion compare to other image finetuning methods?"
        steps={[
          { formula: '\\text{TI: 1-10 KB output, 4-8 GB VRAM}', explanation: 'Smallest adapter. Learns a concept embedding only. Limited expressiveness.' },
          { formula: '\\text{LoRA: 10-200 MB output, 8-16 GB VRAM}', explanation: 'Medium adapter. Modifies attention layers. Good balance of quality and efficiency.' },
          { formula: '\\text{DreamBooth: full model, 16-24 GB VRAM}', explanation: 'Largest output. Modifies all weights. Best quality for specific subjects.' },
          { formula: '\\text{TI best for: styles, textures, simple concepts}', explanation: 'When you need minimal file size and maximum compatibility.' },
        ]}
        id="example-ti-comparison"
      />

      <NoteBlock
        type="tip"
        title="Initializer Token"
        content="Choose an initializer token that is semantically close to your concept. For a specific dog breed, use 'dog'. For an art style, use 'painting' or 'art'. The initializer gives the optimization a good starting point, leading to faster and better convergence."
        id="note-initializer"
      />

      <WarningBlock
        title="Limited Expressiveness"
        content="Textual inversion can only learn what can be expressed through a single embedding vector. It struggles with complex subjects like specific faces or detailed objects. For those, use LoRA or DreamBooth. TI works best for styles, textures, and simple visual concepts."
        id="warning-limited"
      />

      <NoteBlock
        type="note"
        title="Composability"
        content="Textual inversion embeddings are fully composable: you can use multiple learned tokens in the same prompt, combine them with LoRAs, and they work across different checkpoints of the same model family. This makes them ideal for building a library of reusable concepts."
        id="note-composability"
      />
    </div>
  )
}
