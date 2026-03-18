import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function DatasetPrep() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Training Dataset Preparation</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Dataset quality is the single most important factor in diffusion model fine-tuning.
        A well-curated dataset of 10-20 images with precise captions will outperform a
        sloppy dataset of hundreds. This section covers the full pipeline from image
        collection through captioning and formatting.
      </p>

      <DefinitionBlock
        title="Training Dataset Structure"
        definition="A diffusion training dataset consists of image-caption pairs. Each image should be high quality, properly cropped, and at or above the training resolution. Captions must accurately describe the image content and include the trigger word for the concept being learned."
        id="def-dataset-structure"
      />

      <ExampleBlock
        title="Image Selection Guidelines"
        problem="How should you select and prepare training images?"
        steps={[
          { formula: '\\text{Variety: diverse angles, lighting, backgrounds}', explanation: 'More variety helps the model generalize. Avoid repetitive poses or settings.' },
          { formula: '\\text{Quality: sharp, well-lit, high resolution}', explanation: 'Minimum resolution should match training resolution (512 or 1024). No blur or artifacts.' },
          { formula: '\\text{Quantity: 5-10 (DreamBooth), 20-100 (style LoRA)}', explanation: 'More is not always better. Quality and variety matter more than raw count.' },
          { formula: '\\text{Consistency: same subject or consistent style}', explanation: 'All images should clearly represent the target concept.' },
        ]}
        id="example-image-guidelines"
      />

      <PythonCode
        title="prepare_dataset.py"
        code={`import os
from pathlib import Path
from PIL import Image
import json

def prepare_image_dataset(input_dir, output_dir, resolution=1024):
    """Prepare images for diffusion model training."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [
        f for f in Path(input_dir).iterdir()
        if f.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images")

    for img_path in sorted(image_files):
        img = Image.open(img_path).convert("RGB")

        # Resize maintaining aspect ratio, then center crop
        w, h = img.size
        scale = resolution / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Center crop to square
        left = (new_w - resolution) // 2
        top = (new_h - resolution) // 2
        img = img.crop((left, top, left + resolution, top + resolution))

        out_name = f"{img_path.stem}.png"
        out_path = os.path.join(output_dir, out_name)
        img.save(out_path, "PNG")

        metadata.append({
            "file_name": out_name,
            "text": "",  # filled by captioning step
            "original": str(img_path),
        })

    with open(os.path.join(output_dir, "metadata.jsonl"), "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\\n")

    print(f"Processed {len(metadata)} images to {output_dir}")
    return metadata

prepare_image_dataset("./raw_images", "./training_dataset", resolution=1024)`}
        id="code-prepare"
      />

      <PythonCode
        title="auto_caption.py"
        code={`import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os, json

def auto_caption_dataset(image_dir, trigger_word="sks", concept_class="person"):
    """Auto-caption training images using a vision-language model."""
    model_id = "microsoft/Florence-2-large"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    captions = {}
    image_files = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg"))]

    for fname in sorted(image_files):
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")

        inputs = processor(
            text="<MORE_DETAILED_CAPTION>",
            images=img,
            return_tensors="pt"
        ).to("cuda", torch.float16)

        generated_ids = model.generate(**inputs, max_new_tokens=256, num_beams=3)
        caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Insert trigger word
        caption = caption.replace(
            f"a {concept_class}", f"a {trigger_word} {concept_class}"
        )
        if trigger_word not in caption:
            caption = f"a photo of {trigger_word} {concept_class}, {caption}"

        captions[fname] = caption
        print(f"{fname}: {caption[:100]}...")

        # Save as .txt sidecar file
        txt_path = os.path.join(image_dir, fname.rsplit(".", 1)[0] + ".txt")
        with open(txt_path, "w") as f:
            f.write(caption)

    # Also save as metadata.jsonl for diffusers
    with open(os.path.join(image_dir, "metadata.jsonl"), "w") as f:
        for fname, caption in captions.items():
            f.write(json.dumps({"file_name": fname, "text": caption}) + "\\n")

    print(f"\\nCaptioned {len(captions)} images")
    return captions

auto_caption_dataset("./training_dataset", trigger_word="sks", concept_class="dog")`}
        id="code-caption"
      />

      <NoteBlock
        type="tip"
        title="Always Review Auto-Captions"
        content="Auto-generated captions are a starting point, not the final product. Manually review every caption and correct errors. Remove hallucinated details, fix trigger word placement, and ensure consistency across the dataset. 15 minutes of caption editing saves hours of bad training."
        id="note-review-captions"
      />

      <WarningBlock
        title="Watermarks and Artifacts"
        content="Training images with watermarks, text overlays, or compression artifacts will teach the model to reproduce those artifacts. Always clean your images: remove watermarks, crop out borders, and use high-quality source files."
        id="warning-artifacts"
      />

      <NoteBlock
        type="note"
        title="Aspect Ratio Bucketing"
        content="Modern training scripts support aspect ratio bucketing, which groups images by aspect ratio instead of forcing everything to a square crop. This preserves more of the original composition. Enable it with --resolution=1024 --center_crop=False --random_flip in the training script."
        id="note-bucketing"
      />
    </div>
  )
}
