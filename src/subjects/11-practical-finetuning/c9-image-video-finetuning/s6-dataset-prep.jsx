import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ImageDatasetPrep() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Image Dataset Preparation</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        The quality of image training data has an outsized impact on finetuning results. This
        section covers image collection, captioning, cropping, augmentation, and organizing
        datasets for Stable Diffusion and FLUX LoRA training.
      </p>

      <DefinitionBlock
        title="Image-Caption Pairs"
        definition="Image finetuning datasets consist of image-caption pairs. Each image has an associated text file (same name, .txt extension) containing a caption that describes the image. The caption quality directly affects how well the model learns the concept and how controllable it is via prompts."
        id="def-image-caption"
      />

      <h2 className="text-2xl font-semibold">Auto-Captioning with Vision Models</h2>

      <PythonCode
        title="auto_caption_images.py"
        code={`from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import os
import glob

def caption_with_florence2(image_dir, trigger_word="sks"):
    """Auto-caption images using Florence-2."""
    model_name = "microsoft/Florence-2-large"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    image_files = glob.glob(os.path.join(image_dir, "*.png")) + \\
                  glob.glob(os.path.join(image_dir, "*.jpg"))

    for img_path in image_files:
        image = Image.open(img_path).convert("RGB")

        # Generate detailed caption
        inputs = processor(
            text="<MORE_DETAILED_CAPTION>",
            images=image,
            return_tensors="pt"
        ).to("cuda", torch.float16)

        generated_ids = model.generate(
            **inputs, max_new_tokens=256, num_beams=3
        )
        caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Prepend trigger word
        caption = f"{trigger_word}, {caption}"

        # Save caption file
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        with open(txt_path, "w") as f:
            f.write(caption)

        print(f"{os.path.basename(img_path)}: {caption[:80]}...")

    print(f"\\nCaptioned {len(image_files)} images")

# caption_with_florence2("./training-images", trigger_word="ohwx person")`}
        id="code-auto-caption"
      />

      <PythonCode
        title="image_preprocessing.py"
        code={`from PIL import Image
import os
import glob

def preprocess_images(input_dir, output_dir, target_size=1024,
                      min_quality_size=512):
    """Preprocess and filter images for training."""
    os.makedirs(output_dir, exist_ok=True)

    stats = {"processed": 0, "skipped_small": 0, "skipped_format": 0}

    for img_path in glob.glob(os.path.join(input_dir, "*")):
        ext = os.path.splitext(img_path)[1].lower()
        if ext not in ('.png', '.jpg', '.jpeg', '.webp', '.bmp'):
            stats["skipped_format"] += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            stats["skipped_format"] += 1
            continue

        # Filter small images
        if min(img.size) < min_quality_size:
            stats["skipped_small"] += 1
            continue

        # Resize while maintaining aspect ratio
        # Then center crop to square
        w, h = img.size
        scale = target_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Center crop to target_size x target_size
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        img = img.crop((left, top, left + target_size, top + target_size))

        # Save
        basename = os.path.splitext(os.path.basename(img_path))[0]
        img.save(os.path.join(output_dir, f"{basename}.png"))

        # Copy caption if exists
        caption_src = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(caption_src):
            caption_dst = os.path.join(output_dir, f"{basename}.txt")
            with open(caption_src) as f_in, open(caption_dst, "w") as f_out:
                f_out.write(f_in.read())

        stats["processed"] += 1

    print(f"Processed: {stats['processed']}")
    print(f"Skipped (small): {stats['skipped_small']}")
    print(f"Skipped (format): {stats['skipped_format']}")

# preprocess_images("./raw-images", "./training-images")`}
        id="code-preprocess"
      />

      <ExampleBlock
        title="Dataset Organization"
        problem="How should you organize image datasets for training?"
        steps={[
          { formula: '\\text{training-images/image001.png}', explanation: 'Image file (PNG or JPG, cropped to target resolution).' },
          { formula: '\\text{training-images/image001.txt}', explanation: 'Caption file with same name. Contains text description.' },
          { formula: '\\text{15-50 images for style LoRA}', explanation: 'Styles need more variety. Include different subjects in the target style.' },
          { formula: '\\text{5-20 images for subject LoRA}', explanation: 'Subjects need fewer but high-quality images with varied poses/angles.' },
        ]}
        id="example-organization"
      />

      <NoteBlock
        type="tip"
        title="Caption Quality Tips"
        content="Write captions that describe everything visible: subject, action, background, lighting, style, composition, colors. Use natural language, not tags. Include your trigger word at the start. Example: 'ohwx person sitting at a cafe table, warm afternoon light, candid photo, shallow depth of field'."
        id="note-caption-quality"
      />

      <WarningBlock
        title="Dataset Diversity"
        content="Lack of diversity causes overfitting to specific poses, backgrounds, or lighting. For subject LoRAs: vary backgrounds, angles, lighting, and clothing. For style LoRAs: vary subjects, compositions, and color palettes. A diverse dataset of 20 images outperforms a uniform set of 50."
        id="warning-diversity"
      />
    </div>
  )
}
