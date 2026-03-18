import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function OCRFreeModels() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">OCR-Free Document Understanding</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        OCR-free models process document images directly without requiring a separate OCR
        preprocessing step. Models like Donut, Nougat, and Pix2Struct learn to read text
        from pixels end-to-end, avoiding OCR errors and simplifying the pipeline. This
        approach is particularly effective for complex layouts, handwriting, and non-Latin scripts.
      </p>

      <DefinitionBlock
        title="OCR-Free Document Model"
        definition="An OCR-free document model is an encoder-decoder architecture where the encoder processes raw document images (as patches) and the decoder generates structured text output (JSON, markdown, etc.) directly. The model implicitly learns character recognition, layout understanding, and information extraction in a unified framework."
        id="def-ocr-free"
      />

      <h2 className="text-2xl font-semibold">Donut Architecture</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Donut (Document Understanding Transformer) uses a Swin Transformer encoder for images
        and a BART-style decoder to generate structured JSON from document images. It is
        pretrained with a synthetic document reading task and fine-tuned for specific extraction tasks.
      </p>

      <ExampleBlock
        title="OCR Pipeline vs OCR-Free"
        problem="Compare the processing steps for extracting invoice total from a document image."
        steps={[
          { formula: '\\text{OCR pipeline: Image} \\to \\text{OCR} \\to \\text{Text+Boxes} \\to \\text{LayoutLM} \\to \\text{Extract}', explanation: 'Traditional approach requires OCR as a separate step, introducing potential errors.' },
          { formula: '\\text{OCR-free: Image} \\to \\text{Donut/Nougat} \\to \\text{JSON output}', explanation: 'Single model directly outputs structured data from pixels.' },
          { formula: '\\text{Error propagation: OCR@95\\% acc} \\to \\text{NER@90\\%} = 85.5\\% \\text{ end-to-end}', explanation: 'OCR pipeline compounds errors; OCR-free avoids this cascade.' },
        ]}
        id="example-pipeline-comparison"
      />

      <PythonCode
        title="donut_inference.py"
        code={`from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Donut for document parsing
# processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
# model = VisionEncoderDecoderModel.from_pretrained(
#     "naver-clova-ix/donut-base-finetuned-cord-v2"
# )

# Example: Parse a receipt
# image = Image.open("receipt.png").convert("RGB")
# pixel_values = processor(image, return_tensors="pt").pixel_values

# # Generate structured output
# task_prompt = "<s_cord-v2>"  # Task-specific prompt token
# decoder_input_ids = processor.tokenizer(
#     task_prompt, add_special_tokens=False, return_tensors="pt"
# ).input_ids

# outputs = model.generate(
#     pixel_values,
#     decoder_input_ids=decoder_input_ids,
#     max_length=model.decoder.config.max_position_embeddings,
#     pad_token_id=processor.tokenizer.pad_token_id,
#     eos_token_id=processor.tokenizer.eos_token_id,
# )
# result = processor.token2json(processor.decode(outputs[0]))
# print(result)  # {"menu": [{"nm": "Coffee", "price": "5.00"}, ...], "total": "13.50"}

# Nougat for academic papers (LaTeX output)
# nougat_processor = DonutProcessor.from_pretrained("facebook/nougat-base")
# nougat_model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

# OCR-free model comparison
models = {
    "Donut":     {"encoder": "Swin-B", "decoder": "BART", "output": "JSON", "params": "200M"},
    "Pix2Struct": {"encoder": "ViT", "decoder": "T5", "output": "HTML/Text", "params": "300M"},
    "Nougat":    {"encoder": "Swin-B", "decoder": "mBART", "output": "LaTeX/MD", "params": "250M"},
    "Florence-2": {"encoder": "DaViT", "decoder": "BART", "output": "Multi-task", "params": "770M"},
}
for name, info in models.items():
    print(f"{name}: {info['encoder']} -> {info['decoder']} ({info['params']}) -> {info['output']}")`}
        id="code-donut"
      />

      <NoteBlock
        type="intuition"
        title="Why OCR-Free Works"
        content="OCR is fundamentally a vision task -- recognizing text from pixels. By training an end-to-end model, the vision encoder learns to extract textual features that are optimized for the downstream task, not just generic character recognition. This is especially powerful for degraded documents, handwriting, and complex layouts where traditional OCR struggles."
        id="note-ocr-free-intuition"
      />

      <NoteBlock
        type="note"
        title="Resolution Matters"
        content="OCR-free models are highly sensitive to input resolution. Small text in documents may be unreadable at 224x224. Most document models use higher resolutions (1024x1024 or more) or dynamic tiling to preserve text legibility. This significantly increases computation but is necessary for practical document understanding."
        id="note-resolution"
      />

      <WarningBlock
        title="Structured Output Reliability"
        content="OCR-free models generate structured output (JSON, markdown) autoregressively, which means they can produce malformed outputs. Always validate the generated structure and implement fallback parsing. For production systems, consider constrained decoding or grammar-guided generation."
        id="warning-structured-output"
      />
    </div>
  )
}
