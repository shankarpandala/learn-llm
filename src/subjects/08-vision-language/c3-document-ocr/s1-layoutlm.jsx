import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LayoutLM() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LayoutLM: Document Understanding with Layout</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        LayoutLM extends BERT-style models to understand documents by incorporating 2D positional
        information (bounding box coordinates) alongside text tokens. This enables the model to
        reason about the spatial layout of text on a page, which is critical for understanding
        forms, invoices, receipts, and other structured documents.
      </p>

      <DefinitionBlock
        title="LayoutLM Embedding"
        definition="LayoutLM augments the standard token embedding with 2D position embeddings derived from OCR bounding boxes. For each token, the input representation is $\mathbf{e}_i = \mathbf{e}_{\text{token}} + \mathbf{e}_{\text{1D-pos}} + \mathbf{e}_{\text{x0}} + \mathbf{e}_{\text{y0}} + \mathbf{e}_{\text{x1}} + \mathbf{e}_{\text{y1}} + \mathbf{e}_{\text{w}} + \mathbf{e}_{\text{h}}$ where $(x_0, y_0, x_1, y_1)$ are normalized bounding box coordinates."
        id="def-layoutlm"
      />

      <h2 className="text-2xl font-semibold">LayoutLM Evolution</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The LayoutLM family progressed from text + layout (v1) to incorporating the document
        image (v2, v3). LayoutLMv3 uses a unified architecture that jointly processes text
        tokens, layout positions, and image patches within a single transformer.
      </p>

      <ExampleBlock
        title="LayoutLM Coordinate Encoding"
        problem="A word 'Total' appears at pixel coordinates (320, 580, 410, 610) in a 1000x1000 normalized document. How is this encoded?"
        steps={[
          { formula: '(x_0, y_0, x_1, y_1) = (320, 580, 410, 610)', explanation: 'Coordinates normalized to 0-1000 range.' },
          { formula: 'w = x_1 - x_0 = 90, \\quad h = y_1 - y_0 = 30', explanation: 'Width and height of the bounding box.' },
          { formula: '\\mathbf{e} = \\text{Embed}_{x0}(320) + \\text{Embed}_{y0}(580) + \\ldots', explanation: 'Each coordinate has its own embedding table (size 1001 x D).' },
        ]}
        id="example-layout-coords"
      />

      <PythonCode
        title="layoutlm_document.py"
        code={`from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch

# LayoutLMv3 for document token classification (e.g., form field extraction)
# processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
# model = LayoutLMv3ForTokenClassification.from_pretrained(
#     "microsoft/layoutlmv3-base", num_labels=7
# )

# Simulated OCR output for a receipt
words = ["RECEIPT", "Item", "Qty", "Price", "Coffee", "2", "$5.00",
         "Sandwich", "1", "$8.50", "Total", "$13.50"]
# Bounding boxes: [x0, y0, x1, y1] normalized to 0-1000
boxes = [
    [350, 50, 650, 100],   # RECEIPT (header, centered)
    [50, 150, 200, 180],   # Item
    [300, 150, 400, 180],  # Qty
    [600, 150, 750, 180],  # Price
    [50, 200, 200, 230],   # Coffee
    [300, 200, 350, 230],  # 2
    [600, 200, 750, 230],  # $5.00
    [50, 250, 250, 280],   # Sandwich
    [300, 250, 350, 280],  # 1
    [600, 250, 750, 280],  # $8.50
    [50, 350, 200, 380],   # Total
    [600, 350, 750, 380],  # $13.50
]

# Labels: 0=Other, 1=Header, 2=Key, 3=Value
labels = [1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 3]

print("Document layout:")
for word, box, label in zip(words, boxes, labels):
    label_name = ["Other", "Header", "Key", "Value"][label]
    print(f"  '{word}' at ({box[0]},{box[1]})-({box[2]},{box[3]}) -> {label_name}")

# LayoutLMv3 processes words + boxes + image together
# encoding = processor(
#     Image.new("RGB", (1000, 1000)),
#     words, boxes=boxes, return_tensors="pt"
# )
# outputs = model(**encoding)
# predictions = outputs.logits.argmax(-1)

# Key LayoutLM versions
versions = {
    "LayoutLM v1": "Text + 2D layout (no image)",
    "LayoutLM v2": "Text + layout + image (separate encoders)",
    "LayoutLM v3": "Unified text + layout + image patches",
}
for v, desc in versions.items():
    print(f"{v}: {desc}")`}
        id="code-layoutlm"
      />

      <NoteBlock
        type="tip"
        title="OCR Preprocessing"
        content="LayoutLM requires OCR-extracted text with bounding boxes as input. For best results, use a high-quality OCR engine like Tesseract, PaddleOCR, or cloud APIs (Google Vision, Azure Document Intelligence). The OCR quality directly bounds LayoutLM's performance."
        id="note-ocr-preprocessing"
      />

      <WarningBlock
        title="Coordinate Normalization"
        content="LayoutLM expects bounding box coordinates normalized to the range [0, 1000]. Mismatched normalization (e.g., using raw pixel coordinates or [0, 1] range) is a common source of poor results. Always verify coordinate ranges match what the model expects."
        id="warning-normalization"
      />
    </div>
  )
}
