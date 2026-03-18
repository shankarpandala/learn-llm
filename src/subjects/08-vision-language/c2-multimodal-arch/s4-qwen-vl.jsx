import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function QwenVL() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Qwen-VL: Versatile Vision-Language Model</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Qwen-VL from Alibaba combines a ViT vision encoder with the Qwen LLM through a
        cross-attention resampler. It supports diverse visual tasks including image
        captioning, visual question answering, grounding (bounding box output), and OCR.
        Qwen-VL demonstrates that careful multi-task training with structured outputs enables
        a single model to handle both understanding and localization.
      </p>

      <DefinitionBlock
        title="Qwen-VL Architecture"
        definition="Qwen-VL uses a ViT-bigG vision encoder (1.9B params) with a single-layer cross-attention module that compresses visual features from 256 tokens to a fixed set of 256 compressed visual tokens. These are prepended to text tokens and processed by the Qwen-7B LLM with position-aware bounding box tokens."
        id="def-qwen-vl"
      />

      <h2 className="text-2xl font-semibold">Bounding Box as Text</h2>
      <p className="text-gray-700 dark:text-gray-300">
        A key innovation in Qwen-VL is representing bounding boxes as normalized coordinate
        tokens within the text vocabulary. This allows the model to output grounding
        information as part of its natural language response without requiring a separate
        detection head.
      </p>

      <ExampleBlock
        title="Qwen-VL Grounding Format"
        problem="How does Qwen-VL represent the bounding box of a cat at coordinates (120, 80, 340, 290) in a 640x480 image?"
        steps={[
          { formula: '\\text{Normalize: } (\\frac{120}{640}, \\frac{80}{480}, \\frac{340}{640}, \\frac{290}{480})', explanation: 'Convert pixel coordinates to [0, 1] range.' },
          { formula: '\\text{Quantize to 1000 bins: } (187, 166, 531, 604)', explanation: 'Multiply by 1000 and round to get integer tokens.' },
          { formula: '\\text{Output: } \\texttt{<ref>cat</ref><box>(187,166),(531,604)</box>}', explanation: 'Special tokens wrap the object name and coordinates in the text output.' },
        ]}
        id="example-grounding"
      />

      <PythonCode
        title="qwen_vl_inference.py"
        code={`# Qwen-VL inference with transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# model_id = "Qwen/Qwen-VL-Chat"
# tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, device_map="auto", trust_remote_code=True
# ).eval()

# Qwen-VL uses special image tags in the conversation
# query = tokenizer.from_list_format([
#     {'image': 'https://example.com/photo.jpg'},
#     {'text': 'Describe this image and locate all objects.'},
# ])
# response, history = model.chat(tokenizer, query=query, history=None)
# print(response)

# Simulate Qwen-VL bounding box tokenization
def normalize_bbox(bbox, img_w, img_h, num_bins=1000):
    """Convert pixel bbox to Qwen-VL format."""
    x1, y1, x2, y2 = bbox
    # Normalize and quantize
    nx1 = round(x1 / img_w * num_bins)
    ny1 = round(y1 / img_h * num_bins)
    nx2 = round(x2 / img_w * num_bins)
    ny2 = round(y2 / img_h * num_bins)
    return f"<box>({nx1},{ny1}),({nx2},{ny2})</box>"

def parse_bbox(bbox_str, img_w, img_h, num_bins=1000):
    """Parse Qwen-VL bbox string back to pixel coordinates."""
    import re
    match = re.search(r'\\((\d+),(\d+)\\),\\((\d+),(\d+)\\)', bbox_str)
    if match:
        coords = [int(x) for x in match.groups()]
        return [
            coords[0] / num_bins * img_w, coords[1] / num_bins * img_h,
            coords[2] / num_bins * img_w, coords[3] / num_bins * img_h,
        ]

# Example
bbox = (120, 80, 340, 290)
img_w, img_h = 640, 480
bbox_str = normalize_bbox(bbox, img_w, img_h)
print(f"Bbox string: {bbox_str}")
# <box>(188,167),(531,604)</box>

# Qwen-VL-2 architecture improvements
configs = {
    "Qwen-VL":   {"vision": "ViT-bigG", "resampler": "256 tokens", "llm": "Qwen-7B"},
    "Qwen2-VL":  {"vision": "ViT + NaViT", "resampler": "Dynamic", "llm": "Qwen2-72B"},
}
for name, cfg in configs.items():
    print(f"{name}: {cfg}")`}
        id="code-qwen-vl"
      />

      <NoteBlock
        type="note"
        title="Qwen2-VL Improvements"
        content="Qwen2-VL introduced dynamic resolution support via NaViT-style packing, allowing images of any aspect ratio without distortion. It also added video understanding by treating video frames as a sequence of images with temporal position encoding. The 72B variant achieves GPT-4V-level performance on many benchmarks."
        id="note-qwen2-vl"
      />

      <NoteBlock
        type="tip"
        title="Multi-Image and Interleaved Input"
        content="Qwen-VL natively supports multiple images in a single conversation by inserting image tokens at different positions. This enables tasks like image comparison, multi-step visual reasoning, and document processing where multiple pages must be analyzed together."
        id="note-multi-image"
      />

      <WarningBlock
        title="Trust Remote Code"
        content="Qwen-VL requires trust_remote_code=True because it uses custom model architectures not yet integrated into the core transformers library. Always review the remote code before running in production environments, as it can execute arbitrary Python code."
        id="warning-trust-code"
      />
    </div>
  )
}
