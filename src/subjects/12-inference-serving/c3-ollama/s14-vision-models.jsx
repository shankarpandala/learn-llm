import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function VisionModels() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Ollama with Vision Models</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Ollama supports multimodal vision-language models that can understand images alongside text.
        You can send images via the API and get descriptions, analysis, or answers to questions
        about visual content, all running locally.
      </p>

      <DefinitionBlock
        title="Vision-Language Models in Ollama"
        definition="Ollama supports models like LLaVA, BakLLaVA, and LLaMA 3.2 Vision that process both text and images. Images are sent as base64-encoded data in the 'images' field of the API request. The model's vision encoder processes the image while the language model generates the response."
        id="def-vision"
      />

      <PythonCode
        title="Terminal"
        code={`# Pull a vision model
ollama pull llava:7b                    # LLaVA 1.6 (Mistral-based)
ollama pull llama3.2-vision:11b         # LLaMA 3.2 Vision 11B
ollama pull moondream                   # Small but capable (1.8B)

# Use vision from CLI (drag and drop or path)
ollama run llava:7b "Describe this image: ./photo.jpg"

# Via API with base64 image
# (base64 encoding shown in Python examples below)
curl http://localhost:11434/api/generate -d '{
  "model": "llava:7b",
  "prompt": "What do you see in this image?",
  "images": ["<base64-encoded-image>"]
}'`}
        id="code-vision-setup"
      />

      <PythonCode
        title="vision_api.py"
        code={`import ollama
import base64
from pathlib import Path

def encode_image(image_path):
    """Read and base64-encode an image file."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Method 1: Using the Ollama Python library
response = ollama.chat(
    model="llava:7b",
    messages=[{
        "role": "user",
        "content": "Describe this image in detail.",
        "images": ["./screenshot.png"],  # File path works directly
    }]
)
print(response["message"]["content"])

# Method 2: Using requests with base64
import requests

image_b64 = encode_image("./chart.png")
resp = requests.post("http://localhost:11434/api/chat", json={
    "model": "llava:7b",
    "messages": [{
        "role": "user",
        "content": "What does this chart show? Summarize the key trends.",
        "images": [image_b64],
    }],
    "stream": False,
})
print(resp.json()["message"]["content"])

# Method 3: Multiple images
response = ollama.chat(
    model="llava:7b",
    messages=[{
        "role": "user",
        "content": "Compare these two images. What are the differences?",
        "images": ["./image1.png", "./image2.png"],
    }]
)
print(response["message"]["content"])`}
        id="code-vision-python"
      />

      <ExampleBlock
        title="Vision Model Use Cases"
        problem="What can you do with local vision models?"
        steps={[
          { formula: 'Image description and captioning', explanation: 'Generate alt text, describe photos, catalog visual content.' },
          { formula: 'Document/screenshot understanding', explanation: 'Extract text, understand layouts, read charts and diagrams.' },
          { formula: 'Visual question answering', explanation: 'Ask specific questions about image content.' },
          { formula: 'Code from screenshots', explanation: 'Convert UI mockups or screenshots into code.' },
        ]}
        id="example-use-cases"
      />

      <NoteBlock
        type="tip"
        title="Model Selection for Vision"
        content="LLaMA 3.2 Vision 11B offers the best quality but needs ~8GB. LLaVA 7B is a good balance. Moondream (1.8B) is remarkably capable for its size, running on just 2GB -- ideal for edge devices or when you need fast image understanding."
        id="note-model-selection"
      />

      <WarningBlock
        title="Image Size and Performance"
        content="Large images are resized internally but still increase processing time. Pre-resize images to reasonable dimensions (e.g., 768x768) before sending. Very high-resolution images do not improve accuracy and waste compute."
        id="warning-image-size"
      />
    </div>
  )
}
