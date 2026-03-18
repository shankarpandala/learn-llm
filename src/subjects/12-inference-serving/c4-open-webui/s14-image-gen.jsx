import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ImageGen() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Image Generation Integration</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Open WebUI can integrate with image generation backends like AUTOMATIC1111, ComfyUI,
        and OpenAI's DALL-E. Users can generate images directly in chat conversations, combining
        text and visual AI capabilities in one interface.
      </p>

      <DefinitionBlock
        title="Image Generation in Open WebUI"
        definition="Open WebUI supports multiple image generation backends. When configured, users can request images in chat and the system routes the request to the appropriate backend, displays the generated image inline, and optionally stores it for later reference."
        id="def-image-gen"
      />

      <PythonCode
        title="Terminal"
        code={`# Option 1: AUTOMATIC1111 (Stable Diffusion WebUI)
# Start A1111 with API enabled:
python launch.py --api --listen

# Configure in Open WebUI:
docker run -d -p 3000:8080 \\
    -e IMAGE_GENERATION_ENGINE=automatic1111 \\
    -e AUTOMATIC1111_BASE_URL=http://host.docker.internal:7860 \\
    -e ENABLE_IMAGE_GENERATION=true \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Option 2: ComfyUI
# Start ComfyUI server, then configure:
# -e IMAGE_GENERATION_ENGINE=comfyui
# -e COMFYUI_BASE_URL=http://host.docker.internal:8188

# Option 3: OpenAI DALL-E
# Uses the OpenAI API key already configured
# -e IMAGE_GENERATION_ENGINE=openai
# -e IMAGE_GENERATION_MODEL=dall-e-3

# In the chat, type a message like:
# "Generate an image of a sunset over mountains"
# Or use /image command if configured`}
        id="code-config"
      />

      <PythonCode
        title="image_gen_api.py"
        code={`import requests
import base64
from pathlib import Path

BASE_URL = "http://localhost:3000"
HEADERS = {"Authorization": "Bearer YOUR_TOKEN"}

# Generate an image through Open WebUI's API
resp = requests.post(
    f"{BASE_URL}/api/v1/images/generations",
    headers={**HEADERS, "Content-Type": "application/json"},
    json={
        "prompt": "A serene mountain landscape at sunset, photorealistic",
        "n": 1,
        "size": "512x512",
    },
)

if resp.status_code == 200:
    data = resp.json()
    for i, image in enumerate(data.get("data", [])):
        if "b64_json" in image:
            img_bytes = base64.b64decode(image["b64_json"])
            Path(f"generated_{i}.png").write_bytes(img_bytes)
            print(f"Saved generated_{i}.png")
        elif "url" in image:
            print(f"Image URL: {image['url']}")
else:
    print(f"Error: {resp.status_code} - {resp.text}")

# Using AUTOMATIC1111 directly for more control
A1111_URL = "http://localhost:7860"
resp = requests.post(f"{A1111_URL}/sdapi/v1/txt2img", json={
    "prompt": "a cute robot reading a book, digital art",
    "negative_prompt": "blurry, low quality",
    "steps": 30,
    "width": 512,
    "height": 512,
    "cfg_scale": 7.5,
    "sampler_name": "DPM++ 2M Karras",
})
if resp.status_code == 200:
    images = resp.json().get("images", [])
    for i, img_b64 in enumerate(images):
        Path(f"sd_image_{i}.png").write_bytes(base64.b64decode(img_b64))
        print(f"Saved sd_image_{i}.png")`}
        id="code-api"
      />

      <ExampleBlock
        title="Image Generation Backends"
        problem="Compare the available image generation backends."
        steps={[
          { formula: 'AUTOMATIC1111: full Stable Diffusion control, many models', explanation: 'Most flexible. Supports LoRAs, ControlNet, inpainting. Requires GPU.' },
          { formula: 'ComfyUI: node-based workflows, advanced pipelines', explanation: 'Most powerful for complex generation pipelines. Steeper learning curve.' },
          { formula: 'DALL-E 3: highest quality, no local GPU needed', explanation: 'OpenAI cloud API. Best quality but costs per image.' },
        ]}
        id="example-backends"
      />

      <NoteBlock
        type="tip"
        title="LLM-Enhanced Prompts"
        content="Open WebUI can use the LLM to enhance image generation prompts. A simple request like 'draw a cat' gets expanded into a detailed prompt with style, lighting, and composition details before being sent to the image generator."
        id="note-prompt-enhancement"
      />

      <WarningBlock
        title="GPU Memory Sharing"
        content="Running both an LLM (via Ollama) and Stable Diffusion on the same GPU requires careful memory management. A 7B LLM uses ~5GB and SD 1.5 uses ~4GB -- tight for an 8GB GPU. Consider using smaller LLMs or offloading SD to CPU for generation."
        id="warning-gpu-memory"
      />
    </div>
  )
}
