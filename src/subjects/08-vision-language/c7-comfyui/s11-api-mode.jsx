import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function APIMode() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">ComfyUI API Mode</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        ComfyUI exposes a REST API that accepts workflow JSON and returns generated images
        programmatically. This enables integration into production applications, batch
        processing pipelines, and automated testing. The API accepts the same workflow format
        used internally, making it straightforward to convert visual workflows to API calls.
      </p>

      <DefinitionBlock
        title="ComfyUI API"
        definition="The ComfyUI API server (default port 8188) exposes endpoints for queuing workflows (/prompt), checking status (/queue), retrieving results (/history), and uploading images (/upload/image). Workflows are submitted as JSON containing the node graph, and results are retrieved via WebSocket or polling."
        id="def-comfyui-api"
      />

      <h2 className="text-2xl font-semibold">API Workflow</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The typical API flow is: (1) submit a workflow JSON to /prompt, (2) receive a prompt_id,
        (3) monitor progress via WebSocket, (4) retrieve results from /history when complete.
      </p>

      <ExampleBlock
        title="API Request Flow"
        problem="Submit a txt2img workflow and retrieve the result."
        steps={[
          { formula: '\\text{POST /prompt} \\to \\text{\\{prompt\\_id: "abc123"\\}}', explanation: 'Submit workflow JSON, receive tracking ID.' },
          { formula: '\\text{WebSocket /ws?clientId=xyz} \\to \\text{progress events}', explanation: 'Monitor execution progress in real-time.' },
          { formula: '\\text{GET /history/abc123} \\to \\text{output metadata}', explanation: 'Retrieve completion status and output file references.' },
          { formula: '\\text{GET /view?filename=output.png} \\to \\text{image data}', explanation: 'Download the generated image.' },
        ]}
        id="example-api-flow"
      />

      <PythonCode
        title="comfyui_api_client.py"
        code={`import json
import urllib.request
import urllib.parse
import uuid
import io

COMFYUI_URL = "http://127.0.0.1:8188"

class ComfyUIClient:
    """Simple ComfyUI API client."""

    def __init__(self, server_url=COMFYUI_URL):
        self.server = server_url
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, workflow):
        """Submit a workflow for execution."""
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.server}/prompt",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        response = json.loads(urllib.request.urlopen(req).read())
        return response["prompt_id"]

    def get_history(self, prompt_id):
        """Get execution results."""
        url = f"{self.server}/history/{prompt_id}"
        response = json.loads(urllib.request.urlopen(url).read())
        return response.get(prompt_id, {})

    def get_image(self, filename, subfolder="", folder_type="output"):
        """Download a generated image."""
        params = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type,
        })
        url = f"{self.server}/view?{params}"
        return urllib.request.urlopen(url).read()

    def generate(self, workflow, timeout=120):
        """Submit workflow and wait for results."""
        import time

        prompt_id = self.queue_prompt(workflow)
        print(f"Queued: {prompt_id}")

        # Poll for completion
        start = time.time()
        while time.time() - start < timeout:
            history = self.get_history(prompt_id)
            if history:
                outputs = history.get("outputs", {})
                images = []
                for node_output in outputs.values():
                    if "images" in node_output:
                        for img_info in node_output["images"]:
                            img_data = self.get_image(
                                img_info["filename"],
                                img_info.get("subfolder", ""),
                            )
                            images.append(img_data)
                return images
            time.sleep(1)
        raise TimeoutError(f"Generation timed out after {timeout}s")

# Usage example
# client = ComfyUIClient()
# workflow = {...}  # Your workflow JSON
# images = client.generate(workflow)
# with open("output.png", "wb") as f:
#     f.write(images[0])

# Batch generation with parameter sweeps
def parameter_sweep(base_workflow, param_node, param_name, values):
    """Generate images across a parameter sweep."""
    results = []
    for val in values:
        workflow = json.loads(json.dumps(base_workflow))  # Deep copy
        workflow[param_node]["inputs"][param_name] = val
        results.append({"value": val, "workflow": workflow})
    return results

# Example: CFG sweep
sweep = parameter_sweep(
    base_workflow={"8": {"class_type": "KSampler", "inputs": {"cfg": 7.0}}},
    param_node="8", param_name="cfg",
    values=[3.0, 5.0, 7.0, 10.0, 15.0]
)
print(f"Generated {len(sweep)} workflow variants for CFG sweep")`}
        id="code-api-client"
      />

      <NoteBlock
        type="tip"
        title="WebSocket for Real-Time Progress"
        content="For production applications, use WebSocket connections instead of polling. Connect to ws://host:8188/ws?clientId=YOUR_ID to receive real-time progress events including current step, total steps, and preview images. This enables progress bars and live previews in your application."
        id="note-websocket"
      />

      <WarningBlock
        title="API Security"
        content="ComfyUI's API has no built-in authentication. Never expose it directly to the internet. Use a reverse proxy (nginx, Caddy) with authentication for remote access. The API allows arbitrary code execution via custom nodes, so treat it as a privileged endpoint."
        id="warning-security"
      />
    </div>
  )
}
