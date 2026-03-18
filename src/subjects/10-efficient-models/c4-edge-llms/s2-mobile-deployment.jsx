import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function MobileDeployment() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Mobile Deployment: ONNX, Core ML, and Runtime Optimization</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Deploying LLMs on mobile and edge devices requires converting models to optimized
        formats (ONNX, Core ML, TensorFlow Lite) and leveraging hardware-specific acceleration.
        The goal is to achieve acceptable latency (under 50ms per token) within tight memory
        budgets (4-8 GB RAM shared with the OS and other apps).
      </p>

      <DefinitionBlock
        title="ONNX Runtime"
        definition="ONNX (Open Neural Network Exchange) is a hardware-agnostic model format that represents computation as a directed graph. ONNX Runtime (ORT) executes these graphs with optimizations including operator fusion, constant folding, and hardware-specific kernel selection. For LLMs, ORT provides quantized inference, KV cache management, and beam search as built-in operators."
        id="def-onnx"
      />

      <DefinitionBlock
        title="Core ML"
        definition="Core ML is Apple's on-device ML framework optimizing for the Apple Neural Engine (ANE), GPU, and CPU. It supports INT4 weight-only quantization, attention caching, and dynamic shapes. A 3B model quantized to INT4 can run at 15-30 tokens/second on iPhone 15 Pro using the ANE's 35 TOPS of compute."
        id="def-coreml"
      />

      <ExampleBlock
        title="Mobile Memory Budget"
        problem="Determine if a 2.7B model (INT4) can run on a device with 6 GB RAM, 2 GB reserved for OS."
        steps={[
          {
            formula: '\\text{Available RAM} = 6 - 2 = 4\\text{ GB}',
            explanation: 'OS and background apps reserve roughly 2 GB.'
          },
          {
            formula: '\\text{Model weights} = 2.7B \\times 0.5 = 1.35\\text{ GB}',
            explanation: 'INT4 quantization: 0.5 bytes per parameter.'
          },
          {
            formula: '\\text{KV cache (2K ctx)} \\approx 0.15\\text{ GB}',
            explanation: 'KV cache for 2048 tokens with GQA.'
          },
          {
            formula: '\\text{Runtime overhead} \\approx 0.3\\text{ GB}',
            explanation: 'Activations, framework overhead, and intermediate buffers.'
          },
          {
            formula: '\\text{Total} = 1.35 + 0.15 + 0.3 = 1.8\\text{ GB} < 4\\text{ GB} \\checkmark',
            explanation: 'The model fits with 2.2 GB to spare.'
          }
        ]}
        id="example-mobile-memory"
      />

      <PythonCode
        title="onnx_export_and_optimize.py"
        code={`import torch
import torch.nn as nn

# Step 1: Define a small model for demonstration
class TinyLM(nn.Module):
    def __init__(self, vocab=32000, d=512, layers=4, heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads, dim_feedforward=d*4, batch_first=True)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.embed.weight  # Tie weights

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

model = TinyLM()
model.eval()
params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {params:,}")

# Step 2: Export to ONNX
dummy_input = torch.randint(0, 32000, (1, 128))

torch.onnx.export(
    model,
    dummy_input,
    "tiny_lm.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"},
                  "logits": {0: "batch", 1: "seq_len"}},
    opset_version=17,
)
print("Exported to ONNX")

# Step 3: Optimize with ONNX Runtime
# pip install onnxruntime
import onnxruntime as ort

# Create optimized session
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_filepath = "tiny_lm_optimized.onnx"

session = ort.InferenceSession("tiny_lm.onnx", sess_options)

# Benchmark inference
import time
input_np = dummy_input.numpy()
times = []
for _ in range(10):
    start = time.time()
    outputs = session.run(None, {"input_ids": input_np})
    times.append((time.time() - start) * 1000)

print(f"ONNX Runtime: {sum(times)/len(times):.1f}ms avg ({len(times)} runs)")
print(f"Output shape: {outputs[0].shape}")

# Step 4: Quantize ONNX model to INT8
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "tiny_lm.onnx",
    "tiny_lm_int8.onnx",
    weight_type=QuantType.QInt8,
)

import os
original_size = os.path.getsize("tiny_lm.onnx") / 1e6
quant_size = os.path.getsize("tiny_lm_int8.onnx") / 1e6
print(f"\\nOriginal: {original_size:.1f} MB")
print(f"INT8:     {quant_size:.1f} MB ({original_size/quant_size:.1f}x smaller)")`}
        id="code-onnx-deploy"
      />

      <NoteBlock
        type="tip"
        title="Deployment Framework Comparison"
        content="ONNX Runtime: best cross-platform support, works on Windows/Linux/Android/iOS. Core ML: best on Apple devices (ANE acceleration). TensorFlow Lite: good for Android with GPU delegate. llama.cpp: excellent for LLMs specifically, supports many quantization formats, runs on CPU with SIMD acceleration. MLC LLM: compiles models for multiple backends with near-optimal performance."
        id="note-framework-comparison"
      />

      <NoteBlock
        type="note"
        title="Hardware Acceleration on Mobile"
        content="Modern phones have dedicated ML accelerators: Apple Neural Engine (35 TOPS on A17), Qualcomm Hexagon NPU (45 TOPS on Snapdragon 8 Gen 3), and Samsung Exynos NPU. These achieve 5-10x better performance per watt than running on the mobile GPU or CPU. Framework support for these accelerators varies."
        id="note-hardware-accel"
      />

      <WarningBlock
        title="Thermal Throttling"
        content="Mobile devices throttle performance under sustained load to prevent overheating. An LLM generating tokens continuously can trigger throttling within 30-60 seconds, reducing speed by 30-50%. Design for burst inference (short responses) or implement adaptive batch sizing that backs off when thermal state is elevated."
        id="warning-thermal"
      />
    </div>
  )
}
