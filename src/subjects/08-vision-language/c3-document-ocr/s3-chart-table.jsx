import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ChartTableUnderstanding() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Chart and Table Understanding</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Charts and tables encode data visually in structured formats that require specialized
        understanding beyond basic OCR. Models must interpret axes, legends, data relationships,
        cell structures, and spans. This section covers approaches for extracting structured
        data from visual charts and tables in documents.
      </p>

      <DefinitionBlock
        title="Chart Understanding"
        definition="Chart understanding involves extracting the underlying data table from a chart image, identifying chart type (bar, line, pie, scatter), reading axis labels and values, and answering questions about trends and comparisons. This requires both visual perception and numerical reasoning."
        id="def-chart-understanding"
      />

      <h2 className="text-2xl font-semibold">Table Structure Recognition</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Table structure recognition (TSR) detects the grid structure of tables including rows,
        columns, merged cells, and headers. The task is often decomposed into cell detection
        and relationship classification.
      </p>

      <ExampleBlock
        title="Table Structure Recognition Pipeline"
        problem="Extract a 3x4 table with one merged header from a document image."
        steps={[
          { formula: '\\text{Step 1: Detect table region in the document}', explanation: 'Use an object detector (DETR, YOLO) to localize the table bounding box.' },
          { formula: '\\text{Step 2: Identify rows and columns}', explanation: 'Detect horizontal and vertical separators to establish the grid structure.' },
          { formula: '\\text{Step 3: Detect merged cells (spans)}', explanation: 'Classify cell relationships to find colspan/rowspan attributes.' },
          { formula: '\\text{Step 4: Extract cell content via OCR or VLM}', explanation: 'Read text from each detected cell region.' },
        ]}
        id="example-table-pipeline"
      />

      <PythonCode
        title="chart_table_extraction.py"
        code={`# Table detection and extraction using transformers
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from PIL import Image
import torch

# Microsoft Table Transformer for table detection
# processor = AutoImageProcessor.from_pretrained(
#     "microsoft/table-transformer-detection"
# )
# model = TableTransformerForObjectDetection.from_pretrained(
#     "microsoft/table-transformer-detection"
# )

# image = Image.open("document.png")
# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)

# # Post-process detections
# target_sizes = torch.tensor([image.size[::-1]])
# results = processor.post_process_object_detection(
#     outputs, threshold=0.7, target_sizes=target_sizes
# )[0]

# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     print(f"Table detected: score={score:.3f}, box={box.tolist()}")

# Chart QA with Pix2Struct
# from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
# chart_model = Pix2StructForConditionalGeneration.from_pretrained(
#     "google/deplot"  # DePlot: chart -> data table
# )
# chart_processor = Pix2StructProcessor.from_pretrained("google/deplot")

# Simulated chart data extraction
def parse_chart_description(chart_type, data_points):
    """Convert extracted chart data to structured format."""
    result = {
        "chart_type": chart_type,
        "data": data_points,
        "summary": {}
    }

    values = [d["value"] for d in data_points]
    result["summary"] = {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "trend": "increasing" if values[-1] > values[0] else "decreasing"
    }
    return result

# Example: bar chart extraction
chart_data = parse_chart_description("bar", [
    {"label": "Q1", "value": 150},
    {"label": "Q2", "value": 230},
    {"label": "Q3", "value": 180},
    {"label": "Q4", "value": 310},
])
print(f"Chart type: {chart_data['chart_type']}")
print(f"Data: {chart_data['data']}")
print(f"Summary: {chart_data['summary']}")`}
        id="code-chart-table"
      />

      <NoteBlock
        type="note"
        title="DePlot and ChartQA"
        content="Google's DePlot converts chart images directly to linearized data tables, which can then be processed by LLMs for question answering. ChartQA benchmarks show that combining a chart-to-table model with an LLM reasoning module significantly outperforms end-to-end approaches."
        id="note-deplot"
      />

      <WarningBlock
        title="Numerical Precision"
        content="Chart understanding models often struggle with precise numerical reading, especially for values between grid lines or in dense charts. Always validate extracted numbers against any available textual annotations in the chart, and consider using OCR on axis labels as a cross-check."
        id="warning-precision"
      />
    </div>
  )
}
