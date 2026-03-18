import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Guardrails() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Input/Output Guardrails</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Guardrails are programmatic safety layers that validate, filter, and constrain LLM
        inputs and outputs. They provide defense-in-depth beyond model-level alignment,
        catching failures at the application boundary.
      </p>

      <DefinitionBlock
        title="Guardrails"
        definition="Programmable constraints applied to LLM inputs (pre-processing) and outputs (post-processing) that enforce safety policies, format requirements, and content restrictions. They operate as middleware between the user and the model."
        id="def-guardrails"
      />

      <h2 className="text-2xl font-semibold">Types of Guardrails</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Guardrails can be categorized by their mechanism: rule-based (regex, keyword matching),
        ML-based (classifiers for toxicity, PII detection), and LLM-based (using another model
        to evaluate safety). The false positive rate is a critical metric:
      </p>
      <BlockMath math="\text{FPR} = \frac{\text{benign inputs blocked}}{\text{total benign inputs}}" />
      <p className="text-gray-700 dark:text-gray-300">
        A guardrail system must balance safety (low false negative rate) with usability (low
        false positive rate).
      </p>

      <ExampleBlock
        title="Guardrail Pipeline Design"
        problem="Design a complete guardrail system for a production chatbot."
        steps={[
          { formula: '\\text{Input: rate limit} \\to \\text{PII detection} \\to \\text{topic filter}', explanation: 'Pre-processing guardrails screen user input before it reaches the model.' },
          { formula: '\\text{Model: system prompt + constrained generation}', explanation: 'The LLM processes the screened input with safety-aligned system prompt.' },
          { formula: '\\text{Output: toxicity check} \\to \\text{hallucination filter} \\to \\text{format validation}', explanation: 'Post-processing guardrails verify model output before showing to user.' },
          { formula: '\\text{Fallback: safe default response if any check fails}', explanation: 'Graceful degradation when guardrails trigger.' },
        ]}
        id="example-pipeline"
      />

      <PythonCode
        title="guardrails_implementation.py"
        code={`# Production guardrails implementation
import re
from dataclasses import dataclass

@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""
    modified_text: str = None

# --- Input Guardrails ---

class PiiDetector:
    """Detect and redact personally identifiable information."""

    PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    }

    def detect(self, text: str) -> GuardrailResult:
        found = []
        redacted = text
        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                found.append(f"{pii_type}: {len(matches)} found")
                redacted = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted)
        if found:
            return GuardrailResult(False, f"PII detected: {', '.join(found)}", redacted)
        return GuardrailResult(True)

class TopicFilter:
    """Block off-topic or restricted queries."""

    def __init__(self, allowed_topics: list[str], blocked_keywords: list[str]):
        self.allowed_topics = allowed_topics
        self.blocked_keywords = [kw.lower() for kw in blocked_keywords]

    def check(self, text: str) -> GuardrailResult:
        lower_text = text.lower()
        for keyword in self.blocked_keywords:
            if keyword in lower_text:
                return GuardrailResult(False, f"Blocked keyword: {keyword}")
        return GuardrailResult(True)

# --- Output Guardrails ---

class OutputValidator:
    """Validate model output meets requirements."""

    def __init__(self, max_length: int = 2000, required_disclaimer: str = None):
        self.max_length = max_length
        self.required_disclaimer = required_disclaimer

    def validate(self, output: str) -> GuardrailResult:
        if len(output) > self.max_length:
            truncated = output[:self.max_length] + "... [truncated]"
            return GuardrailResult(False, "Output too long", truncated)
        if self.required_disclaimer and self.required_disclaimer not in output:
            modified = output + f"\\n\\n{self.required_disclaimer}"
            return GuardrailResult(False, "Missing disclaimer", modified)
        return GuardrailResult(True)

# --- Guardrails Pipeline ---

class GuardrailsPipeline:
    """Complete input/output guardrails pipeline."""

    def __init__(self, model_fn):
        self.model_fn = model_fn
        self.pii_detector = PiiDetector()
        self.topic_filter = TopicFilter(
            allowed_topics=["product", "support", "billing"],
            blocked_keywords=["competitor_name", "internal_secret"],
        )
        self.output_validator = OutputValidator(
            max_length=2000,
            required_disclaimer="Note: I'm an AI assistant.",
        )
        self.fallback = "I'm sorry, I can't help with that. Please contact support."

    def process(self, user_input: str) -> str:
        # Input guardrails
        pii_result = self.pii_detector.detect(user_input)
        if not pii_result.passed:
            user_input = pii_result.modified_text  # Use redacted version

        topic_result = self.topic_filter.check(user_input)
        if not topic_result.passed:
            return self.fallback

        # Model call
        output = self.model_fn(user_input)

        # Output guardrails
        output_result = self.output_validator.validate(output)
        if not output_result.passed and output_result.modified_text:
            return output_result.modified_text

        return output

# Usage
pipeline = GuardrailsPipeline(model_fn=lambda x: f"Response to: {x}")
print(pipeline.process("My email is test@example.com, help with billing"))
print(pipeline.process("Tell me about competitor_name pricing"))`}
        id="code-guardrails"
      />

      <NoteBlock
        type="tip"
        title="Guardrails AI Framework"
        content="The Guardrails AI library (guardrails-ai) provides pre-built validators for common patterns: JSON schema validation, PII detection, toxicity filtering, and custom validators. NVIDIA NeMo Guardrails offers dialog-level safety management with programmable rails for topic control and safety."
        id="note-guardrails-ai"
      />

      <WarningBlock
        title="Guardrails Have Overhead"
        content="Each guardrail layer adds latency and potential failure modes. PII regex patterns can produce false positives (blocking valid addresses). ML-based classifiers add inference time. Design guardrails with clear bypass conditions and monitoring to detect when they are too aggressive or too permissive."
        id="warning-overhead"
      />

      <NoteBlock
        type="note"
        title="Monitoring and Iteration"
        content="Log all guardrail triggers with enough context for review. Track false positive rates by category. Set up alerts for unusual trigger volumes (may indicate attacks). Regularly review and update rules as new attack patterns emerge and as your application's requirements evolve."
        id="note-monitoring"
      />
    </div>
  )
}
