import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Pipelines() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Custom Pipelines & Filters</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Pipelines in Open WebUI let you intercept and transform messages before they reach the
        model (inlet) or after the model responds (outlet). This enables content filtering,
        logging, custom routing, translation, and advanced processing workflows.
      </p>

      <DefinitionBlock
        title="Pipelines"
        definition="A pipeline is a Python class with inlet (pre-processing) and/or outlet (post-processing) methods that modify the message flow. Inlet filters can modify, reject, or augment user messages. Outlet filters can modify, annotate, or log model responses."
        id="def-pipelines"
      />

      <ExampleBlock
        title="Pipeline Types"
        problem="What kinds of pipelines can you build?"
        steps={[
          { formula: 'Filter pipeline: content moderation, PII removal', explanation: 'Block or redact sensitive content before it reaches the model.' },
          { formula: 'Augmentation pipeline: add context, inject RAG results', explanation: 'Enrich prompts with additional data before generation.' },
          { formula: 'Routing pipeline: choose model based on query', explanation: 'Automatically route to specialized models based on task type.' },
          { formula: 'Logging pipeline: audit, analytics, cost tracking', explanation: 'Log all interactions for compliance or optimization.' },
        ]}
        id="example-types"
      />

      <PythonCode
        title="content_filter_pipeline.py"
        code={`# Content moderation pipeline
# Add via Admin > Workspace > Functions > Add Pipeline

"""
title: Content Filter
description: Filters inappropriate content and PII
author: your-name
version: 0.1.0
"""

import re
from pydantic import BaseModel, Field
from typing import Optional


class Pipeline:
    class Valves(BaseModel):
        block_keywords: str = Field(
            default="password,secret,ssn",
            description="Comma-separated list of blocked keywords"
        )
        redact_emails: bool = Field(
            default=True,
            description="Redact email addresses"
        )

    def __init__(self):
        self.name = "Content Filter"
        self.valves = self.Valves()

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Pre-process user message before sending to LLM."""
        messages = body.get("messages", [])
        blocked_words = [
            w.strip() for w in self.valves.block_keywords.split(",")
        ]

        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"]

                # Check for blocked keywords
                for word in blocked_words:
                    if word.lower() in content.lower():
                        raise Exception(
                            f"Message blocked: contains restricted term"
                        )

                # Redact email addresses
                if self.valves.redact_emails:
                    content = re.sub(
                        r'[\\w.-]+@[\\w.-]+\\.\\w+',
                        '[EMAIL REDACTED]',
                        content
                    )
                msg["content"] = content

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Post-process model response."""
        messages = body.get("messages", [])
        for msg in messages:
            if msg["role"] == "assistant":
                # Add disclaimer to responses
                if "medical" in msg["content"].lower():
                    msg["content"] += (
                        "\\n\\n*Disclaimer: This is not medical advice. "
                        "Consult a healthcare professional.*"
                    )
        return body`}
        id="code-filter"
      />

      <PythonCode
        title="logging_pipeline.py"
        code={`# Logging and analytics pipeline
"""
title: Usage Logger
description: Logs all conversations for analytics
author: your-name
version: 0.1.0
"""

import json
import time
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class Pipeline:
    class Valves(BaseModel):
        log_file: str = Field(
            default="/app/backend/data/usage_log.jsonl",
            description="Path to log file"
        )

    def __init__(self):
        self.name = "Usage Logger"
        self.valves = self.Valves()

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Log incoming requests."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "request",
            "user": user.get("name", "unknown") if user else "unknown",
            "model": body.get("model", "unknown"),
            "message_count": len(body.get("messages", [])),
            "last_message_length": len(
                body.get("messages", [{}])[-1].get("content", "")
            ),
        }

        with open(self.valves.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\\n")

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Log responses."""
        messages = body.get("messages", [])
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]

        if assistant_msgs:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "type": "response",
                "user": user.get("name", "unknown") if user else "unknown",
                "response_length": len(assistant_msgs[-1].get("content", "")),
            }
            with open(self.valves.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\\n")

        return body`}
        id="code-logging"
      />

      <NoteBlock
        type="tip"
        title="Pipeline Chaining"
        content="Multiple pipelines can be active simultaneously and they execute in order. For example: content filter (inlet) -> RAG augmentation (inlet) -> model generates response -> logging (outlet) -> response formatting (outlet). Order matters for correctness."
        id="note-chaining"
      />

      <WarningBlock
        title="Pipeline Errors Block Messages"
        content="If a pipeline raises an exception in the inlet, the message is blocked and the user sees an error. Test pipelines thoroughly before enabling them in production. A buggy filter pipeline can make the entire chat interface unusable."
        id="warning-errors"
      />
    </div>
  )
}
