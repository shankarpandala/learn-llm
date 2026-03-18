import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function ResponsibleDisclosure() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Responsible Disclosure</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        When researchers or users discover vulnerabilities in AI systems, responsible disclosure
        ensures that findings are reported constructively to enable fixes before public exposure.
        This practice, adapted from cybersecurity, is increasingly important for AI safety.
      </p>

      <DefinitionBlock
        title="Responsible Disclosure"
        definition="A vulnerability reporting process where the discoverer privately notifies the system owner, provides reasonable time for remediation, and coordinates public disclosure timing. In the AI context, this extends to model vulnerabilities, safety bypasses, and harmful capabilities."
        id="def-responsible-disclosure"
      />

      <DefinitionBlock
        title="Coordinated Vulnerability Disclosure (CVD)"
        definition="A structured process involving the reporter, the vendor, and optionally a coordinator (like CERT). It establishes timelines, communication channels, and expectations for both parties. Standard practice allows 90 days for remediation before public disclosure."
        id="def-cvd"
      />

      <h2 className="text-2xl font-semibold">The AI Disclosure Process</h2>
      <p className="text-gray-700 dark:text-gray-300">
        AI vulnerability disclosure differs from traditional software bugs because "patches"
        may require retraining or fundamental architectural changes, and because vulnerabilities
        (like jailbreaks) can affect all users simultaneously.
      </p>

      <ExampleBlock
        title="AI Vulnerability Disclosure Workflow"
        problem="You discover a reproducible jailbreak that bypasses safety training on a major LLM."
        steps={[
          { formula: '\\text{1. Document: Reproduce, record steps, assess impact}', explanation: 'Create a clear, minimal reproduction case. Classify severity (information leak vs. harmful content generation).' },
          { formula: '\\text{2. Report: Use vendor security channel}', explanation: 'Contact the AI provider via their bug bounty program, security@, or responsible disclosure policy.' },
          { formula: '\\text{3. Collaborate: Work with the team on verification}', explanation: 'Provide additional details if requested. Help verify that proposed fixes actually address the issue.' },
          { formula: '\\text{4. Disclosure: Coordinate timing of public write-up}', explanation: 'Agree on a disclosure timeline. Publish findings after the fix, omitting exploitation details that could cause harm.' },
        ]}
        id="example-workflow"
      />

      <PythonCode
        title="vulnerability_report_template.py"
        code={`# Template for structured AI vulnerability reporting
# Use this when filing responsible disclosure reports

from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class AIVulnerabilityReport:
    """Structured vulnerability report for AI systems."""

    # Identification
    reporter: str
    report_date: str = field(default_factory=lambda: datetime.now().isoformat())
    system_affected: str = ""
    model_version: str = ""

    # Classification
    vulnerability_type: str = ""  # jailbreak, data_leak, bias, safety_bypass
    severity: str = ""  # critical, high, medium, low
    reproducibility: str = ""  # always, sometimes, rare

    # Details
    description: str = ""
    reproduction_steps: list[str] = field(default_factory=list)
    expected_behavior: str = ""
    actual_behavior: str = ""

    # Impact assessment
    affected_users: str = ""  # all, specific_config, edge_case
    potential_harm: str = ""
    mitigating_factors: str = ""

    # Evidence (redacted for safety)
    sample_prompts: list[str] = field(default_factory=list)
    sample_outputs_redacted: list[str] = field(default_factory=list)

    def to_report(self) -> str:
        """Generate formatted disclosure report."""
        sections = [
            f"# AI Vulnerability Report",
            f"**Date:** {self.report_date}",
            f"**Reporter:** {self.reporter}",
            f"**System:** {self.system_affected} ({self.model_version})",
            f"",
            f"## Classification",
            f"- **Type:** {self.vulnerability_type}",
            f"- **Severity:** {self.severity}",
            f"- **Reproducibility:** {self.reproducibility}",
            f"",
            f"## Description",
            f"{self.description}",
            f"",
            f"## Reproduction Steps",
        ]
        for i, step in enumerate(self.reproduction_steps, 1):
            sections.append(f"{i}. {step}")
        sections.extend([
            f"",
            f"## Impact",
            f"- **Affected users:** {self.affected_users}",
            f"- **Potential harm:** {self.potential_harm}",
            f"- **Mitigating factors:** {self.mitigating_factors}",
        ])
        return "\\n".join(sections)

# Example report (with hypothetical, benign content)
report = AIVulnerabilityReport(
    reporter="Security Researcher",
    system_affected="Example LLM Service",
    model_version="v2.1",
    vulnerability_type="safety_bypass",
    severity="medium",
    reproducibility="always",
    description="The model can be induced to ignore content policy via [technique].",
    reproduction_steps=[
        "Send a standard prompt to establish context",
        "Apply [specific technique] in follow-up message",
        "Model responds without applying safety guidelines",
    ],
    expected_behavior="Model should refuse or redirect",
    actual_behavior="Model complies with restricted request",
    affected_users="all",
    potential_harm="Generation of policy-violating content",
    mitigating_factors="Requires specific multi-turn interaction",
)

print(report.to_report())

# Major AI labs' disclosure channels (as of 2025)
DISCLOSURE_CHANNELS = {
    "OpenAI": "https://openai.com/security/disclosure",
    "Anthropic": "https://www.anthropic.com/responsible-disclosure",
    "Google DeepMind": "https://bughunters.google.com",
    "Meta AI": "https://www.facebook.com/whitehat",
    "Microsoft": "https://msrc.microsoft.com/report",
}`}
        id="code-disclosure"
      />

      <NoteBlock
        type="note"
        title="Bug Bounty Programs for AI"
        content="Major AI companies now offer bug bounty programs specifically for AI safety vulnerabilities. OpenAI's program (via Bugcrowd) covers API abuse, data exposure, and safety bypasses. Anthropic and Google have similar programs. Payouts for critical AI safety findings can range from $2,000 to $20,000+."
        id="note-bug-bounty"
      />

      <WarningBlock
        title="When NOT to Publicly Disclose"
        content="Never publicly share: working exploits for harmful content generation, techniques that could enable real-world harm (weapons, CSAM, etc.), zero-day vulnerabilities before vendor notification, or private system prompts that reveal security architecture. The goal of research is to improve safety, not to demonstrate harm."
        id="warning-when-not"
      />

      <NoteBlock
        type="tip"
        title="Building a Disclosure Culture"
        content="If you build AI systems, create a clear disclosure policy, respond to reports within 48 hours, credit reporters publicly (with permission), and fix critical issues within 30 days. The ML commons community and AI Incident Database (incidentdatabase.ai) are valuable resources for tracking and learning from AI safety incidents."
        id="note-culture"
      />
    </div>
  )
}
