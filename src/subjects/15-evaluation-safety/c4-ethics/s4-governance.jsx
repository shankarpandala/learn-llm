import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Governance() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">AI Governance and Regulation</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        As AI systems become more capable and widely deployed, governments and international
        bodies are developing regulatory frameworks. Understanding the governance landscape
        is essential for anyone building or deploying LLM-based systems.
      </p>

      <DefinitionBlock
        title="AI Governance"
        definition="The set of rules, practices, frameworks, and institutions that guide the development, deployment, and use of AI systems. It encompasses technical standards, legal regulations, industry self-governance, and international coordination."
        id="def-governance"
      />

      <DefinitionBlock
        title="EU AI Act"
        definition="The world's first comprehensive AI regulation (adopted 2024), establishing a risk-based framework. AI systems are classified as minimal, limited, high, or unacceptable risk, with escalating requirements for transparency, testing, and human oversight."
        id="def-eu-ai-act"
      />

      <h2 className="text-2xl font-semibold">Global Regulatory Landscape</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Different jurisdictions take different approaches. The EU favors comprehensive
        regulation, the US relies more on sector-specific guidance and executive orders,
        and China has implemented targeted regulations for specific AI applications
        (deepfakes, recommendation algorithms, generative AI).
      </p>

      <ExampleBlock
        title="EU AI Act Risk Classification"
        problem="Classify common LLM applications under the EU AI Act risk framework."
        steps={[
          { formula: '\\text{Unacceptable: Social scoring, real-time biometric ID}', explanation: 'Banned outright. Includes manipulation of vulnerable groups and untargeted scraping for facial recognition.' },
          { formula: '\\text{High risk: Hiring tools, credit scoring, education}', explanation: 'Requires conformity assessments, risk management, human oversight, and documentation. Most enterprise LLM uses may fall here.' },
          { formula: '\\text{Limited risk: Chatbots, deepfake generators}', explanation: 'Transparency obligations: users must be informed they are interacting with AI.' },
          { formula: '\\text{Minimal risk: Spam filters, AI in games}', explanation: 'No specific requirements, though general-purpose AI models have separate obligations.' },
        ]}
        id="example-risk-classification"
      />

      <PythonCode
        title="compliance_checklist.py"
        code={`# AI governance compliance checklist and documentation framework

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class RiskLevel(Enum):
    MINIMAL = "minimal"
    LIMITED = "limited"
    HIGH = "high"
    UNACCEPTABLE = "unacceptable"

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    IN_PROGRESS = "in_progress"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "n/a"

@dataclass
class ModelCard:
    """Structured model documentation for governance compliance."""
    model_name: str
    version: str
    developer: str
    release_date: str

    # Intended use
    primary_use: str = ""
    out_of_scope_uses: list[str] = field(default_factory=list)
    target_users: str = ""

    # Risk classification
    risk_level: RiskLevel = RiskLevel.MINIMAL
    risk_justification: str = ""

    # Performance and limitations
    eval_metrics: dict = field(default_factory=dict)
    known_limitations: list[str] = field(default_factory=list)
    bias_evaluation: str = ""

    # Training data
    training_data_description: str = ""
    data_licenses: list[str] = field(default_factory=list)

    # Environmental impact
    training_compute: str = ""
    estimated_co2: str = ""

    def generate_card(self) -> str:
        lines = [
            f"# Model Card: {self.model_name} v{self.version}",
            f"Developer: {self.developer} | Released: {self.release_date}",
            f"\\n## Intended Use\\n{self.primary_use}",
            f"\\n## Risk Level: {self.risk_level.value}\\n{self.risk_justification}",
            f"\\n## Limitations",
        ]
        for lim in self.known_limitations:
            lines.append(f"- {lim}")
        lines.append(f"\\n## Bias Evaluation\\n{self.bias_evaluation}")
        lines.append(f"\\n## Environmental Impact")
        lines.append(f"Compute: {self.training_compute}")
        lines.append(f"Estimated CO2: {self.estimated_co2}")
        return "\\n".join(lines)

@dataclass
class ComplianceChecklist:
    """Regulatory compliance checklist for AI deployment."""

    requirements: dict[str, ComplianceStatus] = field(default_factory=dict)

    def __post_init__(self):
        # EU AI Act high-risk requirements
        eu_requirements = {
            "risk_management_system": ComplianceStatus.IN_PROGRESS,
            "data_governance": ComplianceStatus.IN_PROGRESS,
            "technical_documentation": ComplianceStatus.IN_PROGRESS,
            "record_keeping": ComplianceStatus.IN_PROGRESS,
            "transparency_to_users": ComplianceStatus.IN_PROGRESS,
            "human_oversight_measures": ComplianceStatus.IN_PROGRESS,
            "accuracy_robustness_security": ComplianceStatus.IN_PROGRESS,
            "bias_testing_completed": ComplianceStatus.IN_PROGRESS,
            "post_market_monitoring": ComplianceStatus.IN_PROGRESS,
            "incident_reporting_process": ComplianceStatus.IN_PROGRESS,
        }
        self.requirements.update(eu_requirements)

    def update(self, requirement: str, status: ComplianceStatus):
        self.requirements[requirement] = status

    def report(self) -> str:
        lines = ["=== Compliance Status Report ==="]
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        for req, status in self.requirements.items():
            icon = {"compliant": "[OK]", "in_progress": "[..]",
                    "non_compliant": "[!!]", "n/a": "[--]"}
            lines.append(f"  {icon[status.value]} {req}: {status.value}")
        total = len(self.requirements)
        done = sum(1 for s in self.requirements.values()
                   if s == ComplianceStatus.COMPLIANT)
        lines.append(f"\\nProgress: {done}/{total} ({100*done/total:.0f}%)")
        return "\\n".join(lines)

# Example usage
card = ModelCard(
    model_name="AcmeBot",
    version="2.0",
    developer="Acme Corp",
    release_date="2025-01-15",
    primary_use="Customer support chatbot for SaaS products",
    risk_level=RiskLevel.LIMITED,
    risk_justification="User-facing chatbot requires transparency obligations",
    known_limitations=["English only", "May hallucinate product features",
                       "Not suitable for legal or medical advice"],
    bias_evaluation="Tested across 6 demographic groups, DI ratio > 0.85",
    training_compute="256 H100 GPUs for 14 days",
    estimated_co2="~12 tonnes CO2eq",
)
print(card.generate_card())

checklist = ComplianceChecklist()
checklist.update("transparency_to_users", ComplianceStatus.COMPLIANT)
checklist.update("technical_documentation", ComplianceStatus.COMPLIANT)
print("\\n" + checklist.report())`}
        id="code-governance"
      />

      <NoteBlock
        type="note"
        title="Key Regulatory Frameworks"
        content="Major frameworks include: EU AI Act (comprehensive risk-based regulation), US Executive Order 14110 on AI Safety (reporting requirements for frontier models), China's Interim Measures for Generative AI, the UK's pro-innovation approach through sector regulators, and the OECD AI Principles (non-binding international guidelines)."
        id="note-frameworks"
      />

      <WarningBlock
        title="Compliance Is Not Optional"
        content="The EU AI Act carries penalties of up to 35 million euros or 7% of global revenue for violations. Even in less regulated jurisdictions, failure to address AI risks can result in product liability claims, reputational damage, and loss of user trust. Build governance into your development process from the start."
        id="warning-penalties"
      />

      <NoteBlock
        type="tip"
        title="Practical Governance Steps"
        content="Start with: (1) create model cards documenting capabilities and limitations, (2) implement bias testing in your CI/CD pipeline, (3) establish an incident response process, (4) maintain an audit trail of training data and model versions, (5) provide clear AI disclosure to users, and (6) designate a responsible AI officer or team."
        id="note-practical"
      />

    </div>
  )
}
