import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function SystemPrompts() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">System Prompt Design</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        System prompts define the behavior, personality, constraints, and capabilities of an LLM
        application. A well-designed system prompt is the primary lever for controlling model
        behavior in production without fine-tuning.
      </p>

      <DefinitionBlock
        title="System Prompt"
        definition="A special message in the chat format (role: 'system') that provides persistent instructions to the model. Unlike user messages, system prompts are typically hidden from end users and set the overall behavior policy for the conversation."
        id="def-system-prompt"
      />

      <h2 className="text-2xl font-semibold">Anatomy of Effective System Prompts</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Strong system prompts contain several key components: role definition, behavioral
        constraints, output formatting rules, and edge case handling. The order and specificity
        of instructions significantly affect compliance.
      </p>

      <ExampleBlock
        title="System Prompt Structure"
        problem="Design a system prompt for a customer support chatbot for a SaaS product."
        steps={[
          { formula: '\\text{1. Role: You are a helpful support agent for Acme Cloud...}', explanation: 'Establish identity and domain context upfront.' },
          { formula: '\\text{2. Constraints: Only answer questions about Acme products...}', explanation: 'Define boundaries to prevent off-topic usage.' },
          { formula: '\\text{3. Format: Use bullet points, include links to docs...}', explanation: 'Specify output structure for consistency.' },
          { formula: '\\text{4. Escalation: If unsure, say "Let me connect you..."}', explanation: 'Handle uncertainty gracefully with fallback instructions.' },
        ]}
        id="example-system-structure"
      />

      <PythonCode
        title="system_prompt_patterns.py"
        code={`from openai import OpenAI

client = OpenAI()

# Pattern 1: Role-based system prompt
SUPPORT_AGENT = """You are a customer support agent for Acme Cloud Services.

## Your Role
- Help users with billing, account, and technical questions
- Be friendly, professional, and concise

## Constraints
- ONLY answer questions related to Acme Cloud Services
- NEVER share internal pricing formulas or system architecture
- If asked about competitors, politely redirect to Acme features
- If you don't know the answer, say: "Let me connect you with a specialist."

## Response Format
- Use bullet points for multi-step instructions
- Include relevant documentation links as [Doc: topic](https://docs.acme.cloud/topic)
- Keep responses under 200 words unless detailed steps are needed"""

# Pattern 2: Structured output system prompt
JSON_EXTRACTOR = """You are a data extraction assistant. Extract structured information
from user-provided text and return ONLY valid JSON.

## Output Schema
{
  "entities": [{"name": str, "type": str, "confidence": float}],
  "sentiment": "positive" | "negative" | "neutral",
  "key_topics": [str],
  "summary": str (max 50 words)
}

## Rules
- Always return valid JSON, no markdown formatting
- Set confidence between 0.0 and 1.0
- If information is missing, use null
- Never add information not present in the source text"""

# Pattern 3: Persona with guardrails
TUTOR_PROMPT = """You are a patient math tutor for high school students.

## Teaching Approach
- Never give the final answer directly
- Ask guiding questions to help students discover solutions
- Break complex problems into smaller steps
- Celebrate progress and correct mistakes gently

## Boundaries
- Only help with math topics (algebra, geometry, calculus, statistics)
- If asked to do homework for the student, explain that you'll guide them instead
- Redirect non-math questions politely"""

def chat_with_system(system_prompt, user_message, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content

# Test the support agent
print(chat_with_system(
    SUPPORT_AGENT,
    "How do I reset my password?"
))

# Test boundary enforcement
print(chat_with_system(
    SUPPORT_AGENT,
    "What do you think about AWS pricing?"
))`}
        id="code-system-prompts"
      />

      <NoteBlock
        type="tip"
        title="System Prompt Best Practices"
        content="Use markdown headers for organization. Put the most critical instructions first. Use explicit NEVER/ALWAYS directives for hard constraints. Test with adversarial inputs. Version-control your system prompts. Include examples of desired behavior within the prompt when possible."
        id="note-best-practices"
      />

      <WarningBlock
        title="System Prompts Are Not Security Boundaries"
        content="System prompts can be extracted or overridden through prompt injection. Never rely solely on system prompts for security-critical constraints. Implement server-side validation, output filtering, and access controls as defense in depth. Treat system prompt content as potentially visible to users."
        id="warning-not-security"
      />

      <NoteBlock
        type="note"
        title="Temperature and System Prompt Interaction"
        content="Low temperature (0.0-0.3) makes the model follow system prompt instructions more strictly. Higher temperature (0.7-1.0) allows more creative interpretation. For structured output tasks, use temperature 0. For creative tasks, experiment with 0.7-0.9."
        id="note-temperature"
      />
    </div>
  )
}
