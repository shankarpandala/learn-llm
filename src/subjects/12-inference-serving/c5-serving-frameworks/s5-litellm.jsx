import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function LiteLLM() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">LiteLLM: Unified LLM Proxy</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        LiteLLM provides a unified OpenAI-compatible interface to over 100 LLM providers. It
        acts as a proxy layer that translates between the OpenAI API format and each provider's
        native API. This means you write your code once and can switch between OpenAI, Anthropic,
        local models, and dozens of other providers by changing a single model string.
      </p>

      <DefinitionBlock
        title="LiteLLM"
        definition="LiteLLM is an open-source proxy server and Python SDK that provides a unified OpenAI-format API for 100+ LLM providers including OpenAI, Anthropic, Cohere, vLLM, Ollama, Azure, Bedrock, and Vertex AI. It handles authentication, retries, fallbacks, load balancing, spend tracking, and rate limiting."
        id="def-litellm"
      />

      <PythonCode
        title="Terminal"
        code={`# Install LiteLLM
pip install 'litellm[proxy]'

# Start the proxy server with a config file
cat > litellm_config.yaml << 'EOF'
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-your-openai-key

  - model_name: claude
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: sk-ant-your-key

  - model_name: local-llama
    litellm_params:
      model: openai/meta-llama/Llama-3.1-8B-Instruct
      api_base: http://localhost:8000/v1
      api_key: none

  - model_name: ollama-model
    litellm_params:
      model: ollama/llama3.2
      api_base: http://localhost:11434

  # Load balancing: multiple deployments for one model name
  - model_name: fast-model
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: sk-key1
  - model_name: fast-model
    litellm_params:
      model: anthropic/claude-sonnet-4-20250514
      api_key: sk-ant-key2

litellm_settings:
  drop_params: true
  set_verbose: false

general_settings:
  master_key: sk-litellm-master-key
EOF

litellm --config litellm_config.yaml --port 4000

# Docker deployment
docker run -p 4000:4000 \\
    -v $(pwd)/litellm_config.yaml:/app/config.yaml \\
    ghcr.io/berriai/litellm:main-latest \\
    --config /app/config.yaml`}
        id="code-litellm-setup"
      />

      <PythonCode
        title="litellm_client.py"
        code={`from openai import OpenAI
import litellm

# Option 1: Use as a proxy (any OpenAI SDK client works)
client = OpenAI(
    base_url="http://localhost:4000",
    api_key="sk-litellm-master-key",
)

# Route to different backends by model name
for model in ["gpt-4", "claude", "local-llama"]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10,
    )
    print(f"{model}: {resp.choices[0].message.content}")

# Option 2: Use as a Python SDK (no proxy needed)
response = litellm.completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "What is LiteLLM?"}],
    max_tokens=100,
)
print(response.choices[0].message.content)

# Fallback chain: try providers in order
response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    fallbacks=["anthropic/claude-sonnet-4-20250514", "ollama/llama3.2"],
    max_tokens=50,
)
print(response.choices[0].message.content)

# Spend tracking
from litellm import budget_manager
cost = litellm.completion_cost(completion_response=response)
print(f"Request cost: USD {cost:.6f}")`}
        id="code-litellm-client"
      />

      <ExampleBlock
        title="LiteLLM Key Features"
        problem="What problems does LiteLLM solve?"
        steps={[
          { formula: 'Unified API: one interface for 100+ providers', explanation: 'Write code once, switch providers by changing the model string.' },
          { formula: 'Load balancing: distribute across deployments', explanation: 'Multiple backends for the same model name; LiteLLM round-robins.' },
          { formula: 'Fallbacks: automatic failover on errors', explanation: 'If one provider fails, transparently retry with another.' },
          { formula: 'Spend tracking: per-key and per-user budgets', explanation: 'Track costs across all providers in one dashboard.' },
          { formula: 'Virtual keys: issue API keys with limits', explanation: 'Create team/user keys with rate limits and budget caps.' },
        ]}
        id="example-features"
      />

      <NoteBlock
        type="tip"
        title="Virtual Keys for Teams"
        content="LiteLLM can issue virtual API keys with per-key rate limits, model access controls, and budget limits. This lets you give each team or application its own key while routing through your centralized proxy. Generate keys via the /key/generate API endpoint."
        id="note-virtual-keys"
      />

      <WarningBlock
        title="Parameter Translation"
        content="Not all parameters translate perfectly between providers. For example, tool calling syntax differs between OpenAI and Anthropic. Enable drop_params: true in config to silently drop unsupported parameters rather than erroring. Test critical features with each backend before relying on automatic translation."
        id="warning-params"
      />
    </div>
  )
}
