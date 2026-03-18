import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function WebSearch() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Web Search Integration</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Open WebUI can augment LLM responses with real-time web search results. When enabled,
        the system searches the web for relevant information and includes it in the context,
        helping models provide up-to-date answers.
      </p>

      <DefinitionBlock
        title="Web Search in Open WebUI"
        definition="Web search integration fetches live search results, scrapes relevant pages, and injects the content into the LLM prompt. This gives local models access to current information beyond their training cutoff date."
        id="def-web-search"
      />

      <PythonCode
        title="Terminal"
        code={`# Configure web search via environment variables
# Supports multiple search engines:

# Option 1: SearXNG (self-hosted, privacy-focused, free)
docker run -d -p 8888:8080 \\
    --name searxng \\
    -v searxng-data:/etc/searxng \\
    searxng/searxng

docker run -d -p 3000:8080 \\
    -e ENABLE_RAG_WEB_SEARCH=true \\
    -e RAG_WEB_SEARCH_ENGINE=searxng \\
    -e SEARXNG_QUERY_URL=http://searxng:8080/search?q=<query>&format=json \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Option 2: Google PSE (requires API key)
# -e RAG_WEB_SEARCH_ENGINE=google_pse
# -e GOOGLE_PSE_API_KEY=your-key
# -e GOOGLE_PSE_ENGINE_ID=your-engine-id

# Option 3: Brave Search API
# -e RAG_WEB_SEARCH_ENGINE=brave
# -e BRAVE_SEARCH_API_KEY=your-key

# Option 4: DuckDuckGo (no API key needed)
# -e RAG_WEB_SEARCH_ENGINE=duckduckgo`}
        id="code-config"
      />

      <PythonCode
        title="docker-compose-search.py"
        code={`# Docker Compose with SearXNG for fully self-hosted web search
compose_config = """
services:
  searxng:
    image: searxng/searxng
    volumes:
      - searxng-data:/etc/searxng
    environment:
      - SEARXNG_BASE_URL=http://searxng:8080/

  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    volumes:
      - open_webui_data:/app/backend/data
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - ENABLE_RAG_WEB_SEARCH=true
      - RAG_WEB_SEARCH_ENGINE=searxng
      - SEARXNG_QUERY_URL=http://searxng:8080/search?q=<query>&format=json
      - RAG_WEB_SEARCH_RESULT_COUNT=5
      - RAG_WEB_SEARCH_CONCURRENT_REQUESTS=5
    depends_on:
      - ollama
      - searxng

volumes:
  ollama_data:
  open_webui_data:
  searxng-data:
"""

with open("docker-compose.yml", "w") as f:
    f.write(compose_config)

print("Created docker-compose.yml with web search support")
print("Run: docker compose up -d")`}
        id="code-compose"
      />

      <ExampleBlock
        title="How Web Search Augmentation Works"
        problem="What happens when a user asks a question with web search enabled?"
        steps={[
          { formula: 'Query reformulation: LLM converts user message to search query', explanation: 'The model extracts key search terms from the conversational input.' },
          { formula: 'Search: query is sent to the configured search engine', explanation: 'Returns top N URLs and snippets.' },
          { formula: 'Scraping: web pages are fetched and text is extracted', explanation: 'HTML is cleaned to plain text, limited to a reasonable length.' },
          { formula: 'Injection: search results are added to the prompt context', explanation: 'The LLM receives both the user question and relevant web content.' },
        ]}
        id="example-flow"
      />

      <NoteBlock
        type="tip"
        title="Toggle Per-Message"
        content="You can enable or disable web search per message using the toggle in the chat input area. This is useful when you only need current information for specific questions, not every message in a conversation."
        id="note-toggle"
      />

      <WarningBlock
        title="Web Search Adds Latency"
        content="Each web search adds 2-5 seconds to response time (search + scraping + processing). The retrieved content also uses context window tokens. Keep RAG_WEB_SEARCH_RESULT_COUNT low (3-5) to balance freshness against speed."
        id="warning-latency"
      />
    </div>
  )
}
