import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function RAG() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">RAG in Open WebUI</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Open WebUI has built-in Retrieval-Augmented Generation (RAG) capabilities. Upload documents,
        and the system automatically chunks, embeds, and retrieves relevant passages to augment
        the LLM's responses with your specific knowledge.
      </p>

      <DefinitionBlock
        title="RAG in Open WebUI"
        definition="Open WebUI's RAG pipeline: (1) upload documents (PDF, TXT, DOCX, etc.), (2) automatic chunking and embedding using a local or remote embedding model, (3) storage in a built-in vector database (ChromaDB), (4) at query time, relevant chunks are retrieved and injected into the prompt."
        id="def-rag"
      />

      <ExampleBlock
        title="RAG Workflow"
        problem="How does document Q&A work in Open WebUI?"
        steps={[
          { formula: 'Upload: drag PDF/DOCX into chat or use + button', explanation: 'Documents are processed and stored automatically.' },
          { formula: 'Chunk: documents are split into ~500 token chunks', explanation: 'Overlapping chunks ensure context is not lost at boundaries.' },
          { formula: 'Embed: chunks are converted to vectors via embedding model', explanation: 'Uses Ollama embedding model or OpenAI embeddings.' },
          { formula: 'Retrieve: user query is embedded and matched against chunks', explanation: 'Top-k most similar chunks are retrieved (default k=4).' },
          { formula: 'Augment: retrieved chunks are prepended to the prompt', explanation: 'The LLM sees relevant context and generates a grounded answer.' },
        ]}
        id="example-workflow"
      />

      <PythonCode
        title="Terminal"
        code={`# Configure RAG settings via environment variables
docker run -d -p 3000:8080 \\
    -e RAG_EMBEDDING_MODEL=nomic-embed-text \\
    -e RAG_EMBEDDING_ENGINE=ollama \\
    -e CHUNK_SIZE=500 \\
    -e CHUNK_OVERLAP=50 \\
    -e RAG_TOP_K=4 \\
    -e RAG_RELEVANCE_THRESHOLD=0.3 \\
    -v open-webui:/app/backend/data \\
    --name open-webui \\
    ghcr.io/open-webui/open-webui:main

# Make sure the embedding model is available in Ollama
ollama pull nomic-embed-text

# In the UI:
# 1. Click the + button in chat to upload a document
# 2. Or go to Workspace > Documents to manage a knowledge base
# 3. Create collections to group related documents
# 4. Reference a collection with # in chat: "#my-collection"`}
        id="code-config"
      />

      <PythonCode
        title="rag_api.py"
        code={`import requests

BASE_URL = "http://localhost:3000/api/v1"
HEADERS = {"Authorization": "Bearer YOUR_TOKEN"}

# Upload a document for RAG
with open("report.pdf", "rb") as f:
    resp = requests.post(
        f"{BASE_URL}/files/",
        headers=HEADERS,
        files={"file": ("report.pdf", f, "application/pdf")},
    )
    file_data = resp.json()
    print(f"Uploaded: {file_data.get('id')}")

# Create a knowledge collection
resp = requests.post(
    f"{BASE_URL}/knowledge/create",
    headers=HEADERS,
    json={
        "name": "Q3 Reports",
        "description": "Quarterly financial reports",
    },
)
collection = resp.json()

# Chat with document context
resp = requests.post(
    f"{BASE_URL}/../api/chat/completions",
    headers={**HEADERS, "Content-Type": "application/json"},
    json={
        "model": "llama3.2",
        "messages": [
            {"role": "user", "content": "What were the Q3 revenue numbers?"}
        ],
        "files": [{"type": "file", "id": file_data["id"]}],
        "stream": False,
    },
)
print(resp.json()["choices"][0]["message"]["content"])`}
        id="code-rag-api"
      />

      <NoteBlock
        type="tip"
        title="Optimizing RAG Quality"
        content="For better RAG results: (1) use domain-appropriate chunk sizes (smaller for Q&A, larger for summarization), (2) increase top_k if answers span multiple sections, (3) use a strong embedding model like nomic-embed-text, and (4) organize related documents into collections."
        id="note-quality"
      />

      <WarningBlock
        title="RAG Is Not Perfect"
        content="RAG retrieval can miss relevant passages if the query uses different terminology than the document. It may also retrieve irrelevant chunks that confuse the model. Always verify important answers against the source document. Consider adjusting the relevance threshold if you get too many false positives."
        id="warning-limitations"
      />
    </div>
  )
}
