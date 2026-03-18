import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function Pipeline() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">The Retrieve-Augment-Generate Pipeline</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        A RAG system consists of three core stages: retrieving relevant documents from a knowledge
        base, augmenting the user query with this context, and generating a grounded response.
        Understanding each stage and how they connect is essential for building effective RAG applications.
      </p>

      <DefinitionBlock
        title="RAG Pipeline"
        definition="The RAG pipeline transforms a user query $q$ through three stages: (1) Retrieve: find documents $D = \{d_1, \dots, d_k\}$ where $d_i = \arg\max_{d \in \mathcal{D}} \text{sim}(e_q, e_d)$, (2) Augment: construct prompt $p = f(q, D)$, (3) Generate: produce answer $a = \text{LLM}(p)$."
        id="def-pipeline"
      />

      <h2 className="text-2xl font-semibold">Stage 1: Retrieval</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The retrieval stage converts the query into an embedding and searches a vector database
        for the most similar document chunks. The similarity is typically measured using cosine
        similarity or dot product:
      </p>
      <BlockMath math="\text{sim}(q, d) = \frac{e_q \cdot e_d}{\|e_q\| \|e_d\|}" />

      <ExampleBlock
        title="End-to-End RAG Pipeline"
        problem="Build a RAG pipeline that answers questions about a technical document."
        steps={[
          { formula: '\\text{Chunk: } \\mathcal{D} \\to \\{c_1, c_2, \\dots, c_n\\}', explanation: 'Split documents into overlapping chunks of manageable size.' },
          { formula: 'e_{c_i} = \\text{Embed}(c_i) \\in \\mathbb{R}^d', explanation: 'Compute dense embeddings for each chunk.' },
          { formula: 'D_k = \\text{top-}k(\\text{sim}(e_q, e_{c_i}))', explanation: 'Retrieve the k most similar chunks to the query embedding.' },
          { formula: 'a = \\text{LLM}(\\text{prompt}(q, D_k))', explanation: 'Generate the answer conditioned on query and retrieved context.' },
        ]}
        id="example-pipeline"
      />

      <PythonCode
        title="full_rag_pipeline.py"
        code={`# Complete RAG pipeline with LangChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Step 1: Prepare documents
documents = [
    "Transformers use self-attention to process sequences in parallel.",
    "The attention mechanism computes weighted sums over all positions.",
    "Layer normalization stabilizes training of deep transformer networks.",
    "Positional encodings provide sequence order information to the model.",
    "Multi-head attention allows attending to different representation subspaces.",
]

# Step 2: Chunk and embed (here docs are already small)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50
)
splits = text_splitter.create_documents(documents)

# Step 3: Store in vector database
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 4: Build generation chain
prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the provided context.

Context: {context}
Question: {question}
Answer:"""
)

def format_docs(docs):
    return "\\n\\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

# Step 5: Query
answer = rag_chain.invoke("How does self-attention work?")
print(answer)`}
        id="code-full-pipeline"
      />

      <NoteBlock
        type="tip"
        title="Choosing the Right k"
        content="The number of retrieved documents k is a critical hyperparameter. Too few and you miss relevant context; too many and you overwhelm the LLM with noise or exceed context limits. Start with k=3-5 for most applications and tune based on retrieval quality metrics."
        id="note-choosing-k"
      />

      <h2 className="text-2xl font-semibold">Stage 2: Augmentation</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The augmentation stage constructs the final prompt by combining the user query with
        retrieved context. Prompt engineering here is crucial: the template must instruct the
        model to use the context faithfully, cite sources when possible, and acknowledge when
        the context is insufficient.
      </p>

      <WarningBlock
        title="Context Window Limits"
        content="Retrieved chunks must fit within the model's context window along with the system prompt, query, and space for the response. A model with 8K context can practically use about 4-5K tokens for retrieved context. Always calculate your token budget before designing the pipeline."
        id="warning-context-limits"
      />

      <NoteBlock
        type="note"
        title="Indexing Pipeline vs. Query Pipeline"
        content="RAG has two distinct pipelines: the indexing pipeline (offline) handles document loading, chunking, embedding, and storage. The query pipeline (online) handles query embedding, retrieval, augmentation, and generation. Separating these concerns allows independent optimization of each stage."
        id="note-two-pipelines"
      />
    </div>
  )
}
