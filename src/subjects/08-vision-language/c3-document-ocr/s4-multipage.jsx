import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function MultipageDocuments() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Multi-Page Document Understanding</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Real-world documents like contracts, reports, and academic papers span multiple pages.
        Multi-page understanding requires models to maintain context across page boundaries,
        resolve cross-page references, and aggregate information from different sections.
        This extends single-page models with long-context strategies and page-aware architectures.
      </p>

      <DefinitionBlock
        title="Multi-Page Document Understanding"
        definition="Multi-page document understanding processes a sequence of page images $\{P_1, P_2, \ldots, P_K\}$ to answer questions or extract information that may require reasoning across pages. The challenge is encoding all pages within a manageable context while preserving layout and cross-page relationships."
        id="def-multipage"
      />

      <h2 className="text-2xl font-semibold">Strategies for Multi-Page Processing</h2>
      <p className="text-gray-700 dark:text-gray-300">
        Three main approaches handle multi-page documents: (1) process pages independently and
        aggregate, (2) concatenate all page tokens into one long sequence, or (3) use hierarchical
        encoding with page-level and document-level representations.
      </p>

      <ExampleBlock
        title="Token Budget for Multi-Page Documents"
        problem="A 10-page document with 1024x1024 images at patch size 16. How many visual tokens total?"
        steps={[
          { formula: '\\text{Tokens per page} = (1024 / 16)^2 = 64^2 = 4{,}096', explanation: 'Each page produces 4096 visual patch tokens.' },
          { formula: '\\text{Total} = 10 \\times 4{,}096 = 40{,}960 \\text{ visual tokens}', explanation: 'All pages together exceed most context windows.' },
          { formula: '\\text{With Perceiver (64 queries)}: 10 \\times 64 = 640 \\text{ tokens}', explanation: 'Compression via Perceiver Resampler makes multi-page feasible.' },
        ]}
        id="example-multipage-tokens"
      />

      <PythonCode
        title="multipage_processing.py"
        code={`import torch
import torch.nn as nn
from typing import List

class MultiPageDocEncoder(nn.Module):
    """Hierarchical multi-page document encoder."""
    def __init__(self, page_dim=768, doc_dim=768, num_queries=64, max_pages=20):
        super().__init__()
        # Per-page compression (Perceiver-style)
        self.page_queries = nn.Parameter(torch.randn(num_queries, page_dim) * 0.02)
        self.page_cross_attn = nn.MultiheadAttention(page_dim, 12, batch_first=True)
        self.page_norm = nn.LayerNorm(page_dim)

        # Page position embedding
        self.page_pos = nn.Embedding(max_pages, page_dim)

        # Document-level transformer
        doc_layer = nn.TransformerEncoderLayer(
            doc_dim, 12, doc_dim * 4, batch_first=True, norm_first=True
        )
        self.doc_encoder = nn.TransformerEncoder(doc_layer, num_layers=4)

    def encode_page(self, page_features):
        """Compress single page features to fixed-size representation."""
        B = page_features.shape[0]
        queries = self.page_queries.unsqueeze(0).expand(B, -1, -1)
        queries = self.page_norm(queries)
        compressed, _ = self.page_cross_attn(queries, page_features, page_features)
        return compressed  # (B, num_queries, D)

    def forward(self, pages: List[torch.Tensor]):
        """Process list of page features into document representation.

        Args:
            pages: List of K tensors, each (B, N_patches, D)
        Returns:
            (B, K * num_queries, D) document representation
        """
        page_reps = []
        for i, page in enumerate(pages):
            compressed = self.encode_page(page)  # (B, Q, D)
            # Add page position embedding
            compressed = compressed + self.page_pos(
                torch.tensor(i, device=compressed.device)
            )
            page_reps.append(compressed)

        # Concatenate all page representations
        doc_tokens = torch.cat(page_reps, dim=1)  # (B, K*Q, D)
        return self.doc_encoder(doc_tokens)

# Example: 5-page document
encoder = MultiPageDocEncoder(num_queries=32, max_pages=20)
pages = [torch.randn(2, 4096, 768) for _ in range(5)]  # 5 pages of 4096 tokens each

doc_repr = encoder(pages)
print(f"Input: 5 pages x 4096 tokens = {5 * 4096} total tokens")
print(f"Output: {doc_repr.shape}")  # (2, 160, 768) = 5 pages x 32 queries
print(f"Compression ratio: {5 * 4096 / doc_repr.shape[1]:.0f}x")`}
        id="code-multipage"
      />

      <NoteBlock
        type="tip"
        title="Practical Multi-Page Approaches"
        content="For production systems, consider hybrid approaches: use OCR + text-based LLM for text-heavy pages, and VLM for pages with complex layouts, figures, or charts. Route pages through different pipelines based on content type to balance cost and accuracy."
        id="note-practical-multipage"
      />

      <NoteBlock
        type="note"
        title="Long-Context VLMs"
        content="Models like Qwen2-VL and InternVL2 support processing multiple images within their extended context windows (32K+ tokens). Combined with dynamic resolution, they can handle 10-20 page documents directly, though at significant computational cost."
        id="note-long-context"
      />

      <WarningBlock
        title="Cross-Page Reference Resolution"
        content="Multi-page processing with per-page compression can lose cross-page references like 'see Table 2 on page 5' or running totals. Ensure your approach maintains enough information for cross-page reasoning, potentially with explicit page linking mechanisms."
        id="warning-cross-page"
      />
    </div>
  )
}
