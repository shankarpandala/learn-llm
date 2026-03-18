import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

export default function VisualizingEmbeddings() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Visualizing Embedding Spaces</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Word embeddings live in high-dimensional spaces (typically 100-300 dimensions) that are
        impossible to visualize directly. Dimensionality reduction techniques like PCA, t-SNE, and
        UMAP project these vectors into 2D or 3D while preserving meaningful structure, enabling
        us to inspect clusters, analogies, and other patterns visually.
      </p>

      <DefinitionBlock
        title="Principal Component Analysis (PCA)"
        definition="PCA finds orthogonal directions of maximum variance in the data. Given embedding matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$, PCA computes the top $k$ eigenvectors of the covariance matrix $\mathbf{C} = \frac{1}{n}\mathbf{X}^\top \mathbf{X}$ and projects onto them: $\mathbf{Z} = \mathbf{X}\mathbf{U}_k$ where $\mathbf{U}_k \in \mathbb{R}^{d \times k}$."
        notation="$k$ is typically 2 or 3 for visualization. PCA is linear and fast but may miss nonlinear structure."
        id="def-pca"
      />

      <DefinitionBlock
        title="t-SNE (t-Distributed Stochastic Neighbor Embedding)"
        definition="t-SNE (van der Maaten & Hinton, 2008) preserves local neighborhood structure by matching pairwise similarity distributions. It uses Gaussian kernels in high-D and a Student-t kernel in low-D, minimizing the KL divergence between the two distributions: $\\text{KL}(P \\| Q) = \\sum_{i \\neq j} p_{ij} \\log \\frac{p_{ij}}{q_{ij}}$."
        id="def-tsne"
      />

      <NoteBlock
        type="tip"
        title="Perplexity in t-SNE"
        content="The perplexity parameter (typically 5-50) controls the balance between local and global structure. Lower perplexity focuses on very local neighborhoods; higher values consider broader context. Always try multiple perplexity values before drawing conclusions from a t-SNE plot."
        id="note-perplexity"
      />

      <WarningBlock
        title="Interpreting t-SNE Carefully"
        content="Distances between clusters in t-SNE plots are NOT meaningful -- only the existence of clusters is. The algorithm can produce different layouts with different random seeds or perplexity values. Never conclude that two clusters are 'far apart' based solely on their visual separation in a t-SNE plot."
        id="warn-tsne"
      />

      <DefinitionBlock
        title="UMAP (Uniform Manifold Approximation and Projection)"
        definition="UMAP (McInnes et al., 2018) is a manifold learning technique based on Riemannian geometry and algebraic topology. Like t-SNE it preserves local structure, but it also better preserves global structure and is significantly faster, making it practical for very large embedding collections."
        id="def-umap"
      />

      <PythonCode
        title="visualize_embeddings.py"
        id="code-viz"
        code={`import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Simulated word embeddings (in practice, load from gensim or file)
np.random.seed(42)
words = [
    "king", "queen", "prince", "princess",    # royalty
    "man", "woman", "boy", "girl",             # gender
    "cat", "dog", "fish", "bird",              # animals
    "car", "truck", "bus", "bicycle",          # vehicles
]

# Create embeddings with cluster structure
d = 50
embeddings = np.random.randn(len(words), d) * 0.3
# Add cluster offsets to create semantic groups
for i in range(4):   embeddings[i] += np.random.randn(d) * 0.1 + 1.0
for i in range(4,8): embeddings[i] += np.random.randn(d) * 0.1 + 0.5
for i in range(8,12): embeddings[i] += np.random.randn(d) * 0.1 - 0.5
for i in range(12,16): embeddings[i] += np.random.randn(d) * 0.1 - 1.0

# --- PCA ---
pca = PCA(n_components=2)
coords_pca = pca.fit_transform(embeddings)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot PCA
ax = axes[0]
colors = ["#e74c3c"]*4 + ["#3498db"]*4 + ["#2ecc71"]*4 + ["#f39c12"]*4
ax.scatter(coords_pca[:, 0], coords_pca[:, 1], c=colors, s=100, zorder=5)
for i, word in enumerate(words):
    ax.annotate(word, (coords_pca[i, 0]+0.05, coords_pca[i, 1]+0.05), fontsize=9)
ax.set_title(f"PCA (explains {pca.explained_variance_ratio_.sum()*100:.1f}% var)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

# --- t-SNE ---
tsne = TSNE(n_components=2, perplexity=5, random_state=42, n_iter=1000)
coords_tsne = tsne.fit_transform(embeddings)

ax = axes[1]
ax.scatter(coords_tsne[:, 0], coords_tsne[:, 1], c=colors, s=100, zorder=5)
for i, word in enumerate(words):
    ax.annotate(word, (coords_tsne[i, 0]+0.5, coords_tsne[i, 1]+0.5), fontsize=9)
ax.set_title("t-SNE (perplexity=5)")
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")

plt.tight_layout()
plt.savefig("embedding_visualization.png", dpi=150)
plt.show()`}
      />

      <ExampleBlock
        title="Variance Explained by PCA"
        problem="If the first two principal components explain 15% and 8% of the variance respectively in a 300-dim embedding space, how should we interpret the projection?"
        steps={[
          { formula: '$\\text{Total explained} = 15\\% + 8\\% = 23\\%$', explanation: 'Only about a quarter of the information is captured.' },
          { formula: '$\\text{Remaining} = 77\\%$ across 298 dimensions', explanation: 'The majority of the structure lives in dimensions we cannot see.' },
          { formula: 'Conclusion: clusters visible in PCA are real signal', explanation: 'But absence of structure in 2D does not imply absence in high-D. Use PCA for confirmation, not discovery.' },
        ]}
        id="ex-pca-variance"
      />

      <NoteBlock
        type="note"
        title="When to Use Which Method"
        content="Use PCA for a quick global overview and when you need reproducible, interpretable axes. Use t-SNE for exploring local cluster structure in moderate-sized sets (up to ~10k points). Use UMAP for large-scale visualization (100k+ points) where you need both local and global structure preserved, and faster runtime."
        id="note-which-method"
      />
    </div>
  )
}
