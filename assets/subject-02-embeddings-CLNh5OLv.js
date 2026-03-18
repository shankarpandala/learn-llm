import{j as e}from"./vendor-DWbzdFaj.js";import{D as i,W as o,P as a,N as n,E as r,T as s}from"./subject-01-text-fundamentals-DG6tAvii.js";import{r as t}from"./vendor-katex-BYl39Yo6.js";function d(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"One-Hot to Dense Vectors"}),e.jsxs("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:["The simplest way to represent words numerically is the one-hot encoding: assign each word in a vocabulary of size ",e.jsx(t.InlineMath,{math:"V"})," a unique index and represent it as a binary vector with a single 1 at that index and 0s everywhere else. While straightforward, this approach has fundamental limitations that motivate the move to dense, distributed representations."]}),e.jsx(i,{title:"One-Hot Encoding",definition:"Given a vocabulary $\\mathcal{V}$ of size $V$, the one-hot representation of the $i$-th word is a vector $\\mathbf{e}_i \\in \\{0,1\\}^V$ where $e_{ij} = 1$ if $j = i$ and $e_{ij} = 0$ otherwise.",notation:"$\\mathbf{e}_i$ denotes the one-hot vector for word $i$; the inner product $\\mathbf{e}_i^\\top \\mathbf{e}_j = 0$ for all $i \\neq j$.",id:"def-one-hot"}),e.jsx(o,{title:"The Curse of Orthogonality",content:"Because every pair of one-hot vectors is orthogonal, the cosine similarity between any two distinct words is exactly zero. This means 'cat' is as different from 'dog' as it is from 'quantum' -- the representation captures no semantic relationships whatsoever.",id:"warn-orthogonality"}),e.jsx(a,{title:"one_hot_demo.py",id:"code-one-hot",code:`import numpy as np

# Build a tiny vocabulary
vocab = ["king", "queen", "man", "woman", "apple"]
word_to_idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# One-hot encode
def one_hot(word):
    vec = np.zeros(V)
    vec[word_to_idx[word]] = 1.0
    return vec

# Demonstrate the problem: all similarities are 0
king = one_hot("king")
queen = one_hot("queen")
apple = one_hot("apple")

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

print(f"sim(king, queen) = {cosine_sim(king, queen):.2f}")   # 0.00
print(f"sim(king, apple) = {cosine_sim(king, apple):.2f}")   # 0.00
print(f"Memory for V=50,000: {50_000 * 50_000 * 4 / 1e9:.1f} GB (float32)")
# For a real vocabulary, this is extremely wasteful`}),e.jsx(i,{title:"Distributional Hypothesis",definition:"Words that occur in similar linguistic contexts tend to have similar meanings (Harris, 1954; Firth, 1957). Formally, if two words $w_i$ and $w_j$ share similar context distributions $P(\\text{context} \\mid w_i) \\approx P(\\text{context} \\mid w_j)$, then $w_i$ and $w_j$ are semantically related.",id:"def-distributional"}),e.jsx(n,{type:"historical",title:"Firth's Famous Quote",content:"'You shall know a word by the company it keeps' -- J.R. Firth (1957). This insight underpins virtually all modern embedding methods. Rather than defining meaning through symbolic rules, we learn meaning from statistical patterns of co-occurrence in large corpora.",id:"note-firth"}),e.jsx(r,{title:"From Sparse to Dense: Embedding Lookup",problem:"Show that multiplying a one-hot vector by an embedding matrix $\\mathbf{W} \\in \\mathbb{R}^{V \\times d}$ simply selects the corresponding row, yielding a dense vector in $\\mathbb{R}^d$.",steps:[{formula:"$\\mathbf{e}_i^\\top \\mathbf{W} = \\mathbf{W}[i, :]$",explanation:"The one-hot vector acts as a row selector."},{formula:"$\\mathbf{W}[i, :] \\in \\mathbb{R}^d$ where $d \\ll V$",explanation:"The result is a dense vector of dimension d (typically 50-300), much smaller than V (often 50,000+)."},{formula:"$\\cos(\\mathbf{W}[i,:],\\, \\mathbf{W}[j,:]) \\neq 0$ in general",explanation:"Dense vectors can express graded similarity between words, unlike one-hot vectors."}],id:"ex-embedding-lookup"}),e.jsx(a,{title:"embedding_lookup.py",id:"code-embedding-lookup",code:`import numpy as np

V, d = 5, 3  # tiny vocab, 3-dim embeddings
vocab = ["king", "queen", "man", "woman", "apple"]

# Random embedding matrix (in practice, learned from data)
np.random.seed(42)
W = np.random.randn(V, d)

# One-hot lookup is equivalent to indexing
word_idx = 0  # "king"
e_king = np.zeros(V)
e_king[word_idx] = 1.0

# These two are identical:
via_matmul = e_king @ W
via_index  = W[word_idx]

print("Via matmul:", via_matmul)
print("Via index: ", via_index)
print("Equal?", np.allclose(via_matmul, via_index))

# Now similarity is meaningful (once W is learned)
print(f"\\nDense sim(king, queen) = {np.dot(W[0], W[1]) / (np.linalg.norm(W[0]) * np.linalg.norm(W[1])):.3f}")`}),e.jsx(n,{type:"intuition",title:"Why Dense Representations Work",content:"Dense embeddings compress word identity into a low-dimensional space where geometric relationships encode semantic ones. Each dimension can be thought of as capturing a latent feature -- formality, animacy, gender -- that is shared across many words. This sharing is what makes embeddings generalize: even words seen rarely can be placed near semantically similar neighbors.",id:"note-why-dense"})]})}const k=Object.freeze(Object.defineProperty({__proto__:null,default:d},Symbol.toStringTag,{value:"Module"}));function l(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Word2Vec: CBOW & Skip-gram"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Word2Vec (Mikolov et al., 2013) introduced two efficient architectures for learning word embeddings from raw text: Continuous Bag-of-Words (CBOW) and Skip-gram. Both exploit the distributional hypothesis by training a shallow neural network to predict words from their context or vice versa. The learned weight matrices become the word embeddings."}),e.jsx(i,{title:"Continuous Bag-of-Words (CBOW)",definition:"CBOW predicts a target word $w_t$ given its surrounding context words $\\{w_{t-c}, \\dots, w_{t-1}, w_{t+1}, \\dots, w_{t+c}\\}$. The context vectors are averaged and passed through a linear layer with softmax to produce a probability distribution over the vocabulary.",notation:"Context window size $c$; input embeddings $\\mathbf{W} \\in \\mathbb{R}^{V \\times d}$; output weights $\\mathbf{W}' \\in \\mathbb{R}^{d \\times V}$.",id:"def-cbow"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The CBOW objective maximizes:"}),e.jsx(t.BlockMath,{math:String.raw`\mathcal{L}_{\text{CBOW}} = \frac{1}{T}\sum_{t=1}^{T} \log P(w_t \mid w_{t-c}, \dots, w_{t+c})`}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"where the conditional probability uses softmax over the vocabulary:"}),e.jsx(t.BlockMath,{math:String.raw`P(w_t \mid \text{ctx}) = \frac{\exp(\mathbf{v}'_{w_t}{}^\top \bar{\mathbf{v}}_{\text{ctx}})}{\sum_{w=1}^{V} \exp(\mathbf{v}'_{w}{}^\top \bar{\mathbf{v}}_{\text{ctx}})}`}),e.jsx(i,{title:"Skip-gram",definition:"Skip-gram reverses CBOW: given a center word $w_t$, it predicts each context word $w_{t+j}$ (for $-c \\\\leq j \\\\leq c$, $j \\\\neq 0$) independently. This makes Skip-gram especially effective for rare words because each word appears as a center word in multiple training examples.",id:"def-skipgram"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The Skip-gram objective maximizes:"}),e.jsx(t.BlockMath,{math:String.raw`\mathcal{L}_{\text{SG}} = \frac{1}{T}\sum_{t=1}^{T} \sum_{\substack{-c \leq j \leq c \\ j \neq 0}} \log P(w_{t+j} \mid w_t)`}),e.jsx(n,{type:"intuition",title:"CBOW vs Skip-gram Trade-offs",content:"CBOW is faster to train and performs slightly better on frequent words because it averages context (smoothing noise). Skip-gram is slower but excels on rare words and small datasets because each center-context pair creates a separate training example. In practice, Skip-gram with negative sampling is the most widely used variant.",id:"note-tradeoffs"}),e.jsx(r,{title:"Training Pairs from a Sentence",problem:"Given the sentence 'the cat sat on the mat' and window size $c=2$, generate the Skip-gram training pairs for center word 'sat'.",steps:[{formula:"Center: sat, Context: the",explanation:"Two positions to the left (t-2)."},{formula:"Center: sat, Context: cat",explanation:"One position to the left (t-1)."},{formula:"Center: sat, Context: on",explanation:"One position to the right (t+1)."},{formula:"Center: sat, Context: the",explanation:"Two positions to the right (t+2)."}],id:"ex-training-pairs"}),e.jsx(a,{title:"word2vec_gensim.py",id:"code-word2vec",code:`from gensim.models import Word2Vec

# Sample corpus (list of tokenized sentences)
corpus = [
    ["the", "king", "rules", "the", "kingdom"],
    ["the", "queen", "rules", "with", "wisdom"],
    ["a", "man", "and", "a", "woman", "walked"],
    ["the", "prince", "is", "son", "of", "the", "king"],
    ["the", "princess", "is", "daughter", "of", "the", "queen"],
]

# Train Skip-gram model (sg=1); use sg=0 for CBOW
model = Word2Vec(
    sentences=corpus,
    vector_size=50,    # embedding dimension
    window=3,          # context window size
    min_count=1,       # include all words
    sg=1,              # 1 = Skip-gram, 0 = CBOW
    epochs=100,        # more epochs for tiny corpus
    seed=42,
)

# Access the learned embedding
king_vec = model.wv["king"]
print(f"Embedding shape: {king_vec.shape}")  # (50,)

# Find most similar words
similar = model.wv.most_similar("king", topn=3)
for word, score in similar:
    print(f"  {word}: {score:.3f}")

# Vector arithmetic (may need larger corpus for good results)
result = model.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1
)
print(f"\\nking - man + woman = {result[0][0]} ({result[0][1]:.3f})")`}),e.jsx(o,{title:"Softmax Bottleneck",content:"The full softmax requires computing a dot product with every word in the vocabulary for each training step, making it O(V) per example. For vocabularies of 100k+ words, this is prohibitively expensive. This motivates the negative sampling and hierarchical softmax approximations covered in the next section.",id:"warn-softmax"}),e.jsx(n,{type:"historical",title:"Impact of Word2Vec",content:"Word2Vec's 2013 release was a watershed moment for NLP. It demonstrated that simple, shallow models trained on large corpora could produce embeddings capturing complex semantic relationships. The resulting 'word vectors' became a standard component in virtually all NLP pipelines until the rise of contextual embeddings with ELMo and BERT.",id:"note-history"})]})}const j=Object.freeze(Object.defineProperty({__proto__:null,default:l},Symbol.toStringTag,{value:"Module"}));function c(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Training Objectives & Negative Sampling"}),e.jsxs("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:["Computing the full softmax over a vocabulary of size ",e.jsx(t.InlineMath,{math:"V"})," at every training step is computationally prohibitive. Negative sampling (NEG) and Noise Contrastive Estimation (NCE) replace this expensive normalization with a much cheaper binary classification task, reducing the cost from ",e.jsx(t.InlineMath,{math:"O(V)"})," to ",e.jsx(t.InlineMath,{math:"O(k)"})," per training example, where ",e.jsx(t.InlineMath,{math:"k"})," is the number of negative samples."]}),e.jsx(i,{title:"Noise Contrastive Estimation (NCE)",definition:"NCE reformulates density estimation as a binary classification problem: distinguish real data samples from noise samples drawn from a known distribution $P_n(w)$. For each positive (center, context) pair, we draw $k$ negative samples from $P_n$ and train a logistic classifier to separate them.",id:"def-nce"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The Skip-gram with negative sampling (SGNS) objective for a center word"," ",e.jsx(t.InlineMath,{math:"w"})," and true context word ",e.jsx(t.InlineMath,{math:"c"})," is:"]}),e.jsx(t.BlockMath,{math:String.raw`\mathcal{L}_{\text{NEG}} = \log \sigma(\mathbf{v}_c'{}^\top \mathbf{v}_w) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)}\!\left[\log \sigma(-\mathbf{v}_{w_i}'{}^\top \mathbf{v}_w)\right]`}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["where ",e.jsx(t.InlineMath,{math:"\\sigma(x) = 1/(1 + e^{-x})"})," is the sigmoid function and"," ",e.jsx(t.InlineMath,{math:"P_n(w) \\propto f(w)^{3/4}"})," is the noise distribution (unigram raised to the 3/4 power)."]}),e.jsx(i,{title:"Negative Sampling Distribution",definition:"The noise distribution is the unigram distribution raised to the 3/4 power: $P_n(w) = \\\\frac{f(w)^{3/4}}{\\\\sum_{w'} f(w')^{3/4}}$, where $f(w)$ is the frequency of word $w$ in the corpus. The 3/4 exponent smooths the distribution, giving rare words a higher sampling probability than their raw frequency would suggest.",id:"def-noise-dist"}),e.jsx(r,{title:"Effect of the 3/4 Power",problem:"Compare sampling probabilities for a frequent word (f=0.01) and a rare word (f=0.0001) under raw unigram vs. smoothed distribution.",steps:[{formula:"$\\text{Raw ratio} = 0.01 / 0.0001 = 100$",explanation:"The frequent word is 100x more likely to be sampled."},{formula:"$0.01^{0.75} = 0.01778$, $0.0001^{0.75} = 0.000562$",explanation:"Apply the 3/4 power to both frequencies."},{formula:"$\\text{Smoothed ratio} = 0.01778 / 0.000562 \\approx 31.6$",explanation:"The ratio drops from 100 to ~32, giving rare words more negative samples and better gradient signal."}],id:"ex-smoothing"}),e.jsx(s,{title:"SGNS Implicitly Factorizes the PMI Matrix",statement:"Levy & Goldberg (2014) showed that Skip-gram with negative sampling, when fully converged, implicitly factorizes a shifted pointwise mutual information (PMI) matrix: $\\mathbf{v}_w \\cdot \\mathbf{v}_c' = \\text{PMI}(w, c) - \\log k$, where $k$ is the number of negative samples.",corollaries:["This connects neural embedding methods to classical count-based distributional semantics.","The number of negative samples k acts as an implicit regularizer on the PMI values."],id:"thm-pmi"}),e.jsx(a,{title:"negative_sampling_from_scratch.py",id:"code-neg-sampling",code:`import numpy as np

np.random.seed(42)

# Simulate a tiny vocabulary with word frequencies
vocab = ["the", "king", "queen", "man", "woman", "throne"]
freqs = np.array([100, 20, 18, 25, 22, 5], dtype=float)

# Compute noise distribution: f(w)^(3/4) / Z
smoothed = freqs ** 0.75
noise_dist = smoothed / smoothed.sum()

print("Noise distribution (3/4 smoothing):")
for w, p in zip(vocab, noise_dist):
    print(f"  {w:8s}: {p:.4f}")

# Draw negative samples for a training pair
k = 5  # number of negative samples
center_idx = 1   # "king"
context_idx = 2  # "queen" (positive pair)

neg_indices = np.random.choice(len(vocab), size=k, p=noise_dist)
print(f"\\nPositive pair: ({vocab[center_idx]}, {vocab[context_idx]})")
print(f"Negative samples: {[vocab[i] for i in neg_indices]}")

# Simple gradient step (illustrative)
d = 10  # embedding dim
W_center = np.random.randn(len(vocab), d) * 0.1   # center embeddings
W_context = np.random.randn(len(vocab), d) * 0.1  # context embeddings
lr = 0.01

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

# Forward: positive pair
v_c = W_center[center_idx]
v_ctx = W_context[context_idx]
pos_score = sigmoid(np.dot(v_c, v_ctx))
loss = -np.log(pos_score + 1e-8)

# Forward: negative pairs
for ni in neg_indices:
    v_neg = W_context[ni]
    neg_score = sigmoid(-np.dot(v_c, v_neg))
    loss -= np.log(neg_score + 1e-8)

print(f"\\nLoss (1 step): {loss:.4f}")`}),e.jsx(n,{type:"tip",title:"Choosing k (Number of Negatives)",content:"Mikolov et al. recommend k=5-20 for small datasets and k=2-5 for large datasets. More negatives provide a better approximation of the full softmax gradient but increase computation. In practice, k=5 is a robust default.",id:"note-choosing-k"}),e.jsx(o,{title:"Self-Negatives",content:"When sampling negatives, you may accidentally sample the true context word as a negative. For large vocabularies this is rare and can be safely ignored. For very small vocabularies, you should explicitly filter out the positive word from the negative sample set.",id:"warn-self-neg"})]})}const $=Object.freeze(Object.defineProperty({__proto__:null,default:c},Symbol.toStringTag,{value:"Module"}));function m(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Visualizing Embedding Spaces"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"Word embeddings live in high-dimensional spaces (typically 100-300 dimensions) that are impossible to visualize directly. Dimensionality reduction techniques like PCA, t-SNE, and UMAP project these vectors into 2D or 3D while preserving meaningful structure, enabling us to inspect clusters, analogies, and other patterns visually."}),e.jsx(i,{title:"Principal Component Analysis (PCA)",definition:"PCA finds orthogonal directions of maximum variance in the data. Given embedding matrix $\\mathbf{X} \\in \\mathbb{R}^{n \\times d}$, PCA computes the top $k$ eigenvectors of the covariance matrix $\\mathbf{C} = \\frac{1}{n}\\mathbf{X}^\\top \\mathbf{X}$ and projects onto them: $\\mathbf{Z} = \\mathbf{X}\\mathbf{U}_k$ where $\\mathbf{U}_k \\in \\mathbb{R}^{d \\times k}$.",notation:"$k$ is typically 2 or 3 for visualization. PCA is linear and fast but may miss nonlinear structure.",id:"def-pca"}),e.jsx(i,{title:"t-SNE (t-Distributed Stochastic Neighbor Embedding)",definition:"t-SNE (van der Maaten & Hinton, 2008) preserves local neighborhood structure by matching pairwise similarity distributions. It uses Gaussian kernels in high-D and a Student-t kernel in low-D, minimizing the KL divergence between the two distributions: $\\\\text{KL}(P \\\\| Q) = \\\\sum_{i \\\\neq j} p_{ij} \\\\log \\\\frac{p_{ij}}{q_{ij}}$.",id:"def-tsne"}),e.jsx(n,{type:"tip",title:"Perplexity in t-SNE",content:"The perplexity parameter (typically 5-50) controls the balance between local and global structure. Lower perplexity focuses on very local neighborhoods; higher values consider broader context. Always try multiple perplexity values before drawing conclusions from a t-SNE plot.",id:"note-perplexity"}),e.jsx(o,{title:"Interpreting t-SNE Carefully",content:"Distances between clusters in t-SNE plots are NOT meaningful -- only the existence of clusters is. The algorithm can produce different layouts with different random seeds or perplexity values. Never conclude that two clusters are 'far apart' based solely on their visual separation in a t-SNE plot.",id:"warn-tsne"}),e.jsx(i,{title:"UMAP (Uniform Manifold Approximation and Projection)",definition:"UMAP (McInnes et al., 2018) is a manifold learning technique based on Riemannian geometry and algebraic topology. Like t-SNE it preserves local structure, but it also better preserves global structure and is significantly faster, making it practical for very large embedding collections.",id:"def-umap"}),e.jsx(a,{title:"visualize_embeddings.py",id:"code-viz",code:`import numpy as np
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
plt.show()`}),e.jsx(r,{title:"Variance Explained by PCA",problem:"If the first two principal components explain 15% and 8% of the variance respectively in a 300-dim embedding space, how should we interpret the projection?",steps:[{formula:"$\\text{Total explained} = 15\\% + 8\\% = 23\\%$",explanation:"Only about a quarter of the information is captured."},{formula:"$\\text{Remaining} = 77\\%$ across 298 dimensions",explanation:"The majority of the structure lives in dimensions we cannot see."},{formula:"Conclusion: clusters visible in PCA are real signal",explanation:"But absence of structure in 2D does not imply absence in high-D. Use PCA for confirmation, not discovery."}],id:"ex-pca-variance"}),e.jsx(n,{type:"note",title:"When to Use Which Method",content:"Use PCA for a quick global overview and when you need reproducible, interpretable axes. Use t-SNE for exploring local cluster structure in moderate-sized sets (up to ~10k points). Use UMAP for large-scale visualization (100k+ points) where you need both local and global structure preserved, and faster runtime.",id:"note-which-method"})]})}const S=Object.freeze(Object.defineProperty({__proto__:null,default:m},Symbol.toStringTag,{value:"Module"}));function p(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"GloVe: Global Vectors for Word Representation"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"GloVe (Pennington et al., 2014) bridges the gap between count-based methods (like LSA) and prediction-based methods (like Word2Vec). It directly factorizes the log of the global word-word co-occurrence matrix using a weighted least squares objective, producing embeddings that capture both local context patterns and global corpus statistics."}),e.jsx(i,{title:"Co-occurrence Matrix",definition:"The co-occurrence matrix $\\mathbf{X}$ has entries $X_{ij}$ counting the number of times word $j$ appears in the context of word $i$ across the entire corpus. Context is defined by a symmetric window of size $c$. Closer context words may be given higher weight via a decaying function $1/d$ where $d$ is the distance.",notation:"$X_{ij}$ = co-occurrence count; $X_i = \\sum_k X_{ik}$ = total co-occurrences for word $i$; $P_{ij} = X_{ij}/X_i$ = co-occurrence probability.",id:"def-cooccurrence"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The key insight of GloVe is that ratios of co-occurrence probabilities encode meaning. For words ",e.jsx(t.InlineMath,{math:"i"})," (ice) and ",e.jsx(t.InlineMath,{math:"j"})," (steam), the ratio ",e.jsx(t.InlineMath,{math:"P_{ik}/P_{jk}"})," is large when ",e.jsx(t.InlineMath,{math:"k"})," (solid) relates to ice but not steam, and small when ",e.jsx(t.InlineMath,{math:"k"})," (gas) relates to steam but not ice."]}),e.jsx(i,{title:"GloVe Objective Function",definition:"GloVe minimizes the weighted least squares cost: $J = \\\\sum_{i,j=1}^{V} f(X_{ij})\\\\left(\\\\mathbf{w}_i^\\\\top \\\\tilde{\\\\mathbf{w}}_j + b_i + \\\\tilde{b}_j - \\\\log X_{ij}\\\\right)^2$ where $\\\\mathbf{w}_i$ and $\\\\tilde{\\\\mathbf{w}}_j$ are word and context vectors, $b_i$ and $\\\\tilde{b}_j$ are biases, and $f$ is a weighting function.",id:"def-glove-obj"}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["The weighting function ",e.jsx(t.InlineMath,{math:"f(x)"})," prevents very frequent co-occurrences from dominating the objective:"]}),e.jsx(t.BlockMath,{math:String.raw`f(x) = \begin{cases} (x/x_{\max})^{\alpha} & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}`}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["with ",e.jsx(t.InlineMath,{math:"x_{\\max} = 100"})," and ",e.jsx(t.InlineMath,{math:"\\alpha = 3/4"})," as recommended defaults."]}),e.jsx(r,{title:"Co-occurrence Probability Ratios",problem:"Given co-occurrence data: $P(\\text{solid}|\\text{ice}) = 1.9 \\times 10^{-4}$, $P(\\text{solid}|\\text{steam}) = 2.2 \\times 10^{-5}$, $P(\\text{gas}|\\text{ice}) = 6.6 \\times 10^{-5}$, $P(\\text{gas}|\\text{steam}) = 7.8 \\times 10^{-4}$. Compute the ratios.",steps:[{formula:"$P(\\text{solid}|\\text{ice}) / P(\\text{solid}|\\text{steam}) = 8.9$",explanation:'"solid" is strongly associated with ice over steam.'},{formula:"$P(\\text{gas}|\\text{ice}) / P(\\text{gas}|\\text{steam}) = 0.085$",explanation:'"gas" is strongly associated with steam over ice.'},{formula:"$P(\\text{water}|\\text{ice}) / P(\\text{water}|\\text{steam}) \\approx 1.0$",explanation:"Neutral words that relate equally to both yield ratios near 1."}],id:"ex-ratios"}),e.jsx(a,{title:"glove_cooccurrence.py",id:"code-glove",code:`import numpy as np
from collections import Counter, defaultdict

# Build co-occurrence matrix from a small corpus
corpus = [
    "the king sat on the throne".split(),
    "the queen wore the crown".split(),
    "the king and queen ruled the kingdom".split(),
    "a man and a woman walked to the throne".split(),
]

# Build vocabulary
word_counts = Counter(w for sent in corpus for w in sent)
vocab = sorted(word_counts.keys())
word_to_idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)

# Co-occurrence matrix with window size 2
window = 2
cooccur = np.zeros((V, V))

for sent in corpus:
    for i, word in enumerate(sent):
        wi = word_to_idx[word]
        for j in range(max(0, i - window), min(len(sent), i + window + 1)):
            if i != j:
                wj = word_to_idx[sent[j]]
                distance = abs(i - j)
                cooccur[wi][wj] += 1.0 / distance  # distance weighting

print(f"Vocabulary: {vocab}")
print(f"Co-occurrence matrix shape: {cooccur.shape}")

# Show co-occurrences for 'king'
king_idx = word_to_idx["king"]
print(f"\\nCo-occurrences with 'king':")
for w, idx in sorted(word_to_idx.items()):
    if cooccur[king_idx][idx] > 0:
        print(f"  {w:10s}: {cooccur[king_idx][idx]:.2f}")

# GloVe weighting function
def glove_weight(x, x_max=100, alpha=0.75):
    return np.where(x < x_max, (x / x_max) ** alpha, 1.0)

# Show weights for different co-occurrence counts
counts = [1, 5, 10, 50, 100, 500]
print(f"\\nGloVe weights f(x) for x_max=100:")
for c in counts:
    print(f"  f({c:3d}) = {glove_weight(c):.4f}")`}),e.jsx(n,{type:"note",title:"GloVe vs Word2Vec in Practice",content:"Both GloVe and Word2Vec produce high-quality embeddings and often perform comparably on downstream tasks. GloVe's main advantage is interpretability: its objective explicitly connects to co-occurrence statistics. Word2Vec (SGNS) is often easier to train incrementally on streaming data. Pre-trained GloVe vectors (6B, 42B, 840B tokens) are freely available from the Stanford NLP group.",id:"note-comparison"}),e.jsx(o,{title:"Memory Requirements",content:"The full co-occurrence matrix is V x V, which for a 400k vocabulary requires 640 GB in float32. In practice, the matrix is very sparse, and only non-zero entries are stored. Still, building the matrix for large corpora requires careful engineering and can be the main memory bottleneck.",id:"warn-memory"})]})}const T=Object.freeze(Object.defineProperty({__proto__:null,default:p},Symbol.toStringTag,{value:"Module"}));function h(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"FastText: Subword Embeddings"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:'FastText (Bojanowski et al., 2017) extends Word2Vec by representing each word as a bag of character n-grams. This simple modification yields two major benefits: it can generate embeddings for out-of-vocabulary (OOV) words, and it naturally captures morphological patterns (e.g., the shared structure in "teach", "teacher", "teaching").'}),e.jsx(i,{title:"Character N-gram Representation",definition:"A word $w$ is represented as the set of its character n-grams $\\mathcal{G}_w$ (for $n$ between $n_{\\min}$ and $n_{\\max}$), plus the whole word itself. The word embedding is the sum of its n-gram vectors: $\\mathbf{v}_w = \\mathbf{z}_w + \\sum_{g \\in \\mathcal{G}_w} \\mathbf{z}_g$ where $\\mathbf{z}_w$ is the whole-word vector and $\\mathbf{z}_g$ are n-gram vectors.",notation:"Special boundary markers < and > are added: 'where' becomes '<where>'. Default n-gram range: $n_{\\min}=3$, $n_{\\max}=6$.",id:"def-subword"}),e.jsx(r,{title:"Character N-grams for 'where'",problem:"Compute the character n-grams for the word 'where' with $n_{\\min}=3$ and $n_{\\max}=5$, using boundary markers.",steps:[{formula:"Marked form: <where>",explanation:"Add boundary markers < and >."},{formula:"n=3: <wh, whe, her, ere, re>",explanation:"All character trigrams."},{formula:"n=4: <whe, wher, here, ere>",explanation:"All character 4-grams."},{formula:"n=5: <wher, where, here>",explanation:"All character 5-grams."},{formula:"Total: 13 n-grams + whole word",explanation:"The word vector is the sum of all 14 vectors."}],id:"ex-ngrams"}),e.jsx(n,{type:"intuition",title:"Why Subwords Help with OOV Words",content:"When encountering an unseen word like 'unforgettably', FastText decomposes it into known n-grams (e.g., 'unf', 'for', 'get', 'tab', 'bly') that overlap with training words like 'unfair', 'forget', 'table', and 'ably'. The resulting embedding is a meaningful composition of these shared morphological fragments, rather than a zero vector or random initialization.",id:"note-oov"}),e.jsx(a,{title:"fasttext_demo.py",id:"code-fasttext",code:`import numpy as np

# Demonstrate character n-gram extraction
def get_ngrams(word, n_min=3, n_max=6):
    """Extract character n-grams with boundary markers."""
    marked = f"<{word}>"
    ngrams = []
    for n in range(n_min, n_max + 1):
        for i in range(len(marked) - n + 1):
            ngrams.append(marked[i:i+n])
    return ngrams

# Show n-grams for different words
words = ["where", "teacher", "teaching", "unteachable"]
for w in words:
    ng = get_ngrams(w)
    print(f"{w:15s} -> {len(ng)} n-grams")
    print(f"  Sample: {ng[:5]}...")

# Show shared n-grams between related words
def shared_ngrams(w1, w2):
    ng1 = set(get_ngrams(w1))
    ng2 = set(get_ngrams(w2))
    shared = ng1 & ng2
    return shared

shared = shared_ngrams("teacher", "teaching")
print(f"\\nShared n-grams between 'teacher' and 'teaching':")
print(f"  {sorted(shared)}")
print(f"  Count: {len(shared)}")

# Simulate FastText embedding computation
np.random.seed(42)
d = 10
ngram_vectors = {}  # pretend these are learned

def get_fasttext_embedding(word):
    """Compute word embedding as sum of n-gram vectors."""
    ngrams = get_ngrams(word) + [word]  # n-grams + whole word
    total = np.zeros(d)
    for ng in ngrams:
        if ng not in ngram_vectors:
            # Hash-based bucket (FastText uses hashing)
            h = hash(ng) % 2_000_000
            np.random.seed(h)
            ngram_vectors[ng] = np.random.randn(d) * 0.1
        total += ngram_vectors[ng]
    return total / len(ngrams)  # average for stability

# Even an OOV word gets a reasonable embedding
emb_teach = get_fasttext_embedding("teach")
emb_teacher = get_fasttext_embedding("teacher")
emb_quantum = get_fasttext_embedding("quantum")

cos = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"\\nsim(teach, teacher) = {cos(emb_teach, emb_teacher):.3f}")
print(f"sim(teach, quantum)  = {cos(emb_teach, emb_quantum):.3f}")`}),e.jsx(o,{title:"Hashing Collisions",content:"FastText uses a hash function to map n-grams to a fixed number of buckets (default 2 million). Different n-grams may collide into the same bucket, sharing a vector. This is usually benign for large bucket counts but can degrade quality if the vocabulary is very large relative to the number of buckets.",id:"warn-hashing"}),e.jsx(n,{type:"note",title:"Pre-trained FastText Models",content:"Facebook AI Research released pre-trained FastText vectors for 157 languages, trained on Common Crawl and Wikipedia. These models are especially valuable for morphologically rich languages (Finnish, Turkish, Arabic) where a single lemma can have dozens of surface forms. The subword approach handles these naturally.",id:"note-pretrained"}),e.jsx(n,{type:"tip",title:"Using FastText in Python",content:"Install the official fasttext library (pip install fasttext) for training, or use gensim's FastText wrapper. For just loading pre-trained vectors, gensim.models.fasttext.load_facebook_model() handles the .bin format which preserves subword information for OOV inference.",id:"note-usage"})]})}const E=Object.freeze(Object.defineProperty({__proto__:null,default:h},Symbol.toStringTag,{value:"Module"}));function g(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"ELMo & Contextual Embeddings"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:'ELMo (Embeddings from Language Models, Peters et al., 2018) marked a paradigm shift from static word embeddings to contextual representations. Unlike Word2Vec or GloVe, which assign a single vector per word type, ELMo produces different vectors for each word token depending on its surrounding context. The word "bank" gets different representations in "river bank" versus "bank account."'}),e.jsx(i,{title:"Contextual Word Embedding",definition:"A contextual embedding is a function $f: (w, C) \\\\to \\\\mathbb{R}^d$ that maps a word $w$ together with its context $C$ (the surrounding sentence or passage) to a vector. Unlike static embeddings where $f(w) = \\\\mathbf{v}_w$ is fixed, contextual embeddings produce different vectors for the same word in different contexts.",id:"def-contextual"}),e.jsx(i,{title:"ELMo Architecture",definition:"ELMo uses a 2-layer bidirectional LSTM trained as a language model. The forward LSTM models $P(w_t | w_1, \\\\dots, w_{t-1})$ and the backward LSTM models $P(w_t | w_{t+1}, \\\\dots, w_T)$. The final ELMo representation for word $t$ is a task-specific weighted combination of all layers: $\\\\text{ELMo}_t^{\\\\text{task}} = \\\\gamma^{\\\\text{task}} \\\\sum_{j=0}^{L} s_j^{\\\\text{task}} \\\\mathbf{h}_{t,j}$.",notation:"$\\\\mathbf{h}_{t,0}$ = character CNN layer; $\\\\mathbf{h}_{t,1}, \\\\mathbf{h}_{t,2}$ = biLSTM layers; $s_j$ = softmax-normalized weights; $\\\\gamma$ = scalar.",id:"def-elmo"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The biLM objective maximizes the joint log-likelihood of both directions:"}),e.jsx(t.BlockMath,{math:String.raw`\mathcal{L} = \sum_{t=1}^{T} \left( \log P(w_t \mid w_1, \dots, w_{t-1};\, \Theta_{\text{fwd}}) + \log P(w_t \mid w_{t+1}, \dots, w_T;\, \Theta_{\text{bwd}}) \right)`}),e.jsx(n,{type:"intuition",title:"What Each Layer Captures",content:"Peters et al. showed that different biLSTM layers capture different types of information. Layer 0 (character CNN) captures morphology and character patterns. Layer 1 captures syntax -- POS tagging benefits most from this layer. Layer 2 captures semantics -- word sense disambiguation benefits most from this layer. The learned task-specific weights allow downstream models to mix these information types optimally.",id:"note-layers"}),e.jsx(r,{title:"Context-Dependent Representations",problem:"Show how ELMo produces different vectors for the word 'bank' in two sentences: (A) 'He sat by the river bank' and (B) 'She went to the bank to deposit money'.",steps:[{formula:"Sentence A context: river, sat, by",explanation:"The biLSTM processes left-to-right and right-to-left, incorporating the nature-related context."},{formula:"Sentence B context: deposit, money, went",explanation:"The financial context words push the hidden states in a different direction."},{formula:"$\\mathbf{h}^A_{\\text{bank}} \\neq \\mathbf{h}^B_{\\text{bank}}$",explanation:"The resulting ELMo vectors are different despite being the same word, resolving the ambiguity."}],id:"ex-context"}),e.jsx(a,{title:"elmo_conceptual.py",id:"code-elmo",code:`import numpy as np

# Conceptual demonstration of ELMo layer combination
# (Using allennlp for real ELMo is heavyweight; we illustrate the key idea)

np.random.seed(42)
d = 256  # hidden dimension

# Simulate ELMo layer outputs for "bank" in two contexts
# Layer 0: character-level (similar for same word)
h0_river = np.random.randn(d)
h0_money = h0_river + np.random.randn(d) * 0.05  # nearly identical

# Layer 1: syntactic (somewhat different)
h1_river = np.random.randn(d) * 0.8
h1_money = np.random.randn(d) * 0.8  # different syntax context

# Layer 2: semantic (very different for polysemous words)
h2_river = np.random.randn(d) * 0.6
h2_money = np.random.randn(d) * 0.6  # very different meaning

# Task-specific weights (learned during fine-tuning)
# For sentiment analysis, semantics matters most
sentiment_weights = np.array([0.1, 0.2, 0.7])  # favor layer 2
sentiment_weights = np.exp(sentiment_weights) / np.exp(sentiment_weights).sum()
gamma = 1.2  # task scalar

def elmo_combine(h0, h1, h2, weights, gamma):
    """Compute task-specific ELMo representation."""
    return gamma * (weights[0] * h0 + weights[1] * h1 + weights[2] * h2)

elmo_river = elmo_combine(h0_river, h1_river, h2_river, sentiment_weights, gamma)
elmo_money = elmo_combine(h0_money, h1_money, h2_money, sentiment_weights, gamma)

cos = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("ELMo representations for 'bank':")
print(f"  Cosine similarity (river vs money context): {cos(elmo_river, elmo_money):.3f}")
print(f"  Layer 0 similarity (char-level): {cos(h0_river, h0_money):.3f}")
print(f"  Layer 2 similarity (semantic):   {cos(h2_river, h2_money):.3f}")

# For POS tagging, syntax matters more
pos_weights = np.array([0.1, 0.7, 0.2])  # favor layer 1
pos_weights = np.exp(pos_weights) / np.exp(pos_weights).sum()

elmo_river_pos = elmo_combine(h0_river, h1_river, h2_river, pos_weights, gamma)
elmo_money_pos = elmo_combine(h0_money, h1_money, h2_money, pos_weights, gamma)

print(f"\\nWith POS-tagging weights:")
print(f"  Cosine similarity: {cos(elmo_river_pos, elmo_money_pos):.3f}")
print("  (Higher because both are nouns -- syntax is more similar than semantics)")`}),e.jsx(o,{title:"ELMo is Not an Encoder",content:"ELMo is a feature extractor, not a fine-tunable encoder. The biLSTM weights are frozen after pre-training; only the layer mixing weights and scalar are learned per task. This makes it fast to adapt but limits its expressiveness compared to fully fine-tuned models like BERT.",id:"warn-frozen"}),e.jsx(n,{type:"historical",title:"From ELMo to BERT",content:"ELMo's success in 2018 (improving state-of-the-art on 6 NLP benchmarks by simply concatenating ELMo vectors to existing model inputs) demonstrated the power of pre-trained contextual representations. This directly inspired BERT (2018), which replaced LSTMs with Transformers and introduced full fine-tuning, further advancing the paradigm.",id:"note-to-bert"})]})}const M=Object.freeze(Object.defineProperty({__proto__:null,default:g},Symbol.toStringTag,{value:"Module"}));function f(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Sentence & Document Embeddings"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"While word embeddings capture the meaning of individual tokens, many applications require fixed-size representations of entire sentences, paragraphs, or documents. The challenge is to compose variable-length sequences of word vectors into a single vector that preserves semantic meaning for tasks like semantic search, clustering, and similarity comparison."}),e.jsx(i,{title:"Sentence Embedding",definition:"A sentence embedding is a function $f: \\mathcal{S} \\to \\mathbb{R}^d$ mapping a variable-length sequence of tokens to a fixed-dimensional vector such that semantically similar sentences are mapped to nearby points: $\\text{sim}(s_1, s_2) \\text{ high} \\implies \\cos(f(s_1), f(s_2)) \\text{ high}$.",id:"def-sent-emb"}),e.jsx(i,{title:"Average Pooling Baseline",definition:"The simplest sentence embedding averages the word embeddings: $\\mathbf{s} = \\frac{1}{n}\\sum_{i=1}^{n} \\mathbf{v}_{w_i}$. Despite its simplicity, this often works surprisingly well, especially with weighted variants like SIF (Smooth Inverse Frequency) that down-weight common words: $\\mathbf{s} = \\frac{1}{n}\\sum_{i=1}^{n} \\frac{a}{a + p(w_i)} \\mathbf{v}_{w_i}$ where $a$ is a hyperparameter and $p(w_i)$ is word frequency.",id:"def-avg-pool"}),e.jsx(a,{title:"sentence_embedding_basics.py",id:"code-avg",code:`import numpy as np
from collections import Counter

# Simulated word embeddings (50-dim)
np.random.seed(42)
d = 50
word_vecs = {
    "the": np.random.randn(d) * 0.1,
    "cat": np.random.randn(d) + np.array([1]*25 + [0]*25),
    "sat": np.random.randn(d) * 0.5,
    "on": np.random.randn(d) * 0.1,
    "mat": np.random.randn(d) + np.array([0]*25 + [1]*25),
    "dog": np.random.randn(d) + np.array([1]*25 + [0.1]*25),
    "lay": np.random.randn(d) * 0.5,
    "rug": np.random.randn(d) + np.array([0]*25 + [0.9]*25),
}

# Word frequencies for SIF weighting
word_freq = {"the": 0.07, "cat": 0.001, "sat": 0.002, "on": 0.03,
             "mat": 0.0005, "dog": 0.002, "lay": 0.001, "rug": 0.0004}

def avg_embedding(sentence, word_vecs):
    """Simple average of word vectors."""
    words = sentence.lower().split()
    vecs = [word_vecs[w] for w in words if w in word_vecs]
    return np.mean(vecs, axis=0) if vecs else np.zeros(d)

def sif_embedding(sentence, word_vecs, word_freq, a=1e-3):
    """SIF-weighted average (Arora et al., 2017)."""
    words = sentence.lower().split()
    vecs = []
    for w in words:
        if w in word_vecs:
            weight = a / (a + word_freq.get(w, 1e-4))
            vecs.append(weight * word_vecs[w])
    return np.mean(vecs, axis=0) if vecs else np.zeros(d)

# Compare three sentences
s1 = "the cat sat on the mat"
s2 = "the dog lay on the rug"
s3 = "the mat sat on the cat"  # same words, different meaning!

cos = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

print("=== Simple Average ===")
e1, e2, e3 = avg_embedding(s1, word_vecs), avg_embedding(s2, word_vecs), avg_embedding(s3, word_vecs)
print(f"sim(s1, s2) = {cos(e1, e2):.3f}  (similar meaning)")
print(f"sim(s1, s3) = {cos(e1, e3):.3f}  (same words, different meaning)")

print("\\n=== SIF Weighted ===")
e1, e2, e3 = sif_embedding(s1, word_vecs, word_freq), sif_embedding(s2, word_vecs, word_freq), sif_embedding(s3, word_vecs, word_freq)
print(f"sim(s1, s2) = {cos(e1, e2):.3f}")
print(f"sim(s1, s3) = {cos(e1, e3):.3f}")
print("\\nNote: averaging ignores word order, so s1 and s3 are identical!")`}),e.jsx(o,{title:"Bag-of-Words Loses Word Order",content:"Averaging word embeddings creates a bag-of-words representation that is invariant to word order. 'The cat sat on the mat' and 'The mat sat on the cat' produce identical embeddings. For tasks where order matters, use encoder-based models like Sentence-BERT.",id:"warn-order"}),e.jsx(i,{title:"Sentence-BERT (SBERT)",definition:"Sentence-BERT (Reimers & Gurevych, 2019) fine-tunes a pre-trained BERT model using siamese and triplet networks to produce semantically meaningful sentence embeddings. It uses mean pooling over BERT token outputs and trains with a combination of classification (NLI) and regression (STS) objectives, making cosine similarity directly meaningful.",id:"def-sbert"}),e.jsx(a,{title:"sentence_transformers_demo.py",id:"code-sbert",code:`# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pre-trained sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode sentences
sentences = [
    "The cat sat on the mat.",
    "A dog was lying on the rug.",
    "Machine learning is a subfield of AI.",
    "Neural networks learn from data.",
    "The feline rested upon the carpet.",  # paraphrase of sentence 1
]

embeddings = model.encode(sentences)
print(f"Embedding shape: {embeddings.shape}")  # (5, 384)

# Compute pairwise cosine similarities
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)

print("\\nPairwise cosine similarities:")
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        print(f"  [{i}] vs [{j}]: {sim_matrix[i][j]:.3f}  "
              f"({sentences[i][:30]}... vs {sentences[j][:30]}...)")

# Semantic search: find most similar to a query
query = "A pet was resting on the floor"
query_emb = model.encode([query])
scores = cosine_similarity(query_emb, embeddings)[0]
best = np.argmax(scores)
print(f"\\nQuery: '{query}'")
print(f"Best match: '{sentences[best]}' (score: {scores[best]:.3f})")`}),e.jsx(n,{type:"tip",title:"Choosing a Sentence Embedding Model",content:"For general-purpose English tasks, 'all-MiniLM-L6-v2' offers an excellent speed/quality trade-off (384 dims, 80MB). For maximum quality, use 'all-mpnet-base-v2' (768 dims). For multilingual support, use 'paraphrase-multilingual-MiniLM-L12-v2'. For instruction-following embeddings (where you describe the task), consider Instructor or E5 models.",id:"note-choosing"}),e.jsx(n,{type:"note",title:"Beyond Sentence-BERT",content:"Recent developments include instruction-tuned embeddings (Instructor, E5) that condition on task descriptions, and Matryoshka embeddings that support flexible dimensionality -- you can truncate the vector to any prefix length (e.g., 64, 128, 256 dims) while maintaining quality, enabling cost-quality trade-offs at query time.",id:"note-beyond"})]})}const N=Object.freeze(Object.defineProperty({__proto__:null,default:f},Symbol.toStringTag,{value:"Module"}));function u(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Analogies & Vector Arithmetic"}),e.jsxs("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:["One of the most striking properties of word embeddings is that semantic relationships are encoded as consistent vector offsets. The famous example"," ",e.jsx(t.InlineMath,{math:"\\vec{\\text{king}} - \\vec{\\text{man}} + \\vec{\\text{woman}} \\approx \\vec{\\text{queen}}"})," ",'demonstrates that the direction from "man" to "woman" captures a gender relationship that can be transferred to other word pairs. This section explores the mathematics and limitations of embedding analogies.']}),e.jsx(i,{title:"Vector Analogy",definition:"An analogy 'a is to b as c is to ?' is solved by finding the word $d$ whose embedding is closest to $\\mathbf{v}_b - \\mathbf{v}_a + \\mathbf{v}_c$: $d^* = \\arg\\max_{d \\in \\mathcal{V} \\setminus \\{a,b,c\\}} \\cos(\\mathbf{v}_d, \\mathbf{v}_b - \\mathbf{v}_a + \\mathbf{v}_c)$.",notation:"The offset $\\mathbf{v}_b - \\mathbf{v}_a$ encodes the relationship between $a$ and $b$. Adding this offset to $\\mathbf{v}_c$ yields a target point near $\\mathbf{v}_d$.",id:"def-analogy"}),e.jsx(r,{title:"Classic Analogy Examples",problem:"What relationships do these vector offsets encode?",steps:[{formula:"$\\vec{\\text{king}} - \\vec{\\text{man}} + \\vec{\\text{woman}} \\approx \\vec{\\text{queen}}$",explanation:"Gender relationship: the male-female offset transfers from common nouns to royalty."},{formula:"$\\vec{\\text{Paris}} - \\vec{\\text{France}} + \\vec{\\text{Italy}} \\approx \\vec{\\text{Rome}}$",explanation:"Capital-country relationship: a geographic relational offset."},{formula:"$\\vec{\\text{walking}} - \\vec{\\text{walk}} + \\vec{\\text{swim}} \\approx \\vec{\\text{swimming}}$",explanation:"Morphological relationship: the tense offset generalizes across verbs."}],id:"ex-analogies"}),e.jsx("p",{className:"text-gray-700 dark:text-gray-300",children:"The analogy works because the cosine similarity objective can be decomposed:"}),e.jsx(t.BlockMath,{math:String.raw`\cos(\mathbf{v}_d, \mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c) \propto \mathbf{v}_d^\top\mathbf{v}_b - \mathbf{v}_d^\top\mathbf{v}_a + \mathbf{v}_d^\top\mathbf{v}_c`}),e.jsxs("p",{className:"text-gray-700 dark:text-gray-300",children:["This means we seek a word ",e.jsx(t.InlineMath,{math:"d"})," that is similar to ",e.jsx(t.InlineMath,{math:"b"})," and"," ",e.jsx(t.InlineMath,{math:"c"})," but dissimilar from ",e.jsx(t.InlineMath,{math:"a"}),"."]}),e.jsx(a,{title:"analogy_demo.py",id:"code-analogy",code:`import numpy as np
from gensim.models import KeyedVectors

# Load pre-trained Word2Vec (Google News, 300-dim)
# Download: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM
# wv = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# For demo: simulate with small random embeddings
np.random.seed(42)
d = 100
words = ["king", "queen", "man", "woman", "prince", "princess",
         "paris", "france", "rome", "italy", "berlin", "germany",
         "walk", "walking", "swim", "swimming", "run", "running"]

# Create embeddings with structure
vecs = {}
gender_dir = np.random.randn(d)
gender_dir /= np.linalg.norm(gender_dir)
royal_dir = np.random.randn(d)
royal_dir /= np.linalg.norm(royal_dir)

base = np.random.randn(d) * 0.3
vecs["man"]   = base + 0.0 * gender_dir + 0.0 * royal_dir
vecs["woman"] = base + 1.0 * gender_dir + 0.0 * royal_dir
vecs["king"]  = base + 0.0 * gender_dir + 1.0 * royal_dir
vecs["queen"] = base + 1.0 * gender_dir + 1.0 * royal_dir
# Add noise to all
for w in vecs:
    vecs[w] += np.random.randn(d) * 0.1

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def solve_analogy(a, b, c, vecs, exclude=None):
    """Solve: a is to b as c is to ?"""
    target = vecs[b] - vecs[a] + vecs[c]
    exclude = exclude or {a, b, c}
    best_word, best_sim = None, -1
    for w, v in vecs.items():
        if w in exclude:
            continue
        sim = cosine_sim(target, v)
        if sim > best_sim:
            best_word, best_sim = w, sim
    return best_word, best_sim

# Test: man -> woman :: king -> ?
answer, score = solve_analogy("man", "woman", "king", vecs)
print(f"man : woman :: king : {answer} (sim={score:.3f})")

# Show the offset vectors
gender_offset = vecs["woman"] - vecs["man"]
royal_offset = vecs["queen"] - vecs["king"]
print(f"\\nCosine(woman-man, queen-king) = {cosine_sim(gender_offset, royal_offset):.3f}")
print("(High similarity = consistent gender direction)")`}),e.jsx(o,{title:"Analogies Are Not Always Reliable",content:"The analogy task has a success rate of only about 60-75% on standard benchmarks, even with good embeddings. Many failures stem from polysemy (words with multiple meanings), frequency imbalances, or relationships that are not well-captured by linear offsets. Do not treat analogy accuracy as the sole measure of embedding quality.",id:"warn-reliability"}),e.jsx(n,{type:"note",title:"Alternative: 3CosAdd vs 3CosMul",content:"The standard additive method (3CosAdd) can be improved by the multiplicative method (3CosMul) proposed by Levy & Goldberg (2014): d* = argmax cos(d,b)*cos(d,c) / (cos(d,a) + epsilon). This avoids the issue where one large similarity term can dominate the additive formulation.",id:"note-cosmul"}),e.jsx(n,{type:"intuition",title:"Why Linear Offsets Emerge",content:"Levy & Goldberg (2014) showed that Word2Vec implicitly factorizes a PMI matrix, and PMI differences correspond to log-probability ratios. When a consistent relationship (like gender) shifts co-occurrence patterns by a constant factor across word pairs, it manifests as a constant vector offset in the embedding space. The linearity is a consequence of the log-bilinear structure of the training objective.",id:"note-why-linear"})]})}const C=Object.freeze(Object.defineProperty({__proto__:null,default:u},Symbol.toStringTag,{value:"Module"}));function b(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Bias in Word Embeddings"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:'Word embeddings trained on large corpora absorb and amplify societal biases present in the training text. Bolukbasi et al. (2016) demonstrated that embeddings encode harmful stereotypes: "man is to computer programmer as woman is to homemaker." Understanding and mitigating these biases is critical when embeddings are used in real-world applications like hiring, search, and recommendation systems.'}),e.jsx(i,{title:"Embedding Bias",definition:"Bias in embeddings manifests as systematic geometric relationships between word vectors that reflect societal stereotypes rather than definitional facts. Formally, a gender bias exists when $\\cos(\\vec{w}, \\vec{he} - \\vec{she})$ is large for stereotypically male words (e.g., 'surgeon') and negative for stereotypically female words (e.g., 'nurse'), even though these associations are not definitional.",id:"def-bias"}),e.jsx(r,{title:"Detecting Gender Bias via Analogies",problem:"The analogy framework reveals bias when it completes stereotypical associations.",steps:[{formula:"$\\vec{\\text{man}} - \\vec{\\text{woman}} + \\vec{\\text{nurse}} \\approx \\vec{\\text{doctor}}$",explanation:"Implies nurse is to woman as doctor is to man -- a harmful stereotype."},{formula:"$\\vec{\\text{he}} - \\vec{\\text{she}} + \\vec{\\text{receptionist}} \\approx \\vec{\\text{boss}}$",explanation:"Encodes a gendered occupational hierarchy."},{formula:"$\\cos(\\vec{\\text{engineer}}, \\vec{\\text{he}}) > \\cos(\\vec{\\text{engineer}}, \\vec{\\text{she}})$",explanation:'Direct similarity measurement shows "engineer" is closer to male pronouns.'}],id:"ex-bias-analogies"}),e.jsx(i,{title:"WEAT (Word Embedding Association Test)",definition:"WEAT (Caliskan et al., 2017) measures bias by computing the differential association between two sets of target words (e.g., male vs female names) and two sets of attribute words (e.g., career vs family terms): $s(X,Y,A,B) = \\\\sum_{x \\\\in X} s(x,A,B) - \\\\sum_{y \\\\in Y} s(y,A,B)$ where $s(w,A,B) = \\\\text{mean}_{a \\\\in A} \\\\cos(w,a) - \\\\text{mean}_{b \\\\in B} \\\\cos(w,b)$.",id:"def-weat"}),e.jsx(a,{title:"bias_detection.py",id:"code-bias",code:`import numpy as np

np.random.seed(42)
d = 100

# Simulate embeddings with an embedded gender direction
gender_dir = np.random.randn(d)
gender_dir /= np.linalg.norm(gender_dir)

def make_vec(base_seed, gender_component=0.0):
    np.random.seed(base_seed)
    v = np.random.randn(d) * 0.5
    v += gender_component * gender_dir
    return v / np.linalg.norm(v)

# Words with stereotypical gender associations (from training data)
word_vecs = {
    "he":           make_vec(1, gender_component=-1.0),
    "she":          make_vec(2, gender_component=1.0),
    "engineer":     make_vec(3, gender_component=-0.5),
    "nurse":        make_vec(4, gender_component=0.6),
    "doctor":       make_vec(5, gender_component=-0.3),
    "teacher":      make_vec(6, gender_component=0.3),
    "programmer":   make_vec(7, gender_component=-0.6),
    "receptionist": make_vec(8, gender_component=0.5),
    "scientist":    make_vec(9, gender_component=-0.4),
    "librarian":    make_vec(10, gender_component=0.4),
}

cos = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Measure gender direction projection for each occupation
gender_direction = word_vecs["he"] - word_vecs["she"]
gender_direction /= np.linalg.norm(gender_direction)

print("Gender bias scores (projection onto he-she direction):")
print("(Negative = male-biased, Positive = female-biased)")
print("-" * 50)
occupations = ["engineer", "nurse", "doctor", "teacher",
               "programmer", "receptionist", "scientist", "librarian"]
for occ in occupations:
    proj = np.dot(word_vecs[occ], gender_direction)
    bar = "+" * int(abs(proj) * 30)
    direction = "M" if proj < 0 else "F"
    print(f"  {occ:15s}: {proj:+.3f} [{direction}] {bar}")

# Simple debiasing: remove gender component from occupation words
print("\\n--- After Hard Debiasing ---")
for occ in occupations:
    v = word_vecs[occ]
    # Remove projection onto gender direction
    v_debiased = v - np.dot(v, gender_direction) * gender_direction
    proj = np.dot(v_debiased, gender_direction)
    print(f"  {occ:15s}: {proj:+.6f}")`}),e.jsx(i,{title:"Hard Debiasing (Bolukbasi et al., 2016)",definition:"Hard debiasing neutralizes bias by (1) identifying a gender subspace via PCA on gendered word pairs, (2) zeroing out the gender component for 'neutral' words (occupations, adjectives), and (3) equalizing definitional pairs (e.g., he/she, king/queen) to be equidistant from neutral words. For word $w$ and gender direction $g$: $w_{\\text{debiased}} = w - (w \\cdot g)\\, g$.",id:"def-hard-debias"}),e.jsx(o,{title:"Debiasing Has Limitations",content:"Gonen & Goldberg (2019) showed that hard debiasing is 'lipstick on a pig': while direct bias measures decrease, the debiased embeddings still cluster words by gender. Neighborhood-based metrics reveal that most of the bias information remains encoded in indirect geometric relationships. More sophisticated approaches like contextual debiasing and data-level interventions are needed.",id:"warn-lipstick"}),e.jsx(n,{type:"note",title:"Types of Bias",content:"Embedding bias extends well beyond gender. Studies have documented racial bias (African-American names associated with negative attributes), religious bias, age bias, and disability bias in standard embeddings. The WEAT framework can measure any of these by choosing appropriate target and attribute word sets.",id:"note-types"}),e.jsx(n,{type:"tip",title:"Best Practices",content:"Always audit embeddings for bias before deployment. Use multiple measurement methods (WEAT, analogy tests, cluster analysis). Consider training on more balanced corpora, applying post-hoc debiasing, and evaluating downstream task fairness rather than just intrinsic embedding metrics. Document known biases in model cards.",id:"note-practices"})]})}const P=Object.freeze(Object.defineProperty({__proto__:null,default:b},Symbol.toStringTag,{value:"Module"}));function w(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Evaluation Methods for Word Embeddings"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"How do we know if one set of word embeddings is better than another? Evaluation methods fall into two categories: intrinsic evaluations that test embedding properties directly (similarity, analogies) and extrinsic evaluations that measure performance on downstream NLP tasks. A good embedding should perform well on both, though they do not always agree."}),e.jsx(i,{title:"Intrinsic Evaluation",definition:"Intrinsic evaluation measures the quality of embeddings directly by testing geometric properties of the vector space. Common intrinsic tasks include word similarity (correlating cosine similarity with human judgments), analogy completion ($a:b :: c:?$), and categorization (clustering words into semantic categories).",id:"def-intrinsic"}),e.jsx(i,{title:"Extrinsic Evaluation",definition:"Extrinsic evaluation measures embedding quality indirectly by using embeddings as features in a downstream task (e.g., sentiment analysis, NER, parsing) and comparing task performance. Better embeddings should yield better downstream results, though the relationship is not always monotonic.",id:"def-extrinsic"}),e.jsx(r,{title:"Word Similarity Benchmarks",problem:"Key benchmarks for measuring word similarity correlation:",steps:[{formula:"WordSim-353 (Finkelstein et al., 2001)",explanation:'353 word pairs rated 0-10 by humans. Mixes similarity and relatedness (e.g., "car"-"gasoline" rated high).'},{formula:"SimLex-999 (Hill et al., 2015)",explanation:'999 pairs rated for genuine similarity (not just relatedness). "Car"-"gasoline" scores low; "car"-"automobile" scores high.'},{formula:"MEN (Bruni et al., 2014)",explanation:"3,000 pairs with crowd-sourced ratings. Large size gives more statistical power."},{formula:"Metric: Spearman $\\rho$ correlation",explanation:"Rank correlation between human scores and cosine similarities. Higher is better."}],id:"ex-benchmarks"}),e.jsx(n,{type:"intuition",title:"Similarity vs Relatedness",content:"WordSim-353 conflates two distinct notions: similarity (car/automobile -- same concept) and relatedness (car/gasoline -- associated but different). SimLex-999 was specifically designed to test only similarity. This distinction matters because different embeddings may excel at one but not the other, and downstream tasks may need either.",id:"note-sim-rel"}),e.jsx(a,{title:"embedding_evaluation.py",id:"code-eval",code:`import numpy as np
from scipy.stats import spearmanr

# Simulated word similarity dataset (like SimLex-999)
# Format: (word1, word2, human_score)
similarity_pairs = [
    ("happy", "joyful", 9.2),
    ("happy", "sad", 1.5),
    ("car", "automobile", 8.8),
    ("car", "gasoline", 3.1),  # related but not similar
    ("dog", "cat", 6.5),
    ("dog", "computer", 0.8),
    ("king", "queen", 7.0),
    ("king", "throne", 4.2),   # related but not similar
    ("big", "large", 8.5),
    ("big", "small", 2.0),
]

# Simulated embeddings (in practice, load real ones)
np.random.seed(42)
d = 50

def make_similar_vecs(base_seed, similarity):
    """Create two vectors with approximately given cosine similarity."""
    np.random.seed(base_seed)
    v1 = np.random.randn(d)
    v1 /= np.linalg.norm(v1)
    # Mix v1 with random vector to control similarity
    noise = np.random.randn(d)
    noise /= np.linalg.norm(noise)
    v2 = similarity * v1 + (1 - abs(similarity)) * noise
    v2 /= np.linalg.norm(v2)
    return v1, v2

# Generate embeddings that roughly correlate with human scores
word_vecs = {}
for i, (w1, w2, score) in enumerate(similarity_pairs):
    target_sim = (score / 10.0) * 0.8 + np.random.randn() * 0.1
    v1, v2 = make_similar_vecs(i * 100, target_sim)
    word_vecs[w1] = v1
    word_vecs[w2] = v2

# Compute cosine similarities
cos = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

human_scores = []
model_scores = []
print(f"{'Pair':30s} {'Human':>6s} {'Cosine':>7s}")
print("-" * 48)
for w1, w2, score in similarity_pairs:
    model_sim = cos(word_vecs[w1], word_vecs[w2])
    human_scores.append(score)
    model_scores.append(model_sim)
    print(f"{w1 + ' / ' + w2:30s} {score:6.1f} {model_sim:7.3f}")

# Spearman rank correlation
rho, pvalue = spearmanr(human_scores, model_scores)
print(f"\\nSpearman rho: {rho:.3f} (p={pvalue:.4f})")
print(f"Interpretation: {'strong' if rho > 0.6 else 'moderate' if rho > 0.4 else 'weak'} correlation")

# Analogy evaluation (accuracy metric)
print("\\n--- Analogy Accuracy ---")
print("Standard benchmarks: Google Analogy (19,544 pairs)")
print("Categories: semantic (capital-country, gender) + syntactic (tense, plural)")
print("Typical accuracy: Word2Vec ~75%, GloVe ~75%, FastText ~78%")`}),e.jsx(o,{title:"Intrinsic-Extrinsic Disconnect",content:"Higher intrinsic scores do not always predict better downstream performance. Embeddings optimized for word similarity may not be optimal for NER or parsing. Chiu et al. (2016) showed that intrinsic metrics explain only a fraction of the variance in extrinsic task performance. Always evaluate on your actual downstream task.",id:"warn-disconnect"}),e.jsx(n,{type:"note",title:"The MTEB Benchmark",content:"The Massive Text Embedding Benchmark (MTEB, Muennighoff et al., 2023) provides a comprehensive evaluation framework covering 8 task types (classification, clustering, pair classification, reranking, retrieval, STS, summarization, and zero-shot classification) across 58 datasets and 112 languages. It is the current standard for comparing embedding models.",id:"note-mteb"}),e.jsx(n,{type:"tip",title:"Practical Evaluation Advice",content:"Start with MTEB scores to shortlist candidate models. Then evaluate on a held-out sample from your specific domain. Key metrics to check: retrieval recall@k for search applications, Spearman correlation for similarity tasks, and clustering quality (V-measure) for topic modeling. Always compare against a simple TF-IDF baseline.",id:"note-practical"})]})}const B=Object.freeze(Object.defineProperty({__proto__:null,default:w},Symbol.toStringTag,{value:"Module"}));function x(){return e.jsxs("div",{className:"mx-auto max-w-4xl space-y-8 px-4 py-8",children:[e.jsx("h1",{className:"text-3xl font-bold",children:"Domain-Specific Embeddings"}),e.jsx("p",{className:"text-lg text-gray-700 dark:text-gray-300",children:"General-purpose embeddings trained on Wikipedia or news corpora often perform poorly on specialized domains like biomedicine, law, or finance. Domain-specific vocabulary, jargon, and word sense distributions differ significantly from general text. Training or adapting embeddings on domain corpora can yield substantial improvements for specialized NLP tasks."}),e.jsx(i,{title:"Domain Adaptation of Embeddings",definition:"Domain adaptation for embeddings involves either (1) training from scratch on a domain-specific corpus, (2) fine-tuning pre-trained general embeddings on domain text, or (3) combining general and domain embeddings via concatenation or retrofitting. The goal is to capture domain-specific semantics: in medicine, 'discharge' means leaving a hospital, not electrical discharge.",id:"def-domain-adapt"}),e.jsx(r,{title:"Domain-Specific Nearest Neighbors",problem:"The word 'cell' has very different neighbors in general vs. biomedical embeddings.",steps:[{formula:"General embeddings: cell -> phone, battery, jail, prison",explanation:'General text is dominated by "cell phone" and "prison cell" contexts.'},{formula:"BioWordVec: cell -> cells, cellular, tissue, lymphocyte",explanation:"Biomedical text is dominated by biological cell contexts."},{formula:"Financial embeddings: cell -> spreadsheet, table, column, row",explanation:'In financial documents, "cell" refers to spreadsheet cells.'}],id:"ex-neighbors"}),e.jsx(n,{type:"note",title:"Notable Domain-Specific Embeddings",content:"BioWordVec/BioSentVec: trained on PubMed + MIMIC-III clinical notes for biomedical NLP. SciBERT: BERT pre-trained on Semantic Scholar papers for scientific text. LegalBERT: trained on legal corpora (court opinions, legislation). FinBERT: trained on financial news and SEC filings. ClinicalBERT: trained on clinical notes for healthcare applications.",id:"note-examples"}),e.jsx(a,{title:"domain_embedding_training.py",id:"code-domain",code:`import numpy as np
from gensim.models import Word2Vec
from collections import Counter

# Simulate a domain-specific corpus (biomedical)
bio_corpus = [
    "the patient presented with acute myocardial infarction".split(),
    "echocardiography revealed left ventricular dysfunction".split(),
    "treatment included aspirin and beta blockers".split(),
    "the cell culture showed abnormal proliferation".split(),
    "biopsy confirmed malignant cell growth in tissue".split(),
    "blood pressure was elevated at admission".split(),
    "the patient was discharged after seven days".split(),
    "lab results showed elevated white blood cell count".split(),
    "ct scan revealed no metastatic lesions".split(),
    "follow up mri confirmed treatment response".split(),
]

# Train domain-specific Word2Vec
bio_model = Word2Vec(
    sentences=bio_corpus,
    vector_size=50,
    window=5,
    min_count=1,
    sg=1,
    epochs=200,
    seed=42,
)

# General corpus for comparison
general_corpus = [
    "the cat sat on the mat in the room".split(),
    "she went to the bank to deposit money".split(),
    "the cell phone battery was dead".split(),
    "he was discharged from the army last year".split(),
    "blood is thicker than water they say".split(),
    "the patient customer waited in line".split(),
    "treatment of guests was excellent at the hotel".split(),
    "the growth of the company exceeded expectations".split(),
]

general_model = Word2Vec(
    sentences=general_corpus,
    vector_size=50,
    window=5,
    min_count=1,
    sg=1,
    epochs=200,
    seed=42,
)

# Compare nearest neighbors
def show_neighbors(model, word, topn=3, label=""):
    if word in model.wv:
        neighbors = model.wv.most_similar(word, topn=topn)
        nlist = ", ".join(f"{w}({s:.2f})" for w, s in neighbors)
        print(f"  [{label}] {word} -> {nlist}")
    else:
        print(f"  [{label}] {word} -> not in vocabulary")

print("Nearest neighbors comparison:")
for word in ["cell", "patient", "treatment", "blood"]:
    show_neighbors(bio_model, word, label="Bio")
    show_neighbors(general_model, word, label="Gen")
    print()`}),e.jsx(a,{title:"embedding_retrofit.py",id:"code-retrofit",code:`import numpy as np

# Retrofitting: adjust pre-trained embeddings using a domain lexicon
# (Faruqui et al., 2015)

np.random.seed(42)
d = 50

# Simulated pre-trained embeddings
vocab = ["aspirin", "ibuprofen", "tylenol", "headache", "fever",
         "inflammation", "pain", "tablet", "dose", "prescription"]
original = {w: np.random.randn(d) * 0.5 for w in vocab}

# Domain knowledge: synonym pairs from medical ontology (e.g., UMLS)
synonyms = [
    ("aspirin", "ibuprofen"),     # both NSAIDs
    ("headache", "pain"),          # related symptoms
    ("fever", "inflammation"),     # related conditions
    ("tablet", "dose"),            # drug administration
    ("tylenol", "aspirin"),        # both analgesics
]

def retrofit(embeddings, lexicon, n_iters=10, alpha=1.0):
    """Retrofit embeddings to bring synonyms closer together."""
    new_vecs = {w: v.copy() for w, v in embeddings.items()}

    # Build adjacency list
    neighbors = {w: [] for w in embeddings}
    for w1, w2 in lexicon:
        if w1 in neighbors and w2 in neighbors:
            neighbors[w1].append(w2)
            neighbors[w2].append(w1)

    for iteration in range(n_iters):
        for word in embeddings:
            if not neighbors[word]:
                continue
            # Average of original vector and synonym vectors
            neighbor_sum = sum(new_vecs[n] for n in neighbors[word])
            num_neighbors = len(neighbors[word])
            new_vecs[word] = (alpha * embeddings[word] + neighbor_sum) / (alpha + num_neighbors)

    return new_vecs

cos = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

retrofitted = retrofit(original, synonyms)

print("Similarity changes after retrofitting:")
print(f"{'Pair':30s} {'Before':>8s} {'After':>8s} {'Change':>8s}")
print("-" * 58)
for w1, w2 in synonyms:
    before = cos(original[w1], original[w2])
    after = cos(retrofitted[w1], retrofitted[w2])
    print(f"{w1 + ' / ' + w2:30s} {before:8.3f} {after:8.3f} {after-before:+8.3f}")`}),e.jsx(o,{title:"Domain Data Scarcity",content:"Domain-specific corpora are often much smaller than general corpora (millions vs billions of tokens). Training embeddings from scratch on small corpora produces noisy vectors. Prefer fine-tuning pre-trained embeddings or using retrofitting when domain data is limited. For very specialized domains, consider using pre-trained contextual models (e.g., SciBERT, BioBERT) instead of static embeddings.",id:"warn-scarcity"}),e.jsx(n,{type:"tip",title:"Practical Recommendations",content:"Start with a pre-trained model closest to your domain (e.g., PubMedBERT for biomedical). If no domain model exists, fine-tune a general model on your corpus with a low learning rate. Use domain ontologies (UMLS for medicine, FIBO for finance) for retrofitting. Always evaluate on domain-specific benchmarks, not general ones like SimLex-999.",id:"note-practical"})]})}const q=Object.freeze(Object.defineProperty({__proto__:null,default:x},Symbol.toStringTag,{value:"Module"}));export{j as a,$ as b,S as c,T as d,E as e,M as f,N as g,C as h,P as i,B as j,q as k,k as s};
