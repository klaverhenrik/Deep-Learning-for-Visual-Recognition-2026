# Lecture 12
# Vision Transformers

*Deep Learning for Visual Recognition · Aarhus University*

These notes build the Vision Transformer from first principles: starting with why CNNs have inductive biases, introducing the attention mechanism as a differentiable database lookup, assembling the Transformer encoder layer by layer, and then showing how images are turned into token sequences for ViT. The Swin Transformer's solution to the quadratic attention bottleneck is derived carefully, and every major concept is grounded in working PyTorch code.

---

## 1  Inductive Bias: What CNNs Assume and What Transformers Don't

### 1.1  The Inductive Bias of CNNs

Every machine learning model encodes inductive biases — assumptions about the structure of the problem that are baked into the architecture before any training happens. CNNs have two strong inductive biases:

- **Locality**: Convolutional filters operate on small local patches. The model assumes that nearby pixels are more related than distant ones. This is why early layers detect edges and textures — spatially local patterns — while global relationships must be built up gradually through many layers.
- **Translation equivariance**: The same filter is applied at every spatial location (parameter sharing). This assumes that a feature detector (e.g. an edge detector) should work the same way wherever the feature appears in the image.

These biases are incredibly useful when training data is limited — the model does not need to learn that spatial proximity implies correlation, because it is hard-coded. But they are also limitations: there are tasks where distant pixels are highly informative about each other (a dog's ear tells you a lot about where its nose will be), and tasks where texture is a poor classification signal (a photo of an elephant on sand and an elephant on grass may differ more in texture than in semantic content).

### 1.2  Transformers: Less Bias, More Flexibility

Transformers were originally designed for NLP (Vaswani et al., 2017) and have no built-in assumptions about spatial structure. They treat the input as a flat sequence of tokens and let every token attend to every other token — a global receptive field from the very first layer. This means:

- **No locality assumption** — a token representing a corner of the image can directly attend to a token from the opposite corner in a single layer.
- **No translation equivariance** — different positions are distinguished by position embeddings rather than by sharing filter weights.
- **Fully parallelisable** — unlike RNNs, all token interactions are computed simultaneously as matrix multiplications, making Transformers very fast on modern hardware.

The trade-off: less bias means more data needed. CNNs can learn reasonable features from ImageNet's 1.2M images. A ViT trained from scratch on ImageNet alone performs worse than a ResNet of comparable size — it needs $10$–$300\times$ more data to surpass CNNs. Once pre-trained on massive datasets, however, large ViTs outperform large CNNs in both accuracy and training compute efficiency.

> **The bias-data trade-off in one sentence.** A CNN is like a physics student who already knows Newton's laws and just needs to estimate the constants. A Transformer is like a scientist who knows nothing and must infer the laws themselves — given enough data, the Transformer's inferred laws can be better, but the journey requires far more observations.

---

## 2  The Attention Mechanism

### 2.1  Attention as a Differentiable Database Lookup

The best way to understand attention is as a differentiable, learnable database lookup. Imagine you have a database of key–value pairs, and a query. A hard lookup returns the value for the single matching key. Soft attention performs a weighted retrieval: it scores the query against every key, normalises the scores with softmax, and returns a weighted sum of all values.

In neural network terms, all three — queries, keys, and values — are computed from the input tokens by learned linear projections:

$$\mathbf{K} = \mathbf{X} \mathbf{W}_k \qquad \text{(keys: what does each token contain?)}$$

$$\mathbf{V} = \mathbf{X} \mathbf{W}_v \qquad \text{(values: what information does each token carry?)}$$

$$\mathbf{Q} = \mathbf{Y} \mathbf{W}_q \qquad \text{(queries: what is each output token looking for?)}$$

where $\mathbf{X}$ is the input sequence, $\mathbf{Y}$ is the output sequence, and $\mathbf{W}_k$, $\mathbf{W}_v$, $\mathbf{W}_q$ are learned weight matrices. When $\mathbf{X} = \mathbf{Y}$ — when the query sequence is the same as the key/value sequence — this is self-attention. When the queries come from a different sequence than the keys and values — e.g. the decoder querying the encoder — this is cross-attention.

### 2.2  Scaled Dot-Product Attention

The alignment score between a query $\mathbf{q}$ and a key $\mathbf{k}$ is their dot product. After computing all pairwise dot products, we normalise with softmax to obtain attention weights, then compute the weighted sum of values:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \cdot \mathbf{V}$$

The scaling by $1/\sqrt{d_k}$ (where $d_k$ is the dimension of the key vectors) is critical. Without it, as $d_k$ grows, the dot products grow in magnitude, pushing the softmax into its flat saturation region where gradients vanish — the same problem that motivated replacing sigmoid with ReLU, just appearing in a different context. Dividing by $\sqrt{d_k}$ keeps the variance of the dot products at approximately 1.

The matrix dimensions for a sequence of $n$ tokens with embedding dimension $d$:

$$\mathbf{Q}: (n, d_k) \quad \mathbf{K}: (n, d_k) \quad \mathbf{V}: (n, d_v)$$

$$\mathbf{Q}\mathbf{K}^T: (n, n) \qquad \leftarrow \text{attention weight matrix } (n^2 \text{ entries!})$$

$$\text{softmax}(\mathbf{Q}\mathbf{K}^T / \sqrt{d_k}) \cdot \mathbf{V}: (n, d_v) \qquad \leftarrow \text{output}$$

The attention weight matrix has $n^2$ entries — the computational and memory cost of attention is quadratic in sequence length. For a $256 \times 256$ image treated as individual pixels, $n = 65{,}536$ and the attention matrix has over 4 billion entries. This is why naive pixel-level Transformers are impractical for high-resolution images, motivating the patch-based approach used by ViT.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ════════════════════════════════════════════════════════════════════
# SCALED DOT-PRODUCT ATTENTION — built from scratch
# Every line corresponds to the formula above.
# ════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: (batch, n_queries, d_k)
        K: (batch, n_keys,    d_k)
        V: (batch, n_keys,    d_v)
        mask: optional (batch, n_queries, n_keys) bool mask
    Returns:
        output: (batch, n_queries, d_v)
        weights: (batch, n_queries, n_keys)  ← for visualisation
    """
    d_k = Q.size(-1)

    # Step 1: Alignment scores — dot product of Q with every K
    # Q: (batch, n_q, d_k)  ×  Kᵀ: (batch, d_k, n_k) → (batch, n_q, n_k)
    scores = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 2: Optional masking (used in decoder to prevent attending to future tokens)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # Step 3: Normalise scores to attention weights
    weights = F.softmax(scores, dim=-1)   # (batch, n_q, n_k)

    # Step 4: Weighted sum of values
    output = torch.bmm(weights, V)         # (batch, n_q, d_v)

    return output, weights

# ── Concrete example: 3 tokens, d_k=d_v=4 ────────────────────────────
batch, n, d_k, d_v = 2, 3, 4, 4
Q = torch.randn(batch, n, d_k)
K = torch.randn(batch, n, d_k)
V = torch.randn(batch, n, d_v)

out, w = scaled_dot_product_attention(Q, K, V)
print(f'Output shape:  {out.shape}')     # (2, 3, 4)
print(f'Weights shape: {w.shape}')       # (2, 3, 3)  ← n×n attention matrix
print(f'Weights sum to 1: {w.sum(-1)}')  # each row sums to 1.0

# ── Self-attention: Q, K, V all come from the same sequence ──────────
# When X = Y, this is self-attention
W_q = nn.Linear(d_k, d_k, bias=False)
W_k = nn.Linear(d_k, d_k, bias=False)
W_v = nn.Linear(d_v, d_v, bias=False)

X = torch.randn(batch, n, d_k)   # input token embeddings
Q_self = W_q(X)
K_self = W_k(X)
V_self = W_v(X)
out_self, _ = scaled_dot_product_attention(Q_self, K_self, V_self)
print(f'Self-attention output: {out_self.shape}')  # (2, 3, 4) — same shape as input

# PyTorch 2.0+ has an optimised fused implementation:
out_fused = F.scaled_dot_product_attention(Q, K, V)  # uses Flash Attention if available
print(f'Fused attention output: {out_fused.shape}')
```

*Code 1 – Scaled dot-product attention from scratch. Each step maps to the formula: (1) compute $\mathbf{Q}\mathbf{K}^T/\sqrt{d_k}$ alignment scores, (2) optional masking, (3) softmax to get weights, (4) weighted sum of values. Note that the attention weight matrix has shape $(n, n)$ — this is the source of the $O(n^2)$ complexity.*

### 2.3  Multi-Head Attention

A single attention head computes one set of query/key/value projections and one weighted combination. Multi-head attention runs $h$ attention operations in parallel, each with different learned projections, then concatenates the results and projects back to the original dimension:

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, \mathbf{W}_O$$

$$\text{where} \quad \text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_{qi},\, \mathbf{K}\mathbf{W}_{ki},\, \mathbf{V}\mathbf{W}_{vi})$$

The intuition: different heads can attend to different aspects of the relationships between tokens simultaneously. One head might model syntactic dependencies, another semantic similarity, another spatial proximity. The final projection $\mathbf{W}_O$ blends these views into a single representation. In practice $d_k$ for each head is $d_\text{model}/h$, so total compute stays the same as a single head with dimension $d_\text{model}$.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention or cross-attention.
    All the projection matrices are shown explicitly.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads   # dimension per head

        # Separate projections for Q, K, V for all heads at once
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # (d_model → d_model)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # output projection

    def split_heads(self, x, batch):
        """Reshape (batch, seq, d_model) → (batch, n_heads, seq, d_k)."""
        x = x.view(batch, -1, self.n_heads, self.d_k)
        return x.transpose(1, 2)   # (batch, n_heads, seq, d_k)

    def forward(self, Q_in, K_in, V_in, mask=None):
        batch = Q_in.size(0)

        # 1. Project inputs to Q, K, V
        Q = self.split_heads(self.W_q(Q_in), batch)  # (batch, h, n_q, d_k)
        K = self.split_heads(self.W_k(K_in), batch)  # (batch, h, n_k, d_k)
        V = self.split_heads(self.W_v(V_in), batch)  # (batch, h, n_k, d_k)

        # 2. Attention for each head in parallel
        scores  = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        weights = F.softmax(scores, dim=-1)            # (batch, h, n_q, n_k)
        context = torch.matmul(weights, V)             # (batch, h, n_q, d_k)

        # 3. Concatenate heads and project
        context = context.transpose(1,2).contiguous() # (batch, n_q, h, d_k)
        context = context.view(batch, -1, self.n_heads * self.d_k)
        return self.W_o(context)                       # (batch, n_q, d_model)

# ── Quick test ────────────────────────────────────────────────────────
mha = MultiHeadAttention(d_model=256, n_heads=8)   # 8 heads, 32-dim each
x   = torch.randn(4, 16, 256)   # batch=4, seq_len=16, d_model=256
out = mha(x, x, x)              # self-attention: Q=K=V=x
print(f'Input shape:  {x.shape}')   # (4, 16, 256)
print(f'Output shape: {out.shape}') # (4, 16, 256) — same!

n_params = sum(p.numel() for p in mha.parameters())
print(f'Parameters: {n_params:,}')  # 4 × 256² = 262,144
```

*Code 2 – Multi-head attention from scratch, with every projection matrix made explicit. The `split_heads` reshape is the key step: it divides the `d_model` dimension into $h$ independent heads, each computing attention in a $d_k$-dimensional subspace. The output shape always equals the input shape.*

---

## 3  The Transformer Encoder

The Transformer encoder is a stack of identical blocks. Each block contains two sub-layers with residual connections and layer normalisation around each:

- Multi-head self-attention (token-to-token interaction)
- Position-wise feed-forward network (per-token non-linear transformation)

### 3.1  Layer Normalisation vs Batch Normalisation

The Transformer uses layer normalisation rather than batch normalisation. The key difference is the axis of normalisation:

- **Batch norm**: normalises across the batch dimension — computes mean and variance over all examples in the batch for each feature position. Requires a large enough batch to estimate statistics reliably. Behaves differently at train and test time.
- **Layer norm**: normalises across the feature dimension — for each individual token, computes mean and variance across its $d_\text{model}$ features. Independent of batch size. Same behaviour at train and test time. This makes it suitable for variable-length sequences and small batches, both common in NLP and ViT training.

### 3.2  The Feed-Forward Sub-layer

After self-attention, each token representation is passed through a small position-wise feed-forward network (FFN) independently — the same two-layer MLP is applied to each token separately. Typically the inner dimension is $4\times$ the model dimension:

$$\text{FFN}(\mathbf{x}) = \max(0,\, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\, \mathbf{W}_2 + \mathbf{b}_2$$

Self-attention is linear — it only recombines information across tokens but cannot transform the content of each token's representation. The FFN is what provides the non-linear transformations that allow each token to be processed into a richer representation. Without the FFN, stacking more self-attention layers would barely help.

### 3.3  Residual Connections

Each sub-layer wraps its computation in a residual connection: $\text{output} = \text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))$. This serves the same purpose as in ResNets — gradients can flow directly through the addition, mitigating vanishing gradients in deep stacks. Without residual connections, Transformer encoders deeper than a few layers are difficult to train.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block:
    x → LayerNorm → MultiHeadSelfAttention → Add → LayerNorm → FFN → Add
    (Pre-norm variant, common in modern implementations)
    """
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        # Sub-layer 1: multi-head self-attention
        self.attn    = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout, batch_first=True)
        self.norm1   = nn.LayerNorm(d_model)

        # Sub-layer 2: position-wise feed-forward network
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),               # GELU is now more common than ReLU in Transformers
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        # Pre-norm: normalise BEFORE each sub-layer
        # Sub-layer 1: self-attention (Q = K = V = x)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm,
                                 key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)   # residual connection

        # Sub-layer 2: feed-forward network
        x = x + self.dropout(self.ffn(self.norm2(x)))   # residual connection
        return x

# ── Full Transformer encoder = stack of blocks ───────────────────────
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=8, ffn_dim=1024,
                 n_layers=6, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)  # final layer norm

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# ── Shape trace through the encoder ──────────────────────────────────
encoder = TransformerEncoder(d_model=256, n_heads=8, ffn_dim=1024, n_layers=6)
x = torch.randn(4, 16, 256)   # batch=4, seq_len=16, d_model=256
out = encoder(x)
print(f'Encoder input:  {x.shape}')    # (4, 16, 256)
print(f'Encoder output: {out.shape}')  # (4, 16, 256)  — shape preserved!

# The encoder maps each token to a new representation that is informed
# by all other tokens in the sequence via self-attention.

n_params = sum(p.numel() for p in encoder.parameters())
print(f'Encoder parameters: {n_params:,}')  # ≈ 2.4M for these settings
```

*Code 3 – A complete Transformer encoder block and full encoder stack. Note GELU activation in the FFN (smoother than ReLU, widely used in Transformers), the pre-norm variant (LayerNorm before each sub-layer rather than after), and the use of `nn.LayerNorm` rather than `nn.BatchNorm`.*

---

## 4  Positional Embeddings

### 4.1  The Permutation Invariance Problem

Self-attention is permutation invariant: shuffling the order of the input tokens produces the same output embeddings (just in a different order). The attention weight from token $i$ to token $j$ depends only on their content (via $\mathbf{Q}$ and $\mathbf{K}$) — not on their positions. This means the model has no way to distinguish 'the cat sat on the mat' from 'the mat sat on the cat'. For images, this is even worse: position is everything.

The solution is to add a positional embedding to each token before it enters the Transformer. The positional embedding encodes the token's position as a vector that is added to its content embedding, giving the model access to both content and position simultaneously.

### 4.2  Sinusoidal Position Embeddings

The original Transformer paper used fixed sinusoidal embeddings. For a token at position $k$ in a sequence, its $d$-dimensional positional embedding is:

$$\text{PE}(k,\, 2i) = \sin\!\left(\frac{k}{n^{2i/d}}\right)$$

$$\text{PE}(k,\, 2i+1) = \cos\!\left(\frac{k}{n^{2i/d}}\right)$$

where $i$ indexes the dimension ($0 \leq i < d/2$) and $n = 10{,}000$. The key insight is that each pair of dimensions $(\text{PE}(k, 2i),\, \text{PE}(k, 2i+1))$ encodes position as a rotating 2D vector — as $k$ increases, the vector rotates. Different dimensions rotate at different speeds (determined by $n^{2i/d}$): small $i$ gives fast rotation (high frequency), large $i$ gives slow rotation (low frequency). Together, these encode position like a continuous analogue of a binary counter, with the crucial advantage that the encoding generalises naturally to positions longer than those seen during training.

### 4.3  Learned Position Embeddings

ViT uses a simpler approach: learn a separate positional embedding vector for each position. These are initialised randomly and trained end-to-end with the rest of the model. Learned embeddings tend to outperform sinusoidal ones slightly on standard benchmarks, but they do not generalise to longer sequences at inference time without interpolation.

```python
import torch
import torch.nn as nn
import math

# ── Sinusoidal position embeddings ────────────────────────────────────
def sinusoidal_position_embedding(seq_len, d_model, n=10000):
    """
    Compute sinusoidal positional encoding.
    Returns: (seq_len, d_model) tensor — add to token embeddings.
    """
    pe = torch.zeros(seq_len, d_model)
    k  = torch.arange(seq_len).unsqueeze(1).float()      # (seq_len, 1)
    i  = torch.arange(0, d_model, 2).float()             # even dims
    div_term = torch.pow(n, i / d_model)                 # n^(2i/d)

    pe[:, 0::2] = torch.sin(k / div_term)   # even dims → sine
    pe[:, 1::2] = torch.cos(k / div_term)   # odd dims  → cosine
    return pe

# Visualise: nearby positions have similar embeddings
pe = sinusoidal_position_embedding(seq_len=16, d_model=32)
print(f'PE shape: {pe.shape}')   # (16, 32)

# Cosine similarity between position 0 and all other positions
p0    = pe[0]
sims  = torch.nn.functional.cosine_similarity(p0.unsqueeze(0), pe, dim=1)
print('Similarity of pos 0 with pos 0,1,2,...,15:')
print([round(s.item(), 3) for s in sims])
# → nearby positions have high similarity, distant ones have low similarity

# ── Learned position embeddings (used in ViT) ─────────────────────────
class LearnedPositionEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        # One learnable vector per position — initialised randomly
        self.pe = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        """Add positional embedding to token embeddings x: (batch, seq, d)."""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)  # [0, 1, ..., n-1]
        return x + self.pe(positions)   # broadcast over batch dimension

lpe = LearnedPositionEmbedding(max_seq_len=256, d_model=128)
x   = torch.randn(4, 16, 128)   # batch=4, 16 tokens
x_with_pos = lpe(x)
print(f'Input with positions: {x_with_pos.shape}')  # (4, 16, 128) — same shape
```

*Code 4 – Sinusoidal and learned positional embeddings. The cosine similarity check confirms a key property of sinusoidal embeddings: nearby positions (0 and 1) are more similar than distant positions (0 and 15). Both approaches produce a `(seq_len, d_model)` tensor that is added to the token embeddings before entering the Transformer.*

---

## 5  Vision Transformer (ViT)

### 5.1  The Core Idea: Images as Sequences of Patches

The challenge in applying Transformers to images is the sequence length. A standard Transformer can handle sequences of a few hundred tokens; an image has thousands or millions of pixels. ViT (Dosovitskiy et al., 2020, 'An Image Is Worth 16×16 Words') solves this by treating the image not as individual pixels but as a sequence of non-overlapping patches. Each patch is flattened and linearly projected to the model dimension $d$, becoming one token:

$$\text{Image: } (H, W, C) \;\to\; N \text{ patches of size } (P, P, C)$$

$$N = \frac{H \times W}{P^2} \qquad \text{(number of patches = number of tokens)}$$

$$\text{Each patch: flatten to } (P^2 \times C)\text{, then linear projection to } d$$

For a $224 \times 224$ image with $P=16$, $N = (224/16)^2 = 196$ patches. This is manageable. The $O(n^2)$ attention cost scales with $196^2$, not $224^4$.

### 5.2  The CLS Token

For classification, ViT prepends a learnable CLS token to the sequence of patch tokens. After passing through the Transformer encoder, the CLS token's output representation aggregates information from all patches via self-attention and is passed to a classification head. The CLS token is analogous to the [CLS] token in BERT — it is not a patch from the image, but a 'receptacle' that the Transformer fills with a global summary of the image.

### 5.3  Why ViT Has a Global Receptive Field from Layer 1

A CNN's receptive field grows gradually with depth — early layers see only a $3 \times 3$ or $5 \times 5$ region, and global context only emerges after many layers. ViT's self-attention allows every patch token to attend to every other patch token in a single layer. From the very first Transformer block, each patch's representation is informed by all other patches in the image. This 'global receptive field from layer 1' is one of ViT's key structural differences from CNNs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTransformer(nn.Module):
    """
    ViT for image classification.
    Implements the full pipeline:
    image → patches → linear embedding → + position + CLS → Transformer → classify
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 n_classes=1000, d_model=768, n_heads=12, n_layers=12,
                 ffn_dim=3072, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0, 'img_size must be divisible by patch_size'
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2  # e.g. 196 for 224/16

        # ── Step 1: Patch embedding ───────────────────────────────────
        # A single Conv2d with kernel=stride=patch_size gives non-overlapping patches
        # and projects them to d_model simultaneously.
        self.patch_embed = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size
        )  # output: (batch, d_model, H/P, W/P)

        # ── Step 2: CLS token ─────────────────────────────────────────
        # Learnable 1×d_model vector prepended to the patch sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # ── Step 3: Positional embeddings ─────────────────────────────
        # n_patches + 1 because CLS token also gets a position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        self.pos_drop  = nn.Dropout(dropout)

        # ── Step 4: Transformer encoder ───────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_model)

        # ── Step 5: Classification head ───────────────────────────────
        # Only the CLS token output is used for classification
        self.head = nn.Linear(d_model, n_classes)

        # Initialise weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch = x.size(0)

        # 1. Patch embedding: (B, C, H, W) → (B, d_model, H/P, W/P)
        x = self.patch_embed(x)             # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)    # (B, N, d_model)  N=n_patches

        # 2. Prepend CLS token
        cls = self.cls_token.expand(batch, -1, -1)  # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)            # (B, N+1, d_model)

        # 3. Add positional embeddings
        x = self.pos_drop(x + self.pos_embed)       # (B, N+1, d_model)

        # 4. Transformer encoder
        x = self.transformer(x)                     # (B, N+1, d_model)
        x = self.norm(x)

        # 5. Classify using only the CLS token (index 0)
        cls_out = x[:, 0]                           # (B, d_model)
        return self.head(cls_out)                   # (B, n_classes)

# ── ViT-Base/16: the standard ViT configuration ───────────────────────
# img_size=224, patch_size=16 → 196 patches → 197 tokens (with CLS)
# d_model=768, n_heads=12, n_layers=12, ffn_dim=3072
vit = VisionTransformer(img_size=224, patch_size=16, n_classes=1000,
                         d_model=768,  n_heads=12,   n_layers=12,
                         ffn_dim=3072)
x   = torch.randn(2, 3, 224, 224)   # 2 RGB images
logits = vit(x)
print(f'Input:  {x.shape}')           # (2, 3, 224, 224)
print(f'Output: {logits.shape}')      # (2, 1000)  — class scores

n_params = sum(p.numel() for p in vit.parameters())
print(f'Parameters: {n_params/1e6:.0f}M')   # ≈ 86M for ViT-B/16

# ── A compact ViT for smaller images (e.g. CIFAR-10) ─────────────────
vit_small = VisionTransformer(img_size=32, patch_size=4, n_classes=10,
                               d_model=256, n_heads=8, n_layers=6,
                               ffn_dim=512)
x_small = torch.randn(8, 3, 32, 32)
print(vit_small(x_small).shape)   # (8, 10)
```

*Code 5 – Complete ViT implementation. The key architectural decisions: (1) `Conv2d` with `kernel_size=stride=patch_size` extracts patches and projects them in a single step; (2) CLS token is prepended before positional embeddings; (3) only CLS token output goes to the classifier. The ViT-B/16 configuration has ~86M parameters — similar to a large ResNet.*

---

## 6  ViT vs CNN: When Does Each Win?

The original ViT paper ran a definitive experiment comparing ViT and ResNet (BiT) across dataset sizes. The results show a clean crossover:

- **Small data (ImageNet, 1.2M images)**: ResNets outperform ViTs. The CNN's inductive biases (locality, translation equivariance) act as free regularisation — the model does not need to discover these from data. ViT is under-regularised and overfits.
- **Medium data (ImageNet-21k, 14M images)**: Large ViTs match large ResNets. The extra data compensates for the lack of inductive bias.
- **Large data (JFT-300M, 300M images)**: Large ViTs outperform large ResNets, while also training faster (ViT-L/16 takes 2,500 TPU-v3 core days vs ResNet's 9,900). Matrix multiplication is more hardware-friendly than convolution, giving ViTs better hardware utilisation.

The conclusion is sometimes framed as 'ViTs have less inductive bias than CNNs, so they need more data.' A more nuanced reading: ViTs learn better representations when given enough data, because they can discover the optimal biases for each task rather than being constrained to locality and translation equivariance. Locality is a very good prior for natural images, but it is not the optimal prior — with enough data, learned representations can do better.

> **ImageGPT: the naive baseline that showed pixel-level Transformers are impractical.** Before ViT, OpenAI tried treating each pixel as a token (imageGPT). A $256 \times 256$ image has 65,536 tokens; the attention matrix has $65{,}536^2 \approx 4$ billion entries. Even using $64 \times 64$ images, training a 48-layer model required 768 GB of memory for attention matrices for a single example. imageGPT produced interesting results but demonstrated conclusively that pixel-level attention is infeasible at scale — motivating the patch-based approach of ViT.

---

## 7  The Swin Transformer: Efficient Hierarchical ViT

### 7.1  Two Problems with Standard ViT

Standard ViT has two structural limitations for dense prediction tasks (detection, segmentation):

- **Single-scale features**: ViT produces a sequence of token representations at a single resolution throughout. CNNs naturally produce hierarchical feature maps at multiple scales (stride 4, 8, 16, 32), which are essential for detecting objects at different sizes in detection and segmentation frameworks.
- **Quadratic attention cost**: For a $224 \times 224$ image with $P=16$, ViT has 196 tokens and manages fine. But for dense prediction with $P=4$ (finer detail), the same image produces 3,136 tokens and the attention matrix has $\sim 10$M entries per head per layer — too expensive for practical training.

### 7.2  Window Attention

Swin Transformer (Liu et al., 2021) restricts self-attention to non-overlapping local windows of $M \times M$ patches each. With $M=7$ (Swin's default), each window has 49 tokens instead of the full $\sim 3{,}000$. The attention cost per window is $M^4$ instead of $(H/P)^2(W/P)^2$, and since there are $(H/P)/M \times (W/P)/M$ windows, the total attention cost is:

$$\text{Total} = \frac{H/P}{M} \times \frac{W/P}{M} \times M^4 = M^2 \times \frac{H}{P} \times \frac{W}{P}$$

This is linear in image size for fixed $M$, rather than quadratic. For a $224 \times 224$ image with $P=4$ and $M=7$, this is $7^2 \times 56 \times 56 = 137{,}200$ — manageable, and constant regardless of image resolution.

### 7.3  Shifted Windows

Window attention solves the complexity problem but creates a new one: patches in different windows never interact. The receptive field is limited to $M \times M$ patches per layer, and global context is lost. Swin's solution is shifted window attention: alternate between regular windows and windows shifted by $(M/2, M/2)$ pixels in successive blocks.

In odd-numbered blocks, standard windows are used. In even-numbered blocks, the grid is shifted so that new windows span different combinations of patches. Over multiple layers, every patch gradually accumulates information from all other patches — global receptive field is achieved, but at linear rather than quadratic cost.

### 7.4  Hierarchical Feature Maps

Swin also addresses the single-scale limitation by merging patches at each stage, similar to CNN downsampling:

- Stage 1: $4 \times 4$ patches $\to$ feature map of size $H/4 \times W/4 \times C$
- Stage 2: merge $2 \times 2$ groups of patches $\to$ $H/8 \times W/8 \times 2C$
- Stage 3: merge again $\to$ $H/16 \times W/16 \times 4C$
- Stage 4: merge again $\to$ $H/32 \times W/32 \times 8C$

This produces a pyramid of feature maps at 4 different scales, directly compatible with detection and segmentation frameworks like FPN and Mask R-CNN — the same frameworks used with CNN backbones. This is why Swin became the dominant vision backbone for dense prediction tasks.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ── Window attention: the core computational unit of Swin ────────────
def window_partition(x, window_size):
    """
    Partition a feature map into non-overlapping windows.
    Args:
        x: (B, H, W, C)  — spatial feature map
        window_size: M   — window size (M×M patches per window)
    Returns:
        windows: (num_windows*B, M, M, C)
    """
    B, H, W, C = x.shape
    # Reshape into (B, H/M, M, W/M, M, C)
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # Permute to (B, H/M, W/M, M, M, C) then merge batch and window dims
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows   # (num_windows*B, M, M, C)

def window_reverse(windows, window_size, H, W):
    """Reverse of window_partition."""
    B_ = windows.shape[0]
    B  = int(B_ / (H * W / window_size / window_size))
    x  = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x  = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """Self-attention within non-overlapping windows of M×M patches."""
    def __init__(self, d_model, window_size, n_heads):
        super().__init__()
        self.window_size = window_size
        self.n_heads     = n_heads
        self.d_k         = d_model // n_heads
        self.scale       = self.d_k ** -0.5

        self.qkv  = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)

        # Relative position bias (Swin uses relative rather than absolute positions)
        self.rel_bias_table = nn.Parameter(
            torch.zeros((2*window_size-1)**2, n_heads))
        nn.init.trunc_normal_(self.rel_bias_table, std=0.02)

    def forward(self, x):
        """
        x: (num_windows*B, M*M, d_model)
        Returns: same shape
        """
        BW, N, C = x.shape   # BW = num_windows × batch, N = M² tokens

        # Compute Q, K, V for all heads at once
        qkv = self.qkv(x).chunk(3, dim=-1)   # 3 × (BW, N, C)
        Q, K, V = [t.view(BW, N, self.n_heads, self.d_k).permute(0,2,1,3)
                   for t in qkv]              # each: (BW, h, N, d_k)

        # Scaled dot-product attention
        attn = (Q @ K.transpose(-2,-1)) * self.scale   # (BW, h, N, N)
        attn = attn.softmax(dim=-1)

        out = (attn @ V).transpose(1,2).reshape(BW, N, C)
        return self.proj(out)

# ── Demonstration: window partitioning reduces attention cost ─────────
H, W, C, M = 56, 56, 96, 7   # 56×56 feature map, 7×7 windows
B = 2
x = torch.randn(B, H, W, C)

# Partition into windows
windows = window_partition(x, M)            # (B×(H/M)×(W/M), M, M, C)
print(f'Feature map: {x.shape}')            # (2, 56, 56, 96)
print(f'After partition: {windows.shape}')  # (2*8*8=128, 7, 7, 96)

# Attention cost comparison:
n_full    = H * W              # 3136 tokens for global attention
n_window  = M * M              # 49   tokens per window
n_windows = (H//M) * (W//M)    # 64   windows

global_cost = n_full ** 2
window_cost = n_windows * (n_window ** 2)
print(f'Global attention cost: {global_cost:,}')    # 9,834,496
print(f'Window attention cost: {window_cost:,}')    # 200,704  (~50x less!)
print(f'Speedup: {global_cost / window_cost:.1f}×')
```

*Code 6 – Window attention for Swin Transformer. The `window_partition` function reshapes the feature map into non-overlapping $M \times M$ windows. The cost comparison at the bottom shows the key benefit: window attention (200,704) is $\sim 50\times$ cheaper than global attention (9,834,496) for a $56 \times 56$ feature map.*

---

## 8  Self-Attention in CNNs

Self-attention is not exclusive to Transformers — it can be added to CNNs as an additional module. The idea is to apply attention over the spatial positions of a CNN feature map, allowing any spatial position to attend to any other position, regardless of distance.

### 8.1  How It Works

Given a CNN feature map of shape $(C, H, W)$, we can apply self-attention over the $H \times W$ spatial positions (treating each position as a token with a $C$-dimensional embedding):

- Compute $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ by applying $1 \times 1$ convolutions to the feature map (equivalent to per-position linear projections).
- Reshape from $(C, H, W)$ to $(H \times W, C)$ — the $H \times W$ positions become the sequence.
- Apply scaled dot-product attention over the $H \times W$ tokens.
- Reshape back to $(C, H, W)$ and add a residual connection to the original feature map.

The key practical trick in SAGAN and related models is to initialise the residual weight $\gamma = 0$. This means the attention module initially does nothing (passes input unchanged) and is only gradually incorporated as training progresses — preventing it from destabilising an already partially-trained CNN.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialSelfAttention(nn.Module):
    """
    Self-attention over spatial positions of a CNN feature map.
    Used in SAGAN and many other architectures.
    Input/output: (B, C, H, W)
    """
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        d = max(in_channels // reduction, 1)

        # Q, K, V projections via 1×1 convolutions
        self.W_q = nn.Conv2d(in_channels, d, 1, bias=False)
        self.W_k = nn.Conv2d(in_channels, d, 1, bias=False)
        self.W_v = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        # Output projection
        self.W_o = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        # Learnable residual weight, initialised to 0
        # This lets the module start as an identity (no attention)
        # and gradually incorporate attention during training.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W              # number of spatial positions (= tokens)

        # 1. Compute Q, K, V
        Q = self.W_q(x).view(B, -1, N).permute(0, 2, 1)  # (B, N, d)
        K = self.W_k(x).view(B, -1, N).permute(0, 2, 1)  # (B, N, d)
        V = self.W_v(x).view(B, -1, N).permute(0, 2, 1)  # (B, N, C)

        # 2. Scaled dot-product attention over spatial positions
        d_k   = Q.size(-1)
        attn  = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5)  # (B, N, N)
        attn  = F.softmax(attn, dim=-1)

        # 3. Weighted sum of values
        out   = torch.bmm(attn, V)                   # (B, N, C)
        out   = out.permute(0, 2, 1).view(B, C, H, W)
        out   = self.W_o(out)

        # 4. Residual: γ=0 at init means this starts as identity
        return self.gamma * out + x

# ── Add spatial self-attention to a CNN feature map ───────────────────
conv = nn.Conv2d(64, 128, 3, padding=1)
attn = SpatialSelfAttention(128, reduction=8)

x   = torch.randn(4, 64, 32, 32)
x   = F.relu(conv(x))             # regular CNN feature map
out = attn(x)                     # spatial self-attention
print(f'Input:  {x.shape}')       # (4, 128, 32, 32)
print(f'Output: {out.shape}')     # (4, 128, 32, 32) — same shape
print(f'gamma at init: {attn.gamma.item():.1f}')   # 0.0 — starts as identity
```

*Code 7 – Spatial self-attention for CNNs. The `gamma=0` initialisation is critical: it means the module starts as an identity transform (no attention effect) and the attention contribution is gradually increased by gradient descent. This prevents the fresh attention module from disrupting a partially-trained CNN's features.*

---

## 9  Object Detection with Transformers: DETR

DETR (Detection Transformer, Carion et al. 2020) was the first end-to-end object detector to eliminate hand-designed components like anchor boxes and non-maximum suppression. Its architecture combines a CNN backbone (for feature extraction) with a Transformer encoder-decoder:

- **CNN backbone**: A ResNet extracts a feature map from the input image, which is flattened into a sequence of spatial tokens and passed to the Transformer encoder.
- **Transformer encoder**: Standard self-attention over the CNN feature tokens, allowing global context to be incorporated.
- **Transformer decoder**: A fixed set of $N$ learnable 'object queries' (typically $N=100$) attend to the encoder output via cross-attention. Each query learns to detect one object. The decoder outputs $N$ embeddings, one per query.
- **Prediction heads**: Each of the $N$ decoder embeddings is independently passed through small FFNs that predict a bounding box and a class label (or 'no object').

DETR matches Faster R-CNN on COCO with no custom post-processing — a significant simplification. The main downside is slow convergence (300+ epochs vs ~36 for Faster R-CNN) and poor small-object performance, which later works (Deformable DETR, DAB-DETR) addressed.

---

## 10  Summary

This lecture introduced Vision Transformers — a fundamentally different approach to visual recognition that replaces convolutional feature extraction with a global attention mechanism applied to sequences of image patches.

| Architecture | Key idea | Receptive field | Best when |
|---|---|---|---|
| CNN (ResNet etc) | Local filters + pooling | Grows with depth | Small–medium data; dense tasks |
| ViT | Patches → tokens → global attention | Global from layer 1 | Large data; classification |
| Swin Transformer | Window attention + shifted windows | Grows via shifted windows; hierarchical | Large data; detection & segmentation |
| DETR | CNN + Transformer encoder-decoder | Global via attention | End-to-end detection, no anchors |
| CNN + self-attention (SAGAN etc) | Add attention to CNN feature maps | CNN local + attention global | When global context needed in CNN |

The key insight to carry forward: Vision Transformers and CNNs are not competing paradigms — they are complementary tools with different inductive biases. CNNs excel with limited data because locality and translation equivariance are good priors for natural images. ViTs excel with large data because they can learn task-specific representations unconstrained by those priors. Modern practice increasingly uses hybrid approaches: CNN features as input tokens to Transformers, attention gates in CNNs, and hierarchical Transformer backbones (Swin) that inherit the multi-scale feature pyramid of CNNs. Understanding both architectures deeply — and when to use each — is the hallmark of a practitioner who can solve new problems rather than just apply existing recipes.

---

## References

- Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
- Dosovitskiy, A. et al. (2020). An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale. ICLR 2021.
- Liu, Z. et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV.
- Carion, N. et al. (2020). End-to-End Object Detection with Transformers (DETR). ECCV.
- Zhang, H. et al. (2019). Self-Attention Generative Adversarial Networks (SAGAN). ICML.
- Chen, M. et al. (2020). Generative Pretraining from Pixels (imageGPT). ICML.
- Illustrated Transformer blog post: jalammar.github.io/illustrated-transformer/
- Lilian Weng's attention overview: lilianweng.github.io/posts/2018-06-24-attention/