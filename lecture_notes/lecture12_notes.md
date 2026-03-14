# Lecture 12 — Vision Transformers

*Deep Learning for Visual Recognition · Aarhus University*

---

These notes build the Vision Transformer from first principles: starting with why CNNs have inductive biases, introducing attention as a differentiable database lookup, assembling the Transformer encoder, and showing how images become token sequences for ViT. The Swin Transformer's solution to the quadratic attention bottleneck is derived carefully.

---

## 1  Inductive Bias: What CNNs Assume

Every ML model encodes inductive biases — assumptions baked into the architecture before training. CNNs have two strong ones:

- **Locality**: nearby pixels are more related than distant ones — filters operate on small local patches
- **Translation equivariance**: the same filter is applied everywhere — a feature detector works regardless of position

These biases are incredibly useful with limited data: the model doesn't need to learn that spatial proximity implies correlation. But they are also limitations: tasks where distant pixels are highly informative (a dog's ear reveals where its nose will be) benefit from global context.

**Transformers** have no built-in spatial assumptions — they let every token attend to every other token from the first layer. The trade-off: less bias means more data needed. A ViT trained from scratch on ImageNet's 1.2M images performs worse than a ResNet of comparable size. With 14M–300M images, large ViTs surpass large ResNets in both accuracy and training compute efficiency (matrix multiplication is more hardware-friendly than convolution).

---

## 2  Scaled Dot-Product Attention

Attention is best understood as a **differentiable database lookup**:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- **Keys** $K = XW_k$: what each token contains
- **Values** $V = XW_v$: what information each token carries
- **Queries** $Q = YW_q$: what each output token is looking for

The scaling by $1/\sqrt{d_k}$ prevents the dot products from growing large with embedding dimension (which would push softmax into its flat saturation region — the same issue that motivated replacing sigmoid with ReLU, in a different context).

The attention weight matrix has shape $(n, n)$ — **quadratic in sequence length**. For a 256×256 image treated as individual pixels, $n = 65{,}536$ and the matrix has over 4 billion entries. This is why pixel-level Transformers are impractical.

```python
import torch, torch.nn as nn, torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores  = torch.bmm(Q, K.transpose(-2,-1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    weights = F.softmax(scores, dim=-1)   # (B, n_q, n_k) rows sum to 1
    output  = torch.bmm(weights, V)       # (B, n_q, d_v)
    return output, weights

# Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x, B):
        return x.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, Q_in, K_in, V_in):
        B = Q_in.size(0)
        Q = self.split_heads(self.W_q(Q_in), B)
        K = self.split_heads(self.W_k(K_in), B)
        V = self.split_heads(self.W_v(V_in), B)
        scores  = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, V).transpose(1,2).contiguous()
        context = context.view(B, -1, self.n_heads * self.d_k)
        return self.W_o(context)
```

---

## 3  Transformer Encoder

Each encoder block:
1. **Multi-head self-attention** — token-to-token interaction
2. **Position-wise FFN** — $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$ — per-token non-linearity
3. **Residual connections + LayerNorm** around each sub-layer

Self-attention is linear — it only recombines information across tokens. The FFN adds non-linearity and allows each token to be transformed into a richer representation. Without the FFN, stacking attention layers barely helps.

**LayerNorm** (not BatchNorm): normalises across the feature dimension for each individual token, independent of batch size. Same behaviour at train and test time. Suitable for variable-length sequences.

```python
import torch, torch.nn as nn

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x_n = self.norm1(x)
        x = x + self.drop(self.attn(x_n, x_n, x_n)[0])
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x

encoder_block = TransformerEncoderBlock(256, 8, 1024)
x = torch.randn(4, 16, 256)
print(encoder_block(x).shape)  # (4, 16, 256) — shape preserved
```

---

## 4  Positional Embeddings

Self-attention is **permutation invariant**: shuffling the input tokens produces the same output embeddings in a different order. For images, position is everything, so we must inject positional information explicitly.

**Sinusoidal embeddings** (original Transformer paper): for token at position $k$ in a $d$-dimensional space:

$$\text{PE}(k, 2i)   = \sin\!\left(\frac{k}{n^{2i/d}}\right)$$

$$\text{PE}(k, 2i+1) = \cos\!\left(\frac{k}{n^{2i/d}}\right)$$

where $n = 10{,}000$. Each pair of dimensions $(2i, 2i+1)$ encodes position as a rotating 2D vector, with rotation frequency decreasing with $i$ — analogous to binary counting with continuous values.

**Learned embeddings** (ViT): learn a separate positional embedding vector per position. Slightly better on standard benchmarks; doesn't generalise to longer sequences without interpolation.

---

## 5  Vision Transformer (ViT)

**Core idea**: treat an image as a sequence of $N$ non-overlapping patches, each flattened and linearly projected to a token:

$$N = \frac{H \times W}{P^2}, \quad \text{each patch: } (P^2 \times C) \to d$$

For a 224×224 image with $P=16$: $N = 196$ patches — manageable. The $O(n^2)$ attention cost scales with $196^2$, not $224^4$.

A learnable **CLS token** is prepended to the sequence. After the Transformer encoder, the CLS token's output aggregates global image information and is passed to the classification head.

```python
import torch, torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, n_classes=1000,
                 d_model=768, n_heads=12, n_layers=12, ffn_dim=3072, dropout=0.1):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_ch, d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        self.pos_drop    = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)         # (B, N, d)
        x = torch.cat([self.cls_token.expand(B,-1,-1), x], dim=1) # prepend CLS
        x = self.pos_drop(x + self.pos_embed)
        x = self.norm(self.transformer(x))
        return self.head(x[:, 0])   # CLS token → class scores

vit = VisionTransformer()
x   = torch.randn(2, 3, 224, 224)
print(vit(x).shape)  # (2, 1000)
print(f'ViT-B/16: {sum(p.numel() for p in vit.parameters())/1e6:.0f}M params')  # ≈86M
```

---

## 6  ViT vs CNN: When Each Wins

| Pre-training data | Result |
|---|---|
| ImageNet (1.2M images) | ResNets outperform ViTs |
| ImageNet-21k (14M images) | Large ViTs match large ResNets |
| JFT-300M (300M images) | Large ViTs outperform large ResNets |

**Claim**: ViTs have less inductive bias than CNNs, so need more pre-training data to learn good features. With enough data, learned representations can surpass hand-engineered spatial biases.

**Efficiency**: ViT-L/16 takes 2,500 TPU-v3 core-days to train; equivalent ResNet takes 9,900. Matrix multiplication is more hardware-friendly than convolution.

---

## 7  Swin Transformer

### 7.1  Two Problems with Standard ViT

1. **Single-scale features**: ViT produces tokens at one resolution throughout. CNNs produce hierarchical feature pyramids (stride 4, 8, 16, 32) essential for detection/segmentation.
2. **Quadratic cost**: with $P=4$ (finer detail), a 224×224 image produces 3,136 tokens; the attention matrix has ~10M entries per head — too expensive.

### 7.2  Window Attention: $O(n)$ Instead of $O(n^2)$

Restrict self-attention to non-overlapping windows of $M \times M$ patches ($M=7$ in Swin). Total attention cost:

$$\underbrace{\frac{H}{MP} \cdot \frac{W}{MP}}_{\text{number of windows}} \times \underbrace{M^4}_{\text{cost per window}} = M^2 \cdot \frac{H}{P} \cdot \frac{W}{P}$$

This is **linear in image size** for fixed $M$. For a 224×224 image with $P=4$, $M=7$: window attention costs $49 \times 56 \times 56 \approx 137{,}000$ vs global attention's $3{,}136^2 \approx 9{,}800{,}000$ — a 71× reduction.

### 7.3  Shifted Windows

Window attention prevents communication across windows. Solution: alternate between regular and shifted windows in successive blocks. Over multiple layers, every patch accumulates information from all other patches — global context achieved at linear cost.

**Relative position bias** (instead of absolute positional embeddings):

$$\text{Attention} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + B\right)V$$

where $B \in \mathbb{R}^{M^2 \times M^2}$ is a learned bias encoding relative position between patches.

```python
import torch, torch.nn.functional as F

def window_partition(x, M):
    """(B, H, W, C) → (B*num_windows, M, M, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H//M, M, W//M, M, C)
    return x.permute(0,1,3,2,4,5).contiguous().view(-1, M, M, C)

H, W, C, M = 56, 56, 96, 7
x = torch.randn(2, H, W, C)
wins = window_partition(x, M)
print(f'Feature map: {x.shape} → Windows: {wins.shape}')  # (2*64, 7, 7, 96)

# Cost comparison
n_full   = H * W
n_win    = M * M
n_wins   = (H//M) * (W//M)
print(f'Global: {n_full**2:,}  Window: {n_wins * n_win**2:,}  ({n_full**2/(n_wins*n_win**2):.0f}× reduction)')
```

---

## 8  Object Detection with Transformers: DETR

DETR (Carion et al., 2020) was the first end-to-end object detector to eliminate anchor boxes and NMS:

1. **CNN backbone** extracts spatial features → flatten to a sequence
2. **Transformer encoder**: global self-attention over spatial tokens
3. **Transformer decoder**: N=100 learnable **object queries** attend to encoder output via cross-attention; each query learns to detect one object
4. **Prediction heads**: each decoder output → class + box via independent FFNs

No anchors, no NMS, no hand-engineered components. Achieves Faster R-CNN-level accuracy. Main weakness: slow convergence (300+ epochs vs ~36 for Faster R-CNN).

---

## 9  Summary

| Architecture | Key idea | Receptive field | Best when |
|---|---|---|---|
| CNN (ResNet) | Local filters + pooling | Grows with depth | Small–medium data; dense tasks |
| ViT | Patches → tokens → global attention | Global from layer 1 | Large data; classification |
| Swin Transformer | Window attention + shifted windows | Grows via shifts; hierarchical | Large data; detection & segmentation |
| DETR | CNN + Transformer encoder-decoder | Global via attention | End-to-end detection |

The main practical benefit of ViTs is probably **speed**: matrix multiply is more hardware-friendly than convolution, so ViTs with the same FLOPs as CNNs can train and run much faster on modern GPUs/TPUs. Vision Transformers are an evolution, not a revolution — they solve the same problems as CNNs, just with a different set of inductive biases.

## References

- Dosovitskiy, A. et al. (2020). An Image Is Worth 16×16 Words (ViT). ICLR 2021.
- Liu, Z. et al. (2021). Swin Transformer. ICCV.
- Carion, N. et al. (2020). DETR. ECCV.
- Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
- Illustrated Transformer: jalammar.github.io/illustrated-transformer/
