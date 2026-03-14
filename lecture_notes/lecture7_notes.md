# Lecture 7
# CNN Architectures

*Deep Learning for Visual Recognition · Aarhus University*

These notes trace the evolutionary arc of CNN architectures from AlexNet (2012) to MobileNet — showing how each design decision was motivated by the failures of the previous generation. Transfer learning is covered first because it is the practical context in which most of these architectures are actually used.

---

## 1  Transfer Learning

### 1.1  The Core Idea

Training a deep CNN from scratch requires large datasets (millions of images) and significant compute (days of GPU time). Transfer learning circumvents both requirements by re-using a network that was already trained on a large dataset — most commonly ImageNet — and adapting it to a new task. The key insight from Donahue et al. (DeCAF, 2014) is that a CNN trained on ImageNet develops representations that are generally useful far beyond the original task: early layers learn low-level detectors (edges, colours, textures) that appear in all natural images, while later layers learn progressively more task-specific combinations of those features.

The practical consequence is that even a simple linear classifier trained on top of a frozen ImageNet encoder can achieve competitive accuracy on new visual tasks — without any fine-tuning of the convolutional layers. For small datasets, this is often the correct approach.

### 1.2  The Generic-to-Specific Gradient

Features become progressively more task-specific with depth. This creates a principled strategy for deciding which layers to fine-tune:

- **Conv1–Conv2 (early layers)**: Learn edge detectors and colour blobs. These are universal and should almost never be modified — they would just learn the same things again if fine-tuned.
- **Conv3–Conv4 (middle layers)**: Learn textures, simple patterns, object parts. Moderately transferable; worth fine-tuning if the new task differs significantly from ImageNet.
- **Conv5 / FC layers (late layers)**: Learn class-specific combinations. These are the most task-specific and are the first candidates for replacement or fine-tuning.

### 1.3  The Four Scenarios

The decision of what to fine-tune depends on two factors: how much labelled data you have, and how similar your dataset is to the pre-training dataset (ImageNet).

| Scenario | Data situation | Strategy |
|---|---|---|
| 1 — Freeze encoder | Small dataset, similar domain | Replace + train only last FC layer. Freeze everything else. |
| 2 — Train more FC | Larger dataset, similar domain | Replace + train multiple FC layers (smaller to avoid overfit). |
| 3 — Fixed extractor | Any size, any domain | Remove FC layers entirely. Use encoder outputs as fixed features for another classifier (e.g. k-NN). |
| 4 — Full fine-tune | Large dataset, different domain | Replace FC layers, train them first, then unfreeze conv layers and fine-tune all at 1/10 original LR. |

### 1.4  Fine-Tuning Protocol

A reliable two-step fine-tuning protocol (following Jeremy Howard's fast.ai approach):

- **Step 1 — Transfer learning**: Freeze all pretrained layers. Replace and randomly initialise the classification head. Train only the head for a few epochs with a normal learning rate. This gets the head weights into a reasonable range before unleashing gradients on the pretrained layers.
- **Step 2 — Fine-tuning**: Unfreeze all layers (or the deeper half). Continue training with a much lower learning rate — typically 1/10 of the learning rate used in step 1. The low learning rate prevents the pretrained representations from being destroyed by large gradient updates.

> **Why reduce the learning rate for fine-tuning?** The pretrained weights encode millions of images' worth of useful structure. Large gradient updates would overwrite this structure with noise from your small dataset. A low learning rate makes small corrections to the representations rather than replacing them entirely — think of it as polishing rather than repainting.

```python
import torch
import torch.nn as nn
import torchvision.models as models

# ════════════════════════════════════════════════════════════════════
# THE COMPLETE TRANSFER LEARNING + FINE-TUNING WORKFLOW
# ════════════════════════════════════════════════════════════════════

# ── Step 0: Load pretrained model ────────────────────────────────────
# weights='IMAGENET1K_V2' loads the best available ImageNet weights
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

N_CLASSES = 10   # your dataset's number of classes

# ── SCENARIO 1: Small dataset — replace + freeze everything ──────────
# 1. Freeze ALL pretrained layers
for param in model.parameters():
    param.requires_grad = False

# 2. Replace the classification head (always required)
# ResNet50's final FC has in_features=2048
model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
# model.fc is newly created, so requires_grad=True by default

# Only FC parameters are updated — tiny number of parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Trainable: {trainable:,} / {total:,} ({trainable/total:.1%})')

# ── SCENARIO 4: Full fine-tuning — two-step protocol ─────────────────
model2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model2.fc = nn.Linear(model2.fc.in_features, N_CLASSES)

# --- Step 1: Transfer learning (head only) ---------------------------
for param in model2.parameters():
    param.requires_grad = False
for param in model2.fc.parameters():
    param.requires_grad = True

opt_step1 = torch.optim.Adam(model2.fc.parameters(), lr=1e-3)
# ... train for ~5 epochs ...

# --- Step 2: Fine-tuning (all layers, low LR) ------------------------
for param in model2.parameters():
    param.requires_grad = True

# Use discriminative learning rates: lower LR for earlier layers
opt_step2 = torch.optim.Adam([
    {'params': model2.layer1.parameters(), 'lr': 1e-5},  # early: very low
    {'params': model2.layer2.parameters(), 'lr': 1e-5},
    {'params': model2.layer3.parameters(), 'lr': 1e-4},  # middle: low
    {'params': model2.layer4.parameters(), 'lr': 1e-4},
    {'params': model2.fc.parameters(),     'lr': 1e-3},  # head: normal
])
# ... fine-tune for ~10 epochs ...

# ── SCENARIO 3: Fixed feature extractor ──────────────────────────────
# Remove the FC layer to get 2048-d feature vectors
feature_extractor = nn.Sequential(*list(model.children())[:-1],
                                   nn.Flatten())
feature_extractor.eval()

# Extract features for all images (do this once, offline)
with torch.no_grad():
    x = torch.randn(8, 3, 224, 224)   # simulated batch
    features = feature_extractor(x)
    print(f'Feature shape: {features.shape}')  # (8, 2048)
# Then train any classifier (sklearn, k-NN, SVM) on these features
```

*Code 1 – Complete transfer learning workflow. Scenario 1 (freeze + replace head) is the starting point for any new project. Scenario 4's discriminative learning rates (lower LR for earlier layers) is the advanced technique used in practice — earlier layers get smaller updates because their features are already more optimal.*

---

## 2  AlexNet (2012): The CNN Revolution

AlexNet (Krizhevsky, Sutskever & Hinton, 2012) won the 2012 ILSVRC competition with a top-5 error of 15.3%, compared to 26.2% for the second-place entry — a gap so large it convinced the community that deep learning was the future of computer vision. It was the first CNN to win ILSVRC and the paper that triggered the modern deep learning era.

### 2.1  Architecture

The full architecture processes $227 \times 227 \times 3$ input images through five convolutional layers followed by three fully connected layers:

| Layer | Output shape | Parameters | Notes |
|---|---|---|---|
| Input | $227 \times 227 \times 3$ | 0 | Raw RGB image |
| CONV1 | $55 \times 55 \times 96$ | 35K | $11 \times 11$ filters, stride 4 |
| POOL1 | $27 \times 27 \times 96$ | 0 | $3 \times 3$ max pool, stride 2 |
| CONV2 | $27 \times 27 \times 256$ | 615K | $5 \times 5$ filters, pad 2 |
| POOL2 | $13 \times 13 \times 256$ | 0 | $3 \times 3$ max pool, stride 2 |
| CONV3 | $13 \times 13 \times 384$ | 885K | $3 \times 3$ filters, pad 1 |
| CONV4 | $13 \times 13 \times 384$ | 1,327K | $3 \times 3$ filters, pad 1 |
| CONV5 | $13 \times 13 \times 256$ | 885K | $3 \times 3$ filters, pad 1 |
| POOL3 | $6 \times 6 \times 256$ | 0 | $3 \times 3$ max pool, stride 2 |
| FC6 | 4096 | 37,753K | Flatten $6 \times 6 \times 256 = 9216 \to 4096$ |
| FC7 | 4096 | 16,781K | $4096 \to 4096$ |
| FC8 | 1000 | 4,097K | $4096 \to 1000$ class scores |
| **Total** | — | **62.4M** | ~249 MB at 4 bytes/param |

### 2.2  Key Innovations

- **ReLU activations**: First major network to use ReLU instead of tanh/sigmoid. As noted in Lecture 5, this produced $\sim 6\times$ faster convergence. AlexNet trained in a week with ReLU; it would have taken months with sigmoid.
- **Dropout ($p=0.5$) on FC6 and FC7**: With 62M parameters and only 1.2M training images, regularisation was essential. Dropout was applied to both large FC layers.
- **Data augmentation**: Random $227 \times 227$ crops from $256 \times 256$ images, plus horizontal flips. Effectively multiplied the training set size by $2{,}048\times$.
- **GPU training**: Training was split across two GTX 580 GPUs (3GB each) — the first large-scale use of GPU training for deep networks.
- **The FC layers dominate parameter count**: FC6 alone has 37.8M parameters — 60% of the entire network. This insight motivated later architectures to eliminate FC layers entirely.

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """AlexNet from scratch, following the original paper architecture."""
    def __init__(self, num_classes=1000, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            # CONV1: 96 filters, 11×11, stride 4
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # CONV2: 256 filters, 5×5, pad 2
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # CONV3,4,5: 3×3, pad 1 — no pooling between them
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),  # FC6
            nn.Dropout(dropout),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),          # FC7
            nn.Linear(4096, num_classes),                           # FC8
        )

    def forward(self, x):
        x = self.features(x)         # conv layers
        x = x.view(x.size(0), -1)   # flatten 6×6×256 → 9216
        return self.classifier(x)    # FC layers

net   = AlexNet()
x     = torch.randn(1, 3, 227, 227)
print(net(x).shape)   # (1, 1000)

# Parameter count per layer
total = 0
for name, m in net.named_modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        n = sum(p.numel() for p in m.parameters())
        total += n
        print(f'{name:30s}: {n:>12,} params')
print(f'Total: {total:,}')   # ≈ 62.4M

# Note: torchvision provides a pretrained version:
# models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
```

*Code 2 – AlexNet from scratch. The parameter count loop makes the FC layer dominance concrete: the three conv layers after pool2 have ~3M parameters combined, while FC6 alone has 37.8M. This motivated VGGNet's question: can we get the same depth with far fewer parameters?*

---

## 3  VGGNet (2014): Depth with Small Filters

VGGNet (Simonyan & Zisserman, 2014) asked a simple but powerful question: if deeper networks are better, what is the most principled way to go deeper? The answer was to use exclusively $3 \times 3$ convolutions with stride 1, padding 1, and $2 \times 2$ max pooling with stride 2 — the simplest possible building block — and just stack more of them.

### 3.1  The Key Insight: Stacked 3×3 Convolutions

A single $7 \times 7$ convolution and three stacked $3 \times 3$ convolutions have the same effective receptive field on the input — both see a $7 \times 7$ region of the image. But the parameter counts differ significantly:

$$\text{One } 7{\times}7 \text{ conv } (C \text{ channels):} \quad 7 \times 7 \times C \times C = 49C^2 \text{ parameters}$$

$$\text{Three } 3{\times}3 \text{ convs } (C \text{ channels):} \quad 3 \times 3 \times 3 \times C \times C = 27C^2 \text{ parameters}$$

Three stacked $3 \times 3$ convolutions use 45% fewer parameters than a single $7 \times 7$ convolution with the same receptive field. They also apply three ReLU activations instead of one, introducing more non-linearity and therefore more representational power. This insight — that depth through small filters beats width through large filters — became the dominant design principle for all subsequent CNN architectures.

### 3.2  Architecture and Parameter Profile

VGG16 follows a clean pattern: groups of 2–3 conv layers with $3 \times 3$ filters, each group followed by a $2 \times 2$ max pool that halves the spatial dimensions and doubles the number of channels. The progression $64 \to 128 \to 256 \to 512 \to 512$ means that as the feature maps shrink, the number of channels grows to compensate, keeping the total information roughly constant per spatial location.

The 138M parameter count is more than double AlexNet's 62M, but the distribution is revealing: the first FC layer ($7 \times 7 \times 512 \to 4096$) alone accounts for 102.8M parameters — 74% of the total. The entire convolutional backbone has only ~15M parameters. This made the FC7 features highly popular for transfer learning: extract 4096-dimensional FC7 activations as generic image features and train any downstream classifier on top.

```python
import torch
import torch.nn as nn
from torchvision import models

# ── VGG16 from scratch ────────────────────────────────────────────────
def make_vgg_block(in_ch, out_ch, n_convs):
    # One VGG stage: n_convs x (Conv3x3 + BN + ReLU) + MaxPool.
    layers = []
    for _ in range(n_convs):
        layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1),
                   nn.BatchNorm2d(out_ch),
                   nn.ReLU(inplace=True)]
        in_ch = out_ch
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            make_vgg_block(  3,  64, 2),   # 224→112  64 channels
            make_vgg_block( 64, 128, 2),   # 112→56  128 channels
            make_vgg_block(128, 256, 3),   #  56→28  256 channels
            make_vgg_block(256, 512, 3),   #  28→14  512 channels
            make_vgg_block(512, 512, 3),   #  14→ 7  512 channels
        )
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 4096),        nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # flatten 7×7×512
        return self.classifier(x)

vgg = VGG16()
total = sum(p.numel() for p in vgg.parameters())
conv_params = sum(p.numel() for m in vgg.features.modules()
                  for p in m.parameters() if isinstance(m, nn.Conv2d))
fc_params   = sum(p.numel() for p in vgg.classifier.parameters())
print(f'Total parameters:  {total:,}')       # ≈ 138M
print(f'Conv parameters:   {conv_params:,}') # ≈  15M (11%)
print(f'FC parameters:     {fc_params:,}')   # ≈ 123M (89%!)

# ── Effective receptive field of stacked 3×3 convs ───────────────────
# One 3×3: sees 3×3 = 9 pixels of input
# Two 3×3: sees 5×5 = 25 pixels of input
# Three 3×3: sees 7×7 = 49 pixels of input  ← same as one 7×7
# But with fewer parameters and more non-linearities
print(f'Three 3×3 vs one 7×7 params (C=512): {3*3*3*512*512:,} vs {7*7*512*512:,}')
# 7,077,888 vs 12,845,056 — 45% fewer parameters
```

*Code 2 – VGG16 from scratch using the `make_vgg_block` helper. The parameter breakdown makes the FC layer bottleneck starkly visible: 89% of parameters are in 3 FC layers. The final print confirms the parameter efficiency argument: three stacked $3 \times 3$ convolutions use 45% fewer parameters than one $7 \times 7$ convolution with the same receptive field.*

---

## 4  GoogLeNet / Inception V1 (2014): Wider Rather Than Deeper

GoogLeNet (Szegedy et al., 2014) took a different approach to improving efficiency: rather than going deeper like VGGNet, it asked whether a single layer could capture features at multiple scales simultaneously. The answer was the Inception module — and the result was a network with only 5 million parameters ($12\times$ fewer than AlexNet) that outperformed VGGNet on classification.

### 4.1  The 1×1 Convolution as a Bottleneck

A $1 \times 1$ convolution applies a learned linear combination across the channel dimension at each spatial location — it changes the depth of a feature map without changing its spatial dimensions. This makes it a powerful dimension reduction tool:

$$\text{Standard } 5{\times}5 \text{ conv } (480 \to 48 \text{ ch}, 14{\times}14): \quad 14 \times 14 \times 48 \times 5 \times 5 \times 480 = 112.9\text{M ops}$$

$$\text{With } 1{\times}1 \text{ bottleneck } (480 \to 16 \to 48): \quad 14 \times 14 \times 16 \times 1 \times 1 \times 480 = 1.5\text{M ops}$$

$$\phantom{\text{With } 1{\times}1 \text{ bottleneck } (480 \to 16 \to 48):} \quad +\; 14 \times 14 \times 48 \times 5 \times 5 \times 16 = 3.8\text{M ops}$$

$$\text{Total: } 5.3\text{M ops} \quad (21\times \text{ fewer!})$$

The $1 \times 1$ convolution reduces the 480-channel input to 16 channels before the expensive $5 \times 5$ convolution. This bottleneck cuts compute by $21\times$ with minimal loss in representational power. The same principle appears in ResNet's bottleneck blocks and MobileNet's depthwise separable convolutions.

### 4.2  The Inception Module

The Inception module runs four parallel branches on the same input feature map and concatenates their outputs:

- **Branch 1**: $1 \times 1$ convolution — captures cross-channel patterns at each spatial position.
- **Branch 2**: $1 \times 1$ bottleneck $\to$ $3 \times 3$ convolution — captures local spatial features.
- **Branch 3**: $1 \times 1$ bottleneck $\to$ $5 \times 5$ convolution — captures larger spatial features.
- **Branch 4**: $3 \times 3$ max pooling $\to$ $1 \times 1$ convolution — captures pooled spatial structure.

All four branches produce feature maps of the same spatial size (padding ensures this) but different depths, which are then concatenated along the channel dimension. This lets each module learn to extract the most useful combination of feature scales automatically, rather than having the architect hardcode filter sizes.

### 4.3  Global Average Pooling Replaces FC Layers

AlexNet and VGGNet end with large FC layers. GoogLeNet instead applies global average pooling before the final classifier: the last $7 \times 7 \times 1024$ feature map is averaged spatially to produce a 1024-dimensional vector. This eliminates the need for any FC layer parameters except the final $1024 \to 1000$ classifier, reducing overfitting drastically (from 37.8M FC6 parameters in AlexNet to essentially zero).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Inception module with 1×1 bottlenecks ────────────────────────────
class InceptionModule(nn.Module):
    """
    The Inception module: 4 parallel branches concatenated at output.
    All branches preserve spatial dimensions (same padding).
    """
    def __init__(self, in_ch, ch_1x1, ch_3x3_reduce, ch_3x3,
                                     ch_5x5_reduce, ch_5x5, ch_pool):
        super().__init__()
        # Branch 1: just 1×1 conv
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, ch_1x1, 1), nn.BatchNorm2d(ch_1x1), nn.ReLU(inplace=True))

        # Branch 2: 1×1 bottleneck → 3×3
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, ch_3x3_reduce, 1), nn.BatchNorm2d(ch_3x3_reduce), nn.ReLU(inplace=True),
            nn.Conv2d(ch_3x3_reduce, ch_3x3, 3, padding=1), nn.BatchNorm2d(ch_3x3), nn.ReLU(inplace=True))

        # Branch 3: 1×1 bottleneck → 5×5
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, ch_5x5_reduce, 1), nn.BatchNorm2d(ch_5x5_reduce), nn.ReLU(inplace=True),
            nn.Conv2d(ch_5x5_reduce, ch_5x5, 5, padding=2), nn.BatchNorm2d(ch_5x5), nn.ReLU(inplace=True))

        # Branch 4: 3×3 max pool → 1×1 to reduce channels
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, ch_pool, 1), nn.BatchNorm2d(ch_pool), nn.ReLU(inplace=True))

    def forward(self, x):
        # All 4 branches run in parallel, then concatenate on channel dim
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

# ── Test: all branches must produce same spatial dimensions ───────────
# inception_3a in original GoogLeNet: in=192, outputs 256 channels
inc = InceptionModule(192, 64, 96, 128, 16, 32, 32)
x   = torch.randn(2, 192, 28, 28)
out = inc(x)
print(f'Input:  {x.shape}')    # (2, 192, 28, 28)
print(f'Output: {out.shape}')  # (2, 256, 28, 28) = 64+128+32+32

# ── Global Average Pooling: replace the 37M-param FC layer ───────────
# After the last inception module: feature map is (batch, 1024, 7, 7)
# Global average pooling → (batch, 1024)
# Final FC → (batch, 1000)  — only 1024×1000 = 1M parameters

gap = nn.AdaptiveAvgPool2d(1)   # (B, C, H, W) → (B, C, 1, 1)
features = torch.randn(2, 1024, 7, 7)
pooled   = gap(features).flatten(1)  # (2, 1024)
print(f'After GAP: {pooled.shape}')   # (2, 1024) — no spatial dims
print(f'FC params: {1024*1000:,}')    # 1,024,000 vs 37,748,736 in AlexNet
```

*Code 3 – The Inception module with all four parallel branches. The output channel count is the sum of all branch outputs: $64+128+32+32=256$. Global Average Pooling (`nn.AdaptiveAvgPool2d(1)`) replaces the 37M-parameter FC layer with a 1M-parameter one — a $37\times$ parameter reduction at this layer alone.*

---

## 5  ResNet (2015): The Skip Connection Revolution

ResNet (He et al., 2015) solved the problem that had been quietly stopping progress: deeper networks were consistently worse than shallower ones — not because of overfitting, but because of the vanishing gradient problem making them harder to optimise. The solution was elegantly simple: add a direct path for gradients to flow.

### 5.1  The Degradation Problem

The observation that triggered ResNet: a 56-layer plain CNN (no skip connections) has higher training error than an 18-layer plain CNN on CIFAR-10. This is not overfitting — the training error is higher, not just the validation error. Something about deep networks makes them genuinely harder to optimise.

The hypothesis: it should be possible to construct a 56-layer network that performs at least as well as an 18-layer network, by making the extra 38 layers identity mappings (they pass their input through unchanged). If a plain network cannot learn identity mappings, the optimisation landscape must be pathological. The solution: make identity the default, and let the network learn residuals.

### 5.2  The Residual Block

A residual block defines its output as:

$$h_\mathbf{w}(\mathbf{x}) = F(\mathbf{x}) + \mathbf{x}$$

where $F(\mathbf{x})$ is the output of two (or three) convolutional layers applied to $\mathbf{x}$, and $\mathbf{x}$ is the skip connection that bypasses those layers entirely. Instead of learning the full mapping $h_\mathbf{w}(\mathbf{x})$, the block only needs to learn the residual $F(\mathbf{x}) = h_\mathbf{w}(\mathbf{x}) - \mathbf{x}$. If the identity is optimal (the block should do nothing), the network only needs to drive $F(\mathbf{x}) \to 0$, which is much easier than learning an explicit identity transformation through a stack of non-linear layers.

During backpropagation, gradients flow through both the residual path and the skip connection. The skip connection provides a direct highway for gradients to reach early layers without being attenuated by the chain of weight multiplications — solving the vanishing gradient problem for very deep networks.

### 5.3  Projection Shortcuts for Dimension Changes

The addition $\mathbf{x} + F(\mathbf{x})$ requires that $\mathbf{x}$ and $F(\mathbf{x})$ have the same shape. When a block changes the spatial resolution (stride 2) or the number of channels, the skip connection uses a $1 \times 1$ convolution (called a projection shortcut) to match dimensions:

- **Type 1 (solid line in the paper)**: The block preserves shape — skip connection is just the identity, $\mathbf{x} + F(\mathbf{x})$ directly.
- **Type 2 (dashed line)**: The block downsamples or changes depth — skip connection applies a $1 \times 1$ conv with stride 2 to match the output dimensions.

### 5.4  The Bottleneck Block (ResNet-50+)

For deeper variants (ResNet-50, 101, 152), a bottleneck design reduces computation. Instead of two $3 \times 3$ convolutions, each block uses $1{\times}1 \to 3{\times}3 \to 1{\times}1$: the first $1 \times 1$ reduces channels by $4\times$, the $3 \times 3$ operates in this reduced space, and the final $1 \times 1$ restores the channel count. This keeps parameter counts manageable even at 152 layers.

```python
import torch
import torch.nn as nn

# ── Basic residual block (ResNet-18/34) ──────────────────────────────
class BasicBlock(nn.Module):
    expansion = 1   # output channels = planes * expansion

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        # Projection shortcut: needed when stride≠1 or channels change
        self.shortcut = nn.Identity()   # Type 1: identity
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(  # Type 2: 1×1 conv
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))         # F(x)
        out = out + self.shortcut(x)            # F(x) + x  ← skip connection
        return torch.relu(out)                  # ReLU after addition

# ── Bottleneck block (ResNet-50/101/152) ─────────────────────────────
class Bottleneck(nn.Module):
    expansion = 4   # output channels = planes * 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        # 1×1 reduce → 3×3 spatial → 1×1 expand
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return torch.relu(out + self.shortcut(x))

# ── Verify: skip connection allows gradient flow through identity ──────
torch.manual_seed(0)
block = BasicBlock(64, 64, stride=1)  # no shape change → pure identity skip
x = torch.randn(2, 64, 28, 28, requires_grad=True)
out = block(x)
out.sum().backward()

# Gradient flows directly through x.grad even before weight updates
print(f'Input grad norm: {x.grad.norm():.4f}')  # non-zero because of skip

# Bottleneck parameter efficiency
bb_basic = BasicBlock(256, 256)    # two 3×3 convs: 256→256
bb_neck  = Bottleneck(256, 64)     # 1×1→3×3→1×1 bottleneck
p_basic  = sum(p.numel() for p in bb_basic.parameters())
p_neck   = sum(p.numel() for p in bb_neck.parameters())
print(f'Basic block params:      {p_basic:,}')  # ≈ 1.2M
print(f'Bottleneck block params: {p_neck:,}')   # ≈  70K (17× fewer!)
```

*Code 4 – `BasicBlock` and `Bottleneck` residual blocks. The single most important line in both is `out + self.shortcut(x)` — this is the skip connection that solved the degradation problem. The gradient check confirms that gradients flow through $\mathbf{x}$ even without passing through the convolutional layers.*

---

## 6  MobileNet (2017): Efficient CNNs for Deployment

All architectures so far were designed for maximum accuracy on ImageNet, without concern for deployment efficiency. MobileNet (Howard et al., 2017) asked a different question: what is the most accurate network that can run in real time on a mobile phone or embedded device? The answer was depthwise separable convolution.

### 6.1  Depthwise Separable Convolution

A standard convolution mixes spatial and channel information simultaneously: each output feature map is a function of all input channels over a spatial neighbourhood. Depthwise separable convolution factorises this into two separate operations:

- **Depthwise convolution**: Apply one filter per input channel, independently. Each input channel gets its own spatial filter, producing one output channel. This captures spatial features within each channel but does not mix channels.
- **Pointwise convolution**: Apply a $1 \times 1$ convolution to combine information across channels. This mixes the depthwise outputs into new feature combinations.

The two-step process has the same receptive field as a standard convolution but far fewer operations:

$$\text{Standard } 3{\times}3 \text{ conv } (M \to N \text{ ch}, H{\times}W): \quad \text{Cost} = H \times W \times M \times N \times 3 \times 3$$

$$\text{Depthwise separable } (M \to N \text{ ch}): \quad \text{Depthwise cost} = H \times W \times M \times 3 \times 3$$

$$\phantom{\text{Depthwise separable } (M \to N \text{ ch}): \quad} \text{Pointwise cost} = H \times W \times M \times N \times 1 \times 1$$

$$\text{Ratio} = \frac{1}{N} + \frac{1}{9} \approx \frac{1}{9} \quad \text{for large } N$$

For a $3 \times 3$ kernel, depthwise separable convolution uses $8$–$9\times$ fewer operations than standard convolution, with only a small accuracy penalty (typically 1–2%). MobileNetV2 (2018) added inverted residual blocks — residual connections between thin bottleneck layers — combining MobileNet's efficiency with ResNet's depth-training stability.

```python
import torch
import torch.nn as nn

# ── Depthwise Separable Convolution ──────────────────────────────────
class DepthwiseSeparableConv(nn.Module):
    """
    Standard conv  → Depthwise + Pointwise (same output, much cheaper).
    Reduction: ~8-9× fewer multiplications for a 3×3 kernel.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Depthwise: one filter per input channel (groups=in_ch)
        self.depthwise  = nn.Conv2d(in_ch, in_ch, 3,
                                    stride=stride, padding=1,
                                    groups=in_ch, bias=False)  # groups=in_ch is key
        self.bn1        = nn.BatchNorm2d(in_ch)
        # Pointwise: mix channels with 1×1 conv
        self.pointwise  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2        = nn.BatchNorm2d(out_ch)
        self.relu       = nn.ReLU6(inplace=True)  # ReLU capped at 6 for quantisation

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x

# ── MobileNetV1 block ─────────────────────────────────────────────────
class MobileNetV1(nn.Module):
    """MobileNetV1 backbone — replace all conv layers with DS conv."""
    def __init__(self, num_classes=1000):
        super().__init__()
        def conv_bn(in_ch, out_ch, stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            DepthwiseSeparableConv(32,  64,  1),
            DepthwiseSeparableConv(64,  128, 2),
            DepthwiseSeparableConv(128, 128, 1),
            DepthwiseSeparableConv(128, 256, 2),
            DepthwiseSeparableConv(256, 256, 1),
            DepthwiseSeparableConv(256, 512, 2),
            *[DepthwiseSeparableConv(512, 512, 1) for _ in range(5)],
            DepthwiseSeparableConv(512, 1024, 2),
            DepthwiseSeparableConv(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x): return self.model(x)

mobile = MobileNetV1()
total  = sum(p.numel() for p in mobile.parameters())
print(f'MobileNetV1 parameters: {total/1e6:.1f}M')   # ≈ 4.2M

# ── Cost comparison: standard conv vs depthwise separable ─────────────
def conv_cost(H, W, M, N, K):
    return H * W * M * N * K * K

def ds_conv_cost(H, W, M, N, K):
    return H*W*M*K*K + H*W*M*N  # depthwise + pointwise

H, W, M, N, K = 14, 14, 512, 512, 3
std = conv_cost(H, W, M, N, K)
ds  = ds_conv_cost(H, W, M, N, K)
print(f'Standard conv:          {std:,} ops')
print(f'Depthwise separable:    {ds:,} ops')
print(f'Reduction factor:       {std/ds:.1f}×')   # ≈ 8.6×
```

*Code 5 – Depthwise separable convolution. The key line is `groups=in_ch` in the depthwise conv: setting `groups` equal to the number of input channels makes each filter responsible for exactly one channel. The cost comparison confirms the $\sim 8.6\times$ reduction for a $3 \times 3$ kernel with 512 channels.*

---

## 7  U-Net: Skip Connections for Dense Prediction

U-Net (Ronneberger et al., 2015) was designed for biomedical image segmentation, where you need to produce a dense per-pixel output (a segmentation mask) rather than a single class label. It addresses a fundamental limitation of standard encoder-decoder architectures: the bottleneck throws away spatial information.

### 7.1  The Problem with Plain Autoencoders

A plain convolutional autoencoder compresses the input to a small bottleneck and then expands back to the original resolution. The spatial information discarded during compression (exact pixel locations, fine boundaries, subtle textures) cannot be recovered during expansion — the decoder has no access to the high-resolution features computed in the encoder. For segmentation tasks that require pixel-precise boundaries, this is catastrophic.

### 7.2  Skip Connections Across the Bottleneck

U-Net's solution is skip connections from each encoder stage directly to the corresponding decoder stage at the same resolution. When the decoder upsamples from $14 \times 14$ to $28 \times 28$, it concatenates the upsampled features with the encoder's $28 \times 28$ feature map, giving it direct access to the high-resolution spatial details from early in the encoding process. This is different from ResNet's skip connections — U-Net concatenates rather than adds, preserving both the upsampled features and the encoder features separately.

The resulting architecture looks like a 'U' in its layer-size profile (narrow at the bottleneck, wide at the ends), which gave it its name. The skip connections allow the decoder to combine high-level semantic information from the bottleneck with low-level spatial information from the encoder, producing sharp, accurate segmentation boundaries.

```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    U-Net for semantic segmentation.
    Encoder downsamples with max pooling.
    Decoder upsamples with ConvTranspose2d + skip connections (concatenate).
    """
    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs  = nn.ModuleList()

        # ── Encoder: double conv + pool ───────────────────────────────
        in_ch = in_channels
        for f in features:
            self.encoders.append(self._double_conv(in_ch, f))
            self.pools.append(nn.MaxPool2d(2, 2))
            in_ch = f

        # ── Bottleneck ────────────────────────────────────────────────
        self.bottleneck = self._double_conv(features[-1], features[-1]*2)

        # ── Decoder: upsample + concatenate + double conv ─────────────
        for f in reversed(features):
            # Upsample from f*2 channels → f channels
            self.upconvs.append(
                nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            # After concat with encoder features: f + f = 2f channels
            self.decoders.append(self._double_conv(f*2, f))

        self.head = nn.Conv2d(features[0], out_channels, 1)  # final 1×1

    def _double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),)

    def forward(self, x):
        # ── Encoder: save feature maps for skip connections ───────────
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)     # save for skip connection
            x = pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]     # reverse for decoder

        # ── Decoder: upsample + concatenate + double conv ─────────────
        for upconv, dec, skip in zip(self.upconvs, self.decoders, skips):
            x    = upconv(x)              # upsample
            x    = torch.cat([skip, x], dim=1)  # concatenate skip connection
            x    = dec(x)                 # double conv

        return self.head(x)   # per-pixel output

# ── Test: output has same spatial dimensions as input ─────────────────
unet = UNet(in_channels=1, out_channels=1)   # grayscale → binary mask
x    = torch.randn(2, 1, 256, 256)
out  = unet(x)
print(f'Input:  {x.shape}')    # (2, 1, 256, 256)
print(f'Output: {out.shape}')  # (2, 1, 256, 256) — same spatial size!

total = sum(p.numel() for p in unet.parameters())
print(f'U-Net parameters: {total/1e6:.1f}M')   # ≈ 31M

# The skip connections are visible in the forward pass:
# Each decoder step concatenates the upsampled features with
# the saved encoder features at the same resolution.
# This gives the decoder access to precise spatial details
# that were discarded during downsampling.
```

*Code 6 – Complete U-Net. The critical lines are `skips.append(x)` in the encoder (saving high-resolution features) and `torch.cat([skip, x], dim=1)` in the decoder (concatenating them back in). This concatenation is what distinguishes U-Net from a plain autoencoder — the decoder never has to guess what was lost during downsampling.*

---

## 8  Architectural Evolution: A Retrospective

Each architecture in this lecture was motivated by a specific failure mode of its predecessor. Reading the progression as a narrative makes every design decision logical rather than arbitrary:

| Architecture | Year | Params | Key innovation | Motivated by |
|---|---|---|---|---|
| LeNet | 1998 | 60K | CNN + pooling | First CNN — proof of concept |
| AlexNet | 2012 | 62M | ReLU, dropout, GPU | LeNet too shallow for ImageNet scale |
| VGGNet | 2014 | 138M | $3\times3$ filters, depth 19 | AlexNet too wide; depth more efficient |
| GoogLeNet | 2014 | 5M | Inception, GAP, $1\times1$ conv | VGGNet too many parameters in FC layers |
| ResNet | 2015 | 25M | Skip connections | Plain deep nets degrade (vanishing gradient) |
| MobileNet | 2017 | 4.2M | Depthwise separable conv | ResNet too expensive for mobile devices |
| MobileNetV2 | 2018 | 3.4M | Inverted residuals | MobileNetV1 had no residual connections |
| U-Net | 2015 | 31M | Encoder-decoder + skips | Autoencoders lose spatial detail at bottleneck |

Two meta-lessons emerge from this history. First, parameter efficiency always wins: every successful architecture found ways to do more with fewer parameters — smaller filters (VGG), bottleneck layers (GoogLeNet, ResNet), separable convolutions (MobileNet). Second, the right inductive bias matters: skip connections (ResNet, U-Net) encode the belief that identity mappings should be easy to learn; depthwise separable convolutions encode the belief that spatial and channel mixing are independent operations. Each bias turned out to be correct for its use case. In practice today, ResNet or one of its variants is the default choice for most tasks — a reliable, well-understood baseline that has been deployed in countless production systems.

---

## References

- Krizhevsky, A. et al. (2012). ImageNet Classification with Deep Convolutional Neural Networks (AlexNet). NeurIPS.
- Simonyan, K. & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition (VGGNet). ICLR 2015.
- Szegedy, C. et al. (2014). Going Deeper with Convolutions (GoogLeNet). CVPR 2015.
- He, K. et al. (2015). Deep Residual Learning for Image Recognition (ResNet). CVPR 2016.
- Howard, A. et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.
- Ronneberger, O. et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
- Donahue, J. et al. (2014). DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition. ICML.
- PyTorch model zoo: docs.pytorch.org/vision/main/models.html