# Lecture 4 — Convolutional Neural Networks

*Deep Learning for Visual Recognition · Aarhus University*

---

These notes introduce convolutional neural networks (CNNs): why fully-connected networks are impractical for images, how convolution exploits spatial structure through locality and parameter sharing, and how stacking CONV→POOL→FC blocks builds up a full image classifier. The notes end with transfer learning as the practical default for most real-world tasks.

---

## 1  Why Not Fully-Connected Networks for Images?

A 224×224 RGB image has $224 \times 224 \times 3 = 150{,}528$ input values. A single fully-connected layer with 1,000 hidden units would need $150{,}528 \times 1{,}000 \approx 150M$ parameters — for just the first layer. This is:

- **Computationally expensive** to train and run
- **Sample inefficient**: too many parameters relative to the amount of labelled data
- **Ignores spatial structure**: a fully-connected layer treats pixel (0,0) and pixel (223,223) as equally related, discarding the fact that nearby pixels are more correlated than distant ones

CNNs exploit two structural properties of images to address all three problems.

---

## 2  The Two Key Properties

### 2.1  Locality (Sparse Interactions)

Natural images have strong **local** correlations: pixels close to each other are much more likely to be related (part of the same edge, texture, or object) than pixels far apart. A convolutional filter operates on a small local patch (e.g. 3×3 or 5×5), connecting each output neuron to only a small receptive field in the input — not the entire image.

### 2.2  Translation Equivariance (Parameter Sharing)

A feature detector (e.g. a horizontal edge detector) should work the same way wherever the edge appears in the image. CNNs implement this by using the **same filter weights** at every spatial position. This parameter sharing reduces the number of parameters from $H \times W \times C_\text{in} \times C_\text{out}$ (one set per position) to just $K \times K \times C_\text{in} \times C_\text{out}$ (one set shared everywhere).

---

## 3  Convolution

### 3.1  The Operation

A 2D convolution of input feature map $\mathbf{X}$ (shape $H \times W \times C_\text{in}$) with filter $\mathbf{F}$ (shape $K \times K \times C_\text{in}$) produces one output channel:

$$(\mathbf{X} * \mathbf{F})_{i,j} = \sum_{c=1}^{C_\text{in}} \sum_{p=0}^{K-1} \sum_{q=0}^{K-1} X_{i+p,\, j+q,\, c} \cdot F_{p,q,c}$$

Applying $C_\text{out}$ such filters produces an output of shape $H' \times W' \times C_\text{out}$.

### 3.2  Output Size

Given input size $H$, filter size $K$, padding $P$, and stride $S$:

$$H_\text{out} = \left\lfloor \frac{H + 2P - K}{S} \right\rfloor + 1$$

Common configurations:
- **Same padding**: $P = (K-1)/2$, $S=1$ → output same size as input
- **Valid (no padding)**: $P=0$ → output smaller than input
- **Strided convolution**: $S=2$ → halves spatial dimensions (replaces pooling)

### 3.3  Number of Parameters

A convolutional layer with $C_\text{in}$ input channels, $C_\text{out}$ output channels, and $K \times K$ filters has:

$$\text{Parameters} = K \times K \times C_\text{in} \times C_\text{out} + C_\text{out} \quad \text{(weights + biases)}$$

This is independent of the input image size — the same filter is applied everywhere.

---

## 4  Pooling

Pooling layers reduce spatial dimensions, providing spatial invariance and reducing the number of activations. **Max pooling** takes the maximum value in each $K \times K$ window:

$$\text{MaxPool}(i,j) = \max_{p,q \in [0,K)} X_{i \cdot S + p,\; j \cdot S + q}$$

Max pooling has **no learnable parameters**. A 2×2 max pool with stride 2 halves both height and width. Average pooling uses the mean instead of the maximum.

---

## 5  A Complete CNN Block

A typical CNN block is: **CONV → BatchNorm → ReLU → (optional) Pool**

Stacking multiple such blocks builds a feature hierarchy:
- Early layers: edges, corners, colour blobs
- Middle layers: textures, parts
- Late layers: object-level features

After the convolutional backbone, the spatial feature map is **flattened** and passed through one or more fully-connected layers that produce class scores.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── CNN from scratch: understanding each dimension ────────────────────
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1: 3 → 32 channels, 32×32 → 16×16
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # same padding
        self.pool1 = nn.MaxPool2d(2, 2)   # halve spatial dims

        # Block 2: 32 → 64 channels, 16×16 → 8×8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: 64 → 128 channels, 8×8 → 4×4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Classifier head
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        # x: (B, 3, 32, 32)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (B, 32, 16, 16)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (B, 64, 8, 8)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # (B, 128, 4, 4)

        x = x.view(x.size(0), -1)         # flatten: (B, 128*4*4=2048)
        x = F.relu(self.fc1(x))           # (B, 256)
        return self.fc2(x)                # (B, 10) — class logits

# ── Shape trace ───────────────────────────────────────────────────────
model = ConvNet()
x = torch.randn(4, 3, 32, 32)   # batch of 4 CIFAR-10 images
print(model(x).shape)            # (4, 10)

# ── Parameter count per layer ─────────────────────────────────────────
total = 0
for name, m in model.named_modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        n = sum(p.numel() for p in m.parameters())
        total += n
        print(f'{name:12s}: {n:>8,} params')
print(f'Total:        {total:>8,} params')
# conv1:      896   (3*3*3*32 + 32)
# conv2:   18,496   (3*3*32*64 + 64)
# conv3:   73,856   (3*3*64*128 + 128)
# fc1:    524,544   (2048*256 + 256)  ← FC layers dominate!
# fc2:      2,570   (256*10 + 10)
```

---

## 6  The 1×1 Convolution

A 1×1 convolution applies a learned linear combination **across the channel dimension** at each spatial location independently. It does not mix spatial information — only channel information:

$$\mathbf{y}_{i,j} = \mathbf{W} \cdot \mathbf{x}_{i,j} + \mathbf{b}$$

where $\mathbf{x}_{i,j} \in \mathbb{R}^{C_\text{in}}$ is the feature vector at position $(i,j)$.

Uses:
- **Dimension reduction**: reduce $C_\text{in} \to C_\text{out}$ cheaply (used as bottleneck in GoogLeNet/ResNet)
- **Replace FC layers**: a 1×1 conv with $C_\text{out} = K$ classes is equivalent to applying a $K$-way classifier at every spatial position

---

## 7  LeNet-5 and AlexNet

### LeNet-5 (1998)

The first successful CNN for digit recognition. Architecture: `CONV-POOL-CONV-POOL-FC-FC`. Used 5×5 filters and tanh activations. Demonstrated that learned convolutional features substantially outperformed hand-crafted features for MNIST.

### AlexNet (2012)

Won ILSVRC 2012 with top-5 error of 15.3% (vs 26.2% for the runner-up), triggering the deep learning revolution. Key innovations:
- First large-scale use of **ReLU** (6× faster convergence than tanh)
- **Dropout** (p=0.5) on FC layers for regularisation
- **GPU training** (split across two GTX 580s)
- Heavy **data augmentation** (random crops + horizontal flips)

AlexNet architecture for 227×227×3 input:

| Layer | Output size | Parameters |
|---|---|---|
| CONV1 (11×11, s=4, 96 filters) | 55×55×96 | 34,944 |
| POOL1 (3×3, s=2) | 27×27×96 | 0 |
| CONV2 (5×5, p=2, 256 filters) | 27×27×256 | 614,656 |
| POOL2 | 13×13×256 | 0 |
| CONV3 (3×3, p=1, 384 filters) | 13×13×384 | 885,120 |
| CONV4 (3×3, p=1, 384 filters) | 13×13×384 | 1,327,488 |
| CONV5 (3×3, p=1, 256 filters) | 13×13×256 | 884,992 |
| POOL3 | 6×6×256 | 0 |
| FC6 | 4096 | **37,752,832** |
| FC7 | 4096 | 16,781,312 |
| FC8 | 1000 | 4,097,000 |
| **Total** | | **~62.4M** |

Note that FC6 alone accounts for ~60% of all parameters — a strong motivation for removing FC layers in later architectures.

---

## 8  Transfer Learning

Training a CNN from scratch requires millions of labelled images and days of GPU time. **Transfer learning** reuses a network trained on a large dataset (typically ImageNet) as a feature extractor for a new task.

The key observation: early CNN layers learn universal low-level features (edges, colours, textures) while later layers learn task-specific high-level features. For a new task, the early layers can be reused as-is; only the later layers need adapting.

**Decision guide:**

| Data size | Data similarity to ImageNet | Strategy |
|---|---|---|
| Small | Similar | Replace + train last FC layer only |
| Small | Different | Replace + train more FC layers |
| Large | Similar | Fine-tune all layers (low LR) |
| Large | Different | Fine-tune all layers (normal LR) |

```python
import torch
import torch.nn as nn
import torchvision.models as models

# ── Transfer learning with ResNet-50 ─────────────────────────────────
N_CLASSES = 5   # your dataset

# Load pretrained weights
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# ── Scenario 1: small dataset — freeze backbone, train head only ──────
for param in model.parameters():
    param.requires_grad = False

# Replace final FC layer (in_features=2048 for ResNet50)
model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
# model.fc is newly created → requires_grad=True by default

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Trainable: {trainable:,} / {total:,} ({trainable/total:.2%})')

# ── Scenario 2: fine-tune all layers with discriminative learning rates
model2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model2.fc = nn.Linear(model2.fc.in_features, N_CLASSES)
for param in model2.parameters():
    param.requires_grad = True

opt = torch.optim.Adam([
    {'params': model2.layer1.parameters(), 'lr': 1e-5},  # early: very low
    {'params': model2.layer2.parameters(), 'lr': 1e-5},
    {'params': model2.layer3.parameters(), 'lr': 1e-4},  # middle: low
    {'params': model2.layer4.parameters(), 'lr': 1e-4},
    {'params': model2.fc.parameters(),     'lr': 1e-3},  # head: normal
])
print('Discriminative LR optimiser configured')
```

---

## 9  Summary

| Component | Role | Key parameter |
|---|---|---|
| Conv layer | Extract local features | Filter size $K$, number of filters $C_\text{out}$ |
| Padding | Control output size | $P = (K-1)/2$ for same padding |
| Stride | Downsample | $S=2$ halves spatial dims |
| Max pooling | Spatial invariance + downsampling | Pool size, stride |
| Batch normalisation | Stabilise training | Learnable $\gamma$, $\beta$ |
| FC layer | Map features to class scores | $C_\text{in} \times C_\text{out}$ |
| 1×1 conv | Channel mixing / dimension reduction | $C_\text{out}$ |

The convolution operation's output size formula $H_\text{out} = \lfloor(H + 2P - K)/S\rfloor + 1$ is the single most practically useful formula in this lecture — computing it correctly prevents shape mismatches that are the most common CNN implementation bug.

## References

- LeCun, Y. et al. (1998). Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE.
- Krizhevsky, A. et al. (2012). ImageNet Classification with Deep Convolutional Neural Networks (AlexNet). NeurIPS.
- CS231n Stanford: cs231n.github.io/convolutional-networks/
