# Lecture 7 — CNN Architectures

*Deep Learning for Visual Recognition · Aarhus University*

---

These notes trace the evolutionary arc of CNN architectures from AlexNet (2012) to MobileNet, showing how each design decision was motivated by the failure mode of the previous generation. Transfer learning is covered first because it is the practical context in which most of these architectures are actually used.

---

## 1  Transfer Learning

### 1.1  The Core Idea

Training a deep CNN from scratch requires millions of labelled images and days of GPU time. Transfer learning reuses a network already trained on a large dataset — typically ImageNet — and adapts it to a new task. Early CNN layers learn universal low-level features (edges, colours, textures) that appear in all natural images; later layers learn task-specific combinations. This means early layers can be reused as-is, and only later layers need adapting.

### 1.2  The Four Scenarios

| Scenario | Data | Strategy |
|---|---|---|
| 1 — Freeze encoder | Small, similar to ImageNet | Replace + train last FC layer only |
| 2 — Train more FC | Larger, similar to ImageNet | Replace + train multiple FC layers |
| 3 — Fixed extractor | Any size, any domain | Remove FC layers, use encoder as fixed feature extractor |
| 4 — Full fine-tune | Large, different from ImageNet | Replace FC, train head, then unfreeze all at 1/10 LR |

### 1.3  Fine-Tuning Protocol

A reliable two-step approach:

1. **Transfer learning step**: freeze all pretrained layers; replace and train only the classification head for a few epochs with a normal learning rate. This brings the head weights into a reasonable range before touching the pretrained representations.
2. **Fine-tuning step**: unfreeze all layers and continue training with a much lower learning rate (typically 1/10 of the step-1 rate). The low LR prevents the pretrained representations from being destroyed by large gradient updates.

> **Why reduce the learning rate for fine-tuning?** The pretrained weights encode millions of images' worth of useful structure. Large gradient updates overwrite this structure with noise from your small dataset. A low LR makes small corrections rather than replacements.

```python
import torch
import torch.nn as nn
import torchvision.models as models

N_CLASSES = 10

# ── Step 1: Transfer learning (head only) ────────────────────────────
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, N_CLASSES)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Step 1 — trainable: {trainable:,} / {total:,} ({trainable/total:.1%})')

opt_step1 = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

# ── Step 2: Fine-tune all layers with discriminative LRs ─────────────
for param in model.parameters():
    param.requires_grad = True

opt_step2 = torch.optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3},
])
print('Step 2 — discriminative LR optimiser configured')
```

---

## 2  AlexNet (2012)

AlexNet won the 2012 ILSVRC competition with a top-5 error of 15.3% vs 26.2% for the runner-up, triggering the deep learning revolution. Key innovations: first large-scale use of **ReLU** (~6× faster convergence than tanh), **Dropout** on FC layers, GPU training, and heavy **data augmentation**.

Full architecture for 227×227×3 input (62.4M parameters total — 60% in FC6 alone):

| Layer | Output | Parameters |
|---|---|---|
| CONV1 (11×11, s=4, 96 filters) | 55×55×96 | 34,944 |
| POOL1 (3×3, s=2) | 27×27×96 | 0 |
| CONV2 (5×5, p=2, 256 filters) | 27×27×256 | 614,656 |
| POOL2 | 13×13×256 | 0 |
| CONV3–5 (3×3, p=1) | 13×13×256 | ~3.1M |
| POOL3 | 6×6×256 | 0 |
| FC6 | 4096 | **37,752,832** |
| FC7 | 4096 | 16,781,312 |
| FC8 | 1000 | 4,097,000 |

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256*6*6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

net = AlexNet()
print(f'Parameters: {sum(p.numel() for p in net.parameters()):,}')  # ≈ 62.4M
```

---

## 3  VGGNet (2014)

VGG asked: if deeper is better, what is the most principled way to go deeper? Answer: use exclusively **3×3 convolutions** with stride 1 and 2×2 max pooling.

**Key insight** — three stacked 3×3 convolutions have the same effective receptive field as one 7×7 convolution, but use 45% fewer parameters and apply three ReLU non-linearities instead of one:

$$3 \times (3 \times 3 \times C \times C) = 27C^2 \quad \text{vs} \quad 7 \times 7 \times C \times C = 49C^2$$

VGG16 follows the pattern: groups of 2–3 conv layers, each group followed by 2×2 max pool that halves spatial dimensions and doubles channels: 64→128→256→512→512.

The 138M parameter count is more than double AlexNet's, but the **distribution** reveals the FC bottleneck: the three FC layers account for 89% of parameters (7×7×512→4096 is 102.8M parameters alone).

```python
def make_vgg_block(in_ch, out_ch, n_convs):
    layers = []
    for _ in range(n_convs):
        layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1),
                   nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        in_ch = out_ch
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            make_vgg_block(3,   64,  2),   # 224→112
            make_vgg_block(64,  128, 2),   # 112→56
            make_vgg_block(128, 256, 3),   #  56→28
            make_vgg_block(256, 512, 3),   #  28→14
            make_vgg_block(512, 512, 3),   #  14→7
        )
        self.classifier = nn.Sequential(
            nn.Linear(7*7*512, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, 4096),    nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

vgg = VGG16()
conv_params = sum(p.numel() for m in vgg.features.modules()
                  for p in m.parameters() if isinstance(m, nn.Conv2d))
fc_params   = sum(p.numel() for p in vgg.classifier.parameters())
print(f'Conv params: {conv_params:,} ({conv_params/(conv_params+fc_params):.0%})')
print(f'FC params:   {fc_params:,}   ({fc_params/(conv_params+fc_params):.0%})')
```

---

## 4  GoogLeNet / Inception V1 (2014)

GoogLeNet asked: rather than going deeper, can a single layer capture features at multiple scales simultaneously? The answer was the **Inception module** — four parallel branches concatenated at output — combined with **1×1 bottleneck convolutions** to keep computation manageable.

**1×1 convolution as a bottleneck** — a 5×5 conv without bottleneck costs $14 \times 14 \times 48 \times 5 \times 5 \times 480 = 112.9\text{M}$ ops. With a 1×1 bottleneck reducing 480→16 channels first:

$$14 \times 14 \times 16 \times 1 \times 1 \times 480 = 1.5\text{M} \quad \text{(bottleneck)}$$
$$14 \times 14 \times 48 \times 5 \times 5 \times 16 = 3.8\text{M} \quad \text{(5×5)}$$

Total: 5.3M ops — a **21× reduction**.

**Global Average Pooling** replaces FC layers: instead of flattening the 7×7×1024 feature map into a 50,176-d vector and applying a 50M-parameter FC layer, average each channel spatially to get a 1024-d vector, then apply a 1024→1000 classifier (1M parameters). GoogLeNet achieves 5M total parameters — **12× fewer than AlexNet**.

```python
class InceptionModule(nn.Module):
    def __init__(self, in_ch, ch_1x1, ch_3x3_r, ch_3x3, ch_5x5_r, ch_5x5, ch_pool):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, ch_1x1, 1), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(in_ch, ch_3x3_r, 1), nn.ReLU(True),
                                 nn.Conv2d(ch_3x3_r, ch_3x3, 3, padding=1), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(in_ch, ch_5x5_r, 1), nn.ReLU(True),
                                 nn.Conv2d(ch_5x5_r, ch_5x5, 5, padding=2), nn.ReLU(True))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                 nn.Conv2d(in_ch, ch_pool, 1), nn.ReLU(True))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

inc = InceptionModule(192, 64, 96, 128, 16, 32, 32)
x   = torch.randn(2, 192, 28, 28)
print(inc(x).shape)  # (2, 256, 28, 28) = 64+128+32+32
```

---

## 5  ResNet (2015)

ResNet solved the **degradation problem**: adding more layers to a plain CNN makes training *worse*, not just slower. A 56-layer plain network has higher training error than an 18-layer one — not overfitting, but an optimisation failure.

**The observation**: a deeper network should be able to replicate the shallower one by making the extra layers identity mappings. If plain networks cannot learn identities, the loss landscape must be pathological.

**The solution** — residual blocks:

$$h(\mathbf{x}) = F(\mathbf{x}) + \mathbf{x}$$

Instead of learning the full mapping $h(\mathbf{x})$, the block only needs to learn the **residual** $F(\mathbf{x}) = h(\mathbf{x}) - \mathbf{x}$. If the identity is optimal, the network only needs to drive $F(\mathbf{x}) \to 0$, which is much easier than learning an explicit identity through non-linear layers. The skip connection also provides a direct gradient highway through time — the same insight as LSTMs (just applied spatially rather than temporally).

**Bottleneck block** (ResNet-50+): 1×1 → 3×3 → 1×1 convolutions. The first 1×1 reduces channels by 4×, the 3×3 operates in this reduced space, and the final 1×1 restores channels. This is the same bottleneck idea as GoogLeNet, giving 17× fewer parameters than a basic block at the same depth.

```python
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))   # skip connection

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes*4)
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes*4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes*4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*4))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return torch.relu(out + self.shortcut(x))
```

---

## 6  MobileNet (2017)

All previous architectures targeted maximum accuracy without regard for deployment efficiency. MobileNet asked: what is the most accurate network that can run in real time on a mobile device?

**Depthwise separable convolution** factorises a standard convolution into:
1. **Depthwise convolution**: one filter per input channel (groups = in_channels) — captures spatial features within each channel independently
2. **Pointwise convolution**: 1×1 conv that mixes channels

The cost reduction for a 3×3 kernel:

$$\frac{\text{Depthwise separable cost}}{\text{Standard conv cost}} = \frac{H \cdot W \cdot M \cdot 9 + H \cdot W \cdot M \cdot N}{H \cdot W \cdot M \cdot N \cdot 9} = \frac{1}{N} + \frac{1}{9} \approx \frac{1}{9}$$

For large $N$ (e.g. 512), this gives roughly **8–9× fewer operations** with only a small accuracy penalty.

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                                   groups=in_ch, bias=False)  # groups=in_ch is key
        self.bn1       = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2       = nn.BatchNorm2d(out_ch)
        self.relu      = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x

# Cost comparison
H, W, M, N, K = 14, 14, 512, 512, 3
std_cost = H * W * M * N * K * K
ds_cost  = H*W*M*K*K + H*W*M*N
print(f'Standard: {std_cost:,}  DS: {ds_cost:,}  Reduction: {std_cost/ds_cost:.1f}×')
```

---

## 7  U-Net (2015)

U-Net was designed for biomedical image segmentation: produce a per-pixel output mask from an input image. The key challenge is combining **semantic understanding** (what is this object?) from deep features with **precise localisation** (exactly which pixels?) from shallow features.

A plain encoder-decoder loses spatial detail at the bottleneck — the decoder must reconstruct boundaries from a 32× coarser feature map, producing blurry edges. U-Net adds **concatenation skip connections** from each encoder stage directly to the corresponding decoder stage:

```python
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64,128,256,512]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs  = nn.ModuleList()
        in_c = in_ch
        for f in features:
            self.encoders.append(self._double_conv(in_c, f))
            self.pools.append(nn.MaxPool2d(2, 2))
            in_c = f
        self.bottleneck = self._double_conv(features[-1], features[-1]*2)
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f*2, f, 2, stride=2))
            self.decoders.append(self._double_conv(f*2, f))
        self.head = nn.Conv2d(features[0], out_ch, 1)

    def _double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x);  skips.append(x);  x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = torch.cat([skip, up(x)], dim=1)
            x = dec(x)
        return self.head(x)

unet = UNet()
x = torch.randn(2, 1, 256, 256)
print(unet(x).shape)  # (2, 1, 256, 256) — same H×W as input
```

---

## 8  Architectural Evolution Summary

| Architecture | Year | Params | Key innovation | Motivated by |
|---|---|---|---|---|
| LeNet | 1998 | 60K | CNN + pooling | First CNN — proof of concept |
| AlexNet | 2012 | 62M | ReLU, dropout, GPU | LeNet too shallow for ImageNet |
| VGGNet | 2014 | 138M | 3×3 filters, depth 19 | AlexNet large filters inefficient |
| GoogLeNet | 2014 | 5M | Inception, GAP, 1×1 bottleneck | VGGNet too many FC parameters |
| ResNet | 2015 | 25M | Skip connections | Plain deep nets degrade |
| MobileNet | 2017 | 4.2M | Depthwise separable conv | ResNet too slow for mobile |
| MobileNetV2 | 2018 | 3.4M | Inverted residuals | MobileNetV1 no skip connections |
| U-Net | 2015 | 31M | Concatenation skip connections | Autoencoders lose spatial detail |

**Two meta-lessons**: (1) parameter efficiency always wins — every architecture found ways to do more with fewer parameters; (2) the right inductive bias matters — skip connections (ResNet, U-Net) and separable convolutions (MobileNet) each encode correct assumptions about how to process images.

## References

- Krizhevsky et al. (2012). AlexNet. NeurIPS.
- Simonyan & Zisserman (2014). VGGNet. ICLR 2015.
- Szegedy et al. (2014). GoogLeNet. CVPR 2015.
- He et al. (2015). ResNet. CVPR 2016.
- Howard et al. (2017). MobileNets.
- Ronneberger et al. (2015). U-Net. MICCAI.
