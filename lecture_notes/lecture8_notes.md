# Lecture 8 — Object Detection and Segmentation

*Deep Learning for Visual Recognition · Aarhus University*

---

These notes trace the evolution from naive sliding window detection through the full R-CNN family to YOLO's single-pass approach, then pivot to semantic segmentation with FCNs and U-Net. Each architectural step is motivated by the failure mode it addresses.

---

## 1  The Four Core Vision Tasks

| Task | Input | Output | Evaluation |
|---|---|---|---|
| Classification | Image | Class label | Accuracy, top-5 error |
| Classification + Localisation | Image | Label + one bounding box | IoU ≥ 0.5 + correct class |
| Object detection | Image | Labels + boxes (multiple) | mAP at IoU thresholds |
| Semantic segmentation | Image | Per-pixel class label | Mean IoU over classes |
| Instance segmentation | Image | Per-pixel label + instance ID | Mask AP at IoU thresholds |

### 1.1  Intersection over Union (IoU)

$$\text{IoU} = \frac{\text{Area}(\text{Predicted} \cap \text{Ground Truth})}{\text{Area}(\text{Predicted} \cup \text{Ground Truth})}$$

IoU = 1.0 means perfect overlap; IoU = 0 means no overlap. A detection counts as a true positive when IoU ≥ 0.5 (mAP@0.5) or averaged over IoU from 0.5 to 0.95 (COCO standard).

### 1.2  Mean Average Precision (mAP)

For each class: rank all detections by confidence, sweep the threshold to produce a precision-recall curve, and compute the area under it (AP). Average AP over all classes = mAP.

```python
import torch
import torchvision.ops as ops

def iou(box1, box2):
    x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-8)

# Non-Maximum Suppression: keep highest-scoring non-overlapping boxes
boxes  = torch.tensor([[100,100,200,200],[110,110,210,210],[300,300,400,400]], dtype=torch.float32)
scores = torch.tensor([0.9, 0.75, 0.85])
kept   = ops.nms(boxes, scores, iou_threshold=0.5)
print(f'Kept indices after NMS: {kept.tolist()}')  # [0, 2] — boxes[1] suppressed
```

---

## 2  Efficient Sliding Window: Fully Convolutional Networks

### 2.1  The FC Layer Problem

A CNN classifier ending with FC layers expects a fixed input size. Feeding a larger image produces a larger feature map and breaks the matrix multiplication in the FC layer.

### 2.2  The Fix: Replace FC with 1×1 Convolutions

A 1×1 convolution with $N$ filters computes exactly the same linear transformation as an FC layer from $C$ inputs to $N$ outputs — but works on **any spatial size**. For a larger input, it produces a grid of predictions rather than a single vector: one prediction per receptive field position, equivalent to running the classifier at every sliding window position in a single forward pass.

```python
import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Conv2d(256, 1000, kernel_size=1)  # 1×1 replaces FC

    def forward(self, x):
        return self.classifier(self.backbone(x))   # (B, 1000, H/8, W/8)

fcn = FCN()
for H, W in [(128,128), (256,256), (256,384)]:
    out = fcn(torch.randn(1, 3, H, W))
    print(f'Input {H}×{W} → {tuple(out.shape[2:])} spatial cells')
```

---

## 3  Localisation as Regression

For single-object localisation, add a **regression head** to the classification backbone that predicts four box coordinates $(c_x, c_y, w, h)$ as fractions of image size. Train with a multi-task loss:

$$J = J_\text{classification} + J_\text{regression}, \quad J_\text{regression} = \text{Smooth L1}(\hat{b}, b)$$

Smooth L1 (Huber loss) is less sensitive to outlier predictions than MSE:

$$\text{SmoothL1}(x) = \begin{cases} 0.5 x^2 & |x| < 1 \\ |x| - 0.5 & |x| \geq 1 \end{cases}$$

```python
import torch, torch.nn as nn, torchvision.models as models

class LocalisationNet(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features  = nn.Sequential(*list(backbone.children())[:-1])
        self.cls_head  = nn.Linear(2048, num_classes)
        self.bbox_head = nn.Linear(2048, 4)

    def forward(self, x):
        f   = self.features(x).flatten(1)
        return self.cls_head(f), self.bbox_head(f).sigmoid()

model  = LocalisationNet()
images = torch.randn(4, 3, 224, 224)
cls_pred, box_pred = model(images)
print(f'Class logits: {cls_pred.shape}')  # (4, 20)
print(f'Box coords:   {box_pred.shape}')  # (4, 4) — in [0,1]
```

---

## 4  The R-CNN Family: Two-Stage Detection

### 4.1  Selective Search (Region Proposals)

Instead of exhaustive sliding window, propose ~2,000 candidate regions likely to contain objects using bottom-up pixel clustering. Higher recall than sliding window with far fewer evaluations.

### 4.2  R-CNN (2014)

1. Generate ~2,000 proposals with selective search
2. Warp each proposal to 227×227 and run through AlexNet → 4096-d feature vector
3. Classify with per-class SVM
4. Refine box coordinates with linear regressor

**Fatal weakness**: 47 seconds per image (2,000 separate CNN forward passes). Training is not end-to-end (SVM/regressor trained separately from CNN).

### 4.3  Fast R-CNN (2015)

**Insight**: all proposals come from the same image. Run the CNN **once** on the full image, then extract features for each proposal from the shared feature map using **RoI Pooling**.

**RoI Pooling**: for a proposal projected onto the shared feature map, divide it into a fixed $h \times w$ grid and max-pool within each cell, producing a fixed-size $C \times h \times w$ tensor regardless of proposal size.

Also replaces SVM + regressor with a joint multi-task loss trained end-to-end.

```python
import torch, torch.nn.functional as F

def roi_pool(feature_map, proposal, output_size=(2, 2)):
    """
    Single-proposal RoI Pooling from scratch.
    feature_map: (C, H, W)
    proposal:    (x1, y1, x2, y2) in feature-map coordinates
    Returns:     (C, out_h, out_w)
    """
    C, H, W = feature_map.shape
    out_h, out_w = output_size
    x1, y1, x2, y2 = proposal
    bin_h = (y2 - y1) / out_h
    bin_w = (x2 - x1) / out_w
    output = torch.zeros(C, out_h, out_w)
    for i in range(out_h):
        for j in range(out_w):
            r0 = max(int(y1 + i * bin_h), 0);       r1 = min(int(y1 + (i+1)*bin_h), H)
            c0 = max(int(x1 + j * bin_w), 0);       c1 = min(int(x1 + (j+1)*bin_w), W)
            if r1 > r0 and c1 > c0:
                output[:, i, j] = feature_map[:, r0:r1, c0:c1].flatten(1).max(1).values
    return output

fmap  = torch.randn(4, 8, 8)
out_A = roi_pool(fmap, (0, 0, 8, 8), output_size=(2, 2))
out_B = roi_pool(fmap, (1, 1, 4, 4), output_size=(2, 2))
print(f'Large proposal → {tuple(out_A.shape)}')  # (4, 2, 2)
print(f'Small proposal → {tuple(out_B.shape)}')  # (4, 2, 2) — same shape!

# torchvision built-in (batched, fast):
import torchvision.ops as ops
fmap_batch = fmap.unsqueeze(0)
boxes = torch.tensor([[0, 0., 0., 8., 8.], [0, 1., 1., 4., 4.]], dtype=torch.float32)
pooled = ops.roi_pool(fmap_batch, boxes, output_size=(2,2), spatial_scale=1.0)
print(f'torchvision roi_pool: {tuple(pooled.shape)}')  # (2, 4, 2, 2)
```

### 4.4  Faster R-CNN (2015)

Replaces selective search with a learned **Region Proposal Network (RPN)** that shares the backbone. Jointly trained with four losses:

| Loss | What it trains |
|---|---|
| RPN classification | Is each anchor foreground or background? |
| RPN regression | Anchor → proposed box offset |
| Detection classification | Class of each RoI |
| Detection regression | RoI → final box offset |

250× faster than R-CNN at the same accuracy.

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

# Fine-tune for a custom dataset (e.g. 5 classes including background)
model.roi_heads.box_predictor = FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features, num_classes=5)

# Inference: list of image tensors → list of prediction dicts
model.eval()
images = [torch.rand(3, 800, 600)]
with torch.no_grad():
    preds = model(images)
print(f'Detections: {len(preds[0]["boxes"])}')

# Training: also pass targets
model.train()
targets = [{'boxes': torch.tensor([[50.,50.,200.,200.]]), 'labels': torch.tensor([1])}]
losses = model(images, targets)  # returns dict of 4 losses
print({k: f'{v.item():.3f}' for k, v in losses.items()})
```

---

## 5  One-Stage Detection: YOLO and RetinaNet

### 5.1  YOLO

**One forward pass** from image pixels to class labels and bounding boxes. Divide the image into a $7 \times 7$ grid; each cell predicts $B$ boxes and $C$ class probabilities simultaneously. Post-process with NMS.

Strength: real-time speed (45–150+ fps). Weakness: poor small-object detection (each cell predicts only one object).

### 5.2  RetinaNet: Focal Loss

One-stage detectors suffer from extreme class imbalance: ~100,000 anchor boxes per image of which only ~10 contain objects. The easy background class dominates the loss.

**Focal loss** down-weights easy examples:

$$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

When $\gamma = 0$: standard cross-entropy. When $\gamma = 2$: easy examples ($p_t \approx 1$) are suppressed by $(1-p_t)^2 \approx 0$; hard examples ($p_t \approx 0$) retain full loss weight.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, reduction='none')
        p_t  = torch.exp(-ce)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce
        return loss.mean()

# Demonstrate class imbalance problem
focal = FocalLoss(gamma=2.0)
ce    = nn.CrossEntropyLoss()
logits  = torch.cat([torch.tensor([[0.1, 0.9]]),
                     torch.tensor([[10., -10.]]).repeat(99, 1)])
targets = torch.cat([torch.tensor([1]), torch.zeros(99, dtype=torch.long)])
print(f'CE loss:    {ce(logits, targets):.4f}  (dominated by easy negatives)')
print(f'Focal loss: {focal(logits, targets):.4f}  (hard positive dominates)')
```

---

## 6  Semantic Segmentation

### 6.1  FCN with Skip Connections

An FCN backbone (no FC layers) produces a feature map at 1/32 of input resolution. Upsampling by 32× is too coarse for accurate boundaries. **Skip connections** fuse predictions from multiple scales:

- FCN-32s: upsample /32 prediction by 32× — blurry
- FCN-16s: upsample /32 by 2×, add to /16 prediction, upsample by 16× — sharper
- FCN-8s: further fuse /8 prediction — best boundaries

```python
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models

class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.block1_3 = vgg.features[:17]   # → /8
        self.block4   = vgg.features[17:24] # → /16
        self.block5   = vgg.features[24:]   # → /32
        self.score3 = nn.Conv2d(256, num_classes, 1)
        self.score4 = nn.Conv2d(512, num_classes, 1)
        self.score5 = nn.Sequential(
            nn.Conv2d(512, 4096, 7, padding=3), nn.ReLU(True), nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),           nn.ReLU(True), nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, 1))

    def forward(self, x):
        H, W = x.shape[2:]
        p3 = self.block1_3(x)
        p4 = self.block4(p3)
        p5 = self.block5(p4)
        s5 = self.score5(p5)
        s4 = self.score4(p4)
        s3 = self.score3(p3)
        s5_up = F.interpolate(s5, size=s4.shape[2:], mode='bilinear', align_corners=False)
        fused4 = s5_up + s4
        f4_up  = F.interpolate(fused4, size=s3.shape[2:], mode='bilinear', align_corners=False)
        fused3 = f4_up + s3
        return F.interpolate(fused3, size=(H,W), mode='bilinear', align_corners=False)

fcn = FCN8s()
x   = torch.randn(2, 3, 320, 320)
print(fcn(x).shape)  # (2, 21, 320, 320)
```

### 6.2  Upsampling Methods

| Method | Parameters | Notes |
|---|---|---|
| Nearest-neighbour | 0 | Simple; blocky artefacts |
| Bilinear | 0 | Smooth; common default |
| Transposed conv | Learnable | Most powerful; checkerboard risk if kernel overlaps |
| Max-unpooling | 0 | Requires storing pool positions; sharp edges |

---

## 7  Instance Segmentation: Mask R-CNN

Semantic segmentation labels every pixel but cannot distinguish different instances of the same class. **Mask R-CNN** extends Faster R-CNN with a third head:

1. Shared backbone (ResNet + FPN for multi-scale features)
2. RPN proposes candidate regions
3. **RoI Align** (bilinear interpolation, not quantisation — critical for mask precision) extracts fixed-size features
4. Three parallel heads: class prediction, box regression, **binary mask prediction** (28×28 FCN per class)

The mask head predicts a mask for each class independently — no competition between classes — and only the predicted class's mask is used.

---

## 8  Datasets

| Dataset | Classes | Images | Objects/image | Notes |
|---|---|---|---|---|
| PASCAL VOC 2010 | 20 | ~20K | 2.4 | Detection + segmentation |
| ImageNet Det (ILSVRC) | 200 | ~470K | 1.1 | Detection challenge |
| MS COCO 2014 | 80 | ~120K | 7.2 | Richest; standard benchmark |

---

## 9  Architectural Evolution Summary

| Model | Year | Key idea | Bottleneck fixed |
|---|---|---|---|
| Sliding window | pre-2014 | Exhaustive search | N/A — the baseline |
| R-CNN | 2014 | CNN features per proposal | Traditional detectors |
| Fast R-CNN | 2015 | Shared conv + RoI Pooling | R-CNN: 2000 separate passes |
| Faster R-CNN | 2015 | Learned RPN | Slow CPU region proposals |
| YOLO | 2015 | Single-pass global regression | Two-stage speed bottleneck |
| RetinaNet | 2017 | Focal loss | One-stage class imbalance |
| FCN | 2015 | Fully convolutional + upsample | Per-pixel classification impossible with FC |
| FCN-8s | 2015 | Skip connections | FCN-32s blurry boundaries |
| Mask R-CNN | 2017 | Mask head + RoI Align | Semantic ≠ instance segmentation |

**Three unifying principles**: (1) **computation sharing** — reduce how many times the backbone runs per image; (2) **multi-scale processing** — objects appear at many sizes, process at multiple resolutions; (3) **semantic/spatial tension** — deep features know *what*, shallow features know *where*, every architecture is a different way to combine both.

## References

- Girshick et al. (2014). R-CNN. CVPR.
- Girshick (2015). Fast R-CNN. ICCV.
- Ren et al. (2015). Faster R-CNN. NeurIPS.
- He et al. (2017). Mask R-CNN. ICCV.
- Long et al. (2015). FCN for Semantic Segmentation. CVPR.
- Redmon et al. (2015). YOLO. CVPR 2016.
- Lin et al. (2017). Focal Loss / RetinaNet. ICCV.
