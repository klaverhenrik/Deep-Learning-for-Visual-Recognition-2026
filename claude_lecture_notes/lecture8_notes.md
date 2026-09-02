# Lecture 8
# Object Detection and Segmentation

*Deep Learning for Visual Recognition · Aarhus University*

These notes trace the evolution from naive sliding window detection through the full R-CNN family to YOLO's single-pass approach, then pivot to semantic segmentation with FCNs and U-Net. Each architectural step is motivated by the failure mode it addresses.

---

## 1  The Four Core Computer Vision Tasks

Before diving into architectures it is worth being precise about what each task requires, because the output format determines everything else — the loss function, the network head, and the evaluation metric.

| Task | Input | Output | Evaluation |
|---|---|---|---|
| Classification | Image | Class label | Accuracy, top-5 error |
| Classification + Loc | Image | Label + one bounding box | IoU ≥ 0.5 + correct class |
| Object detection | Image | Labels + boxes (multiple) | mAP at IoU thresholds |
| Semantic segmentation | Image | Per-pixel class label | Mean IoU over classes |
| Instance segmentation | Image | Per-pixel label + instance ID | Mask AP at IoU thresholds |

### 1.1  Intersection over Union (IoU)

The standard metric for measuring how well a predicted bounding box matches a ground-truth box is Intersection over Union:

$$\text{IoU} = \frac{\text{Area}(\text{Predicted} \cap \text{Ground Truth})}{\text{Area}(\text{Predicted} \cup \text{Ground Truth})}$$

IoU = 1.0 means perfect overlap; IoU = 0 means no overlap at all. The conventional threshold for a detection to count as a true positive is IoU ≥ 0.5. mAP@0.5 reports the mean average precision when using this threshold; COCO also reports mAP averaged over IoU thresholds from 0.5 to 0.95 in steps of 0.05.

### 1.2  Mean Average Precision (mAP)

mAP is the standard detection metric. For each class: rank all detections across the entire test set by confidence score; sweep the confidence threshold to compute precision and recall at each point; compute the area under the precision-recall curve (AP). Average AP over all classes gives mAP. A detection counts as a true positive only if its IoU with a ground-truth box exceeds the threshold and that ground-truth box has not already been matched.

```python
import torch
import torchvision.ops as ops

# ── Intersection over Union ───────────────────────────────────────────
def iou(box1, box2):
    """
    Compute IoU between two boxes.
    Boxes: (x1, y1, x2, y2) format, where (x1,y1) is top-left.
    """
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)

    # Union
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter / (union + 1e-8)

pred_box  = [50, 50, 200, 200]
gt_box    = [80, 80, 220, 220]
print(f'IoU: {iou(pred_box, gt_box):.3f}')   # >0.5 = good match

# ── Non-Maximum Suppression (NMS) ─────────────────────────────────────
# After detection, many overlapping boxes are predicted for the same object.
# NMS keeps only the highest-scoring box from each cluster of overlaps.
boxes  = torch.tensor([[100,100,200,200],[110,110,210,210],[300,300,400,400]],
                       dtype=torch.float32)
scores = torch.tensor([0.9, 0.75, 0.85])

# torchvision.ops.nms returns indices of kept boxes
kept_idx = ops.nms(boxes, scores, iou_threshold=0.5)
print(f'After NMS: kept boxes at indices {kept_idx.tolist()}')
# boxes[0] and boxes[1] overlap heavily → only boxes[0] (score 0.9) is kept
# boxes[2] does not overlap → also kept

# ── torchvision detection metrics ────────────────────────────────────
# For mAP calculation, use torchmetrics or torchvision built-ins:
# from torchmetrics.detection.mean_ap import MeanAveragePrecision
# metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])
```

*Code 1 – IoU from scratch and Non-Maximum Suppression with `torchvision.ops.nms`. NMS is applied after every detection pass: sort predictions by confidence, keep the highest-scoring box, suppress all others with IoU > threshold, repeat. It is the post-processing step that converts hundreds of overlapping candidates into a clean final set of detections.*

---

## 2  From Sliding Window to Fully Convolutional Networks

### 2.1  The Naive Sliding Window and Its Problems

The conceptually simplest approach to detection: slide a window across the image at multiple positions and scales, run a CNN classifier on each crop, and report detections wherever the confidence is high. This is both intuitive and completely impractical:

- **Speed**: one full CNN forward pass per window position × number of scales. For a $224 \times 224$ image with 16 scales and stride 4, that is thousands of forward passes per image.
- **Scale**: objects appear at different sizes — a person close to the camera is large; one far away is small. Exhaustively searching all scales multiplies the cost further.
- **Localisation**: tight bounding boxes only happen by chance, since the window grid is fixed.

### 2.2  The FC Layer Problem

A standard CNN classifier ends with fully connected layers that expect a fixed-size input vector. If the backbone is designed for $128 \times 128$ images and produces a $1 \times 1 \times 1024$ feature map, feeding a $256 \times 384$ full image produces a $2 \times 3 \times 1024$ feature map — and the matrix multiplication in the FC layer ($1000 \times 1024$) cannot be applied to a $2 \times 3 \times 1024$ tensor. The architecture literally breaks on any input size other than the training size.

### 2.3  1×1 Convolutions Fix This

The fix is to replace every FC layer with an equivalent $1 \times 1$ convolution. A $1 \times 1$ convolution with $N$ filters over a $C$-channel feature map computes exactly the same linear transformation as an FC layer from $C$ inputs to $N$ outputs — but works on any spatial size. If the feature map has spatial size $H \times W$ rather than $1 \times 1$, the $1 \times 1$ conv simply applies that same linear transformation at every position independently, producing an $H \times W \times N$ output volume rather than a $1 \times N$ vector.

The outputs of this Fully Convolutional Network (FCN) are a spatial grid of class scores, one per cell, where each cell corresponds to a receptive field in the input image. This is equivalent to running the original sliding window classifier at every grid position — but using a single forward pass that shares all computation across positions. The savings are enormous: instead of $H \times W$ forward passes, just one.

```python
import torch
import torch.nn as nn
import torchvision.models as models

# ── Converting a classifier to a Fully Convolutional Network ─────────

# Standard classifier: only works on fixed input size
class FixedSizeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # 128 → 64
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # 64 → 32
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # 32 → 16
        )
        # FC layer: ONLY works when backbone output is exactly 16×16
        self.classifier = nn.Linear(256 * 16 * 16, 1000)  # BROKEN for other sizes

    def forward(self, x):
        x = self.backbone(x).flatten(1)   # breaks if input != 128×128
        return self.classifier(x)

# FCN: works on ANY input size
class FullyConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Replace FC with 1×1 conv: works on any spatial size!
        self.classifier = nn.Conv2d(256, 1000, kernel_size=1)  # 1×1 conv

    def forward(self, x):
        feats = self.backbone(x)          # (B, 256, H/8, W/8)
        scores = self.classifier(feats)   # (B, 1000, H/8, W/8) — spatial grid!
        return scores

fcn = FullyConvolutionalNetwork()

# Works on any input size:
for h, w in [(128, 128), (256, 256), (256, 384)]:
    x = torch.randn(1, 3, h, w)
    out = fcn(x)
    print(f'Input {h}×{w} → output {tuple(out.shape[2:])} spatial cells')
# 128×128 → (16,16) cells  — each sees a 128/16 = 8px receptive field
# 256×256 → (32,32) cells
# 256×384 → (32,48) cells  — efficient sliding window in one pass!

# To use as a plain classifier (for training), add global average pooling:
gap_out = fcn(torch.randn(1, 3, 128, 128)).mean(dim=[2,3])  # (1, 1000)
print(f'After GAP: {gap_out.shape}')   # single class score vector
```

*Code 2 – The FC-to-$1\times1$-conv conversion. The `FixedSizeClassifier` breaks for non-$128\times128$ inputs; the FCN works on any size and produces a spatial grid of predictions. Each cell in the output corresponds to one sliding-window position, but all positions are computed in a single forward pass.*

---

## 3  Localisation as Regression

For images containing a single object of known class, bounding box localisation can be treated as a regression problem: predict four numbers $(x, y, \text{width}, \text{height})$ that define the box. This is added to a standard classifier as a second head sharing the same convolutional backbone.

### 3.1  The Dual-Head Architecture

The procedure:

- **Step 1**: Download a pretrained CNN classifier (AlexNet, VGG, ResNet).
- **Step 2**: Attach a new FC regression head that outputs 4 numbers (box coordinates). The classification head is kept intact.
- **Step 3**: Train the regression head with L2 loss (MSE between predicted and true box coordinates). Optionally fine-tune the whole network.
- **Step 4 (inference)**: Run both heads. The classification head gives the class; the regression head gives the box.

Two variants of the regression head are common. Class-agnostic: one shared box prediction regardless of class (4 outputs). Class-specific: one box prediction per class ($4C$ outputs). Class-specific is more accurate but more expensive.

```python
import torch
import torch.nn as nn
import torchvision.models as models

# ── Classification + Localisation with dual heads ────────────────────
class ClassificationAndLocalisation(nn.Module):
    def __init__(self, num_classes=20, pretrained=True):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

        # Shared feature extractor (everything except the final FC)
        self.features   = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim        = 2048  # ResNet50 output dimension

        # Classification head: C class scores
        self.cls_head   = nn.Linear(feat_dim, num_classes)

        # Regression head: 4 box coordinates (class-agnostic)
        # Outputs: (cx, cy, w, h) as fractions of image size
        self.bbox_head  = nn.Linear(feat_dim, 4)

    def forward(self, x):
        feats   = self.features(x).flatten(1)  # (B, 2048)
        cls_out = self.cls_head(feats)          # (B, num_classes) — logits
        box_out = self.bbox_head(feats).sigmoid()  # (B, 4) in [0,1]
        return cls_out, box_out

def localisation_loss(cls_pred, cls_target, box_pred, box_target,
                      cls_weight=1.0, box_weight=1.0):
    """
    Multi-task loss: classification (cross-entropy) + localisation (smooth L1).
    Smooth L1 is less sensitive to outliers than MSE.
    """
    cls_loss = nn.CrossEntropyLoss()(cls_pred, cls_target)
    # Smooth L1 = 0.5*x^2 for |x|<1, else |x|-0.5  (Huber loss)
    box_loss = nn.SmoothL1Loss()(box_pred, box_target)
    return cls_weight * cls_loss + box_weight * box_loss

# ── Training step ─────────────────────────────────────────────────────
model = ClassificationAndLocalisation(num_classes=20)
opt   = torch.optim.Adam(model.parameters(), lr=1e-4)

images   = torch.randn(4, 3, 224, 224)
cls_gt   = torch.randint(0, 20, (4,))
box_gt   = torch.rand(4, 4)   # (cx, cy, w, h) normalised to [0,1]

opt.zero_grad()
cls_pred, box_pred = model(images)
loss = localisation_loss(cls_pred, cls_gt, box_pred, box_gt)
loss.backward()
opt.step()
print(f'Loss: {loss.item():.4f}  (cls + bbox regression)')
```

*Code 3 – Dual-head classification + localisation. Smooth L1 loss is used instead of MSE for box regression because it is more robust to outlier predictions during early training. The sigmoid on `box_head` output constrains predictions to $[0,1]$, interpreted as fractions of image dimensions.*

---

## 4  The R-CNN Family: Two-Stage Detection

Object detection extends localisation to multiple objects of unknown class. The dominant approach for several years was the R-CNN family: first propose candidate regions, then classify each proposal independently. Each generation fixed the primary bottleneck of the previous one.

### 4.1  Region Proposals: Selective Search

Rather than exhaustively evaluating every possible window, selective search proposes a small set of candidate regions (~2,000 per image) that are likely to contain objects. It works bottom-up: start with pixels grouped by colour similarity, then progressively merge similar adjacent regions, converting each cluster into a bounding box. The key property is high recall — even if the precision is low (most proposals contain no object), as long as almost all real objects have at least one proposal nearby, the detector has a chance to find them.

### 4.2  R-CNN (2014)

R-CNN (Girshick et al., 2014) was the first deep learning-based object detector to achieve dramatically better accuracy than traditional methods:

- **Step 1**: Generate ~2,000 region proposals with selective search.
- **Step 2**: Warp each proposal to $227 \times 227$ and run through AlexNet to extract a 4,096-d feature vector.
- **Step 3**: Classify each feature vector with a per-class SVM.
- **Step 4**: Refine box coordinates with a separate linear regressor.

R-CNN's critical weakness: 47 seconds per image at test time. Running 2,000 independent forward passes through AlexNet is simply not practical. Two further problems: the SVM and regressor are trained separately from the CNN features (not end-to-end), and selective search is not learned — it can miss objects or generate poor proposals with no way for the network to correct this.

### 4.3  Fast R-CNN (2015)

Fast R-CNN's core insight: the 2,000 proposals for one image all come from the same image. Rather than running the backbone 2,000 times, run it once to get a feature map for the whole image, then extract features for each proposal by projecting it onto that shared feature map.

The remaining problem: different proposals have different shapes, but the FC layers downstream expect a fixed-size input. The solution is RoI Pooling:

- Project the proposal coordinates onto the feature map (scale by the stride of the backbone).
- Divide the projected region into a fixed $h \times w$ grid (e.g. $7 \times 7$).
- Max-pool within each grid cell, producing a fixed $C \times 7 \times 7$ tensor regardless of the original proposal size.

```python
import torch
import torch.nn.functional as F

# ── RoI Pooling from scratch ──────────────────────────────────────────
# Goal: given a variable-size region proposal on the shared feature map,
# always produce a FIXED output size — e.g. (C, 7, 7).

def roi_pool(feature_map, proposal, output_size=(2, 2)):
    """
    Single-proposal RoI Pooling.
    feature_map: (C, H, W)
    proposal:    (x1, y1, x2, y2) in feature-map pixel coords
    output_size: (out_h, out_w) fixed output grid
    Returns:     (C, out_h, out_w)
    """
    C, H, W = feature_map.shape
    out_h, out_w = output_size
    x1, y1, x2, y2 = proposal

    # Size of each bin in the output grid
    bin_h = (y2 - y1) / out_h
    bin_w = (x2 - x1) / out_w

    output = torch.zeros(C, out_h, out_w)

    for i in range(out_h):       # row of output grid
        for j in range(out_w):   # column of output grid
            # Pixel range for this bin (integer boundaries)
            r0 = max(int(y1 + i * bin_h), 0)
            r1 = min(int(y1 + (i+1) * bin_h), H)
            c0 = max(int(x1 + j * bin_w), 0)
            c1 = min(int(x1 + (j+1) * bin_w), W)

            if r1 > r0 and c1 > c0:
                # Max-pool the bin across all channels
                output[:, i, j] = feature_map[:, r0:r1, c0:c1].flatten(1).max(1).values

    return output

# ── Two proposals of different sizes → same output shape ─────────────
torch.manual_seed(0)
fmap = torch.randn(4, 8, 8)   # 4-channel 8x8 feature map

# Large proposal: covers most of the feature map
out_large = roi_pool(fmap, proposal=(0, 0, 8, 8), output_size=(2, 2))
# Small proposal: top-left corner only
out_small = roi_pool(fmap, proposal=(1, 1, 4, 4), output_size=(2, 2))

print(f"Feature map: {tuple(fmap.shape)}")
print(f"Large proposal (0,0,8,8) → {tuple(out_large.shape)}")
print(f"Small proposal (1,1,4,4) → {tuple(out_small.shape)}")
# Both produce (4, 2, 2) regardless of proposal size — this is the point!

# ── torchvision built-in (batched, fast) ─────────────────────────────
import torchvision.ops as ops

# boxes: (N, 5) — each row is [batch_idx, x1, y1, x2, y2]
fmap_batch = fmap.unsqueeze(0)   # (1, 4, 8, 8)
boxes = torch.tensor([[0, 0., 0., 8., 8.],
                       [0, 1., 1., 4., 4.]], dtype=torch.float32)

pooled = ops.roi_pool(fmap_batch, boxes, output_size=(2,2), spatial_scale=1.0)
print(f"torchvision roi_pool: {tuple(pooled.shape)}")
# (2, 4, 2, 2) — 2 proposals, each (4, 2, 2)

# spatial_scale converts image coordinates to feature-map coordinates.
# If the backbone downsamples by stride 16, use spatial_scale=1/16.
# A box at image pixels (160,160,320,320) becomes (10,10,20,20) on the feature map.
```

*Code 4 – RoI Pooling step by step. The manual loop makes the grid division concrete: the proposal is divided into `out_h × out_w` equal bins and max-pooled within each. Both proposals — large (0,0,8,8) and small (1,1,4,4) — produce an identical (4,2,2) output. The torchvision version handles batches and is what you use in practice; `spatial_scale` converts image-space box coordinates to feature-map coordinates.*

Fast R-CNN also trains the whole system end-to-end with a multi-task loss: cross-entropy for class prediction and smooth L1 for box regression. This replaces R-CNN's separate SVM/regressor training pipeline.

Fast R-CNN achieves $9\times$ faster training and $213\times$ faster testing than R-CNN. Its remaining bottleneck: selective search for region proposals still runs on CPU and takes ~2 seconds per image — slower than the CNN forward pass itself.

### 4.4  Faster R-CNN (2015)

Faster R-CNN replaces selective search with a Region Proposal Network (RPN) — a small CNN that shares the backbone feature map and learns to propose regions. The RPN slides a $3 \times 3$ window over the backbone feature map and at each position outputs scores and box offsets for $k$ anchor boxes (typically $k=9$ from 3 scales × 3 aspect ratios).

The full system has four losses, jointly trained:

- **RPN classification**: is each anchor a foreground (object) or background?
- **RPN regression**: predict offsets from the anchor to the ground-truth box.
- **Detection classification**: what class is each RoI proposal?
- **Detection regression**: refine the RoI box coordinates to the object.

The result is $250\times$ faster than original R-CNN at the same accuracy. With a VGG16 backbone it runs at ~5fps; with a ResNet-101 backbone it achieves ~3fps but with significantly better accuracy. Faster R-CNN with a strong backbone remained state-of-the-art for several years and is still widely used in research.

```python
import torch
import torchvision
from torchvision.models.detection import (fasterrcnn_resnet50_fpn,
                                           FasterRCNN_ResNet50_FPN_Weights)

# ── Using a pretrained Faster R-CNN ──────────────────────────────────
model = fasterrcnn_resnet50_fpn(
    weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

# ── Inference mode ────────────────────────────────────────────────────
model.eval()
images = [torch.rand(3, 800, 600), torch.rand(3, 600, 800)]

with torch.no_grad():
    predictions = model(images)

for i, pred in enumerate(predictions):
    print(f'Image {i}: {len(pred["boxes"])} detections')
    print(f'  boxes:  {pred["boxes"].shape}')   # (N, 4) in xyxy format
    print(f'  labels: {pred["labels"].shape}')  # (N,)   COCO class ids
    print(f'  scores: {pred["scores"].shape}')  # (N,)   confidence

# ── Training mode: pass image list + target dicts ────────────────────
model.train()
targets = [
    {'boxes':  torch.tensor([[10, 20, 100, 200], [50, 60, 150, 250]],
                             dtype=torch.float32),
     'labels': torch.tensor([1, 2])},   # COCO class ids
    {'boxes':  torch.tensor([[30, 40, 200, 300]], dtype=torch.float32),
     'labels': torch.tensor([3])},
]

loss_dict = model(images, targets)
# loss_dict contains four losses:
# 'loss_classifier', 'loss_box_reg',    ← detection head losses
# 'loss_objectness', 'loss_rpn_box_reg'  ← RPN losses
total_loss = sum(loss_dict.values())
print('\nLoss components:')
for k, v in loss_dict.items():
    print(f'  {k}: {v.item():.4f}')
print(f'Total loss: {total_loss.item():.4f}')

# ── Fine-tuning for a custom dataset ─────────────────────────────────
# 1. Change the number of output classes
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 5  # your dataset (including background)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
print(f'Custom detector: {num_classes} classes (incl. background)')
```

*Code 4 – Faster R-CNN from torchvision. In eval mode, pass a list of image tensors and receive prediction dicts. In train mode, also pass target dicts with boxes and labels — the model returns a dict of four losses. Fine-tuning for a custom dataset requires only replacing the `box_predictor` head with the correct number of output classes.*

---

## 5  One-Stage Detection: YOLO and SSD

Two-stage detectors (R-CNN family) are accurate but slow — they first propose regions, then classify each one. One-stage detectors skip the proposal stage entirely and predict class labels and box coordinates for a fixed set of anchor boxes in a single forward pass. This makes them much faster (real-time at 30–150fps) at some cost to accuracy, especially for small objects.

### 5.1  YOLO — You Only Look Once

YOLO (Redmon et al., 2015) formulates detection as a single regression problem from image pixels to box coordinates and class probabilities simultaneously:

- Divide the image into a $7 \times 7$ grid of cells.
- Each cell predicts $B$ bounding boxes (each with $x, y, w, h$, confidence) and $C$ class probabilities, producing a $7 \times 7 \times (B \times 5 + C)$ output tensor.
- A cell is 'responsible' for detecting an object if the object's centre falls in that cell.
- Post-process with NMS to eliminate duplicate detections.

YOLO's main strengths: extremely fast (the original processes 45fps; later versions exceed 150fps), global reasoning (each cell sees the full image context via the backbone, unlike sliding window methods), and learns generalisable representations that transfer well to new domains. Its weakness is poor performance on small or densely packed objects, since each cell only predicts one object.

### 5.2  SSD and RetinaNet

SSD (Single Shot Detector) improves on YOLO by predicting from multiple feature map scales simultaneously — small objects are detected from high-resolution early feature maps; large objects from low-resolution late feature maps. This multi-scale approach substantially improves small-object recall.

RetinaNet addresses the fundamental problem of class imbalance in one-stage training: a typical image has ~100,000 anchor boxes of which only ~10 contain objects. The background class dominates the loss and drowns out signal from the rare positive examples. RetinaNet introduces the focal loss: down-weight easy negatives (where the model is already confident it is background) so that hard positives dominate the gradient. This simple modification gave one-stage detectors accuracy comparable to two-stage methods for the first time.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── YOLO-style single-cell prediction head (simplified) ──────────────
class YOLOHead(nn.Module):
    """
    Predict B boxes and C classes for each cell in the feature grid.
    Output shape: (B_batch, S, S, B*5 + C)
    where S = grid size, B = boxes per cell, C = num classes
    """
    def __init__(self, in_channels, S=7, B=2, C=20):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.conv = nn.Conv2d(in_channels, B*5 + C, kernel_size=1)

    def forward(self, x):
        # x: (batch, in_channels, S, S)
        out = self.conv(x)                    # (batch, B*5+C, S, S)
        out = out.permute(0, 2, 3, 1)         # (batch, S, S, B*5+C)
        return out

# ── Focal Loss (RetinaNet) ────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal loss = -(1-p_t)^gamma * log(p_t)
    Down-weights easy examples (high p_t) so hard examples dominate.
    gamma=0 reduces to standard cross-entropy.
    gamma=2 is the standard RetinaNet setting.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits: (N, C), targets: (N,)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # (N,)
        p_t     = torch.exp(-ce_loss)           # predicted probability of true class
        # Weight: (1 - p_t)^gamma
        # Easy examples: p_t ≈ 1 → weight ≈ 0 → ignored
        # Hard examples: p_t ≈ 0 → weight ≈ 1 → full loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss   = self.alpha * focal_weight * ce_loss
        return focal_loss.mean()

# ── Demonstrate the class imbalance problem ───────────────────────────
focal = FocalLoss(alpha=0.25, gamma=2.0)
ce    = nn.CrossEntropyLoss()

# Simulate: 1 hard positive (model wrong) vs 99 easy negatives (model very right)
logits  = torch.cat([
    torch.tensor([[0.1, 0.9]]),    # hard positive: model predicts class 0, truth is 1
    torch.tensor([[10.0, -10.0]]).repeat(99, 1)  # easy negatives: model very confident
])
targets = torch.cat([torch.tensor([1]), torch.zeros(99, dtype=torch.long)])

ce_val    = ce(logits, targets)
focal_val = focal(logits, targets)
print(f'Cross-entropy loss: {ce_val:.4f}   (dominated by easy negatives)')
print(f'Focal loss:         {focal_val:.4f}  (hard positive dominates)')
# Focal loss is much smaller overall but the hard example contributes more

# Using pretrained RetinaNet from torchvision:
# from torchvision.models.detection import retinanet_resnet50_fpn
# model = retinanet_resnet50_fpn(weights='DEFAULT')
```

*Code 5 – YOLO-style prediction head and RetinaNet focal loss. The focal loss demonstration shows the core class imbalance problem: with standard cross-entropy, 99 easy negative examples swamp the signal from 1 hard positive. Focal loss suppresses the easy examples, letting the hard example dominate the gradient.*

---

## 6  Semantic Segmentation

Semantic segmentation assigns a class label to every pixel in the image. This is strictly harder than classification: the output has the same spatial resolution as the input ($H \times W$), and every pixel must be classified independently. The same global-vs-local tension from detection appears here: the network needs both high-level semantic understanding (what is this object?) and precise spatial localisation (exactly which pixels does it occupy?).

### 6.1  Fully Convolutional Networks for Segmentation

The FCN approach (Long, Shelhamer & Darrell, 2015) converts a classification network to a fully convolutional one and adds an upsampling head. The backbone produces a feature map at 1/32 of the input resolution (for VGGNet with 5 max-pool layers), and the decoder upsamples it back to full resolution using transposed convolution.

The problem: $32\times$ upsampling loses too much spatial detail. The decoder must reconstruct fine-grained boundaries from a feature map that was $32\times$ coarser — and this is genuinely hard. The reconstructed boundaries are blurry, and small objects are often missed entirely.

### 6.2  Skip Connections: FCN-32s → FCN-16s → FCN-8s

The solution is skip connections that fuse predictions from the deep, semantically rich but coarse feature maps with predictions from shallower, spatially precise but less semantic feature maps. FCN-8s uses information from three levels:

- The /32 prediction is upsampled $\times 2$ and added to the /16 feature map.
- That sum is upsampled $\times 2$ again and added to the /8 feature map.
- The resulting /8 prediction is upsampled $\times 8$ to reach the input resolution.

Combining fine (where) and coarse (what) information lets the model produce class boundaries that are both semantically accurate and spatially precise. FCN-8s significantly outperforms FCN-32s at sharp boundaries.

### 6.3  Upsampling Methods

Three approaches for recovering spatial resolution in the decoder:

- **Nearest-neighbour / bilinear upsampling**: No learnable parameters. Simple but effective for $2\times$ upsampling. Bilinear is smoother than nearest-neighbour.
- **Transposed convolution (deconvolution)**: Learnable upsampling. Takes a single value from the low-resolution input, multiplies all filter weights by it, and adds the result to the high-resolution output at overlapping positions. Can learn the optimal upsampling kernel for the task. Watch out for checkerboard artefacts when the filter size produces overlapping receptive fields — solved by using kernel size = stride.
- **Max-unpooling**: Records the positions of maxima during downsampling (switch variables) and places values back at those recorded positions during upsampling. Preserves sharp edges but requires storing the switch variables from the encoder.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ── FCN-style segmentation with skip connections ──────────────────────
class FCN8s(nn.Module):
    """
    FCN-8s: semantic segmentation via fully convolutional network
    with skip connections from /8 and /16 feature maps.
    Uses VGG16 backbone.
    """
    def __init__(self, num_classes=21):  # 21 = 20 PASCAL VOC + background
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Split VGG16 features into blocks by pooling layers
        # pool3 output: /8 of input resolution
        # pool4 output: /16 of input resolution
        # pool5 output: /32 of input resolution
        self.block1_3 = vgg.features[:17]   # up to and including pool3
        self.block4   = vgg.features[17:24] # pool4 block
        self.block5   = vgg.features[24:]   # pool5 block

        # 1×1 convolutions to produce per-class scores at each scale
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        # Replace VGG FC layers with 1×1 convs (FCN conversion)
        self.score_pool5 = nn.Sequential(
            nn.Conv2d(512, 4096, 7, padding=3), nn.ReLU(inplace=True), nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),            nn.ReLU(inplace=True), nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, 1),
        )

    def forward(self, x):
        H, W = x.shape[2:]

        # Forward through each block, save intermediate outputs
        p3 = self.block1_3(x)    # /8
        p4 = self.block4(p3)     # /16
        p5 = self.block5(p4)     # /32

        # Class scores at each scale
        s5 = self.score_pool5(p5)              # /32 (coarse, semantic)
        s4 = self.score_pool4(p4)              # /16
        s3 = self.score_pool3(p3)              # /8  (fine, spatial)

        # Fuse: upsample coarser predictions and add to finer ones
        s5_up = F.interpolate(s5, size=s4.shape[2:], mode='bilinear',
                              align_corners=False)
        fused4 = s5_up + s4              # /16: semantics + some spatial

        f4_up  = F.interpolate(fused4, size=s3.shape[2:], mode='bilinear',
                               align_corners=False)
        fused3 = f4_up + s3              # /8:  semantics + more spatial

        # Upsample to full input resolution
        out = F.interpolate(fused3, size=(H, W), mode='bilinear',
                            align_corners=False)
        return out   # (B, num_classes, H, W) — per-pixel class logits

# ── Test ──────────────────────────────────────────────────────────────
fcn = FCN8s(num_classes=21)
x   = torch.randn(2, 3, 320, 320)
out = fcn(x)
print(f'Input:  {x.shape}')    # (2,  3, 320, 320)
print(f'Output: {out.shape}')  # (2, 21, 320, 320) — same H×W!

# Training: CrossEntropyLoss on per-pixel predictions
seg_mask = torch.randint(0, 21, (2, 320, 320))
loss = nn.CrossEntropyLoss(ignore_index=255)(out, seg_mask)  # 255 = ignore
print(f'Segmentation loss: {loss.item():.4f}')

# Use pretrained FCN from torchvision:
# from torchvision.models.segmentation import fcn_resnet50
# model = fcn_resnet50(weights='DEFAULT')  # COCO pre-trained
```

*Code 6 – FCN-8s with three-scale skip connections. The fuse pattern — upsample coarse prediction, add to fine prediction — is the core of all skip-connection segmentation models. `CrossEntropyLoss` with `ignore_index=255` is the standard training loss, where 255 flags unlabelled or ambiguous pixels.*

---

## 7  U-Net: A Better Encoder-Decoder for Segmentation

FCN-8s improved on naive FCN by fusing predictions from three feature map scales via addition-based skip connections, but its training pipeline is complex (multiple stages, not cleanly end-to-end) and the decoder is relatively shallow. U-Net (Ronneberger et al., 2015) addressed both issues with a symmetric encoder-decoder architecture and skip connections that concatenate rather than add encoder features into the decoder.

### 7.1  The What vs Where Problem

The encoder compresses the image from $128 \times 128 \times 3$ down to $8 \times 8 \times 256$ through successive Conv+Pool blocks. Each halving of spatial dimensions doubles the channel depth, keeping the total information roughly constant. By the bottleneck, the network has learned deep semantic representations — it knows what objects are in the image — but has almost entirely discarded spatial information: it no longer knows precisely where the boundaries lie.

The decoder reverses this: transposed convolutions upsample from $8 \times 8$ back to $128 \times 128$. But the decoder alone cannot recover fine-grained boundaries from a coarse $8 \times 8$ representation. This is the fundamental bottleneck any encoder-decoder faces.

### 7.2  Concatenation Skip Connections

U-Net's solution: at every decoder stage, concatenate the upsampled feature map with the corresponding encoder feature map at the same resolution. This gives the decoder direct access to the high-resolution spatial detail computed in the encoder — boundary edges, fine textures, precise locations — without having to reconstruct it from the coarse bottleneck representation.

This is subtly different from ResNet's skip connections (which add) and FCN's skip connections (which also add after a $1 \times 1$ projection). Concatenation preserves both streams separately: the upsampled semantic features and the encoder spatial features remain distinguishable in the concatenated tensor, and the subsequent double-conv can learn to combine them optimally. Addition forces the two streams to directly cancel or reinforce each other.

```python
import torch
import torch.nn as nn

# ── U-Net: symmetric encoder-decoder with concatenation skips ─────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        self.upconvs  = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # ── Encoder (contracting path) ────────────────────────────────
        in_ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(in_ch, f))
            self.pools.append(nn.MaxPool2d(2, 2))
            in_ch = f

        # ── Bottleneck ────────────────────────────────────────────────
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # ── Decoder (expanding path) ──────────────────────────────────
        for f in reversed(features):
            # Transposed conv: upsample from 2f channels to f
            self.upconvs.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            # After concat: f (upsampled) + f (skip) = 2f channels
            self.decoders.append(DoubleConv(f*2, f))

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # ── Encode: save each encoder output as a skip connection ──────
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)   # <── save for skip connection
            x = pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]   # reverse: deepest encoder matches first decoder

        # ── Decode: upsample, CONCATENATE skip, double-conv ────────────
        for upconv, dec, skip in zip(self.upconvs, self.decoders, skips):
            x = upconv(x)                        # upsample
            x = torch.cat([skip, x], dim=1)      # CONCAT skip connection
            x = dec(x)                           # double conv

        return self.head(x)   # (B, out_channels, H, W)

# ── Test: output matches input spatial dimensions ─────────────────────
unet = UNet(in_channels=1, out_channels=1)    # grayscale → binary mask
x    = torch.randn(2, 1, 256, 256)
out  = unet(x)
print(f'Input:  {x.shape}')    # (2, 1, 256, 256)
print(f'Output: {out.shape}')  # (2, 1, 256, 256) — pixel-perfect same size

# Parameter count
n = sum(p.numel() for p in unet.parameters())
print(f'Parameters: {n/1e6:.1f}M')   # ≈ 31M

# ── Verify skip connections are doing something ───────────────────────
# Count decoder input channels: f_up + f_skip = 2f per stage
for i, (upconv, dec) in enumerate(zip(unet.upconvs, unet.decoders)):
    up_out = upconv.out_channels
    dec_in = dec.net[0].in_channels
    print(f'Decoder {i}: upconv→{up_out}ch  concat→{dec_in}ch  ({up_out}+{up_out}={dec_in})')
# Decoder 0: upconv→512ch  concat→1024ch (512+512=1024)
# Decoder 1: upconv→256ch  concat→ 512ch (256+256=512)  ...

# ── Training with binary cross-entropy (for binary segmentation) ──────
optimiser  = torch.optim.Adam(unet.parameters(), lr=1e-4)
bce_loss   = nn.BCEWithLogitsLoss()
mask_gt    = torch.randint(0, 2, (2, 1, 256, 256)).float()

optimiser.zero_grad()
pred = unet(x)
loss = bce_loss(pred, mask_gt)   # BCEWithLogitsLoss — no sigmoid needed
loss.backward()
optimiser.step()
print(f'Training loss: {loss.item():.4f}')

# For multi-class segmentation, use out_channels=num_classes
# and nn.CrossEntropyLoss() with integer target masks.
```

*Code 7 – Complete U-Net with concatenation skip connections. The critical lines are `skips.append(x)` in the encoder loop and `torch.cat([skip, x], dim=1)` in the decoder loop — these are what distinguishes U-Net from a plain autoencoder. The decoder channel count check confirms that each concatenation doubles the channel dimension, which is the key signature of U-Net vs FCN-style addition.*

### 7.3  FCN vs U-Net: When to Use Which

Both FCN-8s and U-Net use skip connections to combine semantic and spatial information, but they differ in the kind of skip and the decoder depth. FCN adds a $1 \times 1$-projected encoder feature to a coarse decoder prediction — shallow decoder, addition-based. U-Net concatenates the full encoder feature map into a rich symmetric decoder — deeper decoder, concatenation-based, more parameters.

| | FCN-8s | U-Net | When to prefer |
|---|---|---|---|
| Skip type | Addition ($1\times1$ projected) | Concatenation (full channels) | Concat: richer fusion of spatial + semantic |
| Decoder depth | Shallow (just upsample) | Symmetric to encoder | U-Net: more capacity to recover spatial detail |
| Training | Multi-stage, complex | End-to-end | U-Net: simpler, more stable |
| Use case | General natural images | Biomedical, precise boundaries | U-Net when pixel-perfect boundaries matter |
| Params | ~130M (VGG16 backbone) | ~31M (typical) | U-Net: more parameter-efficient |

---

## 8  Instance Segmentation: Mask R-CNN

Semantic segmentation labels every pixel but cannot distinguish different instances of the same class — all dogs get the same label regardless of which dog they belong to. Instance segmentation assigns both a class label and an instance ID to each pixel, requiring the model to simultaneously detect and segment every individual object.

Mask R-CNN (He et al., 2017) extends Faster R-CNN with a third head that predicts a binary segmentation mask for each detected bounding box:

- **Stage 1 (shared backbone + FPN)**: Extract multi-scale feature maps.
- **Stage 2 (RPN)**: Propose candidate object regions.
- **Stage 3 (RoI Align)**: Extract fixed-size features for each proposal. Uses bilinear interpolation rather than quantisation (unlike RoI Pooling), which is critical for mask precision.
- **Three heads in parallel**: class prediction, box regression, and a small FCN that produces a $28 \times 28$ binary mask for each proposal.

The mask head predicts a mask for each class independently (no competition between classes), and only the mask of the predicted class is used. This decoupling prevents the mask prediction from being confused by classification uncertainty. RoI Align's bilinear interpolation (vs RoI Pooling's quantisation) gives a significant improvement in mask quality — the discretisation error in RoI Pooling causes misaligned features that are acceptable for classification but noticeable for pixel-precise masks.

---

## 9  Datasets and Evaluation

| Dataset | Classes | Images | Objects/image | Notes |
|---|---|---|---|---|
| PASCAL VOC 2010 | 20 | ~20K | 2.4 | Detection + segmentation; still used as a benchmark |
| ImageNet Det (ILSVRC) | 200 | ~470K | 1.1 | Detection challenge 2014; broader classes |
| MS COCO 2014 | 80 | ~120K | 7.2 | Richest annotations; standard benchmark for detection, segmentation, keypoints |

MS COCO is the current standard benchmark for object detection and instance segmentation. Its 80 classes cover common objects in everyday scenes; with an average of 7.2 objects per image it is far more densely annotated than PASCAL VOC. COCO mAP averages over IoU thresholds from 0.5 to 0.95 in 0.05 steps, which penalises imprecise localisation more heavily than PASCAL's single mAP@0.5.

---

## 10  Architectural Evolution Summary

The history of detection and segmentation architectures follows the same pattern as classification: each generation identifies the bottleneck of the previous one and designs a targeted solution.

| Model | Year | Key idea | Bottleneck fixed |
|---|---|---|---|
| Sliding window | pre-2014 | Exhaustive search at multiple scales | N/A — the baseline |
| OverFeat | 2013 | $1\times1$ conv efficient sliding window | Slow per-position forward passes |
| R-CNN | 2014 | CNN features per region proposal | Traditional detectors (HOG+SVM) |
| Fast R-CNN | 2015 | Shared conv features + RoI Pooling | R-CNN: 2000 separate forward passes |
| Faster R-CNN | 2015 | Learned RPN replaces selective search | Fast R-CNN: slow CPU region proposals |
| YOLO | 2015 | Single-pass global regression | Two-stage speed bottleneck |
| SSD | 2016 | Multi-scale predictions | YOLO: poor small-object detection |
| RetinaNet | 2017 | Focal loss for class imbalance | One-stage accuracy gap vs two-stage |
| FCN | 2015 | Fully convolutional for pixel prediction | Per-pixel classification impossible with FC layers |
| FCN-8s | 2015 | Skip connections (fine + coarse) | FCN-32s: coarse upsampling, blurry boundaries |
| U-Net | 2015 | Concat skip connections, symmetric | FCN: complicated multi-stage training |
| Mask R-CNN | 2017 | Mask head + RoI Align | Semantic segmentation ≠ instance segmentation |

Three principles unite this entire lecture. First, computation sharing: every improvement from R-CNN to Faster R-CNN to one-stage detectors reduces the number of times the expensive backbone runs per image. Second, multi-scale processing: objects appear at many sizes, and every successful architecture (SSD, FPN, skip-connection FCNs) processes the image at multiple resolutions to handle this. Third, the semantic/spatial tension: deep features know what things are but lose where they are; shallow features know where but not what. Every segmentation architecture — FCN skip connections, U-Net concatenation, Mask R-CNN's RoI Align — is a different engineering solution to the same fundamental problem of combining both.

---

## References

- Redmon, J. et al. (2015). You Only Look Once: Unified, Real-Time Object Detection. CVPR 2016.
- Ren, S. et al. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS.
- Girshick, R. et al. (2014). Rich Feature Hierarchies for Accurate Object Detection (R-CNN). CVPR.
- Girshick, R. (2015). Fast R-CNN. ICCV.
- He, K. et al. (2017). Mask R-CNN. ICCV.
- Long, J. et al. (2015). Fully Convolutional Networks for Semantic Segmentation. CVPR.
- Liu, W. et al. (2016). SSD: Single Shot MultiBox Detector. ECCV.
- Lin, T. et al. (2017). Focal Loss for Dense Object Detection (RetinaNet). ICCV.
- Ronneberger, O. et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
- torchvision detection models: docs.pytorch.org/vision/main/models.html#object-detection